import logging
from collections.abc import Sequence
from typing import Any

import casadi as ca

from ..input_validation import (
    validate_constraint_input_format,
    validate_string_not_empty,
)
from ..tl_types import FloatArray, NumericArrayLike, PhaseID
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .constraints_problem import (
    _get_cross_phase_event_constraints_function,
    _get_phase_path_constraints_function,
)
from .state import ConstraintInput, MultiPhaseVariableState
from .variables_problem import StateVariableImpl, TimeVariableImpl


logger = logging.getLogger(__name__)


def _validate_phase_exists(phases: dict[PhaseID, Any], phase_id: PhaseID) -> None:
    if phase_id not in phases:
        raise ValueError(f"Phase {phase_id} does not exist")


def _validate_constraint_inputs(name: str, boundary: ConstraintInput, context: str) -> None:
    validate_string_not_empty(name, f"{context} name")
    validate_constraint_input_format(boundary, f"{context} '{name}' boundary")


def _log_constraint_addition(
    constraint_count: int, phase_id: PhaseID, constraint_type: str
) -> None:
    logger.debug(
        "Added %d %s constraint(s) to phase %d", constraint_count, constraint_type, phase_id
    )


def _validate_constraint_expressions_not_empty(
    constraint_expressions: tuple, phase_id: PhaseID, constraint_type: str
) -> None:
    if not constraint_expressions:
        raise ValueError(
            f"Phase {phase_id} {constraint_type}_constraints() requires at least one constraint expression"
        )


def _process_symbolic_time_constraints(
    phase_def: Any, phase_id: PhaseID, cross_phase_constraints: list[ca.MX]
) -> None:
    if phase_def.t0_constraint.is_symbolic():
        if (
            phase_def.sym_time_initial is None
            or phase_def.t0_constraint.symbolic_expression is None
        ):
            raise ValueError(f"Phase {phase_id} has an undefined symbolic time initial expression.")
        constraint_expr = phase_def.sym_time_initial - phase_def.t0_constraint.symbolic_expression
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic time initial constraint for phase {phase_id}")

    if phase_def.tf_constraint.is_symbolic():
        if phase_def.sym_time_final is None or phase_def.tf_constraint.symbolic_expression is None:
            raise ValueError(f"Phase {phase_id} has an undefined symbolic time final expression.")
        constraint_expr = phase_def.sym_time_final - phase_def.tf_constraint.symbolic_expression
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic time final constraint for phase {phase_id}")


def _process_single_symbolic_boundary_constraint(
    var_name: str,
    constraint_type: str,
    symbolic_expr: ca.MX,
    phase_def: Any,
    phase_id: PhaseID,
    cross_phase_constraints: list[ca.MX],
) -> None:
    state_index = phase_def.state_name_to_index[var_name]

    if constraint_type == "initial":
        state_initial_sym = phase_def.get_ordered_state_initial_symbols()[state_index]
        constraint_expr = state_initial_sym - symbolic_expr
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic initial constraint for phase {phase_id} state '{var_name}'")

    elif constraint_type == "final":
        state_final_sym = phase_def.get_ordered_state_final_symbols()[state_index]
        constraint_expr = state_final_sym - symbolic_expr
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic final constraint for phase {phase_id} state '{var_name}'")

    elif constraint_type == "boundary":
        state_initial_sym = phase_def.get_ordered_state_initial_symbols()[state_index]
        state_final_sym = phase_def.get_ordered_state_final_symbols()[state_index]

        initial_constraint = state_initial_sym - symbolic_expr
        final_constraint = state_final_sym - symbolic_expr

        cross_phase_constraints.extend([initial_constraint, final_constraint])
        logger.debug(
            f"Added automatic boundary constraints for phase {phase_id} state '{var_name}'"
        )


def _process_symbolic_state_constraints(
    phase_def: Any, phase_id: PhaseID, cross_phase_constraints: list[ca.MX]
) -> None:
    for var_name, constraint_type, symbolic_expr in phase_def.symbolic_boundary_constraints:
        _process_single_symbolic_boundary_constraint(
            var_name, constraint_type, symbolic_expr, phase_def, phase_id, cross_phase_constraints
        )


def _validate_phase_requirements(phases: dict[PhaseID, Any]) -> None:
    if not phases:
        raise ValueError("Problem must have at least one phase defined")

    for phase_id, phase_def in phases.items():
        if not phase_def.dynamics_expressions:
            raise ValueError(f"Phase {phase_id} must have dynamics defined")
        if not phase_def.mesh_configured:
            raise ValueError(f"Phase {phase_id} must have mesh configured")


def _process_symbolic_constraints_for_all_phases(
    phases: dict[PhaseID, Any], cross_phase_constraints: list[ca.MX]
) -> None:
    logger.debug("Processing symbolic boundary constraints for automatic cross-phase linking")

    for phase_id, phase_def in phases.items():
        _process_symbolic_time_constraints(phase_def, phase_id, cross_phase_constraints)
        _process_symbolic_state_constraints(phase_def, phase_id, cross_phase_constraints)


class Phase:
    def __init__(self, problem: "Problem", phase_id: PhaseID) -> None:
        self.problem = problem
        self.phase_id = phase_id
        self._phase_def = self.problem._multiphase_state.set_phase(self.phase_id)

    def time(
        self, initial: ConstraintInput = 0.0, final: ConstraintInput = None
    ) -> TimeVariableImpl:
        return variables_problem.create_phase_time_variable(self._phase_def, initial, final)

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> StateVariableImpl:
        return variables_problem._create_phase_state_variable(
            self._phase_def, name, initial, final, boundary
        )

    def control(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        return variables_problem.create_phase_control_variable(self._phase_def, name, boundary)

    def dynamics(
        self,
        dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int | StateVariableImpl],
    ) -> None:
        variables_problem._set_phase_dynamics(self._phase_def, dynamics_dict)
        logger.info(
            "Dynamics defined for phase %d with %d state variables",
            self.phase_id,
            len(dynamics_dict),
        )

    def add_integral(self, integrand_expr: ca.MX | float | int) -> ca.MX:
        return variables_problem._set_phase_integral(self._phase_def, integrand_expr)

    def path_constraints(self, *constraint_expressions: ca.MX | float | int) -> None:
        _validate_constraint_expressions_not_empty(constraint_expressions, self.phase_id, "path")

        for expr in constraint_expressions:
            constraints_problem.add_path_constraint(self._phase_def, expr)

        _log_constraint_addition(len(constraint_expressions), self.phase_id, "path")

    def event_constraints(self, *constraint_expressions: ca.MX | float | int) -> None:
        _validate_constraint_expressions_not_empty(constraint_expressions, self.phase_id, "event")

        for expr in constraint_expressions:
            constraints_problem.add_event_constraint(self.problem._multiphase_state, expr)

        _log_constraint_addition(len(constraint_expressions), self.phase_id, "event")

    def mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        logger.info(
            "Setting mesh for phase %d: %d intervals", self.phase_id, len(polynomial_degrees)
        )
        mesh._configure_phase_mesh(self._phase_def, polynomial_degrees, mesh_points)


class Problem:
    def __init__(self, name: str = "Multiphase Problem") -> None:
        validate_string_not_empty(name, "Problem name")
        self.name = name
        logger.debug("Created multiphase problem: '%s'", name)

        self._multiphase_state = MultiPhaseVariableState()
        self._initial_guess_container = [None]
        self.solver_options: dict[str, Any] = {}

    def set_phase(self, phase_id: PhaseID) -> Phase:
        if phase_id in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} already exists")

        logger.debug("Adding phase %d to problem '%s'", phase_id, self.name)
        return Phase(self, phase_id)

    def parameter(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        _validate_constraint_inputs(name, boundary, "Parameter")

        param_var = variables_problem._create_static_parameter(
            self._multiphase_state.static_parameters, name, boundary
        )
        logger.debug("Static parameter created: name='%s'", name)
        return param_var

    def minimize(self, objective_expr: ca.MX | float | int) -> None:
        variables_problem._set_multiphase_objective(self._multiphase_state, objective_expr)
        logger.info("Multiphase objective function defined")

    def guess(
        self,
        phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_initial_times: dict[PhaseID, float] | None = None,
        phase_terminal_times: dict[PhaseID, float] | None = None,
        phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
        static_parameters: FloatArray | None = None,
    ) -> None:
        components = []
        if phase_states is not None:
            components.append(f"states({len(phase_states)} phases)")
        if phase_controls is not None:
            components.append(f"controls({len(phase_controls)} phases)")
        if static_parameters is not None:
            components.append(f"parameters({len(static_parameters)})")

        logger.info("Setting multiphase initial guess: %s", ", ".join(components))

        initial_guess_problem._set_multiphase_initial_guess(
            self._initial_guess_container,
            self._multiphase_state,
            phase_states=phase_states,
            phase_controls=phase_controls,
            phase_initial_times=phase_initial_times,
            phase_terminal_times=phase_terminal_times,
            phase_integrals=phase_integrals,
            static_parameters=static_parameters,
        )

    @property
    def _phases(self) -> dict[PhaseID, Any]:
        return self._multiphase_state.phases

    @property
    def _static_parameters(self) -> Any:
        return self._multiphase_state.static_parameters

    @property
    def _cross_phase_constraints(self) -> list[ca.MX]:
        return self._multiphase_state.cross_phase_constraints

    @property
    def _num_phases(self) -> int:
        return len(self._multiphase_state.phases)

    @property
    def initial_guess(self):
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        self._initial_guess_container[0] = value

    def _get_phase_ids(self) -> list[PhaseID]:
        return self._multiphase_state._get_phase_ids()

    def _get_phase_variable_counts(self, phase_id: PhaseID) -> tuple[int, int]:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return self._multiphase_state.phases[phase_id].get_variable_counts()

    def _get_total_variable_counts(self) -> tuple[int, int, int]:
        return self._multiphase_state._get_total_variable_counts()

    def _get_phase_ordered_state_names(self, phase_id: PhaseID) -> list[str]:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return self._multiphase_state.phases[phase_id].state_names.copy()

    def _get_phase_ordered_control_names(self, phase_id: PhaseID) -> list[str]:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return self._multiphase_state.phases[phase_id].control_names.copy()

    def _get_phase_dynamics_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)

        static_parameter_symbols = (
            self._multiphase_state.static_parameters.get_ordered_parameter_symbols()
        )
        return solver_interface._get_phase_dynamics_function(
            self._multiphase_state.phases[phase_id], static_parameter_symbols
        )

    def _get_objective_function(self) -> Any:
        return solver_interface._get_multiphase_objective_function(self._multiphase_state)

    def _get_phase_integrand_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)

        static_parameter_symbols = (
            self._multiphase_state.static_parameters.get_ordered_parameter_symbols()
        )
        return solver_interface._get_phase_integrand_function(
            self._multiphase_state.phases[phase_id], static_parameter_symbols
        )

    def _get_phase_path_constraints_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return _get_phase_path_constraints_function(self._multiphase_state.phases[phase_id])

    def _get_cross_phase_event_constraints_function(self) -> Any:
        return _get_cross_phase_event_constraints_function(self._multiphase_state)

    def validate_multiphase_configuration(self) -> None:
        _process_symbolic_constraints_for_all_phases(
            self._multiphase_state.phases, self._multiphase_state.cross_phase_constraints
        )

        _validate_phase_requirements(self._multiphase_state.phases)

        if self._multiphase_state.objective_expression is None:
            raise ValueError("Problem must have objective function defined")

        logger.debug(
            "Multiphase configuration validated: %d phases, %d cross-phase constraints",
            len(self._multiphase_state.phases),
            len(self._multiphase_state.cross_phase_constraints),
        )
