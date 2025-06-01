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
    get_cross_phase_event_constraints_function,
    get_phase_path_constraints_function,
)
from .state import ConstraintInput, MultiPhaseVariableState, PhaseDefinition
from .variables_problem import StateVariableImpl, TimeVariableImpl


logger = logging.getLogger(__name__)


class PhaseContext:
    """Context manager for phase-specific variable definition."""

    def __init__(self, problem: "Problem", phase_id: PhaseID) -> None:
        self.problem = problem
        self.phase_id = phase_id
        self._phase_def: PhaseDefinition | None = None

    def __enter__(self) -> "PhaseContext":
        self.problem._current_phase_id = self.phase_id
        if self.phase_id not in self.problem._multiphase_state.phases:
            self._phase_def = self.problem._multiphase_state.add_phase(self.phase_id)
        else:
            self._phase_def = self.problem._multiphase_state.phases[self.phase_id]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.problem._current_phase_id = None

    def time(
        self, initial: ConstraintInput = 0.0, final: ConstraintInput = None
    ) -> TimeVariableImpl:
        """Define time variable for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        return variables_problem.create_phase_time_variable(self._phase_def, initial, final)

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> StateVariableImpl:
        """Define state variable for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        return variables_problem.create_phase_state_variable(
            self._phase_def, name, initial, final, boundary
        )

    def control(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        """Define control variable for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        return variables_problem.create_phase_control_variable(self._phase_def, name, boundary)

    def dynamics(
        self,
        dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int],
    ) -> None:
        """Define dynamics for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        variables_problem.set_phase_dynamics(self._phase_def, dynamics_dict)
        logger.info(
            "Dynamics defined for phase %d with %d state variables",
            self.phase_id,
            len(dynamics_dict),
        )

    def add_integral(self, integrand_expr: ca.MX | float | int) -> ca.MX:
        """Add integral expression for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        return variables_problem.add_phase_integral(self._phase_def, integrand_expr)

    def subject_to(self, constraint_expr: ca.MX | float | int) -> None:
        """Add path constraint for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        constraints_problem.add_phase_path_constraint(self._phase_def, constraint_expr)
        logger.debug("Path constraint added to phase %d", self.phase_id)

    def set_mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """Configure mesh for this phase."""
        if self._phase_def is None:
            raise ValueError("Phase definition not initialized")
        logger.info(
            "Setting mesh for phase %d: %d intervals", self.phase_id, len(polynomial_degrees)
        )
        mesh.configure_phase_mesh(self._phase_def, polynomial_degrees, mesh_points)


class Problem:
    """Main class for defining multiphase optimal control problems."""

    def __init__(self, name: str = "Multiphase Problem") -> None:
        """Initialize a new multiphase problem instance."""
        validate_string_not_empty(name, "Problem name")
        self.name = name
        logger.debug("Created multiphase problem: '%s'", name)

        # Core state management
        self._multiphase_state = MultiPhaseVariableState()
        self._current_phase_id: PhaseID | None = None
        self._initial_guess_container = [None]
        self.solver_options: dict[str, Any] = {}

    def phase(self, phase_id: PhaseID) -> PhaseContext:
        """Create a phase context for defining phase-specific variables and constraints."""
        return PhaseContext(self, phase_id)

    def parameter(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        """Define a static parameter that spans across all phases."""
        validate_string_not_empty(name, "Parameter name")
        validate_constraint_input_format(boundary, f"parameter '{name}' boundary")

        param_var = variables_problem.create_static_parameter(
            self._multiphase_state.static_parameters, name, boundary
        )
        logger.debug("Static parameter created: name='%s'", name)
        return param_var

    def minimize(self, objective_expr: ca.MX | float | int) -> None:
        """Define the multiphase objective function to minimize."""
        variables_problem.set_multiphase_objective(self._multiphase_state, objective_expr)
        logger.info("Multiphase objective function defined")

    def subject_to(self, constraint_expr: ca.MX | float | int) -> None:
        """Add a cross-phase constraint to the problem."""
        constraints_problem.add_cross_phase_constraint(self._multiphase_state, constraint_expr)
        logger.debug(
            "Cross-phase constraint added: total=%d",
            len(self._multiphase_state.cross_phase_constraints),
        )

    def set_initial_guess(
        self,
        phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_initial_times: dict[PhaseID, float] | None = None,
        phase_terminal_times: dict[PhaseID, float] | None = None,
        phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
        static_parameters: FloatArray | None = None,
    ) -> None:
        """Set initial guess for the multiphase optimization variables."""
        components = []
        if phase_states is not None:
            components.append(f"states({len(phase_states)} phases)")
        if phase_controls is not None:
            components.append(f"controls({len(phase_controls)} phases)")
        if static_parameters is not None:
            components.append(f"parameters({len(static_parameters)})")

        logger.info("Setting multiphase initial guess: %s", ", ".join(components))

        initial_guess_problem.set_multiphase_initial_guess(
            self._initial_guess_container,
            self._multiphase_state,
            phase_states=phase_states,
            phase_controls=phase_controls,
            phase_initial_times=phase_initial_times,
            phase_terminal_times=phase_terminal_times,
            phase_integrals=phase_integrals,
            static_parameters=static_parameters,
        )

    # ========================================================================
    # PROTOCOL INTERFACE - Required by ProblemProtocol
    # ========================================================================

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

    def get_phase_ids(self) -> list[PhaseID]:
        return self._multiphase_state.get_phase_ids()

    def get_phase_variable_counts(self, phase_id: PhaseID) -> tuple[int, int]:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].get_variable_counts()

    def get_total_variable_counts(self) -> tuple[int, int, int]:
        return self._multiphase_state.get_total_variable_counts()

    def get_phase_ordered_state_names(self, phase_id: PhaseID) -> list[str]:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].state_names.copy()

    def get_phase_ordered_control_names(self, phase_id: PhaseID) -> list[str]:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].control_names.copy()

    def get_phase_dynamics_function(self, phase_id: PhaseID) -> Any:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")

        static_parameter_symbols = (
            self._multiphase_state.static_parameters.get_ordered_parameter_symbols()
        )
        return solver_interface.get_phase_dynamics_function(
            self._multiphase_state.phases[phase_id], static_parameter_symbols
        )

    def get_objective_function(self) -> Any:
        return solver_interface.get_multiphase_objective_function(self._multiphase_state)

    def get_phase_integrand_function(self, phase_id: PhaseID) -> Any:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")

        static_parameter_symbols = (
            self._multiphase_state.static_parameters.get_ordered_parameter_symbols()
        )
        return solver_interface.get_phase_integrand_function(
            self._multiphase_state.phases[phase_id], static_parameter_symbols
        )

    def get_phase_path_constraints_function(self, phase_id: PhaseID) -> Any:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return get_phase_path_constraints_function(self._multiphase_state.phases[phase_id])

    def get_cross_phase_event_constraints_function(self) -> Any:
        return get_cross_phase_event_constraints_function(self._multiphase_state)

    def validate_multiphase_configuration(self) -> None:
        """Validate the multiphase problem configuration with automatic symbolic processing."""
        logger.debug("Processing symbolic boundary constraints for automatic cross-phase linking")

        # Process symbolic boundary constraints and convert to cross-phase constraints
        for phase_id, phase_def in self._multiphase_state.phases.items():
            # Process time constraints
            if phase_def.t0_constraint.is_symbolic():
                if (
                    phase_def.sym_time_initial is None
                    or phase_def.t0_constraint.symbolic_expression is None
                ):
                    raise ValueError(
                        f"Phase {phase_id} has an undefined symbolic time initial expression."
                    )
                constraint_expr = (
                    phase_def.sym_time_initial - phase_def.t0_constraint.symbolic_expression
                )
                self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                logger.debug(f"Added automatic time initial constraint for phase {phase_id}")

            if phase_def.tf_constraint.is_symbolic():
                if (
                    phase_def.sym_time_final is None
                    or phase_def.tf_constraint.symbolic_expression is None
                ):
                    raise ValueError(
                        f"Phase {phase_id} has an undefined symbolic time final expression."
                    )
                constraint_expr = (
                    phase_def.sym_time_final - phase_def.tf_constraint.symbolic_expression
                )
                self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                logger.debug(f"Added automatic time final constraint for phase {phase_id}")

            # Process state symbolic boundary constraints
            for var_name, constraint_type, symbolic_expr in phase_def.symbolic_boundary_constraints:
                state_index = phase_def.state_name_to_index[var_name]

                if constraint_type == "initial":
                    state_initial_sym = phase_def.get_ordered_state_initial_symbols()[state_index]
                    constraint_expr = state_initial_sym - symbolic_expr
                    self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                    logger.debug(
                        f"Added automatic initial constraint for phase {phase_id} state '{var_name}'"
                    )

                elif constraint_type == "final":
                    state_final_sym = phase_def.get_ordered_state_final_symbols()[state_index]
                    constraint_expr = state_final_sym - symbolic_expr
                    self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                    logger.debug(
                        f"Added automatic final constraint for phase {phase_id} state '{var_name}'"
                    )

                elif constraint_type == "boundary":
                    state_initial_sym = phase_def.get_ordered_state_initial_symbols()[state_index]
                    state_final_sym = phase_def.get_ordered_state_final_symbols()[state_index]

                    initial_constraint = state_initial_sym - symbolic_expr
                    final_constraint = state_final_sym - symbolic_expr

                    self._multiphase_state.cross_phase_constraints.extend(
                        [initial_constraint, final_constraint]
                    )
                    logger.debug(
                        f"Added automatic boundary constraints for phase {phase_id} state '{var_name}'"
                    )

        # Final validation
        if not self._multiphase_state.phases:
            raise ValueError("Problem must have at least one phase defined")

        for phase_id, phase_def in self._multiphase_state.phases.items():
            if not phase_def.dynamics_expressions:
                raise ValueError(f"Phase {phase_id} must have dynamics defined")
            if not phase_def.mesh_configured:
                raise ValueError(f"Phase {phase_id} must have mesh configured")

        if self._multiphase_state.objective_expression is None:
            raise ValueError("Problem must have objective function defined")

        logger.debug(
            "Multiphase configuration validated: %d phases, %d cross-phase constraints",
            len(self._multiphase_state.phases),
            len(self._multiphase_state.cross_phase_constraints),
        )
