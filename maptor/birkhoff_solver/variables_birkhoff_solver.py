from collections.abc import Callable
from dataclasses import dataclass

import casadi as ca

from ..mtor_types import PhaseID, ProblemProtocol
from ..utils.constants import LARGE_VALUE, TIME_PRECISION
from .types_birkhoff_solver import (
    _BirkhoffMultiPhaseVariable,
    _BirkhoffPhaseIntervalBundle,
    _BirkhoffPhaseVariable,
)


@dataclass
class _BirkhoffBoundConstraint:
    lower: float
    upper: float
    is_fixed: bool

    @classmethod
    def from_bounds(cls, bounds: tuple[float, float]) -> "_BirkhoffBoundConstraint":
        lower, upper = bounds
        return cls(lower=lower, upper=upper, is_fixed=(lower == upper))


@dataclass
class _BirkhoffVariableCreationContext:
    opti: ca.Opti
    problem: ProblemProtocol
    phase_id: PhaseID
    num_states: int
    num_controls: int
    num_mesh_intervals: int
    num_integrals: int
    grid_points: tuple[float, ...]


@dataclass
class _BirkhoffVariableCreator:
    creator: Callable[[_BirkhoffVariableCreationContext], ca.MX | list[ca.MX] | None]
    constraint_applier: (
        Callable[[ca.Opti, list[ca.MX], _BirkhoffVariableCreationContext], None] | None
    ) = None


def _apply_bound_constraints(
    opti: ca.Opti, variable: ca.MX, constraint: _BirkhoffBoundConstraint
) -> None:
    if constraint.is_fixed:
        opti.subject_to(variable == constraint.lower)
    else:
        if constraint.lower > -LARGE_VALUE:
            opti.subject_to(variable >= constraint.lower)
        if constraint.upper < LARGE_VALUE:
            opti.subject_to(variable <= constraint.upper)


def _apply_time_constraints(
    opti: ca.Opti, time_variables: list[ca.MX], context: _BirkhoffVariableCreationContext
) -> None:
    initial_time_var, terminal_time_var = time_variables
    phase_def = context.problem._phases[context.phase_id]

    t0_constraint = _BirkhoffBoundConstraint.from_bounds(phase_def.t0_bounds)
    tf_constraint = _BirkhoffBoundConstraint.from_bounds(phase_def.tf_bounds)

    _apply_bound_constraints(opti, initial_time_var, t0_constraint)
    _apply_bound_constraints(opti, terminal_time_var, tf_constraint)

    opti.subject_to(terminal_time_var > initial_time_var + TIME_PRECISION)


def _create_time_variables(context: _BirkhoffVariableCreationContext) -> list[ca.MX]:
    initial_time_variable = context.opti.variable()
    terminal_time_variable = context.opti.variable()
    return [initial_time_variable, terminal_time_variable]


def _create_state_variables(context: _BirkhoffVariableCreationContext) -> list[ca.MX]:
    state_variables = []
    for _ in range(context.num_mesh_intervals + 1):
        state_var = context.opti.variable(context.num_states)
        state_variables.append(state_var)
    return state_variables


def _create_virtual_variables(context: _BirkhoffVariableCreationContext) -> list[ca.MX]:
    virtual_variables = []
    num_grid_points = len(context.grid_points)
    for _ in range(num_grid_points):
        virtual_var = context.opti.variable(context.num_states)
        virtual_variables.append(virtual_var)
    return virtual_variables


def _create_control_variables(context: _BirkhoffVariableCreationContext) -> list[ca.MX]:
    control_variables = []
    num_grid_points = len(context.grid_points)
    for _ in range(num_grid_points):
        control_var = context.opti.variable(context.num_controls)
        control_variables.append(control_var)
    return control_variables


def _create_integral_variables(context: _BirkhoffVariableCreationContext) -> ca.MX | None:
    if context.num_integrals > 0:
        return (
            context.opti.variable(context.num_integrals)
            if context.num_integrals > 1
            else context.opti.variable()
        )
    return None


def _create_static_parameter_variables(context: _BirkhoffVariableCreationContext) -> ca.MX:
    _, _, num_static_params = context.problem._get_total_variable_counts()
    return (
        context.opti.variable(num_static_params)
        if num_static_params > 1
        else context.opti.variable()
    )


def _create_variable_creators() -> dict[str, _BirkhoffVariableCreator]:
    return {
        "time": _BirkhoffVariableCreator(
            creator=_create_time_variables,
            constraint_applier=_apply_time_constraints,
        ),
        "states": _BirkhoffVariableCreator(
            creator=_create_state_variables,
            constraint_applier=None,
        ),
        "virtuals": _BirkhoffVariableCreator(
            creator=_create_virtual_variables,
            constraint_applier=None,
        ),
        "controls": _BirkhoffVariableCreator(
            creator=_create_control_variables,
            constraint_applier=None,
        ),
        "integrals": _BirkhoffVariableCreator(
            creator=_create_integral_variables,
            constraint_applier=None,
        ),
    }


def _create_phase_variables_unified(
    context: _BirkhoffVariableCreationContext,
) -> dict[str, ca.MX | list[ca.MX] | None]:
    creators = _create_variable_creators()
    variables: dict[str, ca.MX | list[ca.MX] | None] = {}

    for var_type, creator_obj in creators.items():
        created_vars = creator_obj.creator(context)
        variables[var_type] = created_vars

        if creator_obj.constraint_applier is not None and isinstance(created_vars, list):
            creator_obj.constraint_applier(context.opti, created_vars, context)

    return variables


def _create_variable_context(
    opti: ca.Opti, problem: ProblemProtocol, phase_id: PhaseID, grid_points: tuple[float, ...]
) -> _BirkhoffVariableCreationContext:
    num_states, num_controls = problem._get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    return _BirkhoffVariableCreationContext(
        opti=opti,
        problem=problem,
        phase_id=phase_id,
        num_states=num_states,
        num_controls=num_controls,
        num_mesh_intervals=num_mesh_intervals,
        num_integrals=num_integrals,
        grid_points=grid_points,
    )


def _setup_birkhoff_phase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    grid_points: tuple[float, ...],
) -> _BirkhoffPhaseVariable:
    context = _create_variable_context(opti, problem, phase_id, grid_points)
    variables = _create_phase_variables_unified(context)

    time_vars = variables["time"]
    if not isinstance(time_vars, list) or len(time_vars) != 2:
        raise ValueError(f"Expected time variables to be list of 2 MX, got {type(time_vars)}")
    initial_time, terminal_time = time_vars

    state_vars = variables["states"]
    if not isinstance(state_vars, list):
        raise ValueError(f"Expected state variables to be list of MX, got {type(state_vars)}")
    state_at_mesh_nodes = state_vars

    virtual_vars = variables["virtuals"]
    if not isinstance(virtual_vars, list):
        raise ValueError(f"Expected virtual variables to be list of MX, got {type(virtual_vars)}")
    virtual_variables = virtual_vars

    control_vars = variables["controls"]
    if not isinstance(control_vars, list):
        raise ValueError(f"Expected control variables to be list of MX, got {type(control_vars)}")
    control_variables = control_vars

    integral_vars = variables["integrals"]
    integral_variables = integral_vars if isinstance(integral_vars, ca.MX) else None

    return _BirkhoffPhaseVariable(
        phase_id=phase_id,
        initial_time=initial_time,
        terminal_time=terminal_time,
        state_at_mesh_nodes=state_at_mesh_nodes,
        control_variables=control_variables,
        integral_variables=integral_variables,
        virtual_variables=virtual_variables,
    )


def _setup_birkhoff_multiphase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
    grid_points_per_phase: dict[PhaseID, tuple[float, ...]],
) -> _BirkhoffMultiPhaseVariable:
    multiphase_vars = _BirkhoffMultiPhaseVariable()

    for phase_id in problem._get_phase_ids():
        if phase_id not in grid_points_per_phase:
            raise ValueError(f"No grid points provided for phase {phase_id}")

        grid_points = grid_points_per_phase[phase_id]
        phase_vars = _setup_birkhoff_phase_optimization_variables(
            opti, problem, phase_id, grid_points
        )
        multiphase_vars.phase_variables[phase_id] = phase_vars

    _, _, num_static_params = problem._get_total_variable_counts()
    if num_static_params > 0:
        static_context = _BirkhoffVariableCreationContext(
            opti=opti,
            problem=problem,
            phase_id=0,
            num_states=0,
            num_controls=0,
            num_mesh_intervals=0,
            num_integrals=0,
            grid_points=(),
        )
        multiphase_vars.static_parameters = _create_static_parameter_variables(static_context)

    return multiphase_vars


@dataclass
class _BirkhoffInteriorNodeContext:
    opti: ca.Opti
    phase_id: PhaseID
    num_states: int
    state_at_mesh_nodes: list[ca.MX]
    mesh_interval_index: int


def _setup_birkhoff_interior_nodes(context: _BirkhoffInteriorNodeContext) -> ca.MX:
    state_initial = context.state_at_mesh_nodes[context.mesh_interval_index]
    state_final = context.state_at_mesh_nodes[context.mesh_interval_index + 1]

    state_matrix = ca.horzcat(state_initial, state_final)
    return ca.MX(state_matrix)


def setup_birkhoff_phase_interval_state_variables(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    num_states: int,
    state_at_mesh_nodes: list[ca.MX],
) -> _BirkhoffPhaseIntervalBundle:
    context = _BirkhoffInteriorNodeContext(
        opti=opti,
        phase_id=phase_id,
        num_states=num_states,
        state_at_mesh_nodes=state_at_mesh_nodes,
        mesh_interval_index=mesh_interval_index,
    )

    state_matrix = _setup_birkhoff_interior_nodes(context)
    interior_nodes_var = None

    return state_matrix, interior_nodes_var
