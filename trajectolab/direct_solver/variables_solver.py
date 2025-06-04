import casadi as ca

from ..tl_types import PhaseID, ProblemProtocol
from ..utils.constants import MINIMUM_TIME_INTERVAL
from .types_solver import (
    MultiPhaseVariableReferences,
    PhaseVariableReferences,
    _PhaseIntervalBundle,
)


def _setup_multiphase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
) -> MultiPhaseVariableReferences:
    multiphase_vars = MultiPhaseVariableReferences()

    phase_ids = problem._get_phase_ids()
    for phase_id in phase_ids:
        phase_vars = _setup_phase_optimization_variables(opti, problem, phase_id)
        multiphase_vars.phase_variables[phase_id] = phase_vars

    total_states, total_controls, num_static_params = problem.get_total_variable_counts()
    if num_static_params > 0:
        multiphase_vars.static_parameters = __create_static_parameter_variables(
            opti, num_static_params
        )

    return multiphase_vars


def _setup_phase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> PhaseVariableReferences:
    num_states, num_controls = problem.get_phase_variable_counts(phase_id)

    phase_def = problem._phases[phase_id]
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    initial_time, terminal_time = _create_phase_time_variables(opti, problem, phase_id)
    state_at_mesh_nodes = _create_phase_global_state_variables(
        opti, num_states, num_mesh_intervals, phase_id
    )
    control_variables = _create_phase_control_variables(opti, problem, phase_id, num_mesh_intervals)
    integral_variables = _create_phase_integral_variables(opti, num_integrals, phase_id)

    return PhaseVariableReferences(
        phase_id=phase_id,
        initial_time=initial_time,
        terminal_time=terminal_time,
        state_at_mesh_nodes=state_at_mesh_nodes,
        control_variables=control_variables,
        integral_variables=integral_variables,
    )


def _create_interior_state_variables(
    opti: ca.Opti,
    num_states: int,
    num_interior_nodes: int,
    phase_id: PhaseID,
) -> ca.MX:
    interior_nodes_var = opti.variable(num_states, num_interior_nodes)
    return interior_nodes_var


def _populate_interior_state_columns(
    state_columns: list[ca.MX],
    interior_nodes_var: ca.MX,
    num_interior_nodes: int,
) -> None:
    for i in range(num_interior_nodes):
        state_columns[i + 1] = interior_nodes_var[:, i]


def _setup_interior_nodes_for_interval(
    opti: ca.Opti,
    state_columns: list[ca.MX],
    num_colloc_nodes: int,
    num_states: int,
    phase_id: PhaseID,
) -> ca.MX | None:
    # Complex interior node creation extracted to eliminate deep nesting
    if num_colloc_nodes <= 1:
        return None

    num_interior_nodes = num_colloc_nodes - 1

    if num_interior_nodes <= 0:
        return None

    interior_nodes_var = _create_interior_state_variables(
        opti, num_states, num_interior_nodes, phase_id
    )

    _populate_interior_state_columns(state_columns, interior_nodes_var, num_interior_nodes)

    return interior_nodes_var


def setup_phase_interval_state_variables(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    num_states: int,
    num_colloc_nodes: int,
    state_at_global_mesh_nodes: list[ca.MX],
) -> _PhaseIntervalBundle:
    # State variable setup for single mesh interval with extraction to eliminate deep nesting
    state_columns: list[ca.MX] = [ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)]

    state_columns[0] = state_at_global_mesh_nodes[mesh_interval_index]

    # Complex interior node logic moved to separate function
    interior_nodes_var = _setup_interior_nodes_for_interval(
        opti, state_columns, num_colloc_nodes, num_states, phase_id
    )

    state_columns[num_colloc_nodes] = state_at_global_mesh_nodes[mesh_interval_index + 1]

    state_matrix = ca.horzcat(*state_columns)
    state_matrix = ca.MX(state_matrix)

    return state_matrix, interior_nodes_var


def _create_phase_time_variables(
    opti: ca.Opti, problem: ProblemProtocol, phase_id: PhaseID
) -> tuple[ca.MX, ca.MX]:
    # Time variables with bounds enforcement for temporal constraints
    phase_def = problem._phases[phase_id]
    t0_bounds = phase_def.t0_bounds
    tf_bounds = phase_def.tf_bounds

    initial_time_variable: ca.MX = opti.variable()
    terminal_time_variable: ca.MX = opti.variable()

    # Fixed time constraints for known schedules
    if t0_bounds[0] == t0_bounds[1]:
        opti.subject_to(initial_time_variable == t0_bounds[0])
    else:
        if t0_bounds[0] > -1e5:
            opti.subject_to(initial_time_variable >= t0_bounds[0])
        if t0_bounds[1] < 1e5:
            opti.subject_to(initial_time_variable <= t0_bounds[1])

    if tf_bounds[0] == tf_bounds[1]:
        opti.subject_to(terminal_time_variable == tf_bounds[0])
    else:
        if tf_bounds[0] > -1e5:
            opti.subject_to(terminal_time_variable >= tf_bounds[0])
        if tf_bounds[1] < 1e5:
            opti.subject_to(terminal_time_variable <= tf_bounds[1])

    # Minimum interval prevents singular coordinate transformations
    opti.subject_to(terminal_time_variable > initial_time_variable + MINIMUM_TIME_INTERVAL)

    return initial_time_variable, terminal_time_variable


def _create_phase_global_state_variables(
    opti: ca.Opti, num_states: int, num_mesh_intervals: int, phase_id: PhaseID
) -> list[ca.MX]:
    # Global mesh node states enable continuity enforcement across intervals
    state_variables = []
    for _i in range(num_mesh_intervals + 1):
        state_var = opti.variable(num_states)
        state_variables.append(state_var)

    return state_variables


def _create_phase_control_variables(
    opti: ca.Opti, problem: ProblemProtocol, phase_id: PhaseID, num_mesh_intervals: int
) -> list[ca.MX]:
    # Control variables per mesh interval with interval-specific polynomial degrees
    _, num_controls = problem.get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]

    control_variables = []
    for k in range(num_mesh_intervals):
        num_colloc_points = phase_def.collocation_points_per_interval[k]

        control_var = opti.variable(num_controls, num_colloc_points)
        control_variables.append(control_var)

    return control_variables


def _create_phase_integral_variables(
    opti: ca.Opti, num_integrals: int, phase_id: PhaseID
) -> ca.MX | None:
    # Integral variables for performance indices and path integrals
    if num_integrals > 0:
        integral_var = opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        return integral_var
    return None


def __create_static_parameter_variables(opti: ca.Opti, num_static_params: int) -> ca.MX:
    # Static parameters shared across all phases for design optimization
    static_params_var = (
        opti.variable(num_static_params) if num_static_params > 1 else opti.variable()
    )
    return static_params_var
