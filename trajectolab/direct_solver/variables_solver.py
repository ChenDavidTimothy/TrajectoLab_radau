import casadi as ca

from ..exceptions import DataIntegrityError
from ..tl_types import PhaseID, ProblemProtocol
from ..utils.constants import MINIMUM_TIME_INTERVAL
from .types_solver import (
    MultiPhaseVariableReferences,
    PhaseVariableReferences,
    _PhaseIntervalBundle,
)


def setup_multiphase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
) -> MultiPhaseVariableReferences:
    """
    Set up all optimization variables for multiphase problem following CGPOPS structure.

    Creates unified NLP decision vector: z = [z^(1), z^(2), ..., z^(P), s_1, ..., s_n_s]
    where each z^(p) = [Y^(p), U^(p), Q^(p), t_0^(p), t_f^(p)]
    """

    multiphase_vars = MultiPhaseVariableReferences()

    # Set up variables for each phase in order
    phase_ids = problem.get_phase_ids()
    for phase_id in phase_ids:
        phase_vars = _setup_phase_optimization_variables(opti, problem, phase_id)
        multiphase_vars.phase_variables[phase_id] = phase_vars

    # Set up static parameters
    total_states, total_controls, num_static_params = problem.get_total_variable_counts()
    if num_static_params > 0:
        multiphase_vars.static_parameters = _create_static_parameter_variables(
            opti, num_static_params
        )

    return multiphase_vars


def _setup_phase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> PhaseVariableReferences:
    """Set up optimization variables for a single phase."""
    # Get phase information
    num_states, num_controls = problem.get_phase_variable_counts(phase_id)

    # Get phase mesh information (assuming this is available through problem protocol)
    phase_def = problem._phases[phase_id]  # Access internal phase definition
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    # Data integrity check
    if len(phase_def.collocation_points_per_interval) != num_mesh_intervals:
        raise DataIntegrityError(
            f"Phase {phase_id} collocation points count doesn't match mesh intervals",
            "TrajectoLab phase mesh configuration inconsistency",
        )

    # Create phase optimization variables
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
    """Create interior state variables for mesh interval. EXTRACTED to reduce nesting."""
    interior_nodes_var = opti.variable(num_states, num_interior_nodes)

    if interior_nodes_var is None:
        raise DataIntegrityError(
            f"Failed to create interior_nodes_var for phase {phase_id}",
            "CasADi variable creation failure",
        )

    return interior_nodes_var


def _populate_interior_state_columns(
    state_columns: list[ca.MX],
    interior_nodes_var: ca.MX,
    num_interior_nodes: int,
) -> None:
    """Populate state columns with interior node variables. EXTRACTED to reduce nesting."""
    for i in range(num_interior_nodes):
        state_columns[i + 1] = interior_nodes_var[:, i]


def setup_phase_interval_state_variables(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    num_states: int,
    num_colloc_nodes: int,
    state_at_global_mesh_nodes: list[ca.MX],
) -> _PhaseIntervalBundle:
    """
    Set up state variables for a single mesh interval within a phase.
    REFACTORED using EXTRACTION and INVERSION to eliminate 4+ level nesting.
    """
    # INVERSION: Early returns for validation (guard clauses)
    if mesh_interval_index < 0:
        raise DataIntegrityError(
            f"Phase {phase_id} mesh interval index cannot be negative: {mesh_interval_index}",
            "TrajectoLab interval setup error",
        )

    if mesh_interval_index >= len(state_at_global_mesh_nodes):
        raise DataIntegrityError(
            f"Phase {phase_id} mesh interval index {mesh_interval_index} exceeds available mesh nodes ({len(state_at_global_mesh_nodes)})",
            "TrajectoLab interval setup inconsistency",
        )

    if (mesh_interval_index + 1) >= len(state_at_global_mesh_nodes):
        raise DataIntegrityError(
            f"Phase {phase_id} terminal mesh node for interval {mesh_interval_index} not available",
            "TrajectoLab interval setup inconsistency",
        )

    # Initialize state columns
    state_columns: list[ca.MX] = [ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)]

    # First column is the state at the start of the interval
    state_columns[0] = state_at_global_mesh_nodes[mesh_interval_index]

    # INVERSION: Handle simple case first (early return path)
    interior_nodes_var: ca.MX | None = None
    if num_colloc_nodes <= 1:
        # No interior nodes needed - go straight to final setup
        pass
    else:
        # EXTRACTION: Complex interior node logic moved to separate functions
        num_interior_nodes = num_colloc_nodes - 1
        if num_interior_nodes > 0:
            interior_nodes_var = _create_interior_state_variables(
                opti, num_states, num_interior_nodes, phase_id
            )
            _populate_interior_state_columns(state_columns, interior_nodes_var, num_interior_nodes)

    # Last column is the state at the end of the interval
    state_columns[num_colloc_nodes] = state_at_global_mesh_nodes[mesh_interval_index + 1]

    # Combine all state columns into a matrix
    state_matrix = ca.horzcat(*state_columns)
    state_matrix = ca.MX(state_matrix)

    return state_matrix, interior_nodes_var


def _create_phase_time_variables(
    opti: ca.Opti, problem: ProblemProtocol, phase_id: PhaseID
) -> tuple[ca.MX, ca.MX]:
    """Create time variables for a specific phase with bounds."""
    # Get phase time bounds
    phase_def = problem._phases[phase_id]
    t0_bounds = phase_def.t0_bounds
    tf_bounds = phase_def.tf_bounds

    # Create CasADi variables with phase-specific names
    initial_time_variable: ca.MX = opti.variable()
    terminal_time_variable: ca.MX = opti.variable()

    if initial_time_variable is None or terminal_time_variable is None:
        raise DataIntegrityError(
            f"Failed to create time variables for phase {phase_id}",
            "CasADi variable creation failure",
        )

    # Apply initial time bounds
    if t0_bounds[0] == t0_bounds[1]:
        opti.subject_to(initial_time_variable == t0_bounds[0])
    else:
        if t0_bounds[0] > -1e5:
            opti.subject_to(initial_time_variable >= t0_bounds[0])
        if t0_bounds[1] < 1e5:
            opti.subject_to(initial_time_variable <= t0_bounds[1])

    # Apply final time bounds
    if tf_bounds[0] == tf_bounds[1]:
        opti.subject_to(terminal_time_variable == tf_bounds[0])
    else:
        if tf_bounds[0] > -1e5:
            opti.subject_to(terminal_time_variable >= tf_bounds[0])
        if tf_bounds[1] < 1e5:
            opti.subject_to(terminal_time_variable <= tf_bounds[1])

    # Always enforce minimum time interval
    opti.subject_to(terminal_time_variable > initial_time_variable + MINIMUM_TIME_INTERVAL)

    return initial_time_variable, terminal_time_variable


def _create_phase_global_state_variables(
    opti: ca.Opti, num_states: int, num_mesh_intervals: int, phase_id: PhaseID
) -> list[ca.MX]:
    """Create state variables at global mesh nodes for a specific phase."""
    state_variables = []
    for i in range(num_mesh_intervals + 1):
        state_var = opti.variable(num_states)
        if state_var is None:
            raise DataIntegrityError(
                f"Failed to create state variable at mesh node {i} for phase {phase_id}",
                "CasADi variable creation failure",
            )
        state_variables.append(state_var)

    return state_variables


def _create_phase_control_variables(
    opti: ca.Opti, problem: ProblemProtocol, phase_id: PhaseID, num_mesh_intervals: int
) -> list[ca.MX]:
    """Create control variables for each interval in a specific phase."""
    _, num_controls = problem.get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]

    # Data integrity check
    if len(phase_def.collocation_points_per_interval) != num_mesh_intervals:
        raise DataIntegrityError(
            f"Phase {phase_id} collocation points configuration doesn't match mesh intervals",
            "TrajectoLab mesh configuration inconsistency",
        )

    control_variables = []
    for k in range(num_mesh_intervals):
        num_colloc_points = phase_def.collocation_points_per_interval[k]

        control_var = opti.variable(num_controls, num_colloc_points)
        if control_var is None:
            raise DataIntegrityError(
                f"Failed to create control variable for phase {phase_id} interval {k}",
                "CasADi variable creation failure",
            )
        control_variables.append(control_var)

    return control_variables


def _create_phase_integral_variables(
    opti: ca.Opti, num_integrals: int, phase_id: PhaseID
) -> ca.MX | None:
    """Create integral variables for a specific phase if needed."""
    if num_integrals > 0:
        integral_var = opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        if integral_var is None:
            raise DataIntegrityError(
                f"Failed to create integral variables for phase {phase_id}",
                "CasADi variable creation failure",
            )
        return integral_var
    return None


def _create_static_parameter_variables(opti: ca.Opti, num_static_params: int) -> ca.MX:
    """Create static parameter variables that span all phases."""
    static_params_var = (
        opti.variable(num_static_params) if num_static_params > 1 else opti.variable()
    )
    if static_params_var is None:
        raise DataIntegrityError(
            "Failed to create static parameter variables", "CasADi variable creation failure"
        )
    return static_params_var
