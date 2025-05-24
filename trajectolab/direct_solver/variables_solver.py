"""
Variable setup functions for the direct solver - UNIFIED CONSTRAINT API with ENHANCED ERROR HANDLING.
Updated to handle unified constraint specification and new time bounds system.
Added targeted configuration validation guard clauses.
"""

import casadi as ca

from ..exceptions import ConfigurationError, DataIntegrityError
from ..input_validation import validate_problem_dimensions, validate_time_bounds
from ..tl_types import CasadiMX, CasadiOpti, ListOfCasadiMX, ProblemProtocol
from ..utils.constants import MINIMUM_TIME_INTERVAL
from .types_solver import VariableReferences, _IntervalBundle


def setup_optimization_variables(
    opti: CasadiOpti,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> VariableReferences:
    """Set up all optimization variables for the problem using unified constraint API with enhanced validation."""
    # Guard clause: Validate CasADi optimization object
    if opti is None:
        raise ConfigurationError(
            "CasADi optimization object cannot be None", "TrajectoLab solver setup error"
        )

    # Guard clause: Validate mesh intervals
    if not isinstance(num_mesh_intervals, int) or num_mesh_intervals <= 0:
        raise ConfigurationError(
            f"Number of mesh intervals must be positive integer, got {num_mesh_intervals}",
            "TrajectoLab mesh configuration error",
        )

    # Get variable counts from unified storage
    num_states, num_controls = problem.get_variable_counts()
    num_integrals = problem._num_integrals

    # Validate problem dimensions
    validate_problem_dimensions(num_states, num_controls, num_integrals)

    # Guard clause: Check mesh configuration consistency
    if len(problem.collocation_points_per_interval) != num_mesh_intervals:
        raise DataIntegrityError(
            f"Collocation points count ({len(problem.collocation_points_per_interval)}) doesn't match mesh intervals ({num_mesh_intervals})",
            "TrajectoLab mesh configuration inconsistency",
        )

    # Create time variables
    initial_time, terminal_time = _create_time_variables(opti, problem)

    # Create state variables at global mesh nodes
    state_at_mesh_nodes = _create_global_state_variables(opti, num_states, num_mesh_intervals)

    # Create control variables for each interval
    control_variables = _create_control_variables(opti, problem, num_mesh_intervals)

    # Create integral variables if needed
    integral_variables = _create_integral_variables(opti, num_integrals)

    return VariableReferences(
        initial_time=initial_time,
        terminal_time=terminal_time,
        state_at_mesh_nodes=state_at_mesh_nodes,
        control_variables=control_variables,
        integral_variables=integral_variables,
    )


def setup_interval_state_variables(
    opti: CasadiOpti,
    mesh_interval_index: int,
    num_states: int,
    num_colloc_nodes: int,
    state_at_global_mesh_nodes: ListOfCasadiMX,
) -> _IntervalBundle:
    """Set up state variables for a single mesh interval with enhanced validation."""
    # Guard clause: Validate inputs
    if not isinstance(mesh_interval_index, int) or mesh_interval_index < 0:
        raise ConfigurationError(
            f"Mesh interval index must be non-negative integer, got {mesh_interval_index}",
            "TrajectoLab interval setup error",
        )

    if not isinstance(num_states, int) or num_states < 0:
        raise ConfigurationError(
            f"Number of states must be non-negative integer, got {num_states}",
            "TrajectoLab state configuration error",
        )

    if not isinstance(num_colloc_nodes, int) or num_colloc_nodes <= 0:
        raise ConfigurationError(
            f"Number of collocation nodes must be positive integer, got {num_colloc_nodes}",
            "TrajectoLab collocation configuration error",
        )

    # Guard clause: Validate mesh node availability
    if mesh_interval_index >= len(state_at_global_mesh_nodes):
        raise DataIntegrityError(
            f"Mesh interval index {mesh_interval_index} exceeds available mesh nodes ({len(state_at_global_mesh_nodes)})",
            "TrajectoLab interval setup inconsistency",
        )

    if (mesh_interval_index + 1) >= len(state_at_global_mesh_nodes):
        raise DataIntegrityError(
            f"Terminal mesh node for interval {mesh_interval_index} not available",
            "TrajectoLab interval setup inconsistency",
        )

    # Initialize state columns
    state_columns: list[CasadiMX] = [ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)]

    # First column is the state at the start of the interval
    state_columns[0] = state_at_global_mesh_nodes[mesh_interval_index]

    # Create interior state variables if needed
    interior_nodes_var: CasadiMX | None = None
    if num_colloc_nodes > 1:
        num_interior_nodes = num_colloc_nodes - 1
        if num_interior_nodes > 0:
            interior_nodes_var = opti.variable(num_states, num_interior_nodes)
            if interior_nodes_var is None:
                raise DataIntegrityError(
                    "Failed to create interior_nodes_var", "CasADi variable creation failure"
                )
            for i in range(num_interior_nodes):
                state_columns[i + 1] = interior_nodes_var[:, i]

    # Last column is the state at the end of the interval
    state_columns[num_colloc_nodes] = state_at_global_mesh_nodes[mesh_interval_index + 1]

    # Combine all state columns into a matrix and ensure type is MX
    state_matrix = ca.horzcat(*state_columns)
    state_matrix = ca.MX(state_matrix)

    return state_matrix, interior_nodes_var


def _create_time_variables(opti: CasadiOpti, problem: ProblemProtocol) -> tuple[CasadiMX, CasadiMX]:
    """Create time variables with bounds using unified constraint API with enhanced validation."""
    # Guard clause: Validate problem time bounds
    t0_bounds = problem._t0_bounds
    tf_bounds = problem._tf_bounds

    if not isinstance(t0_bounds, tuple) or len(t0_bounds) != 2:
        raise ConfigurationError(
            f"Initial time bounds must be tuple of length 2, got {t0_bounds}",
            "TrajectoLab time bounds configuration error",
        )

    if not isinstance(tf_bounds, tuple) or len(tf_bounds) != 2:
        raise ConfigurationError(
            f"Final time bounds must be tuple of length 2, got {tf_bounds}",
            "TrajectoLab time bounds configuration error",
        )

    # Validate and apply time bounds with enhanced validation
    validate_time_bounds(t0_bounds, tf_bounds)

    # Guard clause: Check for physically meaningful time duration
    max_possible_duration = tf_bounds[1] - t0_bounds[0]
    if max_possible_duration < MINIMUM_TIME_INTERVAL:
        raise ConfigurationError(
            f"Maximum possible time duration ({max_possible_duration}) is below minimum required ({MINIMUM_TIME_INTERVAL})",
            f"Time bounds allow no valid solutions: t0 ∈ {t0_bounds}, tf ∈ {tf_bounds}",
        )

    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()

    # Guard clause: Validate CasADi variable creation
    if initial_time_variable is None or terminal_time_variable is None:
        raise DataIntegrityError(
            "Failed to create time variables", "CasADi variable creation failure"
        )

    # Apply initial time bounds
    if t0_bounds[0] == t0_bounds[1]:
        # Fixed initial time
        opti.subject_to(initial_time_variable == t0_bounds[0])
    else:
        # Range constraint for initial time
        if t0_bounds[0] > -1e5:  # Not unbounded below
            opti.subject_to(initial_time_variable >= t0_bounds[0])
        if t0_bounds[1] < 1e5:  # Not unbounded above
            opti.subject_to(initial_time_variable <= t0_bounds[1])

    # Apply final time bounds
    if tf_bounds[0] == tf_bounds[1]:
        # Fixed final time
        opti.subject_to(terminal_time_variable == tf_bounds[0])
    else:
        # Range constraint for final time
        if tf_bounds[0] > -1e5:  # Not unbounded below
            opti.subject_to(terminal_time_variable >= tf_bounds[0])
        if tf_bounds[1] < 1e5:  # Not unbounded above
            opti.subject_to(terminal_time_variable <= tf_bounds[1])

    # Always enforce minimum time interval
    opti.subject_to(terminal_time_variable > initial_time_variable + MINIMUM_TIME_INTERVAL)

    return initial_time_variable, terminal_time_variable


def _create_global_state_variables(
    opti: CasadiOpti, num_states: int, num_mesh_intervals: int
) -> ListOfCasadiMX:
    """Create state variables at global mesh nodes with enhanced validation."""
    # Guard clause: Validate inputs
    if not isinstance(num_states, int) or num_states < 0:
        raise ConfigurationError(
            f"Number of states must be non-negative integer, got {num_states}",
            "TrajectoLab state configuration error",
        )

    if not isinstance(num_mesh_intervals, int) or num_mesh_intervals <= 0:
        raise ConfigurationError(
            f"Number of mesh intervals must be positive integer, got {num_mesh_intervals}",
            "TrajectoLab mesh configuration error",
        )

    # Create state variables at mesh nodes (num_intervals + 1 nodes)
    state_variables = []
    for i in range(num_mesh_intervals + 1):
        state_var = opti.variable(num_states)
        if state_var is None:
            raise DataIntegrityError(
                f"Failed to create state variable at mesh node {i}",
                "CasADi variable creation failure",
            )
        state_variables.append(state_var)

    return state_variables


def _create_control_variables(
    opti: CasadiOpti, problem: ProblemProtocol, num_mesh_intervals: int
) -> ListOfCasadiMX:
    """Create control variables for each interval with enhanced validation."""
    # Guard clause: Validate inputs
    if not isinstance(num_mesh_intervals, int) or num_mesh_intervals <= 0:
        raise ConfigurationError(
            f"Number of mesh intervals must be positive integer, got {num_mesh_intervals}",
            "TrajectoLab mesh configuration error",
        )

    _, num_controls = problem.get_variable_counts()

    # Guard clause: Validate control configuration
    if not isinstance(num_controls, int) or num_controls < 0:
        raise ConfigurationError(
            f"Number of controls must be non-negative integer, got {num_controls}",
            "TrajectoLab control configuration error",
        )

    # Guard clause: Validate collocation points configuration
    if len(problem.collocation_points_per_interval) != num_mesh_intervals:
        raise DataIntegrityError(
            f"Collocation points configuration ({len(problem.collocation_points_per_interval)}) doesn't match mesh intervals ({num_mesh_intervals})",
            "TrajectoLab mesh configuration inconsistency",
        )

    control_variables = []
    for k in range(num_mesh_intervals):
        num_colloc_points = problem.collocation_points_per_interval[k]

        # Guard clause: Validate individual interval configuration
        if not isinstance(num_colloc_points, int) or num_colloc_points <= 0:
            raise ConfigurationError(
                f"Collocation points for interval {k} must be positive integer, got {num_colloc_points}",
                "TrajectoLab polynomial degree configuration error",
            )

        control_var = opti.variable(num_controls, num_colloc_points)
        if control_var is None:
            raise DataIntegrityError(
                f"Failed to create control variable for interval {k}",
                "CasADi variable creation failure",
            )
        control_variables.append(control_var)

    return control_variables


def _create_integral_variables(opti: CasadiOpti, num_integrals: int) -> CasadiMX | None:
    """Create integral variables if needed with enhanced validation."""
    # Guard clause: Validate input
    if not isinstance(num_integrals, int) or num_integrals < 0:
        raise ConfigurationError(
            f"Number of integrals must be non-negative integer, got {num_integrals}",
            "TrajectoLab integral configuration error",
        )

    if num_integrals > 0:
        integral_var = opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        if integral_var is None:
            raise DataIntegrityError(
                "Failed to create integral variables", "CasADi variable creation failure"
            )
        return integral_var
    return None
