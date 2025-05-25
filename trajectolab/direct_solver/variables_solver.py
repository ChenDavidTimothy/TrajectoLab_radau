"""
Optimization variable setup and configuration for the direct solver.
"""

import casadi as ca

from ..exceptions import DataIntegrityError
from ..input_validation import validate_casadi_optimization_object, validate_mesh_interval_count
from ..tl_types import CasadiMX, CasadiOpti, ListOfCasadiMX, ProblemProtocol
from ..utils.constants import MINIMUM_TIME_INTERVAL
from .types_solver import VariableReferences, _IntervalBundle


def setup_optimization_variables(
    opti: CasadiOpti,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> VariableReferences:
    """
    Set up all optimization variables for the problem.

    NOTE: Assumes comprehensive validation already done by validate_problem_ready_for_solving()
    """
    # Only validate things not covered by comprehensive validation
    validate_casadi_optimization_object(opti, "variable setup")
    validate_mesh_interval_count(num_mesh_intervals, "mesh intervals")

    # Get variable counts (already validated)
    num_states, num_controls = problem.get_variable_counts()
    num_integrals = problem._num_integrals

    # Data integrity check (internal consistency, not user configuration)
    if len(problem.collocation_points_per_interval) != num_mesh_intervals:
        raise DataIntegrityError(
            f"Collocation points count ({len(problem.collocation_points_per_interval)}) doesn't match mesh intervals ({num_mesh_intervals})",
            "TrajectoLab mesh configuration inconsistency",
        )

    # Create optimization variables
    initial_time, terminal_time = _create_time_variables(opti, problem)
    state_at_mesh_nodes = _create_global_state_variables(opti, num_states, num_mesh_intervals)
    control_variables = _create_control_variables(opti, problem, num_mesh_intervals)
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
    """
    Set up state variables for a single mesh interval.

    NOTE: Basic parameter validation assumed already done at entry point
    """
    # Only validate specific context that wasn't covered by entry point validation
    if mesh_interval_index < 0:
        raise DataIntegrityError(
            f"Mesh interval index cannot be negative: {mesh_interval_index}",
            "TrajectoLab interval setup error",
        )

    # Data integrity checks (internal consistency)
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

    # Combine all state columns into a matrix
    state_matrix = ca.horzcat(*state_columns)
    state_matrix = ca.MX(state_matrix)

    return state_matrix, interior_nodes_var


def _create_time_variables(opti: CasadiOpti, problem: ProblemProtocol) -> tuple[CasadiMX, CasadiMX]:
    """
    Create time variables with bounds.

    NOTE: Time bounds validation assumed already done by validate_problem_ready_for_solving()
    """
    # Get time bounds (already validated)
    t0_bounds = problem._t0_bounds
    tf_bounds = problem._tf_bounds

    # Create CasADi variables
    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()

    # Data integrity check (CasADi interface)
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
    """
    Create state variables at global mesh nodes.

    NOTE: Parameter validation assumed already done
    """
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
    """
    Create control variables for each interval.

    NOTE: Parameter validation assumed already done
    """
    _, num_controls = problem.get_variable_counts()

    # Data integrity check (internal consistency)
    if len(problem.collocation_points_per_interval) != num_mesh_intervals:
        raise DataIntegrityError(
            f"Collocation points configuration ({len(problem.collocation_points_per_interval)}) doesn't match mesh intervals ({num_mesh_intervals})",
            "TrajectoLab mesh configuration inconsistency",
        )

    control_variables = []
    for k in range(num_mesh_intervals):
        num_colloc_points = problem.collocation_points_per_interval[k]

        control_var = opti.variable(num_controls, num_colloc_points)
        if control_var is None:
            raise DataIntegrityError(
                f"Failed to create control variable for interval {k}",
                "CasADi variable creation failure",
            )
        control_variables.append(control_var)

    return control_variables


def _create_integral_variables(opti: CasadiOpti, num_integrals: int) -> CasadiMX | None:
    """
    Create integral variables if needed.

    NOTE: Parameter validation assumed already done
    """
    if num_integrals > 0:
        integral_var = opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        if integral_var is None:
            raise DataIntegrityError(
                "Failed to create integral variables", "CasADi variable creation failure"
            )
        return integral_var
    return None
