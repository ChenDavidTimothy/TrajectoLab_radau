"""
Variable setup functions for the direct solver.
"""

import casadi as ca

from ..input_validation import validate_problem_dimensions, validate_time_bounds
from ..tl_types import CasadiMX, CasadiOpti, ListOfCasadiMX, ProblemProtocol
from ..utils.constants import MINIMUM_TIME_INTERVAL
from .types import VariableReferences, _IntervalBundle


def setup_optimization_variables(
    opti: CasadiOpti,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> VariableReferences:
    """
    Set up all optimization variables for the problem.

    Args:
        opti: CasADi optimization object
        problem: Problem definition
        num_mesh_intervals: Number of mesh intervals

    Returns:
        Container with all variable references

    Raises:
        ValueError: If problem dimensions are invalid
    """
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Validate problem dimensions
    validate_problem_dimensions(num_states, num_controls, num_integrals)

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
    """
    Set up state variables for a single mesh interval.

    Args:
        opti: CasADi optimization object
        mesh_interval_index: Index of mesh interval
        num_states: Number of state variables
        num_colloc_nodes: Number of collocation nodes
        state_at_global_mesh_nodes: Global state variables

    Returns:
        Tuple of (state_matrix, interior_nodes_variable)

    Raises:
        ValueError: If interior nodes creation fails
    """
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
                raise ValueError("Failed to create interior_nodes_var")
            for i in range(num_interior_nodes):
                state_columns[i + 1] = interior_nodes_var[:, i]

    # Last column is the state at the end of the interval
    state_columns[num_colloc_nodes] = state_at_global_mesh_nodes[mesh_interval_index + 1]

    # Combine all state columns into a matrix
    state_matrix = ca.horzcat(*state_columns)

    return state_matrix, interior_nodes_var


def _create_time_variables(opti: CasadiOpti, problem: ProblemProtocol) -> tuple[CasadiMX, CasadiMX]:
    """Create time variables with bounds."""
    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()

    # Validate and apply time bounds
    validate_time_bounds(problem._t0_bounds, problem._tf_bounds)
    opti.subject_to(initial_time_variable >= problem._t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem._t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem._tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem._tf_bounds[1])
    opti.subject_to(terminal_time_variable > initial_time_variable + MINIMUM_TIME_INTERVAL)

    return initial_time_variable, terminal_time_variable


def _create_global_state_variables(
    opti: CasadiOpti, num_states: int, num_mesh_intervals: int
) -> ListOfCasadiMX:
    """Create state variables at global mesh nodes."""
    return [opti.variable(num_states) for _ in range(num_mesh_intervals + 1)]


def _create_control_variables(
    opti: CasadiOpti, problem: ProblemProtocol, num_mesh_intervals: int
) -> ListOfCasadiMX:
    """Create control variables for each interval."""
    num_controls = len(problem._controls)
    return [
        opti.variable(num_controls, problem.collocation_points_per_interval[k])
        for k in range(num_mesh_intervals)
    ]


def _create_integral_variables(opti: CasadiOpti, num_integrals: int) -> CasadiMX | None:
    """Create integral variables if needed."""
    if num_integrals > 0:
        return opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
    return None
