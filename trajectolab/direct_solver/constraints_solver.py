"""
Constraint application functions for collocation, path, and event constraints.
"""

from collections.abc import Sequence

import casadi as ca

from trajectolab.input_validation import validate_dynamics_output, validate_interval_length
from trajectolab.radau import RadauBasisComponents
from trajectolab.tl_types import (
    Constraint,
    DynamicsCallable,
    FloatArray,
    PathConstraintsCallable,
    ProblemParameters,
    ProblemProtocol,
)


def apply_constraint(opti: ca.Opti, constraint: Constraint) -> None:
    """Apply a constraint to the optimization problem."""
    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)


def apply_collocation_constraints(
    opti: ca.Opti,
    mesh_interval_index: int,
    state_at_nodes: ca.MX,
    control_variables: ca.MX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    dynamics_function: DynamicsCallable,
    problem_parameters: ProblemParameters,
    problem: ProblemProtocol | None = None,
) -> None:
    """Apply collocation constraints for a single mesh interval using differential form."""
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    diff_matrix: ca.DM = ca.DM(basis_components.differentiation_matrix)

    # Validate interval length
    validate_interval_length(
        global_normalized_mesh_nodes[mesh_interval_index],
        global_normalized_mesh_nodes[mesh_interval_index + 1],
        mesh_interval_index,
    )

    # Calculate state derivatives at collocation points using differentiation matrix
    state_derivative_at_colloc: ca.MX = ca.mtimes(state_at_nodes, diff_matrix.T)

    # Calculate global segment length and time scaling
    global_segment_length: float = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    tau_to_time_scaling: ca.MX = (
        (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
    )

    # Prepare parameters to pass to dynamics function
    dynamics_params = dict(problem_parameters)

    # Apply constraints at each collocation point
    for i_colloc in range(num_colloc_nodes):
        state_at_colloc: ca.MX = state_at_nodes[:, i_colloc]
        control_at_colloc: ca.MX = control_variables[:, i_colloc]

        # Calculate physical time at this collocation point
        local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
        global_colloc_tau_val: ca.MX = (
            global_segment_length / 2 * local_colloc_tau_val
            + (
                global_normalized_mesh_nodes[mesh_interval_index + 1]
                + global_normalized_mesh_nodes[mesh_interval_index]
            )
            / 2
        )
        physical_time_at_colloc: ca.MX = (
            terminal_time_variable - initial_time_variable
        ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

        # Get dynamics
        state_derivative_rhs: list[ca.MX] | ca.MX | Sequence[ca.MX] = dynamics_function(
            state_at_colloc, control_at_colloc, physical_time_at_colloc, dynamics_params
        )

        # Validate and format dynamics output
        num_states = state_at_nodes.shape[0]
        state_derivative_rhs_vector: ca.MX = validate_dynamics_output(
            state_derivative_rhs, num_states
        )

        # Apply constraint
        opti.subject_to(
            state_derivative_at_colloc[:, i_colloc]
            == tau_to_time_scaling * state_derivative_rhs_vector
        )


def apply_path_constraints(
    opti: ca.Opti,
    mesh_interval_index: int,
    state_at_nodes: ca.MX,
    control_variables: ca.MX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    path_constraints_function: PathConstraintsCallable,
    problem_parameters: ProblemParameters,
    problem: ProblemProtocol | None = None,
) -> None:
    """Apply path constraints for a single mesh interval."""
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    # Prepare parameters to pass to path constraints function
    constraint_params = dict(problem_parameters)

    for i_colloc in range(num_colloc_nodes):
        state_at_colloc: ca.MX = state_at_nodes[:, i_colloc]
        control_at_colloc: ca.MX = control_variables[:, i_colloc]

        # Calculate physical time at this collocation point
        local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
        global_colloc_tau_val: ca.MX = (
            global_segment_length / 2 * local_colloc_tau_val
            + (
                global_normalized_mesh_nodes[mesh_interval_index + 1]
                + global_normalized_mesh_nodes[mesh_interval_index]
            )
            / 2
        )
        physical_time_at_colloc: ca.MX = (
            terminal_time_variable - initial_time_variable
        ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

        # Get and apply path constraints
        path_constraints_result: list[Constraint] | Constraint = path_constraints_function(
            state_at_colloc,
            control_at_colloc,
            physical_time_at_colloc,
            constraint_params,
        )

        constraints_to_apply = (
            path_constraints_result
            if isinstance(path_constraints_result, list)
            else [path_constraints_result]
        )

        for constraint in constraints_to_apply:
            apply_constraint(opti, constraint)


def apply_event_constraints(
    opti: ca.Opti,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    initial_state: ca.MX,
    terminal_state: ca.MX,
    integral_variables: ca.MX | None,
    problem: ProblemProtocol,
) -> None:
    """Apply event constraints to the optimization problem."""
    event_constraints_function = problem.get_event_constraints_function()
    if event_constraints_function is None:
        return

    event_constraints_result: list[Constraint] | Constraint = event_constraints_function(
        initial_time_variable,
        terminal_time_variable,
        initial_state,
        terminal_state,
        integral_variables,
        problem._parameters,
    )

    event_constraints_to_apply = (
        event_constraints_result
        if isinstance(event_constraints_result, list)
        else [event_constraints_result]
    )

    for event_constraint in event_constraints_to_apply:
        apply_constraint(opti, event_constraint)
