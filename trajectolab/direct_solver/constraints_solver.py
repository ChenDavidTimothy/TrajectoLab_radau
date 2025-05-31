# trajectolab/direct_solver/constraints_solver.py
"""
Constraint application functions for multiphase collocation, path, and event constraints.
"""

from collections.abc import Callable, Sequence

import casadi as ca

from trajectolab.input_validation import validate_dynamics_output, validate_interval_length
from trajectolab.radau import RadauBasisComponents
from trajectolab.tl_types import (
    Constraint,
    FloatArray,
    PhaseID,
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


def apply_phase_collocation_constraints(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    state_at_nodes: ca.MX,
    control_variables: ca.MX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    dynamics_function: Callable[..., list[ca.MX]],
    problem: ProblemProtocol | None = None,
    static_parameters_vec: ca.MX | None = None,  # CRITICAL FIX: Add static parameters support
) -> None:
    """Apply collocation constraints for a single mesh interval within a phase."""
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

        # CRITICAL FIX: Pass static parameters to dynamics function
        state_derivative_rhs: list[ca.MX] | ca.MX | Sequence[ca.MX] = dynamics_function(
            state_at_colloc, control_at_colloc, physical_time_at_colloc, static_parameters_vec
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


def apply_phase_path_constraints(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    state_at_nodes: ca.MX,
    control_variables: ca.MX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    path_constraints_function: Callable[..., list[Constraint]],
    problem: ProblemProtocol | None = None,
    static_parameters_vec: ca.MX | None = None,
    static_parameter_symbols: list[ca.MX] | None = None,
) -> None:
    """Apply path constraints for a single mesh interval within a phase."""
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

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

        # Get and apply path constraints - CRITICAL FIX: Pass time variables
        path_constraints_result: list[Constraint] | Constraint = path_constraints_function(
            state_at_colloc,
            control_at_colloc,
            physical_time_at_colloc,
            static_parameters_vec,
            static_parameter_symbols,
            initial_time_variable,  # NEW: Pass initial time variable
            terminal_time_variable,  # NEW: Pass terminal time variable
        )

        constraints_to_apply = (
            path_constraints_result
            if isinstance(path_constraints_result, list)
            else [path_constraints_result]
        )

        for constraint in constraints_to_apply:
            apply_constraint(opti, constraint)


def apply_multiphase_cross_phase_event_constraints(
    opti: ca.Opti,
    phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]],
    static_parameters: ca.MX | None,
    problem: ProblemProtocol,
) -> None:
    """Apply cross-phase event constraints to the multiphase optimization problem."""
    cross_phase_constraints_function = problem.get_cross_phase_event_constraints_function()
    if cross_phase_constraints_function is None:
        print("DEBUG: No cross-phase constraints function found")
        return

    print(
        f"DEBUG: Applying cross-phase constraints with {len(problem._cross_phase_constraints)} raw constraints"
    )

    cross_phase_constraints_result: list[Constraint] | Constraint = (
        cross_phase_constraints_function(
            phase_endpoint_data,
            static_parameters,
        )
    )

    constraints_to_apply = (
        cross_phase_constraints_result
        if isinstance(cross_phase_constraints_result, list)
        else [cross_phase_constraints_result]
    )

    print(f"DEBUG: About to apply {len(constraints_to_apply)} processed cross-phase constraints")
    for i, constraint in enumerate(constraints_to_apply):
        print(f"DEBUG: Constraint {i}: {constraint}")
        apply_constraint(opti, constraint)
