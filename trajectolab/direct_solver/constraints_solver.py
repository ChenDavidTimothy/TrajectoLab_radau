"""
Constraint application functions for the direct solver.
"""

from collections.abc import Sequence

import casadi as ca

from ..input_validation import validate_dynamics_output, validate_interval_length
from ..radau import RadauBasisComponents
from ..tl_types import (
    CasadiDM,
    CasadiMatrix,
    CasadiMX,
    CasadiOpti,
    Constraint,
    DynamicsCallable,
    FloatArray,
    PathConstraintsCallable,
    ProblemParameters,
    ProblemProtocol,
)

# Add these imports
from ..utils.variable_scaling import (
    ProblemScalingInfo,
    get_scaled_variable_bounds,
)


def apply_scaled_path_constraints(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    path_constraints_function: PathConstraintsCallable | None,
    problem_parameters: ProblemParameters,
    scaling_info: ProblemScalingInfo,
    problem: ProblemProtocol,
) -> None:
    """
    Apply path constraints with variable scaling support.

    This function applies path constraints in the scaled variable space,
    properly transforming bounds and custom constraints.
    """
    # Apply standard path constraints if they exist
    if path_constraints_function is not None:
        apply_path_constraints(
            opti,
            mesh_interval_index,
            state_at_nodes,
            control_variables,
            basis_components,
            global_normalized_mesh_nodes,
            initial_time_variable,
            terminal_time_variable,
            path_constraints_function,
            problem_parameters,
        )

    # Apply scaled variable bounds
    if scaling_info.scaling_enabled:
        _apply_scaled_variable_bounds(
            opti,
            mesh_interval_index,
            state_at_nodes,
            control_variables,
            basis_components,
            scaling_info,
            problem,
        )
    else:
        # Apply original bounds without scaling
        _apply_original_variable_bounds(
            opti, mesh_interval_index, state_at_nodes, control_variables, basis_components, problem
        )


def _apply_scaled_variable_bounds(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    scaling_info: ProblemScalingInfo,
    problem: ProblemProtocol,
) -> None:
    """Apply variable bounds in the scaled space."""
    num_colloc_nodes = len(basis_components.collocation_nodes)

    # Apply scaled state bounds at all nodes
    state_names = sorted(problem._states.keys(), key=lambda n: problem._states[n]["index"])
    for node_idx in range(state_at_nodes.shape[1]):
        state_at_node = state_at_nodes[:, node_idx]
        for i, name in enumerate(state_names):
            if name in scaling_info.state_scaling:
                scaling = scaling_info.state_scaling[name]
                if scaling.lower_bound is not None and scaling.upper_bound is not None:
                    try:
                        lower_scaled, upper_scaled = get_scaled_variable_bounds(scaling)
                        opti.subject_to(state_at_node[i] >= lower_scaled)
                        opti.subject_to(state_at_node[i] <= upper_scaled)
                    except ValueError:
                        # Skip bounds if calculation fails
                        pass

    # Apply scaled control bounds at collocation points
    control_names = sorted(problem._controls.keys(), key=lambda n: problem._controls[n]["index"])
    for colloc_idx in range(num_colloc_nodes):
        control_at_colloc = control_variables[:, colloc_idx]
        for i, name in enumerate(control_names):
            if name in scaling_info.control_scaling:
                scaling = scaling_info.control_scaling[name]
                if scaling.lower_bound is not None and scaling.upper_bound is not None:
                    try:
                        lower_scaled, upper_scaled = get_scaled_variable_bounds(scaling)
                        opti.subject_to(control_at_colloc[i] >= lower_scaled)
                        opti.subject_to(control_at_colloc[i] <= upper_scaled)
                    except ValueError:
                        # Skip bounds if calculation fails
                        pass


def _apply_original_variable_bounds(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    problem: ProblemProtocol,
) -> None:
    """Apply variable bounds in the original (unscaled) space."""
    num_colloc_nodes = len(basis_components.collocation_nodes)

    # Apply original state bounds
    state_names = sorted(problem._states.keys(), key=lambda n: problem._states[n]["index"])
    for node_idx in range(state_at_nodes.shape[1]):
        state_at_node = state_at_nodes[:, node_idx]
        for i, name in enumerate(state_names):
            state_info = problem._states[name]
            if state_info.get("lower") is not None:
                opti.subject_to(state_at_node[i] >= state_info["lower"])
            if state_info.get("upper") is not None:
                opti.subject_to(state_at_node[i] <= state_info["upper"])

    # Apply original control bounds
    control_names = sorted(problem._controls.keys(), key=lambda n: problem._controls[n]["index"])
    for colloc_idx in range(num_colloc_nodes):
        control_at_colloc = control_variables[:, colloc_idx]
        for i, name in enumerate(control_names):
            control_info = problem._controls[name]
            if control_info.get("lower") is not None:
                opti.subject_to(control_at_colloc[i] >= control_info["lower"])
            if control_info.get("upper") is not None:
                opti.subject_to(control_at_colloc[i] <= control_info["upper"])


def apply_constraint(opti: CasadiOpti, constraint: Constraint) -> None:
    """
    Apply a constraint to the optimization problem.

    Args:
        opti: CasADi optimization object
        constraint: Constraint to apply
    """
    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)


def apply_collocation_constraints(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    dynamics_function: DynamicsCallable,
    problem_parameters: ProblemParameters,
) -> None:
    """
    Apply collocation constraints for a single mesh interval using differential form.

    Args:
        opti: CasADi optimization object
        mesh_interval_index: Index of mesh interval
        state_at_nodes: State variables at all nodes in interval
        control_variables: Control variables for interval
        basis_components: Radau basis components
        global_normalized_mesh_nodes: Global mesh nodes
        initial_time_variable: Initial time variable
        terminal_time_variable: Terminal time variable
        dynamics_function: Dynamics function
        problem_parameters: Problem parameters

    Raises:
        ValueError: If interval length is invalid
    """
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    diff_matrix: CasadiDM = ca.DM(basis_components.differentiation_matrix)

    # Validate interval length
    validate_interval_length(
        global_normalized_mesh_nodes[mesh_interval_index],
        global_normalized_mesh_nodes[mesh_interval_index + 1],
        mesh_interval_index,
    )

    # Calculate state derivatives at collocation points using differentiation matrix
    state_derivative_at_colloc: CasadiMX = ca.mtimes(state_at_nodes, diff_matrix.T)

    # Calculate global segment length and time scaling
    global_segment_length: float = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    tau_to_time_scaling: CasadiMX = (
        (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
    )

    # Apply constraints at each collocation point
    for i_colloc in range(num_colloc_nodes):
        state_at_colloc: CasadiMX = state_at_nodes[:, i_colloc]
        control_at_colloc: CasadiMX = control_variables[:, i_colloc]

        # Calculate physical time at this collocation point
        local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
        global_colloc_tau_val: CasadiMX = (
            global_segment_length / 2 * local_colloc_tau_val
            + (
                global_normalized_mesh_nodes[mesh_interval_index + 1]
                + global_normalized_mesh_nodes[mesh_interval_index]
            )
            / 2
        )
        physical_time_at_colloc: CasadiMX = (
            terminal_time_variable - initial_time_variable
        ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

        # Get dynamics and apply constraint
        state_derivative_rhs: list[CasadiMX] | CasadiMX | Sequence[CasadiMX] = dynamics_function(
            state_at_colloc, control_at_colloc, physical_time_at_colloc, problem_parameters
        )

        # Validate and format dynamics output
        num_states = state_at_nodes.shape[0]
        state_derivative_rhs_vector: CasadiMX = validate_dynamics_output(
            state_derivative_rhs, num_states
        )

        # Apply collocation constraint: state_derivative = time_scaling * dynamics
        opti.subject_to(
            state_derivative_at_colloc[:, i_colloc]
            == tau_to_time_scaling * state_derivative_rhs_vector
        )


def apply_path_constraints(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    path_constraints_function: PathConstraintsCallable,
    problem_parameters: ProblemParameters,
) -> None:
    """
    Apply path constraints for a single mesh interval.

    Args:
        opti: CasADi optimization object
        mesh_interval_index: Index of mesh interval
        state_at_nodes: State variables at all nodes in interval
        control_variables: Control variables for interval
        basis_components: Radau basis components
        global_normalized_mesh_nodes: Global mesh nodes
        initial_time_variable: Initial time variable
        terminal_time_variable: Terminal time variable
        path_constraints_function: Path constraints function
        problem_parameters: Problem parameters
    """
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    for i_colloc in range(num_colloc_nodes):
        state_at_colloc: CasadiMX = state_at_nodes[:, i_colloc]
        control_at_colloc: CasadiMX = control_variables[:, i_colloc]

        # Calculate physical time at this collocation point
        local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
        global_colloc_tau_val: CasadiMX = (
            global_segment_length / 2 * local_colloc_tau_val
            + (
                global_normalized_mesh_nodes[mesh_interval_index + 1]
                + global_normalized_mesh_nodes[mesh_interval_index]
            )
            / 2
        )
        physical_time_at_colloc: CasadiMX = (
            terminal_time_variable - initial_time_variable
        ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

        # Get and apply path constraints
        path_constraints_result: list[Constraint] | Constraint = path_constraints_function(
            state_at_colloc,
            control_at_colloc,
            physical_time_at_colloc,
            problem_parameters,
        )

        constraints_to_apply = (
            path_constraints_result
            if isinstance(path_constraints_result, list)
            else [path_constraints_result]
        )

        for constraint in constraints_to_apply:
            apply_constraint(opti, constraint)


def apply_event_constraints(
    opti: CasadiOpti,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    initial_state: CasadiMX,
    terminal_state: CasadiMX,
    integral_variables: CasadiMX | None,
    problem: ProblemProtocol,
) -> None:
    """
    Apply event constraints to the optimization problem.

    Args:
        opti: CasADi optimization object
        initial_time_variable: Initial time variable
        terminal_time_variable: Terminal time variable
        initial_state: Initial state vector
        terminal_state: Terminal state vector
        integral_variables: Integral variables (if any)
        problem: Problem definition
    """
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
