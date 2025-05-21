"""
Fixed collocation constraints application with proper scaling validation.
"""

import logging

import casadi as ca

from trajectolab.input_validation import validate_dynamics_output, validate_interval_length
from trajectolab.radau import RadauBasisComponents
from trajectolab.tl_types import (
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


# Configure constraints-specific logger
constraints_logger = logging.getLogger("trajectolab.constraints")
if not constraints_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    constraints_logger.addHandler(handler)
    constraints_logger.setLevel(logging.INFO)


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
    problem: ProblemProtocol | None = None,
) -> None:
    """
    Apply collocation constraints for a single mesh interval using differential form.
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

    # Check if scaling is enabled using consistent property
    use_scaling = False
    if problem is not None and hasattr(problem, "use_scaling"):
        use_scaling = problem.use_scaling

    constraints_logger.info(f"Interval {mesh_interval_index}: use_scaling={use_scaling}")

    # Extract state names for scaling if needed
    state_names = []
    if problem is not None and hasattr(problem, "_states"):
        state_names = list(problem._states.keys())
        constraints_logger.debug(f"Found {len(state_names)} state names: {state_names}")

    # Prepare parameters to pass to dynamics function - include scaling info
    dynamics_params = dict(problem_parameters)
    if problem is not None:
        # Add scaling information to parameters for dynamics function
        dynamics_params["_use_scaling"] = use_scaling
        dynamics_params["_scaling_object"] = (
            problem._scaling if hasattr(problem, "_scaling") else None
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

        # Get dynamics - unscaling now happens in the dynamics_function wrapper
        state_derivative_rhs: list[CasadiMX] | CasadiMX | Sequence[CasadiMX] = dynamics_function(
            state_at_colloc, control_at_colloc, physical_time_at_colloc, dynamics_params
        )

        # Validate and format dynamics output
        num_states = state_at_nodes.shape[0]
        state_derivative_rhs_vector: CasadiMX = validate_dynamics_output(
            state_derivative_rhs, num_states
        )

        # Apply scaled or unscaled constraints as appropriate
        if use_scaling and problem is not None and hasattr(problem, "_scaling"):
            # Apply scaled constraints using imported function
            from trajectolab.scaling import apply_scaling_to_defect_constraints

            apply_scaling_to_defect_constraints(
                opti,
                problem._scaling,
                state_names,
                state_derivative_at_colloc,
                state_derivative_rhs_vector,
                tau_to_time_scaling,
                i_colloc,
            )
            constraints_logger.debug(
                f"Applied scaled defect constraints for collocation point {i_colloc}"
            )
        else:
            # Apply unscaled constraint (original implementation)
            constraints_logger.debug(
                f"Applied unscaled defect constraints for collocation point {i_colloc}"
            )
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
    problem: ProblemProtocol | None = None,
) -> None:
    """
    Apply path constraints for a single mesh interval.
    """
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    # Check if scaling is enabled using consistent property
    use_scaling = False
    if problem is not None and hasattr(problem, "use_scaling"):
        use_scaling = problem.use_scaling

    # Prepare parameters to pass to path constraints function - include scaling info
    constraint_params = dict(problem_parameters)
    if problem is not None:
        # Add scaling information to parameters
        constraint_params["_use_scaling"] = use_scaling
        constraint_params["_scaling_object"] = (
            problem._scaling if hasattr(problem, "_scaling") else None
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

        # Get and apply path constraints - passes scaling info through parameters
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
