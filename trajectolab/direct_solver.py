from __future__ import annotations

from collections.abc import Sequence

import casadi as ca

from .input_validation import (
    validate_and_set_integral_guess,
    validate_dynamics_output,
    validate_interval_length,
    validate_mesh_configuration,
    validate_problem_dimensions,
    validate_time_bounds,
)
from .radau import RadauBasisComponents, compute_radau_collocation_components
from .solution_extraction import extract_and_format_solution
from .tl_types import (
    CasadiDM,
    CasadiMatrix,
    CasadiMX,
    CasadiOpti,
    CasadiOptiSol,
    DynamicsCallable,
    EventConstraint,
    EventConstraintsCallable,
    FloatArray,
    InitialGuess,
    IntegralIntegrandCallable,
    ListOfCasadiMX,
    ObjectiveCallable,
    OptimalControlSolution,
    PathConstraint,
    PathConstraintsCallable,
    ProblemParameters,
    ProblemProtocol,
)
from .utils.constants import MINIMUM_TIME_INTERVAL


def apply_constraint(opti: CasadiOpti, constraint: PathConstraint | EventConstraint) -> None:
    """Apply a constraint to the optimization problem."""
    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)


def _setup_optimization_variables(
    opti: CasadiOpti,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> tuple[CasadiMX, CasadiMX, ListOfCasadiMX, ListOfCasadiMX, CasadiMX | None]:
    """
    Set up all optimization variables for the problem.

    Returns:
        Tuple of (initial_time, terminal_time, state_at_mesh_nodes,
                  control_variables, integral_variables)
    """
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Validate problem dimensions
    validate_problem_dimensions(num_states, num_controls, num_integrals)

    # Create time variables
    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()

    # Validate and apply time bounds
    validate_time_bounds(problem._t0_bounds, problem._tf_bounds)
    opti.subject_to(initial_time_variable >= problem._t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem._t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem._tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem._tf_bounds[1])
    opti.subject_to(terminal_time_variable > initial_time_variable + MINIMUM_TIME_INTERVAL)

    # Create state variables at global mesh nodes
    state_at_global_mesh_nodes_variables: ListOfCasadiMX = [
        opti.variable(num_states) for _ in range(num_mesh_intervals + 1)
    ]

    # Create control variables for each interval
    control_at_local_collocation_nodes_all_intervals_variables: ListOfCasadiMX = [
        opti.variable(num_controls, problem.collocation_points_per_interval[k])
        for k in range(num_mesh_intervals)
    ]

    # Create integral variables if needed
    integral_decision_variables: CasadiMX | None = None
    if num_integrals > 0:
        integral_decision_variables = (
            opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        )

    return (
        initial_time_variable,
        terminal_time_variable,
        state_at_global_mesh_nodes_variables,
        control_at_local_collocation_nodes_all_intervals_variables,
        integral_decision_variables,
    )


def _setup_interval_state_variables(
    opti: CasadiOpti,
    mesh_interval_index: int,
    num_states: int,
    num_colloc_nodes: int,
    state_at_global_mesh_nodes: ListOfCasadiMX,
) -> tuple[CasadiMatrix, CasadiMX | None]:
    """
    Set up state variables for a single mesh interval.

    Returns:
        Tuple of (state_at_nodes_matrix, interior_nodes_variable)
    """
    current_interval_state_columns: list[CasadiMX] = [
        ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)
    ]

    # First column is the state at the start of the interval
    current_interval_state_columns[0] = state_at_global_mesh_nodes[mesh_interval_index]

    # Create interior state variables if needed
    interior_nodes_var: CasadiMX | None = None
    if num_colloc_nodes > 1:
        num_interior_nodes: int = num_colloc_nodes - 1
        if num_interior_nodes > 0:
            interior_nodes_var = opti.variable(num_states, num_interior_nodes)
            if interior_nodes_var is None:
                raise ValueError("Failed to create interior_nodes_var")
            for i in range(num_interior_nodes):
                current_interval_state_columns[i + 1] = interior_nodes_var[:, i]

    # Last column is the state at the end of the interval
    current_interval_state_columns[num_colloc_nodes] = state_at_global_mesh_nodes[
        mesh_interval_index + 1
    ]

    # Combine all state columns into a matrix
    state_at_nodes: CasadiMatrix = ca.horzcat(*current_interval_state_columns)

    return state_at_nodes, interior_nodes_var


def _apply_collocation_constraints(
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
    """Apply collocation constraints for a single mesh interval."""
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    diff_matrix: CasadiDM = ca.DM(basis_components.differentiation_matrix)

    # Validate interval length
    validate_interval_length(
        global_normalized_mesh_nodes[mesh_interval_index],
        global_normalized_mesh_nodes[mesh_interval_index + 1],
        mesh_interval_index,
    )

    # Calculate state derivatives at collocation points
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

        # Apply collocation constraint
        opti.subject_to(
            state_derivative_at_colloc[:, i_colloc]
            == tau_to_time_scaling * state_derivative_rhs_vector
        )


def _apply_path_constraints(
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
    """Apply path constraints for a single mesh interval."""
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
        path_constraints_result: list[PathConstraint] | PathConstraint = path_constraints_function(
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


def _setup_integrals(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    integral_integrand_function: IntegralIntegrandCallable,
    problem_parameters: ProblemParameters,
    num_integrals: int,
    accumulated_integral_expressions: list[CasadiMX],
) -> None:
    """Set up integral calculations for a single mesh interval."""
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    quad_weights = basis_components.quadrature_weights.flatten()

    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    tau_to_time_scaling: CasadiMX = (
        (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
    )

    for integral_index in range(num_integrals):
        quad_sum: CasadiMX = ca.MX(0)

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

            # Calculate integrand and add to quadrature sum
            weight: float = quad_weights[i_colloc]
            integrand_value: CasadiMX = integral_integrand_function(
                state_at_colloc,
                control_at_colloc,
                physical_time_at_colloc,
                integral_index,
                problem_parameters,
            )
            quad_sum += weight * integrand_value

        accumulated_integral_expressions[integral_index] += tau_to_time_scaling * quad_sum


def _apply_initial_guess(
    opti: CasadiOpti,
    problem: ProblemProtocol,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    state_at_global_mesh_nodes: ListOfCasadiMX,
    state_interior_vars: list[CasadiMX | None],
    control_variables: ListOfCasadiMX,
    integral_variables: CasadiMX | None,
    num_mesh_intervals: int,
) -> None:
    """Apply initial guess to optimization variables."""
    if problem.initial_guess is None:
        return

    ig: InitialGuess = problem.initial_guess
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Time variables
    if ig.initial_time_variable is not None:
        opti.set_initial(initial_time_variable, ig.initial_time_variable)
    if ig.terminal_time_variable is not None:
        opti.set_initial(terminal_time_variable, ig.terminal_time_variable)

    # States
    if ig.states is not None:
        if len(ig.states) != num_mesh_intervals:
            raise ValueError(
                f"States guess must have {num_mesh_intervals} arrays, got {len(ig.states)}"
            )

        # Global mesh nodes
        for k in range(num_mesh_intervals):
            state_guess_k = ig.states[k]
            expected_shape = (num_states, problem.collocation_points_per_interval[k] + 1)
            if state_guess_k.shape != expected_shape:
                raise ValueError(
                    f"State guess for interval {k} has shape {state_guess_k.shape}, "
                    f"expected {expected_shape}"
                )

            if k == 0:
                opti.set_initial(state_at_global_mesh_nodes[0], state_guess_k[:, 0])
            opti.set_initial(state_at_global_mesh_nodes[k + 1], state_guess_k[:, -1])

        # Interior state approximation nodes
        for k in range(num_mesh_intervals):
            interior_var = state_interior_vars[k]
            if interior_var is not None:
                state_guess_k = ig.states[k]
                num_interior_nodes = interior_var.shape[1]

                if state_guess_k.shape[1] >= num_interior_nodes + 2:
                    interior_guess = state_guess_k[:, 1 : 1 + num_interior_nodes]
                    opti.set_initial(interior_var, interior_guess)
                else:
                    raise ValueError(
                        f"State guess for interval {k} has {state_guess_k.shape[1]} nodes, "
                        f"but needs at least {num_interior_nodes + 2} for interior nodes"
                    )

    # Controls
    if ig.controls is not None:
        if len(ig.controls) != num_mesh_intervals:
            raise ValueError(
                f"Controls guess must have {num_mesh_intervals} arrays, got {len(ig.controls)}"
            )

        for k in range(num_mesh_intervals):
            control_guess_k = ig.controls[k]
            expected_shape = (num_controls, problem.collocation_points_per_interval[k])

            if control_guess_k.shape != expected_shape:
                raise ValueError(
                    f"Control guess for interval {k} has shape {control_guess_k.shape}, "
                    f"expected {expected_shape}"
                )

            opti.set_initial(control_variables[k], control_guess_k)

    # Integrals
    if ig.integrals is not None and num_integrals > 0 and integral_variables is not None:
        validate_and_set_integral_guess(opti, integral_variables, ig.integrals, num_integrals)


def _store_optimization_references(
    opti: CasadiOpti,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    integral_variables: CasadiMX | None,
    state_at_nodes_all_intervals: list[CasadiMatrix],
    control_variables: ListOfCasadiMX,
    local_state_tau_all_intervals: list[FloatArray],
    local_collocation_tau_all_intervals: list[FloatArray],
    global_normalized_mesh_nodes: FloatArray,
    objective_value: CasadiMX,
) -> None:
    """Store references needed for solution extraction."""
    opti.initial_time_variable_reference = initial_time_variable
    opti.terminal_time_variable_reference = terminal_time_variable

    if integral_variables is not None:
        opti.integral_variables_object_reference = integral_variables
    else:
        opti.integral_variables_object_reference = None

    opti.state_at_local_approximation_nodes_all_intervals_variables = state_at_nodes_all_intervals
    opti.control_at_local_collocation_nodes_all_intervals_variables = control_variables
    opti.metadata_local_state_approximation_nodes_tau = local_state_tau_all_intervals
    opti.metadata_local_collocation_nodes_tau = local_collocation_tau_all_intervals
    opti.metadata_global_normalized_mesh_nodes = global_normalized_mesh_nodes
    opti.symbolic_objective_function_reference = objective_value


def solve_single_phase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """
    Solves a single-phase optimal control problem using Radau pseudospectral collocation.

    Args:
        problem: The optimal control problem definition

    Returns:
        An OptimalControlSolution object containing the solution

    Raises:
        ValueError: If problem configuration is invalid
    """
    # Validate problem is properly configured
    if not hasattr(problem, "_mesh_configured") or not problem._mesh_configured:
        raise ValueError(
            "Problem mesh must be explicitly configured before solving. "
            "Call problem.set_mesh(polynomial_degrees, mesh_points)"
        )

    # Validate initial guess if provided
    if problem.initial_guess is not None:
        problem.validate_initial_guess()

    # Initialize optimization problem
    opti: CasadiOpti = ca.Opti()

    # Extract problem data
    num_states: int = len(problem._states)
    # num_controls: int = len(problem._controls)
    num_integrals: int = problem._num_integrals

    if not problem.collocation_points_per_interval:
        raise ValueError("Problem must include 'collocation_points_per_interval'.")

    num_collocation_nodes_per_interval: list[int] = problem.collocation_points_per_interval
    num_mesh_intervals: int = len(num_collocation_nodes_per_interval)

    # Validate mesh configuration
    if problem.global_normalized_mesh_nodes is None:
        raise ValueError("Global normalized mesh nodes must be set")

    global_normalized_mesh_nodes = problem.global_normalized_mesh_nodes
    validate_mesh_configuration(
        num_collocation_nodes_per_interval,
        global_normalized_mesh_nodes,
        num_mesh_intervals,
    )

    # Get problem functions
    dynamics_function: DynamicsCallable = problem.get_dynamics_function()
    objective_function: ObjectiveCallable = problem.get_objective_function()
    path_constraints_function: PathConstraintsCallable | None = (
        problem.get_path_constraints_function()
    )
    event_constraints_function: EventConstraintsCallable | None = (
        problem.get_event_constraints_function()
    )
    integral_integrand_function: IntegralIntegrandCallable | None = problem.get_integrand_function()
    problem_parameters: ProblemParameters = problem._parameters

    # Set up optimization variables
    (
        initial_time_variable,
        terminal_time_variable,
        state_at_global_mesh_nodes_variables,
        control_at_local_collocation_nodes_all_intervals_variables,
        integral_decision_variables,
    ) = _setup_optimization_variables(opti, problem, num_mesh_intervals)

    # Initialize storage for interval data
    state_at_local_approximation_nodes_all_intervals_variables: list[CasadiMatrix] = []
    state_at_interior_local_approximation_nodes_all_intervals_variables: list[CasadiMX | None] = []
    accumulated_integral_expressions: list[CasadiMX] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )
    local_state_approximation_nodes_tau_all_intervals: list[FloatArray] = []
    local_collocation_nodes_tau_all_intervals: list[FloatArray] = []

    # Process each mesh interval
    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes: int = num_collocation_nodes_per_interval[mesh_interval_index]

        # Set up state variables for this interval
        state_at_nodes, interior_nodes_var = _setup_interval_state_variables(
            opti,
            mesh_interval_index,
            num_states,
            num_colloc_nodes,
            state_at_global_mesh_nodes_variables,
        )

        state_at_local_approximation_nodes_all_intervals_variables.append(state_at_nodes)
        state_at_interior_local_approximation_nodes_all_intervals_variables.append(
            interior_nodes_var
        )

        # Get Radau collocation components
        basis_components: RadauBasisComponents = compute_radau_collocation_components(
            num_colloc_nodes
        )
        state_nodes_tau: FloatArray = basis_components.state_approximation_nodes.flatten()
        colloc_nodes_tau: FloatArray = basis_components.collocation_nodes.flatten()

        local_state_approximation_nodes_tau_all_intervals.append(state_nodes_tau)
        local_collocation_nodes_tau_all_intervals.append(colloc_nodes_tau)

        # Apply collocation constraints
        _apply_collocation_constraints(
            opti,
            mesh_interval_index,
            state_at_nodes,
            control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index],
            basis_components,
            global_normalized_mesh_nodes,
            initial_time_variable,
            terminal_time_variable,
            dynamics_function,
            problem_parameters,
        )

        # Apply path constraints if they exist
        if path_constraints_function:
            _apply_path_constraints(
                opti,
                mesh_interval_index,
                state_at_nodes,
                control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index],
                basis_components,
                global_normalized_mesh_nodes,
                initial_time_variable,
                terminal_time_variable,
                path_constraints_function,
                problem_parameters,
            )

        # Set up integrals if they exist
        if num_integrals > 0 and integral_integrand_function:
            _setup_integrals(
                opti,
                mesh_interval_index,
                state_at_nodes,
                control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index],
                basis_components,
                global_normalized_mesh_nodes,
                initial_time_variable,
                terminal_time_variable,
                integral_integrand_function,
                problem_parameters,
                num_integrals,
                accumulated_integral_expressions,
            )

    # Set integral constraints
    if (
        num_integrals > 0
        and integral_integrand_function
        and integral_decision_variables is not None
    ):
        if num_integrals == 1:
            opti.subject_to(integral_decision_variables == accumulated_integral_expressions[0])
        else:
            for i in range(num_integrals):
                opti.subject_to(
                    integral_decision_variables[i] == accumulated_integral_expressions[i]
                )

    # Set objective
    initial_state: CasadiMX = state_at_global_mesh_nodes_variables[0]
    terminal_state: CasadiMX = state_at_global_mesh_nodes_variables[num_mesh_intervals]

    objective_value: CasadiMX = objective_function(
        initial_time_variable,
        terminal_time_variable,
        initial_state,
        terminal_state,
        integral_decision_variables,
        problem_parameters,
    )
    opti.minimize(objective_value)

    # Apply event constraints
    if event_constraints_function:
        event_constraints_result: list[EventConstraint] | EventConstraint = (
            event_constraints_function(
                initial_time_variable,
                terminal_time_variable,
                initial_state,
                terminal_state,
                integral_decision_variables,
                problem_parameters,
            )
        )
        event_constraints_to_apply = (
            event_constraints_result
            if isinstance(event_constraints_result, list)
            else [event_constraints_result]
        )
        for event_constraint in event_constraints_to_apply:
            apply_constraint(opti, event_constraint)

    # Apply initial guess if provided
    _apply_initial_guess(
        opti,
        problem,
        initial_time_variable,
        terminal_time_variable,
        state_at_global_mesh_nodes_variables,
        state_at_interior_local_approximation_nodes_all_intervals_variables,
        control_at_local_collocation_nodes_all_intervals_variables,
        integral_decision_variables,
        num_mesh_intervals,
    )

    # Set solver options
    solver_options_to_use: dict[str, object] = problem.solver_options or {}
    opti.solver("ipopt", solver_options_to_use)

    # Store references for solution extraction
    _store_optimization_references(
        opti,
        initial_time_variable,
        terminal_time_variable,
        integral_decision_variables,
        state_at_local_approximation_nodes_all_intervals_variables,
        control_at_local_collocation_nodes_all_intervals_variables,
        local_state_approximation_nodes_tau_all_intervals,
        local_collocation_nodes_tau_all_intervals,
        global_normalized_mesh_nodes,
        objective_value,
    )

    # Solve the problem
    solution_obj: OptimalControlSolution
    try:
        solver_solution: CasadiOptiSol = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        solution_obj = extract_and_format_solution(
            solver_solution,
            opti,
            problem,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        solution_obj = extract_and_format_solution(
            None,
            opti,
            problem,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
        solution_obj.success = False
        solution_obj.message = f"Solver runtime error: {e}"

        try:
            if hasattr(opti, "debug") and opti.debug is not None:
                if initial_time_variable is not None:
                    solution_obj.initial_time_variable = float(
                        opti.debug.value(initial_time_variable)
                    )
                if terminal_time_variable is not None:
                    solution_obj.terminal_time_variable = float(
                        opti.debug.value(terminal_time_variable)
                    )
        except Exception as debug_e:
            print(f"  Could not retrieve debug values after solver error: {debug_e}")

    # Store mesh information in solution
    solution_obj.num_collocation_nodes_list_at_solve_time = list(num_collocation_nodes_per_interval)
    solution_obj.global_mesh_nodes_at_solve_time = global_normalized_mesh_nodes.copy()
    return solution_obj
