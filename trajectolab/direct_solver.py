from __future__ import annotations

from collections.abc import Sequence

import casadi as ca

from .constraint_utils import apply_constraint
from .input_validation import (
    validate_and_set_integral_guess,
    validate_dynamics_output,
    validate_mesh_configuration,
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

    opti: CasadiOpti = ca.Opti()

    # Extract necessary problem data
    num_states: int = len(problem._states)
    num_controls: int = len(problem._controls)
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

    # Get vectorized functions directly from problem
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

    # Create time variables
    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()
    opti.subject_to(initial_time_variable >= problem._t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem._t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem._tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem._tf_bounds[1])
    opti.subject_to(terminal_time_variable > initial_time_variable + 1e-6)

    # Create state variables
    state_at_global_mesh_nodes_variables: ListOfCasadiMX = [
        opti.variable(num_states) for _ in range(num_mesh_intervals + 1)
    ]
    state_at_local_approximation_nodes_all_intervals_variables: list[CasadiMatrix] = []
    state_at_interior_local_approximation_nodes_all_intervals_variables: list[CasadiMX | None] = []

    # Create control variables
    control_at_local_collocation_nodes_all_intervals_variables: ListOfCasadiMX = [
        opti.variable(num_controls, num_collocation_nodes_per_interval[k])
        for k in range(num_mesh_intervals)
    ]

    # Create integral variables
    integral_decision_variables: CasadiMX | None = None
    if num_integrals > 0:
        integral_decision_variables = (
            opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        )

    accumulated_integral_expressions: list[CasadiMX] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )
    local_state_approximation_nodes_tau_all_intervals: list[FloatArray] = []
    local_collocation_nodes_tau_all_intervals: list[FloatArray] = []

    # Process each mesh interval
    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes: int = num_collocation_nodes_per_interval[mesh_interval_index]
        current_interval_state_columns: list[CasadiMX] = [
            ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)
        ]
        current_interval_state_columns[0] = state_at_global_mesh_nodes_variables[
            mesh_interval_index
        ]

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

        state_at_interior_local_approximation_nodes_all_intervals_variables.append(
            interior_nodes_var
        )
        current_interval_state_columns[num_colloc_nodes] = state_at_global_mesh_nodes_variables[
            mesh_interval_index + 1
        ]

        state_at_nodes: CasadiMatrix = ca.horzcat(*current_interval_state_columns)
        state_at_local_approximation_nodes_all_intervals_variables.append(state_at_nodes)

        # Get Radau collocation components
        basis_components: RadauBasisComponents = compute_radau_collocation_components(
            num_colloc_nodes
        )
        state_nodes_tau: FloatArray = basis_components.state_approximation_nodes.flatten()
        colloc_nodes_tau: FloatArray = basis_components.collocation_nodes.flatten()
        quad_weights: FloatArray = basis_components.quadrature_weights.flatten()
        diff_matrix: CasadiDM = ca.DM(basis_components.differentiation_matrix)

        local_state_approximation_nodes_tau_all_intervals.append(state_nodes_tau)
        local_collocation_nodes_tau_all_intervals.append(colloc_nodes_tau)

        # Apply collocation constraints
        state_derivative_at_colloc: CasadiMX = ca.mtimes(state_at_nodes, diff_matrix.T)

        global_segment_length: float = (
            global_normalized_mesh_nodes[mesh_interval_index + 1]
            - global_normalized_mesh_nodes[mesh_interval_index]
        )

        if global_segment_length <= 1e-9:
            raise ValueError(
                f"Mesh interval {mesh_interval_index} has zero or negative length: {global_segment_length}"
            )

        tau_to_time_scaling: CasadiMX = (
            (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
        )

        # Process each collocation point
        for i_colloc in range(num_colloc_nodes):
            state_at_colloc: CasadiMX = state_at_nodes[:, i_colloc]
            control_at_colloc: CasadiMX = (
                control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index][
                    :, i_colloc
                ]
            )

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

            # Apply dynamics constraint
            state_derivative_rhs: list[CasadiMX] | CasadiMX | Sequence[CasadiMX] = (
                dynamics_function(
                    state_at_colloc, control_at_colloc, physical_time_at_colloc, problem_parameters
                )
            )
            state_derivative_rhs_vector: CasadiMX = validate_dynamics_output(
                state_derivative_rhs, num_states
            )

            opti.subject_to(
                state_derivative_at_colloc[:, i_colloc]
                == tau_to_time_scaling * state_derivative_rhs_vector
            )

            # Apply path constraints
            if path_constraints_function:
                path_constraints_result: list[PathConstraint] | PathConstraint = (
                    path_constraints_function(
                        state_at_colloc,
                        control_at_colloc,
                        physical_time_at_colloc,
                        problem_parameters,
                    )
                )
                constraints_to_apply = (
                    path_constraints_result
                    if isinstance(path_constraints_result, list)
                    else [path_constraints_result]
                )
                for constraint in constraints_to_apply:
                    apply_constraint(opti, constraint)

        # Process integrals
        if num_integrals > 0 and integral_integrand_function:
            for integral_index in range(num_integrals):
                quad_sum: CasadiMX = ca.MX(0)
                for i_colloc in range(num_colloc_nodes):
                    state_at_colloc_for_integral: CasadiMX = state_at_nodes[:, i_colloc]
                    control_at_colloc_for_integral: CasadiMX = (
                        control_at_local_collocation_nodes_all_intervals_variables[
                            mesh_interval_index
                        ][:, i_colloc]
                    )

                    local_colloc_tau_val_for_integral: float = colloc_nodes_tau[i_colloc]
                    global_colloc_tau_val_for_integral: CasadiMX = (
                        global_segment_length / 2 * local_colloc_tau_val_for_integral
                        + (
                            global_normalized_mesh_nodes[mesh_interval_index + 1]
                            + global_normalized_mesh_nodes[mesh_interval_index]
                        )
                        / 2
                    )
                    physical_time_at_colloc_for_integral: CasadiMX = (
                        terminal_time_variable - initial_time_variable
                    ) / 2 * global_colloc_tau_val_for_integral + (
                        terminal_time_variable + initial_time_variable
                    ) / 2

                    weight: float = quad_weights[i_colloc]
                    integrand_value: CasadiMX = integral_integrand_function(
                        state_at_colloc_for_integral,
                        control_at_colloc_for_integral,
                        physical_time_at_colloc_for_integral,
                        integral_index,
                        problem_parameters,
                    )
                    quad_sum += weight * integrand_value
                accumulated_integral_expressions[integral_index] += tau_to_time_scaling * quad_sum

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
    if problem.initial_guess is not None:
        ig: InitialGuess = problem.initial_guess

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
                expected_shape = (num_states, num_collocation_nodes_per_interval[k] + 1)
                if state_guess_k.shape != expected_shape:
                    raise ValueError(
                        f"State guess for interval {k} has shape {state_guess_k.shape}, "
                        f"expected {expected_shape}"
                    )

                if k == 0:
                    opti.set_initial(state_at_global_mesh_nodes_variables[0], state_guess_k[:, 0])
                opti.set_initial(state_at_global_mesh_nodes_variables[k + 1], state_guess_k[:, -1])

            # Interior state approximation nodes
            for k in range(num_mesh_intervals):
                interior_var = state_at_interior_local_approximation_nodes_all_intervals_variables[
                    k
                ]
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
                expected_shape = (num_controls, num_collocation_nodes_per_interval[k])

                if control_guess_k.shape != expected_shape:
                    raise ValueError(
                        f"Control guess for interval {k} has shape {control_guess_k.shape}, "
                        f"expected {expected_shape}"
                    )

                opti.set_initial(
                    control_at_local_collocation_nodes_all_intervals_variables[k], control_guess_k
                )

        # Integrals
        if (
            ig.integrals is not None
            and num_integrals > 0
            and integral_decision_variables is not None
        ):
            validate_and_set_integral_guess(
                opti, integral_decision_variables, ig.integrals, num_integrals
            )

    # Set solver options
    solver_options_to_use: dict[str, object] = problem.solver_options or {}
    opti.solver("ipopt", solver_options_to_use)

    # Store references for solution extraction
    opti.initial_time_variable_reference = initial_time_variable
    opti.terminal_time_variable_reference = terminal_time_variable
    if integral_decision_variables is not None:
        opti.integral_variables_object_reference = integral_decision_variables
    else:
        opti.integral_variables_object_reference = None

    opti.state_at_local_approximation_nodes_all_intervals_variables = (
        state_at_local_approximation_nodes_all_intervals_variables
    )
    opti.control_at_local_collocation_nodes_all_intervals_variables = (
        control_at_local_collocation_nodes_all_intervals_variables
    )
    opti.metadata_local_state_approximation_nodes_tau = (
        local_state_approximation_nodes_tau_all_intervals
    )
    opti.metadata_local_collocation_nodes_tau = local_collocation_nodes_tau_all_intervals
    opti.metadata_global_normalized_mesh_nodes = global_normalized_mesh_nodes
    opti.symbolic_objective_function_reference = objective_value

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
