"""
Core orchestration for the direct solver.
"""

from typing import cast

import casadi as ca

from ..radau import RadauBasisComponents, compute_radau_collocation_components
from ..solution_extraction import extract_and_format_solution
from ..tl_types import (
    CasadiMX,
    CasadiOpti,
    CasadiOptiSol,
    FloatArray,
    OptimalControlSolution,
    ProblemProtocol,
)
from .constraints_solver import (
    apply_collocation_constraints,
    apply_event_constraints,
    apply_path_constraints,
    apply_scaled_collocation_constraints,
)
from .initial_guess_solver import apply_initial_guess
from .integrals_solver import apply_integral_constraints, setup_integrals
from .types_solver import MetadataBundle, VariableReferences
from .variables_solver import setup_interval_state_variables, setup_optimization_variables


def solve_single_phase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """Solve a single-phase optimal control problem using Radau pseudospectral collocation."""
    # Validate problem configuration
    _validate_problem_configuration(problem)

    # Initialize scaling ONCE if enabled
    scaling_manager = None
    if getattr(problem, "scaling_enabled", True):
        if hasattr(problem, "initialize_scaling"):
            problem.initialize_scaling()
        if hasattr(problem, "scaling_manager"):
            scaling_manager = problem.scaling_manager

    # Initialize optimization problem
    opti: CasadiOpti = ca.Opti()

    # Extract problem metadata
    num_mesh_intervals = len(problem.collocation_points_per_interval)
    num_integrals = problem._num_integrals

    # Set up optimization variables
    variables = setup_optimization_variables(opti, problem, num_mesh_intervals)

    # Initialize containers for interval processing
    metadata = MetadataBundle()
    accumulated_integral_expressions: list[CasadiMX] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )

    # Process each mesh interval
    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = problem.collocation_points_per_interval[mesh_interval_index]

        # Set up state variables for this interval
        state_at_nodes, interior_nodes_var = setup_interval_state_variables(
            opti,
            mesh_interval_index,
            len(problem._states),
            num_colloc_nodes,
            variables.state_at_mesh_nodes,
        )

        # Store state variables and interior nodes
        variables.state_matrices.append(state_at_nodes)
        variables.interior_variables.append(interior_nodes_var)

        # Get Radau collocation components
        basis_components = compute_radau_collocation_components(num_colloc_nodes)

        # Store metadata
        metadata.local_state_tau.append(basis_components.state_approximation_nodes)
        metadata.local_control_tau.append(basis_components.collocation_nodes)

        # Apply collocation constraints with scaling if enabled
        if scaling_manager is not None and scaling_manager.is_initialized:
            # Use scaled collocation constraints (Rule 3)
            apply_scaled_collocation_constraints(
                opti,
                mesh_interval_index,
                state_at_nodes,
                variables.control_variables[mesh_interval_index],
                basis_components,
                problem.global_normalized_mesh_nodes,
                variables.initial_time,
                variables.terminal_time,
                problem.get_dynamics_function(),
                problem._parameters,
                scaling_manager,
            )
        else:
            # Use unscaled collocation constraints
            apply_collocation_constraints(
                opti,
                mesh_interval_index,
                state_at_nodes,
                variables.control_variables[mesh_interval_index],
                basis_components,
                problem.global_normalized_mesh_nodes,
                variables.initial_time,
                variables.terminal_time,
                problem.get_dynamics_function(),
                problem._parameters,
            )

        # Apply path constraints (unscaled for now)
        path_constraints_function = problem.get_path_constraints_function()
        if path_constraints_function is not None:
            apply_path_constraints(
                opti,
                mesh_interval_index,
                state_at_nodes,
                variables.control_variables[mesh_interval_index],
                basis_components,
                problem.global_normalized_mesh_nodes,
                variables.initial_time,
                variables.terminal_time,
                path_constraints_function,
                problem._parameters,
            )

        # Set up integrals (unscaled for now)
        integral_integrand_function = problem.get_integrand_function()
        if num_integrals > 0 and integral_integrand_function is not None:
            setup_integrals(
                opti,
                mesh_interval_index,
                state_at_nodes,
                variables.control_variables[mesh_interval_index],
                basis_components,
                problem.global_normalized_mesh_nodes,
                variables.initial_time,
                variables.terminal_time,
                integral_integrand_function,
                problem._parameters,
                num_integrals,
                accumulated_integral_expressions,
            )

    # Store global mesh nodes
    metadata.global_mesh_nodes = cast(FloatArray, problem.global_normalized_mesh_nodes)

    # Set up objective and event constraints
    _setup_objective_and_event_constraints(opti, variables, problem, num_mesh_intervals)

    # Apply integral constraints if needed
    if num_integrals > 0 and variables.integral_variables is not None:
        apply_integral_constraints(
            opti, variables.integral_variables, accumulated_integral_expressions, num_integrals
        )

    # Apply initial guess with proper scaling
    if scaling_manager is not None and scaling_manager.is_initialized:
        # Import function only when needed
        from .initial_guess_solver import apply_scaled_initial_guess

        apply_scaled_initial_guess(opti, variables, problem, num_mesh_intervals, scaling_manager)
    else:
        apply_initial_guess(opti, variables, problem, num_mesh_intervals)

    # Configure solver and store references
    _configure_solver_and_store_references(opti, variables, metadata, problem)

    # Execute solve
    solution = _execute_solve(opti, problem, num_mesh_intervals)

    # Unscale solution values if scaling was used
    if scaling_manager is not None and scaling_manager.is_initialized and solution.success:
        _unscale_solution(solution, scaling_manager)

    return solution


def _unscale_solution(solution: OptimalControlSolution, scaling_manager) -> None:
    """
    Unscale solution values to convert from scaled variables back to physical units.

    Args:
        solution: Solution with scaled values
        scaling_manager: Scaling manager with scaling factors
    """
    # Unscale state trajectories
    if solution.states and solution.time_states is not None:
        for i in range(len(solution.states)):
            if i < len(scaling_manager.state_scales):
                state_traj = solution.states[i]
                solution.states[i] = (
                    state_traj - scaling_manager.state_shifts[i]
                ) / scaling_manager.state_scales[i]

    # Unscale control trajectories
    if solution.controls and solution.time_controls is not None:
        for i in range(len(solution.controls)):
            if i < len(scaling_manager.control_scales):
                control_traj = solution.controls[i]
                solution.controls[i] = (
                    control_traj - scaling_manager.control_shifts[i]
                ) / scaling_manager.control_scales[i]

    # Unscale solved state trajectories if available
    if solution.solved_state_trajectories_per_interval is not None:
        for k in range(len(solution.solved_state_trajectories_per_interval)):
            scaled_states = solution.solved_state_trajectories_per_interval[k]
            solution.solved_state_trajectories_per_interval[k] = scaling_manager.unscale_state(
                scaled_states
            )

    # Unscale solved control trajectories if available
    if solution.solved_control_trajectories_per_interval is not None:
        for k in range(len(solution.solved_control_trajectories_per_interval)):
            scaled_controls = solution.solved_control_trajectories_per_interval[k]
            solution.solved_control_trajectories_per_interval[k] = scaling_manager.unscale_control(
                scaled_controls
            )


def _validate_problem_configuration(problem: ProblemProtocol) -> None:
    """Validate that the problem is properly configured."""
    if not hasattr(problem, "_mesh_configured") or not problem._mesh_configured:
        raise ValueError(
            "Problem mesh must be explicitly configured before solving. "
            "Call problem.set_mesh(polynomial_degrees, mesh_points)"
        )

    if problem.initial_guess is not None:
        problem.validate_initial_guess()

    if not problem.collocation_points_per_interval:
        raise ValueError("Problem must include 'collocation_points_per_interval'.")

    if problem.global_normalized_mesh_nodes is None:
        raise ValueError("Global normalized mesh nodes must be set")


def _process_mesh_intervals(
    opti: CasadiOpti,
    variables: VariableReferences,
    metadata: MetadataBundle,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
    accumulated_integral_expressions: list[CasadiMX],
) -> None:
    """Process each mesh interval to set up constraints and integrals."""
    num_states = len(problem._states)
    num_integrals = problem._num_integrals
    dynamics_function = problem.get_dynamics_function()
    path_constraints_function = problem.get_path_constraints_function()
    integral_integrand_function = problem.get_integrand_function()
    global_mesh_nodes = cast(FloatArray, problem.global_normalized_mesh_nodes)

    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = problem.collocation_points_per_interval[mesh_interval_index]

        # Set up state variables for this interval
        state_at_nodes, interior_nodes_var = setup_interval_state_variables(
            opti, mesh_interval_index, num_states, num_colloc_nodes, variables.state_at_mesh_nodes
        )

        # Store state variables and interior nodes
        variables.state_matrices.append(state_at_nodes)
        variables.interior_variables.append(interior_nodes_var)

        # Get Radau collocation components
        basis_components: RadauBasisComponents = compute_radau_collocation_components(
            num_colloc_nodes
        )

        # Store metadata
        metadata.local_state_tau.append(basis_components.state_approximation_nodes)
        metadata.local_control_tau.append(basis_components.collocation_nodes)

        # Apply collocation constraints
        apply_collocation_constraints(
            opti,
            mesh_interval_index,
            state_at_nodes,
            variables.control_variables[mesh_interval_index],
            basis_components,
            global_mesh_nodes,
            variables.initial_time,
            variables.terminal_time,
            dynamics_function,
            problem._parameters,
        )

        # Apply path constraints if they exist
        if path_constraints_function is not None:
            apply_path_constraints(
                opti,
                mesh_interval_index,
                state_at_nodes,
                variables.control_variables[mesh_interval_index],
                basis_components,
                global_mesh_nodes,
                variables.initial_time,
                variables.terminal_time,
                path_constraints_function,
                problem._parameters,
            )

        # Set up integrals if they exist
        if num_integrals > 0 and integral_integrand_function is not None:
            setup_integrals(
                opti,
                mesh_interval_index,
                state_at_nodes,
                variables.control_variables[mesh_interval_index],
                basis_components,
                global_mesh_nodes,
                variables.initial_time,
                variables.terminal_time,
                integral_integrand_function,
                problem._parameters,
                num_integrals,
                accumulated_integral_expressions,
            )

    # Store global mesh nodes
    metadata.global_mesh_nodes = global_mesh_nodes


def _setup_objective_and_event_constraints(
    opti: CasadiOpti,
    variables: VariableReferences,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> None:
    """Set up the objective function and apply event constraints."""
    # Get objective function
    objective_function = problem.get_objective_function()

    # Set up objective
    initial_state: CasadiMX = variables.state_at_mesh_nodes[0]
    terminal_state: CasadiMX = variables.state_at_mesh_nodes[num_mesh_intervals]

    objective_value: CasadiMX = objective_function(
        variables.initial_time,
        variables.terminal_time,
        initial_state,
        terminal_state,
        variables.integral_variables,
        problem._parameters,
    )

    opti.minimize(objective_value)

    # Apply event constraints
    apply_event_constraints(
        opti,
        variables.initial_time,
        variables.terminal_time,
        initial_state,
        terminal_state,
        variables.integral_variables,
        problem,
    )


def _configure_solver_and_store_references(
    opti: CasadiOpti,
    variables: VariableReferences,
    metadata: MetadataBundle,
    problem: ProblemProtocol,
) -> None:
    """Configure solver options and store references for solution extraction."""
    # Set solver options
    solver_options_to_use: dict[str, object] = problem.solver_options or {}
    opti.solver("ipopt", solver_options_to_use)

    # Store references for solution extraction
    opti.initial_time_variable_reference = variables.initial_time
    opti.terminal_time_variable_reference = variables.terminal_time
    opti.integral_variables_object_reference = variables.integral_variables
    opti.state_at_local_approximation_nodes_all_intervals_variables = variables.state_matrices
    opti.control_at_local_collocation_nodes_all_intervals_variables = variables.control_variables
    opti.metadata_local_state_approximation_nodes_tau = metadata.local_state_tau
    opti.metadata_local_collocation_nodes_tau = metadata.local_control_tau
    opti.metadata_global_normalized_mesh_nodes = metadata.global_mesh_nodes

    # Get objective expression for storage
    objective_function = problem.get_objective_function()
    num_mesh_intervals = len(problem.collocation_points_per_interval)
    initial_state = variables.state_at_mesh_nodes[0]
    terminal_state = variables.state_at_mesh_nodes[num_mesh_intervals]

    objective_expression = objective_function(
        variables.initial_time,
        variables.terminal_time,
        initial_state,
        terminal_state,
        variables.integral_variables,
        problem._parameters,
    )
    opti.symbolic_objective_function_reference = objective_expression


def _execute_solve(
    opti: CasadiOpti, problem: ProblemProtocol, num_mesh_intervals: int
) -> OptimalControlSolution:
    """Execute the solve and handle results."""
    global_mesh_nodes = cast(FloatArray, problem.global_normalized_mesh_nodes)
    collocation_points = problem.collocation_points_per_interval

    try:
        solver_solution: CasadiOptiSol = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        solution_obj = extract_and_format_solution(
            solver_solution, opti, problem, collocation_points, global_mesh_nodes
        )
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        solution_obj = extract_and_format_solution(
            None, opti, problem, collocation_points, global_mesh_nodes
        )
        solution_obj.success = False
        solution_obj.message = f"Solver runtime error: {e}"

        # Try to retrieve debug values if available
        try:
            if hasattr(opti, "debug") and opti.debug is not None:
                if hasattr(opti, "initial_time_variable_reference"):
                    solution_obj.initial_time_variable = float(
                        opti.debug.value(opti.initial_time_variable_reference)
                    )
                if hasattr(opti, "terminal_time_variable_reference"):
                    solution_obj.terminal_time_variable = float(
                        opti.debug.value(opti.terminal_time_variable_reference)
                    )
        except Exception as debug_e:
            print(f"  Could not retrieve debug values after solver error: {debug_e}")

    # Store mesh information in solution
    solution_obj.num_collocation_nodes_list_at_solve_time = list(collocation_points)
    solution_obj.global_mesh_nodes_at_solve_time = global_mesh_nodes.copy()

    return solution_obj
