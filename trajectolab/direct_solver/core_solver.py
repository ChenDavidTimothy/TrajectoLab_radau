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
)
from .initial_guess_solver import apply_initial_guess
from .integrals_solver import apply_integral_constraints, setup_integrals
from .types_solver import MetadataBundle, VariableReferences
from .variables_solver import setup_interval_state_variables, setup_optimization_variables


def solve_single_phase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """
    Solve a single-phase optimal control problem using Radau pseudospectral collocation.

    Args:
        problem: The optimal control problem definition

    Returns:
        An OptimalControlSolution object containing the solution

    Raises:
        ValueError: If problem configuration is invalid
        RuntimeError: If solver fails
    """
    # Validate problem configuration
    _validate_problem_configuration(problem)

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
    _process_mesh_intervals(
        opti, variables, metadata, problem, num_mesh_intervals, accumulated_integral_expressions
    )

    # Set up objective and event constraints
    _setup_objective_and_event_constraints(opti, variables, problem, num_mesh_intervals)

    # Apply integral constraints if needed
    if num_integrals > 0 and variables.integral_variables is not None:
        apply_integral_constraints(
            opti, variables.integral_variables, accumulated_integral_expressions, num_integrals
        )

    # Apply initial guess
    apply_initial_guess(opti, variables, problem, num_mesh_intervals)

    # Configure solver and store references
    _configure_solver_and_store_references(opti, variables, metadata, problem)

    # Execute solve
    solution_obj = _execute_solve(opti, problem, num_mesh_intervals)

    return solution_obj


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
            problem,
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
    print("\n=== SETTING UP OBJECTIVE AND CONSTRAINTS ===")

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
