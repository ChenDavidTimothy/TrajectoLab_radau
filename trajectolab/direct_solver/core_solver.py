"""
Core orchestration for the direct solver - ENHANCED WITH FAIL-FAST.
Added targeted error handling for critical TrajectoLab operations.
"""

import logging
from typing import cast

import casadi as ca

from ..exceptions import ConfigurationError, DataIntegrityError, SolutionExtractionError
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


logger = logging.getLogger(__name__)


def solve_single_phase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """
    Solve a single-phase optimal control problem using Radau pseudospectral collocation.

    Enhanced with fail-fast error handling for critical TrajectoLab operations.
    """
    # Critical validation with fail-fast
    _validate_problem_configuration(problem)

    # Initialize optimization problem
    opti: CasadiOpti = ca.Opti()

    # Extract problem metadata with validation
    num_mesh_intervals = len(problem.collocation_points_per_interval)
    num_integrals = problem._num_integrals

    # Guard clause: Check for valid mesh intervals
    if num_mesh_intervals == 0:
        raise ConfigurationError(
            "Problem has no mesh intervals configured",
            "Call problem.set_mesh() with valid polynomial degrees and mesh points",
        )

    try:
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

    except Exception as e:
        # If this is already a TrajectoLab error, re-raise it
        if isinstance(e, ConfigurationError | DataIntegrityError):
            raise
        # Otherwise, wrap as DataIntegrityError for TrajectoLab setup failures
        raise DataIntegrityError(
            f"Failed to set up optimization problem: {e}", "TrajectoLab problem construction error"
        ) from e

    # Execute solve
    solution_obj = _execute_solve(opti, problem, num_mesh_intervals)

    return solution_obj


def _validate_problem_configuration(problem: ProblemProtocol) -> None:
    """
    Validate that the problem is properly configured with fail-fast.
    """
    # Guard clause: Mesh configuration is essential
    if not hasattr(problem, "_mesh_configured") or not problem._mesh_configured:
        raise ConfigurationError(
            "Problem mesh must be explicitly configured before solving",
            "Call problem.set_mesh(polynomial_degrees, mesh_points)",
        )

    # Guard clause: Initial guess validation if present
    if problem.initial_guess is not None:
        try:
            problem.validate_initial_guess()
        except Exception as e:
            raise ConfigurationError(
                f"Initial guess validation failed: {e}", "Fix initial guess or mesh configuration"
            ) from e

    # Guard clause: Essential configuration
    if not problem.collocation_points_per_interval:
        raise ConfigurationError(
            "Problem must include 'collocation_points_per_interval'", "Internal configuration error"
        )

    if problem.global_normalized_mesh_nodes is None:
        raise ConfigurationError(
            "Global normalized mesh nodes must be set", "Internal mesh configuration error"
        )


def _process_mesh_intervals(
    opti: CasadiOpti,
    variables: VariableReferences,
    metadata: MetadataBundle,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
    accumulated_integral_expressions: list[CasadiMX],
) -> None:
    """
    Process each mesh interval to set up constraints and integrals.
    Enhanced with data integrity validation.
    """
    num_states, _ = problem.get_variable_counts()
    num_integrals = problem._num_integrals

    # Guard clause: Get essential functions
    dynamics_function = problem.get_dynamics_function()
    if dynamics_function is None:
        raise ConfigurationError(
            "Dynamics function is not defined", "Define problem dynamics before solving"
        )

    path_constraints_function = problem.get_path_constraints_function()
    integral_integrand_function = problem.get_integrand_function()
    global_mesh_nodes = cast(FloatArray, problem.global_normalized_mesh_nodes)

    # Critical validation of mesh nodes
    if len(global_mesh_nodes) != num_mesh_intervals + 1:
        raise DataIntegrityError(
            f"Mesh nodes count ({len(global_mesh_nodes)}) doesn't match expected ({num_mesh_intervals + 1})",
            "Mesh configuration inconsistency",
        )

    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = problem.collocation_points_per_interval[mesh_interval_index]

        # Guard clause: Valid collocation nodes
        if num_colloc_nodes <= 0:
            raise ConfigurationError(
                f"Invalid number of collocation nodes ({num_colloc_nodes}) for interval {mesh_interval_index}",
                "Polynomial degree must be positive",
            )

        try:
            # Set up state variables for this interval
            state_at_nodes, interior_nodes_var = setup_interval_state_variables(
                opti,
                mesh_interval_index,
                num_states,
                num_colloc_nodes,
                variables.state_at_mesh_nodes,
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

        except Exception as e:
            # Wrap interval processing errors with context
            if isinstance(e, ConfigurationError | DataIntegrityError):
                raise
            raise DataIntegrityError(
                f"Failed to process mesh interval {mesh_interval_index}: {e}",
                "TrajectoLab interval setup error",
            ) from e

    # Store global mesh nodes
    metadata.global_mesh_nodes = global_mesh_nodes


def _setup_objective_and_event_constraints(
    opti: CasadiOpti,
    variables: VariableReferences,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> None:
    """
    Set up the objective function and apply event constraints.
    """
    logger.info("Setting up objective and constraints")

    # Guard clause: Objective function is required
    objective_function = problem.get_objective_function()
    if objective_function is None:
        raise ConfigurationError(
            "Objective function is not defined", "Define problem objective before solving"
        )

    # Set up objective
    initial_state: CasadiMX = variables.state_at_mesh_nodes[0]
    terminal_state: CasadiMX = variables.state_at_mesh_nodes[num_mesh_intervals]

    try:
        objective_value: CasadiMX = objective_function(
            variables.initial_time,
            variables.terminal_time,
            initial_state,
            terminal_state,
            variables.integral_variables,
            problem._parameters,
        )

        opti.minimize(objective_value)

    except Exception as e:
        raise DataIntegrityError(
            f"Failed to set up objective function: {e}", "Objective function evaluation error"
        ) from e

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
    """
    Configure solver options and store references for solution extraction.
    """
    # Set solver options
    solver_options_to_use: dict[str, object] = problem.solver_options or {}

    try:
        opti.solver("ipopt", solver_options_to_use)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to configure solver: {e}", "Invalid solver options"
        ) from e

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
    """
    Execute the solve and handle results with enhanced error handling.
    """
    global_mesh_nodes = cast(FloatArray, problem.global_normalized_mesh_nodes)
    collocation_points = problem.collocation_points_per_interval

    try:
        solver_solution: CasadiOptiSol = opti.solve()
        logger.info("NLP problem solved successfully")

        # Enhanced solution extraction with error handling
        try:
            solution_obj = extract_and_format_solution(
                solver_solution, opti, problem, collocation_points, global_mesh_nodes
            )
        except Exception as e:
            raise SolutionExtractionError(
                f"Failed to extract solution: {e}", "TrajectoLab solution processing error"
            ) from e

    except RuntimeError as e:
        logger.error(f"Solver failed: {e}")

        # Create failed solution object
        try:
            solution_obj = extract_and_format_solution(
                None, opti, problem, collocation_points, global_mesh_nodes
            )
        except Exception as extract_error:
            raise SolutionExtractionError(
                f"Failed to extract solution after solver failure: {extract_error}",
                "Critical solution extraction error",
            ) from extract_error

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
        except Exception as e:
            # Debug value extraction is not critical, but log the exception
            logger.warning("Failed to extract debug values: %s", e)

    # Store mesh information in solution
    solution_obj.num_collocation_nodes_list_at_solve_time = list(collocation_points)
    solution_obj.global_mesh_nodes_at_solve_time = global_mesh_nodes.copy()

    return solution_obj
