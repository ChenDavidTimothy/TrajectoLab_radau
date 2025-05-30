"""
Core orchestration for the direct pseudospectral solver using Radau collocation.
"""

import logging

import casadi as ca

from ..exceptions import DataIntegrityError, SolutionExtractionError
from ..radau import RadauBasisComponents, compute_radau_collocation_components
from ..solution_extraction import extract_and_format_solution
from ..tl_types import (
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


# Library logger
logger = logging.getLogger(__name__)


def solve_single_phase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """Solve a single-phase optimal control problem using Radau pseudospectral collocation."""

    # Log solver start (DEBUG - developer cares about internal operations)
    logger.debug("Starting Radau collocation solver")

    # Initialize optimization problem
    opti: ca.Opti = ca.Opti()
    num_mesh_intervals = len(problem.collocation_points_per_interval)
    num_integrals = problem._num_integrals

    # Log problem structure (DEBUG)
    logger.debug("Problem structure: intervals=%d, integrals=%d", num_mesh_intervals, num_integrals)

    try:
        # Set up optimization variables
        logger.debug("Setting up optimization variables")
        variables = setup_optimization_variables(opti, problem, num_mesh_intervals)

        # Initialize containers for interval processing
        metadata = MetadataBundle()
        accumulated_integral_expressions: list[ca.MX] = (
            [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
        )

        # Process each mesh interval
        logger.debug("Processing %d mesh intervals", num_mesh_intervals)
        _process_mesh_intervals(
            opti, variables, metadata, problem, num_mesh_intervals, accumulated_integral_expressions
        )

        # Set up objective and event constraints
        logger.debug("Setting up objective and constraints")
        _setup_objective_and_event_constraints(opti, variables, problem, num_mesh_intervals)

        # Apply integral constraints if needed
        if num_integrals > 0 and variables.integral_variables is not None:
            logger.debug("Applying integral constraints: count=%d", num_integrals)
            apply_integral_constraints(
                opti, variables.integral_variables, accumulated_integral_expressions, num_integrals
            )

        # Apply initial guess
        logger.debug("Applying initial guess")
        apply_initial_guess(opti, variables, problem, num_mesh_intervals)

        # Configure solver
        logger.debug("Configuring NLP solver")
        _configure_solver_and_store_references(opti, variables, metadata, problem)

    except Exception as e:
        logger.error("Failed to set up optimization problem: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Failed to set up optimization problem: {e}", "TrajectoLab problem construction error"
        ) from e

    # Execute solve
    logger.debug("Executing NLP solve")
    solution_obj = _execute_solve(opti, problem, num_mesh_intervals)

    return solution_obj


def _process_mesh_intervals(
    opti: ca.Opti,
    variables: VariableReferences,
    metadata: MetadataBundle,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
    accumulated_integral_expressions: list[ca.MX],
) -> None:
    """
    Process each mesh interval to set up constraints and integrals.

    NOTE: Parameter validation assumed already done
    """
    num_states, _ = problem.get_variable_counts()
    num_integrals = problem._num_integrals

    # Get essential functions (already validated that they exist)
    dynamics_function = problem.get_dynamics_function()
    path_constraints_function = problem.get_path_constraints_function()
    integral_integrand_function = problem.get_integrand_function()
    global_mesh_nodes = problem.global_normalized_mesh_nodes

    # Data integrity validation (internal consistency)
    if len(global_mesh_nodes) != num_mesh_intervals + 1:
        raise DataIntegrityError(
            f"Mesh nodes count ({len(global_mesh_nodes)}) doesn't match expected ({num_mesh_intervals + 1})",
            "Mesh configuration inconsistency",
        )

    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = problem.collocation_points_per_interval[mesh_interval_index]

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

            # Get Radau collocation components (validation done in compute_radau_collocation_components)
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
                    num_integrals,
                    accumulated_integral_expressions,
                )

        except Exception as e:
            # Wrap interval processing errors with context
            if isinstance(e, DataIntegrityError):
                raise
            raise DataIntegrityError(
                f"Failed to process mesh interval {mesh_interval_index}: {e}",
                "TrajectoLab interval setup error",
            ) from e

    # Store global mesh nodes
    metadata.global_mesh_nodes = global_mesh_nodes


def _setup_objective_and_event_constraints(
    opti: ca.Opti,
    variables: VariableReferences,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> None:
    """
    Set up the objective function and apply event constraints.

    NOTE: Assumes objective function exists (validated at entry point)
    """
    logger.info("Setting up objective and constraints")

    # Get objective function (already validated that it exists)
    objective_function = problem.get_objective_function()

    # Set up objective
    initial_state: ca.MX = variables.state_at_mesh_nodes[0]
    terminal_state: ca.MX = variables.state_at_mesh_nodes[num_mesh_intervals]

    try:
        objective_value: ca.MX = objective_function(
            variables.initial_time,
            variables.terminal_time,
            initial_state,
            terminal_state,
            variables.integral_variables,
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
    opti: ca.Opti,
    variables: VariableReferences,
    metadata: MetadataBundle,
    problem: ProblemProtocol,
) -> None:
    """Configure solver options and store references for solution extraction."""
    # Set solver options (already validated)
    solver_options_to_use: dict[str, object] = problem.solver_options or {}

    try:
        opti.solver("ipopt", solver_options_to_use)
    except Exception as e:
        raise DataIntegrityError(
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
    )
    opti.symbolic_objective_function_reference = objective_expression


def _execute_solve(
    opti: ca.Opti, problem: ProblemProtocol, num_mesh_intervals: int
) -> OptimalControlSolution:
    """Execute the solve and handle results."""
    global_mesh_nodes = problem.global_normalized_mesh_nodes
    collocation_points = problem.collocation_points_per_interval

    try:
        # Attempt solve
        solver_solution: ca.OptiSol = opti.solve()

        # Log successful solve (DEBUG - internal operation success)
        logger.debug("NLP solver completed successfully")

        # Extract solution
        try:
            solution_obj = extract_and_format_solution(
                solver_solution, opti, problem, collocation_points, global_mesh_nodes
            )
            logger.debug("Solution extraction completed")
        except Exception as e:
            logger.error("Solution extraction failed: %s", str(e))
            raise SolutionExtractionError(
                f"Failed to extract solution: {e}", "TrajectoLab solution processing error"
            ) from e

    except RuntimeError as e:
        # Log solver failure (WARNING - recoverable, solution object still created)
        logger.warning("NLP solver failed: %s", str(e))

        # Create failed solution object
        try:
            solution_obj = extract_and_format_solution(
                None, opti, problem, collocation_points, global_mesh_nodes
            )
        except Exception as extract_error:
            logger.error(
                "Critical: Solution extraction failed after solver failure: %s", str(extract_error)
            )
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
                logger.debug("Retrieved debug values from failed solve")
        except Exception:
            logger.debug("Could not extract debug values from failed solve")

    # Store mesh information in solution
    solution_obj.num_collocation_nodes_list_at_solve_time = list(collocation_points)
    solution_obj.global_mesh_nodes_at_solve_time = global_mesh_nodes.copy()

    return solution_obj
