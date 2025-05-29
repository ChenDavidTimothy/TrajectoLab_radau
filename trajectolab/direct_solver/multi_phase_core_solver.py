"""
Core orchestration for the multi-phase direct pseudospectral solver using Radau collocation.

This module implements the faithful CGPOPS multi-phase solver structure, creating
the unified NLP from CGPOPS Equation (31) with proper block-diagonal constraint
organization from Section 4.2.

The solver creates a single optimization problem containing all phases:
z = [z^(1), ..., z^(P), s₁, ..., sₙₛ]ᵀ

With constraint structure maintaining the block-diagonal sparsity pattern essential
for computational efficiency in large multi-phase problems.
"""

import logging

import casadi as ca

from ..exceptions import ConfigurationError, DataIntegrityError, SolutionExtractionError
from ..tl_types import MultiPhaseOptimalControlSolution, MultiPhaseProblemProtocol
from .multi_phase_constraints_solver import apply_all_multi_phase_constraints
from .multi_phase_variables_solver import (
    create_multi_phase_metadata_bundle,
    setup_multi_phase_interval_state_variables,
    setup_multi_phase_optimization_variables,
    validate_multi_phase_variable_structure,
)
from .types_solver import MultiPhaseMetadataBundle, MultiPhaseVariableReferences


# Library logger
logger = logging.getLogger(__name__)


def solve_multi_phase_radau_collocation(
    problem: MultiPhaseProblemProtocol,
) -> MultiPhaseOptimalControlSolution:
    """
    Solve a multi-phase optimal control problem using Radau pseudospectral collocation.

    Implements the faithful CGPOPS multi-phase solver methodology from Section 2,
    creating a unified NLP with hierarchical decision vector structure from Equation (31)
    and block-diagonal constraint organization from Section 4.2.

    The solver creates a single optimization problem containing:
    - All phase variables z^(1), ..., z^(P)
    - Global static parameters s₁, ..., sₙₛ
    - Phase-internal constraints (diagonal blocks)
    - Inter-phase event constraints (off-diagonal coupling)
    - Multi-phase objective function φ(E^(1), ..., E^(P), s)

    Args:
        problem: Multi-phase problem protocol with validated structure

    Returns:
        MultiPhaseOptimalControlSolution containing complete multi-phase results

    Raises:
        ConfigurationError: If problem structure is invalid for solving
        DataIntegrityError: If internal solver setup fails
        SolutionExtractionError: If solution extraction fails

    Note:
        This function assumes the problem has been validated through
        validate_multi_phase_problem_structure() before calling.
    """
    # Log solver start (DEBUG - developer cares about internal operations)
    logger.debug("Starting multi-phase Radau collocation solver")

    # Initialize optimization problem - single unified NLP for all phases
    opti: ca.Opti = ca.Opti()
    phase_count = problem.get_phase_count()

    # Log problem structure (DEBUG)
    logger.debug(
        "Multi-phase problem structure: %d phases, %d global parameters, %d inter-phase constraints",
        phase_count,
        len(problem.global_parameters),
        len(problem.inter_phase_constraints),
    )

    try:
        # Set up optimization variables - CGPOPS hierarchical structure
        logger.debug("Setting up multi-phase optimization variables")
        variables = setup_multi_phase_optimization_variables(opti, problem)

        # Validate variable structure before proceeding
        validate_multi_phase_variable_structure(variables, problem.phases)

        # Set up interval state variables for all phases
        logger.debug("Setting up interval state variables for all phases")
        setup_multi_phase_interval_state_variables(opti, variables)

        # Create metadata bundle for constraint tracking and solution extraction
        logger.debug("Creating multi-phase metadata bundle")
        metadata = create_multi_phase_metadata_bundle(variables, problem.phases)

        # Apply all multi-phase constraints - block-diagonal structure
        logger.debug("Applying all multi-phase constraints")
        apply_all_multi_phase_constraints(opti, variables, problem.phases, problem, metadata)

        # Set up multi-phase objective function
        logger.debug("Setting up multi-phase objective function")
        _setup_multi_phase_objective(opti, variables, problem, metadata)

        # Apply initial guess if provided
        logger.debug("Applying multi-phase initial guess")
        _apply_multi_phase_initial_guess(opti, variables, problem)

        # Configure solver options
        logger.debug("Configuring multi-phase NLP solver")
        _configure_multi_phase_solver(opti, variables, metadata, problem)

    except Exception as e:
        logger.error("Failed to set up multi-phase optimization problem: %s", str(e))
        if isinstance(e, (ConfigurationError, DataIntegrityError)):
            raise
        raise DataIntegrityError(
            f"Failed to set up multi-phase optimization problem: {e}",
            "TrajectoLab multi-phase problem construction error",
        ) from e

    # Execute solve
    logger.debug("Executing multi-phase NLP solve")
    solution_obj = _execute_multi_phase_solve(opti, problem, variables, metadata)

    return solution_obj


def _setup_multi_phase_objective(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Set up multi-phase objective function.

    Implements the global objective function from CGPOPS Equation (17):
    J = φ(E^(1), ..., E^(P), s)

    Where E^(p) are phase endpoint vectors and s are global static parameters.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problem: Multi-phase problem protocol
        metadata: Multi-phase metadata bundle

    Raises:
        ConfigurationError: If objective setup fails
    """
    logger.info("Setting up multi-phase objective function")

    try:
        # Get global objective function
        objective_function = problem.get_global_objective_function()

        # Construct phase endpoint vectors E^(1), ..., E^(P)
        phase_endpoint_vectors = []
        for phase_vars in variables.phase_variables:
            if not phase_vars.state_at_mesh_nodes:
                raise ConfigurationError(
                    f"Phase {phase_vars.phase_index} missing state variables for objective evaluation",
                    "Multi-phase objective setup error",
                )

            # Create endpoint vector components (will be used by objective function)
            initial_state = phase_vars.state_at_mesh_nodes[0]  # Y_1^(p)
            final_state = phase_vars.state_at_mesh_nodes[-1]  # Y_{N^(p)+1}^(p)
            initial_time = phase_vars.initial_time  # t_0^(p)
            terminal_time = phase_vars.terminal_time  # t_f^(p)
            integrals = phase_vars.integral_variables  # Q^(p)

            endpoint_data = {
                "initial_state": initial_state,
                "initial_time": initial_time,
                "final_state": final_state,
                "terminal_time": terminal_time,
                "integrals": integrals,
                "phase_index": phase_vars.phase_index,
            }
            phase_endpoint_vectors.append(endpoint_data)

        # Get global parameters
        global_params = problem.global_parameters

        # Evaluate global objective function
        objective_value = objective_function(phase_endpoint_vectors, global_params)

        if objective_value is None:
            raise ConfigurationError(
                "Multi-phase objective function returned None",
                "Multi-phase objective evaluation error",
            )

        # Set objective in optimization problem
        opti.minimize(objective_value)

        # Store objective expression in metadata
        metadata.global_objective_expression = objective_value

        logger.debug("Multi-phase objective function set up successfully")

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Multi-phase objective setup failed: {e}", "TrajectoLab multi-phase objective error"
        ) from e


def _apply_multi_phase_initial_guess(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
) -> None:
    """
    Apply initial guess to multi-phase optimization variables.

    Applies initial guesses from individual phases plus any global parameter
    initial values to the unified multi-phase optimization problem.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problem: Multi-phase problem protocol

    Note:
        This function handles the case where phases may have different
        initial guess availability - some phases may have guesses while others don't.
    """
    logger.debug("Applying initial guess to multi-phase variables")

    try:
        # Apply initial guess for each phase
        phases_with_guess = 0

        for phase_idx, (phase_vars, phase_problem) in enumerate(
            zip(variables.phase_variables, problem.phases, strict=False)
        ):
            if phase_problem.initial_guess is not None:
                logger.debug("Applying initial guess for phase %d", phase_idx)

                # Use existing single-phase initial guess application
                from .initial_guess_solver import apply_initial_guess

                # Convert phase variables to single-phase format for compatibility
                single_phase_vars = phase_vars.to_single_phase_reference()

                # Apply initial guess for this phase
                apply_initial_guess(
                    opti=opti,
                    variables=single_phase_vars,
                    problem=phase_problem,
                    num_mesh_intervals=phase_vars.num_mesh_intervals,
                )

                phases_with_guess += 1
            else:
                logger.debug("Phase %d has no initial guess", phase_idx)

        # Apply global parameter initial values if provided
        if variables.global_parameters is not None and problem.global_parameters:
            logger.debug("Applying global parameter initial values")

            # Set global parameter values as initial guess
            param_values = []
            for param_name in variables.global_parameter_names:
                if param_name in problem.global_parameters:
                    param_values.append(problem.global_parameters[param_name])
                else:
                    param_values.append(0.0)  # Default initial value

            if param_values:
                param_vector = ca.DM(param_values)
                opti.set_initial(variables.global_parameters, param_vector)
                logger.debug("Set initial values for %d global parameters", len(param_values))

        logger.debug(
            "Initial guess application completed: %d/%d phases with guess",
            phases_with_guess,
            variables.phase_count,
        )

    except Exception as e:
        logger.warning("Initial guess application failed (continuing without guess): %s", str(e))
        # Don't raise exception - solver can proceed without initial guess


def _configure_multi_phase_solver(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
    problem: MultiPhaseProblemProtocol,
) -> None:
    """
    Configure solver options and store references for solution extraction.

    Sets up IPOPT solver with appropriate options for multi-phase problems
    and stores all necessary references for solution extraction.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle
        problem: Multi-phase problem protocol

    Raises:
        ConfigurationError: If solver configuration fails
    """
    logger.debug("Configuring multi-phase solver")

    try:
        # Set solver options
        solver_options = getattr(problem, "solver_options", {}) or {}

        # Add default options suitable for multi-phase problems
        default_options = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            # Multi-phase problems may need higher iteration limits
            "ipopt.max_iter": solver_options.get("ipopt.max_iter", 5000),
        }

        # Merge user options with defaults (user options take precedence)
        final_options = {**default_options, **solver_options}

        # Configure solver
        opti.solver("ipopt", final_options)

        logger.debug("Configured IPOPT solver with options: %s", final_options)

        # Store references for solution extraction
        _store_solution_extraction_references(opti, variables, metadata, problem)

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Multi-phase solver configuration failed: {e}",
            "TrajectoLab multi-phase solver setup error",
        ) from e


def _store_solution_extraction_references(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
    problem: MultiPhaseProblemProtocol,
) -> None:
    """
    Store references needed for multi-phase solution extraction.

    Stores all variable references, metadata, and problem information
    needed to extract the complete multi-phase solution after solving.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle
        problem: Multi-phase problem protocol
    """
    logger.debug("Storing solution extraction references")

    # Store multi-phase variable references
    opti.multi_phase_variables = variables
    opti.multi_phase_metadata = metadata
    opti.multi_phase_problem = problem

    # Store individual phase references for compatibility
    opti.phase_initial_time_variables = [pv.initial_time for pv in variables.phase_variables]
    opti.phase_terminal_time_variables = [pv.terminal_time for pv in variables.phase_variables]
    opti.phase_state_matrices = [pv.state_matrices for pv in variables.phase_variables]
    opti.phase_control_variables = [pv.control_variables for pv in variables.phase_variables]
    opti.phase_integral_variables = [pv.integral_variables for pv in variables.phase_variables]

    # Store global references
    opti.global_parameters_variable = variables.global_parameters
    opti.global_parameter_names = variables.global_parameter_names.copy()

    # Store metadata for solution analysis
    opti.phase_metadata = [pm.to_single_phase_metadata() for pm in metadata.phase_metadata]
    opti.inter_phase_constraint_count = metadata.inter_phase_constraint_count
    opti.total_nlp_variables = metadata.total_nlp_variables
    opti.total_nlp_constraints = metadata.total_nlp_constraints

    # Store objective expression
    opti.multi_phase_objective_expression = metadata.global_objective_expression

    logger.debug(
        "Stored solution extraction references: %d variables, %d constraints",
        metadata.total_nlp_variables,
        metadata.total_nlp_constraints,
    )


def _execute_multi_phase_solve(
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
) -> MultiPhaseOptimalControlSolution:
    """
    Execute the multi-phase solve and handle results.

    Attempts to solve the unified multi-phase NLP and creates appropriate
    solution objects for both successful and failed solves.

    Args:
        opti: CasADi optimization object
        problem: Multi-phase problem protocol
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle

    Returns:
        MultiPhaseOptimalControlSolution with results

    Raises:
        SolutionExtractionError: If solution extraction fails critically
    """
    logger.debug("Executing multi-phase NLP solve")

    # Log solve statistics
    logger.info(
        "Solving multi-phase NLP: %d phases, %d variables, %d constraints",
        variables.phase_count,
        metadata.total_nlp_variables,
        metadata.total_nlp_constraints,
    )

    try:
        # Attempt solve
        solver_solution: ca.OptiSol = opti.solve()

        # Log successful solve (DEBUG - internal operation success)
        logger.debug("Multi-phase NLP solver completed successfully")

        # Extract solution
        try:
            solution_obj = _extract_multi_phase_solution(
                solver_solution, opti, problem, variables, metadata, success=True
            )
            logger.debug("Multi-phase solution extraction completed")

        except Exception as e:
            logger.error("Multi-phase solution extraction failed: %s", str(e))
            raise SolutionExtractionError(
                f"Failed to extract multi-phase solution: {e}",
                "TrajectoLab multi-phase solution processing error",
            ) from e

    except RuntimeError as e:
        # Log solver failure (WARNING - recoverable, solution object still created)
        logger.warning("Multi-phase NLP solver failed: %s", str(e))

        # Create failed solution object
        try:
            solution_obj = _extract_multi_phase_solution(
                None, opti, problem, variables, metadata, success=False
            )
            solution_obj.message = f"Multi-phase solver runtime error: {e}"

        except Exception as extract_error:
            logger.error(
                "Critical: Multi-phase solution extraction failed after solver failure: %s",
                str(extract_error),
            )
            raise SolutionExtractionError(
                f"Failed to extract multi-phase solution after solver failure: {extract_error}",
                "Critical multi-phase solution extraction error",
            ) from extract_error

        # Try to retrieve debug values if available
        try:
            if hasattr(opti, "debug") and opti.debug is not None:
                _extract_debug_values(opti, solution_obj)
                logger.debug("Retrieved debug values from failed multi-phase solve")
        except Exception:
            logger.debug("Could not extract debug values from failed multi-phase solve")

    # Log final result
    if solution_obj.success:
        logger.info(
            "Multi-phase solve completed successfully: objective=%.6e, %d phases",
            solution_obj.objective or 0.0,
            solution_obj.phase_count,
        )
    else:
        logger.warning("Multi-phase solve failed: %s", solution_obj.message)

    return solution_obj


def _extract_multi_phase_solution(
    solver_solution: ca.OptiSol | None,
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
    success: bool,
) -> MultiPhaseOptimalControlSolution:
    """
    Extract multi-phase solution from solver results.

    Creates a comprehensive MultiPhaseOptimalControlSolution containing
    results from all phases plus global multi-phase information.

    Args:
        solver_solution: CasADi solver solution (None if solve failed)
        opti: CasADi optimization object
        problem: Multi-phase problem protocol
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle
        success: Whether the solve was successful

    Returns:
        MultiPhaseOptimalControlSolution with extracted results

    Note:
        This is a placeholder implementation. Full solution extraction
        will be implemented in Phase 4A: Multi-Phase Solution Extraction.
    """
    logger.debug("Extracting multi-phase solution")

    # Create solution object
    solution = MultiPhaseOptimalControlSolution()
    solution.success = success
    solution.phase_count = variables.phase_count

    if success and solver_solution is not None:
        try:
            # Extract basic solution information
            solution.objective = (
                float(solver_solution.value(metadata.global_objective_expression))
                if metadata.global_objective_expression
                else None
            )

            # Extract global parameters
            if variables.global_parameters is not None:
                global_param_values = solver_solution.value(variables.global_parameters)
                solution.global_parameters = {
                    name: float(global_param_values[i])
                    for i, name in enumerate(variables.global_parameter_names)
                }
            else:
                solution.global_parameters = dict(problem.global_parameters)

            # Placeholder for phase-specific solution extraction
            # This will be fully implemented in Phase 4A
            solution.phase_solutions = []
            solution.phase_endpoints = []

            # Basic phase solution placeholders
            for phase_idx, phase_vars in enumerate(variables.phase_variables):
                # Create placeholder single-phase solution
                from ..tl_types import OptimalControlSolution

                phase_solution = OptimalControlSolution()
                phase_solution.success = True
                phase_solution.message = "Phase solution extracted from multi-phase solve"

                # Extract basic timing information
                try:
                    phase_solution.initial_time_variable = float(
                        solver_solution.value(phase_vars.initial_time)
                    )
                    phase_solution.terminal_time_variable = float(
                        solver_solution.value(phase_vars.terminal_time)
                    )
                except Exception:
                    logger.debug("Could not extract timing for phase %d", phase_idx)

                solution.phase_solutions.append(phase_solution)

                # Create placeholder endpoint vector
                from ..tl_types import PhaseEndpointVector

                endpoint = PhaseEndpointVector(phase_index=phase_idx)
                solution.phase_endpoints.append(endpoint)

            solution.message = f"Multi-phase solve successful: {variables.phase_count} phases"

        except Exception as e:
            logger.warning("Partial solution extraction failed: %s", str(e))
            solution.success = False
            solution.message = f"Solution extraction error: {e}"

    else:
        solution.success = False
        if not success:
            solution.message = solution.message or "Multi-phase solver failed"

    # Store solver metadata
    solution.raw_solution = solver_solution
    solution.opti_object = opti
    solution.phase_count = variables.phase_count

    logger.debug("Multi-phase solution extraction completed: success=%s", solution.success)
    return solution


def _extract_debug_values(opti: ca.Opti, solution: MultiPhaseOptimalControlSolution) -> None:
    """
    Extract debug values from failed solve for analysis.

    Attempts to extract whatever values are available from a failed solve
    to aid in debugging and analysis.

    Args:
        opti: CasADi optimization object with debug information
        solution: Solution object to populate with debug values
    """
    try:
        if not hasattr(opti, "multi_phase_variables"):
            return

        variables = opti.multi_phase_variables

        # Try to extract phase timing information
        for phase_idx, phase_vars in enumerate(variables.phase_variables):
            try:
                if phase_idx < len(solution.phase_solutions):
                    phase_solution = solution.phase_solutions[phase_idx]
                    phase_solution.initial_time_variable = float(
                        opti.debug.value(phase_vars.initial_time)
                    )
                    phase_solution.terminal_time_variable = float(
                        opti.debug.value(phase_vars.terminal_time)
                    )
            except Exception:
                continue  # Skip if debug extraction fails for this phase

        # Try to extract global parameters
        if variables.global_parameters is not None:
            try:
                global_param_values = opti.debug.value(variables.global_parameters)
                solution.global_parameters = {
                    name: float(global_param_values[i])
                    for i, name in enumerate(variables.global_parameter_names)
                }
            except Exception:
                solution.global_parameters = {}

    except Exception:
        # If debug extraction fails completely, continue silently
        pass
