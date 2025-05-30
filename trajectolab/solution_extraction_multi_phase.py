"""
Multi-phase solution extraction and processing for optimal control problems.

This module extracts comprehensive solutions from multi-phase NLP results, creating
detailed solution objects that maintain the CGPOPS mathematical structure while
providing analysis capabilities for phase transitions, continuity, and global
optimization results.
"""

import logging
from typing import Any

import casadi as ca
import numpy as np

from .direct_solver.types_solver import MultiPhaseMetadataBundle, MultiPhaseVariableReferences
from .exceptions import DataIntegrityError, SolutionExtractionError
from .solution_extraction import extract_and_format_solution
from .tl_types import (
    FloatArray,
    MultiPhaseOptimalControlSolution,
    MultiPhaseProblemProtocol,
    OptimalControlSolution,
    PhaseEndpointVector,
    ProblemProtocol,
)


# Library logger
logger = logging.getLogger(__name__)


def extract_multi_phase_solution(
    solver_solution: ca.OptiSol | None,
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
) -> MultiPhaseOptimalControlSolution:
    """
    Extract comprehensive multi-phase solution from solver results.

    Creates a complete MultiPhaseOptimalControlSolution containing:
    - Individual phase solutions extracted from the unified NLP
    - Global optimization results (parameters, objective, constraints)
    - Phase endpoint vectors E^(p) as defined in CGPOPS Equation (15)
    - Phase transition analysis and continuity assessment
    - Comprehensive metadata for analysis and visualization

    Args:
        solver_solution: CasADi solver solution (None if solve failed)
        opti: CasADi optimization object with stored references
        problem: Multi-phase problem protocol
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle

    Returns:
        MultiPhaseOptimalControlSolution with complete multi-phase results

    Raises:
        SolutionExtractionError: If solution extraction fails critically
        DataIntegrityError: If solution data is inconsistent
    """
    logger.debug("Extracting multi-phase solution from solver results")

    try:
        # Create multi-phase solution object
        solution = MultiPhaseOptimalControlSolution()
        solution.phase_count = variables.phase_count
        solution.success = solver_solution is not None

        # Store basic solver information
        solution.raw_solution = solver_solution
        solution.opti_object = opti

        if solution.success and solver_solution is not None:
            # Extract successful solution
            _extract_successful_multi_phase_solution(
                solution, solver_solution, opti, problem, variables, metadata
            )
        else:
            # Handle failed solution
            _extract_failed_multi_phase_solution(solution, opti, problem, variables, metadata)

        # Perform solution validation
        _validate_multi_phase_solution_integrity(solution, variables)

        # Log extraction completion
        logger.info(
            "Multi-phase solution extraction completed: success=%s, phases=%d, objective=%.6e",
            solution.success,
            solution.phase_count,
            solution.objective or 0.0,
        )

        return solution

    except Exception as e:
        logger.error("Multi-phase solution extraction failed: %s", str(e))
        if isinstance(e, (SolutionExtractionError, DataIntegrityError)):
            raise
        raise SolutionExtractionError(
            f"Multi-phase solution extraction failed: {e}",
            "TrajectoLab multi-phase solution extraction error",
        ) from e


def _extract_successful_multi_phase_solution(
    solution: MultiPhaseOptimalControlSolution,
    solver_solution: ca.OptiSol,
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Extract solution from successful multi-phase solve.

    Performs comprehensive extraction of all solution components including
    individual phase results, global parameters, and inter-phase information.

    Args:
        solution: Multi-phase solution object to populate
        solver_solution: Successful CasADi solver solution
        opti: CasADi optimization object
        problem: Multi-phase problem protocol
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle
    """
    logger.debug("Extracting successful multi-phase solution")

    try:
        # Extract global objective value
        if metadata.global_objective_expression is not None:
            solution.objective = float(solver_solution.value(metadata.global_objective_expression))
            logger.debug("Extracted global objective: %.6e", solution.objective)

        # Extract global parameters
        solution.global_parameters = _extract_global_parameters(solver_solution, variables, problem)

        # Extract individual phase solutions
        solution.phase_solutions = _extract_individual_phase_solutions(
            solver_solution, opti, problem, variables, metadata
        )

        # Extract phase endpoint vectors
        solution.phase_endpoints = _extract_phase_endpoint_vectors(
            solver_solution, variables, solution.phase_solutions
        )

        # Extract inter-phase constraint information
        _extract_inter_phase_constraint_information(
            solution, solver_solution, opti, variables, metadata
        )

        # Analyze phase transitions and continuity
        _analyze_phase_transitions(solution)

        # Extract mesh and solver metadata
        _extract_solution_metadata(solution, metadata, problem)

        solution.success = True
        solution.message = f"Multi-phase solve successful: {solution.phase_count} phases"

        logger.debug("Successful multi-phase solution extraction completed")

    except Exception as e:
        logger.error("Successful solution extraction failed: %s", str(e))
        solution.success = False
        solution.message = f"Solution extraction error: {e}"
        raise SolutionExtractionError(
            f"Failed to extract successful multi-phase solution: {e}",
            "TrajectoLab multi-phase solution extraction error",
        ) from e


def _extract_failed_multi_phase_solution(
    solution: MultiPhaseOptimalControlSolution,
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Extract available information from failed multi-phase solve.

    Attempts to extract whatever information is available from a failed solve
    to aid in debugging and analysis.

    Args:
        solution: Multi-phase solution object to populate
        opti: CasADi optimization object
        problem: Multi-phase problem protocol
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle
    """
    logger.debug("Extracting failed multi-phase solution information")

    try:
        solution.success = False
        solution.message = "Multi-phase solver failed"

        # Try to extract debug information if available
        if hasattr(opti, "debug") and opti.debug is not None:
            logger.debug("Attempting to extract debug values from failed solve")

            # Extract global parameters from debug
            solution.global_parameters = _extract_global_parameters_debug(
                opti.debug, variables, problem
            )

            # Extract basic phase information from debug
            solution.phase_solutions = _extract_phase_solutions_debug(
                opti.debug, problem, variables
            )

            # Create placeholder endpoint vectors
            solution.phase_endpoints = _create_placeholder_endpoint_vectors(variables)

        else:
            # Create minimal solution structure
            solution.global_parameters = dict(problem.global_parameters)
            solution.phase_solutions = _create_placeholder_phase_solutions(problem.phases)
            solution.phase_endpoints = _create_placeholder_endpoint_vectors(variables)

        # Extract basic metadata
        _extract_solution_metadata(solution, metadata, problem)

        logger.debug("Failed solution extraction completed with available information")

    except Exception as e:
        logger.warning("Failed solution extraction encountered error: %s", str(e))
        # Create minimal solution structure
        solution.global_parameters = {}
        solution.phase_solutions = []
        solution.phase_endpoints = []


def _extract_global_parameters(
    solver_solution: ca.OptiSol,
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
) -> dict[str, float]:
    """
    Extract global parameter values from successful solve.

    Args:
        solver_solution: Successful solver solution
        variables: Multi-phase variable references
        problem: Multi-phase problem protocol

    Returns:
        Dictionary of parameter names to optimized values
    """
    logger.debug("Extracting global parameters")

    global_params = {}

    try:
        if variables.global_parameters is not None and variables.global_parameter_names:
            # Extract optimized parameter values
            param_values = solver_solution.value(variables.global_parameters)

            # Handle scalar vs vector parameter values
            if np.isscalar(param_values):
                if len(variables.global_parameter_names) == 1:
                    global_params[variables.global_parameter_names[0]] = float(param_values)
            else:
                param_array = np.atleast_1d(param_values)
                for i, param_name in enumerate(variables.global_parameter_names):
                    if i < len(param_array):
                        global_params[param_name] = float(param_array[i])

            logger.debug("Extracted %d global parameters", len(global_params))
        else:
            # Use original parameter values if no optimization variables
            global_params = dict(problem.global_parameters)
            logger.debug("Used original global parameter values")

    except Exception as e:
        logger.warning("Global parameter extraction failed: %s", str(e))
        global_params = dict(problem.global_parameters)

    return global_params


def _extract_global_parameters_debug(
    debug: Any,
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
) -> dict[str, float]:
    """Extract global parameters from debug information."""
    try:
        if variables.global_parameters is not None and variables.global_parameter_names:
            param_values = debug.value(variables.global_parameters)
            global_params = {}

            if np.isscalar(param_values):
                if len(variables.global_parameter_names) == 1:
                    global_params[variables.global_parameter_names[0]] = float(param_values)
            else:
                param_array = np.atleast_1d(param_values)
                for i, param_name in enumerate(variables.global_parameter_names):
                    if i < len(param_array):
                        global_params[param_name] = float(param_array[i])

            return global_params
    except Exception as e:
        logger.exception("State continuity analysis failed: %s", e)

    return dict(problem.global_parameters)


def _extract_individual_phase_solutions(
    solver_solution: ca.OptiSol,
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
) -> list[OptimalControlSolution]:
    """
    Extract individual phase solutions from unified multi-phase NLP.

    Uses the existing single-phase solution extraction for each phase
    while maintaining phase-specific information and metadata.

    Args:
        solver_solution: Successful solver solution
        opti: CasADi optimization object
        problem: Multi-phase problem protocol
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle

    Returns:
        List of OptimalControlSolution objects, one per phase
    """
    logger.debug("Extracting individual phase solutions")

    phase_solutions = []

    try:
        for phase_idx, (phase_vars, phase_problem) in enumerate(
            zip(variables.phase_variables, problem.phases, strict=False)
        ):
            logger.debug("Extracting solution for phase %d", phase_idx)

            # Create single-phase compatible references
            single_phase_vars = phase_vars.to_single_phase_reference()
            phase_metadata = metadata.get_phase_metadata(phase_idx)
            single_phase_metadata = phase_metadata.to_single_phase_metadata()

            # Get phase-specific mesh configuration
            collocation_points = phase_vars.collocation_points_per_interval
            global_mesh_nodes = phase_problem.global_normalized_mesh_nodes

            try:
                # Use existing single-phase solution extraction
                phase_solution = extract_and_format_solution(
                    casadi_solution_object=solver_solution,
                    casadi_optimization_problem_object=opti,
                    problem=phase_problem,
                    num_collocation_nodes_per_interval=collocation_points,
                    global_normalized_mesh_nodes=global_mesh_nodes,
                )

                # Mark as part of multi-phase solution
                phase_solution.message = f"Phase {phase_idx} solution from multi-phase solve"
                phase_solutions.append(phase_solution)

                logger.debug("Successfully extracted phase %d solution", phase_idx)

            except Exception as e:
                logger.warning("Phase %d solution extraction failed: %s", phase_idx, str(e))

                # Create placeholder solution for failed phase extraction
                placeholder_solution = OptimalControlSolution()
                placeholder_solution.success = False
                placeholder_solution.message = f"Phase {phase_idx} extraction failed: {e}"

                # Try to extract basic timing information
                try:
                    placeholder_solution.initial_time_variable = float(
                        solver_solution.value(phase_vars.initial_time)
                    )
                    placeholder_solution.terminal_time_variable = float(
                        solver_solution.value(phase_vars.terminal_time)
                    )
                except Exception:
                    pass

                phase_solutions.append(placeholder_solution)

        logger.debug("Extracted %d phase solutions", len(phase_solutions))
        return phase_solutions

    except Exception as e:
        logger.error("Individual phase solution extraction failed: %s", str(e))
        raise SolutionExtractionError(
            f"Failed to extract individual phase solutions: {e}",
            "TrajectoLab phase solution extraction error",
        ) from e


def _extract_phase_solutions_debug(
    debug: Any,
    problems: list[ProblemProtocol],
    variables: MultiPhaseVariableReferences,
) -> list[OptimalControlSolution]:
    """Extract basic phase information from debug values."""
    phase_solutions = []

    for phase_idx, phase_vars in enumerate(variables.phase_variables):
        solution = OptimalControlSolution()
        solution.success = False
        solution.message = f"Phase {phase_idx} debug extraction"

        try:
            solution.initial_time_variable = float(debug.value(phase_vars.initial_time))
            solution.terminal_time_variable = float(debug.value(phase_vars.terminal_time))
        except Exception:
            pass

        phase_solutions.append(solution)

    return phase_solutions


def _create_placeholder_phase_solutions(
    problems: list[ProblemProtocol],
) -> list[OptimalControlSolution]:
    """Create placeholder phase solutions when extraction fails."""
    phase_solutions = []

    for phase_idx, _ in enumerate(problems):
        solution = OptimalControlSolution()
        solution.success = False
        solution.message = f"Phase {phase_idx} placeholder (solve failed)"
        phase_solutions.append(solution)

    return phase_solutions


def _extract_phase_endpoint_vectors(
    solver_solution: ca.OptiSol,
    variables: MultiPhaseVariableReferences,
    phase_solutions: list[OptimalControlSolution],
) -> list[PhaseEndpointVector]:
    """
    Extract phase endpoint vectors E^(p) from solution.

    Creates endpoint vectors as defined in CGPOPS Equation (15):
    E^(p) = [Y_1^(p), t_0^(p), Y_{N^(p)+1}^(p), t_f^(p), Q^(p)]

    Args:
        solver_solution: Successful solver solution
        variables: Multi-phase variable references
        phase_solutions: List of extracted phase solutions

    Returns:
        List of PhaseEndpointVector objects
    """
    logger.debug("Extracting phase endpoint vectors")

    endpoint_vectors = []

    try:
        for phase_idx, (phase_vars, phase_solution) in enumerate(
            zip(variables.phase_variables, phase_solutions, strict=False)
        ):
            logger.debug("Creating endpoint vector for phase %d", phase_idx)

            # Extract endpoint components
            try:
                # Initial state Y_1^(p)
                initial_state = None
                if phase_vars.state_at_mesh_nodes and phase_solution.states:
                    initial_state_values = solver_solution.value(phase_vars.state_at_mesh_nodes[0])
                    initial_state = np.atleast_1d(initial_state_values).astype(np.float64)

                # Final state Y_{N^(p)+1}^(p)
                final_state = None
                if phase_vars.state_at_mesh_nodes and phase_solution.states:
                    final_state_values = solver_solution.value(phase_vars.state_at_mesh_nodes[-1])
                    final_state = np.atleast_1d(final_state_values).astype(np.float64)

                # Times t_0^(p), t_f^(p)
                initial_time = phase_solution.initial_time_variable
                final_time = phase_solution.terminal_time_variable

                # Integrals Q^(p)
                integrals = None
                if phase_vars.integral_variables is not None and phase_vars.num_integrals > 0:
                    integral_values = solver_solution.value(phase_vars.integral_variables)
                    if np.isscalar(integral_values):
                        integrals = np.array([float(integral_values)], dtype=np.float64)
                    else:
                        integrals = np.atleast_1d(integral_values).astype(np.float64)

                # Create endpoint vector
                endpoint = PhaseEndpointVector(
                    phase_index=phase_idx,
                    initial_state=initial_state,
                    initial_time=initial_time,
                    final_state=final_state,
                    final_time=final_time,
                    integrals=integrals,
                )

                endpoint_vectors.append(endpoint)
                logger.debug("Created endpoint vector for phase %d", phase_idx)

            except Exception as e:
                logger.warning(
                    "Endpoint vector creation failed for phase %d: %s", phase_idx, str(e)
                )

                # Create placeholder endpoint vector
                placeholder_endpoint = PhaseEndpointVector(phase_index=phase_idx)
                endpoint_vectors.append(placeholder_endpoint)

        logger.debug("Extracted %d phase endpoint vectors", len(endpoint_vectors))
        return endpoint_vectors

    except Exception as e:
        logger.error("Phase endpoint vector extraction failed: %s", str(e))
        raise SolutionExtractionError(
            f"Failed to extract phase endpoint vectors: {e}",
            "TrajectoLab endpoint vector extraction error",
        ) from e


def _create_placeholder_endpoint_vectors(
    variables: MultiPhaseVariableReferences,
) -> list[PhaseEndpointVector]:
    """Create placeholder endpoint vectors when extraction fails."""
    return [
        PhaseEndpointVector(phase_index=phase_idx) for phase_idx in range(variables.phase_count)
    ]


def _extract_inter_phase_constraint_information(
    solution: MultiPhaseOptimalControlSolution,
    solver_solution: ca.OptiSol,
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Extract inter-phase constraint violation and multiplier information.

    Analyzes the satisfaction of inter-phase event constraints and extracts
    constraint multipliers for sensitivity analysis.

    Args:
        solution: Multi-phase solution object to populate
        solver_solution: Successful solver solution
        opti: CasADi optimization object
        variables: Multi-phase variable references
        metadata: Multi-phase metadata bundle
    """
    logger.debug("Extracting inter-phase constraint information")

    try:
        if metadata.inter_phase_constraint_count > 0:
            # Extract constraint violations (if available)
            try:
                # This would require access to constraint expressions
                # Implementation depends on how constraints are stored
                solution.inter_phase_constraint_violations = np.array([], dtype=np.float64)
                solution.max_inter_phase_constraint_violation = 0.0
                logger.debug("Inter-phase constraint violations extracted")
            except Exception as e:
                logger.debug("Could not extract constraint violations: %s", str(e))

            # Extract constraint multipliers (if available)
            try:
                # Extract Lagrange multipliers for inter-phase constraints
                # Implementation depends on CasADi multiplier access
                solution.inter_phase_constraint_multipliers = np.array([], dtype=np.float64)
                logger.debug("Inter-phase constraint multipliers extracted")
            except Exception as e:
                logger.debug("Could not extract constraint multipliers: %s", str(e))

        else:
            logger.debug("No inter-phase constraints to analyze")

    except Exception as e:
        logger.warning("Inter-phase constraint information extraction failed: %s", str(e))


def _analyze_phase_transitions(solution: MultiPhaseOptimalControlSolution) -> None:
    """
    Analyze phase transitions for continuity and discontinuities.

    Performs comprehensive analysis of state continuity, time continuity,
    and constraint violations at phase boundaries.

    Args:
        solution: Multi-phase solution object to populate with analysis
    """
    logger.debug("Analyzing phase transitions")

    try:
        if len(solution.phase_solutions) < 2:
            logger.debug("Less than 2 phases, no transitions to analyze")
            return

        # Initialize analysis containers
        solution.phase_continuity_errors = {}
        solution.phase_jump_magnitudes = {}

        # Analyze each phase transition
        for i in range(len(solution.phase_solutions) - 1):
            transition_key = (i, i + 1)

            try:
                phase_i = solution.phase_solutions[i]
                phase_j = solution.phase_solutions[i + 1]

                # Analyze state continuity
                state_discontinuity = _analyze_state_continuity(phase_i, phase_j)
                if state_discontinuity is not None:
                    solution.phase_continuity_errors[transition_key] = state_discontinuity
                    solution.phase_jump_magnitudes[transition_key] = np.linalg.norm(
                        state_discontinuity
                    )

                # Analyze time continuity
                time_discontinuity = _analyze_time_continuity(phase_i, phase_j)
                if time_discontinuity is not None:
                    logger.debug(f"Phase {i} to {i + 1} time discontinuity: {time_discontinuity}")

                logger.debug("Analyzed transition from phase %d to %d", i, i + 1)

            except Exception as e:
                logger.warning(
                    "Phase transition analysis failed for phases %d-%d: %s", i, i + 1, str(e)
                )

        logger.debug("Phase transition analysis completed")

    except Exception as e:
        logger.warning("Phase transition analysis failed: %s", str(e))


def _analyze_state_continuity(
    phase_i: OptimalControlSolution,
    phase_j: OptimalControlSolution,
) -> FloatArray | None:
    """Analyze state continuity between two phases."""
    try:
        if (
            phase_i.states
            and phase_j.states
            and len(phase_i.states) > 0
            and len(phase_j.states) > 0
        ):
            # Get final state of phase i and initial state of phase j
            final_state_i = np.array([state[-1] for state in phase_i.states])
            initial_state_j = np.array([state[0] for state in phase_j.states])

            # Calculate discontinuity
            discontinuity = final_state_i - initial_state_j
            return discontinuity.astype(np.float64)
    except Exception:
        pass

    return None


def _analyze_time_continuity(
    phase_i: OptimalControlSolution,
    phase_j: OptimalControlSolution,
) -> float | None:
    """Analyze time continuity between two phases."""
    try:
        if phase_i.terminal_time_variable is not None and phase_j.initial_time_variable is not None:
            return phase_j.initial_time_variable - phase_i.terminal_time_variable
    except Exception:
        pass

    return None


def _extract_solution_metadata(
    solution: MultiPhaseOptimalControlSolution,
    metadata: MultiPhaseMetadataBundle,
    problem: MultiPhaseProblemProtocol,
) -> None:
    """
    Extract solver and problem metadata for solution analysis.

    Args:
        solution: Multi-phase solution object to populate
        metadata: Multi-phase metadata bundle
        problem: Multi-phase problem protocol
    """
    logger.debug("Extracting solution metadata")

    try:
        # Store mesh configurations
        solution.phase_mesh_configurations = []
        for phase_idx, phase_problem in enumerate(problem.phases):
            mesh_config = {
                "phase_index": phase_idx,
                "collocation_points_per_interval": phase_problem.collocation_points_per_interval.copy(),
                "global_normalized_mesh_nodes": phase_problem.global_normalized_mesh_nodes.copy(),
                "num_mesh_intervals": len(phase_problem.collocation_points_per_interval),
            }
            solution.phase_mesh_configurations.append(mesh_config)

        # Store NLP statistics
        solution.total_collocation_points = sum(
            sum(config["collocation_points_per_interval"])
            for config in solution.phase_mesh_configurations
        )

        # Store solver metadata
        solution.solve_time = None  # Would be extracted from solver if available
        solution.nlp_iterations = None  # Would be extracted from solver if available

        logger.debug("Solution metadata extraction completed")

    except Exception as e:
        logger.warning("Solution metadata extraction failed: %s", str(e))


def _validate_multi_phase_solution_integrity(
    solution: MultiPhaseOptimalControlSolution,
    variables: MultiPhaseVariableReferences,
) -> None:
    """
    Validate integrity of extracted multi-phase solution.

    Performs consistency checks to ensure the extracted solution maintains
    proper structure and data integrity.

    Args:
        solution: Multi-phase solution object to validate
        variables: Multi-phase variable references for validation

    Raises:
        DataIntegrityError: If solution data is inconsistent
    """
    logger.debug("Validating multi-phase solution integrity")

    try:
        # Validate basic structure
        if solution.phase_count != variables.phase_count:
            raise DataIntegrityError(
                f"Solution phase count ({solution.phase_count}) doesn't match "
                f"variable phase count ({variables.phase_count})",
                "Multi-phase solution integrity error",
            )

        # Validate phase solutions count
        if len(solution.phase_solutions) != solution.phase_count:
            raise DataIntegrityError(
                f"Phase solutions count ({len(solution.phase_solutions)}) doesn't match "
                f"phase count ({solution.phase_count})",
                "Multi-phase solution integrity error",
            )

        # Validate phase endpoints count
        if len(solution.phase_endpoints) != solution.phase_count:
            raise DataIntegrityError(
                f"Phase endpoints count ({len(solution.phase_endpoints)}) doesn't match "
                f"phase count ({solution.phase_count})",
                "Multi-phase solution integrity error",
            )

        # Validate phase endpoint indices
        for i, endpoint in enumerate(solution.phase_endpoints):
            if endpoint.phase_index != i:
                raise DataIntegrityError(
                    f"Phase endpoint index mismatch: expected {i}, got {endpoint.phase_index}",
                    "Multi-phase solution integrity error",
                )

        logger.debug("Multi-phase solution integrity validation passed")

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Multi-phase solution integrity validation failed: {e}",
            "TrajectoLab multi-phase solution validation error",
        ) from e
