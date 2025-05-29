"""
Multi-phase optimization variable setup and configuration for the direct solver.

This module implements the faithful CGPOPS multi-phase variable structure from
Equation (31), creating the hierarchical NLP decision vector:
z = [z^(1), ..., z^(P), s₁, ..., sₙₛ]ᵀ

Each phase contributes variables z^(p) = [Y_cols^(p), U_cols^(p), Q^(p), t₀^(p), t_f^(p)]
while global static parameters s₁, ..., sₙₛ are shared across all phases.
"""

import logging
from typing import Any

import casadi as ca

from ..exceptions import ConfigurationError, DataIntegrityError
from ..input_validation import (
    validate_casadi_optimization_object,
    validate_mesh_interval_count,
    validate_polynomial_degree,
)
from ..tl_types import MultiPhaseProblemProtocol, ProblemProtocol
from .types_solver import (
    MultiPhaseMetadataBundle,
    MultiPhaseVariableReferences,
    PhaseMetadataBundle,
    PhaseVariableReferences,
)
from .variables_solver import (
    _create_control_variables,
    _create_global_state_variables,
    _create_integral_variables,
    _create_time_variables,
    setup_interval_state_variables,
)


# Library logger
logger = logging.getLogger(__name__)


def setup_multi_phase_optimization_variables(
    opti: ca.Opti,
    problem: MultiPhaseProblemProtocol,
) -> MultiPhaseVariableReferences:
    """
    Set up multi-phase optimization variables matching CGPOPS structure exactly.

    Creates the hierarchical NLP decision vector from CGPOPS Equation (31):
    z = [z^(1), ..., z^(P), s₁, ..., sₙₛ]ᵀ

    Where each phase contributes:
    z^(p) = [Y_{(:,1)}^(p), ..., Y_{(:,n_y^(p))}^(p), U_{(:,1)}^(p), ..., U_{(:,n_u^(p))}^(p), (Q^(p))ᵀ, t₀^(p), t_f^(p)]

    And global static parameters s₁, ..., sₙₛ are shared across all phases.

    Args:
        opti: CasADi optimization object for variable creation
        problem: Multi-phase problem protocol with phase and parameter information

    Returns:
        MultiPhaseVariableReferences containing the complete hierarchical variable structure

    Raises:
        ConfigurationError: If problem structure is invalid for variable setup
        DataIntegrityError: If internal variable creation fails

    Note:
        This function assumes comprehensive validation has been done by the calling
        solver through validate_multi_phase_problem_structure().
    """
    # Validate inputs
    validate_casadi_optimization_object(opti, "multi-phase variable setup")

    phase_count = problem.get_phase_count()
    if phase_count < 2:
        raise ConfigurationError(
            f"Multi-phase problem must have at least 2 phases, got {phase_count}",
            "Multi-phase variable setup error",
        )

    # Log multi-phase variable setup start (DEBUG - developer info)
    logger.debug(
        "Setting up multi-phase variables: %d phases, %d global parameters",
        phase_count,
        len(problem.global_parameters),
    )

    try:
        # Set up global static parameters first - shared across all phases
        global_parameters = _setup_global_parameters(opti, problem.global_parameters)

        # Set up variables for each phase
        phase_variables = []
        for phase_idx in range(phase_count):
            phase_problem = problem.phases[phase_idx]
            phase_vars = _setup_phase_variables(opti, phase_problem, phase_idx)
            phase_variables.append(phase_vars)

        # Create multi-phase variable references structure
        multi_phase_vars = MultiPhaseVariableReferences(
            phase_variables=phase_variables,
            global_parameters=global_parameters,
            global_parameter_names=list(problem.global_parameters.keys()),
            global_parameter_values=dict(problem.global_parameters),
            phase_count=phase_count,
        )

        # Validate multi-phase variable structure
        multi_phase_vars.validate_phase_consistency()

        # Log successful variable setup (DEBUG)
        total_vars = multi_phase_vars.get_nlp_variable_count()
        logger.debug(
            "Multi-phase variables created successfully: %d total NLP variables", total_vars
        )

        return multi_phase_vars

    except Exception as e:
        logger.error("Multi-phase variable setup failed: %s", str(e))
        if isinstance(e, (ConfigurationError, DataIntegrityError)):
            raise
        raise DataIntegrityError(
            f"Multi-phase variable setup failed: {e}",
            "TrajectoLab multi-phase variable creation error",
        ) from e


def _setup_global_parameters(opti: ca.Opti, global_params: dict[str, float]) -> ca.MX | None:
    """
    Set up global static parameters s₁, ..., sₙₛ shared across all phases.

    Global parameters are optimization variables that appear in dynamics,
    constraints, and objective functions of multiple phases, implementing
    the shared parameter structure from CGPOPS Equations (6) and (31).

    Args:
        opti: CasADi optimization object
        global_params: Dictionary of parameter names to values

    Returns:
        CasADi MX variable containing all global parameters, or None if no parameters

    Raises:
        DataIntegrityError: If parameter creation fails
    """
    if not global_params:
        logger.debug("No global parameters to set up")
        return None

    num_params = len(global_params)
    logger.debug("Setting up %d global parameters: %s", num_params, list(global_params.keys()))

    try:
        # Create global parameter vector s = [s₁, ..., sₙₛ]ᵀ
        global_param_vector = opti.variable(num_params)

        if global_param_vector is None:
            raise DataIntegrityError(
                "Failed to create global parameter vector",
                "CasADi global parameter creation failure",
            )

        # Apply parameter bounds if needed (could be extended to support bounds)
        # For now, global parameters are typically fixed at their values
        # This could be extended to support parameter optimization

        logger.debug("Global parameter vector created: size=%d", num_params)
        return global_param_vector

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Global parameter setup failed: {e}", "TrajectoLab global parameter creation error"
        ) from e


def _setup_phase_variables(
    opti: ca.Opti,
    phase_problem: ProblemProtocol,
    phase_index: int,
) -> PhaseVariableReferences:
    """
    Set up optimization variables for a single phase within multi-phase context.

    Creates variables z^(p) for phase p as defined in CGPOPS Equation (31):
    z^(p) = [Y_{(:,1)}^(p), ..., Y_{(:,n_y^(p))}^(p), U_{(:,1)}^(p), ..., U_{(:,n_u^(p))}^(p), (Q^(p))ᵀ, t₀^(p), t_f^(p)]

    Args:
        opti: CasADi optimization object
        phase_problem: Single-phase problem protocol
        phase_index: Zero-based phase index

    Returns:
        PhaseVariableReferences containing all variables for this phase

    Raises:
        ConfigurationError: If phase structure is invalid
        DataIntegrityError: If variable creation fails
    """
    logger.debug("Setting up variables for phase %d", phase_index)

    # Validate phase structure
    if phase_index < 0:
        raise ConfigurationError(
            f"Phase index must be non-negative, got {phase_index}",
            "Multi-phase variable setup error",
        )

    if not phase_problem._mesh_configured:
        raise ConfigurationError(
            f"Phase {phase_index} mesh not configured. Set mesh before variable setup.",
            "Multi-phase variable setup error",
        )

    # Get phase information
    num_states, num_controls = phase_problem.get_variable_counts()
    num_integrals = phase_problem._num_integrals
    num_mesh_intervals = len(phase_problem.collocation_points_per_interval)
    collocation_points = phase_problem.collocation_points_per_interval.copy()

    # Validate phase mesh configuration
    validate_mesh_interval_count(num_mesh_intervals, f"phase {phase_index} mesh intervals")
    for k, num_colloc in enumerate(collocation_points):
        validate_polynomial_degree(
            num_colloc, f"phase {phase_index} interval {k} collocation points"
        )

    try:
        # Create time variables t₀^(p), t_f^(p)
        initial_time, terminal_time = _create_time_variables(opti, phase_problem)

        # Create global state variables at mesh nodes
        state_at_mesh_nodes = _create_global_state_variables(opti, num_states, num_mesh_intervals)

        # Create control variables for each interval
        control_variables = _create_control_variables(opti, phase_problem, num_mesh_intervals)

        # Create integral variables Q^(p)
        integral_variables = _create_integral_variables(opti, num_integrals)

        # Create phase variable references
        phase_vars = PhaseVariableReferences(
            phase_index=phase_index,
            initial_time=initial_time,
            terminal_time=terminal_time,
            state_at_mesh_nodes=state_at_mesh_nodes,
            control_variables=control_variables,
            integral_variables=integral_variables,
            num_states=num_states,
            num_controls=num_controls,
            num_integrals=num_integrals,
            num_mesh_intervals=num_mesh_intervals,
            collocation_points_per_interval=collocation_points,
        )

        logger.debug(
            "Phase %d variables created: %d states, %d controls, %d integrals, %d intervals",
            phase_index,
            num_states,
            num_controls,
            num_integrals,
            num_mesh_intervals,
        )

        return phase_vars

    except Exception as e:
        logger.error("Phase %d variable setup failed: %s", phase_index, str(e))
        if isinstance(e, (ConfigurationError, DataIntegrityError)):
            raise
        raise DataIntegrityError(
            f"Phase {phase_index} variable setup failed: {e}",
            "TrajectoLab phase variable creation error",
        ) from e


def setup_multi_phase_interval_state_variables(
    opti: ca.Opti,
    phase_variables: MultiPhaseVariableReferences,
) -> None:
    """
    Set up interval state variables for all phases in multi-phase problem.

    For each phase, sets up state matrices connecting mesh nodes through
    Lagrange polynomial interpolation, maintaining the block-diagonal
    structure essential for CGPOPS sparsity exploitation.

    Args:
        opti: CasADi optimization object
        phase_variables: Multi-phase variable references to populate

    Raises:
        DataIntegrityError: If interval variable setup fails
    """
    logger.debug("Setting up interval state variables for %d phases", phase_variables.phase_count)

    try:
        for phase_idx, phase_vars in enumerate(phase_variables.phase_variables):
            logger.debug("Setting up interval variables for phase %d", phase_idx)

            # Set up interval state variables for this phase
            for interval_idx in range(phase_vars.num_mesh_intervals):
                num_colloc_nodes = phase_vars.collocation_points_per_interval[interval_idx]

                # Use existing single-phase function for each interval
                state_matrix, interior_vars = setup_interval_state_variables(
                    opti=opti,
                    mesh_interval_index=interval_idx,
                    num_states=phase_vars.num_states,
                    num_colloc_nodes=num_colloc_nodes,
                    state_at_global_mesh_nodes=phase_vars.state_at_mesh_nodes,
                )

                # Store in phase variables
                phase_vars.state_matrices.append(state_matrix)
                phase_vars.interior_variables.append(interior_vars)

            # Validate interval setup for this phase
            if len(phase_vars.state_matrices) != phase_vars.num_mesh_intervals:
                raise DataIntegrityError(
                    f"Phase {phase_idx} state matrix count mismatch: "
                    f"expected {phase_vars.num_mesh_intervals}, got {len(phase_vars.state_matrices)}",
                    "Multi-phase interval variable setup error",
                )

            if len(phase_vars.interior_variables) != phase_vars.num_mesh_intervals:
                raise DataIntegrityError(
                    f"Phase {phase_idx} interior variables count mismatch: "
                    f"expected {phase_vars.num_mesh_intervals}, got {len(phase_vars.interior_variables)}",
                    "Multi-phase interval variable setup error",
                )

        logger.debug("All phase interval state variables set up successfully")

    except Exception as e:
        logger.error("Multi-phase interval variable setup failed: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Multi-phase interval variable setup failed: {e}",
            "TrajectoLab multi-phase interval setup error",
        ) from e


def create_multi_phase_metadata_bundle(
    phase_variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
) -> MultiPhaseMetadataBundle:
    """
    Create metadata bundle for multi-phase problem solution extraction.

    Aggregates metadata from all phases plus global multi-phase information,
    essential for proper solution extraction and analysis.

    Args:
        phase_variables: Multi-phase variable references
        problems: List of phase problems

    Returns:
        MultiPhaseMetadataBundle containing comprehensive metadata

    Raises:
        ConfigurationError: If metadata creation fails
    """
    if len(problems) != phase_variables.phase_count:
        raise ConfigurationError(
            f"Problem count ({len(problems)}) must match phase count ({phase_variables.phase_count})",
            "Multi-phase metadata creation error",
        )

    logger.debug("Creating multi-phase metadata bundle for %d phases", phase_variables.phase_count)

    try:
        # Create phase metadata bundles
        phase_metadata = []
        total_constraints = 0

        for phase_idx, (phase_vars, problem) in enumerate(
            zip(phase_variables.phase_variables, problems, strict=False)
        ):
            # Calculate phase constraint counts (placeholder - will be filled by constraint setup)
            phase_defect_constraints = (
                sum(phase_vars.collocation_points_per_interval) * phase_vars.num_states
            )
            phase_path_constraints = 0  # Will be updated during constraint setup
            phase_total_constraints = phase_defect_constraints + phase_path_constraints

            # Create phase metadata
            phase_meta = PhaseMetadataBundle(
                phase_index=phase_idx,
                global_mesh_nodes=problem.global_normalized_mesh_nodes.copy(),
                phase_constraint_count=phase_total_constraints,
                phase_defect_constraint_count=phase_defect_constraints,
                phase_path_constraint_count=phase_path_constraints,
            )

            phase_metadata.append(phase_meta)
            total_constraints += phase_total_constraints

        # Create multi-phase metadata
        multi_phase_metadata = MultiPhaseMetadataBundle(
            phase_metadata=phase_metadata,
            inter_phase_constraint_count=0,  # Will be updated during constraint setup
            phase_count=phase_variables.phase_count,
            total_nlp_variables=phase_variables.get_nlp_variable_count(),
            total_nlp_constraints=total_constraints,
        )

        logger.debug(
            "Multi-phase metadata created: %d variables, %d constraints (estimated)",
            multi_phase_metadata.total_nlp_variables,
            multi_phase_metadata.total_nlp_constraints,
        )

        return multi_phase_metadata

    except Exception as e:
        logger.error("Multi-phase metadata creation failed: %s", str(e))
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Multi-phase metadata creation failed: {e}", "TrajectoLab multi-phase metadata error"
        ) from e


def validate_multi_phase_variable_structure(
    phase_variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
) -> None:
    """
    Comprehensive validation of multi-phase variable structure.

    Validates the complete hierarchical NLP variable structure matches
    CGPOPS requirements and maintains internal consistency.

    Args:
        phase_variables: Multi-phase variable references to validate
        problems: List of phase problems for cross-validation

    Raises:
        ConfigurationError: If variable structure is invalid
        DataIntegrityError: If internal consistency is violated
    """
    logger.debug("Validating multi-phase variable structure")

    try:
        # Validate basic structure
        if phase_variables.phase_count != len(problems):
            raise ConfigurationError(
                f"Phase count mismatch: variables={phase_variables.phase_count}, "
                f"problems={len(problems)}",
                "Multi-phase variable validation error",
            )

        if phase_variables.phase_count != len(phase_variables.phase_variables):
            raise DataIntegrityError(
                f"Phase variable count inconsistency: count={phase_variables.phase_count}, "
                f"variables={len(phase_variables.phase_variables)}",
                "Multi-phase variable structure corruption",
            )

        # Validate each phase
        for phase_idx, (phase_vars, problem) in enumerate(
            zip(phase_variables.phase_variables, problems, strict=False)
        ):
            # Validate phase index consistency
            if phase_vars.phase_index != phase_idx:
                raise DataIntegrityError(
                    f"Phase index inconsistency: expected {phase_idx}, "
                    f"got {phase_vars.phase_index}",
                    "Multi-phase variable structure corruption",
                )

            # Validate variable counts match problem
            expected_states, expected_controls = problem.get_variable_counts()
            if phase_vars.num_states != expected_states:
                raise ConfigurationError(
                    f"Phase {phase_idx} state count mismatch: "
                    f"variables={phase_vars.num_states}, problem={expected_states}",
                    "Multi-phase variable validation error",
                )

            if phase_vars.num_controls != expected_controls:
                raise ConfigurationError(
                    f"Phase {phase_idx} control count mismatch: "
                    f"variables={phase_vars.num_controls}, problem={expected_controls}",
                    "Multi-phase variable validation error",
                )

            # Validate mesh consistency
            if phase_vars.num_mesh_intervals != len(problem.collocation_points_per_interval):
                raise ConfigurationError(
                    f"Phase {phase_idx} mesh interval count mismatch: "
                    f"variables={phase_vars.num_mesh_intervals}, "
                    f"problem={len(problem.collocation_points_per_interval)}",
                    "Multi-phase variable validation error",
                )

            # Validate state and control variables exist
            if not phase_vars.state_at_mesh_nodes:
                raise DataIntegrityError(
                    f"Phase {phase_idx} missing state variables",
                    "Multi-phase variable structure corruption",
                )

            if not phase_vars.control_variables:
                raise DataIntegrityError(
                    f"Phase {phase_idx} missing control variables",
                    "Multi-phase variable structure corruption",
                )

            # Validate variable array sizes
            expected_state_nodes = phase_vars.num_mesh_intervals + 1
            if len(phase_vars.state_at_mesh_nodes) != expected_state_nodes:
                raise DataIntegrityError(
                    f"Phase {phase_idx} state node count mismatch: "
                    f"expected {expected_state_nodes}, got {len(phase_vars.state_at_mesh_nodes)}",
                    "Multi-phase variable structure corruption",
                )

            if len(phase_vars.control_variables) != phase_vars.num_mesh_intervals:
                raise DataIntegrityError(
                    f"Phase {phase_idx} control variable count mismatch: "
                    f"expected {phase_vars.num_mesh_intervals}, got {len(phase_vars.control_variables)}",
                    "Multi-phase variable structure corruption",
                )

        # Validate global parameters if present
        if phase_variables.global_parameters is not None:
            expected_param_count = len(phase_variables.global_parameter_names)
            actual_param_size = phase_variables.global_parameters.size1()

            if actual_param_size != expected_param_count:
                raise DataIntegrityError(
                    f"Global parameter count mismatch: "
                    f"names={expected_param_count}, variables={actual_param_size}",
                    "Multi-phase variable structure corruption",
                )

        # Use built-in consistency validation
        phase_variables.validate_phase_consistency()

        logger.debug("Multi-phase variable structure validation successful")

    except Exception as e:
        logger.error("Multi-phase variable validation failed: %s", str(e))
        if isinstance(e, (ConfigurationError, DataIntegrityError)):
            raise
        raise DataIntegrityError(
            f"Multi-phase variable validation failed: {e}",
            "TrajectoLab multi-phase variable validation error",
        ) from e


def get_multi_phase_variable_summary(
    phase_variables: MultiPhaseVariableReferences,
) -> dict[str, Any]:
    """
    Get comprehensive summary of multi-phase variable structure.

    Returns detailed information about the NLP variable structure
    for analysis and debugging purposes.

    Args:
        phase_variables: Multi-phase variable references to summarize

    Returns:
        Dictionary containing detailed variable structure information
    """
    summary = {
        "phase_count": phase_variables.phase_count,
        "total_nlp_variables": phase_variables.get_nlp_variable_count(),
        "total_states": phase_variables.total_states,
        "total_controls": phase_variables.total_controls,
        "total_integrals": phase_variables.total_integrals,
        "total_collocation_points": phase_variables.total_collocation_points,
        "has_global_parameters": phase_variables.global_parameters is not None,
        "global_parameter_count": len(phase_variables.global_parameter_names),
        "global_parameter_names": phase_variables.global_parameter_names.copy(),
        "phases": [],
    }

    # Add phase-specific information
    for phase_vars in phase_variables.phase_variables:
        phase_info = {
            "index": phase_vars.phase_index,
            "num_states": phase_vars.num_states,
            "num_controls": phase_vars.num_controls,
            "num_integrals": phase_vars.num_integrals,
            "num_mesh_intervals": phase_vars.num_mesh_intervals,
            "collocation_points_per_interval": phase_vars.collocation_points_per_interval.copy(),
            "total_collocation_points": sum(phase_vars.collocation_points_per_interval),
            "has_state_matrices": len(phase_vars.state_matrices) > 0,
            "has_interior_variables": len(phase_vars.interior_variables) > 0,
        }
        summary["phases"].append(phase_info)

    return summary
