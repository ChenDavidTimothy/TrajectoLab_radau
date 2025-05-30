"""
Multi-phase constraint application for collocation, path, and event constraints.

This module implements the faithful CGPOPS block-diagonal constraint structure from
Section 4.2, where phase-internal constraints form diagonal blocks and inter-phase
event constraints provide the off-diagonal coupling between phases.

The constraint structure follows CGPOPS exactly:
- Defect constraints Δ^(p) = 0 for each phase (diagonal blocks)
- Path constraints applied at collocation points within each phase (diagonal blocks)
- Inter-phase event constraints b_min ≤ b(E^(1), ..., E^(P), s) ≤ b_max (coupling)
- Integral constraints ρ^(p) = 0 for each phase (diagonal blocks)
"""

import logging

import casadi as ca

from ..exceptions import ConfigurationError, DataIntegrityError
from ..radau import compute_radau_collocation_components
from ..tl_types import (
    MultiPhaseProblemProtocol,
    ProblemProtocol,
)
from .constraints_solver import (
    apply_collocation_constraints,
    apply_path_constraints,
)
from .integrals_solver import apply_integral_constraints, setup_integrals
from .types_solver import MultiPhaseMetadataBundle, MultiPhaseVariableReferences


# Library logger
logger = logging.getLogger(__name__)


def apply_multi_phase_collocation_constraints(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Apply collocation constraints for all phases (diagonal blocks).

    Implements the defect constraints from CGPOPS Equation (18) for each phase:
    Δ^(p) = D^(p)Y^(p) - (t_f^(p) - t_0^(p))/2 * A^(p) = 0

    These constraints form the diagonal blocks of the constraint Jacobian as
    described in CGPOPS Section 4.2, with no coupling between phases.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problems: List of phase problems
        metadata: Multi-phase metadata bundle

    Raises:
        DataIntegrityError: If constraint application fails
    """
    logger.debug("Applying collocation constraints for %d phases", variables.phase_count)

    if len(problems) != variables.phase_count:
        raise ConfigurationError(
            f"Problem count ({len(problems)}) must match phase count ({variables.phase_count})",
            "Multi-phase collocation constraint error",
        )

    try:
        total_defect_constraints = 0

        for phase_idx, (phase_vars, problem) in enumerate(
            zip(variables.phase_variables, problems, strict=False)
        ):
            logger.debug("Applying collocation constraints for phase %d", phase_idx)

            # Get phase metadata
            phase_metadata = metadata.get_phase_metadata(phase_idx)

            # Apply collocation constraints for each mesh interval in this phase
            for interval_idx in range(phase_vars.num_mesh_intervals):
                num_colloc_nodes = phase_vars.collocation_points_per_interval[interval_idx]

                # Get Radau basis components for this interval
                basis_components = compute_radau_collocation_components(num_colloc_nodes)

                # Get state matrix for this interval
                if interval_idx >= len(phase_vars.state_matrices):
                    raise DataIntegrityError(
                        f"Phase {phase_idx} missing state matrix for interval {interval_idx}",
                        "Multi-phase collocation constraint setup error",
                    )

                state_at_nodes = phase_vars.state_matrices[interval_idx]
                control_variables = phase_vars.control_variables[interval_idx]

                # Apply collocation constraints using existing single-phase function
                apply_collocation_constraints(
                    opti=opti,
                    mesh_interval_index=interval_idx,
                    state_at_nodes=state_at_nodes,
                    control_variables=control_variables,
                    basis_components=basis_components,
                    global_normalized_mesh_nodes=problem.global_normalized_mesh_nodes,
                    initial_time_variable=phase_vars.initial_time,
                    terminal_time_variable=phase_vars.terminal_time,
                    dynamics_function=problem.get_dynamics_function(),
                    problem_parameters=problem._parameters,
                    problem=problem,
                )

                # Count defect constraints for this interval
                defect_constraints_count = num_colloc_nodes * phase_vars.num_states
                total_defect_constraints += defect_constraints_count

                logger.debug(
                    "Phase %d interval %d: applied %d defect constraints",
                    phase_idx,
                    interval_idx,
                    defect_constraints_count,
                )

            # Update phase metadata with actual defect constraint count
            phase_metadata.phase_defect_constraint_count = (
                sum(phase_vars.collocation_points_per_interval) * phase_vars.num_states
            )

        logger.debug(
            "Applied %d total defect constraints across all phases", total_defect_constraints
        )

    except Exception as e:
        logger.error("Multi-phase collocation constraint application failed: %s", str(e))
        if isinstance(e, (ConfigurationError, DataIntegrityError)):
            raise
        raise DataIntegrityError(
            f"Multi-phase collocation constraint application failed: {e}",
            "TrajectoLab multi-phase collocation constraint error",
        ) from e


def apply_multi_phase_path_constraints(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Apply path constraints for all phases (diagonal blocks).

    Implements path constraints from CGPOPS Equation (19) applied at collocation
    points within each phase. These constraints form diagonal blocks in the
    constraint Jacobian with no inter-phase coupling.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problems: List of phase problems
        metadata: Multi-phase metadata bundle

    Raises:
        DataIntegrityError: If constraint application fails
    """
    logger.debug("Applying path constraints for %d phases", variables.phase_count)

    try:
        total_path_constraints = 0

        for phase_idx, (phase_vars, problem) in enumerate(
            zip(variables.phase_variables, problems, strict=False)
        ):
            # Check if this phase has path constraints
            path_constraints_function = problem.get_path_constraints_function()
            if path_constraints_function is None:
                logger.debug("Phase %d has no path constraints", phase_idx)
                continue

            logger.debug("Applying path constraints for phase %d", phase_idx)

            # Get phase metadata
            phase_metadata = metadata.get_phase_metadata(phase_idx)
            phase_path_constraint_count = 0

            # Apply path constraints for each mesh interval in this phase
            for interval_idx in range(phase_vars.num_mesh_intervals):
                num_colloc_nodes = phase_vars.collocation_points_per_interval[interval_idx]

                # Get Radau basis components for this interval
                basis_components = compute_radau_collocation_components(num_colloc_nodes)

                # Get state matrix and control variables for this interval
                state_at_nodes = phase_vars.state_matrices[interval_idx]
                control_variables = phase_vars.control_variables[interval_idx]

                # Apply path constraints using existing single-phase function
                apply_path_constraints(
                    opti=opti,
                    mesh_interval_index=interval_idx,
                    state_at_nodes=state_at_nodes,
                    control_variables=control_variables,
                    basis_components=basis_components,
                    global_normalized_mesh_nodes=problem.global_normalized_mesh_nodes,
                    initial_time_variable=phase_vars.initial_time,
                    terminal_time_variable=phase_vars.terminal_time,
                    path_constraints_function=path_constraints_function,
                    problem_parameters=problem._parameters,
                    problem=problem,
                )

                # Estimate path constraint count (actual count will be determined during constraint evaluation)
                estimated_path_constraints = num_colloc_nodes  # Placeholder estimate
                phase_path_constraint_count += estimated_path_constraints

                logger.debug(
                    "Phase %d interval %d: applied path constraints", phase_idx, interval_idx
                )

            # Update phase metadata with path constraint count
            phase_metadata.phase_path_constraint_count = phase_path_constraint_count
            total_path_constraints += phase_path_constraint_count

        logger.debug(
            "Applied path constraints for all phases (estimated count: %d)", total_path_constraints
        )

    except Exception as e:
        logger.error("Multi-phase path constraint application failed: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Multi-phase path constraint application failed: {e}",
            "TrajectoLab multi-phase path constraint error",
        ) from e


def apply_inter_phase_event_constraints(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Apply inter-phase event constraints linking phases.

    Fixed to handle CasADi context issues by reconstructing constraints
    using actual optimization variables instead of stored expressions.
    """
    # Check if there are inter-phase constraints to apply
    if not problem.inter_phase_constraints:
        logger.debug("No inter-phase constraints to apply")
        return

    logger.debug("Applying %d inter-phase event constraints", len(problem.inter_phase_constraints))

    try:
        constraint_count = 0

        for i, constraint_expr in enumerate(problem.inter_phase_constraints):
            if constraint_expr is None:
                raise DataIntegrityError(
                    f"Inter-phase constraint {i} expression is None",
                    "Inter-phase constraint application error",
                )

            # Create substitution map to replace symbolic variables with actual optimization variables
            substitution_map = _create_variable_substitution_map(variables, problem)

            # Reconstruct constraint using actual optimization variables
            try:
                reconstructed_constraint = _substitute_variables_in_constraint(
                    constraint_expr, substitution_map
                )

                # Apply reconstructed constraint
                opti.subject_to(reconstructed_constraint)
                constraint_count += 1

                logger.debug("Applied inter-phase constraint %d", i)

            except Exception as sub_error:
                logger.error("Failed to apply inter-phase constraint %d: %s", i, str(sub_error))
                raise DataIntegrityError(
                    f"Failed to apply inter-phase constraint {i}: {sub_error}",
                    "Inter-phase constraint application error",
                ) from sub_error

        # Update metadata with actual constraint count
        metadata.inter_phase_constraint_count = constraint_count

        logger.debug("Applied %d inter-phase event constraints", constraint_count)

    except Exception as e:
        logger.error("Inter-phase event constraint application failed: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Inter-phase event constraint application failed: {e}",
            "TrajectoLab inter-phase constraint error",
        ) from e


def _create_variable_substitution_map(
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
) -> dict[ca.MX, ca.MX]:
    """
    Create mapping from symbolic variables to actual optimization variables.

    This maps the symbols used in constraint expressions to the actual
    variables created in the current Opti context.
    """
    substitution_map = {}

    try:
        # Map phase variables
        for phase_idx, (phase_vars, phase_problem) in enumerate(
            zip(variables.phase_variables, problem.phases, strict=False)
        ):
            # Get phase variable symbols and names
            state_symbols = phase_problem.get_ordered_state_symbols()
            state_initial_symbols = (
                phase_problem._variable_state.get_ordered_state_initial_symbols()
            )
            state_final_symbols = phase_problem._variable_state.get_ordered_state_final_symbols()

            # Map initial state symbols to actual initial state variables (first mesh node)
            for i, (state_sym, initial_sym) in enumerate(
                zip(state_symbols, state_initial_symbols, strict=False)
            ):
                if len(phase_vars.state_at_mesh_nodes) > 0:
                    # Map both the regular state symbol and initial symbol to the first mesh node
                    substitution_map[initial_sym] = phase_vars.state_at_mesh_nodes[0][i]

            # Map final state symbols to actual final state variables (last mesh node)
            for i, final_sym in enumerate(state_final_symbols):
                if len(phase_vars.state_at_mesh_nodes) > 0:
                    # Map final symbol to the last mesh node
                    substitution_map[final_sym] = phase_vars.state_at_mesh_nodes[-1][i]

            # Map time symbols
            if phase_problem._variable_state.sym_time_initial is not None:
                substitution_map[phase_problem._variable_state.sym_time_initial] = (
                    phase_vars.initial_time
                )

            if phase_problem._variable_state.sym_time_final is not None:
                substitution_map[phase_problem._variable_state.sym_time_final] = (
                    phase_vars.terminal_time
                )

            # Map integral symbols if present
            integral_symbols = phase_problem._variable_state.integral_symbols
            if integral_symbols and phase_vars.integral_variables is not None:
                for i, integral_sym in enumerate(integral_symbols):
                    if phase_vars.num_integrals == 1:
                        substitution_map[integral_sym] = phase_vars.integral_variables
                    else:
                        substitution_map[integral_sym] = phase_vars.integral_variables[i]

        # Map global parameters if present
        if hasattr(problem, "_global_parameter_symbols"):
            for param_name, param_symbol in problem._global_parameter_symbols.items():
                if variables.global_parameters is not None:
                    # Find parameter index
                    if param_name in variables.global_parameter_names:
                        param_idx = variables.global_parameter_names.index(param_name)
                        if variables.global_parameters.size1() == 1:
                            substitution_map[param_symbol] = variables.global_parameters
                        else:
                            substitution_map[param_symbol] = variables.global_parameters[param_idx]
                else:
                    # Use constant value if no optimization variable
                    if param_name in problem.global_parameters:
                        substitution_map[param_symbol] = ca.MX(
                            problem.global_parameters[param_name]
                        )

        logger.debug("Created substitution map with %d entries", len(substitution_map))
        return substitution_map

    except Exception as e:
        logger.error("Failed to create variable substitution map: %s", str(e))
        raise DataIntegrityError(
            f"Failed to create variable substitution map: {e}",
            "Inter-phase constraint substitution error",
        ) from e


def _substitute_variables_in_constraint(
    constraint_expr: ca.MX,
    substitution_map: dict[ca.MX, ca.MX],
) -> ca.MX:
    """
    Substitute symbolic variables with actual optimization variables in constraint.

    This reconstructs the constraint expression using variables from the current Opti context.
    """
    try:
        if not substitution_map:
            logger.warning("Empty substitution map, returning original constraint")
            return constraint_expr

        # Extract old and new variables for substitution
        old_vars = list(substitution_map.keys())
        new_vars = list(substitution_map.values())

        # Perform substitution
        reconstructed_expr = ca.substitute([constraint_expr], old_vars, new_vars)[0]

        logger.debug("Successfully substituted variables in constraint")
        return reconstructed_expr

    except Exception as e:
        logger.error("Variable substitution failed: %s", str(e))
        raise DataIntegrityError(
            f"Variable substitution failed: {e}",
            "Inter-phase constraint variable substitution error",
        ) from e


def debug_print_variable_mapping(
    variables: MultiPhaseVariableReferences,
    problem: MultiPhaseProblemProtocol,
) -> None:
    """
    Debug function to print variable mapping information.
    Call this from apply_inter_phase_event_constraints if needed.
    """
    print("\n=== Variable Mapping Debug ===")

    for phase_idx, (phase_vars, phase_problem) in enumerate(
        zip(variables.phase_variables, problem.phases, strict=False)
    ):
        print(f"\nPhase {phase_idx}:")

        # State variables
        state_names = phase_problem.get_ordered_state_names()
        state_initial_symbols = phase_problem._variable_state.get_ordered_state_initial_symbols()
        state_final_symbols = phase_problem._variable_state.get_ordered_state_final_symbols()

        print(f"  State variables: {state_names}")
        print(f"  Initial symbols: {[s.name() for s in state_initial_symbols]}")
        print(f"  Final symbols: {[s.name() for s in state_final_symbols]}")
        print(f"  Actual mesh nodes: {len(phase_vars.state_at_mesh_nodes)}")

        # Time variables
        if phase_problem._variable_state.sym_time_initial:
            print(f"  Initial time symbol: {phase_problem._variable_state.sym_time_initial.name()}")
        if phase_problem._variable_state.sym_time_final:
            print(f"  Final time symbol: {phase_problem._variable_state.sym_time_final.name()}")

    print("=== End Variable Mapping Debug ===\n")


def apply_multi_phase_integral_constraints(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
    metadata: MultiPhaseMetadataBundle,
    accumulated_integral_expressions: dict[int, list[ca.MX]],
) -> None:
    """
    Apply integral constraints for all phases (diagonal blocks).

    Implements integral constraints from CGPOPS Equation (23) for each phase:
    ρ^(p) = Q^(p) - (t_f^(p) - t_0^(p))/2 * [w^(p)]^T * G^(p) = 0

    These constraints form diagonal blocks in the constraint Jacobian.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problems: List of phase problems
        metadata: Multi-phase metadata bundle
        accumulated_integral_expressions: Dictionary mapping phase index to accumulated expressions

    Raises:
        DataIntegrityError: If integral constraint application fails
    """
    logger.debug("Applying integral constraints for %d phases", variables.phase_count)

    try:
        total_integral_constraints = 0

        for phase_idx, (phase_vars, problem) in enumerate(
            zip(variables.phase_variables, problems, strict=False)
        ):
            # Check if this phase has integrals
            if phase_vars.num_integrals == 0:
                logger.debug("Phase %d has no integrals", phase_idx)
                continue

            if phase_vars.integral_variables is None:
                raise DataIntegrityError(
                    f"Phase {phase_idx} has {phase_vars.num_integrals} integrals but no integral variables",
                    "Multi-phase integral constraint error",
                )

            # Get accumulated integral expressions for this phase
            if phase_idx not in accumulated_integral_expressions:
                raise DataIntegrityError(
                    f"Phase {phase_idx} missing accumulated integral expressions",
                    "Multi-phase integral constraint error",
                )

            phase_accumulated_expressions = accumulated_integral_expressions[phase_idx]

            # Apply integral constraints using existing single-phase function
            apply_integral_constraints(
                opti=opti,
                integral_variables=phase_vars.integral_variables,
                accumulated_integral_expressions=phase_accumulated_expressions,
                num_integrals=phase_vars.num_integrals,
            )

            total_integral_constraints += phase_vars.num_integrals

            logger.debug(
                "Phase %d: applied %d integral constraints", phase_idx, phase_vars.num_integrals
            )

        logger.debug("Applied %d total integral constraints", total_integral_constraints)

    except Exception as e:
        logger.error("Multi-phase integral constraint application failed: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Multi-phase integral constraint application failed: {e}",
            "TrajectoLab multi-phase integral constraint error",
        ) from e


def setup_multi_phase_integrals(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
) -> dict[int, list[ca.MX]]:
    """
    Set up integral calculations for all phases using quadrature.

    Computes integrals for each phase using Radau quadrature rules,
    accumulating the quadrature sums for later constraint application.

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problems: List of phase problems

    Returns:
        Dictionary mapping phase index to accumulated integral expressions

    Raises:
        DataIntegrityError: If integral setup fails
    """
    logger.debug("Setting up integrals for %d phases", variables.phase_count)

    try:
        accumulated_integral_expressions = {}
        total_integrals = 0

        for phase_idx, (phase_vars, problem) in enumerate(
            zip(variables.phase_variables, problems, strict=False)
        ):
            # Check if this phase has integrals
            if phase_vars.num_integrals == 0:
                logger.debug("Phase %d has no integrals to set up", phase_idx)
                continue

            # Initialize accumulated expressions for this phase
            phase_accumulated_expressions = [ca.MX(0) for _ in range(phase_vars.num_integrals)]

            # Get integrand function for this phase
            integrand_function = problem.get_integrand_function()
            if integrand_function is None:
                raise DataIntegrityError(
                    f"Phase {phase_idx} has {phase_vars.num_integrals} integrals but no integrand function",
                    "Multi-phase integral setup error",
                )

            # Set up integrals for each mesh interval in this phase
            for interval_idx in range(phase_vars.num_mesh_intervals):
                num_colloc_nodes = phase_vars.collocation_points_per_interval[interval_idx]

                # Get Radau basis components for this interval
                basis_components = compute_radau_collocation_components(num_colloc_nodes)

                # Get state matrix and control variables for this interval
                state_at_nodes = phase_vars.state_matrices[interval_idx]
                control_variables = phase_vars.control_variables[interval_idx]

                # Set up integrals using existing single-phase function
                setup_integrals(
                    opti=opti,
                    mesh_interval_index=interval_idx,
                    state_at_nodes=state_at_nodes,
                    control_variables=control_variables,
                    basis_components=basis_components,
                    global_normalized_mesh_nodes=problem.global_normalized_mesh_nodes,
                    initial_time_variable=phase_vars.initial_time,
                    terminal_time_variable=phase_vars.terminal_time,
                    integral_integrand_function=integrand_function,
                    problem_parameters=problem._parameters,
                    num_integrals=phase_vars.num_integrals,
                    accumulated_integral_expressions=phase_accumulated_expressions,
                )

            # Store accumulated expressions for this phase
            accumulated_integral_expressions[phase_idx] = phase_accumulated_expressions
            total_integrals += phase_vars.num_integrals

            logger.debug(
                "Phase %d: set up %d integrals across %d intervals",
                phase_idx,
                phase_vars.num_integrals,
                phase_vars.num_mesh_intervals,
            )

        logger.debug("Set up %d total integrals across all phases", total_integrals)
        return accumulated_integral_expressions

    except Exception as e:
        logger.error("Multi-phase integral setup failed: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Multi-phase integral setup failed: {e}",
            "TrajectoLab multi-phase integral setup error",
        ) from e


def apply_all_multi_phase_constraints(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problems: list[ProblemProtocol],
    multi_phase_problem: MultiPhaseProblemProtocol,
    metadata: MultiPhaseMetadataBundle,
) -> None:
    """
    Apply all multi-phase constraints in proper order.

    Main orchestration function that applies all constraint types in the correct
    sequence to build the complete CGPOPS block-diagonal constraint structure:

    1. Phase-internal collocation constraints (diagonal blocks)
    2. Phase-internal path constraints (diagonal blocks)
    3. Phase-internal integral constraints (diagonal blocks)
    4. Inter-phase event constraints (off-diagonal coupling)

    Args:
        opti: CasADi optimization object
        variables: Multi-phase variable references
        problems: List of phase problems
        multi_phase_problem: Multi-phase problem protocol
        metadata: Multi-phase metadata bundle

    Raises:
        DataIntegrityError: If constraint application fails
    """
    logger.debug("Applying all multi-phase constraints")

    try:
        # Step 1: Apply collocation constraints (defect constraints) - diagonal blocks
        logger.debug("Step 1: Applying collocation constraints")
        apply_multi_phase_collocation_constraints(opti, variables, problems, metadata)

        # Step 2: Apply path constraints - diagonal blocks
        logger.debug("Step 2: Applying path constraints")
        apply_multi_phase_path_constraints(opti, variables, problems, metadata)

        # Step 3: Set up and apply integral constraints - diagonal blocks
        logger.debug("Step 3: Setting up and applying integral constraints")
        accumulated_integral_expressions = setup_multi_phase_integrals(opti, variables, problems)
        apply_multi_phase_integral_constraints(
            opti, variables, problems, metadata, accumulated_integral_expressions
        )

        # Step 4: Apply inter-phase event constraints - off-diagonal coupling
        logger.debug("Step 4: Applying inter-phase event constraints")
        apply_inter_phase_event_constraints(opti, variables, multi_phase_problem, metadata)

        # Update final constraint counts in metadata
        _update_constraint_counts(metadata)

        total_constraints = metadata.total_nlp_constraints
        logger.info("Applied all multi-phase constraints: %d total constraints", total_constraints)

    except Exception as e:
        logger.error("Multi-phase constraint application failed: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Multi-phase constraint application failed: {e}",
            "TrajectoLab multi-phase constraint application error",
        ) from e


def _update_constraint_counts(metadata: MultiPhaseMetadataBundle) -> None:
    """
    Update final constraint counts in metadata bundle.

    Calculates total constraint counts across all phases and constraint types
    for accurate metadata reporting and solution analysis.

    Args:
        metadata: Multi-phase metadata bundle to update
    """
    total_phase_constraints = 0

    # Update phase-specific constraint counts and sum them
    for phase_meta in metadata.phase_metadata:
        phase_meta.phase_constraint_count = (
            phase_meta.phase_defect_constraint_count + phase_meta.phase_path_constraint_count
        )
        total_phase_constraints += phase_meta.phase_constraint_count

    # Update total constraint count
    metadata.total_nlp_constraints = total_phase_constraints + metadata.inter_phase_constraint_count

    logger.debug(
        "Updated constraint counts: %d phase constraints, %d inter-phase constraints, %d total",
        total_phase_constraints,
        metadata.inter_phase_constraint_count,
        metadata.total_nlp_constraints,
    )
