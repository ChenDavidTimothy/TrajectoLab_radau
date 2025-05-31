"""
Main multiphase PHS adaptive mesh refinement algorithm implementation.
"""

import logging
from collections.abc import Callable

import numpy as np

from trajectolab.adaptive.phs.data_structures import (
    AdaptiveParameters,
    MultiphaseAdaptiveState,
    extract_and_prepare_array,
)
from trajectolab.adaptive.phs.error_estimation import (
    calculate_gamma_normalizers_for_phase,
    calculate_relative_error_estimate,
    simulate_dynamics_for_phase_interval_error_estimation,
)
from trajectolab.adaptive.phs.initial_guess import (
    propagate_multiphase_solution_to_new_meshes,
)
from trajectolab.adaptive.phs.numerical import (
    PolynomialInterpolant,
)
from trajectolab.adaptive.phs.refinement import (
    h_reduce_intervals,
    h_refine_params,
    p_reduce_interval,
    p_refine_interval,
)
from trajectolab.radau import (
    RadauBasisComponents,
    compute_barycentric_weights,
    compute_radau_collocation_components,
)
from trajectolab.tl_types import (
    FloatArray,
    MultiPhaseInitialGuess,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


logger = logging.getLogger(__name__)


def _extract_multiphase_solution_trajectories(
    solution: OptimalControlSolution, problem: ProblemProtocol
) -> None:
    """Extract state and control trajectories for all phases from unified solution."""
    if solution.raw_solution is None or solution.opti_object is None:
        raise ValueError("Missing raw solution or opti object")

    opti = solution.opti_object
    raw_sol = solution.raw_solution

    # Extract trajectories for each phase
    for phase_id in problem.get_phase_ids():
        num_states, num_controls = problem.get_phase_variable_counts(phase_id)
        phase_def = problem._phases[phase_id]
        polynomial_degrees = phase_def.collocation_points_per_interval

        # Initialize phase trajectory storage if not present
        if phase_id not in solution.phase_solved_state_trajectories_per_interval:
            solution.phase_solved_state_trajectories_per_interval[phase_id] = []
        if phase_id not in solution.phase_solved_control_trajectories_per_interval:
            solution.phase_solved_control_trajectories_per_interval[phase_id] = []

        # Extract state trajectories for this phase
        if hasattr(opti, "multiphase_variables_reference"):
            variables = opti.multiphase_variables_reference
            if phase_id in variables.phase_variables:
                phase_vars = variables.phase_variables[phase_id]

                # Extract state matrices for each interval in this phase
                for k, state_matrix in enumerate(phase_vars.state_matrices):
                    state_vals = raw_sol.value(state_matrix)
                    state_array = extract_and_prepare_array(
                        state_vals, num_states, polynomial_degrees[k] + 1
                    )
                    solution.phase_solved_state_trajectories_per_interval[phase_id].append(
                        state_array
                    )

                # Extract control trajectories for this phase
                if num_controls > 0:
                    for k, control_var in enumerate(phase_vars.control_variables):
                        control_vals = raw_sol.value(control_var)
                        control_array = extract_and_prepare_array(
                            control_vals, num_controls, polynomial_degrees[k]
                        )
                        solution.phase_solved_control_trajectories_per_interval[phase_id].append(
                            control_array
                        )
                else:
                    # No controls for this phase
                    for k in range(len(polynomial_degrees)):
                        solution.phase_solved_control_trajectories_per_interval[phase_id].append(
                            np.empty((0, polynomial_degrees[k]), dtype=np.float64)
                        )


def _create_phase_interpolants(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> tuple[
    list[Callable[[float | FloatArray], FloatArray] | None],
    list[Callable[[float | FloatArray], FloatArray] | None],
]:
    """Create state and control interpolants for all intervals in a specific phase."""
    if phase_id not in solution.phase_solved_state_trajectories_per_interval:
        raise ValueError(f"Missing state trajectories for phase {phase_id}")

    phase_def = problem._phases[phase_id]
    polynomial_degrees = phase_def.collocation_points_per_interval
    num_intervals = len(polynomial_degrees)

    num_states, num_controls = problem.get_phase_variable_counts(phase_id)

    basis_cache: dict[int, RadauBasisComponents] = {}
    control_weights_cache: dict[int, FloatArray] = {}

    state_evaluators: list[Callable[[float | FloatArray], FloatArray] | None] = [
        None
    ] * num_intervals
    control_evaluators: list[Callable[[float | FloatArray], FloatArray] | None] = [
        None
    ] * num_intervals

    states_list = solution.phase_solved_state_trajectories_per_interval[phase_id]
    controls_list = solution.phase_solved_control_trajectories_per_interval.get(phase_id)

    for k in range(num_intervals):
        try:
            N_k = polynomial_degrees[k]

            # Use cached basis components
            if N_k not in basis_cache:
                basis_cache[N_k] = compute_radau_collocation_components(N_k)

            basis = basis_cache[N_k]

            # Create state interpolant
            state_data = states_list[k]
            state_evaluators[k] = PolynomialInterpolant(
                basis.state_approximation_nodes,
                state_data,
                basis.barycentric_weights_for_state_nodes,
            )

            # Create control interpolant
            if num_controls > 0 and controls_list is not None:
                control_data = controls_list[k]

                if N_k not in control_weights_cache:
                    control_weights_cache[N_k] = compute_barycentric_weights(
                        basis.collocation_nodes
                    )

                control_weights = control_weights_cache[N_k]
                control_evaluators[k] = PolynomialInterpolant(
                    basis.collocation_nodes, control_data, control_weights
                )
            else:
                # Empty control interpolant
                control_evaluators[k] = PolynomialInterpolant(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    np.empty((0, 2), dtype=np.float64),
                    None,
                )

        except Exception as e:
            logger.warning(f"Error creating interpolant for phase {phase_id} interval {k}: {e}")
            # Create fallback interpolants
            if state_evaluators[k] is None:
                state_evaluators[k] = PolynomialInterpolant(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    np.full((num_states, 2), np.nan, dtype=np.float64),
                    None,
                )
            if control_evaluators[k] is None:
                control_evaluators[k] = PolynomialInterpolant(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    np.full((num_controls if num_controls > 0 else 0, 2), np.nan, dtype=np.float64),
                    None,
                )

    return state_evaluators, control_evaluators


def _estimate_phase_errors(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    state_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    control_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    adaptive_params: AdaptiveParameters,
    gamma_factors: FloatArray,
) -> list[float]:
    """Estimate errors for all intervals in a specific phase."""
    phase_def = problem._phases[phase_id]
    polynomial_degrees = phase_def.collocation_points_per_interval
    num_intervals = len(polynomial_degrees)
    errors: list[float] = [np.inf] * num_intervals

    for k in range(num_intervals):
        state_eval = state_evaluators[k]
        control_eval = control_evaluators[k]

        if state_eval is None or control_eval is None:
            errors[k] = np.inf
            continue

        # Pass configurable ODE solver and phase_id
        sim_bundle = simulate_dynamics_for_phase_interval_error_estimation(
            phase_id,
            k,
            solution,
            problem,
            state_eval,
            control_eval,
            adaptive_params.get_ode_solver(),
            ode_rtol=adaptive_params.ode_solver_tolerance,
            n_eval_points=adaptive_params.num_error_sim_points,
        )

        # Calculate relative error
        error = calculate_relative_error_estimate(phase_id, k, sim_bundle, gamma_factors)
        errors[k] = error

    return errors


def _refine_phase_mesh(
    phase_id: PhaseID,
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    errors: list[float],
    adaptive_params: AdaptiveParameters,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    state_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    control_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    gamma_factors: FloatArray,
) -> tuple[list[int], FloatArray]:
    """Refine mesh for a specific phase with correct h-reduction ordering."""
    from typing import NamedTuple

    class MergeCandidate(NamedTuple):
        """Candidate for h-reduction merge."""

        first_idx: int
        second_idx: int
        overall_max_error: float
        merged_degree: int

    # Step 1: Identify intervals needing refinement vs reduction
    num_intervals = len(polynomial_degrees)
    intervals_needing_refinement = set()
    intervals_for_reduction = set()

    for k in range(num_intervals):
        error_k = errors[k]
        if np.isnan(error_k) or np.isinf(error_k) or error_k > adaptive_params.error_tolerance:
            intervals_needing_refinement.add(k)
        else:
            intervals_for_reduction.add(k)

    # Step 2: Process refinement intervals (p-refine or h-refine)
    refinement_actions: dict[int, tuple[str, int] | tuple[str, list[int]]] = {}

    for k in intervals_needing_refinement:
        error_k = errors[k]
        N_k = polynomial_degrees[k]

        # Try p-refinement first
        p_result = p_refine_interval(
            error_k, N_k, adaptive_params.error_tolerance, adaptive_params.max_polynomial_degree
        )

        if p_result.was_p_successful:
            refinement_actions[k] = ("p", p_result.actual_Nk_to_use)
        else:
            # P-refinement failed -> use h-refinement
            h_result = h_refine_params(
                p_result.unconstrained_target_Nk, adaptive_params.min_polynomial_degree
            )
            refinement_actions[k] = ("h", h_result.collocation_nodes_for_new_subintervals)

    # Step 3: Identify h-reduction candidates (CRITICAL: Error-based ordering)
    merge_candidates: list[MergeCandidate] = []

    for k in intervals_for_reduction:
        if (k + 1) in intervals_for_reduction and (k + 1) < num_intervals:
            # Both intervals are candidates for reduction
            state_eval_first = state_evaluators[k]
            state_eval_second = state_evaluators[k + 1]
            control_eval_first = control_evaluators[k]
            control_eval_second = control_evaluators[k + 1]

            # Check if merge is possible
            num_states, num_controls = problem.get_phase_variable_counts(phase_id)
            if (
                state_eval_first is not None
                and state_eval_second is not None
                and (
                    num_controls == 0
                    or (control_eval_first is not None and control_eval_second is not None)
                )
            ):
                can_merge = h_reduce_intervals(
                    phase_id,
                    k,
                    solution,
                    problem,
                    adaptive_params,
                    gamma_factors,
                    state_eval_first,
                    control_eval_first,
                    state_eval_second,
                    control_eval_second,
                )

                if can_merge:
                    # Calculate overall maximum error as per specification
                    error_k = errors[k]
                    error_k_plus_1 = errors[k + 1]
                    overall_max_error = max(error_k, error_k_plus_1)

                    merged_degree = max(polynomial_degrees[k], polynomial_degrees[k + 1])
                    merged_degree = max(
                        adaptive_params.min_polynomial_degree,
                        min(adaptive_params.max_polynomial_degree, merged_degree),
                    )

                    merge_candidates.append(
                        MergeCandidate(
                            first_idx=k,
                            second_idx=k + 1,
                            overall_max_error=overall_max_error,
                            merged_degree=merged_degree,
                        )
                    )

    # CRITICAL: Sort merge candidates by overall maximum relative error (ascending order)
    merge_candidates.sort(key=lambda x: x.overall_max_error)

    # Step 4: Process merges in error-based order, checking for conflicts
    merged_intervals = set()  # Track which intervals have been merged
    approved_merges = []

    for candidate in merge_candidates:
        # Check if either interval is already involved in a merge
        if (
            candidate.first_idx not in merged_intervals
            and candidate.second_idx not in merged_intervals
        ):
            approved_merges.append(candidate)
            merged_intervals.add(candidate.first_idx)
            merged_intervals.add(candidate.second_idx)

    # Step 5: Apply p-reduction to remaining intervals
    reduction_actions = {}  # k -> new_N

    for k in intervals_for_reduction:
        if k not in merged_intervals:  # Not merged
            p_reduce = p_reduce_interval(
                polynomial_degrees[k],
                errors[k],
                adaptive_params.error_tolerance,
                adaptive_params.min_polynomial_degree,
                adaptive_params.max_polynomial_degree,
            )
            reduction_actions[k] = p_reduce.new_num_collocation_nodes

    # Step 6: Build new mesh configuration
    next_polynomial_degrees: list[int] = []
    next_mesh_points = [mesh_points[0]]

    k = 0
    while k < num_intervals:
        if k in refinement_actions:
            # Apply refinement
            action_type, action_data = refinement_actions[k]
            if action_type == "p":
                # Handle p-refinement (single integer)
                assert isinstance(action_data, int), (
                    f"Expected int for p-refinement, got {type(action_data)}"
                )
                next_polynomial_degrees.append(action_data)
                next_mesh_points.append(mesh_points[k + 1])
            else:  # h-refinement
                # Handle h-refinement (list of integers)
                assert isinstance(action_data, list), (
                    f"Expected list for h-refinement, got {type(action_data)}"
                )
                next_polynomial_degrees.extend(action_data)
                # Create new mesh nodes for subintervals
                tau_start = mesh_points[k]
                tau_end = mesh_points[k + 1]
                num_subintervals = len(action_data)
                new_nodes = np.linspace(tau_start, tau_end, num_subintervals + 1, dtype=np.float64)
                next_mesh_points.extend(list(new_nodes[1:]))
            k += 1

        elif any(merge.first_idx == k for merge in approved_merges):
            # Apply h-reduction merge
            merge = next(merge for merge in approved_merges if merge.first_idx == k)
            next_polynomial_degrees.append(merge.merged_degree)
            next_mesh_points.append(mesh_points[k + 2])  # Skip shared mesh point
            k += 2  # Skip both merged intervals

        else:
            # Apply p-reduction or keep unchanged
            if k in reduction_actions:
                next_polynomial_degrees.append(reduction_actions[k])
            else:
                next_polynomial_degrees.append(polynomial_degrees[k])
            next_mesh_points.append(mesh_points[k + 1])
            k += 1

    return next_polynomial_degrees, np.array(next_mesh_points, dtype=np.float64)


def solve_multiphase_phs_adaptive_internal(
    problem: ProblemProtocol,
    error_tolerance: float,
    max_iterations: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
    ode_solver_tolerance: float,
    ode_method: str,
    ode_max_step: float | None,
    ode_solver,
    num_error_sim_points: int,
    initial_guess: MultiPhaseInitialGuess | None = None,
) -> OptimalControlSolution:
    """
    Internal multiphase PHS-Adaptive mesh refinement algorithm implementation.
    """

    # Log algorithm start
    logger.info(
        "Starting multiphase adaptive mesh refinement: tolerance=%.1e, max_iter=%d",
        error_tolerance,
        max_iterations,
    )

    # Store adaptive parameters
    adaptive_params = AdaptiveParameters(
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        ode_solver_tolerance=ode_solver_tolerance,
        num_error_sim_points=num_error_sim_points,
        ode_method=ode_method,
        ode_max_step=ode_max_step,
        ode_solver=ode_solver,
    )

    # Initialize multiphase adaptive state
    phase_ids = problem.get_phase_ids()
    adaptive_state = MultiphaseAdaptiveState(
        phase_polynomial_degrees={},
        phase_mesh_points={},
        phase_converged=dict.fromkeys(phase_ids, False),
        iteration=0,
    )

    # Initialize mesh configuration from problem
    for phase_id in phase_ids:
        phase_def = problem._phases[phase_id]
        if not phase_def.mesh_configured:
            raise ValueError(f"Phase {phase_id} mesh must be configured before adaptive solving")

        adaptive_state.phase_polynomial_degrees[phase_id] = list(
            phase_def.collocation_points_per_interval
        )
        adaptive_state.phase_mesh_points[phase_id] = phase_def.global_normalized_mesh_nodes.copy()

    # Log initial mesh configuration
    for phase_id in phase_ids:
        logger.debug(
            "Phase %d initial mesh: degrees=%s, intervals=%d",
            phase_id,
            adaptive_state.phase_polynomial_degrees[phase_id],
            len(adaptive_state.phase_polynomial_degrees[phase_id]),
        )

    # Import unified multiphase solver
    from trajectolab.direct_solver import solve_multiphase_radau_collocation

    # Main adaptive refinement loop
    for iteration in range(max_iterations):
        adaptive_state.iteration = iteration

        # Log iteration start
        logger.info("Multiphase adaptive iteration %d/%d", iteration + 1, max_iterations)

        # Configure all phase meshes in the unified problem
        adaptive_state.configure_problem_meshes(problem)

        # Handle initial guess
        if iteration == 0:
            _handle_first_iteration_initial_guess(problem, initial_guess)
        else:
            if adaptive_state.most_recent_unified_solution is None:
                raise ValueError("No previous unified solution available for propagation")

            logger.debug("Propagating unified solution to new multiphase meshes")
            propagated_guess = propagate_multiphase_solution_to_new_meshes(
                adaptive_state.most_recent_unified_solution,
                problem,
                adaptive_state.phase_polynomial_degrees,
                adaptive_state.phase_mesh_points,
            )
            problem.initial_guess = propagated_guess

        # Solve unified multiphase NLP
        logger.debug("Solving unified multiphase NLP for iteration %d", iteration + 1)
        solution = solve_multiphase_radau_collocation(problem)

        if not solution.success:
            logger.warning(
                "Unified NLP solve failed in iteration %d: %s", iteration + 1, solution.message
            )

            if adaptive_state.most_recent_unified_solution is not None:
                adaptive_state.most_recent_unified_solution.message = f"Adaptive stopped due to solver failure in iteration {iteration + 1}: {solution.message}"
                adaptive_state.most_recent_unified_solution.success = False
                logger.info("Returning last successful unified solution")
                return adaptive_state.most_recent_unified_solution
            else:
                solution.message = (
                    f"Multiphase adaptive failed in first iteration: {solution.message}"
                )
                logger.error("No successful unified solution obtained")
                return solution

        # Extract trajectories from unified solution for all phases
        try:
            _extract_multiphase_solution_trajectories(solution, problem)
            logger.debug("Multiphase solution trajectories extracted successfully")
        except Exception as e:
            logger.error(
                "Failed to extract multiphase trajectories in iteration %d: %s",
                iteration + 1,
                str(e),
            )
            solution.message = f"Failed to extract multiphase trajectories: {e}"
            solution.success = False
            return solution

        # CRITICAL FIX: Store the mesh information that was ACTUALLY used for this solve
        # This must be done BEFORE any mesh refinement
        solution.phase_mesh_intervals = adaptive_state.phase_polynomial_degrees.copy()
        solution.phase_mesh_nodes = adaptive_state.phase_mesh_points.copy()

        # Update most recent successful solution
        adaptive_state.most_recent_unified_solution = solution

        # Process each phase for convergence and refinement
        any_phase_needs_refinement = False

        for phase_id in phase_ids:
            logger.debug(f"Processing phase {phase_id} for iteration {iteration + 1}")

            # Calculate gamma normalization factors for this phase
            gamma_factors = calculate_gamma_normalizers_for_phase(solution, problem, phase_id)
            num_states, _ = problem.get_phase_variable_counts(phase_id)
            if gamma_factors is None and num_states > 0:
                logger.error(
                    "Failed to calculate gamma normalizers for phase %d in iteration %d",
                    phase_id,
                    iteration + 1,
                )
                solution.message = f"Failed to calculate gamma normalizers for phase {phase_id} at iteration {iteration + 1}"
                solution.success = False
                return solution

            safe_gamma = (
                gamma_factors
                if gamma_factors is not None
                else np.ones((num_states, 1), dtype=np.float64)
            )

            # Create interpolants and calculate errors for this phase
            state_evaluators, control_evaluators = _create_phase_interpolants(
                solution, problem, phase_id
            )

            phase_errors = _estimate_phase_errors(
                solution,
                problem,
                phase_id,
                state_evaluators,
                control_evaluators,
                adaptive_params,
                safe_gamma,
            )

            # Log phase error analysis
            max_phase_error = max(phase_errors) if phase_errors else 0.0
            avg_phase_error = np.mean(phase_errors) if phase_errors else 0.0
            logger.info(
                "Phase %d error analysis: max=%.2e, avg=%.2e, tolerance=%.2e",
                phase_id,
                max_phase_error,
                avg_phase_error,
                error_tolerance,
            )

            # Check phase convergence
            phase_converged = all(
                not (np.isnan(error) or np.isinf(error))
                and error <= adaptive_params.error_tolerance
                for error in phase_errors
            )

            adaptive_state.phase_converged[phase_id] = phase_converged

            if not phase_converged:
                any_phase_needs_refinement = True

                # Refine mesh for this phase
                try:
                    logger.debug("Refining mesh for phase %d", phase_id)
                    (
                        adaptive_state.phase_polynomial_degrees[phase_id],
                        adaptive_state.phase_mesh_points[phase_id],
                    ) = _refine_phase_mesh(
                        phase_id,
                        adaptive_state.phase_polynomial_degrees[phase_id],
                        adaptive_state.phase_mesh_points[phase_id],
                        phase_errors,
                        adaptive_params,
                        solution,
                        problem,
                        state_evaluators,
                        control_evaluators,
                        safe_gamma,
                    )

                    # Log mesh refinement result
                    logger.debug(
                        "Phase %d mesh refined: new_degrees=%s, new_intervals=%d",
                        phase_id,
                        adaptive_state.phase_polynomial_degrees[phase_id],
                        len(adaptive_state.phase_polynomial_degrees[phase_id]),
                    )

                except Exception as e:
                    logger.error(
                        "Mesh refinement failed for phase %d in iteration %d: %s",
                        phase_id,
                        iteration + 1,
                        str(e),
                    )
                    if adaptive_state.most_recent_unified_solution is not None:
                        adaptive_state.most_recent_unified_solution.message = (
                            f"Mesh refinement failed for phase {phase_id}: {e}"
                        )
                        adaptive_state.most_recent_unified_solution.success = False
                        return adaptive_state.most_recent_unified_solution

        # Check global convergence
        if not any_phase_needs_refinement:
            # Log convergence success
            logger.info("Multiphase adaptive refinement converged in %d iterations", iteration + 1)

            solution.message = (
                f"Multiphase adaptive mesh converged to tolerance {adaptive_params.error_tolerance:.1e} "
                f"in {iteration + 1} iterations"
            )
            return solution

    # Maximum iterations reached
    logger.warning(
        "Multiphase adaptive refinement reached maximum iterations (%d) without convergence",
        max_iterations,
    )

    if adaptive_state.most_recent_unified_solution is not None:
        max_iter_msg = (
            f"Reached maximum iterations ({max_iterations}) without full convergence "
            f"to tolerance {adaptive_params.error_tolerance:.1e}"
        )
        adaptive_state.most_recent_unified_solution.message = max_iter_msg
        return adaptive_state.most_recent_unified_solution
    else:
        # Create failure solution
        failed_solution = OptimalControlSolution()
        failed_solution.success = False
        failed_solution.message = (
            f"No successful unified solution obtained in {max_iterations} iterations"
        )
        logger.error("Multiphase adaptive refinement failed completely")
        return failed_solution


def _handle_first_iteration_initial_guess(
    problem: ProblemProtocol,
    initial_guess: MultiPhaseInitialGuess | None,
) -> None:
    """
    Handle initial guess for the first iteration of multiphase adaptive.

    Args:
        problem: The multiphase problem object
        initial_guess: User-provided multiphase initial guess
    """
    if problem.initial_guess is not None:
        try:
            from trajectolab.input_validation import validate_multiphase_initial_guess_structure

            validate_multiphase_initial_guess_structure(problem.initial_guess, problem)
        except ValueError as e:
            raise ValueError(f"Initial guess invalid for multiphase mesh: {e}") from e
    elif initial_guess is not None:
        # Set the initial guess and validate
        problem.initial_guess = initial_guess
        try:
            from trajectolab.input_validation import validate_multiphase_initial_guess_structure

            validate_multiphase_initial_guess_structure(problem.initial_guess, problem)
        except ValueError as e:
            raise ValueError(f"Initial guess invalid for multiphase mesh: {e}") from e
    else:
        problem.initial_guess = None
