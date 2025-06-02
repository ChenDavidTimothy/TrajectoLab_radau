# trajectolab/adaptive/phs/algorithm.py - Updated imports section

import logging
from collections.abc import Callable
from typing import cast

import numpy as np

from trajectolab.adaptive.phs.data_structures import (
    AdaptiveParameters,
    MultiphaseAdaptiveState,
    ensure_2d_array,
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
from trajectolab.exceptions import DataIntegrityError
from trajectolab.radau import (
    RadauBasisComponents,
    compute_barycentric_weights,
    compute_radau_collocation_components,
)
from trajectolab.tl_types import (
    AdaptiveAlgorithmData,
    FloatArray,
    MultiPhaseInitialGuess,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


logger = logging.getLogger(__name__)


def _extract_state_matrices_for_phase(
    phase_vars, polynomial_degrees: list[int], raw_sol, num_states: int, phase_id: PhaseID
) -> list[FloatArray]:
    """EXTRACTED: Extract state matrices for a single phase."""
    if len(phase_vars.state_matrices) != len(polynomial_degrees):
        raise DataIntegrityError(
            f"Phase {phase_id} state matrices count ({len(phase_vars.state_matrices)}) != polynomial degrees count ({len(polynomial_degrees)})",
            "Solution extraction state matrix mismatch",
        )

    state_trajectories = []
    for k, state_matrix in enumerate(phase_vars.state_matrices):
        state_vals = raw_sol.value(state_matrix)
        state_array = ensure_2d_array(state_vals, num_states, polynomial_degrees[k] + 1)
        state_trajectories.append(state_array)

    return state_trajectories


def _extract_control_variables_for_phase(
    phase_vars, polynomial_degrees: list[int], raw_sol, num_controls: int, phase_id: PhaseID
) -> list[FloatArray]:
    """EXTRACTED: Extract control variables for a single phase."""
    if num_controls == 0:
        # No controls for this phase
        return [
            np.empty((0, polynomial_degrees[k]), dtype=np.float64)
            for k in range(len(polynomial_degrees))
        ]

    if len(phase_vars.control_variables) != len(polynomial_degrees):
        raise DataIntegrityError(
            f"Phase {phase_id} control variables count ({len(phase_vars.control_variables)}) != polynomial degrees count ({len(polynomial_degrees)})",
            "Solution extraction control matrix mismatch",
        )

    control_trajectories = []
    for k, control_var in enumerate(phase_vars.control_variables):
        control_vals = raw_sol.value(control_var)
        control_array = ensure_2d_array(control_vals, num_controls, polynomial_degrees[k])
        control_trajectories.append(control_array)

    return control_trajectories


def _extract_trajectories_for_single_phase(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    variables,
    raw_sol,
) -> None:
    """EXTRACTED: Extract trajectories for a single phase."""
    # INVERSION: Early return if phase not in variables
    if phase_id not in variables.phase_variables:
        return

    num_states, num_controls = problem.get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]
    polynomial_degrees = phase_def.collocation_points_per_interval
    phase_vars = variables.phase_variables[phase_id]

    # Extract state trajectories
    state_trajectories = _extract_state_matrices_for_phase(
        phase_vars, polynomial_degrees, raw_sol, num_states, phase_id
    )
    solution.phase_solved_state_trajectories_per_interval[phase_id] = state_trajectories

    # Extract control trajectories
    control_trajectories = _extract_control_variables_for_phase(
        phase_vars, polynomial_degrees, raw_sol, num_controls, phase_id
    )
    solution.phase_solved_control_trajectories_per_interval[phase_id] = control_trajectories


def _extract_multiphase_solution_trajectories(
    solution: OptimalControlSolution, problem: ProblemProtocol
) -> None:
    """NEVER-NESTER REFACTORED: Extract trajectories with early returns and extraction."""
    # INVERSION: Early returns for missing data
    if solution.raw_solution is None or solution.opti_object is None:
        raise ValueError("Missing raw solution or opti object")

    if not hasattr(solution.opti_object, "multiphase_variables_reference"):
        raise ValueError("Missing multiphase variables reference")

    opti = solution.opti_object
    raw_sol = solution.raw_solution
    variables = opti.multiphase_variables_reference

    # Initialize solution data structures
    solution.phase_solved_state_trajectories_per_interval = {}
    solution.phase_solved_control_trajectories_per_interval = {}

    # EXTRACTION: Process each phase in separate function
    for phase_id in problem.get_phase_ids():
        _extract_trajectories_for_single_phase(solution, problem, phase_id, variables, raw_sol)


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

    # STREAMLINED: Simple caching without complex patterns
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
                control_weights_cache[N_k] = compute_barycentric_weights(basis.collocation_nodes)

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

        # STREAMLINED: Direct tuple unpacking instead of complex bundle
        (
            success,
            fwd_tau_points,
            fwd_sim_traj,
            fwd_nlp_traj,
            bwd_tau_points,
            bwd_sim_traj,
            bwd_nlp_traj,
        ) = simulate_dynamics_for_phase_interval_error_estimation(
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

        # Calculate relative error using streamlined function
        error = calculate_relative_error_estimate(
            phase_id,
            k,
            success,
            fwd_sim_traj,
            fwd_nlp_traj,
            bwd_sim_traj,
            bwd_nlp_traj,
            gamma_factors,
        )
        errors[k] = error

    return errors


def _identify_refinement_intervals(
    polynomial_degrees: list[int], errors: list[float], adaptive_params: AdaptiveParameters
) -> tuple[set[int], set[int]]:
    """EXTRACTED: Identify intervals that need refinement vs reduction."""
    num_intervals = len(polynomial_degrees)
    intervals_needing_refinement = set()
    intervals_for_reduction = set()

    for k in range(num_intervals):
        error_k = errors[k]
        if np.isnan(error_k) or np.isinf(error_k) or error_k > adaptive_params.error_tolerance:
            intervals_needing_refinement.add(k)
        else:
            intervals_for_reduction.add(k)

    return intervals_needing_refinement, intervals_for_reduction


def _process_refinement_intervals(
    intervals_needing_refinement: set[int],
    polynomial_degrees: list[int],
    errors: list[float],
    adaptive_params: AdaptiveParameters,
) -> dict[int, tuple[str, int] | tuple[str, list[int]]]:
    """EXTRACTED: Process intervals that need refinement."""
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

    return refinement_actions


def _can_intervals_merge(
    k: int,
    intervals_for_reduction: set[int],
    num_intervals: int,
    state_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    control_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> bool:
    """EXTRACTED: Check if two intervals can be merged."""
    # INVERSION: Early returns for conditions that prevent merging
    if (k + 1) not in intervals_for_reduction:
        return False

    if (k + 1) >= num_intervals:
        return False

    state_eval_first = state_evaluators[k]
    state_eval_second = state_evaluators[k + 1]
    control_eval_first = control_evaluators[k]
    control_eval_second = control_evaluators[k + 1]

    if state_eval_first is None or state_eval_second is None:
        return False

    num_states, num_controls = problem.get_phase_variable_counts(phase_id)
    if num_controls > 0:
        if control_eval_first is None or control_eval_second is None:
            return False

    return True


def _create_merge_candidate(
    k: int, errors: list[float], polynomial_degrees: list[int], adaptive_params: AdaptiveParameters
):
    """EXTRACTED: Create a merge candidate."""
    from typing import NamedTuple

    class MergeCandidate(NamedTuple):
        first_idx: int
        second_idx: int
        overall_max_error: float
        merged_degree: int

    error_k = errors[k]
    error_k_plus_1 = errors[k + 1]
    overall_max_error = max(error_k, error_k_plus_1)

    merged_degree = max(polynomial_degrees[k], polynomial_degrees[k + 1])
    merged_degree = max(
        adaptive_params.min_polynomial_degree,
        min(adaptive_params.max_polynomial_degree, merged_degree),
    )

    return MergeCandidate(
        first_idx=k,
        second_idx=k + 1,
        overall_max_error=overall_max_error,
        merged_degree=merged_degree,
    )


def _identify_merge_candidates(
    intervals_for_reduction: set[int],
    polynomial_degrees: list[int],
    errors: list[float],
    adaptive_params: AdaptiveParameters,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    state_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    control_evaluators: list[Callable[[float | FloatArray], FloatArray] | None],
    gamma_factors: FloatArray,
):
    """EXTRACTED: Identify h-reduction merge candidates."""
    merge_candidates = []
    num_intervals = len(polynomial_degrees)

    for k in intervals_for_reduction:
        # Check if merge is possible
        if not _can_intervals_merge(
            k,
            intervals_for_reduction,
            num_intervals,
            state_evaluators,
            control_evaluators,
            problem,
            phase_id,
        ):
            continue

        # Test actual merge feasibility
        can_merge = h_reduce_intervals(
            phase_id,
            k,
            solution,
            problem,
            adaptive_params,
            gamma_factors,
            state_evaluators[k],
            control_evaluators[k],
            state_evaluators[k + 1],
            control_evaluators[k + 1],
        )

        if can_merge:
            candidate = _create_merge_candidate(k, errors, polynomial_degrees, adaptive_params)
            merge_candidates.append(candidate)

    return merge_candidates


def _select_approved_merges(merge_candidates):
    """EXTRACTED: Select merges without conflicts."""
    # Sort by overall maximum relative error (ascending order)
    merge_candidates.sort(key=lambda x: x.overall_max_error)

    merged_intervals = set()
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

    return approved_merges, merged_intervals


def _apply_p_reduction(
    intervals_for_reduction: set[int],
    merged_intervals: set[int],
    polynomial_degrees: list[int],
    errors: list[float],
    adaptive_params: AdaptiveParameters,
) -> dict[int, int]:
    """EXTRACTED: Apply p-reduction to remaining intervals."""
    reduction_actions = {}

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

    return reduction_actions


def _apply_p_refinement(
    k: int,
    action_data: int,
    mesh_points: FloatArray,
    next_polynomial_degrees: list[int],
    next_mesh_points: list[float],
) -> int:
    """EXTRACTED: Apply p-refinement to interval."""
    next_polynomial_degrees.append(action_data)
    next_mesh_points.append(mesh_points[k + 1])
    return k + 1


def _apply_h_refinement(
    k: int,
    action_data: list[int],
    mesh_points: FloatArray,
    next_polynomial_degrees: list[int],
    next_mesh_points: list[float],
) -> int:
    """EXTRACTED: Apply h-refinement to interval."""
    next_polynomial_degrees.extend(action_data)
    # Create new mesh nodes for subintervals
    tau_start = mesh_points[k]
    tau_end = mesh_points[k + 1]
    num_subintervals = len(action_data)
    new_nodes = np.linspace(tau_start, tau_end, num_subintervals + 1, dtype=np.float64)
    next_mesh_points.extend(list(new_nodes[1:]))
    return k + 1


def _apply_h_reduction_merge(
    k: int,
    approved_merges,
    mesh_points: FloatArray,
    next_polynomial_degrees: list[int],
    next_mesh_points: list[float],
) -> int:
    """EXTRACTED: Apply h-reduction merge."""
    merge = next(merge for merge in approved_merges if merge.first_idx == k)
    next_polynomial_degrees.append(merge.merged_degree)
    next_mesh_points.append(mesh_points[k + 2])  # Skip shared mesh point
    return k + 2  # Skip both merged intervals


def _apply_standard_processing(
    k: int,
    reduction_actions: dict[int, int],
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    next_polynomial_degrees: list[int],
    next_mesh_points: list[float],
) -> int:
    """EXTRACTED: Apply standard processing (p-reduction or keep unchanged)."""
    if k in reduction_actions:
        next_polynomial_degrees.append(reduction_actions[k])
    else:
        next_polynomial_degrees.append(polynomial_degrees[k])
    next_mesh_points.append(mesh_points[k + 1])
    return k + 1


def _build_new_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    refinement_actions: dict[int, tuple[str, int] | tuple[str, list[int]]],
    approved_merges,
    reduction_actions: dict[int, int],
) -> tuple[list[int], FloatArray]:
    """EXTRACTED: Build new mesh configuration by applying all actions."""
    next_polynomial_degrees: list[int] = []
    next_mesh_points = [mesh_points[0]]
    num_intervals = len(polynomial_degrees)

    k = 0
    while k < num_intervals:
        if k in refinement_actions:
            # Apply refinement
            action_type, action_data = refinement_actions[k]
            if action_type == "p":
                k = _apply_p_refinement(
                    k, action_data, mesh_points, next_polynomial_degrees, next_mesh_points
                )
            else:  # h-refinement
                k = _apply_h_refinement(
                    k, action_data, mesh_points, next_polynomial_degrees, next_mesh_points
                )

        elif any(merge.first_idx == k for merge in approved_merges):
            # Apply h-reduction merge
            k = _apply_h_reduction_merge(
                k, approved_merges, mesh_points, next_polynomial_degrees, next_mesh_points
            )

        else:
            # Apply p-reduction or keep unchanged
            k = _apply_standard_processing(
                k,
                reduction_actions,
                polynomial_degrees,
                mesh_points,
                next_polynomial_degrees,
                next_mesh_points,
            )

    return next_polynomial_degrees, np.array(next_mesh_points, dtype=np.float64)


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
    """NEVER-NESTER REFACTORED: Refine mesh using extraction pattern."""

    # Step 1: Identify intervals needing refinement vs reduction
    intervals_needing_refinement, intervals_for_reduction = _identify_refinement_intervals(
        polynomial_degrees, errors, adaptive_params
    )

    # Step 2: Process refinement intervals (p-refine or h-refine)
    refinement_actions = _process_refinement_intervals(
        intervals_needing_refinement, polynomial_degrees, errors, adaptive_params
    )

    # Step 3: Identify h-reduction candidates
    merge_candidates = _identify_merge_candidates(
        intervals_for_reduction,
        polynomial_degrees,
        errors,
        adaptive_params,
        solution,
        problem,
        phase_id,
        state_evaluators,
        control_evaluators,
        gamma_factors,
    )

    # Step 4: Process merges in error-based order, checking for conflicts
    approved_merges, merged_intervals = _select_approved_merges(merge_candidates)

    # Step 5: Apply p-reduction to remaining intervals
    reduction_actions = _apply_p_reduction(
        intervals_for_reduction, merged_intervals, polynomial_degrees, errors, adaptive_params
    )

    # Step 6: Build new mesh configuration
    return _build_new_mesh_configuration(
        polynomial_degrees, mesh_points, refinement_actions, approved_merges, reduction_actions
    )


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
    STREAMLINED multiphase PHS-Adaptive mesh refinement algorithm implementation.
    Now stores adaptive algorithm data in solution.
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

    # Import unified multiphase solver
    from trajectolab.direct_solver import solve_multiphase_radau_collocation

    # Variables to store final adaptive data
    final_phase_errors: dict[PhaseID, list[float]] = {}
    final_gamma_factors: dict[PhaseID, FloatArray | None] = {}

    # Main adaptive refinement loop
    for iteration in range(max_iterations):
        adaptive_state.iteration = iteration

        logger.info("Multiphase adaptive iteration %d/%d", iteration + 1, max_iterations)

        # Configure all phase meshes in the unified problem
        adaptive_state.configure_problem_meshes(problem)

        # Handle initial guess
        if iteration == 0:
            _handle_first_iteration_initial_guess(problem, initial_guess)
        else:
            if adaptive_state.most_recent_unified_solution is None:
                raise ValueError("No previous unified solution available for propagation")

            propagated_guess = propagate_multiphase_solution_to_new_meshes(
                adaptive_state.most_recent_unified_solution,
                problem,
                adaptive_state.phase_polynomial_degrees,
                adaptive_state.phase_mesh_points,
            )
            problem.initial_guess = propagated_guess

        # Solve unified multiphase NLP
        solution = solve_multiphase_radau_collocation(problem)

        if not solution.success:
            logger.warning(
                "Unified NLP solve failed in iteration %d: %s", iteration + 1, solution.message
            )

            if adaptive_state.most_recent_unified_solution is not None:
                adaptive_state.most_recent_unified_solution.message = f"Adaptive stopped due to solver failure in iteration {iteration + 1}: {solution.message}"
                adaptive_state.most_recent_unified_solution.success = False
                return cast(OptimalControlSolution, adaptive_state.most_recent_unified_solution)
            else:
                solution.message = (
                    f"Multiphase adaptive failed in first iteration: {solution.message}"
                )
                return solution

        # Store the mesh information that was ACTUALLY used for this solve
        solution.phase_mesh_intervals = {}
        solution.phase_mesh_nodes = {}

        for phase_id in problem.get_phase_ids():
            phase_def = problem._phases[phase_id]
            solution.phase_mesh_intervals[phase_id] = list(
                phase_def.collocation_points_per_interval
            )
            solution.phase_mesh_nodes[phase_id] = phase_def.global_normalized_mesh_nodes.copy()

        # Extract trajectories from unified solution for all phases
        _extract_multiphase_solution_trajectories(solution, problem)

        # Update most recent successful solution
        adaptive_state.most_recent_unified_solution = solution

        # Process each phase for convergence and refinement
        any_phase_needs_refinement = False

        for phase_id in phase_ids:
            # Calculate gamma normalization factors for this phase
            gamma_factors = calculate_gamma_normalizers_for_phase(solution, problem, phase_id)
            num_states, _ = problem.get_phase_variable_counts(phase_id)
            if gamma_factors is None and num_states > 0:
                solution.message = f"Failed to calculate gamma normalizers for phase {phase_id} at iteration {iteration + 1}"
                solution.success = False
                return solution

            safe_gamma = (
                gamma_factors
                if gamma_factors is not None
                else np.ones((num_states, 1), dtype=np.float64)
            )

            # Store gamma factors for final iteration
            final_gamma_factors[phase_id] = gamma_factors

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

            # Store phase errors for final iteration
            final_phase_errors[phase_id] = phase_errors.copy()

            # Check phase convergence
            phase_converged = all(
                not (np.isnan(error) or np.isinf(error))
                and error <= adaptive_params.error_tolerance
                for error in phase_errors
            )

            adaptive_state.phase_converged[phase_id] = phase_converged

            if not phase_converged:
                any_phase_needs_refinement = True

                # Refine mesh for this phase - use the CURRENT mesh configuration
                current_degrees = solution.phase_mesh_intervals[phase_id]
                current_mesh_points = solution.phase_mesh_nodes[phase_id]

                (
                    adaptive_state.phase_polynomial_degrees[phase_id],
                    adaptive_state.phase_mesh_points[phase_id],
                ) = _refine_phase_mesh(
                    phase_id,
                    current_degrees,
                    current_mesh_points,
                    phase_errors,
                    adaptive_params,
                    solution,
                    problem,
                    state_evaluators,
                    control_evaluators,
                    safe_gamma,
                )

        # Check global convergence
        if not any_phase_needs_refinement:
            logger.info("Multiphase adaptive refinement converged in %d iterations", iteration + 1)

            # Store adaptive data in solution
            solution.adaptive_data = AdaptiveAlgorithmData(
                target_tolerance=adaptive_params.error_tolerance,
                total_iterations=iteration + 1,
                converged=True,
                phase_converged=adaptive_state.phase_converged.copy(),
                final_phase_error_estimates=final_phase_errors,
                phase_gamma_factors=final_gamma_factors,
            )

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
        # Store adaptive data for non-converged solution
        adaptive_state.most_recent_unified_solution.adaptive_data = AdaptiveAlgorithmData(
            target_tolerance=adaptive_params.error_tolerance,
            total_iterations=max_iterations,
            converged=False,
            phase_converged=adaptive_state.phase_converged.copy(),
            final_phase_error_estimates=final_phase_errors,
            phase_gamma_factors=final_gamma_factors,
        )

        max_iter_msg = (
            f"Reached maximum iterations ({max_iterations}) without full convergence "
            f"to tolerance {adaptive_params.error_tolerance:.1e}"
        )
        adaptive_state.most_recent_unified_solution.message = max_iter_msg
        return cast(OptimalControlSolution, adaptive_state.most_recent_unified_solution)
    else:
        # Create failure solution
        failed_solution = OptimalControlSolution()
        failed_solution.success = False
        failed_solution.message = (
            f"No successful unified solution obtained in {max_iterations} iterations"
        )
        return failed_solution


def _handle_first_iteration_initial_guess(
    problem: ProblemProtocol,
    initial_guess: MultiPhaseInitialGuess | None,
) -> None:
    """Handle initial guess for the first iteration of multiphase adaptive."""
    if problem.initial_guess is not None:
        # STREAMLINED: Basic validation only
        pass
    elif initial_guess is not None:
        problem.initial_guess = initial_guess
    else:
        problem.initial_guess = None
