import logging
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from trajectolab.adaptive.phs.data_structures import (
    AdaptiveParameters,
    MultiphaseAdaptiveState,
    _ensure_2d_array,
)
from trajectolab.adaptive.phs.error_estimation import (
    _calculate_gamma_normalizers_for_phase,
    _calculate_relative_error_estimate,
    _simulate_dynamics_for_phase_interval_error_estimation,
)
from trajectolab.adaptive.phs.initial_guess import (
    _propagate_multiphase_solution_to_new_meshes,
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
    _compute_barycentric_weights,
    _compute_radau_collocation_components,
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
    """Extract state trajectory matrices for a phase."""
    state_trajectories = []
    for k, state_matrix in enumerate(phase_vars.state_matrices):
        state_vals = raw_sol.value(state_matrix)
        state_array = _ensure_2d_array(state_vals, num_states, polynomial_degrees[k] + 1)
        state_trajectories.append(state_array)

    return state_trajectories


def _extract_control_variables_for_phase(
    phase_vars, polynomial_degrees: list[int], raw_sol, num_controls: int, phase_id: PhaseID
) -> list[FloatArray]:
    """Extract control variable trajectories for a phase."""
    if num_controls == 0:
        return [
            np.empty((0, polynomial_degrees[k]), dtype=np.float64)
            for k in range(len(polynomial_degrees))
        ]

    control_trajectories = []
    for k, control_var in enumerate(phase_vars.control_variables):
        control_vals = raw_sol.value(control_var)
        control_array = _ensure_2d_array(control_vals, num_controls, polynomial_degrees[k])
        control_trajectories.append(control_array)

    return control_trajectories


def _validate_solution_for_extraction(solution: OptimalControlSolution) -> None:
    """Validate solution has required components for trajectory extraction."""
    if solution.raw_solution is None:
        raise ValueError("Raw solution is None")
    if solution.opti_object is None:
        raise ValueError("Opti object is None")
    if not hasattr(solution.opti_object, "multiphase_variables_reference"):
        raise ValueError("Missing multiphase variables reference")


def _extract_multiphase_solution_trajectories(
    solution: OptimalControlSolution, problem: ProblemProtocol
) -> None:
    """Extract trajectory data from solution for all phases."""
    _validate_solution_for_extraction(solution)

    # Assert non-None after validation for type narrowing
    assert solution.opti_object is not None
    assert solution.raw_solution is not None

    opti = solution.opti_object
    raw_sol = solution.raw_solution
    variables = opti.multiphase_variables_reference  # Now type-safe

    solution.phase_solved_state_trajectories_per_interval = {}
    solution.phase_solved_control_trajectories_per_interval = {}

    for phase_id in problem._get_phase_ids():
        if phase_id not in variables.phase_variables:
            continue

        num_states, num_controls = problem._get_phase_variable_counts(phase_id)
        phase_def = problem._phases[phase_id]
        polynomial_degrees = phase_def.collocation_points_per_interval
        phase_vars = variables.phase_variables[phase_id]

        state_trajectories = _extract_state_matrices_for_phase(
            phase_vars, polynomial_degrees, raw_sol, num_states, phase_id
        )
        solution.phase_solved_state_trajectories_per_interval[phase_id] = state_trajectories

        control_trajectories = _extract_control_variables_for_phase(
            phase_vars, polynomial_degrees, raw_sol, num_controls, phase_id
        )
        solution.phase_solved_control_trajectories_per_interval[phase_id] = control_trajectories


def _validate_interpolant_creation_inputs(
    solution: OptimalControlSolution, phase_id: PhaseID
) -> None:
    """Validate inputs for interpolant creation."""
    if phase_id not in solution.phase_solved_state_trajectories_per_interval:
        raise ValueError(f"Missing state trajectories for phase {phase_id}")


def _get_or_create_basis_components(
    basis_cache: dict[int, RadauBasisComponents], N_k: int
) -> RadauBasisComponents:
    """Get basis components from cache or create new ones."""
    if N_k not in basis_cache:
        basis_cache[N_k] = _compute_radau_collocation_components(N_k)
    return basis_cache[N_k]


def _get_or_create_control_weights(
    control_weights_cache: dict[int, FloatArray], basis: RadauBasisComponents, N_k: int
) -> FloatArray:
    """Get control barycentric weights from cache or create new ones."""
    if N_k not in control_weights_cache:
        control_weights_cache[N_k] = _compute_barycentric_weights(basis.collocation_nodes)
    return control_weights_cache[N_k]


def _create_state_interpolant(
    basis: RadauBasisComponents, state_data: FloatArray
) -> PolynomialInterpolant:
    """Create state interpolant using basis components."""
    return PolynomialInterpolant(
        basis.state_approximation_nodes,
        state_data,
        basis.barycentric_weights_for_state_nodes,
    )


def _create_control_interpolant(
    basis: RadauBasisComponents,
    control_data: FloatArray,
    control_weights: FloatArray,
    num_controls: int,
) -> PolynomialInterpolant:
    """Create control interpolant or empty interpolant if no controls."""
    if num_controls > 0 and control_data.size > 0:
        return PolynomialInterpolant(basis.collocation_nodes, control_data, control_weights)

    # Empty control interpolant for phases without controls
    return PolynomialInterpolant(
        np.array([-1.0, 1.0], dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),  # Use actual weights instead of None
    )


def _create_phase_interpolants(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> tuple[
    list[Callable[[float | FloatArray], FloatArray]],
    list[Callable[[float | FloatArray], FloatArray]],
]:
    """Create interpolants for continuous trajectory evaluation."""
    _validate_interpolant_creation_inputs(solution, phase_id)

    phase_def = problem._phases[phase_id]
    polynomial_degrees = phase_def.collocation_points_per_interval
    num_intervals = len(polynomial_degrees)
    num_states, num_controls = problem._get_phase_variable_counts(phase_id)

    # Caches prevent redundant computation across intervals
    basis_cache: dict[int, RadauBasisComponents] = {}
    control_weights_cache: dict[int, FloatArray] = {}

    state_evaluators: list[Callable[[float | FloatArray], FloatArray]] = []
    control_evaluators: list[Callable[[float | FloatArray], FloatArray]] = []

    states_list = solution.phase_solved_state_trajectories_per_interval[phase_id]
    controls_list = solution.phase_solved_control_trajectories_per_interval.get(phase_id)

    for k in range(num_intervals):
        N_k = polynomial_degrees[k]
        basis = _get_or_create_basis_components(basis_cache, N_k)

        # Create state interpolant
        state_data = states_list[k]
        state_evaluators.append(_create_state_interpolant(basis, state_data))

        # Create control interpolant with proper empty arrays instead of None
        if controls_list is not None:
            control_data = controls_list[k]
            control_weights = _get_or_create_control_weights(control_weights_cache, basis, N_k)
            control_evaluators.append(
                _create_control_interpolant(basis, control_data, control_weights, num_controls)
            )
        else:
            # Use proper empty arrays instead of None
            empty_control_data = np.empty((0, N_k), dtype=np.float64)
            empty_weights = np.array([], dtype=np.float64)
            control_evaluators.append(
                _create_control_interpolant(basis, empty_control_data, empty_weights, 0)
            )

    return state_evaluators, control_evaluators


def _estimate_phase_errors(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    state_evaluators: Sequence[Callable[[float | FloatArray], FloatArray] | None],
    control_evaluators: Sequence[Callable[[float | FloatArray], FloatArray] | None],
    adaptive_params: AdaptiveParameters,
    gamma_factors: FloatArray,
) -> list[float]:
    """Estimate errors for all intervals in a phase."""
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

        (
            success,
            fwd_tau_points,
            fwd_sim_traj,
            fwd_nlp_traj,
            bwd_tau_points,
            bwd_sim_traj,
            bwd_nlp_traj,
        ) = _simulate_dynamics_for_phase_interval_error_estimation(
            phase_id,
            k,
            solution,
            problem,
            state_eval,
            control_eval,
            adaptive_params._get_ode_solver(),
            n_eval_points=adaptive_params.num_error_sim_points,
        )

        error = _calculate_relative_error_estimate(
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


def _classify_intervals_by_error(
    polynomial_degrees: list[int], errors: list[float], adaptive_params: AdaptiveParameters
) -> tuple[set[int], set[int]]:
    """Classify intervals into refinement and reduction categories."""
    intervals_needing_refinement = set()
    intervals_for_reduction = set()

    for k, error_k in enumerate(errors):
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
    """Process intervals needing refinement with p/h-refinement strategy."""
    refinement_actions: dict[int, tuple[str, int] | tuple[str, list[int]]] = {}

    for k in intervals_needing_refinement:
        error_k = errors[k]
        N_k = polynomial_degrees[k]

        p_result = p_refine_interval(
            error_k, N_k, adaptive_params.error_tolerance, adaptive_params.max_polynomial_degree
        )

        if p_result.was_p_successful:
            refinement_actions[k] = ("p", p_result.actual_Nk_to_use)
        else:
            # H-refinement when polynomial degree limits reached
            h_result = h_refine_params(
                p_result.unconstrained_target_Nk, adaptive_params.min_polynomial_degree
            )
            refinement_actions[k] = ("h", h_result.collocation_nodes_for_new_subintervals)

    return refinement_actions


class MergeCandidate:
    """Container for merge candidate information."""

    def __init__(
        self, first_idx: int, second_idx: int, overall_max_error: float, merged_degree: int
    ):
        self.first_idx = first_idx
        self.second_idx = second_idx
        self.overall_max_error = overall_max_error
        self.merged_degree = merged_degree


def _create_merge_candidate(
    k: int, errors: list[float], polynomial_degrees: list[int], adaptive_params: AdaptiveParameters
) -> MergeCandidate:
    """Create merge candidate for adjacent intervals."""
    error_k = errors[k]
    error_k_plus_1 = errors[k + 1]
    overall_max_error = max(error_k, error_k_plus_1)

    # Conservative merged degree maintains solution quality
    merged_degree = max(polynomial_degrees[k], polynomial_degrees[k + 1])
    merged_degree = max(
        adaptive_params.min_polynomial_degree,
        min(adaptive_params.max_polynomial_degree, merged_degree),
    )

    return MergeCandidate(k, k + 1, overall_max_error, merged_degree)


def _check_merge_feasibility(
    k: int,
    intervals_for_reduction: set[int],
    num_intervals: int,
    state_evaluators: list[Callable[[float | FloatArray], FloatArray]],
    control_evaluators: list[Callable[[float | FloatArray], FloatArray]],
    problem: ProblemProtocol,
    phase_id: PhaseID,
    solution: OptimalControlSolution,
    adaptive_params: AdaptiveParameters,
    gamma_factors: FloatArray,
) -> bool:
    """Check if intervals k and k+1 can be merged."""
    # Structural checks
    if (k + 1) not in intervals_for_reduction or (k + 1) >= num_intervals:
        return False

    # Bounds checks for evaluator lists
    if k >= len(state_evaluators) or (k + 1) >= len(state_evaluators):
        return False
    if k >= len(control_evaluators) or (k + 1) >= len(control_evaluators):
        return False

    state_eval_k = state_evaluators[k]
    state_eval_k_plus_1 = state_evaluators[k + 1]
    control_eval_k = control_evaluators[k]
    control_eval_k_plus_1 = control_evaluators[k + 1]

    # Mathematical feasibility via simulation
    return h_reduce_intervals(
        phase_id,
        k,
        solution,
        problem,
        adaptive_params,
        gamma_factors,
        state_eval_k,
        control_eval_k,
        state_eval_k_plus_1,
        control_eval_k_plus_1,
    )


def _identify_merge_candidates(
    intervals_for_reduction: set[int],
    polynomial_degrees: list[int],
    errors: list[float],
    adaptive_params: AdaptiveParameters,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    state_evaluators: list[Callable],
    control_evaluators: list[Callable],
    gamma_factors: FloatArray,
) -> list[MergeCandidate]:
    """Identify feasible merge candidates."""
    merge_candidates = []

    for k in intervals_for_reduction:
        if not _check_merge_feasibility(
            k,
            intervals_for_reduction,
            len(polynomial_degrees),
            state_evaluators,
            control_evaluators,
            problem,
            phase_id,
            solution,
            adaptive_params,
            gamma_factors,
        ):
            continue

        candidate = _create_merge_candidate(k, errors, polynomial_degrees, adaptive_params)
        merge_candidates.append(candidate)

    return merge_candidates


def _select_approved_merges(
    merge_candidates: list[MergeCandidate],
) -> tuple[list[MergeCandidate], set[int]]:
    """Select non-conflicting merges ordered by error magnitude."""
    merge_candidates.sort(key=lambda x: x.overall_max_error)

    merged_intervals = set()
    approved_merges = []

    for candidate in merge_candidates:
        # Conflict avoidance ensures each interval participates in at most one merge
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
    """Apply p-reduction to unmerged intervals."""
    reduction_actions = {}

    for k in intervals_for_reduction:
        if k not in merged_intervals:
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
    """Apply p-refinement to interval k."""
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
    """Apply h-refinement to interval k."""
    next_polynomial_degrees.extend(action_data)
    # Uniform subdivision of original interval for new subintervals
    tau_start = mesh_points[k]
    tau_end = mesh_points[k + 1]
    num_subintervals = len(action_data)
    new_nodes = np.linspace(tau_start, tau_end, num_subintervals + 1, dtype=np.float64)
    next_mesh_points.extend(list(new_nodes[1:]))
    return k + 1


def _apply_h_reduction_merge(
    k: int,
    approved_merges: list[MergeCandidate],
    mesh_points: FloatArray,
    next_polynomial_degrees: list[int],
    next_mesh_points: list[float],
) -> int:
    """Apply h-reduction merge to intervals k and k+1."""
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
    """Apply standard interval processing (no refinement/merge)."""
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
    approved_merges: list[MergeCandidate],
    reduction_actions: dict[int, int],
) -> tuple[list[int], FloatArray]:
    """Build new mesh configuration by processing all intervals sequentially."""
    next_polynomial_degrees: list[int] = []
    next_mesh_points = [mesh_points[0]]
    num_intervals = len(polynomial_degrees)

    k = 0
    while k < num_intervals:
        if k in refinement_actions:
            _, action_data = refinement_actions[k]
            if isinstance(action_data, int):
                k = _apply_p_refinement(
                    k, action_data, mesh_points, next_polynomial_degrees, next_mesh_points
                )
            elif isinstance(action_data, list):
                k = _apply_h_refinement(
                    k, action_data, mesh_points, next_polynomial_degrees, next_mesh_points
                )
            else:
                raise ValueError(f"Unexpected action_data type: {type(action_data)}")
            continue

        if any(merge.first_idx == k for merge in approved_merges):
            k = _apply_h_reduction_merge(
                k, approved_merges, mesh_points, next_polynomial_degrees, next_mesh_points
            )
            continue

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
    state_evaluators: list[Callable[[float | FloatArray], FloatArray]],
    control_evaluators: list[Callable[[float | FloatArray], FloatArray]],
    gamma_factors: FloatArray,
) -> tuple[list[int], FloatArray]:
    """Comprehensive mesh adaptation balancing accuracy and computational efficiency."""
    intervals_needing_refinement, intervals_for_reduction = _classify_intervals_by_error(
        polynomial_degrees, errors, adaptive_params
    )

    refinement_actions = _process_refinement_intervals(
        intervals_needing_refinement, polynomial_degrees, errors, adaptive_params
    )

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

    approved_merges, merged_intervals = _select_approved_merges(merge_candidates)

    reduction_actions = _apply_p_reduction(
        intervals_for_reduction, merged_intervals, polynomial_degrees, errors, adaptive_params
    )

    return _build_new_mesh_configuration(
        polynomial_degrees, mesh_points, refinement_actions, approved_merges, reduction_actions
    )


def _initialize_adaptive_state(
    problem: ProblemProtocol, adaptive_params: AdaptiveParameters
) -> MultiphaseAdaptiveState:
    """Initialize adaptive state from problem configuration."""
    phase_ids = problem._get_phase_ids()
    adaptive_state = MultiphaseAdaptiveState(
        phase_polynomial_degrees={},
        phase_mesh_points={},
        phase_converged=dict.fromkeys(phase_ids, False),
        iteration=0,
    )

    # Initial mesh configuration from problem definition
    for phase_id in phase_ids:
        phase_def = problem._phases[phase_id]
        if not phase_def.mesh_configured:
            raise ValueError(f"Phase {phase_id} mesh must be configured before adaptive solving")

        adaptive_state.phase_polynomial_degrees[phase_id] = list(
            phase_def.collocation_points_per_interval
        )
        adaptive_state.phase_mesh_points[phase_id] = phase_def.global_normalized_mesh_nodes.copy()

    return adaptive_state


def _configure_initial_guess(
    iteration: int,
    problem: ProblemProtocol,
    initial_guess: MultiPhaseInitialGuess | None,
    adaptive_state: MultiphaseAdaptiveState,
) -> None:
    """Configure initial guess for current iteration."""
    if iteration == 0:
        if problem.initial_guess is not None:
            pass
        elif initial_guess is not None:
            problem.initial_guess = initial_guess
        else:
            problem.initial_guess = None
    else:
        # Solution propagation maintains trajectory quality across mesh changes
        if adaptive_state.most_recent_unified_solution is None:
            raise ValueError("No previous unified solution available for propagation")

        propagated_guess = _propagate_multiphase_solution_to_new_meshes(
            adaptive_state.most_recent_unified_solution,
            problem,
            adaptive_state.phase_polynomial_degrees,
            adaptive_state.phase_mesh_points,
        )
        problem.initial_guess = propagated_guess


def _store_mesh_information(solution: OptimalControlSolution, problem: ProblemProtocol) -> None:
    """Store mesh information in solution for continuity."""
    solution.phase_mesh_intervals = {}
    solution.phase_mesh_nodes = {}

    for phase_id in problem._get_phase_ids():
        phase_def = problem._phases[phase_id]
        solution.phase_mesh_intervals[phase_id] = list(phase_def.collocation_points_per_interval)
        solution.phase_mesh_nodes[phase_id] = phase_def.global_normalized_mesh_nodes.copy()


def _handle_solver_failure(
    solution: OptimalControlSolution,
    iteration: int,
    adaptive_state: MultiphaseAdaptiveState,
) -> OptimalControlSolution:
    """Handle solver failure during adaptive iteration."""
    logger.warning("Unified NLP solve failed in iteration %d: %s", iteration + 1, solution.message)

    if adaptive_state.most_recent_unified_solution is not None:
        adaptive_state.most_recent_unified_solution.message = f"Adaptive stopped due to solver failure in iteration {iteration + 1}: {solution.message}"
        adaptive_state.most_recent_unified_solution.success = False
        return cast(OptimalControlSolution, adaptive_state.most_recent_unified_solution)
    else:
        solution.message = f"Multiphase adaptive failed in first iteration: {solution.message}"
        return solution


def _process_single_phase_convergence(
    phase_id: PhaseID,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    adaptive_params: AdaptiveParameters,
    adaptive_state: MultiphaseAdaptiveState,
    final_phase_errors: dict[PhaseID, list[float]],
    final_gamma_factors: dict[PhaseID, FloatArray | None],
) -> bool:
    """Process convergence for a single phase. Returns True if phase needs refinement."""
    # Gamma normalization prevents ill-conditioning in error calculations
    gamma_factors = _calculate_gamma_normalizers_for_phase(solution, problem, phase_id)
    num_states, _ = problem._get_phase_variable_counts(phase_id)

    if gamma_factors is None and num_states > 0:
        solution.message = f"Failed to calculate gamma normalizers for phase {phase_id}"
        solution.success = False
        raise ValueError(solution.message)

    safe_gamma = (
        gamma_factors if gamma_factors is not None else np.ones((num_states, 1), dtype=np.float64)
    )

    final_gamma_factors[phase_id] = gamma_factors

    state_evaluators, control_evaluators = _create_phase_interpolants(solution, problem, phase_id)

    phase_errors = _estimate_phase_errors(
        solution,
        problem,
        phase_id,
        state_evaluators,
        control_evaluators,
        adaptive_params,
        safe_gamma,
    )

    final_phase_errors[phase_id] = phase_errors.copy()

    # Phase convergence based on tolerance satisfaction
    phase_converged = all(
        not (np.isnan(error) or np.isinf(error)) and error <= adaptive_params.error_tolerance
        for error in phase_errors
    )

    adaptive_state.phase_converged[phase_id] = phase_converged

    if not phase_converged:
        # Mesh refinement based on current solution accuracy
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
        return True

    return False


def _check_convergence_across_phases(
    problem: ProblemProtocol,
    solution: OptimalControlSolution,
    adaptive_params: AdaptiveParameters,
    adaptive_state: MultiphaseAdaptiveState,
    final_phase_errors: dict[PhaseID, list[float]],
    final_gamma_factors: dict[PhaseID, FloatArray | None],
) -> bool:
    """Check convergence across all phases. Returns True if any phase needs refinement."""
    any_phase_needs_refinement = False

    for phase_id in problem._get_phase_ids():
        phase_needs_refinement = _process_single_phase_convergence(
            phase_id,
            solution,
            problem,
            adaptive_params,
            adaptive_state,
            final_phase_errors,
            final_gamma_factors,
        )
        if phase_needs_refinement:
            any_phase_needs_refinement = True

    return any_phase_needs_refinement


def _create_convergence_result(
    solution: OptimalControlSolution,
    iteration: int,
    adaptive_params: AdaptiveParameters,
    adaptive_state: MultiphaseAdaptiveState,
    final_phase_errors: dict[PhaseID, list[float]],
    final_gamma_factors: dict[PhaseID, FloatArray | None],
) -> OptimalControlSolution:
    """Create successful convergence result."""
    logger.info("Multiphase adaptive refinement converged in %d iterations", iteration + 1)

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


def _create_max_iterations_result(
    adaptive_params: AdaptiveParameters,
    adaptive_state: MultiphaseAdaptiveState,
    final_phase_errors: dict[PhaseID, list[float]],
    final_gamma_factors: dict[PhaseID, FloatArray | None],
) -> OptimalControlSolution:
    """Create result when maximum iterations reached."""
    logger.warning(
        "Multiphase adaptive refinement reached maximum iterations (%d) without convergence",
        adaptive_params.max_iterations,
    )

    if adaptive_state.most_recent_unified_solution is not None:
        adaptive_state.most_recent_unified_solution.adaptive_data = AdaptiveAlgorithmData(
            target_tolerance=adaptive_params.error_tolerance,
            total_iterations=adaptive_params.max_iterations,
            converged=False,
            phase_converged=adaptive_state.phase_converged.copy(),
            final_phase_error_estimates=final_phase_errors,
            phase_gamma_factors=final_gamma_factors,
        )

        max_iter_msg = (
            f"Reached maximum iterations ({adaptive_params.max_iterations}) without full convergence "
            f"to tolerance {adaptive_params.error_tolerance:.1e}"
        )
        adaptive_state.most_recent_unified_solution.message = max_iter_msg
        return cast(OptimalControlSolution, adaptive_state.most_recent_unified_solution)
    else:
        failed_solution = OptimalControlSolution()
        failed_solution.success = False
        failed_solution.message = f"No successful unified solution obtained in {adaptive_params.max_iterations} iterations"
        return failed_solution


def solve_multiphase_phs_adaptive_internal(
    problem: ProblemProtocol,
    error_tolerance: float,
    max_iterations: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
    ode_solver_tolerance: float,
    ode_method: str,
    ode_max_step: float | None,
    ode_atol_factor: float,
    ode_solver,
    num_error_sim_points: int,
    initial_guess: MultiPhaseInitialGuess | None = None,
) -> OptimalControlSolution:
    """Multiphase PHS-Adaptive mesh refinement algorithm implementation."""
    logger.info(
        "Starting multiphase adaptive mesh refinement: tolerance=%.1e, max_iter=%d",
        error_tolerance,
        max_iterations,
    )

    adaptive_params = AdaptiveParameters(
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        ode_solver_tolerance=ode_solver_tolerance,
        num_error_sim_points=num_error_sim_points,
        ode_method=ode_method,
        ode_max_step=ode_max_step,
        ode_atol_factor=ode_atol_factor,
        ode_solver=ode_solver,
    )

    adaptive_state = _initialize_adaptive_state(problem, adaptive_params)

    from trajectolab.direct_solver import _solve_multiphase_radau_collocation

    final_phase_errors: dict[PhaseID, list[float]] = {}
    final_gamma_factors: dict[PhaseID, FloatArray | None] = {}

    # Adaptive refinement loop balances accuracy and computational efficiency
    for iteration in range(max_iterations):
        adaptive_state.iteration = iteration
        logger.info("Multiphase adaptive iteration %d/%d", iteration + 1, max_iterations)

        adaptive_state._configure_problem_meshes(problem)
        _configure_initial_guess(iteration, problem, initial_guess, adaptive_state)

        solution = _solve_multiphase_radau_collocation(problem)

        if not solution.success:
            return _handle_solver_failure(solution, iteration, adaptive_state)

        _store_mesh_information(solution, problem)
        _extract_multiphase_solution_trajectories(solution, problem)
        adaptive_state.most_recent_unified_solution = solution

        # Check convergence across all phases
        any_phase_needs_refinement = _check_convergence_across_phases(
            problem,
            solution,
            adaptive_params,
            adaptive_state,
            final_phase_errors,
            final_gamma_factors,
        )

        # Early termination on convergence
        if not any_phase_needs_refinement:
            return _create_convergence_result(
                solution,
                iteration,
                adaptive_params,
                adaptive_state,
                final_phase_errors,
                final_gamma_factors,
            )

    # Maximum iterations reached
    return _create_max_iterations_result(
        adaptive_params, adaptive_state, final_phase_errors, final_gamma_factors
    )
