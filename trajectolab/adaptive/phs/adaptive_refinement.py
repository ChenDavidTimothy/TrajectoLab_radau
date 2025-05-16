"""Mesh refinement decision algorithms for adaptive mesh refinement."""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from trajectolab.adaptive.phs.adaptive_error import DEFAULT_STATE_CLIP_VALUE
from trajectolab.adaptive.phs.adaptive_interpolation import (
    PolynomialInterpolant,
    extract_and_prepare_array,
)
from trajectolab.trajectolab_types import (
    AdaptiveParameters,
    CasADiDM,
    CasADiOpti,
    NumStates,
    OptimalControlProblem,
    OptimalControlSolution,
    _FloatArray,
    _Matrix,
    _NormalizedTimePoint,
    _Vector,
)

logger = logging.getLogger(__name__)


@dataclass
class PRefineResult:
    """Result of polynomial degree refinement decision."""

    actual_Nk_to_use: int
    was_p_successful: bool
    unconstrained_target_Nk: int


@dataclass
class HRefineResult:
    """Result of mesh interval refinement decision."""

    collocation_nodes_for_new_subintervals: List[int]
    num_new_subintervals: int


@dataclass
class PReduceResult:
    """Result of polynomial degree reduction decision."""

    new_num_collocation_nodes: int
    was_reduction_applied: bool


def p_refine_interval(
    current_Nk: int,
    interval_error: float,
    error_tol: float,
    N_max_degree: int,
) -> PRefineResult:
    """Decide polynomial degree refinement for an interval.

    Args:
        current_Nk: Current polynomial degree
        interval_error: Estimated error for the interval
        error_tol: Error tolerance threshold
        N_max_degree: Maximum allowed polynomial degree

    Returns:
        Decision result with new polynomial degree
    """
    if interval_error <= error_tol:  # Error already acceptable
        return PRefineResult(
            actual_Nk_to_use=current_Nk,
            was_p_successful=False,
            unconstrained_target_Nk=current_Nk,
        )

    # Heuristic for nodes to add
    if np.isinf(interval_error):  # Max out refinement if error is infinite
        nodes_to_add = max(1, N_max_degree - current_Nk)
    elif interval_error > 0:  # Avoid log of zero or negative
        ratio = interval_error / error_tol
        nodes_to_add = max(1, int(np.ceil(np.log10(ratio))))
    else:  # interval_error is zero or very small negative
        nodes_to_add = 0  # No refinement needed

    if nodes_to_add == 0 and interval_error > error_tol:
        nodes_to_add = 1  # ensure at least one node is added if error is above tolerance

    target_Nk = current_Nk + nodes_to_add

    if target_Nk >= N_max_degree:
        return PRefineResult(
            actual_Nk_to_use=N_max_degree,
            was_p_successful=(N_max_degree > current_Nk),
            unconstrained_target_Nk=target_Nk,
        )

    return PRefineResult(
        actual_Nk_to_use=target_Nk,
        was_p_successful=True,
        unconstrained_target_Nk=target_Nk,
    )


def h_refine_params(unconstrained_target_Nk: int, N_min_degree: int) -> HRefineResult:
    """Determine h-refinement parameters.

    Args:
        unconstrained_target_Nk: Target polynomial degree if unconstrained
        N_min_degree: Minimum allowed polynomial degree

    Returns:
        Decision result with collocation nodes for new subintervals
    """
    # Determine number of subintervals needed if each has N_min_degree
    num_subintervals = max(2, int(np.ceil(unconstrained_target_Nk / max(1, N_min_degree))))
    nodes_per_subinterval: List[int] = [N_min_degree] * num_subintervals

    return HRefineResult(
        collocation_nodes_for_new_subintervals=nodes_per_subinterval,
        num_new_subintervals=num_subintervals,
    )


def p_reduce_interval(
    current_Nk: int,
    interval_error: float,
    error_tol: float,
    N_min_degree: int,
    N_max_degree: int,
) -> PReduceResult:
    """Decide polynomial degree reduction for an interval.

    Args:
        current_Nk: Current polynomial degree
        interval_error: Estimated error for the interval
        error_tol: Error tolerance threshold
        N_min_degree: Minimum allowed polynomial degree
        N_max_degree: Maximum allowed polynomial degree

    Returns:
        Decision result with new polynomial degree
    """
    if interval_error > error_tol or current_Nk <= N_min_degree:
        return PReduceResult(new_num_collocation_nodes=current_Nk, was_reduction_applied=False)

    # Heuristic for nodes to remove
    nodes_to_remove_float = 0.0
    if interval_error < 1e-16:  # Essentially zero error
        nodes_to_remove_float = float(current_Nk - N_min_degree)
    elif interval_error > 0:  # interval_error is > 0 and <= error_tol
        ratio = error_tol / interval_error  # ratio >= 1.0
        if ratio > 1.0:  # Ensure log argument is > 0
            nodes_to_remove_float = np.floor(np.log10(ratio))

    nodes_to_remove = max(0, int(nodes_to_remove_float))
    new_Nk = max(N_min_degree, current_Nk - nodes_to_remove)
    was_reduced = new_Nk < current_Nk

    return PReduceResult(new_num_collocation_nodes=new_Nk, was_reduction_applied=was_reduced)


def h_reduce_intervals(
    first_interval_idx: int,
    solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    adaptive_params: "AdaptiveParameters",  # Forward reference
    gamma_factors: _Vector,
    state_evaluator_first_interval: PolynomialInterpolant,
    control_evaluator_first_interval: PolynomialInterpolant,
    state_evaluator_second_interval: PolynomialInterpolant,
    control_evaluator_second_interval: PolynomialInterpolant,
) -> bool:
    """Decide if two adjacent intervals can be merged.

    Args:
        first_interval_idx: Index of the first interval
        solution: Optimization solution
        problem: Optimal control problem definition
        adaptive_params: Adaptation parameters
        gamma_factors: Scaling factors for error normalization
        state_evaluator_first_interval: State interpolant for first interval
        control_evaluator_first_interval: Control interpolant for first interval
        state_evaluator_second_interval: State interpolant for second interval
        control_evaluator_second_interval: Control interpolant for second interval

    Returns:
        True if intervals can be merged, False otherwise
    """
    from scipy.integrate import solve_ivp

    from trajectolab.adaptive.phs.adaptive_coordinates import (
        map_local_interval_tau_to_global_normalized_tau,
    )

    logger.info(
        f"    h-reduction check for intervals {first_interval_idx} and {first_interval_idx+1}."
    )
    error_tol: float = adaptive_params.error_tolerance
    ode_rtol: float = adaptive_params.ode_solver_tolerance
    ode_atol: float = ode_rtol * 1e-1
    num_sim_points: int = adaptive_params.num_error_sim_points

    num_states: NumStates = problem.num_states
    dynamics_function = problem.dynamics_function
    problem_parameters = problem.problem_parameters

    if solution.raw_solution is None:
        logger.warning("      h-reduction failed: Raw solution missing.")
        return False
    if solution.global_normalized_mesh_nodes is None:
        logger.warning("      h-reduction failed: Global mesh nodes missing from solution.")
        return False

    global_mesh: _FloatArray = np.array(solution.global_normalized_mesh_nodes, dtype=np.float64)

    if first_interval_idx + 2 >= len(global_mesh):
        logger.warning(
            f"      h-reduction failed: Not enough mesh points for intervals {first_interval_idx}, {first_interval_idx+1}."
        )
        return False

    # Mesh points for the two intervals being considered for merge
    tau_start_k: _NormalizedTimePoint = global_mesh[first_interval_idx]
    tau_shared_between_k_kp1: _NormalizedTimePoint = global_mesh[first_interval_idx + 1]
    tau_end_kp1: _NormalizedTimePoint = global_mesh[first_interval_idx + 2]

    # Half-lengths of the original intervals in global tau
    beta_k = (tau_shared_between_k_kp1 - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared_between_k_kp1) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        logger.info("      h-reduction check: One of the original intervals has zero length.")
        return False

    t0_sol = solution.initial_time_variable
    tf_sol = solution.terminal_time_variable
    if t0_sol is None or tf_sol is None:
        logger.warning("      h-reduction failed: Solution time variables are None.")
        return False
    t0: float = float(t0_sol)
    tf: float = float(tf_sol)

    # Global time scaling for the entire OCP
    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    # Overall scaling factors for dynamics in each original interval's local tau
    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    def _get_control_value_for_h_reduction(
        control_evaluator: PolynomialInterpolant, local_tau: _NormalizedTimePoint
    ) -> _Vector:
        if control_evaluator.num_vars == 0:
            return np.array([], dtype=np.float64)
        clipped_tau = np.clip(local_tau, -1.0, 1.0)
        u_val_eval: Any = control_evaluator(clipped_tau)
        return np.atleast_1d(u_val_eval.squeeze()).astype(np.float64)

    # RHS for forward simulation over the first part of the merged interval
    def merged_fwd_rhs_interval_k_domain(
        local_tau_k_domain: _NormalizedTimePoint, state_np: _Vector
    ) -> _Vector:
        u_val_np = _get_control_value_for_h_reduction(
            control_evaluator_first_interval, local_tau_k_domain
        )
        state_clipped_np = np.clip(state_np, -DEFAULT_STATE_CLIP_VALUE, DEFAULT_STATE_CLIP_VALUE)

        global_tau_val = map_local_interval_tau_to_global_normalized_tau(
            local_tau_k_domain, tau_start_k, tau_shared_between_k_kp1
        )
        t_actual = alpha * global_tau_val + alpha_0

        state_ca = CasADiDM(state_clipped_np)
        u_ca = CasADiDM(u_val_np if u_val_np.size > 0 else [])
        f_rhs_sym = dynamics_function(state_ca, u_ca, t_actual, problem_parameters)
        return scaling_k * np.array(f_rhs_sym, dtype=np.float64).flatten()

    # RHS for backward simulation over the second part of the merged interval
    def merged_bwd_rhs_interval_kp1_domain(
        local_tau_kp1_domain: _NormalizedTimePoint, state_np: _Vector
    ) -> _Vector:
        u_val_np = _get_control_value_for_h_reduction(
            control_evaluator_second_interval, local_tau_kp1_domain
        )
        state_clipped_np = np.clip(state_np, -DEFAULT_STATE_CLIP_VALUE, DEFAULT_STATE_CLIP_VALUE)

        global_tau_val = map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1_domain, tau_shared_between_k_kp1, tau_end_kp1
        )
        t_actual = alpha * global_tau_val + alpha_0

        state_ca = CasADiDM(state_clipped_np)
        u_ca = CasADiDM(u_val_np if u_val_np.size > 0 else [])
        f_rhs_sym = dynamics_function(state_ca, u_ca, t_actual, problem_parameters)
        return scaling_kp1 * np.array(f_rhs_sym, dtype=np.float64).flatten()

    # Extract initial state for forward simulation
    initial_state_fwd_sim: _Vector
    try:
        if (
            solution.states
            and first_interval_idx < len(solution.states)
            and solution.states[first_interval_idx].size > 0
        ):
            Xk_nlp_from_sol_states: _Matrix = solution.states[first_interval_idx]
            initial_state_fwd_sim = Xk_nlp_from_sol_states[:, 0].flatten().astype(np.float64)
        else:  # Fallback to opti object if solution.states not populated directly
            opti: Optional[CasADiOpti] = solution.opti_object
            raw_sol = solution.raw_solution
            if not (
                problem.collocation_points_per_interval
                and raw_sol
                and opti
                and hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables")
                and opti.state_at_local_approximation_nodes_all_intervals_variables
                and first_interval_idx
                < len(opti.state_at_local_approximation_nodes_all_intervals_variables)
            ):
                logger.warning("      h-reduction failed: Cannot extract initial state data.")
                return False
            Nk_k: int = problem.collocation_points_per_interval[first_interval_idx]
            Xk_nlp_raw = raw_sol.value(
                opti.state_at_local_approximation_nodes_all_intervals_variables[first_interval_idx]
            )
            Xk_nlp = extract_and_prepare_array(Xk_nlp_raw, num_states, Nk_k + 1)
            initial_state_fwd_sim = Xk_nlp[:, 0].flatten()
    except Exception as e:
        logger.error(f"      h-reduction failed: Error getting initial state: {e}")
        return False

    # Forward simulation
    num_fwd_pts_k = max(2, num_sim_points // 2)
    fwd_tau_points_interval_k_local: _Vector = np.linspace(
        -1.0, 1.0, num_fwd_pts_k, dtype=np.float64
    )
    fwd_sim_obj_k = None
    fwd_trajectory_k: _Matrix = np.full(
        (num_states, len(fwd_tau_points_interval_k_local)), np.nan, dtype=np.float64
    )

    try:
        fwd_sim_obj_k = solve_ivp(
            merged_fwd_rhs_interval_k_domain,
            t_span=(-1.0, 1.0),
            y0=initial_state_fwd_sim,
            t_eval=fwd_tau_points_interval_k_local,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )
        if fwd_sim_obj_k.success:
            fwd_trajectory_k = fwd_sim_obj_k.y.astype(np.float64)
        else:
            logger.warning(f"      h-reduction: Fwd IVP for interval {first_interval_idx} failed.")
    except Exception as e:
        logger.error(f"      h-reduction: Exception during fwd IVP: {e}")

    # Extract terminal state for the backward simulation
    terminal_state_bwd_sim: _Vector
    try:
        if (
            solution.states
            and (first_interval_idx + 1) < len(solution.states)
            and solution.states[first_interval_idx + 1].size > 0
        ):
            Xkp1_nlp_from_sol_states: _Matrix = solution.states[first_interval_idx + 1]
            terminal_state_bwd_sim = Xkp1_nlp_from_sol_states[:, -1].flatten().astype(np.float64)
        else:  # Fallback
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if not (
                problem.collocation_points_per_interval
                and raw_sol
                and opti
                and hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables")
                and opti.state_at_local_approximation_nodes_all_intervals_variables
                and (first_interval_idx + 1)
                < len(opti.state_at_local_approximation_nodes_all_intervals_variables)
            ):
                logger.warning("      h-reduction failed: Cannot extract terminal state data.")
                return False
            Nk_kp1: int = problem.collocation_points_per_interval[first_interval_idx + 1]
            Xkp1_nlp_raw = raw_sol.value(
                opti.state_at_local_approximation_nodes_all_intervals_variables[
                    first_interval_idx + 1
                ]
            )
            Xkp1_nlp = extract_and_prepare_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)
            terminal_state_bwd_sim = Xkp1_nlp[:, -1].flatten()
    except Exception as e:
        logger.error(f"      h-reduction failed: Error getting terminal state: {e}")
        return False

    # Backward simulation
    num_bwd_pts_kp1 = max(2, num_sim_points // 2)
    bwd_tau_points_interval_kp1_local: _Vector = np.linspace(
        1.0, -1.0, num_bwd_pts_kp1, dtype=np.float64
    )
    bwd_sim_obj_kp1 = None
    bwd_trajectory_kp1_temp: _Matrix = np.full(
        (num_states, len(bwd_tau_points_interval_kp1_local)), np.nan, dtype=np.float64
    )

    try:
        bwd_sim_obj_kp1 = solve_ivp(
            merged_bwd_rhs_interval_kp1_domain,
            t_span=(1.0, -1.0),
            y0=terminal_state_bwd_sim,
            t_eval=bwd_tau_points_interval_kp1_local,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )
        if bwd_sim_obj_kp1.success:
            bwd_trajectory_kp1_temp = np.fliplr(bwd_sim_obj_kp1.y).astype(np.float64)
    except Exception as e:
        logger.error(f"      h-reduction: Exception during bwd IVP: {e}")

    # Simulation points for comparison
    sorted_bwd_tau_points_kp1_local = np.flip(bwd_tau_points_interval_kp1_local)

    if num_states == 0:  # No states, merge depends only on simulation success
        can_merge = (
            fwd_sim_obj_k is not None
            and fwd_sim_obj_k.success
            and bwd_sim_obj_kp1 is not None
            and bwd_sim_obj_kp1.success
        )
        logger.info(f"      h-reduction check (no states): Simulations successful = {can_merge}.")
        return can_merge

    # Error calculation
    all_errors_list: List[float] = []

    # Errors from forward simulation
    if fwd_sim_obj_k and fwd_sim_obj_k.success and fwd_trajectory_k.size > 0:
        nlp_states_at_fwd_pts_k: _Matrix = state_evaluator_first_interval(
            fwd_tau_points_interval_k_local
        )
        abs_diff_fwd = np.abs(fwd_trajectory_k - nlp_states_at_fwd_pts_k)
        scaled_errors_fwd = gamma_factors.flatten()[:, np.newaxis] * abs_diff_fwd
        all_errors_list.extend(list(scaled_errors_fwd.flatten()))

    # Errors from backward simulation
    if bwd_sim_obj_kp1 and bwd_sim_obj_kp1.success and bwd_trajectory_kp1_temp.size > 0:
        nlp_states_at_bwd_pts_kp1: _Matrix = state_evaluator_second_interval(
            sorted_bwd_tau_points_kp1_local
        )
        abs_diff_bwd = np.abs(bwd_trajectory_kp1_temp - nlp_states_at_bwd_pts_kp1)
        scaled_errors_bwd = gamma_factors.flatten()[:, np.newaxis] * abs_diff_bwd
        all_errors_list.extend(list(scaled_errors_bwd.flatten()))

    if not all_errors_list:
        logger.warning("      h-reduction: No error values collected.")
        return False

    max_error_for_merged_candidate = (
        np.nanmax(all_errors_list).item() if all_errors_list else np.inf
    )

    if np.isnan(max_error_for_merged_candidate):
        logger.warning("      h-reduction check: max_error calculation resulted in NaN.")
        max_error_for_merged_candidate = np.inf

    can_merge_result = max_error_for_merged_candidate <= error_tol
    logger.info(
        f"      h-reduction check result: max_error_merged_candidate = {max_error_for_merged_candidate:.4e}, "
        f"tol = {error_tol:.2e}. Merge approved: {can_merge_result}"
    )
    return can_merge_result
