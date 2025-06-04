import logging
from collections.abc import Callable
from typing import cast

import casadi as ca
import numpy as np

from trajectolab.adaptive.phs.data_structures import (
    AdaptiveParameters,
    HRefineResult,
    PReduceResult,
    PRefineResult,
    ensure_2d_array,
)
from trajectolab.adaptive.phs.error_estimation import _convert_casadi_dynamics_result_to_numpy
from trajectolab.adaptive.phs.numerical import (
    map_local_interval_tau_to_global_normalized_tau,
    map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k,
    map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1,
)
from trajectolab.tl_types import (
    FloatArray,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


__all__ = ["h_reduce_intervals", "h_refine_params", "p_reduce_interval", "p_refine_interval"]

logger = logging.getLogger(__name__)


def _calculate_merge_feasibility_from_errors(
    all_fwd_errors: FloatArray, all_bwd_errors: FloatArray, error_tol: float
) -> tuple[bool, float]:
    """VECTORIZED: Merge feasibility using NumPy operations."""
    if all_fwd_errors.size == 0 and all_bwd_errors.size == 0:
        return False, float(np.inf)  # Cast to Python float

    if (all_fwd_errors.size > 0 and np.any(np.isnan(all_fwd_errors))) or (
        all_bwd_errors.size > 0 and np.any(np.isnan(all_bwd_errors))
    ):
        return False, float(np.inf)  # Cast to Python float

    max_error_val = max(
        np.max(all_fwd_errors) if all_fwd_errors.size > 0 else 0.0,
        np.max(all_bwd_errors) if all_bwd_errors.size > 0 else 0.0,
    )

    is_feasible = bool(max_error_val <= error_tol and max_error_val != np.inf)

    return is_feasible, float(max_error_val)


def _calculate_trajectory_errors_with_gamma(
    X_sim: FloatArray, X_nlp: FloatArray, gamma_factors: FloatArray
) -> FloatArray:
    """VECTORIZED: Return FloatArray for efficiency."""
    if np.any(np.isnan(X_sim)):
        return np.array([], dtype=np.float64)

    X_nlp_flat = X_nlp.flatten() if X_nlp.ndim > 1 else X_nlp
    return (gamma_factors.flatten() * np.abs(X_sim - X_nlp_flat)).astype(np.float64)


def p_refine_interval(
    max_error: float, current_Nk: int, error_tol: float, N_max: int
) -> PRefineResult:
    """Determine new polynomial degree using p-refinement."""
    if max_error <= error_tol:
        return PRefineResult(current_Nk, False, current_Nk)

    nodes_to_add = (
        max(1, N_max - current_Nk)
        if np.isinf(max_error)
        else max(1, int(np.ceil(np.log10(max_error / error_tol))))
    )
    target_Nk = current_Nk + nodes_to_add

    if target_Nk > N_max:
        return PRefineResult(N_max, False, target_Nk)

    return PRefineResult(target_Nk, True, target_Nk)


def h_refine_params(target_Nk: int, N_min: int) -> HRefineResult:
    """Determine parameters for h-refinement."""
    num_subintervals = max(2, int(np.ceil(target_Nk / N_min)))
    return HRefineResult([N_min] * num_subintervals, num_subintervals)


def p_reduce_interval(
    current_Nk: int, max_error: float, error_tol: float, N_min: int, N_max: int
) -> PReduceResult:
    """Determine new polynomial degree using p-reduction per Eq. 36."""
    if max_error > error_tol or current_Nk <= N_min:
        return PReduceResult(current_Nk, False)

    delta = float(N_min + N_max - current_Nk)
    if abs(delta) < 1e-9:
        delta = 1.0

    try:
        ratio = error_tol / max_error
        if ratio >= 1.0:
            power_arg = np.power(ratio, 1.0 / delta)
            nodes_to_remove = int(np.floor(np.log10(power_arg))) if power_arg >= 1.0 else 0
        else:
            nodes_to_remove = 0
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        nodes_to_remove = 0

    new_Nk = max(N_min, current_Nk - max(0, nodes_to_remove))
    return PReduceResult(new_Nk, new_Nk < current_Nk)


def h_reduce_intervals(
    phase_id: PhaseID,
    first_idx: int,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    adaptive_params: AdaptiveParameters,
    gamma_factors: FloatArray,
    state_evaluator_first: Callable[[float | FloatArray], FloatArray],
    control_evaluator_first: Callable[[float | FloatArray], FloatArray] | None,
    state_evaluator_second: Callable[[float | FloatArray], FloatArray],
    control_evaluator_second: Callable[[float | FloatArray], FloatArray] | None,
) -> bool:
    """Check if two adjacent intervals can be merged."""
    if (
        solution.raw_solution is None
        or phase_id not in solution.phase_mesh_nodes
        or first_idx + 2 >= len(solution.phase_mesh_nodes[phase_id])
    ):
        return False

    num_states, _ = problem.get_phase_variable_counts(phase_id)
    phase_dynamics_function = problem.get_phase_dynamics_function(phase_id)
    global_mesh = solution.phase_mesh_nodes[phase_id]

    tau_start_k, tau_shared, tau_end_kp1 = global_mesh[first_idx : first_idx + 3]
    beta_k, beta_kp1 = (tau_shared - tau_start_k) / 2.0, (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        return False

    if (
        phase_id not in solution.phase_initial_times
        or phase_id not in solution.phase_terminal_times
    ):
        return False

    t0, tf = solution.phase_initial_times[phase_id], solution.phase_terminal_times[phase_id]
    alpha, alpha_0 = (tf - t0) / 2.0, (tf + t0) / 2.0
    scaling_k, scaling_kp1 = alpha * beta_k, alpha * beta_kp1

    def _get_control_value(
        control_evaluator: Callable[[float | FloatArray], FloatArray] | None, local_tau: float
    ) -> FloatArray:
        if control_evaluator is None:
            return np.array([], dtype=np.float64)
        return np.atleast_1d(control_evaluator(np.clip(local_tau, -1.0, 1.0)).squeeze())

    def merged_fwd_rhs(local_tau_k: float, state: FloatArray) -> FloatArray:
        u_val = _get_control_value(control_evaluator_first, local_tau_k)
        global_tau = map_local_interval_tau_to_global_normalized_tau(
            local_tau_k, tau_start_k, tau_shared
        )
        dynamics_result = phase_dynamics_function(
            ca.MX(np.clip(state, -1e6, 1e6)), ca.MX(u_val), ca.MX(alpha * global_tau + alpha_0)
        )
        return cast(
            FloatArray,
            scaling_k * _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states),
        )

    def merged_bwd_rhs(local_tau_kp1: float, state: FloatArray) -> FloatArray:
        u_val = _get_control_value(control_evaluator_second, local_tau_kp1)
        global_tau = map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1, tau_shared, tau_end_kp1
        )
        dynamics_result = phase_dynamics_function(
            ca.MX(np.clip(state, -1e6, 1e6)), ca.MX(u_val), ca.MX(alpha * global_tau + alpha_0)
        )
        return cast(
            FloatArray,
            scaling_kp1 * _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states),
        )

    # Get initial and terminal states
    try:
        if phase_id in solution.phase_solved_state_trajectories_per_interval and first_idx < len(
            solution.phase_solved_state_trajectories_per_interval[phase_id]
        ):
            initial_state_fwd = solution.phase_solved_state_trajectories_per_interval[phase_id][
                first_idx
            ][:, 0].flatten()
            terminal_state_bwd = solution.phase_solved_state_trajectories_per_interval[phase_id][
                first_idx + 1
            ][:, -1].flatten()
        else:
            # Fallback extraction
            opti, raw_sol = solution.opti_object, solution.raw_solution
            if opti is None or raw_sol is None:
                return False
            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables:
                return False
            phase_def = problem._phases[phase_id]

            Xk_nlp = ensure_2d_array(
                raw_sol.value(variables.phase_variables[phase_id].state_matrices[first_idx]),
                num_states,
                phase_def.collocation_points_per_interval[first_idx] + 1,
            )
            Xkp1_nlp = ensure_2d_array(
                raw_sol.value(variables.phase_variables[phase_id].state_matrices[first_idx + 1]),
                num_states,
                phase_def.collocation_points_per_interval[first_idx + 1] + 1,
            )
            initial_state_fwd, terminal_state_bwd = (
                Xk_nlp[:, 0].flatten(),
                Xkp1_nlp[:, -1].flatten(),
            )
    except Exception:
        return False

    # Simulation setup
    num_sim_points = adaptive_params.num_error_sim_points
    target_end_tau_k = map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    target_end_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )

    fwd_tau_points = np.linspace(
        -1.0, target_end_tau_k, max(2, num_sim_points // 2), dtype=np.float64
    )
    bwd_tau_points = np.linspace(
        1.0, target_end_tau_kp1, max(2, num_sim_points // 2), dtype=np.float64
    )

    configured_ode_solver = adaptive_params.get_ode_solver()

    # Forward simulation
    try:
        fwd_sim = configured_ode_solver(
            merged_fwd_rhs,
            t_span=(-1.0, target_end_tau_k),
            y0=initial_state_fwd,
            t_eval=fwd_tau_points,
        )
        fwd_success, fwd_trajectory = (
            fwd_sim.success,
            fwd_sim.y
            if fwd_sim.success
            else np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64),
        )
    except (RuntimeError, OverflowError, FloatingPointError):
        fwd_success, fwd_trajectory = (
            False,
            np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64),
        )

    # Backward simulation
    try:
        bwd_sim = configured_ode_solver(
            merged_bwd_rhs,
            t_span=(1.0, target_end_tau_kp1),
            y0=terminal_state_bwd,
            t_eval=bwd_tau_points,
        )
        bwd_success = bwd_sim.success
        bwd_trajectory = (
            np.array(bwd_sim.y[:, ::-1], dtype=np.float64)
            if bwd_success
            else np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64)
        )
    except (RuntimeError, OverflowError, FloatingPointError):
        bwd_success, bwd_trajectory = (
            False,
            np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64),
        )

    if num_states == 0:
        return fwd_success and bwd_success

    # VECTORIZED: Error calculation
    all_fwd_errors = np.concatenate(
        [
            _calculate_trajectory_errors_with_gamma(
                fwd_trajectory[:, i],
                state_evaluator_first(zeta_k)
                if -1.0 <= zeta_k <= 1.0 + 1e-9
                else state_evaluator_second(
                    map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                        zeta_k, tau_start_k, tau_shared, tau_end_kp1
                    )
                ),
                gamma_factors,
            )
            for i, zeta_k in enumerate(fwd_tau_points)
        ]
    )

    all_bwd_errors = np.concatenate(
        [
            _calculate_trajectory_errors_with_gamma(
                bwd_trajectory[:, i],
                state_evaluator_second(zeta_kp1)
                if -1.0 - 1e-9 <= zeta_kp1 <= 1.0
                else state_evaluator_first(
                    map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
                        zeta_kp1, tau_start_k, tau_shared, tau_end_kp1
                    )
                ),
                gamma_factors,
            )
            for i, zeta_kp1 in enumerate(np.flip(bwd_tau_points))
        ]
    )

    can_merge, _ = _calculate_merge_feasibility_from_errors(
        all_fwd_errors, all_bwd_errors, adaptive_params.error_tolerance
    )
    return can_merge
