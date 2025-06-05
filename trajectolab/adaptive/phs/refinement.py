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
    _ensure_2d_array,
)
from trajectolab.adaptive.phs.error_estimation import _convert_casadi_dynamics_result_to_numpy
from trajectolab.adaptive.phs.numerical import (
    _map_local_interval_tau_to_global_normalized_tau,
    _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k,
    _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1,
)
from trajectolab.tl_types import (
    FloatArray,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


__all__ = ["h_reduce_intervals", "h_refine_params", "p_reduce_interval", "p_refine_interval"]

logger = logging.getLogger(__name__)


def _validate_error_arrays(all_fwd_errors: FloatArray, all_bwd_errors: FloatArray) -> bool:
    """Validate error arrays for NaN values."""
    if all_fwd_errors.size == 0 and all_bwd_errors.size == 0:
        return False

    fwd_has_nan = all_fwd_errors.size > 0 and np.any(np.isnan(all_fwd_errors))
    bwd_has_nan = all_bwd_errors.size > 0 and np.any(np.isnan(all_bwd_errors))

    return not (fwd_has_nan or bwd_has_nan)


def _compute_max_error_value(all_fwd_errors: FloatArray, all_bwd_errors: FloatArray) -> float:
    """Compute maximum error value from forward and backward errors."""
    max_fwd = np.max(all_fwd_errors) if all_fwd_errors.size > 0 else 0.0
    max_bwd = np.max(all_bwd_errors) if all_bwd_errors.size > 0 else 0.0
    return max(max_fwd, max_bwd)


def _calculate_merge_feasibility_from_errors(
    all_fwd_errors: FloatArray, all_bwd_errors: FloatArray, error_tol: float
) -> tuple[bool, float]:
    """Calculate merge feasibility from error arrays."""
    if not _validate_error_arrays(all_fwd_errors, all_bwd_errors):
        return False, float(np.inf)

    max_error_val = _compute_max_error_value(all_fwd_errors, all_bwd_errors)
    is_feasible = bool(max_error_val <= error_tol and max_error_val != np.inf)

    return is_feasible, float(max_error_val)


def _calculate_trajectory_errors_with_gamma(
    X_sim: FloatArray, X_nlp: FloatArray, gamma_factors: FloatArray
) -> FloatArray:
    """Calculate trajectory errors with gamma scaling."""
    if np.any(np.isnan(X_sim)):
        return np.array([], dtype=np.float64)

    X_nlp_flat = X_nlp.flatten() if X_nlp.ndim > 1 else X_nlp
    return cast(FloatArray, (gamma_factors.flatten() * np.abs(X_sim - X_nlp_flat)))


def _compute_p_refine_target(
    max_error: float, current_Nk: int, error_tol: float, N_max: int
) -> int:
    """Compute target polynomial degree for p-refinement."""
    if np.isinf(max_error):
        return max(1, N_max - current_Nk)
    return max(1, int(np.ceil(np.log10(max_error / error_tol))))


def p_refine_interval(
    max_error: float, current_Nk: int, error_tol: float, N_max: int
) -> PRefineResult:
    """Determine new polynomial degree using p-refinement."""
    if max_error <= error_tol:
        return PRefineResult(current_Nk, False, current_Nk)

    nodes_to_add = _compute_p_refine_target(max_error, current_Nk, error_tol, N_max)
    target_Nk = current_Nk + nodes_to_add

    if target_Nk > N_max:
        return PRefineResult(N_max, False, target_Nk)

    return PRefineResult(target_Nk, True, target_Nk)


def h_refine_params(target_Nk: int, N_min: int) -> HRefineResult:
    """Determine parameters for h-refinement."""
    num_subintervals = max(2, int(np.ceil(target_Nk / N_min)))
    return HRefineResult([N_min] * num_subintervals, num_subintervals)


def _compute_p_reduce_delta(current_Nk: int, N_min: int, N_max: int) -> float:
    """Compute delta parameter for p-reduction calculation."""
    delta = float(N_min + N_max - current_Nk)
    return delta if abs(delta) >= 1e-9 else 1.0


def _calculate_nodes_to_remove(error_tol: float, max_error: float, delta: float) -> int:
    """Calculate number of nodes to remove in p-reduction."""
    try:
        ratio = error_tol / max_error
        if ratio >= 1.0:
            power_arg = np.power(ratio, 1.0 / delta)
            return int(np.floor(np.log10(power_arg))) if power_arg >= 1.0 else 0
        return 0
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        return 0


def p_reduce_interval(
    current_Nk: int, max_error: float, error_tol: float, N_min: int, N_max: int
) -> PReduceResult:
    """Determine new polynomial degree using p-reduction per Eq. 36."""
    if max_error > error_tol or current_Nk <= N_min:
        return PReduceResult(current_Nk, False)

    delta = _compute_p_reduce_delta(current_Nk, N_min, N_max)
    nodes_to_remove = _calculate_nodes_to_remove(error_tol, max_error, delta)
    new_Nk = max(N_min, current_Nk - max(0, nodes_to_remove))

    return PReduceResult(new_Nk, new_Nk < current_Nk)


def _validate_merge_preconditions(
    solution: OptimalControlSolution, phase_id: PhaseID, first_idx: int
) -> bool:
    """Validate preconditions for interval merging."""
    if (
        solution.raw_solution is None
        or phase_id not in solution.phase_mesh_nodes
        or first_idx + 2 >= len(solution.phase_mesh_nodes[phase_id])
    ):
        return False

    required_attrs = ["phase_initial_times", "phase_terminal_times"]
    return all(phase_id in getattr(solution, attr) for attr in required_attrs)


def _extract_merge_time_parameters(
    solution: OptimalControlSolution, phase_id: PhaseID
) -> tuple[float, float, float, float]:
    """Extract time transformation parameters for merging."""
    t0 = solution.phase_initial_times[phase_id]
    tf = solution.phase_terminal_times[phase_id]
    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0
    return t0, tf, alpha, alpha_0


def _extract_merge_mesh_parameters(
    solution: OptimalControlSolution, phase_id: PhaseID, first_idx: int
) -> tuple[float, float, float, float, float]:
    """Extract mesh parameters for interval merging."""
    global_mesh = solution.phase_mesh_nodes[phase_id]
    tau_start_k, tau_shared, tau_end_kp1 = global_mesh[first_idx : first_idx + 3]
    beta_k = (tau_shared - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        raise ValueError("Interval has zero width")

    return tau_start_k, tau_shared, tau_end_kp1, beta_k, beta_kp1


def _create_control_evaluator(
    control_evaluator: Callable[[float | FloatArray], FloatArray] | None,
) -> Callable[[float], FloatArray]:
    """Create safe control value evaluator."""

    def _get_control_value(local_tau: float) -> FloatArray:
        if control_evaluator is None:
            return np.array([], dtype=np.float64)
        return np.atleast_1d(control_evaluator(np.clip(local_tau, -1.0, 1.0)).squeeze())

    return _get_control_value


def _create_merged_dynamics_functions(
    phase_dynamics_function: Callable,
    control_evaluator_first: Callable,
    control_evaluator_second: Callable,
    alpha: float,
    alpha_0: float,
    tau_start_k: float,
    tau_shared: float,
    tau_end_kp1: float,
    beta_k: float,
    beta_kp1: float,
    num_states: int,
) -> tuple[Callable, Callable]:
    """Create merged forward and backward dynamics functions."""
    get_control_first = _create_control_evaluator(control_evaluator_first)
    get_control_second = _create_control_evaluator(control_evaluator_second)

    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    def merged_fwd_rhs(local_tau_k: float, state: FloatArray) -> FloatArray:
        u_val = get_control_first(local_tau_k)
        global_tau = _map_local_interval_tau_to_global_normalized_tau(
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
        u_val = get_control_second(local_tau_kp1)
        global_tau = _map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1, tau_shared, tau_end_kp1
        )
        dynamics_result = phase_dynamics_function(
            ca.MX(np.clip(state, -1e6, 1e6)), ca.MX(u_val), ca.MX(alpha * global_tau + alpha_0)
        )
        return cast(
            FloatArray,
            scaling_kp1 * _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states),
        )

    return merged_fwd_rhs, merged_bwd_rhs


def _extract_boundary_states_for_merge(
    solution: OptimalControlSolution,
    phase_id: PhaseID,
    first_idx: int,
    problem: ProblemProtocol,
    num_states: int,
) -> tuple[FloatArray, FloatArray]:
    """Extract boundary states for merge simulation."""
    if phase_id in solution.phase_solved_state_trajectories_per_interval and first_idx < len(
        solution.phase_solved_state_trajectories_per_interval[phase_id]
    ):
        initial_state_fwd = solution.phase_solved_state_trajectories_per_interval[phase_id][
            first_idx
        ][:, 0].flatten()
        terminal_state_bwd = solution.phase_solved_state_trajectories_per_interval[phase_id][
            first_idx + 1
        ][:, -1].flatten()
        return initial_state_fwd, terminal_state_bwd

    # Fallback to raw solution extraction
    opti, raw_sol = solution.opti_object, solution.raw_solution
    if opti is None or raw_sol is None:
        raise ValueError("Cannot extract boundary states")

    variables = opti.multiphase_variables_reference
    if phase_id not in variables.phase_variables:
        raise ValueError(f"Phase {phase_id} not in variables")

    phase_def = problem._phases[phase_id]

    Xk_nlp = _ensure_2d_array(
        raw_sol.value(variables.phase_variables[phase_id].state_matrices[first_idx]),
        num_states,
        phase_def.collocation_points_per_interval[first_idx] + 1,
    )
    Xkp1_nlp = _ensure_2d_array(
        raw_sol.value(variables.phase_variables[phase_id].state_matrices[first_idx + 1]),
        num_states,
        phase_def.collocation_points_per_interval[first_idx + 1] + 1,
    )

    return Xk_nlp[:, 0].flatten(), Xkp1_nlp[:, -1].flatten()


def _setup_merge_simulation_points(
    adaptive_params: AdaptiveParameters,
    tau_start_k: float,
    tau_shared: float,
    tau_end_kp1: float,
) -> tuple[FloatArray, FloatArray]:
    """Setup tau points for merge simulation."""
    num_sim_points = adaptive_params.num_error_sim_points
    target_end_tau_k = _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    target_end_tau_kp1 = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )

    fwd_tau_points = np.linspace(
        -1.0, target_end_tau_k, max(2, num_sim_points // 2), dtype=np.float64
    )
    bwd_tau_points = np.linspace(
        1.0, target_end_tau_kp1, max(2, num_sim_points // 2), dtype=np.float64
    )

    return fwd_tau_points, bwd_tau_points


def _run_merge_simulation(
    configured_ode_solver: Callable,
    dynamics_rhs: Callable,
    t_span: tuple[float, float],
    y0: FloatArray,
    t_eval: FloatArray,
    num_states: int,
) -> tuple[bool, FloatArray]:
    """Run merge simulation with error handling."""
    try:
        sim = configured_ode_solver(dynamics_rhs, t_span=t_span, y0=y0, t_eval=t_eval)
        if sim.success:
            return True, sim.y
        return False, np.full((num_states, len(t_eval)), np.nan, dtype=np.float64)
    except (RuntimeError, OverflowError, FloatingPointError):
        return False, np.full((num_states, len(t_eval)), np.nan, dtype=np.float64)


def _run_merge_simulations(
    merged_fwd_rhs: Callable,
    merged_bwd_rhs: Callable,
    initial_state_fwd: FloatArray,
    terminal_state_bwd: FloatArray,
    fwd_tau_points: FloatArray,
    bwd_tau_points: FloatArray,
    adaptive_params: AdaptiveParameters,
    num_states: int,
) -> tuple[bool, bool, FloatArray, FloatArray]:
    """Run both forward and backward merge simulations."""
    configured_ode_solver = adaptive_params._get_ode_solver()

    # Forward simulation
    fwd_success, fwd_trajectory = _run_merge_simulation(
        configured_ode_solver,
        merged_fwd_rhs,
        (-1.0, fwd_tau_points[-1]),
        initial_state_fwd,
        fwd_tau_points,
        num_states,
    )

    # Backward simulation
    bwd_success, bwd_trajectory_raw = _run_merge_simulation(
        configured_ode_solver,
        merged_bwd_rhs,
        (1.0, bwd_tau_points[-1]),
        terminal_state_bwd,
        bwd_tau_points,
        num_states,
    )

    bwd_trajectory = (
        np.array(bwd_trajectory_raw[:, ::-1], dtype=np.float64)
        if bwd_success
        else np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64)
    )

    return fwd_success, bwd_success, fwd_trajectory, bwd_trajectory


def _calculate_merge_errors(
    fwd_trajectory: FloatArray,
    bwd_trajectory: FloatArray,
    fwd_tau_points: FloatArray,
    bwd_tau_points: FloatArray,
    state_evaluator_first: Callable,
    state_evaluator_second: Callable,
    gamma_factors: FloatArray,
    tau_start_k: float,
    tau_shared: float,
    tau_end_kp1: float,
) -> tuple[FloatArray, FloatArray]:
    """Calculate errors for merge feasibility assessment."""
    all_fwd_errors = np.concatenate(
        [
            _calculate_trajectory_errors_with_gamma(
                fwd_trajectory[:, i],
                state_evaluator_first(zeta_k)
                if -1.0 <= zeta_k <= 1.0 + 1e-9
                else state_evaluator_second(
                    _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
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
                    _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
                        zeta_kp1, tau_start_k, tau_shared, tau_end_kp1
                    )
                ),
                gamma_factors,
            )
            for i, zeta_kp1 in enumerate(np.flip(bwd_tau_points))
        ]
    )

    return all_fwd_errors, all_bwd_errors


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
    # Validate preconditions
    if not _validate_merge_preconditions(solution, phase_id, first_idx):
        return False

    num_states, _ = problem._get_phase_variable_counts(phase_id)
    phase_dynamics_function = problem._get_phase_dynamics_function(phase_id)

    # Extract parameters
    t0, tf, alpha, alpha_0 = _extract_merge_time_parameters(solution, phase_id)

    try:
        tau_start_k, tau_shared, tau_end_kp1, beta_k, beta_kp1 = _extract_merge_mesh_parameters(
            solution, phase_id, first_idx
        )
    except ValueError:
        return False

    # Create dynamics functions
    merged_fwd_rhs, merged_bwd_rhs = _create_merged_dynamics_functions(
        phase_dynamics_function,
        control_evaluator_first,
        control_evaluator_second,
        alpha,
        alpha_0,
        tau_start_k,
        tau_shared,
        tau_end_kp1,
        beta_k,
        beta_kp1,
        num_states,
    )

    # Extract boundary states
    try:
        initial_state_fwd, terminal_state_bwd = _extract_boundary_states_for_merge(
            solution, phase_id, first_idx, problem, num_states
        )
    except (Exception, ValueError):
        return False

    # Setup simulation points
    fwd_tau_points, bwd_tau_points = _setup_merge_simulation_points(
        adaptive_params, tau_start_k, tau_shared, tau_end_kp1
    )

    # Run simulations
    fwd_success, bwd_success, fwd_trajectory, bwd_trajectory = _run_merge_simulations(
        merged_fwd_rhs,
        merged_bwd_rhs,
        initial_state_fwd,
        terminal_state_bwd,
        fwd_tau_points,
        bwd_tau_points,
        adaptive_params,
        num_states,
    )

    if num_states == 0:
        return fwd_success and bwd_success

    # Calculate errors
    all_fwd_errors, all_bwd_errors = _calculate_merge_errors(
        fwd_trajectory,
        bwd_trajectory,
        fwd_tau_points,
        bwd_tau_points,
        state_evaluator_first,
        state_evaluator_second,
        gamma_factors,
        tau_start_k,
        tau_shared,
        tau_end_kp1,
    )

    # Determine feasibility
    can_merge, _ = _calculate_merge_feasibility_from_errors(
        all_fwd_errors, all_bwd_errors, adaptive_params.error_tolerance
    )
    return can_merge
