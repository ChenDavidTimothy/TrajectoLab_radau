import logging
from collections.abc import Callable
from typing import cast

import casadi as ca
import numpy as np

from trajectolab.tl_types import (
    FloatArray,
    ODESolverCallable,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


__all__ = [
    "calculate_gamma_normalizers_for_phase",
    "calculate_relative_error_estimate",
    "simulate_dynamics_for_phase_interval_error_estimation",
]

logger = logging.getLogger(__name__)


def _calculate_gamma_normalization_factors(max_state_values: FloatArray) -> FloatArray:
    """Pure gamma calculation - easily testable."""
    gamma_denominator = 1.0 + max_state_values
    return (1.0 / np.maximum(gamma_denominator, np.float64(1e-12))).reshape(-1, 1)


def _find_maximum_state_values_across_phase_intervals(
    Y_solved_list: list[FloatArray],
) -> FloatArray:
    """VECTORIZED: Pure maximum value calculation across intervals."""
    if not Y_solved_list:
        return np.array([], dtype=np.float64)

    valid_intervals = [Xk for Xk in Y_solved_list if Xk.size > 0]
    if not valid_intervals:
        return np.array([], dtype=np.float64)

    # VECTORIZED: Single concatenation + max operation
    return cast(FloatArray, np.max(np.abs(np.concatenate(valid_intervals, axis=1)), axis=1))


def _calculate_trajectory_error_differences(
    sim_trajectory: FloatArray, nlp_trajectory: FloatArray, gamma_factors: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Pure trajectory error calculation - easily testable."""
    abs_diff = np.abs(sim_trajectory - nlp_trajectory)
    scaled_errors = gamma_factors * abs_diff
    max_errors_per_state = (
        np.nanmax(scaled_errors, axis=1)
        if scaled_errors.size > 0
        else np.zeros(gamma_factors.shape[0], dtype=np.float64)
    )
    return abs_diff, max_errors_per_state


def _calculate_combined_error_estimate(
    max_fwd_errors_per_state: FloatArray, max_bwd_errors_per_state: FloatArray
) -> float:
    """VECTORIZED: Combined error using pure NumPy operations."""
    if max_fwd_errors_per_state.shape != max_bwd_errors_per_state.shape:
        raise ValueError("Input arrays must have the same shape.")

    # VECTORIZED: All conditional logic in one expression
    fwd_valid = ~np.isnan(max_fwd_errors_per_state)
    bwd_valid = ~np.isnan(max_bwd_errors_per_state)
    combined_errors = np.where(
        fwd_valid & bwd_valid,
        np.maximum(max_fwd_errors_per_state, max_bwd_errors_per_state),
        np.where(
            fwd_valid,
            max_fwd_errors_per_state,
            np.where(bwd_valid, max_bwd_errors_per_state, np.nan),
        ),
    )

    max_error = float(np.nanmax(combined_errors)) if combined_errors.size > 0 else 0.0
    if np.isnan(max_error):
        return np.inf

    return max(max_error, 1e-15) if 0.0 < max_error < 1e-15 else float(max_error)


def _convert_casadi_dynamics_result_to_numpy(dynamics_result: ca.MX, num_states: int) -> FloatArray:
    """VECTORIZED: Convert dynamics result using optimized CasADi operations."""
    if isinstance(dynamics_result, ca.MX):
        # VECTORIZED: Direct conversion instead of element loops
        result_dm = ca.evalf(dynamics_result)
        return np.array(result_dm.full(), dtype=np.float64).flatten()
    elif isinstance(dynamics_result, list):
        # VECTORIZED: Batch evaluation
        return np.array([float(ca.evalf(expr)) for expr in dynamics_result], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported dynamics result type: {type(dynamics_result)}")


def simulate_dynamics_for_phase_interval_error_estimation(
    phase_id: PhaseID,
    interval_idx: int,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    state_evaluator: Callable[[float | FloatArray], FloatArray],
    control_evaluator: Callable[[float | FloatArray], FloatArray],
    ode_solver: ODESolverCallable,
    n_eval_points: int = 50,
) -> tuple[bool, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Simulate dynamics for error estimation."""
    if not solution.success or solution.raw_solution is None:
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    num_states, _ = problem.get_phase_variable_counts(phase_id)
    phase_dynamics_function = problem.get_phase_dynamics_function(phase_id)

    if (
        phase_id not in solution.phase_initial_times
        or phase_id not in solution.phase_terminal_times
        or phase_id not in solution.phase_mesh_nodes
    ):
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    t0, tf = solution.phase_initial_times[phase_id], solution.phase_terminal_times[phase_id]
    alpha, alpha_0 = (tf - t0) / 2.0, (tf + t0) / 2.0

    global_mesh = solution.phase_mesh_nodes[phase_id]
    if interval_idx + 1 >= len(global_mesh):
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    tau_start, tau_end = global_mesh[interval_idx], global_mesh[interval_idx + 1]
    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < 1e-12:
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    beta_k0, overall_scaling = (tau_end + tau_start) / 2.0, alpha * beta_k

    def dynamics_rhs(tau: float, state: FloatArray) -> FloatArray:
        control = (
            control_evaluator(tau).flatten()
            if control_evaluator(tau).ndim > 1
            else control_evaluator(tau)
        )
        global_tau = beta_k * tau + beta_k0
        physical_time = alpha * global_tau + alpha_0
        dynamics_result = phase_dynamics_function(
            ca.MX(state), ca.MX(control), ca.MX(physical_time)
        )
        state_deriv_np = _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states)
        return cast(FloatArray, overall_scaling * state_deriv_np)

    # Forward and backward simulation
    initial_state = (
        state_evaluator(-1.0).flatten() if state_evaluator(-1.0).ndim > 1 else state_evaluator(-1.0)
    )
    terminal_state = (
        state_evaluator(1.0).flatten() if state_evaluator(1.0).ndim > 1 else state_evaluator(1.0)
    )

    fwd_tau_points = np.linspace(-1, 1, n_eval_points, dtype=np.float64)
    bwd_tau_points = np.linspace(1, -1, n_eval_points, dtype=np.float64)

    try:
        fwd_sim = ode_solver(dynamics_rhs, t_span=(-1, 1), y0=initial_state, t_eval=fwd_tau_points)
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

    try:
        bwd_sim = ode_solver(dynamics_rhs, t_span=(1, -1), y0=terminal_state, t_eval=bwd_tau_points)
        bwd_success, bwd_trajectory = (
            bwd_sim.success,
            np.fliplr(bwd_sim.y)
            if bwd_sim.success
            else np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64),
        )
    except (RuntimeError, OverflowError, FloatingPointError):
        bwd_success, bwd_trajectory = (
            False,
            np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64),
        )

    return (
        fwd_success and bwd_success,
        fwd_tau_points,
        fwd_trajectory,
        state_evaluator(fwd_tau_points),
        np.flip(bwd_tau_points),
        bwd_trajectory,
        state_evaluator(np.flip(bwd_tau_points)),
    )


def calculate_relative_error_estimate(
    phase_id: PhaseID,
    interval_idx: int,
    success: bool,
    fwd_sim_traj: FloatArray,
    fwd_nlp_traj: FloatArray,
    bwd_sim_traj: FloatArray,
    bwd_nlp_traj: FloatArray,
    gamma_factors: FloatArray,
) -> float:
    """Calculate relative error estimate."""
    if not success or fwd_sim_traj.size == 0 or bwd_sim_traj.size == 0:
        return np.inf

    if fwd_sim_traj.shape[0] == 0:
        return 0.0

    _, max_fwd_errors = _calculate_trajectory_error_differences(
        fwd_sim_traj, fwd_nlp_traj, gamma_factors
    )
    _, max_bwd_errors = _calculate_trajectory_error_differences(
        bwd_sim_traj, bwd_nlp_traj, gamma_factors
    )

    max_error = _calculate_combined_error_estimate(max_fwd_errors, max_bwd_errors)
    return np.inf if np.isnan(max_error) else max_error


def calculate_gamma_normalizers_for_phase(
    solution: OptimalControlSolution, problem: ProblemProtocol, phase_id: PhaseID
) -> FloatArray | None:
    """Calculate gamma normalization factors for error estimation."""
    if not solution.success or solution.raw_solution is None:
        return None

    num_states, _ = problem.get_phase_variable_counts(phase_id)
    if num_states == 0:
        return np.array([], dtype=np.float64).reshape(0, 1)

    if phase_id not in solution.phase_solved_state_trajectories_per_interval:
        return None

    Y_solved_list = solution.phase_solved_state_trajectories_per_interval[phase_id]
    if not Y_solved_list:
        return None

    max_abs_values = _find_maximum_state_values_across_phase_intervals(Y_solved_list)
    return _calculate_gamma_normalization_factors(max_abs_values)
