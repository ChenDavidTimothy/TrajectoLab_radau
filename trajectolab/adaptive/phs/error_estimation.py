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
from trajectolab.utils.constants import DEFAULT_ODE_RTOL


__all__ = [
    "calculate_gamma_normalizers_for_phase",
    "calculate_relative_error_estimate",
    "simulate_dynamics_for_phase_interval_error_estimation",
]

logger = logging.getLogger(__name__)


# ========================================================================
# MATHEMATICAL CORE FUNCTIONS - Pure calculations (UNCHANGED)
# ========================================================================


def _calculate_gamma_normalization_factors(max_state_values: FloatArray) -> FloatArray:
    """Pure gamma calculation - easily testable."""
    gamma_denominator = 1.0 + max_state_values
    gamma_factors = 1.0 / np.maximum(gamma_denominator, np.float64(1e-12))
    return gamma_factors.reshape(-1, 1)


def _find_maximum_state_values_across_phase_intervals(
    Y_solved_list: list[FloatArray],
) -> FloatArray:
    """Pure maximum value calculation across intervals in a phase - easily testable."""
    if not Y_solved_list:
        return np.array([], dtype=np.float64)

    # Determine number of states from first non-empty interval
    num_states = 0
    for Xk in Y_solved_list:
        if Xk.size > 0:
            num_states = Xk.shape[0]
            break

    if num_states == 0:
        return np.array([], dtype=np.float64)

    # Find maximum absolute value for each state component
    max_abs_values = np.zeros(num_states, dtype=np.float64)
    first_interval = True

    for Xk in Y_solved_list:
        if Xk.size == 0:
            continue

        max_abs_in_interval = np.max(np.abs(Xk), axis=1)

        if first_interval:
            max_abs_values = max_abs_in_interval
            first_interval = False
        else:
            max_abs_values = np.maximum(max_abs_values, max_abs_in_interval)

    return max_abs_values


def _calculate_trajectory_error_differences(
    sim_trajectory: FloatArray, nlp_trajectory: FloatArray, gamma_factors: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Pure trajectory error calculation - easily testable."""
    # Calculate absolute differences
    abs_diff = np.abs(sim_trajectory - nlp_trajectory)

    # Apply gamma scaling
    scaled_errors = gamma_factors * abs_diff

    # Calculate maximum errors per state
    max_errors_per_state = (
        np.nanmax(scaled_errors, axis=1)
        if scaled_errors.size > 0
        else np.zeros(gamma_factors.shape[0], dtype=np.float64)
    )

    return abs_diff, max_errors_per_state


def _calculate_combined_error_estimate(
    max_fwd_errors_per_state: FloatArray, max_bwd_errors_per_state: FloatArray
) -> float:
    """
    Calculates the combined maximum error from forward and backward simulations.
    Prioritizes valid numerical errors over NaNs for each state component.
    """
    if max_fwd_errors_per_state.shape != max_bwd_errors_per_state.shape:
        raise ValueError("Input arrays must have the same shape.")

    num_states = max_fwd_errors_per_state.shape[0]
    combined_errors_per_state = np.full(num_states, np.nan, dtype=np.float64)

    for i in range(num_states):
        fwd_err = max_fwd_errors_per_state[i]
        bwd_err = max_bwd_errors_per_state[i]

        is_fwd_nan = np.isnan(fwd_err)
        is_bwd_nan = np.isnan(bwd_err)

        if not is_fwd_nan and not is_bwd_nan:
            combined_errors_per_state[i] = np.float64(max(fwd_err, bwd_err))
        elif not is_fwd_nan:  # Only fwd is valid
            combined_errors_per_state[i] = np.float64(fwd_err)
        elif not is_bwd_nan:  # Only bwd is valid
            combined_errors_per_state[i] = np.float64(bwd_err)
        # If both are NaN, combined_errors_per_state[i] remains NaN

    # Aggregate errors across states, ignoring NaNs if other valid errors exist
    if combined_errors_per_state.size > 0:
        max_error = float(np.nanmax(combined_errors_per_state))
    else:  # No states
        max_error = 0.0

    # If all per-state errors were NaN, nanmax returns NaN (with a warning)
    if np.isnan(max_error):
        return np.inf

    # Ensure a minimum error threshold for numerical stability
    MIN_ERROR_THRESHOLD = 1e-15
    if max_error < MIN_ERROR_THRESHOLD and max_error != 0.0:  # Allow true zero error
        max_error = MIN_ERROR_THRESHOLD

    return float(max_error)


def _convert_casadi_dynamics_result_to_numpy(dynamics_result: ca.MX, num_states: int) -> FloatArray:
    """
    Convert optimized dynamics result (ca.MX) to numpy array.

    Handles the new dynamics interface where function returns ca.MX directly
    instead of list[ca.MX].
    """
    if isinstance(dynamics_result, ca.MX):
        # New optimized interface - ca.MX result
        if dynamics_result.shape[0] == num_states and dynamics_result.shape[1] == 1:
            # Correct shape - convert directly
            state_deriv_np = np.array(
                [float(ca.evalf(dynamics_result[i])) for i in range(num_states)], dtype=np.float64
            )
        elif dynamics_result.shape[0] == 1 and dynamics_result.shape[1] == num_states:
            # Transposed - need to transpose first
            state_deriv_np = np.array(
                [float(ca.evalf(dynamics_result[0, i])) for i in range(num_states)],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unexpected dynamics result shape: {dynamics_result.shape}, expected ({num_states}, 1)"
            )
    elif isinstance(dynamics_result, list):
        # Legacy interface - list[ca.MX] (backward compatibility)
        if len(dynamics_result) != num_states:
            raise ValueError(
                f"Dynamics list length {len(dynamics_result)} != expected states {num_states}"
            )
        state_deriv_np = np.array(
            [float(ca.evalf(expr)) for expr in dynamics_result], dtype=np.float64
        )
    else:
        raise ValueError(f"Unsupported dynamics result type: {type(dynamics_result)}")

    return state_deriv_np


# ========================================================================
# STREAMLINED SIMULATION FUNCTIONS
# ========================================================================


def simulate_dynamics_for_phase_interval_error_estimation(
    phase_id: PhaseID,
    interval_idx: int,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    state_evaluator: Callable[[float | FloatArray], FloatArray],
    control_evaluator: Callable[[float | FloatArray], FloatArray],
    ode_solver: ODESolverCallable,
    ode_rtol: float = DEFAULT_ODE_RTOL,
    n_eval_points: int = 50,
) -> tuple[bool, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Updated for optimized dynamics interface.
    Returns: (success, fwd_tau_points, fwd_sim_traj, fwd_nlp_traj, bwd_tau_points, bwd_sim_traj, bwd_nlp_traj)
    """
    if not solution.success or solution.raw_solution is None:
        logger.warning(f"NLP solution unsuccessful for phase {phase_id} interval {interval_idx}")
        # Return empty arrays on failure
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    # Get variable counts for this phase
    num_states, num_controls = problem.get_phase_variable_counts(phase_id)
    phase_dynamics_function = problem.get_phase_dynamics_function(phase_id)

    # Validate time variables for this phase
    if (
        phase_id not in solution.phase_initial_times
        or phase_id not in solution.phase_terminal_times
    ):
        logger.error(f"Missing time variables for phase {phase_id} interval {interval_idx}")
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    # Time transformation parameters for this phase
    t0 = solution.phase_initial_times[phase_id]
    tf = solution.phase_terminal_times[phase_id]
    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    # Validate phase mesh
    if phase_id not in solution.phase_mesh_nodes:
        logger.error(f"Missing mesh for phase {phase_id} interval {interval_idx}")
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    global_mesh = solution.phase_mesh_nodes[phase_id]
    if interval_idx + 1 >= len(global_mesh):
        logger.error(f"Interval {interval_idx} out of bounds for phase {phase_id} mesh")
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    tau_start = global_mesh[interval_idx]
    tau_end = global_mesh[interval_idx + 1]

    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < 1e-12:
        logger.warning(f"Phase {phase_id} interval {interval_idx} has zero length")
        empty = np.array([], dtype=np.float64)
        return False, empty, empty, empty, empty, empty, empty

    beta_k0 = (tau_end + tau_start) / 2.0
    overall_scaling = alpha * beta_k

    def dynamics_rhs(tau: float, state: FloatArray) -> FloatArray:
        """Right-hand side of dynamics ODE in local tau coordinates."""
        # Get control from interpolant
        control = control_evaluator(tau)
        if control.ndim > 1:
            control = control.flatten()

        # Convert to global coordinates and physical time
        global_tau = beta_k * tau + beta_k0
        physical_time = alpha * global_tau + alpha_0

        # Handle optimized dynamics interface (ca.MX result)
        dynamics_result = phase_dynamics_function(
            ca.MX(state), ca.MX(control), ca.MX(physical_time)
        )

        # Convert using new helper function that handles both interfaces
        state_deriv_np = _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states)

        if state_deriv_np.shape[0] != num_states:
            raise ValueError(
                f"Phase {phase_id} dynamics output dimension mismatch. Expected {num_states}, got {state_deriv_np.shape[0]}"
            )

        return cast(FloatArray, overall_scaling * state_deriv_np)

    # Forward simulation
    initial_state = state_evaluator(-1.0)
    if initial_state.ndim > 1:
        initial_state = initial_state.flatten()

    fwd_tau_points = np.linspace(-1, 1, n_eval_points, dtype=np.float64)

    try:
        fwd_sim = ode_solver(
            dynamics_rhs,
            t_span=(-1, 1),
            y0=initial_state,
            t_eval=fwd_tau_points,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_rtol * 1e-2,
        )
        fwd_success = fwd_sim.success
        fwd_trajectory = (
            fwd_sim.y
            if fwd_success
            else np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64)
        )
    except Exception as e:
        logger.error(f"Forward simulation failed: {e}")
        fwd_success = False
        fwd_trajectory = np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64)

    fwd_nlp_trajectory = state_evaluator(fwd_tau_points)

    # Backward simulation
    terminal_state = state_evaluator(1.0)
    if terminal_state.ndim > 1:
        terminal_state = terminal_state.flatten()

    bwd_tau_points = np.linspace(1, -1, n_eval_points, dtype=np.float64)

    try:
        bwd_sim = ode_solver(
            dynamics_rhs,
            t_span=(1, -1),
            y0=terminal_state,
            t_eval=bwd_tau_points,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_rtol * 1e-2,
        )
        bwd_success = bwd_sim.success
        bwd_trajectory = (
            np.fliplr(bwd_sim.y)
            if bwd_success
            else np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64)
        )
    except Exception as e:
        logger.error(f"Backward simulation failed: {e}")
        bwd_success = False
        bwd_trajectory = np.full((num_states, len(bwd_tau_points)), np.nan, dtype=np.float64)

    sorted_bwd_tau_points = np.flip(bwd_tau_points)
    bwd_nlp_trajectory = state_evaluator(sorted_bwd_tau_points)

    overall_success = fwd_success and bwd_success
    return (
        overall_success,
        fwd_tau_points,
        fwd_trajectory,
        fwd_nlp_trajectory,
        sorted_bwd_tau_points,
        bwd_trajectory,
        bwd_nlp_trajectory,
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
    """STREAMLINED: Direct parameters instead of complex bundle object."""
    # Check for failed simulations
    if not success or fwd_sim_traj.size == 0 or bwd_sim_traj.size == 0:
        logger.warning(
            f"Incomplete simulation results for phase {phase_id} interval {interval_idx}"
        )
        return np.inf

    num_states = fwd_sim_traj.shape[0]
    if num_states == 0:
        return 0.0

    # Forward errors using MATHEMATICAL CORE
    fwd_diff, max_fwd_errors = _calculate_trajectory_error_differences(
        fwd_sim_traj, fwd_nlp_traj, gamma_factors
    )

    # Backward errors using MATHEMATICAL CORE
    bwd_diff, max_bwd_errors = _calculate_trajectory_error_differences(
        bwd_sim_traj, bwd_nlp_traj, gamma_factors
    )

    # Combined error using MATHEMATICAL CORE
    max_error = _calculate_combined_error_estimate(max_fwd_errors, max_bwd_errors)

    if np.isnan(max_error):
        logger.warning(
            f"Error calculation resulted in NaN for phase {phase_id} interval {interval_idx}"
        )
        return np.inf

    return max_error


def calculate_gamma_normalizers_for_phase(
    solution: OptimalControlSolution, problem: ProblemProtocol, phase_id: PhaseID
) -> FloatArray | None:
    """Calculates gamma_i normalization factors for error estimation for a specific phase."""
    if not solution.success or solution.raw_solution is None:
        return None

    # Get variable counts for this phase
    num_states, _ = problem.get_phase_variable_counts(phase_id)
    if num_states == 0:
        return np.array([], dtype=np.float64).reshape(0, 1)

    # Get phase-specific state trajectories
    if phase_id not in solution.phase_solved_state_trajectories_per_interval:
        logger.warning(f"Missing solved state trajectories for phase {phase_id} gamma calculation")
        return None

    Y_solved_list = solution.phase_solved_state_trajectories_per_interval[phase_id]
    if not Y_solved_list:
        logger.warning(f"Empty solved state trajectories for phase {phase_id} gamma calculation")
        return None

    # Use MATHEMATICAL CORE for calculation
    max_abs_values = _find_maximum_state_values_across_phase_intervals(Y_solved_list)
    gamma_factors = _calculate_gamma_normalization_factors(max_abs_values)

    return gamma_factors
