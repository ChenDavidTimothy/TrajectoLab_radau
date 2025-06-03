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

# Import from error_estimation instead of duplicate function
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


# ========================================================================
# MATHEMATICAL CORE FUNCTIONS - Pure calculations (UNCHANGED)
# ========================================================================


def _calculate_merge_feasibility_from_errors(
    all_fwd_errors: list[float], all_bwd_errors: list[float], error_tol: float
) -> tuple[bool, float]:
    """
    Determines if two intervals can be merged based on their error estimates.
    A merge is not feasible if any error component is NaN.
    """
    np_fwd_errors = (
        np.array(all_fwd_errors, dtype=np.float64)
        if all_fwd_errors
        else np.array([], dtype=np.float64)
    )
    np_bwd_errors = (
        np.array(all_bwd_errors, dtype=np.float64)
        if all_bwd_errors
        else np.array([], dtype=np.float64)
    )

    # Handle case where both error lists are actually empty
    if not np_fwd_errors.size and not np_bwd_errors.size:
        return False, np.inf

    if np.any(np.isnan(np_fwd_errors)) or np.any(np.isnan(np_bwd_errors)):
        return False, np.inf

    # Proceed if no NaNs are present
    max_fwd_error = np.max(np_fwd_errors) if np_fwd_errors.size > 0 else 0.0
    max_bwd_error = np.max(np_bwd_errors) if np_bwd_errors.size > 0 else 0.0

    max_error = max(max_fwd_error, max_bwd_error)

    if max_error == np.inf:
        can_merge = False
    else:
        can_merge = max_error <= error_tol

    return bool(can_merge), float(max_error)


def _calculate_trajectory_errors_with_gamma(
    X_sim: FloatArray, X_nlp: FloatArray, gamma_factors: FloatArray
) -> list[float]:
    """Pure trajectory error calculation - easily testable."""
    if np.any(np.isnan(X_sim)):
        return []

    if X_nlp.ndim > 1:
        X_nlp = X_nlp.flatten()

    abs_diff = np.abs(X_sim - X_nlp)
    scaled_errors = gamma_factors.flatten() * abs_diff
    return list(scaled_errors)


# ========================================================================
# REFINEMENT DECISION FUNCTIONS - Pure mathematical logic (UNCHANGED)
# ========================================================================


def p_refine_interval(
    max_error: float, current_Nk: int, error_tol: float, N_max: int
) -> PRefineResult:
    """Determines new polynomial degree using p-refinement."""
    # No refinement needed if error is within tolerance
    if max_error <= error_tol:
        return PRefineResult(
            actual_Nk_to_use=current_Nk, was_p_successful=False, unconstrained_target_Nk=current_Nk
        )

    # Calculate number of nodes to add
    if np.isinf(max_error):
        nodes_to_add = max(1, N_max - current_Nk)
    else:
        ratio = max_error / error_tol
        nodes_to_add = max(1, int(np.ceil(np.log10(ratio))))

    target_Nk = current_Nk + nodes_to_add

    # Check if target exceeds maximum allowable
    if target_Nk > N_max:
        return PRefineResult(
            actual_Nk_to_use=N_max, was_p_successful=False, unconstrained_target_Nk=target_Nk
        )

    return PRefineResult(
        actual_Nk_to_use=target_Nk, was_p_successful=True, unconstrained_target_Nk=target_Nk
    )


def h_refine_params(target_Nk: int, N_min: int) -> HRefineResult:
    """Determines parameters for h-refinement (splitting an interval)."""
    num_subintervals = max(2, int(np.ceil(target_Nk / N_min)))
    nodes_per_subinterval = [N_min] * num_subintervals

    return HRefineResult(
        collocation_nodes_for_new_subintervals=nodes_per_subinterval,
        num_new_subintervals=num_subintervals,
    )


def p_reduce_interval(
    current_Nk: int, max_error: float, error_tol: float, N_min: int, N_max: int
) -> PReduceResult:
    """
    Determines new polynomial degree using p-reduction per Eq. 36.
    SAFETY-CRITICAL: Always uses mathematical formula from specification.
    """
    # Only reduce if error is below tolerance and current Nk > minimum
    if max_error > error_tol or current_Nk <= N_min:
        return PReduceResult(new_num_collocation_nodes=current_Nk, was_reduction_applied=False)

    # Calculate reduction control parameter delta per specification
    delta = float(N_min + N_max - current_Nk)
    if abs(delta) < 1e-9:
        delta = 1.0  # Avoid division by zero

    # CRITICAL: Always use Eq. 36 mathematical formula - no special cases
    try:
        ratio = error_tol / max_error  # Should be >= 1 since max_error <= error_tol
        if ratio < 1.0:
            # Mathematical inconsistency - should not happen
            nodes_to_remove = 0
        else:
            # Apply Eq. 36: P_k^- = floor(log10((ε/e_max^(k))^(1/δ)))
            power_arg = np.power(ratio, 1.0 / delta)
            if power_arg >= 1.0:
                nodes_to_remove = int(np.floor(np.log10(power_arg)))
            else:
                nodes_to_remove = 0

    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        # Mathematical operation failed - conservative fallback
        nodes_to_remove = 0

    # Apply reduction with bounds checking
    nodes_to_remove = max(0, nodes_to_remove)
    new_Nk = max(N_min, current_Nk - nodes_to_remove)
    was_reduced = new_Nk < current_Nk

    return PReduceResult(new_num_collocation_nodes=new_Nk, was_reduction_applied=was_reduced)


# ========================================================================
#  H-REDUCTION IMPLEMENTATION
# ========================================================================


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
    """Check if two adjacent intervals in a specific phase can be merged."""

    error_tol = adaptive_params.error_tolerance
    num_sim_points = adaptive_params.num_error_sim_points

    # Get variable counts for this phase
    num_states, _ = problem.get_phase_variable_counts(phase_id)
    phase_dynamics_function = problem.get_phase_dynamics_function(phase_id)

    if solution.raw_solution is None:
        return False

    # Extract mesh and time information for this phase
    if phase_id not in solution.phase_mesh_nodes:
        return False

    global_mesh = solution.phase_mesh_nodes[phase_id]
    if first_idx + 2 >= len(global_mesh):
        return False

    tau_start_k = global_mesh[first_idx]
    tau_shared = global_mesh[first_idx + 1]
    tau_end_kp1 = global_mesh[first_idx + 2]

    beta_k = (tau_shared - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        return False

    # Time transformation parameters for this phase
    if (
        phase_id not in solution.phase_initial_times
        or phase_id not in solution.phase_terminal_times
    ):
        return False

    t0 = solution.phase_initial_times[phase_id]
    tf = solution.phase_terminal_times[phase_id]

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    def _get_control_value(
        control_evaluator: Callable[[float | FloatArray], FloatArray] | None, local_tau: float
    ) -> FloatArray:
        """Get control value from evaluator, with clipping to handle boundary conditions."""
        if control_evaluator is None:
            return np.array([], dtype=np.float64)

        clipped_tau = np.clip(local_tau, -1.0, 1.0)
        u_val = control_evaluator(clipped_tau)
        return np.atleast_1d(u_val.squeeze())

    def merged_fwd_rhs(local_tau_k: float, state: FloatArray) -> FloatArray:
        """RHS for merged domain forward simulation."""
        u_val = _get_control_value(control_evaluator_first, local_tau_k)
        state_clipped = np.clip(state, -1e6, 1e6)
        global_tau = map_local_interval_tau_to_global_normalized_tau(
            local_tau_k, tau_start_k, tau_shared
        )
        t_actual = alpha * global_tau + alpha_0

        # Handle optimized dynamics interface
        dynamics_result = phase_dynamics_function(
            ca.MX(state_clipped), ca.MX(u_val), ca.MX(t_actual)
        )

        # Use shared function from error_estimation
        f_rhs_np = _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states)

        return cast(FloatArray, scaling_k * f_rhs_np)

    def merged_bwd_rhs(local_tau_kp1: float, state: FloatArray) -> FloatArray:
        """RHS for merged domain backward simulation."""
        u_val = _get_control_value(control_evaluator_second, local_tau_kp1)
        state_clipped = np.clip(state, -1e6, 1e6)
        global_tau = map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1, tau_shared, tau_end_kp1
        )
        t_actual = alpha * global_tau + alpha_0

        # Handle optimized dynamics interface
        dynamics_result = phase_dynamics_function(
            ca.MX(state_clipped), ca.MX(u_val), ca.MX(t_actual)
        )

        # Use shared function from error_estimation
        f_rhs_np = _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states)

        return cast(FloatArray, scaling_kp1 * f_rhs_np)

    # Get state values at interval endpoints for this phase
    try:
        if phase_id in solution.phase_solved_state_trajectories_per_interval and first_idx < len(
            solution.phase_solved_state_trajectories_per_interval[phase_id]
        ):
            Xk_nlp = solution.phase_solved_state_trajectories_per_interval[phase_id][first_idx]
        else:
            #  Fallback to raw extraction with less validation
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if opti is None or raw_sol is None:
                return False

            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables or first_idx >= len(
                variables.phase_variables[phase_id].state_matrices
            ):
                return False

            phase_def = problem._phases[phase_id]
            Nk_k = phase_def.collocation_points_per_interval[first_idx]

            Xk_nlp_raw = raw_sol.value(
                variables.phase_variables[phase_id].state_matrices[first_idx]
            )
            Xk_nlp = ensure_2d_array(Xk_nlp_raw, num_states, Nk_k + 1)

        initial_state_fwd = Xk_nlp[:, 0].flatten()
    except Exception:
        return False

    # Forward simulation through merged domain
    target_end_tau_k = map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_fwd_pts = max(2, num_sim_points // 2)
    fwd_tau_points = np.linspace(-1.0, target_end_tau_k, num_fwd_pts, dtype=np.float64)

    # Use configured ODE solver from adaptive_params
    configured_ode_solver = adaptive_params.get_ode_solver()

    # Define these upfront to ensure they're bound in all code paths
    fwd_trajectory = np.full((num_states, num_fwd_pts), np.nan, dtype=np.float64)
    fwd_sim_success = False

    # Let scipy configuration errors bubble up immediately, only catch numerical convergence issues
    try:
        fwd_sim = configured_ode_solver(
            merged_fwd_rhs,
            t_span=(-1.0, target_end_tau_k),
            y0=initial_state_fwd,
            t_eval=fwd_tau_points,
        )

        if fwd_sim.success:
            fwd_trajectory = fwd_sim.y
            fwd_sim_success = True
    except (RuntimeError, OverflowError, FloatingPointError) as e:
        # Only catch numerical convergence/overflow issues, let configuration errors bubble up
        logger.debug(
            f"Forward simulation numerical failure for phase {phase_id} intervals {first_idx}-{first_idx + 1}: {e}"
        )

    # Get terminal state for backward simulation
    try:
        if phase_id in solution.phase_solved_state_trajectories_per_interval and (
            first_idx + 1
        ) < len(solution.phase_solved_state_trajectories_per_interval[phase_id]):
            Xkp1_nlp = solution.phase_solved_state_trajectories_per_interval[phase_id][
                first_idx + 1
            ]
        else:
            #  Fallback to raw extraction with less validation
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if opti is None or raw_sol is None:
                return False

            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables or (first_idx + 1) >= len(
                variables.phase_variables[phase_id].state_matrices
            ):
                return False

            phase_def = problem._phases[phase_id]
            Nk_kp1 = phase_def.collocation_points_per_interval[first_idx + 1]

            Xkp1_nlp_raw = raw_sol.value(
                variables.phase_variables[phase_id].state_matrices[first_idx + 1]
            )
            Xkp1_nlp = ensure_2d_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)

        terminal_state_bwd = Xkp1_nlp[:, -1].flatten()
    except Exception:
        return False

    # Backward simulation through merged domain
    target_end_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_bwd_pts = max(2, num_sim_points // 2)
    bwd_tau_points = np.linspace(1.0, target_end_tau_kp1, num_bwd_pts, dtype=np.float64)
    sorted_bwd_tau_points = np.flip(bwd_tau_points)

    # Define these upfront to ensure they're bound in all code paths
    bwd_trajectory = np.full((num_states, num_bwd_pts), np.nan, dtype=np.float64)
    bwd_sim_success = False

    # Let scipy configuration errors bubble up immediately, only catch numerical convergence issues
    try:
        bwd_sim = configured_ode_solver(
            merged_bwd_rhs,
            t_span=(1.0, target_end_tau_kp1),
            y0=terminal_state_bwd,
            t_eval=bwd_tau_points,
        )

        if bwd_sim.success:
            # Create a new array with explicit 2D shape
            flipped_data = bwd_sim.y[:, ::-1]
            # First ensure we have the right dimensions by accessing shape components
            rows, cols = flipped_data.shape[0], flipped_data.shape[1]
            # Then create a new array with explicit type
            bwd_trajectory = np.array(flipped_data, dtype=np.float64).reshape(rows, cols)
            bwd_sim_success = True
    except (RuntimeError, OverflowError, FloatingPointError) as e:
        # Only catch numerical convergence/overflow issues, let configuration errors bubble up
        logger.debug(
            f"Backward simulation numerical failure for phase {phase_id} intervals {first_idx}-{first_idx + 1}: {e}"
        )

    # For problems with no states, just check if simulations were successful
    if num_states == 0:
        can_merge = fwd_sim_success and bwd_sim_success
        return can_merge

    # Calculate errors for merged domain using MATHEMATICAL CORE
    all_fwd_errors: list[float] = []
    for i, zeta_k in enumerate(fwd_tau_points):
        X_sim = fwd_trajectory[:, i]

        # Get NLP state for comparison
        if -1.0 <= zeta_k <= 1.0 + 1e-9:
            X_nlp = state_evaluator_first(zeta_k)
        else:
            zeta_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                zeta_k, tau_start_k, tau_shared, tau_end_kp1
            )
            X_nlp = state_evaluator_second(zeta_kp1)

        # Use mathematical core function for error calculation
        trajectory_errors = _calculate_trajectory_errors_with_gamma(X_sim, X_nlp, gamma_factors)
        all_fwd_errors.extend(trajectory_errors)

    all_bwd_errors: list[float] = []
    for i, zeta_kp1 in enumerate(sorted_bwd_tau_points):
        X_sim = bwd_trajectory[:, i]

        # Get NLP state for comparison
        if -1.0 - 1e-9 <= zeta_kp1 <= 1.0:
            X_nlp = state_evaluator_second(zeta_kp1)
        else:
            zeta_k = np.float64(
                map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
                    zeta_kp1, tau_start_k, tau_shared, tau_end_kp1
                )
            )
            X_nlp = state_evaluator_first(zeta_k)

        # Use mathematical core function for error calculation
        trajectory_errors = _calculate_trajectory_errors_with_gamma(X_sim, X_nlp, gamma_factors)
        all_bwd_errors.extend(trajectory_errors)

    # Use mathematical core function for merge decision
    can_merge, max_error = _calculate_merge_feasibility_from_errors(
        all_fwd_errors, all_bwd_errors, error_tol
    )

    return can_merge
