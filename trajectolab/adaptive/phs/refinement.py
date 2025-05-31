"""
Mesh refinement strategies including p-refinement, h-refinement, and reduction for multiphase problems.
"""

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
    extract_and_prepare_array,
)
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
from trajectolab.utils.casadi_utils import convert_casadi_to_numpy
from trajectolab.utils.constants import DEFAULT_ODE_ATOL_FACTOR


__all__ = ["h_reduce_intervals", "h_refine_params", "p_reduce_interval", "p_refine_interval"]


logger = logging.getLogger(__name__)


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
# ORCHESTRATION FUNCTIONS - Setup and coordination
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

    # Log h-reduction attempt
    logger.debug(
        "Checking h-reduction for phase %d intervals %d and %d", phase_id, first_idx, first_idx + 1
    )

    error_tol = adaptive_params.error_tolerance
    ode_rtol = adaptive_params.ode_solver_tolerance
    ode_atol = ode_rtol * DEFAULT_ODE_ATOL_FACTOR
    num_sim_points = adaptive_params.num_error_sim_points

    # Get variable counts for this phase
    num_states, _ = problem.get_phase_variable_counts(phase_id)
    phase_dynamics_function = cast(ca.Function, problem.get_phase_dynamics_function(phase_id))

    if solution.raw_solution is None:
        logger.debug("h-reduction failed: Raw solution missing")
        return False

    # Extract mesh and time information for this phase
    if phase_id not in solution.phase_mesh_nodes:
        logger.debug("h-reduction failed: Phase %d mesh is None", phase_id)
        return False

    global_mesh = solution.phase_mesh_nodes[phase_id]
    if first_idx + 2 >= len(global_mesh):
        logger.debug("h-reduction failed: Phase %d insufficient mesh points", phase_id)
        return False

    tau_start_k = global_mesh[first_idx]
    tau_shared = global_mesh[first_idx + 1]
    tau_end_kp1 = global_mesh[first_idx + 2]

    beta_k = (tau_shared - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        logger.debug(
            "h-reduction check: Phase %d one interval has zero length. Merge not possible", phase_id
        )
        return False

    # Time transformation parameters for this phase
    if (
        phase_id not in solution.phase_initial_times
        or phase_id not in solution.phase_terminal_times
    ):
        logger.warning(
            "h-reduction failed: Initial or terminal time missing for phase %d", phase_id
        )
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

        # Use consolidated conversion function
        f_rhs_np = convert_casadi_to_numpy(
            cast(ca.Function, phase_dynamics_function),
            state_clipped,
            u_val,
            t_actual,
        )
        return cast(FloatArray, scaling_k * f_rhs_np)

    def merged_bwd_rhs(local_tau_kp1: float, state: FloatArray) -> FloatArray:
        """RHS for merged domain backward simulation."""
        u_val = _get_control_value(control_evaluator_second, local_tau_kp1)
        state_clipped = np.clip(state, -1e6, 1e6)
        global_tau = map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1, tau_shared, tau_end_kp1
        )
        t_actual = alpha * global_tau + alpha_0

        # Use consolidated conversion function
        f_rhs_np = convert_casadi_to_numpy(phase_dynamics_function, state_clipped, u_val, t_actual)
        return cast(FloatArray, scaling_kp1 * f_rhs_np)

    # Get state values at interval endpoints for this phase
    try:
        if phase_id in solution.phase_solved_state_trajectories_per_interval and first_idx < len(
            solution.phase_solved_state_trajectories_per_interval[phase_id]
        ):
            Xk_nlp = solution.phase_solved_state_trajectories_per_interval[phase_id][first_idx]
        else:
            # Fallback to raw extraction
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if opti is None or raw_sol is None:
                logger.warning("h-reduction failed: Optimization object or raw solution is None")
                return False

            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables or first_idx >= len(
                variables.phase_variables[phase_id].state_matrices
            ):
                logger.warning(
                    "h-reduction failed: Missing state matrix for phase %d interval %d",
                    phase_id,
                    first_idx,
                )
                return False

            phase_def = problem._phases[phase_id]
            Nk_k = phase_def.collocation_points_per_interval[first_idx]

            Xk_nlp_raw = raw_sol.value(
                variables.phase_variables[phase_id].state_matrices[first_idx]
            )
            Xk_nlp = extract_and_prepare_array(Xk_nlp_raw, num_states, Nk_k + 1)

        initial_state_fwd = Xk_nlp[:, 0].flatten()
    except Exception as e:
        logger.warning(
            "h-reduction failed: Error getting initial state for phase %d: %s", phase_id, str(e)
        )
        return False

    # Forward simulation through merged domain
    target_end_tau_k = map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_fwd_pts = max(2, num_sim_points // 2)
    fwd_tau_points = np.linspace(-1.0, target_end_tau_k, num_fwd_pts, dtype=np.float64)

    logger.info(
        "h-reduction: Starting merged forward simulation for phase %d from zeta_k=-1 to %.3f (%d points)",
        phase_id,
        target_end_tau_k,
        num_fwd_pts,
    )

    # Import here to avoid circular imports
    from scipy.integrate import solve_ivp

    # Define these upfront to ensure they're bound in all code paths
    fwd_trajectory = np.full((num_states, num_fwd_pts), np.nan, dtype=np.float64)
    fwd_sim_success = False

    try:
        fwd_sim = solve_ivp(
            merged_fwd_rhs,
            t_span=(-1.0, target_end_tau_k),
            y0=initial_state_fwd,
            t_eval=fwd_tau_points,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )

        if fwd_sim.success:
            fwd_trajectory = fwd_sim.y
            fwd_sim_success = True
        else:
            logger.warning(
                "Phase %d merged forward simulation failed: %s", phase_id, fwd_sim.message
            )
    except Exception as e:
        logger.warning("Exception during phase %d merged forward simulation: %s", phase_id, str(e))

    # Get terminal state for backward simulation
    try:
        if phase_id in solution.phase_solved_state_trajectories_per_interval and (
            first_idx + 1
        ) < len(solution.phase_solved_state_trajectories_per_interval[phase_id]):
            Xkp1_nlp = solution.phase_solved_state_trajectories_per_interval[phase_id][
                first_idx + 1
            ]
        else:
            # Fallback to raw extraction
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if opti is None or raw_sol is None:
                logger.warning("h-reduction failed: Optimization object or raw solution is None")
                return False

            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables or (first_idx + 1) >= len(
                variables.phase_variables[phase_id].state_matrices
            ):
                logger.warning(
                    "h-reduction failed: Missing state matrix for phase %d interval %d",
                    phase_id,
                    first_idx + 1,
                )
                return False

            phase_def = problem._phases[phase_id]
            Nk_kp1 = phase_def.collocation_points_per_interval[first_idx + 1]

            Xkp1_nlp_raw = raw_sol.value(
                variables.phase_variables[phase_id].state_matrices[first_idx + 1]
            )
            Xkp1_nlp = extract_and_prepare_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)

        terminal_state_bwd = Xkp1_nlp[:, -1].flatten()
    except Exception as e:
        logger.warning(
            "h-reduction failed: Error getting terminal state for phase %d: %s", phase_id, str(e)
        )
        return False

    # Backward simulation through merged domain
    target_end_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_bwd_pts = max(2, num_sim_points // 2)
    bwd_tau_points = np.linspace(1.0, target_end_tau_kp1, num_bwd_pts, dtype=np.float64)
    sorted_bwd_tau_points = np.flip(bwd_tau_points)

    logger.info(
        "h-reduction: Starting merged backward simulation for phase %d from zeta_kp1=1 to %.3f (%d points)",
        phase_id,
        target_end_tau_kp1,
        num_bwd_pts,
    )

    # Define these upfront to ensure they're bound in all code paths
    bwd_trajectory = np.full((num_states, num_bwd_pts), np.nan, dtype=np.float64)
    bwd_sim_success = False

    try:
        bwd_sim = solve_ivp(
            merged_bwd_rhs,
            t_span=(1.0, target_end_tau_kp1),
            y0=terminal_state_bwd,
            t_eval=bwd_tau_points,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )

        if bwd_sim.success:
            # Create a new array with explicit 2D shape
            flipped_data = bwd_sim.y[:, ::-1]
            # First ensure we have the right dimensions by accessing shape components
            rows, cols = flipped_data.shape[0], flipped_data.shape[1]
            # Then create a new array with explicit type
            bwd_trajectory = np.array(flipped_data, dtype=np.float64).reshape(rows, cols)
            bwd_sim_success = True
        else:
            logger.warning(
                "Phase %d merged backward simulation failed: %s", phase_id, bwd_sim.message
            )
    except Exception as e:
        logger.warning("Exception during phase %d merged backward simulation: %s", phase_id, str(e))

    # For problems with no states, just check if simulations were successful
    if num_states == 0:
        can_merge = fwd_sim_success and bwd_sim_success
        logger.debug(
            "h-reduction check for phase %d (no states): can_intervals_be_merged=%s",
            phase_id,
            can_merge,
        )
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

    logger.debug(
        "h-reduction result for phase %d: max_error=%.4e, can_merge=%s",
        phase_id,
        max_error,
        can_merge,
    )

    if can_merge:
        logger.debug(
            "h-reduction approved for phase %d intervals %d, %d", phase_id, first_idx, first_idx + 1
        )
    else:
        logger.debug(
            "h-reduction rejected for phase %d: error %.2e > tolerance %.2e",
            phase_id,
            max_error,
            error_tol,
        )

    return can_merge
