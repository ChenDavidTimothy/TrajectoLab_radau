"""
Mesh refinement strategies for the PHS adaptive algorithm - SIMPLIFIED.
Updated to use unified storage system and type system.
"""

import logging
from typing import cast

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
    ControlEvaluator,
    FloatArray,
    GammaFactors,
    OptimalControlSolution,
    ProblemProtocol,
    StateEvaluator,
)
from trajectolab.utils.casadi_utils import CasadiFunction, convert_casadi_to_numpy
from trajectolab.utils.constants import DEFAULT_ODE_ATOL_FACTOR


__all__ = ["h_reduce_intervals", "h_refine_params", "p_reduce_interval", "p_refine_interval"]


logger = logging.getLogger(__name__)


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
    first_idx: int,
    solution: "OptimalControlSolution",
    problem: ProblemProtocol,
    adaptive_params: AdaptiveParameters,
    gamma_factors: GammaFactors,
    state_evaluator_first: StateEvaluator,
    control_evaluator_first: ControlEvaluator | None,
    state_evaluator_second: StateEvaluator,
    control_evaluator_second: ControlEvaluator | None,
) -> bool:
    """Check if two adjacent intervals can be merged."""

    # Log h-reduction attempt (DEBUG - developer info)
    logger.debug("Checking h-reduction for intervals %d and %d", first_idx, first_idx + 1)

    error_tol = adaptive_params.error_tolerance
    ode_rtol = adaptive_params.ode_solver_tolerance
    ode_atol = ode_rtol * DEFAULT_ODE_ATOL_FACTOR
    num_sim_points = adaptive_params.num_error_sim_points

    # Get variable counts from unified storage
    num_states, _ = problem.get_variable_counts()
    casadi_dynamics_function = cast(CasadiFunction, problem.get_dynamics_function())
    problem_parameters = problem._parameters

    if solution.raw_solution is None:
        logger.debug("h-reduction failed: Raw solution missing")
        return False

    # Extract mesh and time information
    global_mesh = solution.global_normalized_mesh_nodes
    if global_mesh is None:
        logger.debug("h-reduction failed: Global mesh is None")
        return False

    tau_start_k = global_mesh[first_idx]
    tau_shared = global_mesh[first_idx + 1]
    tau_end_kp1 = global_mesh[first_idx + 2]

    beta_k = (tau_shared - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        logger.debug("h-reduction check: One interval has zero length. Merge not possible")
        return False

    # Time transformation parameters
    t0 = solution.initial_time_variable
    tf = solution.terminal_time_variable

    # Check for None values before arithmetic operations
    if t0 is None or tf is None:
        print("      h-reduction failed: Initial or terminal time is None.")
        return False

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    def _get_control_value(
        control_evaluator: ControlEvaluator | None, local_tau: float
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
            cast(CasadiFunction, casadi_dynamics_function),
            state_clipped,
            u_val,
            t_actual,
            problem_parameters,
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
        f_rhs_np = convert_casadi_to_numpy(
            casadi_dynamics_function, state_clipped, u_val, t_actual, problem_parameters
        )
        return cast(FloatArray, scaling_kp1 * f_rhs_np)

    # Get state values at interval endpoints
    try:
        Y_solved_list = solution.solved_state_trajectories_per_interval
        Nk_k = problem.collocation_points_per_interval[first_idx]
        Nk_kp1 = problem.collocation_points_per_interval[first_idx + 1]

        if Y_solved_list and first_idx < len(Y_solved_list):
            Xk_nlp = Y_solved_list[first_idx]
        else:  # Fallback
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if opti is None or raw_sol is None:
                print("      h-reduction failed: opti or raw_sol is None.")
                return False

            Xk_nlp_raw = raw_sol.value(
                opti.state_at_local_approximation_nodes_all_intervals_variables[first_idx]
            )
            Xk_nlp = extract_and_prepare_array(Xk_nlp_raw, num_states, Nk_k + 1)

        initial_state_fwd = Xk_nlp[:, 0].flatten()
    except Exception as e:
        print(f"      h-reduction failed: Error getting initial state: {e}")
        return False

    # Forward simulation through merged domain
    target_end_tau_k = map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_fwd_pts = max(2, num_sim_points // 2)
    fwd_tau_points = np.linspace(-1.0, target_end_tau_k, num_fwd_pts, dtype=np.float64)

    print(
        f"      h-reduction: Starting Merged IVP sim from zeta_k=-1 to {target_end_tau_k:.3f} ({num_fwd_pts} pts)"
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
            print(f"      Merged IVP failed: {fwd_sim.message}")
    except Exception as e:
        print(f"      Exception during merged IVP simulation: {e}")

    # Get terminal state for backward simulation
    try:
        if Y_solved_list is not None and (first_idx + 1) < len(Y_solved_list):
            Xkp1_nlp = Y_solved_list[first_idx + 1]
        else:  # Fallback
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if opti is None or raw_sol is None:
                print("      h-reduction failed: opti or raw_sol is None.")
                return False

            Xkp1_nlp_raw = raw_sol.value(
                opti.state_at_local_approximation_nodes_all_intervals_variables[first_idx + 1]
            )
            Xkp1_nlp = extract_and_prepare_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)

        terminal_state_bwd = Xkp1_nlp[:, -1].flatten()
    except Exception as e:
        print(f"      h-reduction failed: Error getting terminal state: {e}")
        return False

    # Backward simulation through merged domain
    target_end_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_bwd_pts = max(2, num_sim_points // 2)
    bwd_tau_points = np.linspace(1.0, target_end_tau_kp1, num_bwd_pts, dtype=np.float64)
    sorted_bwd_tau_points = np.flip(bwd_tau_points)

    print(
        f"      h-reduction: Starting Merged TVP sim from zeta_kp1=1 to {target_end_tau_kp1:.3f} ({num_bwd_pts} pts)"
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
            print(f"      Merged TVP failed: {bwd_sim.message}")
    except Exception as e:
        print(f"      Exception during merged TVP simulation: {e}")

    # For problems with no states, just check if simulations were successful
    if num_states == 0:
        can_merge = fwd_sim_success and bwd_sim_success
        logger.debug("h-reduction check (no states): can_intervals_be_merged=%s", can_merge)
        return can_merge

    # Calculate errors for merged domain
    all_fwd_errors: list[float] = []
    for i, zeta_k in enumerate(fwd_tau_points):
        X_sim = fwd_trajectory[:, i]
        if np.any(np.isnan(X_sim)):
            continue

        # Get NLP state for comparison
        if -1.0 <= zeta_k <= 1.0 + 1e-9:
            X_nlp = state_evaluator_first(zeta_k)
        else:
            zeta_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                zeta_k, tau_start_k, tau_shared, tau_end_kp1
            )
            X_nlp = state_evaluator_second(zeta_kp1)

        if X_nlp.ndim > 1:
            X_nlp = X_nlp.flatten()

        abs_diff = np.abs(X_sim - X_nlp)
        scaled_errors = gamma_factors.flatten() * abs_diff
        all_fwd_errors.extend(scaled_errors.tolist())

    all_bwd_errors: list[float] = []
    for i, zeta_kp1 in enumerate(sorted_bwd_tau_points):
        X_sim = bwd_trajectory[:, i]
        if np.any(np.isnan(X_sim)):
            continue

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

        if X_nlp.ndim > 1:
            X_nlp = X_nlp.flatten()

        abs_diff = np.abs(X_sim - X_nlp)
        scaled_errors = gamma_factors.flatten() * abs_diff
        all_bwd_errors.extend(scaled_errors.tolist())

    # Get maximum error for merged domain
    max_fwd_error = np.nanmax(all_fwd_errors) if all_fwd_errors else np.inf
    max_bwd_error = np.nanmax(all_bwd_errors) if all_bwd_errors else np.inf
    max_error = max(max_fwd_error, max_bwd_error)

    if np.isnan(max_error):
        print("      h-reduction check: max_error calculation resulted in NaN. Merge not approved.")
        max_error = np.inf

    can_merge = max_error <= error_tol

    logger.debug("h-reduction result: max_error=%.4e, can_merge=%s", max_error, can_merge)

    if can_merge:
        logger.debug("h-reduction approved for intervals %d, %d", first_idx, first_idx + 1)
    else:
        logger.debug("h-reduction rejected: error %.2e > tolerance %.2e", max_error, error_tol)

    return can_merge
