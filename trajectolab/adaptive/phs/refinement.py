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
from trajectolab.utils.constants import DEFAULT_ODE_ATOL_FACTOR


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
# NEVER-NESTER REFACTORED H-REDUCTION IMPLEMENTATION
# ========================================================================


def _validate_h_reduction_inputs(
    solution: OptimalControlSolution, phase_id: PhaseID, first_idx: int
) -> tuple[FloatArray, float, float, float, float, float, float]:
    """EXTRACTED: Validate inputs and extract basic parameters."""
    # INVERSION: Early return checks
    if solution.raw_solution is None:
        raise ValueError("Raw solution is None")

    if phase_id not in solution.phase_mesh_nodes:
        raise ValueError(f"Phase {phase_id} not in mesh nodes")

    if (
        phase_id not in solution.phase_initial_times
        or phase_id not in solution.phase_terminal_times
    ):
        raise ValueError(f"Missing time variables for phase {phase_id}")

    # Extract mesh and time information
    global_mesh = solution.phase_mesh_nodes[phase_id]
    if first_idx + 2 >= len(global_mesh):
        raise ValueError(f"Invalid interval index {first_idx} for phase {phase_id}")

    tau_start_k = global_mesh[first_idx]
    tau_shared = global_mesh[first_idx + 1]
    tau_end_kp1 = global_mesh[first_idx + 2]

    beta_k = (tau_shared - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        raise ValueError(f"Zero length interval for phase {phase_id}")

    # Time transformation parameters
    t0 = solution.phase_initial_times[phase_id]
    tf = solution.phase_terminal_times[phase_id]
    alpha = (tf - t0) / 2.0
    (tf + t0) / 2.0

    return global_mesh, tau_start_k, tau_shared, tau_end_kp1, beta_k, beta_kp1, alpha


def _get_control_value(
    control_evaluator: Callable[[float | FloatArray], FloatArray] | None, local_tau: float
) -> FloatArray:
    """EXTRACTED: Get control value from evaluator with boundary handling."""
    if control_evaluator is None:
        return np.array([], dtype=np.float64)

    clipped_tau = np.clip(local_tau, -1.0, 1.0)
    u_val = control_evaluator(clipped_tau)
    return np.atleast_1d(u_val.squeeze())


def _create_merged_dynamics_rhs(
    phase_dynamics_function,
    control_evaluator: Callable[[float | FloatArray], FloatArray] | None,
    tau_start: float,
    tau_end: float,
    alpha: float,
    alpha_0: float,
    scaling: float,
    num_states: int,
) -> Callable[[float, FloatArray], FloatArray]:
    """EXTRACTED: Create RHS function for merged domain simulation."""

    def dynamics_rhs(local_tau: float, state: FloatArray) -> FloatArray:
        u_val = _get_control_value(control_evaluator, local_tau)
        state_clipped = np.clip(state, -1e6, 1e6)
        global_tau = map_local_interval_tau_to_global_normalized_tau(local_tau, tau_start, tau_end)
        t_actual = alpha * global_tau + alpha_0

        # Handle optimized dynamics interface
        dynamics_result = phase_dynamics_function(
            ca.MX(state_clipped), ca.MX(u_val), ca.MX(t_actual)
        )

        # Use shared function from error_estimation
        f_rhs_np = _convert_casadi_dynamics_result_to_numpy(dynamics_result, num_states)

        return cast(FloatArray, scaling * f_rhs_np)

    return dynamics_rhs


def _run_forward_simulation(
    merged_fwd_rhs: Callable[[float, FloatArray], FloatArray],
    initial_state_fwd: FloatArray,
    target_end_tau_k: float,
    num_sim_points: int,
    ode_rtol: float,
    ode_atol: float,
    phase_id: PhaseID,
    first_idx: int,
) -> tuple[FloatArray, bool]:
    """EXTRACTED: Run forward simulation with error handling."""
    num_fwd_pts = max(2, num_sim_points // 2)
    fwd_tau_points = np.linspace(-1.0, target_end_tau_k, num_fwd_pts, dtype=np.float64)

    # Import here to avoid circular imports
    from scipy.integrate import solve_ivp

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
            return fwd_sim.y, True
    except Exception as e:
        logger.debug(
            f"Forward simulation failed for phase {phase_id} intervals {first_idx}-{first_idx + 1}: {e}"
        )

    # Return NaN array on failure
    num_states = len(initial_state_fwd)
    return np.full((num_states, num_fwd_pts), np.nan, dtype=np.float64), False


def _run_backward_simulation(
    merged_bwd_rhs: Callable[[float, FloatArray], FloatArray],
    terminal_state_bwd: FloatArray,
    target_end_tau_kp1: float,
    num_sim_points: int,
    ode_rtol: float,
    ode_atol: float,
    phase_id: PhaseID,
    first_idx: int,
) -> tuple[FloatArray, bool]:
    """EXTRACTED: Run backward simulation with error handling."""
    num_bwd_pts = max(2, num_sim_points // 2)
    bwd_tau_points = np.linspace(1.0, target_end_tau_kp1, num_bwd_pts, dtype=np.float64)

    # Import here to avoid circular imports
    from scipy.integrate import solve_ivp

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
            rows, cols = flipped_data.shape[0], flipped_data.shape[1]
            bwd_trajectory = np.array(flipped_data, dtype=np.float64).reshape(rows, cols)
            return bwd_trajectory, True
    except Exception as e:
        logger.debug(
            f"Backward simulation failed for phase {phase_id} intervals {first_idx}-{first_idx + 1}: {e}"
        )

    # Return NaN array on failure
    num_states = len(terminal_state_bwd)
    return np.full((num_states, num_bwd_pts), np.nan, dtype=np.float64), False


def _extract_initial_and_terminal_states(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    first_idx: int,
    num_states: int,
) -> tuple[FloatArray, FloatArray]:
    """EXTRACTED: Extract initial and terminal states for the intervals."""
    # Get initial state for first interval
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
                raise ValueError("Missing opti or raw solution")

            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables or first_idx >= len(
                variables.phase_variables[phase_id].state_matrices
            ):
                raise ValueError("Invalid phase or interval index")

            phase_def = problem._phases[phase_id]
            Nk_k = phase_def.collocation_points_per_interval[first_idx]

            Xk_nlp_raw = raw_sol.value(
                variables.phase_variables[phase_id].state_matrices[first_idx]
            )
            Xk_nlp = ensure_2d_array(Xk_nlp_raw, num_states, Nk_k + 1)

        initial_state_fwd = Xk_nlp[:, 0].flatten()
    except Exception as e:
        logger.error(f"Failed to extract initial state: {e}")
        raise

    # Get terminal state for second interval
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
                raise ValueError("Missing opti or raw solution")

            variables = opti.multiphase_variables_reference
            if phase_id not in variables.phase_variables or (first_idx + 1) >= len(
                variables.phase_variables[phase_id].state_matrices
            ):
                raise ValueError("Invalid phase or interval index")

            phase_def = problem._phases[phase_id]
            Nk_kp1 = phase_def.collocation_points_per_interval[first_idx + 1]

            Xkp1_nlp_raw = raw_sol.value(
                variables.phase_variables[phase_id].state_matrices[first_idx + 1]
            )
            Xkp1_nlp = ensure_2d_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)

        terminal_state_bwd = Xkp1_nlp[:, -1].flatten()
    except Exception as e:
        logger.error(f"Failed to extract terminal state: {e}")
        raise

    return initial_state_fwd, terminal_state_bwd


def _calculate_forward_errors(
    fwd_trajectory: FloatArray,
    fwd_tau_points: FloatArray,
    state_evaluator_first: Callable[[float | FloatArray], FloatArray],
    state_evaluator_second: Callable[[float | FloatArray], FloatArray],
    gamma_factors: FloatArray,
    tau_start_k: float,
    tau_shared: float,
    tau_end_kp1: float,
) -> list[float]:
    """EXTRACTED: Calculate forward trajectory errors."""
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

    return all_fwd_errors


def _calculate_backward_errors(
    bwd_trajectory: FloatArray,
    sorted_bwd_tau_points: FloatArray,
    state_evaluator_first: Callable[[float | FloatArray], FloatArray],
    state_evaluator_second: Callable[[float | FloatArray], FloatArray],
    gamma_factors: FloatArray,
    tau_start_k: float,
    tau_shared: float,
    tau_end_kp1: float,
) -> list[float]:
    """EXTRACTED: Calculate backward trajectory errors."""
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

    return all_bwd_errors


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
    """NEVER-NESTER REFACTORED: Check if two adjacent intervals can be merged."""

    # EXTRACTION: Validate inputs and extract parameters
    try:
        (global_mesh, tau_start_k, tau_shared, tau_end_kp1, beta_k, beta_kp1, alpha) = (
            _validate_h_reduction_inputs(solution, phase_id, first_idx)
        )
    except (ValueError, KeyError) as e:
        logger.debug(f"Validation failed for phase {phase_id}: {e}")
        return False

    error_tol = adaptive_params.error_tolerance
    ode_rtol = adaptive_params.ode_solver_tolerance
    ode_atol = ode_rtol * DEFAULT_ODE_ATOL_FACTOR
    num_sim_points = adaptive_params.num_error_sim_points

    # Get variable counts and dynamics function
    num_states, _ = problem.get_phase_variable_counts(phase_id)
    phase_dynamics_function = problem.get_phase_dynamics_function(phase_id)

    # Time transformation parameters
    t0 = solution.phase_initial_times[phase_id]
    tf = solution.phase_terminal_times[phase_id]
    alpha_0 = (tf + t0) / 2.0

    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    # EXTRACTION: Create RHS functions
    merged_fwd_rhs = _create_merged_dynamics_rhs(
        phase_dynamics_function,
        control_evaluator_first,
        tau_start_k,
        tau_shared,
        alpha,
        alpha_0,
        scaling_k,
        num_states,
    )

    merged_bwd_rhs = _create_merged_dynamics_rhs(
        phase_dynamics_function,
        control_evaluator_second,
        tau_shared,
        tau_end_kp1,
        alpha,
        alpha_0,
        scaling_kp1,
        num_states,
    )

    # EXTRACTION: Get initial and terminal states
    try:
        initial_state_fwd, terminal_state_bwd = _extract_initial_and_terminal_states(
            solution, problem, phase_id, first_idx, num_states
        )
    except Exception:
        return False

    # EXTRACTION: Forward simulation
    target_end_tau_k = map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )

    fwd_trajectory, fwd_sim_success = _run_forward_simulation(
        merged_fwd_rhs,
        initial_state_fwd,
        target_end_tau_k,
        num_sim_points,
        ode_rtol,
        ode_atol,
        phase_id,
        first_idx,
    )

    # EXTRACTION: Backward simulation
    target_end_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )

    bwd_trajectory, bwd_sim_success = _run_backward_simulation(
        merged_bwd_rhs,
        terminal_state_bwd,
        target_end_tau_kp1,
        num_sim_points,
        ode_rtol,
        ode_atol,
        phase_id,
        first_idx,
    )

    # For problems with no states, just check if simulations were successful
    if num_states == 0:
        return fwd_sim_success and bwd_sim_success

    # EXTRACTION: Calculate errors
    num_fwd_pts = max(2, num_sim_points // 2)
    fwd_tau_points = np.linspace(-1.0, target_end_tau_k, num_fwd_pts, dtype=np.float64)

    all_fwd_errors = _calculate_forward_errors(
        fwd_trajectory,
        fwd_tau_points,
        state_evaluator_first,
        state_evaluator_second,
        gamma_factors,
        tau_start_k,
        tau_shared,
        tau_end_kp1,
    )

    num_bwd_pts = max(2, num_sim_points // 2)
    bwd_tau_points = np.linspace(1.0, target_end_tau_kp1, num_bwd_pts, dtype=np.float64)
    sorted_bwd_tau_points = np.flip(bwd_tau_points)

    all_bwd_errors = _calculate_backward_errors(
        bwd_trajectory,
        sorted_bwd_tau_points,
        state_evaluator_first,
        state_evaluator_second,
        gamma_factors,
        tau_start_k,
        tau_shared,
        tau_end_kp1,
    )

    # Use mathematical core function for merge decision
    can_merge, max_error = _calculate_merge_feasibility_from_errors(
        all_fwd_errors, all_bwd_errors, error_tol
    )

    return can_merge
