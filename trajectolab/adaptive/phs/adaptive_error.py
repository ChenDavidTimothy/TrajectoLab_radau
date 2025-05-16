"""Error estimation utilities for adaptive mesh refinement."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from trajectolab.adaptive.phs.adaptive_interpolation import PolynomialInterpolant
from trajectolab.trajectolab_types import (
    CasADiDM,
    NumStates,
    OptimalControlProblem,
    OptimalControlSolution,
    _Matrix,
    _MeshPoints,
    _NormalizedTimePoint,
    _Vector,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_STATE_CLIP_VALUE = 1e6


@dataclass
class IntervalSimulationBundle:
    """Holds simulation results for error estimation."""

    forward_simulation_local_tau_evaluation_points: _Matrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    state_trajectory_from_forward_simulation: _Matrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    nlp_state_trajectory_evaluated_at_forward_simulation_points: _Matrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    backward_simulation_local_tau_evaluation_points: _Matrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    state_trajectory_from_backward_simulation: _Matrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    nlp_state_trajectory_evaluated_at_backward_simulation_points: _Matrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    are_forward_and_backward_simulations_successful: bool = True

    def __post_init__(self):
        """Ensure arrays have correct shape and type."""
        for field_name, field_val in self.__dict__.items():
            if not isinstance(field_val, np.ndarray) or field_val.size == 0:
                continue

            current_val = field_val
            if current_val.dtype != np.float64:
                current_val = current_val.astype(np.float64)

            if current_val.ndim == 1:
                setattr(self, field_name, current_val.reshape(1, -1))
            elif current_val.ndim > 2:
                raise ValueError(
                    f"Field {field_name} must be 1D or 2D array, got {current_val.ndim}D."
                )
            elif current_val.ndim == 2 and current_val.dtype != np.float64:
                setattr(self, field_name, current_val)


def simulate_dynamics_for_error_estimation(
    interval_idx: int,
    solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    state_evaluator: PolynomialInterpolant,
    control_evaluator: PolynomialInterpolant,
    ode_solver: Callable[..., Any] = solve_ivp,
    ode_rtol: float = 1e-7,
    n_eval_points: int = 50,
) -> IntervalSimulationBundle:
    """Simulate system dynamics for error estimation.

    Args:
        interval_idx: Index of the interval to simulate
        solution: Optimization solution from NLP
        problem: Optimal control problem definition
        state_evaluator: Interpolant for state trajectories
        control_evaluator: Interpolant for control trajectories
        ode_solver: ODE solver function to use
        ode_rtol: Relative tolerance for ODE solver
        n_eval_points: Number of points for error evaluation

    Returns:
        Bundle of simulation results for error calculation
    """
    result = IntervalSimulationBundle(are_forward_and_backward_simulations_successful=False)

    if not solution.success or solution.raw_solution is None:
        logger.warning(
            f"    Interval {interval_idx}: NLP solution unsuccessful for error simulation."
        )
        return result

    num_states: NumStates = problem.num_states
    dynamics_function = problem.dynamics_function
    problem_parameters = problem.problem_parameters

    t0_sol = solution.initial_time_variable
    tf_sol = solution.terminal_time_variable

    if t0_sol is None or tf_sol is None:
        logger.warning(f"    Interval {interval_idx}: Time variables (t0, tf) are None.")
        return result

    t0: float = float(t0_sol)
    tf: float = float(tf_sol)

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    if solution.global_normalized_mesh_nodes is None:
        logger.warning(f"    Interval {interval_idx}: global_normalized_mesh_nodes is None.")
        return result

    global_mesh: _MeshPoints = np.array(solution.global_normalized_mesh_nodes, dtype=np.float64)
    tau_start: _NormalizedTimePoint = global_mesh[interval_idx]
    tau_end: _NormalizedTimePoint = global_mesh[interval_idx + 1]

    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < 1e-12:  # Interval of zero length
        logger.warning(f"    Interval {interval_idx} has zero length. Skipping simulation.")
        # Initialize with correct empty shapes
        result.forward_simulation_local_tau_evaluation_points = np.empty((1, 0), dtype=np.float64)
        result.state_trajectory_from_forward_simulation = np.empty(
            (num_states, 0), dtype=np.float64
        )
        result.nlp_state_trajectory_evaluated_at_forward_simulation_points = np.empty(
            (num_states, 0), dtype=np.float64
        )
        result.backward_simulation_local_tau_evaluation_points = np.empty((1, 0), dtype=np.float64)
        result.state_trajectory_from_backward_simulation = np.empty(
            (num_states, 0), dtype=np.float64
        )
        result.nlp_state_trajectory_evaluated_at_backward_simulation_points = np.empty(
            (num_states, 0), dtype=np.float64
        )
        return result

    beta_k0 = (tau_end + tau_start) / 2.0
    overall_scaling = alpha * beta_k

    def dynamics_rhs(tau_local: _NormalizedTimePoint, state_np: _Vector) -> _Vector:
        """Right-hand side of dynamics ODE in local tau coordinates."""
        # Get control from interpolant - handle any shape
        try:
            control_val_at_tau: Any = control_evaluator(tau_local)
        except Exception as e:
            logger.error(f"Error evaluating control at tau={tau_local}: {e}")
            control_val_at_tau = np.zeros(control_evaluator.num_vars, dtype=np.float64)

        num_controls = control_evaluator.num_vars

        # Process control value to ensure correct shape
        control_np: _Vector
        if num_controls == 0:
            control_np = np.array([], dtype=np.float64)
        elif isinstance(control_val_at_tau, (int, float, np.number)) or (
            hasattr(control_val_at_tau, "ndim")
            and (
                control_val_at_tau.ndim == 0
                or (isinstance(control_val_at_tau, np.ndarray) and control_val_at_tau.size == 1)
            )
        ):
            # Handle scalar control value
            if num_controls == 1:
                control_np = np.array([float(control_val_at_tau)], dtype=np.float64)
            else:
                shape_info = getattr(control_val_at_tau, "shape", type(control_val_at_tau))
                logger.warning(
                    f"Scalar control value but num_controls={num_controls}, shape={shape_info}"
                )
                control_np = np.zeros(num_controls, dtype=np.float64)  # Safe default
        elif hasattr(control_val_at_tau, "ndim"):
            if control_val_at_tau.ndim == 2 and control_val_at_tau.shape[1] == 1:
                control_np = control_val_at_tau.flatten()
            elif control_val_at_tau.ndim == 1:
                control_np = control_val_at_tau
            else:
                shape_info = getattr(control_val_at_tau, "shape", None)
                logger.warning(f"Unexpected control shape from interpolant: {shape_info}")
                control_np = np.zeros(num_controls, dtype=np.float64)  # Safe default
        else:
            # Unknown type fallback
            shape_info = type(control_val_at_tau)
            logger.warning(f"Control value has an unsupported type: {shape_info}")
            control_np = np.zeros(num_controls, dtype=np.float64)  # Safe default

        # Safety clip to avoid extremely large values
        state_clipped_np = np.clip(state_np, -DEFAULT_STATE_CLIP_VALUE, DEFAULT_STATE_CLIP_VALUE)

        # Transform to global coordinates and physical time
        global_tau_val: _NormalizedTimePoint = beta_k * tau_local + beta_k0
        physical_time: float = alpha * global_tau_val + alpha_0

        # Create CasADi DM objects for dynamics function
        state_ca = CasADiDM(state_clipped_np)
        control_ca = CasADiDM(control_np)

        # Evaluate dynamics function
        try:
            state_deriv_symbolic = dynamics_function(
                state_ca, control_ca, physical_time, problem_parameters
            )
            state_deriv_np_flat: _Vector = np.array(
                state_deriv_symbolic, dtype=np.float64
            ).flatten()
        except Exception as e:
            logger.error(f"Error in dynamics function at t={physical_time}: {e}")
            state_deriv_np_flat = np.zeros(num_states, dtype=np.float64)

        # Check dimensions
        if state_deriv_np_flat.shape[0] != num_states:
            logger.error(
                f"Dynamics function output dimension mismatch. Expected {num_states}, "
                f"got {state_deriv_np_flat.shape[0]}. Using zeros."
            )
            return np.zeros(num_states, dtype=np.float64)

        return overall_scaling * state_deriv_np_flat

    # Forward simulation
    try:
        initial_state_val: Any = state_evaluator(-1.0)
        initial_state: _Vector

        if num_states == 0:
            initial_state = np.array([], dtype=np.float64)
        elif isinstance(initial_state_val, (int, float, np.number)) or (
            hasattr(initial_state_val, "ndim") and initial_state_val.ndim == 0
        ):
            # Handle scalar value - turn it into a 1-element array
            if num_states == 1:
                initial_state = np.array([float(initial_state_val)], dtype=np.float64)
            else:
                logger.warning(f"Scalar initial_state_val but num_states={num_states}")
                initial_state = np.zeros(num_states, dtype=np.float64)  # Use zeros for stability
        elif (
            hasattr(initial_state_val, "ndim")
            and initial_state_val.ndim == 2
            and initial_state_val.shape[1] == 1
        ):
            initial_state = initial_state_val.flatten()
        elif (
            hasattr(initial_state_val, "ndim")
            and initial_state_val.ndim == 1
            and initial_state_val.size == num_states
        ):
            initial_state = initial_state_val
        else:
            shape_info = getattr(initial_state_val, "shape", type(initial_state_val))
            logger.warning(f"Unexpected initial_state_val shape: {shape_info}")
            initial_state = np.zeros(num_states, dtype=np.float64)  # Use zeros for stability
    except Exception as e:
        logger.error(f"Error getting initial state: {e}")
        initial_state = np.zeros(num_states, dtype=np.float64)  # Safe fallback

    fwd_tau_points_1d: _Vector = np.linspace(-1.0, 1.0, n_eval_points, dtype=np.float64)

    try:
        fwd_sim = ode_solver(
            dynamics_rhs,
            t_span=(-1.0, 1.0),
            y0=initial_state,
            t_eval=fwd_tau_points_1d,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_rtol * 1e-2,
        )

        result.forward_simulation_local_tau_evaluation_points = fwd_tau_points_1d.reshape(1, -1)
        result.state_trajectory_from_forward_simulation = (
            fwd_sim.y.astype(np.float64)
            if fwd_sim.success and hasattr(fwd_sim, "y")
            else np.full((num_states, len(fwd_tau_points_1d)), np.nan, dtype=np.float64)
        )
        if not fwd_sim.success:
            logger.warning(
                f"    Fwd IVP fail int {interval_idx}: {getattr(fwd_sim, 'message', 'Unknown reason')}"
            )
    except Exception as e:
        logger.error(f"Forward simulation failed: {e}")
        result.forward_simulation_local_tau_evaluation_points = fwd_tau_points_1d.reshape(1, -1)
        result.state_trajectory_from_forward_simulation = np.full(
            (num_states, len(fwd_tau_points_1d)), np.nan, dtype=np.float64
        )
        fwd_sim = None  # Mark as failed

    try:
        nlp_fwd_eval_matrix: Any = state_evaluator(fwd_tau_points_1d)
        result.nlp_state_trajectory_evaluated_at_forward_simulation_points = (
            nlp_fwd_eval_matrix.reshape(num_states, -1)
            if num_states > 0
            else np.empty((0, len(fwd_tau_points_1d)), dtype=np.float64)
        )
    except Exception as e:
        logger.error(f"Error evaluating NLP states for forward points: {e}")
        result.nlp_state_trajectory_evaluated_at_forward_simulation_points = np.full(
            (num_states, len(fwd_tau_points_1d)), np.nan, dtype=np.float64
        )

    # Backward simulation
    try:
        terminal_state_val: Any = state_evaluator(1.0)
        terminal_state: _Vector

        if num_states == 0:
            terminal_state = np.array([], dtype=np.float64)
        elif isinstance(terminal_state_val, (int, float, np.number)) or (
            hasattr(terminal_state_val, "ndim") and terminal_state_val.ndim == 0
        ):
            # Handle scalar value - turn it into a 1-element array
            if num_states == 1:
                terminal_state = np.array([float(terminal_state_val)], dtype=np.float64)
            else:
                logger.warning(f"Scalar terminal_state_val but num_states={num_states}")
                terminal_state = np.zeros(num_states, dtype=np.float64)  # Use zeros for stability
        elif (
            hasattr(terminal_state_val, "ndim")
            and terminal_state_val.ndim == 2
            and terminal_state_val.shape[1] == 1
        ):
            terminal_state = terminal_state_val.flatten()
        elif (
            hasattr(terminal_state_val, "ndim")
            and terminal_state_val.ndim == 1
            and terminal_state_val.size == num_states
        ):
            terminal_state = terminal_state_val
        else:
            shape_info = getattr(terminal_state_val, "shape", type(terminal_state_val))
            logger.warning(f"Unexpected terminal_state_val shape: {shape_info}")
            terminal_state = np.zeros(num_states, dtype=np.float64)  # Use zeros for stability
    except Exception as e:
        logger.error(f"Error getting terminal state: {e}")
        terminal_state = np.zeros(num_states, dtype=np.float64)  # Safe fallback

    bwd_tau_points_1d: _Vector = np.linspace(1.0, -1.0, n_eval_points, dtype=np.float64)

    try:
        bwd_sim = ode_solver(
            dynamics_rhs,
            t_span=(1.0, -1.0),
            y0=terminal_state,
            t_eval=bwd_tau_points_1d,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_rtol * 1e-2,
        )

        sorted_bwd_tau_points_1d: _Vector = np.flip(bwd_tau_points_1d).astype(np.float64)
        result.backward_simulation_local_tau_evaluation_points = sorted_bwd_tau_points_1d.reshape(
            1, -1
        )
        result.state_trajectory_from_backward_simulation = (
            np.fliplr(bwd_sim.y).astype(np.float64)
            if bwd_sim.success and hasattr(bwd_sim, "y")
            else np.full((num_states, len(sorted_bwd_tau_points_1d)), np.nan, dtype=np.float64)
        )
        if not bwd_sim.success:
            logger.warning(
                f"    Bwd IVP fail int {interval_idx}: {getattr(bwd_sim, 'message', 'Unknown reason')}"
            )
    except Exception as e:
        logger.error(f"Backward simulation failed: {e}")
        sorted_bwd_tau_points_1d = np.flip(bwd_tau_points_1d).astype(np.float64)
        result.backward_simulation_local_tau_evaluation_points = sorted_bwd_tau_points_1d.reshape(
            1, -1
        )
        result.state_trajectory_from_backward_simulation = np.full(
            (num_states, len(sorted_bwd_tau_points_1d)), np.nan, dtype=np.float64
        )
        bwd_sim = None  # Mark as failed

    try:
        nlp_bwd_eval_matrix: Any = state_evaluator(sorted_bwd_tau_points_1d)
        result.nlp_state_trajectory_evaluated_at_backward_simulation_points = (
            nlp_bwd_eval_matrix.reshape(num_states, -1)
            if num_states > 0
            else np.empty((0, len(sorted_bwd_tau_points_1d)), dtype=np.float64)
        )
    except Exception as e:
        logger.error(f"Error evaluating NLP states for backward points: {e}")
        result.nlp_state_trajectory_evaluated_at_backward_simulation_points = np.full(
            (num_states, len(sorted_bwd_tau_points_1d)), np.nan, dtype=np.float64
        )

    # Check if simulations were successful
    fwd_success = fwd_sim is not None and hasattr(fwd_sim, "success") and fwd_sim.success
    bwd_success = bwd_sim is not None and hasattr(bwd_sim, "success") and bwd_sim.success
    result.are_forward_and_backward_simulations_successful = fwd_success and bwd_success

    return result


def calculate_relative_error_estimate(
    interval_idx: int,
    sim_bundle: IntervalSimulationBundle,
    gamma_factors: _Vector,  # Column vector (NumStates x 1)
) -> float:
    """Calculate relative error estimate for an interval.

    Args:
        interval_idx: Index of the interval
        sim_bundle: Simulation results bundle
        gamma_factors: Scaling factors for error normalization

    Returns:
        Maximum relative error for the interval
    """
    if (
        not sim_bundle.are_forward_and_backward_simulations_successful
        or sim_bundle.state_trajectory_from_forward_simulation.size == 0
    ):
        logger.warning(f"    Interval {interval_idx}: Simulation results incomplete or failed.")
        return np.inf

    num_states: NumStates = sim_bundle.state_trajectory_from_forward_simulation.shape[0]
    if num_states == 0:
        return 0.0  # No states, no error.

    # Ensure gamma_factors is float64 and correct shape (N_states, 1)
    gamma_factors_col_vec = gamma_factors.astype(np.float64).reshape(num_states, 1)

    # Print some debugging info to help understand data shapes
    fwd_diff_shape = sim_bundle.state_trajectory_from_forward_simulation.shape
    fwd_nlp_shape = sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points.shape
    bwd_diff_shape = sim_bundle.state_trajectory_from_backward_simulation.shape
    bwd_nlp_shape = sim_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points.shape

    logger.debug(
        f"Forward: sim={fwd_diff_shape}, nlp={fwd_nlp_shape}; Backward: sim={bwd_diff_shape}, nlp={bwd_nlp_shape}"
    )

    # Check for shape mismatches
    if (fwd_diff_shape != fwd_nlp_shape) or (bwd_diff_shape != bwd_nlp_shape):
        logger.warning(f"    Interval {interval_idx}: Shape mismatch in error calculation.")
        return np.inf

    # Errors from forward simulation
    fwd_diff: _Matrix = np.abs(
        sim_bundle.state_trajectory_from_forward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points
    )
    # Add debug output to see what's happening
    max_raw_diff = np.nanmax(fwd_diff) if fwd_diff.size > 0 else 0.0
    logger.debug(
        f"    Debug: Forward differences shape: {fwd_diff.shape}, Max raw diff: {max_raw_diff}"
    )

    fwd_errors: _Matrix = gamma_factors_col_vec * fwd_diff
    max_fwd_errors_per_state: _Vector = (
        np.nanmax(fwd_errors, axis=1)
        if fwd_errors.size > 0
        else np.zeros(num_states, dtype=np.float64)
    )

    # Errors from backward simulation
    bwd_diff: _Matrix = np.abs(
        sim_bundle.state_trajectory_from_backward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points
    )
    # Add debug output
    max_raw_diff = np.nanmax(bwd_diff) if bwd_diff.size > 0 else 0.0
    logger.debug(
        f"    Debug: Backward differences shape: {bwd_diff.shape}, Max raw diff: {max_raw_diff}"
    )

    bwd_errors: _Matrix = gamma_factors_col_vec * bwd_diff
    max_bwd_errors_per_state: _Vector = (
        np.nanmax(bwd_errors, axis=1)
        if bwd_errors.size > 0
        else np.zeros(num_states, dtype=np.float64)
    )

    max_errors_per_state_combined: _Vector = np.maximum(
        max_fwd_errors_per_state, max_bwd_errors_per_state
    )

    # Overall maximum error for the interval
    max_error_for_interval: float = (
        np.nanmax(max_errors_per_state_combined).item()
        if max_errors_per_state_combined.size > 0
        else 0.0
    )
    if num_states > 0 and max_errors_per_state_combined.size == 0:
        max_error_for_interval = np.inf

    if np.isnan(max_error_for_interval):
        logger.warning(f"    Interval {interval_idx}: Error calculation resulted in NaN.")
        return np.inf

    # Ensure a minimum floor for error to avoid issues with log in p-refinement
    return max(float(max_error_for_interval), 1e-15) if num_states > 0 else 0.0


def calculate_gamma_normalizers(
    solution: OptimalControlSolution, problem: OptimalControlProblem
) -> Optional[_Vector]:
    """Calculate gamma normalization factors for error estimation.

    Args:
        solution: Optimization solution
        problem: Optimal control problem definition

    Returns:
        Column vector of normalization factors, or None if calculation failed
    """
    if not solution.success:
        logger.warning("    Gamma calculation failed - solution unsuccessful.")
        return None

    num_states: NumStates = problem.num_states
    if num_states == 0:
        return np.array([]).reshape(0, 1).astype(np.float64)

    # Try to use solution.states first
    if solution.states and len(solution.states) > 0:
        # Find max absolute value for each state across all intervals
        max_abs_values_per_state: _Vector = np.zeros(num_states, dtype=np.float64)

        for Xk_trajectory_matrix in solution.states:
            if Xk_trajectory_matrix.size == 0:
                continue
            max_abs_in_interval_k: _Vector = np.max(
                np.abs(Xk_trajectory_matrix.astype(np.float64)), axis=1
            )
            max_abs_values_per_state = np.maximum(max_abs_values_per_state, max_abs_in_interval_k)
    # Fallback to solved_state_trajectories_per_interval if available
    elif (
        hasattr(solution, "solved_state_trajectories_per_interval")
        and solution.solved_state_trajectories_per_interval
    ):
        # Find maximum absolute value for each state component
        max_abs_values_per_state = np.zeros(num_states, dtype=np.float64)
        first_interval = True

        for Xk in solution.solved_state_trajectories_per_interval:
            if Xk.size == 0:
                continue

            max_abs_in_interval = np.max(np.abs(Xk), axis=1)

            if first_interval:
                max_abs_values_per_state = max_abs_in_interval
                first_interval = False
            else:
                max_abs_values_per_state = np.maximum(max_abs_values_per_state, max_abs_in_interval)
    else:
        logger.warning("    Gamma calculation failed - no state trajectories available.")
        return None

    # Normalization factor: gamma_i = 1 / (1 + max|x_i(t)|)
    gamma_denominator: _Vector = 1.0 + max_abs_values_per_state
    gamma_denominator_safe = np.maximum(gamma_denominator, 1e-12)
    gamma_factors_flat: _Vector = 1.0 / gamma_denominator_safe

    return gamma_factors_flat.reshape(-1, 1).astype(np.float64)  # Ensure column vector
