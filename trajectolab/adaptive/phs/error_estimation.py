"""
Error estimation functions using forward and backward dynamics simulation.
"""

import logging
from collections.abc import Callable
from typing import cast

import casadi as ca
import numpy as np

from trajectolab.adaptive.phs.data_structures import IntervalSimulationBundle
from trajectolab.tl_types import (
    FloatArray,
    ODESolverCallable,
    OptimalControlSolution,
    ProblemProtocol,
)
from trajectolab.utils.constants import DEFAULT_ODE_RTOL


__all__ = [
    "calculate_gamma_normalizers",
    "calculate_relative_error_estimate",
    "simulate_dynamics_for_error_estimation",
]

# Set up logging for this module
logger = logging.getLogger(__name__)


def _simulate_forward(
    dynamics_rhs,
    initial_state: FloatArray,
    tau_points: FloatArray,
    ode_solver: ODESolverCallable,
    ode_rtol: float,
) -> tuple[bool, FloatArray]:
    """Simulate forward dynamics."""
    try:
        sim_result = ode_solver(
            dynamics_rhs,
            t_span=(-1, 1),
            y0=initial_state,
            t_eval=tau_points,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_rtol * 1e-2,
        )
        return sim_result.success, sim_result.y if sim_result.success else np.array([])
    except Exception as e:
        logger.error(f"Forward simulation failed: {e}")
        return False, np.array([])


def _simulate_backward(
    dynamics_rhs,
    terminal_state: FloatArray,
    tau_points: FloatArray,
    ode_solver: ODESolverCallable,
    ode_rtol: float,
) -> tuple[bool, FloatArray]:
    """Simulate backward dynamics."""
    try:
        sim_result = ode_solver(
            dynamics_rhs,
            t_span=(1, -1),
            y0=terminal_state,
            t_eval=tau_points,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_rtol * 1e-2,
        )
        return sim_result.success, np.fliplr(sim_result.y) if sim_result.success else np.array([])
    except Exception as e:
        logger.error(f"Backward simulation failed: {e}")
        return False, np.array([])


def simulate_dynamics_for_error_estimation(
    interval_idx: int,
    solution: "OptimalControlSolution",
    problem: ProblemProtocol,
    state_evaluator: Callable[[float | FloatArray], FloatArray],
    control_evaluator: Callable[[float | FloatArray], FloatArray],
    ode_solver: ODESolverCallable,
    ode_rtol: float = DEFAULT_ODE_RTOL,
    n_eval_points: int = 50,
) -> IntervalSimulationBundle:
    """
    Simulates dynamics forward and backward for error estimation - SIMPLIFIED.
    Updated to use unified storage system.
    """
    result = IntervalSimulationBundle(are_forward_and_backward_simulations_successful=False)

    if not solution.success or solution.raw_solution is None:
        logger.warning(f"NLP solution unsuccessful for interval {interval_idx}")
        return result

    # Get variable counts from unified storage
    num_states, num_controls = problem.get_variable_counts()
    casadi_dynamics_function = problem.get_dynamics_function()
    problem_parameters = problem._parameters

    # Validate time variables
    if solution.initial_time_variable is None or solution.terminal_time_variable is None:
        logger.error(f"Missing time variables for interval {interval_idx}")
        return result

    # Time transformation parameters
    t0 = solution.initial_time_variable
    tf = solution.terminal_time_variable
    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    # Validate global mesh
    if solution.global_normalized_mesh_nodes is None:
        logger.error(f"Missing global mesh for interval {interval_idx}")
        return result

    global_mesh = solution.global_normalized_mesh_nodes
    tau_start = global_mesh[interval_idx]
    tau_end = global_mesh[interval_idx + 1]

    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < 1e-12:
        logger.warning(f"Interval {interval_idx} has zero length")
        return result

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

        # Get dynamics result as list[ca.MX] from protocol
        dynamics_result = casadi_dynamics_function(
            ca.MX(state), ca.MX(control), ca.MX(physical_time), problem_parameters
        )

        # Convert list[ca.MX] to numpy - protocol guarantees this is a list
        state_deriv_np = np.array(
            [float(ca.evalf(expr)) for expr in dynamics_result], dtype=np.float64
        )

        if state_deriv_np.shape[0] != num_states:
            raise ValueError(
                f"Dynamics output dimension mismatch. Expected {num_states}, got {state_deriv_np.shape[0]}"
            )

        return cast(FloatArray, overall_scaling * state_deriv_np)

    # Forward simulation
    initial_state = state_evaluator(-1.0)
    if initial_state.ndim > 1:
        initial_state = initial_state.flatten()

    fwd_tau_points = np.linspace(-1, 1, n_eval_points, dtype=np.float64)
    fwd_success, fwd_trajectory = _simulate_forward(
        dynamics_rhs, initial_state, fwd_tau_points, ode_solver, ode_rtol
    )

    # Store forward results
    result.forward_simulation_local_tau_evaluation_points = fwd_tau_points
    if fwd_success:
        result.state_trajectory_from_forward_simulation = fwd_trajectory
    else:
        result.state_trajectory_from_forward_simulation = np.full(
            (num_states, len(fwd_tau_points)), np.nan, dtype=np.float64
        )

    result.nlp_state_trajectory_evaluated_at_forward_simulation_points = state_evaluator(
        fwd_tau_points
    )

    # Backward simulation
    terminal_state = state_evaluator(1.0)
    if terminal_state.ndim > 1:
        terminal_state = terminal_state.flatten()

    bwd_tau_points = np.linspace(1, -1, n_eval_points, dtype=np.float64)
    bwd_success, bwd_trajectory = _simulate_backward(
        dynamics_rhs, terminal_state, bwd_tau_points, ode_solver, ode_rtol
    )

    # Store backward results
    sorted_bwd_tau_points = np.flip(bwd_tau_points)
    result.backward_simulation_local_tau_evaluation_points = sorted_bwd_tau_points

    if bwd_success:
        result.state_trajectory_from_backward_simulation = bwd_trajectory
    else:
        result.state_trajectory_from_backward_simulation = np.full(
            (num_states, len(sorted_bwd_tau_points)), np.nan, dtype=np.float64
        )

    result.nlp_state_trajectory_evaluated_at_backward_simulation_points = state_evaluator(
        sorted_bwd_tau_points
    )

    result.are_forward_and_backward_simulations_successful = fwd_success and bwd_success
    return result


def calculate_relative_error_estimate(
    interval_idx: int, sim_bundle: IntervalSimulationBundle, gamma_factors: FloatArray
) -> float:
    """Calculates the maximum relative error estimate for an interval."""
    # Check for failed simulations
    if (
        not sim_bundle.are_forward_and_backward_simulations_successful
        or sim_bundle.state_trajectory_from_forward_simulation is None
        or sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points is None
        or sim_bundle.state_trajectory_from_backward_simulation is None
        or sim_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points is None
    ):
        logger.warning(f"Incomplete simulation results for interval {interval_idx}")
        return np.inf

    num_states = sim_bundle.state_trajectory_from_forward_simulation.shape[0]
    if num_states == 0:
        return 0.0

    # Forward errors
    fwd_diff = np.abs(
        sim_bundle.state_trajectory_from_forward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points
    )
    logger.debug(f"Forward max difference for interval {interval_idx}: {np.max(fwd_diff):.2e}")

    fwd_errors = gamma_factors * fwd_diff
    max_fwd_errors = (
        np.nanmax(fwd_errors, axis=1)
        if fwd_errors.size > 0
        else np.zeros(num_states, dtype=np.float64)
    )

    # Backward errors
    bwd_diff = np.abs(
        sim_bundle.state_trajectory_from_backward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points
    )
    logger.debug(f"Backward max difference for interval {interval_idx}: {np.max(bwd_diff):.2e}")

    bwd_errors = gamma_factors * bwd_diff
    max_bwd_errors = (
        np.nanmax(bwd_errors, axis=1)
        if bwd_errors.size > 0
        else np.zeros(num_states, dtype=np.float64)
    )

    # Combined error
    max_errors_per_state = np.maximum(max_fwd_errors, max_bwd_errors)
    max_error = np.nanmax(max_errors_per_state) if max_errors_per_state.size > 0 else np.inf

    # Ensure minimum error threshold
    if max_error < 1e-15:
        max_error = 1e-15

    if np.isnan(max_error):
        logger.warning(f"Error calculation resulted in NaN for interval {interval_idx}")
        return np.inf

    return float(max_error)


def calculate_gamma_normalizers(
    solution: "OptimalControlSolution", problem: ProblemProtocol
) -> FloatArray | None:
    """Calculates gamma_i normalization factors for error estimation - SIMPLIFIED."""
    if not solution.success or solution.raw_solution is None:
        return None

    # Get variable counts from unified storage
    num_states, _ = problem.get_variable_counts()
    if num_states == 0:
        return np.array([], dtype=np.float64).reshape(0, 1)

    Y_solved_list = solution.solved_state_trajectories_per_interval
    if not Y_solved_list:
        logger.warning("Missing solved state trajectories for gamma calculation")
        return None

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

    # Calculate gamma factors
    gamma_denominator = 1.0 + max_abs_values
    gamma_factors = 1.0 / np.maximum(gamma_denominator, np.float64(1e-12))

    return gamma_factors.reshape(-1, 1)
