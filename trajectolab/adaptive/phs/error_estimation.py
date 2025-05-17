"""
Error estimation functions for the PHS adaptive algorithm.
"""

from typing import cast

import numpy as np
from scipy.integrate import solve_ivp

from trajectolab.adaptive.phs.data_structures import IntervalSimulationBundle, NumPyDynamicsAdapter
from trajectolab.tl_types import (
    ProblemProtocol,
    _ControlEvaluator,
    _FloatArray,
    _GammaFactors,
    _ODESolverCallable,
    _StateEvaluator,
)

__all__ = [
    "simulate_dynamics_for_error_estimation",
    "calculate_relative_error_estimate",
    "calculate_gamma_normalizers",
]


def simulate_dynamics_for_error_estimation(
    interval_idx: int,
    solution: "OptimalControlSolution",
    problem: ProblemProtocol,
    state_evaluator: _StateEvaluator,
    control_evaluator: _ControlEvaluator,
    ode_solver: _ODESolverCallable = solve_ivp,
    ode_rtol: float = 1e-7,
    n_eval_points: int = 50,
) -> IntervalSimulationBundle:
    """
    Simulates dynamics forward and backward for error estimation.
    Uses pre-computed polynomial interpolants for state and control.
    """
    result = IntervalSimulationBundle(
        are_forward_and_backward_simulations_successful=False
    )  # Default to failure

    if not solution.success or solution.raw_solution is None:
        print(
            f"    Warning: NLP solution unsuccessful for interval {interval_idx} in error simulation."
        )
        return result

    num_states = len(problem._states)
    # Remove the unused variable warning by adding underscore
    _num_controls = len(problem._controls)
    casadi_dynamics_function = problem.get_dynamics_function()
    problem_parameters = problem._parameters

    # Create NumPy adapter for dynamics function
    numpy_dynamics = NumPyDynamicsAdapter(casadi_dynamics_function, problem_parameters)

    # Time transformation parameters
    t0 = solution.initial_time_variable
    tf = solution.terminal_time_variable

    # Check for None before arithmetic operations
    if t0 is None or tf is None:
        print(
            f"    Warning: Initial or terminal time is None for interval {interval_idx}. Skipping simulation."
        )
        return result

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    # Check if global_mesh is None before indexing
    global_mesh = solution.global_normalized_mesh_nodes
    if global_mesh is None:
        print(f"    Warning: Global mesh is None for interval {interval_idx}. Skipping simulation.")
        return result

    tau_start = global_mesh[interval_idx]
    tau_end = global_mesh[interval_idx + 1]

    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < 1e-12:
        print(f"    Warning: Interval {interval_idx} has zero length. Skipping simulation.")
        return result

    beta_k0 = (tau_end + tau_start) / 2.0
    overall_scaling = alpha * beta_k

    def dynamics_rhs(tau: float, state: _FloatArray) -> _FloatArray:
        """Right-hand side of dynamics ODE in local tau coordinates."""
        # Get control from interpolant
        control = control_evaluator(tau)
        if control.ndim > 1:
            control = control.flatten()  # Ensure 1D for dynamics

        # Convert to global coordinates and physical time
        global_tau = beta_k * tau + beta_k0
        physical_time = alpha * global_tau + alpha_0

        # Use the NumPy dynamics adapter
        state_deriv_np = numpy_dynamics(state, control, physical_time)

        if state_deriv_np.shape[0] != num_states:
            raise ValueError(
                f"Dynamics function output dimension mismatch. Expected {num_states}, got {state_deriv_np.shape[0]}."
            )

        return cast(_FloatArray, overall_scaling * state_deriv_np)

    # Forward simulation (IVP)
    initial_state = state_evaluator(-1.0)
    if initial_state.ndim > 1:
        initial_state = initial_state.flatten()

    fwd_tau_points = np.linspace(-1, 1, n_eval_points, dtype=np.float64)

    # Call solve_ivp with kwargs
    fwd_sim = ode_solver(
        dynamics_rhs,
        t_span=(-1, 1),
        y0=initial_state,
        t_eval=fwd_tau_points,
        method="RK45",
        rtol=ode_rtol,
        atol=ode_rtol * 1e-2,
    )

    result.forward_simulation_local_tau_evaluation_points = fwd_tau_points
    if fwd_sim.success:
        result.state_trajectory_from_forward_simulation = fwd_sim.y
    else:
        result.state_trajectory_from_forward_simulation = np.full(
            (num_states, len(fwd_tau_points)), np.nan, dtype=np.float64
        )
        print(f"    Fwd IVP fail int {interval_idx}: {fwd_sim.message}")

    result.nlp_state_trajectory_evaluated_at_forward_simulation_points = state_evaluator(
        fwd_tau_points
    )

    # Backward simulation (TVP)
    terminal_state = state_evaluator(1.0)
    if terminal_state.ndim > 1:
        terminal_state = terminal_state.flatten()

    bwd_tau_points = np.linspace(1, -1, n_eval_points, dtype=np.float64)

    # Call solve_ivp with kwargs
    bwd_sim = ode_solver(
        dynamics_rhs,
        t_span=(1, -1),
        y0=terminal_state,
        t_eval=bwd_tau_points,
        method="RK45",
        rtol=ode_rtol,
        atol=ode_rtol * 1e-2,
    )

    # Create the reversed tau points
    sorted_bwd_tau_points = np.flip(bwd_tau_points)

    if bwd_sim.success:
        result.backward_simulation_local_tau_evaluation_points = np.array(
            sorted_bwd_tau_points, dtype=np.float64
        )
        result.state_trajectory_from_backward_simulation = np.fliplr(bwd_sim.y)
    else:
        result.backward_simulation_local_tau_evaluation_points = np.array(
            sorted_bwd_tau_points, dtype=np.float64
        )
        result.state_trajectory_from_backward_simulation = np.full(
            (num_states, len(sorted_bwd_tau_points)), np.nan, dtype=np.float64
        )
        print(f"    Bwd TVP fail int {interval_idx}: {bwd_sim.message}")

    result.nlp_state_trajectory_evaluated_at_backward_simulation_points = state_evaluator(
        sorted_bwd_tau_points
    )

    result.are_forward_and_backward_simulations_successful = fwd_sim.success and bwd_sim.success
    return result


def calculate_relative_error_estimate(
    interval_idx: int, sim_bundle: IntervalSimulationBundle, gamma_factors: _GammaFactors
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
        print(
            f"    Interval {interval_idx}: Simulation results incomplete or failed. Returning np.inf."
        )
        return np.inf

    num_states = sim_bundle.state_trajectory_from_forward_simulation.shape[0]
    if num_states == 0:
        return 0.0  # No states, no error

    # Forward errors - add debugging to see what's happening
    fwd_diff = np.abs(
        sim_bundle.state_trajectory_from_forward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points
    )
    print(
        f"    DEBUG: Forward differences shape: {fwd_diff.shape}, Max raw diff: {np.max(fwd_diff)}"
    )

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
    print(
        f"    DEBUG: Backward differences shape: {bwd_diff.shape}, Max raw diff: {np.max(bwd_diff)}"
    )

    bwd_errors = gamma_factors * bwd_diff
    max_bwd_errors = (
        np.nanmax(bwd_errors, axis=1)
        if bwd_errors.size > 0
        else np.zeros(num_states, dtype=np.float64)
    )

    # Take maximum of forward and backward errors for each state
    max_errors_per_state = np.maximum(max_fwd_errors, max_bwd_errors)

    # Overall maximum error across all states
    max_error = np.nanmax(max_errors_per_state) if max_errors_per_state.size > 0 else np.inf

    # Ensure we never return exactly zero error (numerical tolerance)
    if max_error < 1e-15:
        max_error = 1e-15

    if np.isnan(max_error):
        print(
            f"    Interval {interval_idx}: Error calculation resulted in NaN. Treating as high error (np.inf)."
        )
        return np.inf

    return float(max_error)


def calculate_gamma_normalizers(
    solution: "OptimalControlSolution", problem: ProblemProtocol
) -> _GammaFactors | None:
    """Calculates gamma_i normalization factors for error estimation."""
    if not solution.success or solution.raw_solution is None:
        return None

    num_states = len(problem._states)
    if num_states == 0:
        return np.array([], dtype=np.float64).reshape(0, 1)  # No states, no gamma

    Y_solved_list = solution.solved_state_trajectories_per_interval
    if not Y_solved_list:
        print("    Warning: solved_state_trajectories_per_interval missing for gamma calculation.")
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
    # Use np.float64 directly to avoid type mismatch
    gamma_factors = 1.0 / np.maximum(gamma_denominator, np.float64(1e-12))  # Avoid division by zero

    return cast(_GammaFactors, gamma_factors.reshape(-1, 1))
