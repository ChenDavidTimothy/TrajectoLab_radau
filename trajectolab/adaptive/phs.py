from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast, overload

import numpy as np
from scipy.integrate import solve_ivp

from trajectolab.adaptive.base import AdaptiveBase
from trajectolab.radau import (
    compute_barycentric_weights,
    compute_radau_collocation_components,
    evaluate_lagrange_polynomial_at_point,
)
from trajectolab.trajectolab_types import (  # Basic types
    ZERO_TOLERANCE,
    OptimalControlProblem,
    OptimalControlSolution,
    _BarycentricWeights,
    _FloatArray,
    _MeshNodesList,
    _NormalizedTimePoint,
    _Vector,
)

# Type aliases specific to PHS module
_InterpolantCallable: TypeAlias = callable[[float | _Vector], _Vector | _FloatArray]


@dataclass
class AdaptiveParameters:
    """Error tolerance and iteration control parameters."""

    error_tolerance: float  # Error tolerance threshold (epsilon_tol)
    max_iterations: int  # Maximum number of refinement iterations (M_max_iterations)
    min_polynomial_degree: int  # Minimum polynomial degree allowed (N_min_poly_degree)
    max_polynomial_degree: int  # Maximum polynomial degree allowed (N_max_poly_degree)
    ode_solver_tolerance: float = 1e-7  # ODE solver tolerance (ode_solver_tol)
    num_error_sim_points: int = 50  # Number of points for error simulation


@dataclass
class IntervalSimulationBundle:
    """Holds results from forward/backward simulations for error estimation."""

    # Initialize with empty arrays that have the correct type for better type inference
    forward_simulation_local_tau_evaluation_points: _Vector = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    state_trajectory_from_forward_simulation: _FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    nlp_state_trajectory_evaluated_at_forward_simulation_points: _FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    backward_simulation_local_tau_evaluation_points: _Vector = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    state_trajectory_from_backward_simulation: _FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    nlp_state_trajectory_evaluated_at_backward_simulation_points: _FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    are_forward_and_backward_simulations_successful: bool = True

    def __post_init__(self) -> None:
        """Ensure all ndarray fields are properly formatted as 2D arrays if not empty."""
        for field_name, field_val in self.__dict__.items():
            if not isinstance(field_val, np.ndarray) or field_val.size == 0:  # Skip if empty
                continue

            # Ensure 2D arrays
            if field_val.ndim == 1:
                setattr(self, field_name, field_val.reshape(1, -1))
            elif field_val.ndim > 2:
                raise ValueError(
                    f"Field {field_name} must be 1D or 2D array, got {field_val.ndim}D."
                )


def _extract_and_prepare_array(
    casadi_value: Any, expected_rows: int, expected_cols: int
) -> _FloatArray:
    """Extracts numerical value from CasADi and ensures correct 2D shape."""
    # Convert to numpy array
    if hasattr(casadi_value, "to_DM"):
        np_array = np.array(casadi_value.to_DM(), dtype=np.float64)
    else:
        np_array = np.array(casadi_value, dtype=np.float64)

    # Handle empty arrays for states/controls
    if expected_rows == 0:
        return np.empty((0, expected_cols), dtype=np.float64)

    # Ensure 2D shape
    if np_array.ndim == 1:
        if len(np_array) == expected_rows:
            np_array = np_array.reshape(expected_rows, 1)
        else:
            np_array = np_array.reshape(1, -1)

    # Transpose if dimensions are swapped
    if np_array.shape[0] != expected_rows and np_array.shape[1] == expected_rows:
        np_array = np_array.T

    # Handle dimension mismatch as best we can
    if np_array.shape != (expected_rows, expected_cols):
        squeezed = np_array.squeeze()
        if squeezed.ndim == 1 and expected_rows == 1 and len(squeezed) == expected_cols:
            np_array = squeezed.reshape(1, expected_cols)

    return cast(_FloatArray, np_array)


class PolynomialInterpolant:
    """
    Callable class that implements Lagrange polynomial interpolation
    using the barycentric formula.
    """

    def __init__(
        self,
        nodes: _Vector,
        values: _FloatArray,
        barycentric_weights: _BarycentricWeights | None = None,
    ) -> None:
        """Creates a Lagrange polynomial interpolant using barycentric formula."""
        # Convert to arrays if needed and ensure 2D values
        self.values_at_nodes: _FloatArray = np.atleast_2d(values)
        self.nodes_array: _Vector = np.asarray(nodes, dtype=np.float64)
        self.num_vars: int = self.values_at_nodes.shape[0]
        self.num_nodes_val: int = self.values_at_nodes.shape[1]
        self.num_nodes_pts: int = len(self.nodes_array)

        if self.num_nodes_pts != self.num_nodes_val:
            raise ValueError(
                f"Mismatch in number of nodes ({self.num_nodes_pts}) and values columns ({self.num_nodes_val})"
            )

        # Compute or use provided barycentric weights
        self.bary_weights: _BarycentricWeights = (
            compute_barycentric_weights(self.nodes_array)
            if barycentric_weights is None
            else np.asarray(barycentric_weights, dtype=np.float64)
        )

        if len(self.bary_weights) != self.num_nodes_pts:
            raise ValueError("Barycentric weights length does not match nodes length")

    @overload
    def __call__(self, points: float) -> _Vector: ...

    @overload
    def __call__(self, points: _Vector) -> _FloatArray: ...

    def __call__(self, points: float | _Vector) -> _Vector | _FloatArray:
        """Evaluates the interpolant at the given point(s)."""
        is_scalar = np.isscalar(points)
        zeta_arr = np.atleast_1d(points)
        result = np.zeros((self.num_vars, len(zeta_arr)), dtype=np.float64)

        for i, zeta in enumerate(zeta_arr):
            L_j = evaluate_lagrange_polynomial_at_point(self.nodes_array, self.bary_weights, zeta)
            result[:, i] = np.dot(self.values_at_nodes, L_j)

        # Return appropriate shape based on input
        return result[:, 0] if is_scalar else result


def get_polynomial_interpolant(
    nodes: _Vector, values: _FloatArray, barycentric_weights: _BarycentricWeights | None = None
) -> PolynomialInterpolant:
    """Creates a Lagrange polynomial interpolant using barycentric formula."""
    return PolynomialInterpolant(nodes, values, barycentric_weights)


@overload
def dummy_evaluator(tau: float) -> _Vector: ...


@overload
def dummy_evaluator(tau: _Vector) -> _FloatArray: ...


def dummy_evaluator(tau: float | _Vector) -> _Vector | _FloatArray:
    """Dummy evaluator for cases when a proper interpolant is unavailable."""
    if np.isscalar(tau):
        return np.zeros((1,), dtype=np.float64)
    return np.zeros((1, len(np.atleast_1d(tau))), dtype=np.float64)


def _simulate_dynamics_for_error_estimation(
    interval_idx: int,
    solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    state_evaluator: _InterpolantCallable,
    control_evaluator: _InterpolantCallable,
    ode_solver: callable = solve_ivp,
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

    num_states = problem.num_states
    # Removed unused variable num_controls = problem.num_controls
    dynamics_function = problem.dynamics_function
    problem_parameters = problem.problem_parameters

    # Time transformation parameters
    t0 = solution.initial_time_variable
    tf = solution.terminal_time_variable
    if t0 is None or tf is None:
        print(f"    Warning: Time variables are None in simulation for interval {interval_idx}.")
        return result

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    global_mesh = (
        np.array(solution.global_normalized_mesh_nodes, dtype=np.float64)
        if solution.global_normalized_mesh_nodes is not None
        else np.array([], dtype=np.float64)
    )

    if global_mesh.size <= interval_idx + 1:
        print(f"    Warning: Global mesh insufficient for interval {interval_idx}.")
        return result

    tau_start = global_mesh[interval_idx]
    tau_end = global_mesh[interval_idx + 1]

    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < ZERO_TOLERANCE:
        print(f"    Warning: Interval {interval_idx} has zero length. Skipping simulation.")
        return result

    beta_k0 = (tau_end + tau_start) / 2.0
    overall_scaling = alpha * beta_k

    def dynamics_rhs(tau: float, state: np.ndarray) -> np.ndarray:
        """Right-hand side of dynamics ODE in local tau coordinates."""
        # Get control from interpolant
        control = control_evaluator(tau)
        if control.ndim > 1:
            control = control.flatten()  # Ensure 1D for dynamics

        # Convert to global coordinates and physical time
        global_tau = beta_k * tau + beta_k0
        physical_time = alpha * global_tau + alpha_0

        # Evaluate dynamics
        state_deriv = dynamics_function(state, control, physical_time, problem_parameters)
        state_deriv_np = np.array(state_deriv, dtype=np.float64).flatten()

        if state_deriv_np.shape[0] != num_states:
            raise ValueError(
                f"Dynamics function output dimension mismatch. Expected {num_states}, got {state_deriv_np.shape[0]}."
            )

        return overall_scaling * state_deriv_np

    # Forward simulation (IVP)
    initial_state = state_evaluator(-1.0)
    if initial_state.ndim > 1:
        initial_state = initial_state.flatten()

    fwd_tau_points = np.linspace(-1, 1, n_eval_points, dtype=np.float64)
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
    result.state_trajectory_from_forward_simulation = (
        fwd_sim.y
        if fwd_sim.success
        else np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64)
    )
    if not fwd_sim.success:
        print(f"    Fwd IVP fail int {interval_idx}: {fwd_sim.message}")

    result.nlp_state_trajectory_evaluated_at_forward_simulation_points = state_evaluator(
        fwd_tau_points
    )

    # Backward simulation (TVP)
    terminal_state = state_evaluator(1.0)
    if terminal_state.ndim > 1:
        terminal_state = terminal_state.flatten()

    bwd_tau_points = np.linspace(1, -1, n_eval_points, dtype=np.float64)
    bwd_sim = ode_solver(
        dynamics_rhs,
        t_span=(1, -1),
        y0=terminal_state,
        t_eval=bwd_tau_points,
        method="RK45",
        rtol=ode_rtol,
        atol=ode_rtol * 1e-2,
    )

    # Initialize sorted_bwd_tau_points to avoid unbound variable error
    sorted_bwd_tau_points = np.flip(bwd_tau_points)
    result.backward_simulation_local_tau_evaluation_points = sorted_bwd_tau_points
    result.state_trajectory_from_backward_simulation = (
        np.fliplr(bwd_sim.y)
        if bwd_sim.success
        else np.full((num_states, len(sorted_bwd_tau_points)), np.nan, dtype=np.float64)
    )
    if not bwd_sim.success:
        print(f"    Bwd TVP fail int {interval_idx}: {bwd_sim.message}")

    result.nlp_state_trajectory_evaluated_at_backward_simulation_points = state_evaluator(
        result.backward_simulation_local_tau_evaluation_points
    )

    result.are_forward_and_backward_simulations_successful = fwd_sim.success and bwd_sim.success
    return result


def calculate_relative_error_estimate(
    interval_idx: int, sim_bundle: IntervalSimulationBundle, gamma_factors: _FloatArray
) -> float:
    """Calculates the maximum relative error estimate for an interval."""
    # Check for failed simulations or unpopulated arrays
    if (
        not sim_bundle.are_forward_and_backward_simulations_successful
        or sim_bundle.state_trajectory_from_forward_simulation.size == 0
        or sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points.size == 0
        or sim_bundle.state_trajectory_from_backward_simulation.size == 0
        or sim_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points.size == 0
    ):
        print(
            f"    Interval {interval_idx}: Simulation results incomplete, failed, or arrays empty. Returning np.inf."
        )
        return float(np.inf)

    num_states = sim_bundle.state_trajectory_from_forward_simulation.shape[0]
    if (
        num_states == 0
    ):  # This check might need review if shape[0] is non-zero for an empty array after change
        return 0.0  # No states, no error

    # Forward errors
    fwd_diff = np.abs(
        sim_bundle.state_trajectory_from_forward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points
    )
    print(
        f"    DEBUG: Forward differences shape: {fwd_diff.shape}, Max raw diff: {np.max(fwd_diff) if fwd_diff.size > 0 else 'N/A (empty)'}"
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
        f"    DEBUG: Backward differences shape: {bwd_diff.shape}, Max raw diff: {np.max(bwd_diff) if bwd_diff.size > 0 else 'N/A (empty)'}"
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
    max_error = np.nanmax(max_errors_per_state) if max_errors_per_state.size > 0 else float(np.inf)

    # Ensure we never return exactly zero error (numerical tolerance)
    if max_error < 1e-15:
        max_error = 1e-15

    if np.isnan(max_error):
        print(
            f"    Interval {interval_idx}: Error calculation resulted in NaN. Treating as high error (np.inf)."
        )
        return float(np.inf)

    return float(max_error)


@dataclass
class PRefineResult:
    """Result of polynomial degree refinement."""

    actual_Nk_to_use: int
    was_p_successful: bool
    unconstrained_target_Nk: int


def p_refine_interval(
    current_Nk: int, max_error: float, error_tol: float, N_max: int
) -> PRefineResult:
    """Determines new polynomial degree using p-refinement."""
    # No refinement needed if error is within tolerance
    if max_error <= error_tol:
        return PRefineResult(
            actual_Nk_to_use=current_Nk,
            was_p_successful=False,
            unconstrained_target_Nk=current_Nk,
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
            actual_Nk_to_use=N_max,
            was_p_successful=False,
            unconstrained_target_Nk=target_Nk,
        )

    return PRefineResult(
        actual_Nk_to_use=target_Nk,
        was_p_successful=True,
        unconstrained_target_Nk=target_Nk,
    )


@dataclass
class HRefineResult:
    """Result of h-refinement (interval splitting)."""

    collocation_nodes_for_new_subintervals: list[int]
    num_new_subintervals: int


def h_refine_params(target_Nk: int, N_min: int) -> HRefineResult:
    """Determines parameters for h-refinement (splitting an interval)."""
    num_subintervals = max(2, int(np.ceil(target_Nk / N_min)))
    nodes_per_subinterval = [N_min] * num_subintervals

    return HRefineResult(
        collocation_nodes_for_new_subintervals=nodes_per_subinterval,
        num_new_subintervals=num_subintervals,
    )


def _map_global_normalized_tau_to_local_interval_tau(
    global_tau: _NormalizedTimePoint,
    global_start: _NormalizedTimePoint,
    global_end: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Maps global tau to local zeta in [-1, 1]. Inverse of Eq. 7 (mesh.txt)."""
    beta = (global_end - global_start) / 2.0
    beta0 = (global_end + global_start) / 2.0

    if abs(beta) < ZERO_TOLERANCE:
        return 0.0

    return float((global_tau - beta0) / beta)


def _map_local_interval_tau_to_global_normalized_tau(
    local_tau: _NormalizedTimePoint,
    global_start: _NormalizedTimePoint,
    global_end: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Maps local zeta in [-1, 1] to global tau. Eq. 7 (mesh.txt)."""
    beta = (global_end - global_start) / 2.0
    beta0 = (global_end + global_start) / 2.0

    return float(beta * local_tau + beta0)


def _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
    local_tau_k: _NormalizedTimePoint,
    global_start_k: _NormalizedTimePoint,
    global_shared: _NormalizedTimePoint,
    global_end_kp1: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Transforms zeta in interval k to zeta in interval k+1. Eq. 30 (mesh.txt)."""
    global_tau = _map_local_interval_tau_to_global_normalized_tau(
        local_tau_k, global_start_k, global_shared
    )
    return _map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_shared, global_end_kp1
    )


def _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
    local_tau_kp1: _NormalizedTimePoint,
    global_start_k: _NormalizedTimePoint,
    global_shared: _NormalizedTimePoint,
    global_end_kp1: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Transforms zeta in interval k+1 to zeta in interval k. Inverse of Eq. 30 (mesh.txt)."""
    global_tau = _map_local_interval_tau_to_global_normalized_tau(
        local_tau_kp1, global_shared, global_end_kp1
    )
    return _map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_start_k, global_shared
    )


def h_reduce_intervals(
    first_idx: int,
    solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    adaptive_params: AdaptiveParameters,
    gamma_factors: _FloatArray,
    state_evaluator_first: _InterpolantCallable,
    control_evaluator_first: _InterpolantCallable,
    state_evaluator_second: _InterpolantCallable,
    control_evaluator_second: _InterpolantCallable,
) -> bool:
    """
    Checks if two adjacent intervals can be merged.
    Returns True if merge is successful (error condition met).
    """
    # Initialize variables to prevent "possibly unbound" errors
    fwd_sim = None
    bwd_sim = None
    fwd_trajectory = None
    bwd_trajectory = None
    sorted_bwd_tau_points = None
    bwd_tau_points = None  # Explicitly initialize

    print(f"    h-reduction check for intervals {first_idx} and {first_idx+1}.")
    error_tol = adaptive_params.error_tolerance
    ode_rtol = adaptive_params.ode_solver_tolerance
    ode_atol = ode_rtol * 1e-1
    num_sim_points = adaptive_params.num_error_sim_points

    num_states = problem.num_states
    # num_controls = problem.num_controls # This was already commented as unused
    dynamics_function = problem.dynamics_function
    problem_parameters = problem.problem_parameters

    if solution.raw_solution is None:
        print("      h-reduction failed: Raw solution missing.")
        return False

    # Extract mesh and time information
    if solution.global_normalized_mesh_nodes is None:
        print("      h-reduction failed: Global mesh nodes missing.")
        return False

    global_mesh = np.array(solution.global_normalized_mesh_nodes, dtype=np.float64)

    if global_mesh.size <= first_idx + 2:
        print(
            f"      h-reduction failed: Global mesh too small for intervals {first_idx} and {first_idx+1}."
        )
        return False

    tau_start_k = global_mesh[first_idx]
    tau_shared = global_mesh[first_idx + 1]
    tau_end_kp1 = global_mesh[first_idx + 2]

    beta_k = (tau_shared - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared) / 2.0

    if abs(beta_k) < ZERO_TOLERANCE or abs(beta_kp1) < ZERO_TOLERANCE:
        print("      h-reduction check: One of the intervals has zero length. Merge not possible.")
        return False

    # Time transformation parameters
    t0 = solution.initial_time_variable
    tf = solution.terminal_time_variable

    if t0 is None or tf is None:
        print("      h-reduction failed: Initial or terminal time is None.")
        return False

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    def _get_control_value(
        control_evaluator: _InterpolantCallable | None, local_tau: _NormalizedTimePoint
    ) -> np.ndarray:
        """Get control value from evaluator, with clipping to handle boundary conditions."""
        if control_evaluator is None:  # Or if it's the dummy_evaluator for no controls
            return np.array([], dtype=np.float64)

        clipped_tau = np.clip(local_tau, -1.0, 1.0)
        u_val = control_evaluator(clipped_tau)
        return np.atleast_1d(u_val.squeeze())

    def merged_fwd_rhs(local_tau_k: float, state: np.ndarray) -> np.ndarray:
        """RHS for merged domain forward simulation."""
        u_val = _get_control_value(control_evaluator_first, local_tau_k)
        state_clipped = np.clip(state, -1e6, 1e6)
        global_tau = _map_local_interval_tau_to_global_normalized_tau(
            local_tau_k, tau_start_k, tau_shared
        )
        t_actual = alpha * global_tau + alpha_0

        f_rhs = dynamics_function(state_clipped, u_val, t_actual, problem_parameters)
        f_rhs_np = np.array(f_rhs, dtype=np.float64).flatten()

        return scaling_k * f_rhs_np

    def merged_bwd_rhs(local_tau_kp1: float, state: np.ndarray) -> np.ndarray:
        """RHS for merged domain backward simulation."""
        u_val = _get_control_value(control_evaluator_second, local_tau_kp1)
        state_clipped = np.clip(state, -1e6, 1e6)
        global_tau = _map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1, tau_shared, tau_end_kp1
        )
        t_actual = alpha * global_tau + alpha_0

        f_rhs = dynamics_function(state_clipped, u_val, t_actual, problem_parameters)
        f_rhs_np = np.array(f_rhs, dtype=np.float64).flatten()

        return scaling_kp1 * f_rhs_np

    # Get state values at interval endpoints
    try:
        Y_solved_list = solution.solved_state_trajectories_per_interval
        Nk_k = (
            problem.collocation_points_per_interval[first_idx]
            if problem.collocation_points_per_interval is not None
            else 0
        )
        Nk_kp1 = (
            problem.collocation_points_per_interval[first_idx + 1]
            if problem.collocation_points_per_interval is not None
            else 0
        )

        if Y_solved_list is not None and first_idx < len(Y_solved_list):
            Xk_nlp = Y_solved_list[first_idx]
        else:  # Fallback
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if (
                opti is not None
                and hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables")
                and opti.state_at_local_approximation_nodes_all_intervals_variables is not None
                and first_idx < len(opti.state_at_local_approximation_nodes_all_intervals_variables)
            ):
                Xk_nlp_raw = raw_sol.value(
                    opti.state_at_local_approximation_nodes_all_intervals_variables[first_idx]
                )
                Xk_nlp = _extract_and_prepare_array(Xk_nlp_raw, num_states, Nk_k + 1)
            else:
                print("      h-reduction failed: Cannot extract state data for first interval")
                return False

        initial_state_fwd = Xk_nlp[:, 0].flatten()
    except Exception as e:
        print(f"      h-reduction failed: Error getting initial state: {e}")
        return False

    # Forward simulation through merged domain
    target_end_tau_k = _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
        1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_fwd_pts = max(
        num_sim_points // 2,
        (
            int(num_sim_points * (target_end_tau_k - (-1.0)) / 2.0)
            if target_end_tau_k > -1.0
            else num_sim_points // 2
        ),
    )
    num_fwd_pts = max(2, num_fwd_pts)
    fwd_tau_points = np.linspace(-1.0, target_end_tau_k, num_fwd_pts, dtype=np.float64)

    print(
        f"      h-reduction: Starting Merged IVP sim from zeta_k=-1 to {target_end_tau_k:.3f} ({num_fwd_pts} pts)"
    )
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
        fwd_trajectory = (
            fwd_sim.y
            if fwd_sim.success
            else np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64)
        )
        if not fwd_sim.success:
            print(f"      Merged IVP failed: {fwd_sim.message}")
    except Exception as e:
        print(f"      Exception during merged IVP simulation: {e}")
        fwd_trajectory = np.full((num_states, len(fwd_tau_points)), np.nan, dtype=np.float64)

    # Get terminal state for backward simulation
    try:
        if Y_solved_list is not None and (first_idx + 1) < len(Y_solved_list):
            Xkp1_nlp = Y_solved_list[first_idx + 1]
        else:  # Fallback
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if (
                opti is not None
                and hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables")
                and opti.state_at_local_approximation_nodes_all_intervals_variables is not None
                and first_idx + 1
                < len(opti.state_at_local_approximation_nodes_all_intervals_variables)
            ):
                Xkp1_nlp_raw = raw_sol.value(
                    opti.state_at_local_approximation_nodes_all_intervals_variables[first_idx + 1]
                )
                Xkp1_nlp = _extract_and_prepare_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)
            else:
                print("      h-reduction failed: Cannot extract state data for second interval")
                return False

        terminal_state_bwd = Xkp1_nlp[:, -1].flatten()
    except Exception as e:
        print(f"      h-reduction failed: Error getting terminal state: {e}")
        return False

    # Backward simulation through merged domain
    target_end_tau_kp1 = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
        -1.0, tau_start_k, tau_shared, tau_end_kp1
    )
    num_bwd_pts = max(
        num_sim_points // 2,
        (
            int(num_sim_points * (1.0 - target_end_tau_kp1) / 2.0)
            if target_end_tau_kp1 < 1.0
            else num_sim_points // 2
        ),
    )
    num_bwd_pts = max(2, num_bwd_pts)
    bwd_tau_points = np.linspace(1.0, target_end_tau_kp1, num_bwd_pts, dtype=np.float64)

    print(
        f"      h-reduction: Starting Merged TVP sim from zeta_kp1=1 to {target_end_tau_kp1:.3f} ({num_bwd_pts} pts)"
    )
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
        sorted_bwd_tau_points = np.flip(bwd_tau_points)
        bwd_trajectory = (
            np.fliplr(bwd_sim.y)
            if bwd_sim.success
            else np.full((num_states, len(sorted_bwd_tau_points)), np.nan, dtype=np.float64)
        )
        if not bwd_sim.success:
            print(f"      Merged TVP failed: {bwd_sim.message}")
    except Exception as e:
        print(f"      Exception during merged TVP simulation: {e}")
        if bwd_tau_points is not None:  # Check if bwd_tau_points was initialized
            sorted_bwd_tau_points = np.flip(bwd_tau_points)
            bwd_trajectory = np.full(
                (num_states, len(sorted_bwd_tau_points)), np.nan, dtype=np.float64
            )
        else:  # Fallback if bwd_tau_points is somehow still None (should not happen with pre-initialization)
            sorted_bwd_tau_points = np.array([], dtype=np.float64)
            bwd_trajectory = np.full((num_states, 0), np.nan, dtype=np.float64)

    # For problems with no states, just check if simulations were successful
    if num_states == 0:
        can_merge = (
            fwd_sim is not None and fwd_sim.success and bwd_sim is not None and bwd_sim.success
        )
        print(f"      h-reduction check (no states): can_intervals_be_merged = {can_merge}")
        return can_merge

    # Calculate errors for merged domain
    all_fwd_errors: list[float] = []
    if (
        fwd_trajectory is not None and fwd_trajectory.size > 0
    ):  # Ensure fwd_trajectory is not None and not empty
        for i, zeta_k in enumerate(fwd_tau_points):
            X_sim = fwd_trajectory[:, i]
            if np.any(np.isnan(X_sim)):
                continue

            # Get NLP state for comparison
            if -1.0 <= zeta_k <= 1.0 + 1e-9:
                X_nlp = state_evaluator_first(zeta_k)
            else:
                zeta_kp1 = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                    zeta_k, tau_start_k, tau_shared, tau_end_kp1
                )
                X_nlp = state_evaluator_second(zeta_kp1)

            if X_nlp.ndim > 1:
                X_nlp = X_nlp.flatten()

            abs_diff = np.abs(X_sim - X_nlp)
            scaled_errors = gamma_factors.flatten() * abs_diff
            all_fwd_errors.extend(scaled_errors.tolist())

    all_bwd_errors: list[float] = []
    if (
        bwd_trajectory is not None
        and bwd_trajectory.size > 0
        and sorted_bwd_tau_points is not None
        and sorted_bwd_tau_points.size > 0
    ):  # Ensure bwd_trajectory and its points are not None/empty
        for i, zeta_kp1 in enumerate(sorted_bwd_tau_points):
            X_sim = bwd_trajectory[:, i]
            if np.any(np.isnan(X_sim)):
                continue

            # Get NLP state for comparison
            if -1.0 - 1e-9 <= zeta_kp1 <= 1.0:
                X_nlp = state_evaluator_second(zeta_kp1)
            else:
                zeta_k = _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
                    zeta_kp1, tau_start_k, tau_shared, tau_end_kp1
                )
                X_nlp = state_evaluator_first(zeta_k)

            if X_nlp.ndim > 1:
                X_nlp = X_nlp.flatten()

            abs_diff = np.abs(X_sim - X_nlp)
            scaled_errors = gamma_factors.flatten() * abs_diff
            all_bwd_errors.extend(scaled_errors.tolist())

    # Get maximum error for merged domain
    max_fwd_error = np.nanmax(all_fwd_errors) if all_fwd_errors else float(np.inf)
    max_bwd_error = np.nanmax(all_bwd_errors) if all_bwd_errors else float(np.inf)
    max_error = float(max(max_fwd_error, max_bwd_error))

    if np.isnan(max_error):
        print("      h-reduction check: max_error calculation resulted in NaN. Merge not approved.")
        max_error = float(np.inf)

    print(f"      h-reduction check result: max_error = {max_error:.4e}")
    can_merge = max_error <= error_tol

    if can_merge:
        print(
            f"      h-reduction condition met. Merge approved for intervals {first_idx}, {first_idx+1}."
        )
    else:
        print(
            f"      h-reduction condition NOT met (error {max_error:.2e} > tol {error_tol:.2e}). Merge failed."
        )

    return can_merge


@dataclass
class PReduceResult:
    """Result of p-reduction."""

    new_num_collocation_nodes: int
    was_reduction_applied: bool


def p_reduce_interval(
    current_Nk: int, max_error: float, error_tol: float, N_min: int, N_max: int
) -> PReduceResult:
    """Determines new polynomial degree for an interval using p-reduction."""
    # Only reduce if error is below tolerance and current Nk > minimum
    if max_error > error_tol or current_Nk <= N_min:
        return PReduceResult(new_num_collocation_nodes=current_Nk, was_reduction_applied=False)

    # Calculate reduction control parameter delta
    delta = float(N_min + N_max - current_Nk)
    if abs(delta) < 1e-9:
        delta = 1.0  # Avoid division by zero

    # Calculate number of nodes to remove
    if max_error < 1e-16:  # Error is essentially zero
        nodes_to_remove = current_Nk - N_min
    else:
        ratio = error_tol / max_error  # Should be >= 1
        if ratio < 1.0:
            nodes_to_remove = 0
        else:
            try:
                log_arg = np.power(ratio, 1.0 / delta)
                nodes_to_remove = np.floor(np.log10(log_arg)) if log_arg >= 1.0 else 0
            except (ValueError, OverflowError, ZeroDivisionError):
                nodes_to_remove = 0

    nodes_to_remove = max(0, int(nodes_to_remove))
    new_Nk = max(N_min, current_Nk - nodes_to_remove)
    was_reduced = new_Nk < current_Nk

    return PReduceResult(new_num_collocation_nodes=new_Nk, was_reduction_applied=was_reduced)


def _generate_robust_default_initial_guess(
    problem: OptimalControlProblem,
    collocation_nodes_list: list[int],
    initial_time_guess: float | None = None,
    terminal_time_guess: float | None = None,
    integral_values_guess: float | list[float] | None = None,
) -> Any:  # Return type will be InitialGuess, but avoid circular import
    """Generates a robust default initial guess with correct dimensions."""
    from trajectolab.direct_solver import InitialGuess

    # Removed unused variable: num_intervals = len(collocation_nodes_list)
    num_states = problem.num_states
    num_controls = problem.num_controls
    num_integrals = problem.num_integrals

    # Get default values
    default_state = getattr(problem.default_initial_guess_values, "state", 0.0)
    default_control = getattr(problem.default_initial_guess_values, "control", 0.0)

    # Initialize state and control trajectories
    states: list[np.ndarray] = []
    controls: list[np.ndarray] = []

    for _idx, Nk in enumerate(collocation_nodes_list):
        # State trajectory for this interval
        state_traj = np.full((num_states, Nk + 1), default_state, dtype=np.float64)
        states.append(state_traj)

        # Control trajectory for this interval
        if num_controls > 0:
            control_traj = np.full((num_controls, Nk), default_control, dtype=np.float64)
        else:
            control_traj = np.empty((0, Nk), dtype=np.float64)
        controls.append(control_traj)

    # Time variable guesses
    if initial_time_guess is None and problem.initial_guess:
        initial_time_guess = problem.initial_guess.initial_time_variable

    if terminal_time_guess is None and problem.initial_guess:
        terminal_time_guess = problem.initial_guess.terminal_time_variable

    # Integral guesses
    final_integral_guess = None
    if num_integrals > 0:
        if integral_values_guess is not None:
            final_integral_guess = integral_values_guess
        else:
            default_integral = getattr(problem.default_initial_guess_values, "integral", 0.0)
            if problem.initial_guess and problem.initial_guess.integrals is not None:
                raw_guess = problem.initial_guess.integrals
            else:
                raw_guess = (
                    [default_integral] * num_integrals if num_integrals > 1 else default_integral
                )

            if num_integrals == 1:
                final_integral_guess = (
                    float(raw_guess)
                    if not isinstance(raw_guess, (list, np.ndarray))
                    else float(raw_guess[0])
                )
            elif isinstance(raw_guess, (list, np.ndarray)) and len(raw_guess) == num_integrals:
                final_integral_guess = np.array(raw_guess, dtype=np.float64)
            else:
                final_integral_guess = np.full(num_integrals, default_integral, dtype=np.float64)

    return InitialGuess(
        initial_time_variable=initial_time_guess,
        terminal_time_variable=terminal_time_guess,
        states=states,
        controls=controls,
        integrals=final_integral_guess,
    )


def _propagate_guess_from_previous(
    prev_solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    target_nodes_list: list[int],
    target_mesh: np.ndarray,
) -> Any:  # Return type will be InitialGuess, but avoid circular import
    """Creates initial guess for current NLP, propagating from previous solution."""
    t0_prop = prev_solution.initial_time_variable
    tf_prop = prev_solution.terminal_time_variable
    integrals_prop = prev_solution.integrals

    # Generate default guess with propagated time and integral values
    guess = _generate_robust_default_initial_guess(
        problem,
        target_nodes_list,
        initial_time_guess=t0_prop,
        terminal_time_guess=tf_prop,
        integral_values_guess=integrals_prop,
    )

    # Check if mesh structure is identical to previous solution
    prev_nodes = prev_solution.num_collocation_nodes_list_at_solve_time
    prev_mesh = prev_solution.global_mesh_nodes_at_solve_time

    can_propagate_trajectories = False
    if prev_nodes is not None and prev_mesh is not None:
        if np.array_equal(target_nodes_list, prev_nodes) and np.allclose(target_mesh, prev_mesh):
            can_propagate_trajectories = True

    if can_propagate_trajectories:
        print(
            "  Mesh structure identical to previous. Propagating state/control trajectories directly."
        )
        prev_states = prev_solution.solved_state_trajectories_per_interval
        prev_controls = prev_solution.solved_control_trajectories_per_interval

        # Propagate state trajectories if available
        if prev_states and len(prev_states) == len(target_nodes_list):
            guess.states = prev_states
        else:
            print("    Warning: Previous states mismatch or missing. Using default states.")

        # Propagate control trajectories if available
        if prev_controls and len(prev_controls) == len(target_nodes_list):
            guess.controls = prev_controls
        else:
            print("    Warning: Previous controls mismatch or missing. Using default controls.")
    else:
        print(
            "  Mesh structure changed. Using robust default for state/control trajectories (times/integrals propagated)."
        )

    return guess


def _calculate_gamma_normalizers(
    solution: OptimalControlSolution, problem: OptimalControlProblem
) -> _FloatArray | None:
    """Calculates gamma_i normalization factors for error estimation."""
    if not solution.success or solution.raw_solution is None:
        return None  # Keep None for this specific return as it signals failure to calculate

    num_states = problem.num_states
    if num_states == 0:
        return np.array([], dtype=np.float64).reshape(0, 1)  # No states, no gamma

    Y_solved_list = solution.solved_state_trajectories_per_interval
    if not Y_solved_list:  # Checks if list is empty or None
        print("    Warning: solved_state_trajectories_per_interval missing for gamma calculation.")
        return None  # Keep None

    # Find maximum absolute value for each state component
    max_abs_values = np.zeros(num_states, dtype=np.float64)
    first_interval = True

    for Xk in Y_solved_list:
        if Xk.size == 0:  # Check for empty array
            continue

        max_abs_in_interval = np.max(np.abs(Xk), axis=1)

        if first_interval:
            max_abs_values = max_abs_in_interval
            first_interval = False
        else:
            max_abs_values = np.maximum(max_abs_values, max_abs_in_interval)

    # Calculate gamma factors
    gamma_denominator = 1.0 + max_abs_values
    gamma_factors = 1.0 / np.maximum(gamma_denominator, ZERO_TOLERANCE)  # Avoid division by zero

    return cast(_FloatArray, gamma_factors.reshape(-1, 1))


class PHSAdaptive(AdaptiveBase):
    """Implements the PHS-Adaptive mesh refinement algorithm."""

    def __init__(
        self,
        error_tolerance: float = 1e-3,
        max_iterations: int = 30,
        min_polynomial_degree: int = 4,
        max_polynomial_degree: int = 16,
        ode_solver_tolerance: float = 1e-7,
        num_error_sim_points: int = 40,
        initial_polynomial_degrees: list[int] | None = None,
        initial_mesh_points: _MeshNodesList | None = None,
        initial_guess: Any = None,  # Type is InitialGuess but avoid circular import
    ) -> None:
        """
        Initialize the PHS-Adaptive mesh refinement algorithm.

        Args:
            error_tolerance: Error tolerance threshold for mesh refinement
            max_iterations: Maximum number of refinement iterations
            min_polynomial_degree: Minimum polynomial degree allowed
            max_polynomial_degree: Maximum polynomial degree allowed
            ode_solver_tolerance: ODE solver tolerance for error estimation
            num_error_sim_points: Number of simulation points for error estimation
            initial_polynomial_degrees: Initial list of polynomial degrees for each interval
            initial_mesh_points: Initial mesh points in normalized time domain [-1, 1]
            initial_guess: Optional initial guess for the solver

        Note:
            If both initial_polynomial_degrees and initial_mesh_points are provided,
            they must be consistent: len(initial_polynomial_degrees) == len(initial_mesh_points) - 1
        """
        super().__init__(initial_guess)
        self.adaptive_params = AdaptiveParameters(
            error_tolerance=error_tolerance,
            max_iterations=max_iterations,
            min_polynomial_degree=min_polynomial_degree,
            max_polynomial_degree=max_polynomial_degree,
            ode_solver_tolerance=ode_solver_tolerance,
            num_error_sim_points=num_error_sim_points,
        )

        # Validate initial mesh configuration if provided
        if initial_polynomial_degrees is not None and initial_mesh_points is not None:
            if len(initial_polynomial_degrees) != len(initial_mesh_points) - 1:
                raise ValueError(
                    "Number of polynomial degrees must be one less than number of mesh points"
                )

        self._initial_polynomial_degrees = initial_polynomial_degrees
        self._initial_mesh_points = initial_mesh_points

    def run(
        self,
        problem: Any,  # Using Any for compatibility with AdaptiveBase
        legacy_problem: OptimalControlProblem,
        initial_solution: OptimalControlSolution | None = None,
    ) -> OptimalControlSolution:
        """Run the PHS-Adaptive mesh refinement algorithm."""
        # 'problem' (first arg) is unused in this method, but kept for potential API compatibility with AdaptiveBase.
        # 'legacy_problem' is the one actually used.
        error_tol = self.adaptive_params.error_tolerance
        max_iterations = self.adaptive_params.max_iterations
        N_min = self.adaptive_params.min_polynomial_degree
        N_max = self.adaptive_params.max_polynomial_degree
        ode_rtol = self.adaptive_params.ode_solver_tolerance
        num_sim_points = self.adaptive_params.num_error_sim_points

        # Initialize mesh configuration
        if self._initial_polynomial_degrees is not None:
            current_nodes_list = list(self._initial_polynomial_degrees)
            current_mesh = (
                np.array(self._initial_mesh_points, dtype=np.float64)
                if self._initial_mesh_points is not None
                else np.linspace(-1, 1, len(current_nodes_list) + 1, dtype=np.float64)
            )
        else:
            current_nodes_list = list(legacy_problem.collocation_points_per_interval or [])
            if not current_nodes_list:  # Ensure at least one interval
                current_nodes_list = [N_min]

            if legacy_problem.global_normalized_mesh_nodes is not None:
                current_mesh = np.array(
                    legacy_problem.global_normalized_mesh_nodes, dtype=np.float64
                )
            else:
                current_mesh = np.linspace(-1, 1, len(current_nodes_list) + 1, dtype=np.float64)

        # Enforce node count limits
        current_nodes_list = [max(N_min, min(N_max, nk)) for nk in current_nodes_list]

        current_problem_definition = legacy_problem  # Use the correctly named problem object
        most_recent_solution = initial_solution
        from trajectolab.direct_solver import solve_single_phase_radau_collocation

        # Main adaptive refinement loop
        for iteration in range(max_iterations):
            print(f"\n--- Adaptive Iteration M = {iteration} ---")
            num_intervals = len(current_nodes_list)

            # Update problem definition with current mesh
            current_problem_definition.collocation_points_per_interval = list(current_nodes_list)
            current_problem_definition.global_normalized_mesh_nodes = list(current_mesh)

            if iteration == 0:
                initial_guess_for_solver = (
                    self.initial_guess
                    if self.initial_guess is not None
                    else _generate_robust_default_initial_guess(
                        current_problem_definition, current_nodes_list
                    )
                )
                if self.initial_guess is not None:
                    print("  Using user-provided initial guess for first iteration.")
                else:
                    print("  No user-provided initial guess. Using robust default.")

            elif not most_recent_solution or not most_recent_solution.success:
                print("  Previous NLP failed. Using robust default initial guess.")
                initial_guess_for_solver = _generate_robust_default_initial_guess(
                    current_problem_definition, current_nodes_list
                )
            else:
                initial_guess_for_solver = _propagate_guess_from_previous(
                    most_recent_solution,
                    current_problem_definition,
                    current_nodes_list,
                    current_mesh,
                )
            current_problem_definition.initial_guess = initial_guess_for_solver

            print(f"  Mesh K={num_intervals}, num_nodes_per_interval = {current_nodes_list}")
            print(f"  Mesh nodes_tau_global = {np.array2string(current_mesh, precision=3)}")

            solution = solve_single_phase_radau_collocation(current_problem_definition)

            if not solution.success:
                error_msg = f"NLP solver failed in adaptive iteration {iteration}. " + (
                    solution.message or "Solver error."
                )
                print(f"  Error: {error_msg} Stopping.")
                # Return the most recent successful solution if available, else the current failed one
                if most_recent_solution:
                    most_recent_solution.message = error_msg
                    most_recent_solution.success = (
                        False  # Mark it as ultimately failed due to this error
                    )
                    return most_recent_solution
                else:
                    # solution.message = error_msg # Already set by solver or above
                    return solution

            for attr_name, default_value in [
                ("solved_state_trajectories_per_interval", []),
                ("solved_control_trajectories_per_interval", []),
                ("num_collocation_nodes_list_at_solve_time", None),
                ("global_mesh_nodes_at_solve_time", None),
            ]:
                if not hasattr(solution, attr_name):
                    setattr(solution, attr_name, default_value)

            try:
                opti = solution.opti_object
                raw_sol = solution.raw_solution

                if (
                    opti is not None
                    and raw_sol is not None
                    and hasattr(
                        opti,
                        "state_at_local_approximation_nodes_all_intervals_variables",
                    )
                    and opti.state_at_local_approximation_nodes_all_intervals_variables is not None
                ):
                    solution.solved_state_trajectories_per_interval = [
                        _extract_and_prepare_array(
                            raw_sol.value(var),
                            current_problem_definition.num_states,
                            current_nodes_list[i] + 1,
                        )
                        for i, var in enumerate(
                            opti.state_at_local_approximation_nodes_all_intervals_variables
                        )
                    ]

                if current_problem_definition.num_controls > 0:
                    if (
                        opti is not None
                        and raw_sol is not None
                        and hasattr(
                            opti,
                            "control_at_local_collocation_nodes_all_intervals_variables",
                        )
                        and opti.control_at_local_collocation_nodes_all_intervals_variables
                        is not None
                    ):
                        solution.solved_control_trajectories_per_interval = [
                            _extract_and_prepare_array(
                                raw_sol.value(var),
                                current_problem_definition.num_controls,
                                current_nodes_list[i],
                            )
                            for i, var in enumerate(
                                opti.control_at_local_collocation_nodes_all_intervals_variables
                            )
                        ]
                else:  # No controls
                    solution.solved_control_trajectories_per_interval = [
                        np.empty((0, current_nodes_list[i]), dtype=np.float64)
                        for i in range(num_intervals)
                    ]

            except Exception as e:
                error_msg = f"Failed to extract trajectories from NLP solution at iter {iteration}: {e}. Stopping."
                print(f"  Error: {error_msg}")
                solution.message = error_msg
                solution.success = False
                return solution

            most_recent_solution = solution
            # Store the current mesh configuration in the solution
            most_recent_solution.num_collocation_nodes_list_at_solve_time = list(current_nodes_list)
            most_recent_solution.global_mesh_nodes_at_solve_time = np.copy(current_mesh)

            gamma_factors = _calculate_gamma_normalizers(solution, current_problem_definition)
            if gamma_factors is None and current_problem_definition.num_states > 0:
                error_msg = f"Failed to calculate gamma normalizers at iter {iteration}. Stopping."
                print(f"  Error: {error_msg}")
                solution.message = error_msg
                solution.success = False
                return solution

            basis_cache: dict[int, Any] = {}
            control_weights_cache: dict[int, _BarycentricWeights] = {}
            # Initialize with dummy_evaluator to give a compatible initial type (callable)
            state_evaluators: list[_InterpolantCallable] = [
                dummy_evaluator for _ in range(num_intervals)
            ]
            control_evaluators: list[_InterpolantCallable] = [
                dummy_evaluator for _ in range(num_intervals)
            ]

            states_list = solution.solved_state_trajectories_per_interval
            controls_list = solution.solved_control_trajectories_per_interval

            for k in range(num_intervals):
                try:
                    Nk = current_nodes_list[k]
                    if Nk not in basis_cache:
                        basis_cache[Nk] = compute_radau_collocation_components(Nk)
                    basis = basis_cache[Nk]

                    if (
                        states_list
                        and k < len(states_list)
                        and states_list[k] is not None
                        and states_list[k].size > 0
                    ):
                        state_evaluators[k] = get_polynomial_interpolant(
                            basis.state_approximation_nodes,
                            states_list[k],
                            basis.barycentric_weights_for_state_nodes,
                        )
                    else:
                        state_evaluators[k] = get_polynomial_interpolant(  # Fallback with NaN
                            np.array([-1.0, 1.0], dtype=np.float64),
                            np.full(
                                (current_problem_definition.num_states, 2), np.nan, dtype=np.float64
                            ),
                            None,
                        )

                    if current_problem_definition.num_controls > 0:
                        if (
                            controls_list
                            and k < len(controls_list)
                            and controls_list[k] is not None
                            and controls_list[k].size > 0
                        ):
                            if Nk not in control_weights_cache:
                                control_weights_cache[Nk] = compute_barycentric_weights(
                                    basis.collocation_nodes
                                )
                            control_evaluators[k] = get_polynomial_interpolant(
                                basis.collocation_nodes,
                                controls_list[k],
                                control_weights_cache[Nk],
                            )
                        else:
                            control_evaluators[k] = get_polynomial_interpolant(  # Fallback with NaN
                                np.array([-1.0, 1.0], dtype=np.float64),
                                np.full(
                                    (current_problem_definition.num_controls, 2),
                                    np.nan,
                                    dtype=np.float64,
                                ),
                                None,
                            )
                    else:
                        control_evaluators[k] = get_polynomial_interpolant(
                            np.array([-1.0, 1.0], dtype=np.float64),
                            np.empty((0, 2), dtype=np.float64),
                            None,
                        )

                except Exception as e:
                    print(f"  Warning: Error creating interpolant for interval {k}: {e}")
                    state_evaluators[k] = get_polynomial_interpolant(
                        np.array([-1.0, 1.0], dtype=np.float64),
                        np.full(
                            (current_problem_definition.num_states, 2), np.nan, dtype=np.float64
                        ),
                        None,
                    )
                    control_evaluators[k] = get_polynomial_interpolant(
                        np.array([-1.0, 1.0], dtype=np.float64),
                        np.full(
                            (
                                (
                                    current_problem_definition.num_controls
                                    if current_problem_definition.num_controls > 0
                                    else 0
                                ),
                                2,
                            ),
                            np.nan,
                            dtype=np.float64,
                        ),
                        None,
                    )

            errors = [float(np.inf)] * num_intervals
            for k in range(num_intervals):
                print(f"  Starting error simulation for interval {k}...")
                # state_eval and control_eval are guaranteed to be callable due to initialization
                state_eval = state_evaluators[k]
                control_eval = control_evaluators[k]

                sim_bundle = _simulate_dynamics_for_error_estimation(
                    k,
                    solution,
                    current_problem_definition,
                    state_eval,
                    control_eval,
                    ode_rtol=ode_rtol,
                    n_eval_points=num_sim_points,
                )

                gamma_safe = (
                    gamma_factors
                    if gamma_factors is not None
                    else np.array([], dtype=np.float64).reshape(0, 1)
                )
                error = calculate_relative_error_estimate(k, sim_bundle, gamma_safe)
                errors[k] = error
                print(f"    Interval {k}: Nk={current_nodes_list[k]}, Error={error:.4e}")

            print(f"  Overall errors: {[f'{e:.2e}' for e in errors]}")

            all_errors_ok = (
                all(e <= error_tol for e in errors if not (np.isnan(e) or np.isinf(e)))
                if errors
                else True
            )

            if all_errors_ok:
                print(f"Mesh converged after {iteration+1} iterations.")
                solution.num_collocation_nodes_per_interval = current_nodes_list.copy()
                solution.global_normalized_mesh_nodes = np.copy(current_mesh)
                solution.message = f"Adaptive mesh converged to tolerance {error_tol:.1e} in {iteration+1} iterations."
                return solution

            next_nodes_list: list[int] = []
            next_mesh = [current_mesh[0]]
            k = 0
            while k < num_intervals:
                error_k = errors[k]
                Nk = current_nodes_list[k]
                print(f"    Processing interval {k}: Nk={Nk}, Error={error_k:.2e}")

                if np.isnan(error_k) or np.isinf(error_k) or error_k > error_tol:
                    print(f"      Interval {k} error > tol. Attempting p-refinement.")
                    p_result = p_refine_interval(Nk, error_k, error_tol, N_max)
                    if p_result.was_p_successful:
                        print(
                            f"        p-refinement applied: Nk {Nk} -> {p_result.actual_Nk_to_use}"
                        )
                        next_nodes_list.append(p_result.actual_Nk_to_use)
                        next_mesh.append(current_mesh[k + 1])
                        k += 1
                    else:
                        print("        p-refinement failed. Attempting h-refinement.")
                        h_result = h_refine_params(p_result.unconstrained_target_Nk, N_min)
                        print(
                            f"          h-refinement: Splitting interval {k} into {h_result.num_new_subintervals} subintervals, each Nk={h_result.collocation_nodes_for_new_subintervals[0]}."
                        )
                        next_nodes_list.extend(h_result.collocation_nodes_for_new_subintervals)
                        new_nodes_for_split = np.linspace(
                            current_mesh[k],
                            current_mesh[k + 1],
                            h_result.num_new_subintervals + 1,
                            dtype=np.float64,
                        )
                        next_mesh.extend(list(new_nodes_for_split[1:]))
                        k += 1
                else:  # Error <= tolerance
                    print(f"    Interval {k} error <= tol.")
                    can_merge = False
                    if k < num_intervals - 1:  # Check eligibility for h-reduction
                        next_error = errors[k + 1]
                        if (
                            not (np.isnan(next_error) or np.isinf(next_error))
                            and next_error <= error_tol
                        ):
                            print(
                                f"      Interval {k+1} also has low error ({next_error:.2e}). Eligible for h-reduction."
                            )
                            state_eval_1 = state_evaluators[k]
                            control_eval_1 = control_evaluators[k]
                            state_eval_2 = state_evaluators[k + 1]
                            control_eval_2 = control_evaluators[k + 1]

                            gamma_safe = (
                                gamma_factors
                                if gamma_factors is not None
                                else np.array([], dtype=np.float64).reshape(0, 1)
                            )
                            can_merge = h_reduce_intervals(
                                k,
                                solution,
                                current_problem_definition,
                                self.adaptive_params,
                                gamma_safe,
                                state_eval_1,
                                control_eval_1,
                                state_eval_2,
                                control_eval_2,
                            )

                    if can_merge:
                        print(f"      h-reduction applied to merge interval {k} and {k+1}.")
                        merged_Nk = max(current_nodes_list[k], current_nodes_list[k + 1])
                        merged_Nk = max(N_min, min(N_max, merged_Nk))
                        next_nodes_list.append(merged_Nk)
                        next_mesh.append(current_mesh[k + 2])
                        k += 2
                    else:  # h-reduction failed or not applicable, try p-reduction
                        print(
                            f"      h-reduction failed or not applicable. Attempting p-reduction for interval {k}."
                        )
                        p_reduce_res = p_reduce_interval(Nk, error_k, error_tol, N_min, N_max)
                        if p_reduce_res.was_reduction_applied:
                            print(
                                f"        p-reduction applied: Nk {Nk} -> {p_reduce_res.new_num_collocation_nodes}"
                            )
                        else:
                            print(f"        p-reduction not applied for Nk {Nk}.")
                        next_nodes_list.append(p_reduce_res.new_num_collocation_nodes)
                        next_mesh.append(current_mesh[k + 1])
                        k += 1

            current_nodes_list = next_nodes_list
            current_mesh = np.array(next_mesh, dtype=np.float64)

            # Mesh sanity checks
            if not hasattr(most_recent_solution, "num_collocation_nodes_per_interval"):
                most_recent_solution.num_collocation_nodes_per_interval = []
            if not hasattr(most_recent_solution, "global_normalized_mesh_nodes"):
                most_recent_solution.global_normalized_mesh_nodes = None

            most_recent_solution.num_collocation_nodes_per_interval = current_nodes_list
            most_recent_solution.global_normalized_mesh_nodes = current_mesh

            if not current_nodes_list and len(current_mesh) > 1:
                error_msg = "Stopped due to mesh inconsistency (empty collocation_nodes but mesh_nodes exist)."
                print(f"  Error: {error_msg} Stopping.")
                most_recent_solution.message = error_msg
                most_recent_solution.success = False
                return most_recent_solution

            if current_nodes_list and len(current_nodes_list) != (len(current_mesh) - 1):
                error_msg = f"Mesh structure inconsistent. num_nodes_list len: {len(current_nodes_list)}, mesh_nodes len-1: {len(current_mesh)-1}."
                print(f"  Error: {error_msg} Stopping.")
                most_recent_solution.message = error_msg
                most_recent_solution.success = False
                return most_recent_solution

            if len(current_nodes_list) > 0:
                unique_nodes, counts = np.unique(
                    np.round(current_mesh, decimals=12), return_counts=True
                )
                if np.any(counts > 1):
                    duplicates = unique_nodes[counts > 1]
                    error_msg = (
                        f"Duplicate mesh nodes found: {duplicates}. Original nodes: {current_mesh}."
                    )
                    print(f"  Error: {error_msg} Stopping.")
                    most_recent_solution.message = error_msg
                    most_recent_solution.success = False
                    return most_recent_solution

                if len(unique_nodes) > 1 and not np.all(np.diff(unique_nodes) > 1e-9):
                    diffs = np.diff(unique_nodes)
                    problem_indices = np.where(diffs <= 1e-9)[0]
                    problem_pairs = (
                        ", ".join(
                            [
                                f"({unique_nodes[i]:.3f}, {unique_nodes[i+1]:.3f})"
                                for i in problem_indices
                            ]
                        )
                        if problem_indices.size > 0
                        else "N/A"
                    )
                    error_msg = f"Mesh nodes not strictly increasing or interval too small. Problem pairs: {problem_pairs}. All nodes: {current_mesh}."
                    print(f"  Error: {error_msg} Stopping.")
                    most_recent_solution.message = error_msg
                    most_recent_solution.success = False
                    return most_recent_solution

        # Max iterations reached
        max_iter_msg = f"Adaptive mesh refinement reached max iterations ({max_iterations}) without full convergence to tolerance {error_tol:.1e}."
        print(max_iter_msg)

        if most_recent_solution:
            most_recent_solution.message = max_iter_msg
            if not hasattr(most_recent_solution, "num_collocation_nodes_per_interval"):
                most_recent_solution.num_collocation_nodes_per_interval = []
            if not hasattr(most_recent_solution, "global_normalized_mesh_nodes"):
                most_recent_solution.global_normalized_mesh_nodes = None

            most_recent_solution.num_collocation_nodes_per_interval = current_nodes_list.copy()
            most_recent_solution.global_normalized_mesh_nodes = np.copy(current_mesh)
            return most_recent_solution
        else:
            from trajectolab.direct_solver import OptimalControlSolution

            failed = OptimalControlSolution()
            failed.success = False
            failed.message = (
                max_iter_msg + " No successful NLP solution obtained throughout iterations."
            )
            failed.num_collocation_nodes_per_interval = current_nodes_list
            failed.global_normalized_mesh_nodes = current_mesh.tolist()
            return failed
