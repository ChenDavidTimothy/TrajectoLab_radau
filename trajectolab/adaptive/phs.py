import logging  # Added
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from trajectolab.adaptive.base import AdaptiveBase
from trajectolab.radau import (
    compute_barycentric_weights,
    compute_radau_collocation_components,
    evaluate_lagrange_polynomial_at_point,
)
from trajectolab.trajectolab_types import (
    CasADiDM,
    CasADiOpti,
    InitialGuess,
    NumControls,
    NumIntegrals,
    NumStates,
    OptimalControlProblem,
    OptimalControlSolution,
    RadauBasisComponents,
    _FloatArray,  # Maintained, assuming specific usage defined in trajectolab_types
    _Matrix,
    _MeshPoints,
    _NormalizedTimePoint,
    _Vector,
)

# Setup logging
logger = logging.getLogger(__name__)
# Example: To see logs, uncomment and configure:
# logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_STATE_CLIP_VALUE = 1e6  # Added for h_reduce_intervals


@dataclass
class AdaptiveParameters:
    error_tolerance: float
    max_iterations: int
    min_polynomial_degree: int
    max_polynomial_degree: int
    ode_solver_tolerance: float = 1e-7
    num_error_sim_points: int = 50


@dataclass
class IntervalSimulationBundle:
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
        for field_name, field_val in self.__dict__.items():
            if not isinstance(field_val, np.ndarray) or field_val.size == 0:
                continue

            current_val = field_val
            if current_val.dtype != np.float64:
                current_val = current_val.astype(np.float64)

            if current_val.ndim == 1:
                # Ensure 2D for _Matrix type, typically as a row vector (1, N)
                # or column vector (N,1) if context implies.
                # Here, default to (1,N) if it's for evaluation points,
                # or needs specific reshape if it's state/control data.
                # For safety, this bundle expects specific shapes from creation.
                # This reshape is a fallback.
                setattr(self, field_name, current_val.reshape(1, -1))
            elif current_val.ndim > 2:
                raise ValueError(
                    f"Field {field_name} must be 1D or 2D array, got {current_val.ndim}D."
                )
            elif current_val.ndim == 2 and current_val.dtype != np.float64:
                setattr(self, field_name, current_val)


def _extract_and_prepare_array(
    casadi_value: Union[CasADiDM, ca.MX, ca.SX, _FloatArray, List[Any], Tuple[Any, ...]],
    expected_rows: int,
    expected_cols: int,
) -> _Matrix:
    """
    Extracts numerical value from CasADi/Python types and ensures correct 2D shape
    and float64 dtype.
    """
    np_array_intermediate: np.ndarray

    if isinstance(casadi_value, (ca.MX, ca.SX)):
        if casadi_value.is_constant() and not casadi_value.is_symbolic():
            try:
                # Prefer to_numpy() if available and appropriate, or to_DM().toarray()
                dm_val = ca.DM(casadi_value)
                np_array_intermediate = dm_val.toarray()
            except Exception as e:
                logger.warning(
                    f"Could not convert symbolic CasADi type {type(casadi_value)} to DM/numpy directly: {e}. Value: {casadi_value}"
                )
                np_array_intermediate = np.array([], dtype=np.float64)
        elif casadi_value.is_symbolic():
            logger.warning(
                f"Attempting to convert symbolic CasADi type {type(casadi_value)} to NumPy. This may fail or yield unexpected results. Value: {casadi_value}"
            )
            # This path is problematic if a numeric array is truly expected.
            # Trying a direct conversion, but it's risky for symbolic expressions.
            try:
                np_array_intermediate = np.array(CasADiDM(casadi_value), dtype=np.float64)
            except (RuntimeError, TypeError, ValueError):
                np_array_intermediate = np.array([], dtype=np.float64)

        else:  # Non-symbolic, non-constant MX/SX (e.g., from opti.value on an expression)
            try:
                # This path might be hit if CasADi MX/SX wraps a DM
                dm_val = ca.DM(casadi_value)
                np_array_intermediate = dm_val.toarray()
            except Exception as e:
                logger.warning(
                    f"Could not convert non-symbolic CasADi type {type(casadi_value)} to DM/numpy: {e}. Value: {casadi_value}"
                )
                np_array_intermediate = np.array([], dtype=np.float64)

    elif isinstance(casadi_value, CasADiDM):
        np_array_intermediate = casadi_value.toarray()  # DM.toarray() is robust
    elif isinstance(casadi_value, np.ndarray):
        np_array_intermediate = casadi_value
    elif isinstance(casadi_value, (List, Tuple)):
        np_array_intermediate = np.array(casadi_value, dtype=np.float64)
    else:
        raise TypeError(f"Unsupported type for casadi_value: {type(casadi_value)}")

    # Ensure float64
    if np_array_intermediate.dtype != np.float64:
        np_array_intermediate = np_array_intermediate.astype(np.float64)

    # Handle empty cases
    if expected_rows == 0 and expected_cols == 0:  # Requesting a 0x0 array
        return np.empty((0, 0), dtype=np.float64)
    if expected_rows == 0 and np_array_intermediate.size > 0:  # Expecting 0 rows, but got data
        # This case might indicate num_states=0 but data is present
        # Or num_controls=0 but data is present.
        # Return (0, actual_cols) or (0,0)
        return np.empty(
            (0, np_array_intermediate.shape[1] if np_array_intermediate.ndim == 2 else 0),
            dtype=np.float64,
        )
    if np_array_intermediate.size == 0:
        return np.empty((expected_rows, expected_cols), dtype=np.float64)

    # Ensure 2D shape
    if np_array_intermediate.ndim == 1:
        # Try to match expected_rows or expected_cols if one of them is 1
        if expected_rows == 1 and (
            expected_cols == 0 or len(np_array_intermediate) == expected_cols
        ):
            np_array_intermediate = np_array_intermediate.reshape(1, -1)
        elif expected_cols == 1 and (
            expected_rows == 0 or len(np_array_intermediate) == expected_rows
        ):
            np_array_intermediate = np_array_intermediate.reshape(-1, 1)
        # If unambiguous (e.g. expected_rows=1, array has N elements, make it 1xN)
        elif expected_rows == 1 and np_array_intermediate.size == expected_cols:
            np_array_intermediate = np_array_intermediate.reshape(1, expected_cols)
        elif expected_cols == 1 and np_array_intermediate.size == expected_rows:
            np_array_intermediate = np_array_intermediate.reshape(expected_rows, 1)
        else:
            # Default to row vector if shape is ambiguous for 1D->2D
            logger.debug(
                f"Ambiguous 1D to 2D conversion. Expected ({expected_rows},{expected_cols}), got 1D array of len {len(np_array_intermediate)}. Defaulting to row vector if elements match, else error."
            )
            if (
                np_array_intermediate.size == expected_cols and expected_rows == 1
            ):  # Common case for single state/control trajectory
                np_array_intermediate = np_array_intermediate.reshape(1, expected_cols)
            elif np_array_intermediate.size == expected_rows and expected_cols == 1:
                np_array_intermediate = np_array_intermediate.reshape(expected_rows, 1)
            elif (
                np_array_intermediate.size == expected_rows * expected_cols
            ):  # if total elements match
                np_array_intermediate = np_array_intermediate.reshape(expected_rows, expected_cols)
            else:
                # Cannot unambiguously reshape, this will likely fail the next check.
                # We force it to be a row vector for now.
                np_array_intermediate = np_array_intermediate.reshape(1, -1)

    # Final shape check and adjustment
    if (
        np_array_intermediate.shape == (expected_cols, expected_rows)
        and expected_rows != expected_cols
    ):
        # If shape is (B,A) but expected (A,B) and not square, transpose.
        np_array_intermediate = np_array_intermediate.T

    if np_array_intermediate.shape != (expected_rows, expected_cols):
        # If total elements match, attempt a final reshape.
        if np_array_intermediate.size == expected_rows * expected_cols:
            try:
                np_array_intermediate = np_array_intermediate.reshape(expected_rows, expected_cols)
            except ValueError as e:
                logger.error(
                    f"Final shape mismatch and reshape failed for _extract_and_prepare_array. "
                    f"Input type: {type(casadi_value)}, Initial shape: {np_array_intermediate.shape}, "
                    f"Target shape: ({expected_rows},{expected_cols}). Error: {e}"
                )
                raise ValueError(
                    f"Array shape mismatch: Expected ({expected_rows},{expected_cols}), "
                    f"got {np_array_intermediate.shape} after processing. Input: {casadi_value}"
                ) from e
        else:
            logger.error(
                f"Final shape and size mismatch for _extract_and_prepare_array. "
                f"Input type: {type(casadi_value)}, Processed shape: {np_array_intermediate.shape}, Size: {np_array_intermediate.size}, "
                f"Target shape: ({expected_rows},{expected_cols}), Target size: {expected_rows * expected_cols}."
            )
            raise ValueError(
                f"Array shape/size mismatch: Expected ({expected_rows},{expected_cols}), "
                f"got {np_array_intermediate.shape}. Input: {casadi_value}"
            )
    return np_array_intermediate


class PolynomialInterpolant:
    values_at_nodes: _Matrix
    nodes_array: _Vector
    bary_weights: _Vector
    num_vars: int
    num_nodes_val: int  # Number of points where values are provided (columns in values_at_nodes)
    num_nodes_pts: int  # Number of distinct nodes (length of nodes_array)

    def __init__(
        self, nodes: _Vector, values: _Matrix, barycentric_weights: Optional[_Vector] = None
    ):
        self.nodes_array = np.asarray(nodes, dtype=np.float64).flatten()  # Ensure 1D

        # Ensure values_at_nodes is 2D and float64
        _values = np.asarray(values, dtype=np.float64)
        if _values.ndim == 0 and _values.size == 1:  # scalar input
            _values = _values.reshape(1, 1)
        elif _values.ndim == 1:
            # Attempt to guess if it's a single variable over multiple nodes (1, N)
            # or multiple variables at a single node (M, 1).
            # Default to (1, N) if nodes array suggests N points.
            if len(self.nodes_array) == _values.size:
                _values = _values.reshape(1, -1)
            else:  # Could be (M,1) or error. For safety, assume (M,1) if nodes_array has 1 point.
                _values = _values.reshape(-1, 1)

        self.values_at_nodes = _values
        self.num_vars, self.num_nodes_val = self.values_at_nodes.shape
        self.num_nodes_pts = len(self.nodes_array)

        if self.num_nodes_pts == 0 and self.num_vars > 0 and self.num_nodes_val > 0:
            raise ValueError(
                "Cannot create interpolant with non-empty values but empty nodes array."
            )
        if (
            self.num_nodes_pts > 0
            and self.num_vars > 0
            and self.num_nodes_pts != self.num_nodes_val
        ):
            raise ValueError(
                f"Mismatch in number of nodes ({self.num_nodes_pts}) and "
                f"columns in values_at_nodes ({self.num_nodes_val})"
            )

        if barycentric_weights is None:
            if self.num_nodes_pts > 0:
                self.bary_weights = compute_barycentric_weights(self.nodes_array)
            else:  # No nodes, no weights
                self.bary_weights = np.array([], dtype=np.float64)
        else:
            self.bary_weights = np.asarray(barycentric_weights, dtype=np.float64)

        if self.num_nodes_pts > 0 and len(self.bary_weights) != self.num_nodes_pts:
            raise ValueError("Barycentric weights length does not match nodes length")

    def __call__(self, points: Union[_NormalizedTimePoint, _Vector]) -> Union[_Matrix, _Vector]:
        is_scalar_input_point = np.isscalar(points)
        zeta_arr: _Vector = np.atleast_1d(np.asarray(points, dtype=np.float64))

        if (
            self.num_vars == 0 or self.num_nodes_pts == 0
        ):  # No variables to interpolate or no basis nodes
            empty_shape_matrix = (self.num_vars, len(zeta_arr))
            # For scalar input with single var, result should be scalar. Here num_vars is 0.
            # If input scalar, output shape should be (num_vars,) which is (0,)
            # If input vector, output shape should be (num_vars, len(zeta_arr))
            return np.empty(
                (self.num_vars,) if is_scalar_input_point else empty_shape_matrix,
                dtype=np.float64,
            )

        result = np.zeros((self.num_vars, len(zeta_arr)), dtype=np.float64)

        for i, zeta_val in enumerate(zeta_arr):
            scalar_zeta: float = float(zeta_val)
            # evaluate_lagrange_polynomial_at_point returns vector L_j of length num_nodes_pts
            L_j: _Vector = evaluate_lagrange_polynomial_at_point(
                self.nodes_array, self.bary_weights, scalar_zeta
            )
            # values_at_nodes (num_vars, num_nodes_val) * L_j (num_nodes_pts)
            # Requires num_nodes_val == num_nodes_pts
            result[:, i] = np.dot(self.values_at_nodes, L_j)

        if self.num_vars == 1:
            final_result_vector = result.flatten()
            return final_result_vector[0] if is_scalar_input_point else final_result_vector

        return result[:, 0] if is_scalar_input_point else result


def get_polynomial_interpolant(
    nodes: _Vector, values: _Matrix, barycentric_weights: Optional[_Vector] = None
) -> PolynomialInterpolant:
    return PolynomialInterpolant(nodes, values, barycentric_weights)


DummyEvaluatorType = Callable[[Union[_NormalizedTimePoint, _Vector]], Union[_Matrix, _Vector]]


def dummy_evaluator(tau: Union[_NormalizedTimePoint, _Vector]) -> Union[_Matrix, _Vector]:
    if np.isscalar(tau):
        return np.zeros(
            0, dtype=np.float64
        )  # Consistent with PolynomialInterpolant num_vars=0 scalar output

    tau_arr = np.atleast_1d(tau)
    # Consistent with PolynomialInterpolant num_vars=0 vector output (0, N_tau)
    return np.zeros((0, len(tau_arr)), dtype=np.float64)


def _simulate_dynamics_for_error_estimation(
    interval_idx: int,
    solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    state_evaluator: PolynomialInterpolant,
    control_evaluator: PolynomialInterpolant,
    ode_solver: Callable[..., Any] = solve_ivp,
    ode_rtol: float = 1e-7,
    n_eval_points: int = 50,
) -> IntervalSimulationBundle:
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
        logger.warning(
            f"    Interval {interval_idx}: Time variables (t0, tf) are None. Skipping simulation."
        )
        return result

    t0: float = float(t0_sol)
    tf: float = float(tf_sol)

    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    if solution.global_normalized_mesh_nodes is None:
        logger.warning(
            f"    Interval {interval_idx}: global_normalized_mesh_nodes is None. Skipping simulation."
        )
        return result

    global_mesh: _MeshPoints = np.array(solution.global_normalized_mesh_nodes, dtype=np.float64)
    tau_start: _NormalizedTimePoint = global_mesh[interval_idx]
    tau_end: _NormalizedTimePoint = global_mesh[interval_idx + 1]

    beta_k = (tau_end - tau_start) / 2.0
    if abs(beta_k) < 1e-12:  # Interval of zero length
        logger.warning(f"    Interval {interval_idx} has zero length. Skipping simulation.")
        # Initialize with correct empty shapes if num_states is known
        result.forward_simulation_local_tau_evaluation_points = np.empty((1, 0), dtype=np.float64)
        result.state_trajectory_from_forward_simulation = np.empty(
            (num_states, 0), dtype=np.float64
        )
        result.nlp_state_trajectory_evaluated_at_forward_simulation_points = np.empty(
            (num_states, 0), dtype=np.float64
        )
        # ... and for backward
        result.backward_simulation_local_tau_evaluation_points = np.empty((1, 0), dtype=np.float64)
        result.state_trajectory_from_backward_simulation = np.empty(
            (num_states, 0), dtype=np.float64
        )
        result.nlp_state_trajectory_evaluated_at_backward_simulation_points = np.empty(
            (num_states, 0), dtype=np.float64
        )
        return result  # Success is still False

    beta_k0 = (tau_end + tau_start) / 2.0
    overall_scaling = alpha * beta_k

    def dynamics_rhs(tau_local: _NormalizedTimePoint, state_np: _Vector) -> _Vector:
        # control_evaluator returns Matrix (num_controls, 1) for scalar tau_local, or (num_controls, N)
        # Here tau_local is scalar from solve_ivp.
        control_val_at_tau: Union[_Matrix, _Vector] = control_evaluator(tau_local)

        control_np: _Vector
        if control_evaluator.num_vars == 0:
            control_np = np.array([], dtype=np.float64)
        elif (
            control_val_at_tau.ndim == 2 and control_val_at_tau.shape[1] == 1
        ):  # Expected (num_controls, 1)
            control_np = control_val_at_tau.flatten()
        elif control_val_at_tau.ndim == 1:  # Expected (num_controls,)
            control_np = control_val_at_tau
        else:  # Should not happen for scalar tau_local if interpolant is correct
            logger.error(f"Unexpected control shape from interpolant: {control_val_at_tau.shape}")
            control_np = np.array([], dtype=np.float64)

        global_tau_val: _NormalizedTimePoint = beta_k * tau_local + beta_k0
        physical_time: float = alpha * global_tau_val + alpha_0

        state_ca = CasADiDM(state_np)  # state_np is 1D from solve_ivp
        control_ca = CasADiDM(control_np)  # control_np should be 1D

        state_deriv_symbolic = dynamics_function(
            state_ca, control_ca, physical_time, problem_parameters
        )
        # Ensure output is flat numpy array
        state_deriv_np_flat: _Vector = np.array(state_deriv_symbolic, dtype=np.float64).flatten()

        if state_deriv_np_flat.shape[0] != num_states:
            raise ValueError(
                f"Dynamics function output dimension mismatch. Expected {num_states}, got {state_deriv_np_flat.shape[0]}."
            )
        return overall_scaling * state_deriv_np_flat

    # state_evaluator returns Matrix (num_states, 1) for scalar input
    initial_state_val_matrix: Union[_Matrix, _Vector] = state_evaluator(-1.0)
    initial_state: _Vector
    if num_states == 0:
        initial_state = np.array([], dtype=np.float64)
    elif initial_state_val_matrix.ndim == 2 and initial_state_val_matrix.shape[1] == 1:
        initial_state = initial_state_val_matrix.flatten()
    elif initial_state_val_matrix.ndim == 1 and initial_state_val_matrix.size == num_states:
        initial_state = initial_state_val_matrix
    else:
        logger.error(f"Unexpected initial_state_val_matrix shape: {initial_state_val_matrix.shape}")
        initial_state = np.full(num_states, np.nan, dtype=np.float64)  # Fallback

    fwd_tau_points_1d: _Vector = np.linspace(-1.0, 1.0, n_eval_points, dtype=np.float64)
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
            f"    Fwd IVP fail int {interval_idx}: {fwd_sim.message if hasattr(fwd_sim, 'message') else 'Unknown reason'}"
        )

    # state_evaluator for vector input returns Matrix (num_states, N_points)
    nlp_fwd_eval_matrix: Union[_Matrix, _Vector] = state_evaluator(fwd_tau_points_1d)
    result.nlp_state_trajectory_evaluated_at_forward_simulation_points = (
        nlp_fwd_eval_matrix.reshape(num_states, -1)
        if num_states > 0
        else np.empty((0, len(fwd_tau_points_1d)), dtype=np.float64)
    )

    terminal_state_val_matrix: Union[_Matrix, _Vector] = state_evaluator(1.0)
    terminal_state: _Vector
    if num_states == 0:
        terminal_state = np.array([], dtype=np.float64)
    elif terminal_state_val_matrix.ndim == 2 and terminal_state_val_matrix.shape[1] == 1:
        terminal_state = terminal_state_val_matrix.flatten()
    elif terminal_state_val_matrix.ndim == 1 and terminal_state_val_matrix.size == num_states:
        terminal_state = terminal_state_val_matrix
    else:
        logger.error(
            f"Unexpected terminal_state_val_matrix shape: {terminal_state_val_matrix.shape}"
        )
        terminal_state = np.full(num_states, np.nan, dtype=np.float64)

    bwd_tau_points_1d: _Vector = np.linspace(1.0, -1.0, n_eval_points, dtype=np.float64)
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
    result.backward_simulation_local_tau_evaluation_points = sorted_bwd_tau_points_1d.reshape(1, -1)
    result.state_trajectory_from_backward_simulation = (
        np.fliplr(bwd_sim.y).astype(np.float64)  # type: ignore[attr-defined]
        if bwd_sim.success and hasattr(bwd_sim, "y")
        else np.full((num_states, len(sorted_bwd_tau_points_1d)), np.nan, dtype=np.float64)
    )
    if not bwd_sim.success:
        logger.warning(
            f"    Bwd IVP fail int {interval_idx}: {bwd_sim.message if hasattr(bwd_sim, 'message') else 'Unknown reason'}"
        )

    nlp_bwd_eval_matrix: Union[_Matrix, _Vector] = state_evaluator(sorted_bwd_tau_points_1d)
    result.nlp_state_trajectory_evaluated_at_backward_simulation_points = (
        nlp_bwd_eval_matrix.reshape(num_states, -1)
        if num_states > 0
        else np.empty((0, len(sorted_bwd_tau_points_1d)), dtype=np.float64)
    )

    result.are_forward_and_backward_simulations_successful = fwd_sim.success and bwd_sim.success
    return result


def calculate_relative_error_estimate(
    interval_idx: int,
    sim_bundle: IntervalSimulationBundle,
    gamma_factors: _Vector,  # Column vector (NumStates x 1), float64
) -> float:
    if (
        not sim_bundle.are_forward_and_backward_simulations_successful
        or sim_bundle.state_trajectory_from_forward_simulation.size == 0
        # Add checks for other relevant arrays if they could be empty despite successful sim
    ):
        logger.warning(
            f"    Interval {interval_idx}: Simulation results incomplete or failed. Error is np.inf."
        )
        return np.inf

    num_states: NumStates = sim_bundle.state_trajectory_from_forward_simulation.shape[0]
    if num_states == 0:
        return 0.0  # No states, no error.

    # Ensure gamma_factors is float64 and correct shape (N_states, 1)
    gamma_factors_col_vec = gamma_factors.astype(np.float64).reshape(num_states, 1)

    fwd_diff: _Matrix = np.abs(
        sim_bundle.state_trajectory_from_forward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points
    )
    # Broadcasting (N_states,1) * (N_states, N_pts) -> (N_states, N_pts)
    fwd_errors: _Matrix = gamma_factors_col_vec * fwd_diff
    max_fwd_errors_per_state: _Vector = (
        np.nanmax(fwd_errors, axis=1)
        if fwd_errors.size > 0
        else np.zeros(num_states, dtype=np.float64)
    )

    bwd_diff: _Matrix = np.abs(
        sim_bundle.state_trajectory_from_backward_simulation
        - sim_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points
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
        else 0.0  # If somehow no errors calculated (e.g. num_states was 0 but bundle was not empty)
    )
    if (
        num_states > 0 and max_errors_per_state_combined.size == 0
    ):  # Should not happen if num_states > 0
        max_error_for_interval = np.inf

    if np.isnan(max_error_for_interval):
        logger.warning(
            f"    Interval {interval_idx}: Error calculation resulted in NaN. Treating as high error (np.inf)."
        )
        return np.inf
    # Ensure a minimum floor for error to avoid issues with log in p-refinement if error is extremely small
    return max(float(max_error_for_interval), 1e-15) if num_states > 0 else 0.0


@dataclass
class PRefineResult:
    actual_Nk_to_use: int
    was_p_successful: bool
    unconstrained_target_Nk: int


def p_refine_interval(
    current_Nk: int,
    interval_error: float,  # Renamed from max_error_val for clarity
    error_tol: float,
    N_max_degree: int,  # Renamed from N_max
) -> PRefineResult:
    if interval_error <= error_tol:  # Error already acceptable
        return PRefineResult(
            actual_Nk_to_use=current_Nk,
            was_p_successful=False,  # No refinement needed/done
            unconstrained_target_Nk=current_Nk,
        )

    # Heuristic for nodes to add
    if np.isinf(interval_error):  # Max out refinement if error is infinite
        nodes_to_add = max(1, N_max_degree - current_Nk)
    elif interval_error > 0:  # Avoid log of zero or negative
        ratio = interval_error / error_tol
        nodes_to_add = max(1, int(np.ceil(np.log10(ratio))))
    else:  # interval_error is zero or very small negative (due to float precision)
        nodes_to_add = 0  # No refinement needed

    if (
        nodes_to_add == 0 and interval_error > error_tol
    ):  # e.g. error is 1.1*tol, log10(1.1) is small
        nodes_to_add = 1  # ensure at least one node is added if error is above tolerance

    target_Nk = current_Nk + nodes_to_add

    if (
        target_Nk >= N_max_degree
    ):  # Use '>=' to include case where current_Nk is already N_max_degree
        return PRefineResult(
            actual_Nk_to_use=N_max_degree,
            was_p_successful=(N_max_degree > current_Nk),  # Successful if degree actually increased
            unconstrained_target_Nk=target_Nk,
        )

    return PRefineResult(
        actual_Nk_to_use=target_Nk,
        was_p_successful=True,
        unconstrained_target_Nk=target_Nk,
    )


@dataclass
class HRefineResult:
    collocation_nodes_for_new_subintervals: List[int]
    num_new_subintervals: int


def h_refine_params(
    unconstrained_target_Nk: int, N_min_degree: int
) -> HRefineResult:  # Renamed N_min
    # Determine number of subintervals needed if each has N_min_degree
    num_subintervals = max(
        2, int(np.ceil(unconstrained_target_Nk / max(1, N_min_degree)))
    )  # Avoid div by zero if N_min_degree is 0
    nodes_per_subinterval: List[int] = [N_min_degree] * num_subintervals

    return HRefineResult(
        collocation_nodes_for_new_subintervals=nodes_per_subinterval,
        num_new_subintervals=num_subintervals,
    )


def _map_global_normalized_tau_to_local_interval_tau(
    global_tau: _NormalizedTimePoint,
    global_interval_start_tau: _NormalizedTimePoint,  # Renamed
    global_interval_end_tau: _NormalizedTimePoint,  # Renamed
) -> _NormalizedTimePoint:
    beta = (global_interval_end_tau - global_interval_start_tau) / 2.0
    beta0 = (global_interval_end_tau + global_interval_start_tau) / 2.0
    if abs(beta) < 1e-12:
        return 0.0  # Midpoint for zero-length interval
    return (global_tau - beta0) / beta


def _map_local_interval_tau_to_global_normalized_tau(
    local_tau: _NormalizedTimePoint,
    global_interval_start_tau: _NormalizedTimePoint,  # Renamed
    global_interval_end_tau: _NormalizedTimePoint,  # Renamed
) -> _NormalizedTimePoint:
    beta = (global_interval_end_tau - global_interval_start_tau) / 2.0
    beta0 = (global_interval_end_tau + global_interval_start_tau) / 2.0
    return beta * local_tau + beta0


# The following two functions for mapping between adjacent intervals are specific
# and seem correctly implemented based on the primary mapping functions.
def _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
    local_tau_k: _NormalizedTimePoint,
    global_start_tau_k: _NormalizedTimePoint,
    global_shared_tau: _NormalizedTimePoint,
    global_end_tau_kp1: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    global_tau = _map_local_interval_tau_to_global_normalized_tau(
        local_tau_k, global_start_tau_k, global_shared_tau
    )
    return _map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_shared_tau, global_end_tau_kp1
    )


def _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
    local_tau_kp1: _NormalizedTimePoint,
    global_start_tau_k: _NormalizedTimePoint,
    global_shared_tau: _NormalizedTimePoint,
    global_end_tau_kp1: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    global_tau = _map_local_interval_tau_to_global_normalized_tau(
        local_tau_kp1, global_shared_tau, global_end_tau_kp1
    )
    return _map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_start_tau_k, global_shared_tau
    )


def h_reduce_intervals(
    first_interval_idx: int,  # Renamed
    solution: OptimalControlSolution,
    problem: OptimalControlProblem,
    adaptive_params: AdaptiveParameters,
    gamma_factors: _Vector,  # Column vector N_states x 1, float64
    state_evaluator_first_interval: PolynomialInterpolant,  # Renamed
    control_evaluator_first_interval: PolynomialInterpolant,  # Renamed
    state_evaluator_second_interval: PolynomialInterpolant,  # Renamed
    control_evaluator_second_interval: PolynomialInterpolant,  # Renamed
) -> bool:
    logger.info(
        f"    h-reduction check for intervals {first_interval_idx} and {first_interval_idx+1}."
    )
    error_tol: float = adaptive_params.error_tolerance
    ode_rtol: float = adaptive_params.ode_solver_tolerance
    ode_atol: float = ode_rtol * 1e-1  # Consistent smaller atol
    num_sim_points: int = adaptive_params.num_error_sim_points

    num_states: NumStates = problem.num_states
    dynamics_function = problem.dynamics_function
    problem_parameters = problem.problem_parameters

    if solution.raw_solution is None:
        logger.warning("      h-reduction failed: Raw solution missing.")
        return False
    if solution.global_normalized_mesh_nodes is None:
        logger.warning("      h-reduction failed: Global mesh nodes missing from solution.")
        return False

    global_mesh: _MeshPoints = np.array(solution.global_normalized_mesh_nodes, dtype=np.float64)

    if first_interval_idx + 2 >= len(global_mesh):
        logger.warning(
            f"      h-reduction failed: Not enough mesh points for intervals {first_interval_idx}, {first_interval_idx+1}."
        )
        return False

    # Mesh points for the two intervals being considered for merge
    tau_start_k: _NormalizedTimePoint = global_mesh[first_interval_idx]
    tau_shared_between_k_kp1: _NormalizedTimePoint = global_mesh[first_interval_idx + 1]  # Renamed
    tau_end_kp1: _NormalizedTimePoint = global_mesh[first_interval_idx + 2]

    # Half-lengths of the original intervals in global tau
    beta_k = (tau_shared_between_k_kp1 - tau_start_k) / 2.0
    beta_kp1 = (tau_end_kp1 - tau_shared_between_k_kp1) / 2.0

    if abs(beta_k) < 1e-12 or abs(beta_kp1) < 1e-12:
        logger.info(
            "      h-reduction check: One of the original intervals has zero length. Merge not sensible."
        )
        return False  # Cannot merge if one interval is effectively non-existent

    t0_sol = solution.initial_time_variable
    tf_sol = solution.terminal_time_variable
    if t0_sol is None or tf_sol is None:
        logger.warning("      h-reduction failed: Solution time variables are None.")
        return False
    t0: float = float(t0_sol)
    tf: float = float(tf_sol)

    # Global time scaling for the entire OCP
    alpha = (tf - t0) / 2.0
    alpha_0 = (tf + t0) / 2.0

    # Overall scaling factors for dynamics in each original interval's local tau
    scaling_k = alpha * beta_k
    scaling_kp1 = alpha * beta_kp1

    def _get_control_value_for_h_reduction(  # Renamed for clarity
        control_evaluator: PolynomialInterpolant, local_tau: _NormalizedTimePoint
    ) -> _Vector:
        if control_evaluator.num_vars == 0:
            return np.array([], dtype=np.float64)
        # Ensure tau is within [-1,1] for interpolant, though values outside can occur during mapping.
        # The interpolant itself should handle this, or this clip is a safety.
        clipped_tau = np.clip(local_tau, -1.0, 1.0)
        u_val_eval: Union[_Matrix, _Vector] = control_evaluator(
            clipped_tau
        )  # Returns (num_controls, 1) for scalar
        return np.atleast_1d(u_val_eval.squeeze()).astype(np.float64)  # Ensure 1D for dynamics

    # RHS for forward simulation over the first part of the merged interval (domain of original interval k)
    def merged_fwd_rhs_interval_k_domain(
        local_tau_k_domain: _NormalizedTimePoint, state_np: _Vector
    ) -> _Vector:
        u_val_np = _get_control_value_for_h_reduction(
            control_evaluator_first_interval, local_tau_k_domain
        )
        # Clipping state values during ODE solve can help stability but might mask issues.
        state_clipped_np = np.clip(state_np, -DEFAULT_STATE_CLIP_VALUE, DEFAULT_STATE_CLIP_VALUE)

        global_tau_val = _map_local_interval_tau_to_global_normalized_tau(
            local_tau_k_domain, tau_start_k, tau_shared_between_k_kp1
        )
        t_actual = alpha * global_tau_val + alpha_0

        state_ca = CasADiDM(state_clipped_np)
        u_ca = CasADiDM(u_val_np if u_val_np.size > 0 else [])
        f_rhs_sym = dynamics_function(state_ca, u_ca, t_actual, problem_parameters)
        return scaling_k * np.array(f_rhs_sym, dtype=np.float64).flatten()

    # RHS for backward simulation over the second part of the merged interval (domain of original interval k+1)
    def merged_bwd_rhs_interval_kp1_domain(
        local_tau_kp1_domain: _NormalizedTimePoint, state_np: _Vector
    ) -> _Vector:
        u_val_np = _get_control_value_for_h_reduction(
            control_evaluator_second_interval, local_tau_kp1_domain
        )
        state_clipped_np = np.clip(state_np, -DEFAULT_STATE_CLIP_VALUE, DEFAULT_STATE_CLIP_VALUE)

        global_tau_val = _map_local_interval_tau_to_global_normalized_tau(
            local_tau_kp1_domain, tau_shared_between_k_kp1, tau_end_kp1
        )
        t_actual = alpha * global_tau_val + alpha_0

        state_ca = CasADiDM(state_clipped_np)
        u_ca = CasADiDM(u_val_np if u_val_np.size > 0 else [])
        f_rhs_sym = dynamics_function(state_ca, u_ca, t_actual, problem_parameters)
        return scaling_kp1 * np.array(f_rhs_sym, dtype=np.float64).flatten()

    # Extract initial state for the forward simulation (start of interval k)
    initial_state_fwd_sim: _Vector
    try:
        if (
            solution.states
            and first_interval_idx < len(solution.states)
            and solution.states[first_interval_idx].size > 0
        ):
            Xk_nlp_from_sol_states: _Matrix = solution.states[first_interval_idx]
            initial_state_fwd_sim = Xk_nlp_from_sol_states[:, 0].flatten().astype(np.float64)
        else:  # Fallback to opti object if solution.states not populated directly
            opti: Optional[CasADiOpti] = solution.opti_object
            raw_sol = solution.raw_solution
            if not (
                problem.collocation_points_per_interval
                and raw_sol
                and opti
                and hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables")
                and opti.state_at_local_approximation_nodes_all_intervals_variables
                and first_interval_idx
                < len(opti.state_at_local_approximation_nodes_all_intervals_variables)
            ):
                logger.warning(
                    "      h-reduction failed: Cannot extract initial state data for first interval."
                )
                return False
            Nk_k: int = problem.collocation_points_per_interval[first_interval_idx]
            Xk_nlp_raw = raw_sol.value(
                opti.state_at_local_approximation_nodes_all_intervals_variables[first_interval_idx]
            )
            Xk_nlp = _extract_and_prepare_array(Xk_nlp_raw, num_states, Nk_k + 1)
            initial_state_fwd_sim = Xk_nlp[:, 0].flatten()
    except Exception as e:
        logger.error(f"      h-reduction failed: Error getting initial state for fwd sim: {e}")
        return False

    # The forward simulation goes from local tau = -1 (start of k)
    # to a point in k's local tau that corresponds to local tau = +1 in k+1 (end of k+1)
    # This means we are simulating across the *entire merged span* but using interval k's local tau coordinates.
    # This seems incorrect. The simulation should be split or use a unified coordinate system.
    # Let's assume the original logic meant to simulate *only* the first interval, then *only* the second for error.
    # For h-reduction, we need to simulate the *merged* interval.
    # The current RHS functions are specific to their original interval's scaling.
    # A more robust h-reduction would define a *new* interpolant over the merged interval [-1, 1]
    # mapping to global tau [tau_start_k, tau_end_kp1].
    # And simulate this new representation.

    # Given the existing structure, it seems it simulates from start of k to end of k,
    # then from end of k+1 to start of k+1, and compares with NLP states from BOTH intervals.

    # For this elegance check, I will follow the existing structure's apparent intent:
    # Simulate first interval fwd, second interval bwd, using their respective dynamics scalings.
    # Then calculate errors against the NLP states from *both* intervals, mapped to the simulation points.

    # Forward simulation for interval k
    # target_end_tau_k = 1.0 (simulating only interval k)
    num_fwd_pts_k = max(2, num_sim_points // 2)  # Or num_sim_points if covering only one interval
    fwd_tau_points_interval_k_local: _Vector = np.linspace(
        -1.0, 1.0, num_fwd_pts_k, dtype=np.float64
    )
    fwd_sim_obj_k = None
    fwd_trajectory_k: _Matrix = np.full(
        (num_states, len(fwd_tau_points_interval_k_local)), np.nan, dtype=np.float64
    )

    try:
        fwd_sim_obj_k = solve_ivp(
            merged_fwd_rhs_interval_k_domain,  # Uses scaling_k
            t_span=(-1.0, 1.0),
            y0=initial_state_fwd_sim,
            t_eval=fwd_tau_points_interval_k_local,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )
        if fwd_sim_obj_k.success:
            fwd_trajectory_k = fwd_sim_obj_k.y.astype(np.float64)
        else:
            logger.warning(
                f"      h-reduction: Fwd IVP for interval {first_interval_idx} failed: {fwd_sim_obj_k.message}"
            )
    except Exception as e:
        logger.error(
            f"      h-reduction: Exception during fwd IVP for interval {first_interval_idx}: {e}"
        )

    # Extract terminal state for the backward simulation (end of interval k+1)
    terminal_state_bwd_sim: _Vector
    try:
        if (
            solution.states
            and (first_interval_idx + 1) < len(solution.states)
            and solution.states[first_interval_idx + 1].size > 0
        ):
            Xkp1_nlp_from_sol_states: _Matrix = solution.states[first_interval_idx + 1]
            terminal_state_bwd_sim = Xkp1_nlp_from_sol_states[:, -1].flatten().astype(np.float64)
        else:  # Fallback
            opti = solution.opti_object
            raw_sol = solution.raw_solution
            if not (
                problem.collocation_points_per_interval
                and raw_sol
                and opti
                and hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables")
                and opti.state_at_local_approximation_nodes_all_intervals_variables
                and (first_interval_idx + 1)
                < len(opti.state_at_local_approximation_nodes_all_intervals_variables)
            ):
                logger.warning(
                    "      h-reduction failed: Cannot extract terminal state data for second interval."
                )
                return False
            Nk_kp1: int = problem.collocation_points_per_interval[first_interval_idx + 1]
            Xkp1_nlp_raw = raw_sol.value(
                opti.state_at_local_approximation_nodes_all_intervals_variables[
                    first_interval_idx + 1
                ]
            )
            Xkp1_nlp = _extract_and_prepare_array(Xkp1_nlp_raw, num_states, Nk_kp1 + 1)
            terminal_state_bwd_sim = Xkp1_nlp[:, -1].flatten()
    except Exception as e:
        logger.error(f"      h-reduction failed: Error getting terminal state for bwd sim: {e}")
        return False

    # Backward simulation for interval k+1
    # target_start_tau_kp1 = -1.0 (simulating only interval k+1, from its end to its start)
    num_bwd_pts_kp1 = max(2, num_sim_points // 2)
    bwd_tau_points_interval_kp1_local: _Vector = np.linspace(
        1.0, -1.0, num_bwd_pts_kp1, dtype=np.float64
    )  # Integrate from local +1 to -1
    bwd_sim_obj_kp1 = None
    bwd_trajectory_kp1_temp: _Matrix = np.full(
        (num_states, len(bwd_tau_points_interval_kp1_local)), np.nan, dtype=np.float64
    )

    try:
        bwd_sim_obj_kp1 = solve_ivp(
            merged_bwd_rhs_interval_kp1_domain,  # Uses scaling_kp1
            t_span=(1.0, -1.0),  # Integrate backward in local tau for interval k+1
            y0=terminal_state_bwd_sim,
            t_eval=bwd_tau_points_interval_kp1_local,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )
        if bwd_sim_obj_kp1.success:
            # Resulting trajectory is for local tau [1, ..., -1]. Flip to be [-1, ..., 1] order.
            bwd_trajectory_kp1_temp = np.fliplr(bwd_sim_obj_kp1.y).astype(np.float64)
    except Exception as e:
        logger.error(
            f"      h-reduction: Exception during bwd IVP for interval {first_interval_idx+1}: {e}"
        )

    # Simulation points for comparison, now in increasing local tau for interval k+1
    sorted_bwd_tau_points_kp1_local = np.flip(bwd_tau_points_interval_kp1_local)

    if num_states == 0:  # No states, merge depends only on simulation success
        can_merge = (
            fwd_sim_obj_k is not None
            and fwd_sim_obj_k.success
            and bwd_sim_obj_kp1 is not None
            and bwd_sim_obj_kp1.success
        )
        logger.info(
            f"      h-reduction check (no states): Simulations successful = {can_merge}. Merging: {can_merge}"
        )
        return can_merge

    # --- Error Calculation ---
    # This part compares the simulated trajectories (fwd_trajectory_k, bwd_trajectory_kp1_temp)
    # with the NLP solution's state interpolants (state_evaluator_first_interval, state_evaluator_second_interval).
    # The critical aspect is that the simulated points and NLP evaluations must be in consistent coordinate systems.

    all_errors_list: List[float] = []

    # Errors from forward simulation of interval k
    if fwd_sim_obj_k and fwd_sim_obj_k.success and fwd_trajectory_k.size > 0:
        nlp_states_at_fwd_pts_k: _Matrix = state_evaluator_first_interval(
            fwd_tau_points_interval_k_local
        )  # (Ns, Npts)
        abs_diff_fwd = np.abs(fwd_trajectory_k - nlp_states_at_fwd_pts_k)
        scaled_errors_fwd = (
            gamma_factors.flatten()[:, np.newaxis] * abs_diff_fwd
        )  # (Ns,1) * (Ns,Npts)
        all_errors_list.extend(list(scaled_errors_fwd.flatten()))

    # Errors from backward simulation of interval k+1
    if bwd_sim_obj_kp1 and bwd_sim_obj_kp1.success and bwd_trajectory_kp1_temp.size > 0:
        # bwd_trajectory_kp1_temp is already ordered for local tau [-1, ..., 1] for interval k+1
        # sorted_bwd_tau_points_kp1_local are the corresponding tau points
        nlp_states_at_bwd_pts_kp1: _Matrix = state_evaluator_second_interval(
            sorted_bwd_tau_points_kp1_local
        )
        abs_diff_bwd = np.abs(bwd_trajectory_kp1_temp - nlp_states_at_bwd_pts_kp1)
        scaled_errors_bwd = gamma_factors.flatten()[:, np.newaxis] * abs_diff_bwd
        all_errors_list.extend(list(scaled_errors_bwd.flatten()))

    if (
        not all_errors_list
    ):  # If simulations failed or num_states was 0 initially but somehow passed
        logger.warning(
            "      h-reduction: No error values collected. Assuming high error, merge not approved."
        )
        return False

    max_error_for_merged_candidate = (
        np.nanmax(all_errors_list).item() if all_errors_list else np.inf
    )

    if np.isnan(max_error_for_merged_candidate):
        logger.warning(
            "      h-reduction check: max_error calculation resulted in NaN. Merge not approved."
        )
        max_error_for_merged_candidate = np.inf

    can_merge_result = max_error_for_merged_candidate <= error_tol
    logger.info(
        f"      h-reduction check result: max_error_merged_candidate = {max_error_for_merged_candidate:.4e}, tol = {error_tol:.2e}. Merge approved: {can_merge_result}"
    )
    return can_merge_result


@dataclass
class PReduceResult:
    new_num_collocation_nodes: int
    was_reduction_applied: bool


def p_reduce_interval(
    current_Nk: int,
    interval_error: float,  # Renamed from max_error_val
    error_tol: float,
    N_min_degree: int,  # Renamed from N_min
    N_max_degree: int,  # Renamed from N_max (though not directly used in original logic, good to have for context)
) -> PReduceResult:
    if interval_error > error_tol or current_Nk <= N_min_degree:
        return PReduceResult(new_num_collocation_nodes=current_Nk, was_reduction_applied=False)

    # Heuristic for nodes to remove
    nodes_to_remove_float = 0.0
    if interval_error < 1e-16:  # Essentially zero error
        nodes_to_remove_float = float(current_Nk - N_min_degree)
    elif interval_error > 0:  # interval_error is > 0 and <= error_tol
        ratio = error_tol / interval_error  # ratio >= 1.0
        # Original formula: delta = float(N_min_degree + N_max_degree - current_Nk)
        # base_for_log = np.power(ratio, 1.0 / delta)
        # nodes_to_remove_float = np.floor(np.log10(base_for_log))
        # This heuristic seems complex and dependent on N_max.
        # A simpler heuristic: if error is, e.g., 0.1 * tol, maybe remove 1 node.
        # If error is 0.01 * tol, maybe remove 2 nodes. (logarithmic reduction)
        # Let's use a simplified version or keep the original if its derivation is known.
        # For now, let's simplify: reduce by 1 if error is < 0.5 * tol, etc.
        # Or, more closely to the spirit:
        # Try to estimate how many nodes could be removed.
        # If ratio is large (error is very small), can remove more.
        # log10(ratio) gives an order of magnitude.
        # A simple approach: nodes_to_remove = floor(log10(error_tol / interval_error))
        # This is similar to p_refine's heuristic in reverse.
        if ratio > 1.0:  # Ensure log argument is > 0
            nodes_to_remove_float = np.floor(
                np.log10(ratio)
            )  # if ratio=10, remove 1. if ratio=100, remove 2.

    nodes_to_remove = max(0, int(nodes_to_remove_float))
    new_Nk = max(N_min_degree, current_Nk - nodes_to_remove)
    was_reduced = new_Nk < current_Nk

    return PReduceResult(new_num_collocation_nodes=new_Nk, was_reduction_applied=was_reduced)


def _generate_robust_default_initial_guess(
    problem: OptimalControlProblem,
    collocation_nodes_list: List[int],
    initial_time_guess: Optional[float] = None,
    terminal_time_guess: Optional[float] = None,
    integral_values_guess: Optional[Union[_Vector, float]] = None,
) -> InitialGuess:
    num_states: NumStates = problem.num_states
    num_controls: NumControls = problem.num_controls
    num_integrals: NumIntegrals = problem.num_integrals

    default_state_val = getattr(problem.default_initial_guess_values, "state", 0.0)
    default_control_val = getattr(problem.default_initial_guess_values, "control", 0.0)

    states_guess_list: List[_Matrix] = []
    controls_guess_list: List[_Matrix] = []

    for Nk_val in collocation_nodes_list:
        # State guess: (num_states, Nk_val + 1)
        state_traj_guess = np.full((num_states, Nk_val + 1), default_state_val, dtype=np.float64)
        states_guess_list.append(state_traj_guess)

        # Control guess: (num_controls, Nk_val)
        if num_controls > 0:
            control_traj_guess = np.full(
                (num_controls, Nk_val), default_control_val, dtype=np.float64
            )
        else:  # No controls
            control_traj_guess = np.empty((0, Nk_val), dtype=np.float64)
        controls_guess_list.append(control_traj_guess)

    # Time guesses
    final_t0_guess: Optional[float] = initial_time_guess
    if final_t0_guess is None and problem.initial_guess:
        final_t0_guess = problem.initial_guess.initial_time_variable

    final_tf_guess: Optional[float] = terminal_time_guess
    if final_tf_guess is None and problem.initial_guess:
        final_tf_guess = problem.initial_guess.terminal_time_variable

    # Integral guesses
    final_integrals_guess: Optional[Union[float, Sequence[float]]] = None  # Corrected type
    if num_integrals > 0:
        if integral_values_guess is not None:
            if isinstance(integral_values_guess, np.ndarray):
                final_integrals_guess = list(integral_values_guess.astype(np.float64))
            else:  # float
                final_integrals_guess = float(integral_values_guess)
        elif problem.initial_guess and problem.initial_guess.integrals is not None:
            # problem.initial_guess.integrals can be float | Sequence[float] | _FloatArray | None
            raw_problem_integrals = problem.initial_guess.integrals
            if isinstance(raw_problem_integrals, np.ndarray):
                final_integrals_guess = list(raw_problem_integrals.astype(np.float64))
            elif isinstance(raw_problem_integrals, (list, tuple)):
                final_integrals_guess = [float(x) for x in raw_problem_integrals]
            elif isinstance(raw_problem_integrals, float):
                final_integrals_guess = raw_problem_integrals
            # else leave as None if type is unexpected
        else:  # Default value for integrals
            default_integral_val = getattr(problem.default_initial_guess_values, "integral", 0.0)
            if num_integrals == 1:  # Expects float
                final_integrals_guess = default_integral_val
            else:  # Expects Sequence[float]
                final_integrals_guess = [default_integral_val] * num_integrals

    return InitialGuess(
        initial_time_variable=final_t0_guess,
        terminal_time_variable=final_tf_guess,
        states=states_guess_list,
        controls=controls_guess_list,
        integrals=final_integrals_guess,
    )


def _propagate_guess_from_previous(
    prev_solution: OptimalControlSolution,
    problem_for_new_nlp: OptimalControlProblem,  # Renamed for clarity
    target_collocation_nodes_list: List[int],  # Renamed
    target_mesh_global_tau: _MeshPoints,  # Renamed, type is np.ndarray[np.float64]
) -> InitialGuess:
    t0_prop: Optional[float] = prev_solution.initial_time_variable
    tf_prop: Optional[float] = prev_solution.terminal_time_variable

    integrals_prop_for_guess: Optional[Union[_Vector, float]] = None
    if prev_solution.integrals is not None:
        if isinstance(prev_solution.integrals, np.ndarray):
            integrals_prop_for_guess = prev_solution.integrals.astype(np.float64)
        else:  # Should be float
            integrals_prop_for_guess = float(prev_solution.integrals)

    # Start with a robust default guess, then overwrite parts if possible
    current_guess = _generate_robust_default_initial_guess(
        problem_for_new_nlp,
        target_collocation_nodes_list,
        initial_time_guess=t0_prop,
        terminal_time_guess=tf_prop,
        integral_values_guess=integrals_prop_for_guess,
    )

    prev_nodes_list = prev_solution.num_collocation_nodes_list_at_solve_time
    prev_mesh_list_type = prev_solution.global_mesh_nodes_at_solve_time

    can_propagate_trajectories = False
    if prev_nodes_list is not None and prev_mesh_list_type is not None:
        prev_mesh_np = np.array(prev_mesh_list_type, dtype=np.float64)
        # Check if mesh structure (nodes per interval AND mesh points) is identical
        if target_collocation_nodes_list == prev_nodes_list and np.allclose(
            target_mesh_global_tau, prev_mesh_np, atol=1e-9, rtol=1e-9
        ):
            can_propagate_trajectories = True

    if can_propagate_trajectories:
        logger.info(
            "  Mesh structure identical to previous. Propagating state/control trajectories directly."
        )
        if prev_solution.states and len(prev_solution.states) == len(target_collocation_nodes_list):
            current_guess.states = [s.astype(np.float64) for s in prev_solution.states]
        else:
            logger.warning(
                "    Warning: Previous states mismatch or missing. Using default states from robust guess."
            )

        if prev_solution.controls and len(prev_solution.controls) == len(
            target_collocation_nodes_list
        ):
            current_guess.controls = [c.astype(np.float64) for c in prev_solution.controls]
        else:
            logger.warning(
                "    Warning: Previous controls mismatch or missing. Using default controls from robust guess."
            )
    else:
        logger.info(
            "  Mesh structure changed. Using robust default for state/control (times/integrals propagated)."
        )

    return current_guess


def _calculate_gamma_normalizers(
    solution: OptimalControlSolution, problem: OptimalControlProblem
) -> Optional[_Vector]:  # Returns a column vector (NumStates x 1) of np.float64, or None
    if not solution.success:  # solution.states can be empty if num_states=0
        logger.warning("    Gamma calculation failed - solution unsuccessful.")
        return None

    num_states: NumStates = problem.num_states
    if num_states == 0:
        return np.array([]).reshape(0, 1).astype(np.float64)

    if (
        not solution.states
    ):  # Should not happen if num_states > 0 and solution succeeded with state extraction
        logger.warning(
            "    Gamma calculation failed - solution.states missing or empty despite num_states > 0."
        )
        return None

    # Find max absolute value for each state across all intervals and all points in trajectories
    # Initialize with very small numbers or zeros
    max_abs_values_per_state: _Vector = np.zeros(num_states, dtype=np.float64)

    for (
        Xk_trajectory_matrix
    ) in solution.states:  # Xk_trajectory_matrix is _Matrix (num_states, Nk+1)
        if Xk_trajectory_matrix.size == 0:  # Should not happen if num_states > 0
            continue
        # Max abs along points (axis=1) for each state in current interval
        max_abs_in_interval_k: _Vector = np.max(
            np.abs(Xk_trajectory_matrix.astype(np.float64)), axis=1
        )
        max_abs_values_per_state = np.maximum(max_abs_values_per_state, max_abs_in_interval_k)

    # Normalization factor: gamma_i = 1 / (1 + max|x_i(t)|)
    gamma_denominator: _Vector = 1.0 + max_abs_values_per_state
    # Prevent division by zero or extremely small numbers if a state is always zero
    gamma_denominator_safe = np.maximum(gamma_denominator, 1e-12)
    gamma_factors_flat: _Vector = 1.0 / gamma_denominator_safe

    return gamma_factors_flat.reshape(-1, 1).astype(np.float64)  # Ensure column vector


class PHSAdaptive(AdaptiveBase):
    adaptive_params: AdaptiveParameters
    _initial_polynomial_degrees: Optional[List[int]]
    _initial_mesh_points_global_tau: Optional[_MeshPoints]  # Stored as float64 NumPy array, Renamed

    def __init__(
        self,
        error_tolerance: float = 1e-3,
        max_iterations: int = 30,
        min_polynomial_degree: int = 4,
        max_polynomial_degree: int = 16,
        ode_solver_tolerance: float = 1e-7,
        num_error_sim_points: int = 40,
        initial_polynomial_degrees: Optional[List[int]] = None,
        initial_mesh_points_global_tau: Optional[Union[_MeshPoints, List[float]]] = None,  # Renamed
        initial_guess: Optional[InitialGuess] = None,
    ):
        super().__init__(initial_guess)
        self.adaptive_params = AdaptiveParameters(
            error_tolerance,
            max_iterations,
            min_polynomial_degree,
            max_polynomial_degree,
            ode_solver_tolerance,
            num_error_sim_points,
        )

        if initial_polynomial_degrees is not None and initial_mesh_points_global_tau is not None:
            if len(initial_polynomial_degrees) != len(initial_mesh_points_global_tau) - 1:
                raise ValueError(
                    "Number of initial polynomial degrees must be one less than the number of initial mesh points."
                )

        self._initial_polynomial_degrees = (
            list(initial_polynomial_degrees) if initial_polynomial_degrees else None
        )
        if initial_mesh_points_global_tau is not None:
            self._initial_mesh_points_global_tau = np.array(
                initial_mesh_points_global_tau, dtype=np.float64
            )
        else:
            self._initial_mesh_points_global_tau = None

    def run(
        self,
        problem: Optional[
            OptimalControlProblem
        ],  # Unused parameter, kept for API compatibility with AdaptiveBase
        legacy_problem_instance: OptimalControlProblem,  # Renamed for clarity
        initial_solution: Optional[OptimalControlSolution] = None,
    ) -> OptimalControlSolution:
        """Run the PHS-Adaptive mesh refinement algorithm."""
        # `problem` arg is unused as per original comment, `legacy_problem_instance` is used.

        error_tol = self.adaptive_params.error_tolerance
        max_iter = self.adaptive_params.max_iterations
        N_min = self.adaptive_params.min_polynomial_degree
        N_max = self.adaptive_params.max_polynomial_degree
        ode_rtol_val = self.adaptive_params.ode_solver_tolerance
        num_sim_pts = self.adaptive_params.num_error_sim_points

        # Initialize current mesh structure
        current_collocation_nodes_list: List[int]
        current_global_mesh_tau: _MeshPoints  # Internally always float64 NumPy array

        if (
            self._initial_polynomial_degrees is not None
            and self._initial_mesh_points_global_tau is not None
        ):
            current_collocation_nodes_list = list(self._initial_polynomial_degrees)
            current_global_mesh_tau = np.copy(self._initial_mesh_points_global_tau)
        else:
            # Fallback to problem's initial mesh or a single default interval
            current_collocation_nodes_list = list(
                legacy_problem_instance.collocation_points_per_interval or [N_min]
            )
            if legacy_problem_instance.global_normalized_mesh_nodes is not None:
                current_global_mesh_tau = np.array(
                    legacy_problem_instance.global_normalized_mesh_nodes, dtype=np.float64
                )
            else:  # Default to a single interval from -1 to 1
                current_global_mesh_tau = np.linspace(
                    -1, 1, len(current_collocation_nodes_list) + 1, dtype=np.float64
                )

        # Ensure initial node counts are within bounds
        current_collocation_nodes_list = [
            max(N_min, min(N_max, nk)) for nk in current_collocation_nodes_list
        ]

        # The OCP definition that will be modified in each iteration
        current_ocp_definition: OptimalControlProblem = legacy_problem_instance
        most_recent_ocp_solution: Optional[OptimalControlSolution] = initial_solution

        # Dynamic import for solver (consider if this is strictly necessary or can be a top-level import)
        try:
            from trajectolab.direct_solver import solve_single_phase_radau_collocation
        except ImportError as e:
            logger.error(f"Failed to import solve_single_phase_radau_collocation: {e}")
            raise

        for iteration_M in range(max_iter):  # Renamed iteration variable
            logger.info(f"\n--- Adaptive Iteration M = {iteration_M} ---")
            num_intervals_K = len(current_collocation_nodes_list)  # Renamed

            # Update OCP definition with current mesh
            current_ocp_definition.collocation_points_per_interval = list(
                current_collocation_nodes_list
            )
            current_ocp_definition.global_normalized_mesh_nodes = list(
                current_global_mesh_tau
            )  # OCP might expect list

            # Prepare initial guess for the NLP solver
            initial_guess_for_nlp: InitialGuess
            user_provided_initial_guess = self.initial_guess  # From AdaptiveBase

            if iteration_M == 0 and user_provided_initial_guess is not None:
                initial_guess_for_nlp = user_provided_initial_guess
                logger.info("  Using user-provided initial guess for first iteration.")
            elif iteration_M == 0:  # No user guess, first iteration
                initial_guess_for_nlp = _generate_robust_default_initial_guess(
                    current_ocp_definition, current_collocation_nodes_list
                )
                logger.info("  Using robust default initial guess for first iteration.")
            elif not most_recent_ocp_solution or not most_recent_ocp_solution.success:
                logger.warning(
                    "  Previous NLP failed or no previous solution. Using robust default initial guess."
                )
                initial_guess_for_nlp = _generate_robust_default_initial_guess(
                    current_ocp_definition, current_collocation_nodes_list
                )
            else:  # Propagate from previous successful solution
                initial_guess_for_nlp = _propagate_guess_from_previous(
                    most_recent_ocp_solution,
                    current_ocp_definition,
                    current_collocation_nodes_list,
                    current_global_mesh_tau,
                )
            current_ocp_definition.initial_guess = initial_guess_for_nlp

            logger.info(f"  Mesh K={num_intervals_K}, N_k = {current_collocation_nodes_list}")
            logger.info(
                f"  Mesh tau_global = {np.array2string(current_global_mesh_tau, precision=3)}"
            )

            # Solve the OCP
            solved_ocp_solution_this_iter: OptimalControlSolution = (
                solve_single_phase_radau_collocation(current_ocp_definition)
            )

            if not solved_ocp_solution_this_iter.success:
                error_msg = (
                    f"NLP solver failed in adaptive iteration {iteration_M}. "
                    f"{solved_ocp_solution_this_iter.message or 'Solver error.'} Stopping."
                )
                logger.error(f"  Error: {error_msg}")
                if most_recent_ocp_solution:  # Return the last good one, marked as failed now
                    most_recent_ocp_solution.message = error_msg
                    most_recent_ocp_solution.success = False
                    return most_recent_ocp_solution
                return solved_ocp_solution_this_iter  # Return the current failed solution

            # --- Populate solution object with state/control trajectories ---
            # This ensures solution.states and solution.controls are consistently np.ndarray lists
            try:
                opti_obj: Optional[CasADiOpti] = solved_ocp_solution_this_iter.opti_object
                raw_casadi_sol = solved_ocp_solution_this_iter.raw_solution

                if (
                    opti_obj
                    and raw_casadi_sol
                    and hasattr(
                        opti_obj, "state_at_local_approximation_nodes_all_intervals_variables"
                    )
                    and opti_obj.state_at_local_approximation_nodes_all_intervals_variables
                ):
                    temp_states_list: List[_Matrix] = []
                    for i, var_sx_or_mx in enumerate(
                        opti_obj.state_at_local_approximation_nodes_all_intervals_variables
                    ):
                        val = raw_casadi_sol.value(var_sx_or_mx)
                        temp_states_list.append(
                            _extract_and_prepare_array(
                                val,
                                current_ocp_definition.num_states,
                                current_collocation_nodes_list[i] + 1,  # Nk+1 points for states
                            )
                        )
                    solved_ocp_solution_this_iter.states = temp_states_list
                elif current_ocp_definition.num_states > 0:  # States expected but not found
                    logger.warning("  Could not extract states from NLP solution object.")
                    solved_ocp_solution_this_iter.states = []  # Ensure it's an empty list
                else:  # num_states is 0
                    solved_ocp_solution_this_iter.states = []

                if current_ocp_definition.num_controls > 0:
                    if (
                        opti_obj
                        and raw_casadi_sol
                        and hasattr(
                            opti_obj, "control_at_local_collocation_nodes_all_intervals_variables"
                        )
                        and opti_obj.control_at_local_collocation_nodes_all_intervals_variables
                    ):
                        temp_controls_list: List[_Matrix] = []
                        for i, var_sx_or_mx in enumerate(
                            opti_obj.control_at_local_collocation_nodes_all_intervals_variables
                        ):
                            val = raw_casadi_sol.value(var_sx_or_mx)
                            temp_controls_list.append(
                                _extract_and_prepare_array(
                                    val,
                                    current_ocp_definition.num_controls,
                                    current_collocation_nodes_list[i],  # Nk points for controls
                                )
                            )
                        solved_ocp_solution_this_iter.controls = temp_controls_list
                    else:  # Controls expected but not found
                        logger.warning("  Could not extract controls from NLP solution object.")
                        solved_ocp_solution_this_iter.controls = []
                else:  # num_controls is 0
                    solved_ocp_solution_this_iter.controls = [
                        np.empty((0, nk), dtype=np.float64) for nk in current_collocation_nodes_list
                    ]

            except Exception as e:
                error_msg = f"Failed to extract trajectories from NLP solution at iter {iteration_M}: {e}. Stopping."
                logger.error(f"  Error: {error_msg}")
                solved_ocp_solution_this_iter.message = error_msg
                solved_ocp_solution_this_iter.success = False
                return solved_ocp_solution_this_iter
            # --- End trajectory population ---

            most_recent_ocp_solution = solved_ocp_solution_this_iter
            # Store mesh used for this successful solve
            most_recent_ocp_solution.num_collocation_nodes_list_at_solve_time = list(
                current_collocation_nodes_list
            )
            most_recent_ocp_solution.global_mesh_nodes_at_solve_time = list(
                np.copy(current_global_mesh_tau)
            )

            gamma_factors_col_vec: Optional[_Vector] = _calculate_gamma_normalizers(
                most_recent_ocp_solution, current_ocp_definition
            )
            if gamma_factors_col_vec is None and current_ocp_definition.num_states > 0:
                error_msg = (
                    f"Failed to calculate gamma normalizers at iter {iteration_M}. Stopping."
                )
                logger.error(f"  Error: {error_msg}")
                most_recent_ocp_solution.message = error_msg
                most_recent_ocp_solution.success = False
                return most_recent_ocp_solution

            # Cache for Radau basis components to avoid recomputation
            radau_basis_cache: dict[int, RadauBasisComponents] = {}  # Renamed

            # Prepare interpolants for state and control trajectories
            state_evaluators_list: List[PolynomialInterpolant] = [None] * num_intervals_K  # type: ignore
            control_evaluators_list: List[PolynomialInterpolant] = [None] * num_intervals_K  # type: ignore

            # Create a single dummy interpolant for controls if no controls exist
            _dummy_control_interpolant_if_no_controls = None
            if current_ocp_definition.num_controls == 0:
                _dummy_nodes = np.array([-1.0, 1.0], dtype=np.float64)
                _dummy_control_values = np.empty((0, 2), dtype=np.float64)  # (0, N_nodes)
                _dummy_control_interpolant_if_no_controls = PolynomialInterpolant(
                    _dummy_nodes, _dummy_control_values
                )

            for k_interval_idx in range(num_intervals_K):  # Renamed
                try:
                    Nk_val_current_interval = current_collocation_nodes_list[
                        k_interval_idx
                    ]  # Renamed
                    if Nk_val_current_interval not in radau_basis_cache:
                        radau_basis_cache[Nk_val_current_interval] = (
                            compute_radau_collocation_components(Nk_val_current_interval)
                        )
                    basis_comps: RadauBasisComponents = radau_basis_cache[Nk_val_current_interval]

                    # State interpolant
                    if (
                        most_recent_ocp_solution.states
                        and k_interval_idx < len(most_recent_ocp_solution.states)
                        and most_recent_ocp_solution.states[k_interval_idx].size > 0
                    ):
                        state_values_k_interval: _Matrix = most_recent_ocp_solution.states[
                            k_interval_idx
                        ]
                        state_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                            basis_comps.state_approximation_nodes,  # Vector (Nk+1)
                            state_values_k_interval,  # Matrix (num_states, Nk+1)
                            basis_comps.barycentric_weights_for_state_nodes,
                        )
                    elif current_ocp_definition.num_states > 0:  # States expected but missing
                        logger.warning(
                            f"  State trajectory missing for interval {k_interval_idx}. Using NaN-filled interpolant."
                        )
                        empty_state_vals = np.full(
                            (
                                current_ocp_definition.num_states,
                                len(basis_comps.state_approximation_nodes),
                            ),
                            np.nan,
                            dtype=np.float64,
                        )
                        state_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                            basis_comps.state_approximation_nodes,
                            empty_state_vals,
                            basis_comps.barycentric_weights_for_state_nodes,
                        )
                    else:  # num_states == 0
                        empty_state_vals = np.empty(
                            (0, len(basis_comps.state_approximation_nodes)), dtype=np.float64
                        )
                        state_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                            basis_comps.state_approximation_nodes,
                            empty_state_vals,
                            basis_comps.barycentric_weights_for_state_nodes,
                        )

                    # Control interpolant
                    if current_ocp_definition.num_controls > 0:
                        if (
                            most_recent_ocp_solution.controls
                            and k_interval_idx < len(most_recent_ocp_solution.controls)
                            and most_recent_ocp_solution.controls[k_interval_idx].size > 0
                        ):
                            control_values_k_interval: _Matrix = most_recent_ocp_solution.controls[
                                k_interval_idx
                            ]
                            # Barycentric weights for control nodes (collocation_nodes) might differ from state nodes
                            control_bary_weights = compute_barycentric_weights(
                                basis_comps.collocation_nodes
                            )
                            control_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                                basis_comps.collocation_nodes,  # Vector (Nk)
                                control_values_k_interval,  # Matrix (num_controls, Nk)
                                control_bary_weights,
                            )
                        else:  # Controls expected but missing
                            logger.warning(
                                f"  Control trajectory missing for interval {k_interval_idx}. Using NaN-filled interpolant."
                            )
                            empty_ctrl_vals = np.full(
                                (
                                    current_ocp_definition.num_controls,
                                    len(basis_comps.collocation_nodes),
                                ),
                                np.nan,
                                dtype=np.float64,
                            )
                            control_bary_weights = compute_barycentric_weights(
                                basis_comps.collocation_nodes
                            )
                            control_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                                basis_comps.collocation_nodes, empty_ctrl_vals, control_bary_weights
                            )
                    else:  # No controls
                        control_evaluators_list[k_interval_idx] = _dummy_control_interpolant_if_no_controls  # type: ignore

                except Exception as e:
                    logger.error(f"  Error creating interpolant for interval {k_interval_idx}: {e}")
                    # Fallback to dummy interpolants if error occurs
                    # This requires a more robust dummy that matches expected num_vars if possible
                    # For simplicity, let's ensure the list is filled. A better dummy would be ideal.
                    _dummy_nodes = np.array([-1.0, 1.0], dtype=np.float64)
                    _s_vals = np.empty((current_ocp_definition.num_states, 2), dtype=np.float64)
                    state_evaluators_list[k_interval_idx] = PolynomialInterpolant(
                        _dummy_nodes, _s_vals
                    )
                    if _dummy_control_interpolant_if_no_controls:
                        control_evaluators_list[k_interval_idx] = (
                            _dummy_control_interpolant_if_no_controls
                        )
                    else:
                        _c_vals = np.empty(
                            (current_ocp_definition.num_controls, 2), dtype=np.float64
                        )
                        control_evaluators_list[k_interval_idx] = PolynomialInterpolant(
                            _dummy_nodes, _c_vals
                        )

            # Estimate error for each interval
            interval_error_estimates: List[float] = [np.inf] * num_intervals_K  # Renamed

            for k_interval_idx in range(num_intervals_K):
                logger.info(f"  Starting error simulation for interval {k_interval_idx}...")
                sim_bundle = _simulate_dynamics_for_error_estimation(
                    k_interval_idx,
                    most_recent_ocp_solution,
                    current_ocp_definition,
                    state_evaluators_list[k_interval_idx],
                    control_evaluators_list[k_interval_idx],
                    ode_rtol=ode_rtol_val,
                    n_eval_points=num_sim_pts,
                )

                # Ensure gamma_factors_col_vec is valid for error calculation
                gamma_for_calc: _Vector
                if current_ocp_definition.num_states > 0:
                    if (
                        gamma_factors_col_vec is not None
                        and gamma_factors_col_vec.size == current_ocp_definition.num_states
                    ):
                        gamma_for_calc = gamma_factors_col_vec
                    else:
                        logger.warning(
                            f"   Gamma factors invalid or missing for interval {k_interval_idx} with {current_ocp_definition.num_states} states. Error will be inf."
                        )
                        gamma_for_calc = np.full(
                            (current_ocp_definition.num_states, 1), np.nan, dtype=np.float64
                        )  # This will lead to inf error
                else:  # No states, no gamma factors needed
                    gamma_for_calc = np.empty((0, 1), dtype=np.float64)

                error_val_this_interval = calculate_relative_error_estimate(
                    k_interval_idx, sim_bundle, gamma_for_calc
                )
                interval_error_estimates[k_interval_idx] = error_val_this_interval
                logger.info(
                    f"    Interval {k_interval_idx}: Nk={current_collocation_nodes_list[k_interval_idx]}, Error={error_val_this_interval:.4e}"
                )

            logger.info(f"  Overall errors: {[f'{e:.2e}' for e in interval_error_estimates]}")

            # Check for convergence: all errors below tolerance
            # Ignore NaN/Inf for "all errors ok" if they are being refined anyway.
            # Strict check: if any error is NaN/Inf, it's not "ok" unless num_intervals is 0.
            converged_this_iteration = False
            if not interval_error_estimates:  # No intervals (e.g. problem setup issue)
                converged_this_iteration = True  # Vacuously true
            else:
                converged_this_iteration = all(
                    err <= error_tol
                    for err in interval_error_estimates
                    if not (np.isnan(err) or np.isinf(err))
                )
                # If any unhandled NaN/Inf remains, it's not truly converged.
                if any(np.isnan(err) or np.isinf(err) for err in interval_error_estimates):
                    converged_this_iteration = False

            if converged_this_iteration:
                success_msg = f"Adaptive mesh converged to tolerance {error_tol:.1e} in {iteration_M+1} iterations."
                logger.info(success_msg)
                most_recent_ocp_solution.num_collocation_nodes_per_interval = list(
                    current_collocation_nodes_list
                )
                most_recent_ocp_solution.global_normalized_mesh_nodes = list(
                    np.copy(current_global_mesh_tau)
                )
                most_recent_ocp_solution.message = success_msg
                return most_recent_ocp_solution

            # --- Mesh Adaptation Logic (p, h refinement/reduction) ---
            next_collocation_nodes_proposal: List[int] = []  # Renamed
            next_global_mesh_tau_proposal: List[_NormalizedTimePoint] = [
                current_global_mesh_tau[0]
            ]  # Renamed

            current_interval_pointer = 0  # Renamed
            while current_interval_pointer < num_intervals_K:
                error_k = interval_error_estimates[current_interval_pointer]
                Nk_k = current_collocation_nodes_list[current_interval_pointer]
                logger.info(
                    f"    Adapting interval {current_interval_pointer}: Nk={Nk_k}, Error={error_k:.2e}"
                )

                if (
                    np.isnan(error_k) or np.isinf(error_k) or error_k > error_tol
                ):  # High error or invalid
                    logger.info(
                        f"      Interval {current_interval_pointer} error > tol. Attempting p-refinement."
                    )
                    p_refine_res = p_refine_interval(Nk_k, error_k, error_tol, N_max)
                    if p_refine_res.was_p_successful:
                        logger.info(
                            f"        p-refinement applied: Nk {Nk_k} -> {p_refine_res.actual_Nk_to_use}"
                        )
                        next_collocation_nodes_proposal.append(p_refine_res.actual_Nk_to_use)
                        next_global_mesh_tau_proposal.append(
                            current_global_mesh_tau[current_interval_pointer + 1]
                        )
                        current_interval_pointer += 1
                    else:  # p-refinement failed or hit N_max, try h-refinement
                        logger.info(
                            "        p-refinement failed or insufficient. Attempting h-refinement."
                        )
                        h_refine_res = h_refine_params(p_refine_res.unconstrained_target_Nk, N_min)
                        logger.info(
                            f"          h-refinement: Splitting int {current_interval_pointer} into {h_refine_res.num_new_subintervals} "
                            f"subintervals, each Nk={h_refine_res.collocation_nodes_for_new_subintervals[0]}."
                        )
                        next_collocation_nodes_proposal.extend(
                            h_refine_res.collocation_nodes_for_new_subintervals
                        )
                        # Create new mesh points for the split interval
                        new_split_mesh_segment = np.linspace(
                            current_global_mesh_tau[
                                current_interval_pointer
                            ],  # Start of current interval
                            current_global_mesh_tau[
                                current_interval_pointer + 1
                            ],  # End of current interval
                            h_refine_res.num_new_subintervals + 1,
                            dtype=np.float64,
                        )
                        next_global_mesh_tau_proposal.extend(
                            list(new_split_mesh_segment[1:])
                        )  # Add new internal points and end point
                        current_interval_pointer += 1
                else:  # Low error: error_k <= error_tol
                    logger.info(f"      Interval {current_interval_pointer} error <= tol.")
                    merged_intervals_in_this_step = False
                    # Check for h-reduction with the next interval if possible
                    if current_interval_pointer < num_intervals_K - 1:
                        error_kp1 = interval_error_estimates[current_interval_pointer + 1]
                        if (
                            not (np.isnan(error_kp1) or np.isinf(error_kp1))
                            and error_kp1 <= error_tol
                        ):
                            logger.info(
                                f"      Interval {current_interval_pointer+1} also low error ({error_kp1:.2e}). Checking h-reduction."
                            )
                            gamma_for_hr_calc = gamma_factors_col_vec
                            if current_ocp_definition.num_states > 0 and (
                                gamma_for_hr_calc is None
                                or gamma_for_hr_calc.size != current_ocp_definition.num_states
                            ):
                                logger.warning(
                                    "       Cannot perform h-reduction: gamma factors invalid for this check."
                                )
                                can_merge_flag = False
                            else:
                                can_merge_flag = h_reduce_intervals(
                                    current_interval_pointer,
                                    most_recent_ocp_solution,
                                    current_ocp_definition,
                                    self.adaptive_params,
                                    (
                                        gamma_for_hr_calc
                                        if gamma_for_hr_calc is not None
                                        else np.empty((0, 1))
                                    ),  # Pass valid empty if no states
                                    state_evaluators_list[current_interval_pointer],
                                    control_evaluators_list[current_interval_pointer],
                                    state_evaluators_list[current_interval_pointer + 1],
                                    control_evaluators_list[current_interval_pointer + 1],
                                )
                            if can_merge_flag:
                                logger.info(
                                    f"      h-reduction approved. Merging intervals {current_interval_pointer} and {current_interval_pointer+1}."
                                )
                                # New Nk for merged interval: max of the two, bounded by N_min, N_max
                                merged_Nk = max(
                                    current_collocation_nodes_list[current_interval_pointer],
                                    current_collocation_nodes_list[current_interval_pointer + 1],
                                )
                                merged_Nk = max(N_min, min(N_max, merged_Nk))
                                next_collocation_nodes_proposal.append(merged_Nk)
                                # The new interval spans from start of k to end of k+1.
                                # The end point is current_global_mesh_tau[current_interval_pointer + 2]
                                next_global_mesh_tau_proposal.append(
                                    current_global_mesh_tau[current_interval_pointer + 2]
                                )
                                current_interval_pointer += 2  # Advanced past two merged intervals
                                merged_intervals_in_this_step = True

                    if not merged_intervals_in_this_step:  # No merge, or was last interval
                        logger.info(
                            f"      h-reduction not applied or not applicable. Attempting p-reduction for interval {current_interval_pointer}."
                        )
                        p_reduce_res = p_reduce_interval(Nk_k, error_k, error_tol, N_min, N_max)
                        if p_reduce_res.was_reduction_applied:
                            logger.info(
                                f"        p-reduction applied: Nk {Nk_k} -> {p_reduce_res.new_num_collocation_nodes}"
                            )
                        else:
                            logger.info(f"        p-reduction not applied for Nk {Nk_k}.")
                        next_collocation_nodes_proposal.append(
                            p_reduce_res.new_num_collocation_nodes
                        )
                        next_global_mesh_tau_proposal.append(
                            current_global_mesh_tau[current_interval_pointer + 1]
                        )
                        current_interval_pointer += 1
            # --- End Mesh Adaptation Logic ---

            current_collocation_nodes_list = next_collocation_nodes_proposal
            current_global_mesh_tau = np.array(next_global_mesh_tau_proposal, dtype=np.float64)

            # Update solution object with the proposed mesh for the next iteration (or if loop terminates here)
            if most_recent_ocp_solution:
                most_recent_ocp_solution.num_collocation_nodes_per_interval = list(
                    current_collocation_nodes_list
                )
                most_recent_ocp_solution.global_normalized_mesh_nodes = list(
                    current_global_mesh_tau
                )

            # --- Mesh Sanity Checks ---
            if (
                not current_collocation_nodes_list and len(current_global_mesh_tau) > 1
            ):  # No intervals defined, but mesh points exist
                error_msg = "Adaptive process stopped: Mesh inconsistency (no collocation_nodes_list but mesh_nodes exist)."
                logger.error(f"  Error: {error_msg}")
                if most_recent_ocp_solution:
                    most_recent_ocp_solution.message = error_msg
                    most_recent_ocp_solution.success = False
                else:  # Should not happen if loop ran once
                    most_recent_ocp_solution = OptimalControlSolution(
                        success=False, message=error_msg
                    )
                return most_recent_ocp_solution

            if current_collocation_nodes_list and (
                len(current_collocation_nodes_list) != (len(current_global_mesh_tau) - 1)
            ):
                error_msg = (
                    f"Adaptive process stopped: Mesh structure inconsistent. "
                    f"Num intervals from nodes_list: {len(current_collocation_nodes_list)}, "
                    f"Num intervals from mesh_tau: {len(current_global_mesh_tau)-1}."
                )
                logger.error(f"  Error: {error_msg}")
                if most_recent_ocp_solution:
                    most_recent_ocp_solution.message = error_msg
                    most_recent_ocp_solution.success = False
                else:
                    most_recent_ocp_solution = OptimalControlSolution(
                        success=False,
                        message=error_msg,
                        num_collocation_nodes_per_interval=list(current_collocation_nodes_list),
                        global_normalized_mesh_nodes=list(current_global_mesh_tau),
                    )
                return most_recent_ocp_solution

            if len(current_global_mesh_tau) > 1:
                # Check for duplicate or non-increasing mesh points (after rounding for stability)
                unique_mesh_nodes_rounded = np.round(
                    current_global_mesh_tau, decimals=10
                )  # Adjust decimals if needed
                if len(np.unique(unique_mesh_nodes_rounded)) != len(current_global_mesh_tau):
                    error_msg = f"Adaptive process stopped: Duplicate mesh nodes found after rounding: {current_global_mesh_tau}."
                    logger.error(f"  Error: {error_msg}")
                    # Further logic to handle or return failed solution
                    if most_recent_ocp_solution:
                        most_recent_ocp_solution.message = error_msg
                        most_recent_ocp_solution.success = False
                    else:
                        most_recent_ocp_solution = OptimalControlSolution(
                            success=False, message=error_msg
                        )
                    return most_recent_ocp_solution

                diffs = np.diff(current_global_mesh_tau)
                if not np.all(diffs > 1e-9):  # Check for strictly increasing and non-tiny intervals
                    problem_indices = np.where(diffs <= 1e-9)[0]
                    problem_pairs_str = ", ".join(
                        [
                            f"({current_global_mesh_tau[i]:.3e}, {current_global_mesh_tau[i+1]:.3e})"
                            for i in problem_indices
                        ]
                    )
                    error_msg = (
                        f"Adaptive process stopped: Mesh nodes not strictly increasing or interval too small. "
                        f"Problem pairs at indices {problem_indices}: {problem_pairs_str}. Mesh: {current_global_mesh_tau}."
                    )
                    logger.error(f"  Error: {error_msg}")
                    if most_recent_ocp_solution:
                        most_recent_ocp_solution.message = error_msg
                        most_recent_ocp_solution.success = False
                    else:
                        most_recent_ocp_solution = OptimalControlSolution(
                            success=False, message=error_msg
                        )
                    return most_recent_ocp_solution
            # --- End Mesh Sanity Checks ---

        # Loop finished due to max_iterations
        max_iter_msg = (
            f"Adaptive mesh refinement reached max iterations ({max_iter}) "
            f"without full convergence to tolerance {error_tol:.1e}."
        )
        logger.warning(max_iter_msg)
        if most_recent_ocp_solution:
            most_recent_ocp_solution.message = max_iter_msg
            # Success status might still be true if the last NLP solve was good, but overall convergence not met.
            # Caller should check error vs tolerance on returned solution if strict convergence is required.
            return most_recent_ocp_solution
        else:  # No solution obtained at all
            return OptimalControlSolution(
                success=False,
                message=max_iter_msg
                + " No successful NLP solution obtained throughout iterations.",
                num_collocation_nodes_per_interval=list(current_collocation_nodes_list),
                global_normalized_mesh_nodes=list(current_global_mesh_tau),
            )
