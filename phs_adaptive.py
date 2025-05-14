# phs_adaptive.py
# This script implements an PHS-Adaptive mesh refinement algorithm
# for solving optimal control problems, aligned with mesh.txt.

import numpy as np
import casadi as ca
from typing import List, Dict, Any, Callable, Tuple, Union, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# Assuming solver_radau.py and radau_pseudospectral_basis.py are accessible
from rpm_solver import solve_single_phase_radau_collocation, OptimalControlSolution, OptimalControlProblem, InitialGuess, DefaultGuessValues
from rpm_basis import compute_radau_collocation_components, _compute_barycentric_weights, _evaluate_lagrange_polynomial_at_point

class AdaptiveParameters:
    """Class representing parameters for adaptive mesh refinement."""
    def __init__(self, epsilon_tol, M_max_iterations, N_min_poly_degree, N_max_poly_degree, 
                 ode_solver_tol=1e-7, num_error_sim_points=50):
        self.epsilon_tol = epsilon_tol
        self.M_max_iterations = M_max_iterations
        self.N_min_poly_degree = N_min_poly_degree
        self.N_max_poly_degree = N_max_poly_degree
        self.ode_solver_tol = ode_solver_tol
        self.num_error_sim_points = num_error_sim_points

# --- Dataclass for structured simulation results ---
@dataclass
class IntervalSimulationBundle:
    """
    Holds results from forward/backward simulations for error estimation.
    Corresponds to Section 4.1 in mesh.txt
    """
    forward_simulation_local_tau_evaluation_points: Optional[np.ndarray] = None
    state_trajectory_from_forward_simulation: Optional[np.ndarray] = None
    nlp_state_trajectory_evaluated_at_forward_simulation_points: Optional[np.ndarray] = None
    backward_simulation_local_tau_evaluation_points: Optional[np.ndarray] = None
    state_trajectory_from_backward_simulation: Optional[np.ndarray] = None
    nlp_state_trajectory_evaluated_at_backward_simulation_points: Optional[np.ndarray] = None
    are_forward_and_backward_simulations_successful: bool = True

    def __post_init__(self):
        # Ensure all ndarray fields are actually numpy arrays and 2D, if not None
        for field_name, field_def in self.__dataclass_fields__.items():
            # Check if the type is Optional[np.ndarray] or np.ndarray for robustness
            is_numpy_field = (hasattr(field_def.type, '__args__') and \
                              len(field_def.type.__args__) > 0 and \
                              field_def.type.__args__[0] is np.ndarray) or \
                             (field_def.type is np.ndarray)

            if is_numpy_field:
                val = getattr(self, field_name)
                if val is None:
                    continue  # Skip None values
                
                if not isinstance(val, np.ndarray):
                   setattr(self, field_name, np.array(val))
               
                current_val = getattr(self, field_name)
                if current_val.ndim == 1:
                    # Promote to 2D array (row vector)
                    # This assumes if 1D, it's like a single state's trajectory or points for one dimension
                    # If it's (num_vars,) for a single point, this might need adjustment based on usage
                    # However, typical usage here is (num_vars, num_points) or (1, num_points)
                    setattr(self, field_name, current_val.reshape(1, -1) if current_val.shape[0] != 0 else np.empty((0,0))) # ensure (1, N) or (0,0)
                elif current_val.ndim > 2:
                    raise ValueError(
                        f"Field {field_name} must be 1D or 2D array, "
                        f"got {current_val.ndim}D."
                    )
                # Ensure that if it's supposed to be (num_vars, num_points) and it's (num_points, num_vars), it's handled
                # This part is tricky without knowing expected num_vars. The original _extract_and_prepare_array handles this better.
                # For now, the __post_init__ will primarily ensure it's a 2D numpy array if not None.


# --- Helper: Extract and Prepare Array from NLP Solution ---
def _extract_and_prepare_array(
        casadi_value_to_extract_and_format: Any, # CasADi DM or numeric
        expected_number_of_rows: int, # num_states or num_controls
        expected_number_of_columns: int # Nk+1 for states, Nk for controls
    ) -> np.ndarray:
        """
        Extracts numerical value from CasADi, converts to NumPy array,
        and ensures correct 2D shape (rows, cols).
        """
        if hasattr(casadi_value_to_extract_and_format, 'to_DM'):
            numpy_array_from_casadi_value = np.array(casadi_value_to_extract_and_format.to_DM())
        else:
            numpy_array_from_casadi_value = np.array(casadi_value_to_extract_and_format)

        if expected_number_of_rows == 0: # No states or no controls
            return np.empty((0, expected_number_of_columns))

        if expected_number_of_rows == 1 and numpy_array_from_casadi_value.ndim == 1: # Single state/control, ensure 2D
            numpy_array_from_casadi_value = numpy_array_from_casadi_value.reshape(1, -1)
        elif numpy_array_from_casadi_value.ndim == 1 and len(numpy_array_from_casadi_value) == expected_number_of_rows: # Possibly a column vector for multiple states/controls at a single point
            numpy_array_from_casadi_value = numpy_array_from_casadi_value.reshape(expected_number_of_rows, 1) # Should be (num_rows, num_cols)

        # Ensure shape is (expected_number_of_rows, expected_number_of_columns)
        # GPOPS-II style often has (num_points, num_vars), CasADi often (num_vars, num_points)
        # This script expects (num_vars, num_points)
        if numpy_array_from_casadi_value.shape[0] != expected_number_of_rows and numpy_array_from_casadi_value.shape[1] == expected_number_of_rows:
            numpy_array_from_casadi_value = numpy_array_from_casadi_value.T

        if numpy_array_from_casadi_value.shape != (expected_number_of_rows, expected_number_of_columns) and not (expected_number_of_rows == 0):
             # This case should ideally be handled by caller or indicate a problem
             # For robustness, if one dim is 1 and can be squeezed, try that.
             squeezed_val = numpy_array_from_casadi_value.squeeze()
             if squeezed_val.ndim == 1 and expected_number_of_rows == 1 and len(squeezed_val) == expected_number_of_columns:
                 numpy_array_from_casadi_value = squeezed_val.reshape(1, expected_number_of_columns)
             # else:
                 # print(f"Warning: Shape mismatch for array. Expected ({expected_number_of_rows}, {expected_number_of_columns}), got {numpy_array_from_casadi_value.shape}. Check data source.")
                 # This might still lead to errors downstream if not perfectly aligned.
    
        return numpy_array_from_casadi_value

# --- Helper: Polynomial Interpolation (Corresponds to general numerical methods) ---
class PolynomialInterpolant:
    """
    A callable class that implements Lagrange polynomial interpolation
    using the barycentric formula.
    """
    def __init__(self, 
                interpolation_definition_nodes_local_tau: np.ndarray,
                known_values_at_interpolation_definition_nodes: np.ndarray,
                barycentric_weights_for_definition_nodes: Optional[np.ndarray] = None):
        """
        Creates a Lagrange polynomial interpolant using barycentric formula.

        Args:
            interpolation_definition_nodes_local_tau: Points at which function values are known.
            known_values_at_interpolation_definition_nodes: Function values at nodes. Shape (num_vars, num_nodes).
            barycentric_weights_for_definition_nodes: Pre-computed barycentric weights for nodes.
        """
        if not isinstance(known_values_at_interpolation_definition_nodes, np.ndarray):
            self.values_at_nodes = np.array(known_values_at_interpolation_definition_nodes)
        else:
            self.values_at_nodes = known_values_at_interpolation_definition_nodes

        if self.values_at_nodes.ndim == 1: # Ensure known_values_at_interpolation_definition_nodes is 2D (num_vars, num_nodes)
            self.values_at_nodes = self.values_at_nodes.reshape(1, -1)
        self.num_vars, self.num_nodes_val = self.values_at_nodes.shape

        if not isinstance(interpolation_definition_nodes_local_tau, np.ndarray):
            self.nodes_array = np.array(interpolation_definition_nodes_local_tau)
        else:
            self.nodes_array = interpolation_definition_nodes_local_tau
        self.num_nodes_pts = len(self.nodes_array)

        if self.num_nodes_pts != self.num_nodes_val:
            raise ValueError(f"Mismatch in number of nodes ({self.num_nodes_pts}) and number of values columns ({self.num_nodes_val})")

        if barycentric_weights_for_definition_nodes is None:
            self.bary_weights = _compute_barycentric_weights(self.nodes_array)
        else:
            self.bary_weights = np.array(barycentric_weights_for_definition_nodes) if not isinstance(barycentric_weights_for_definition_nodes, np.ndarray) else barycentric_weights_for_definition_nodes
            if len(self.bary_weights) != self.num_nodes_pts:
                raise ValueError("Provided barycentric_weights_for_definition_nodes length does not match nodes length.")

    def __call__(self, local_tau_evaluation_point_or_points: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluates the interpolant at the given point(s).

        Args:
            local_tau_evaluation_point_or_points: Point(s) at which to evaluate the interpolant.

        Returns:
            Interpolated value(s) at the evaluation point(s).
        """
        is_scalar_input = np.isscalar(local_tau_evaluation_point_or_points)
        zeta_eval_arr = np.atleast_1d(local_tau_evaluation_point_or_points)
        # values_at_nodes_arr is shape (num_vars, num_nodes_val)
        # L_j_at_zeta_pt (from _evaluate_lagrange_polynomial_at_point) is shape (num_nodes_val,)
        # np.dot(values_at_nodes_arr, L_j_at_zeta_pt) results in shape (num_vars,)
        # interpolated_values_at_evaluation_points will be shape (num_vars, len(zeta_eval_arr))
        interpolated_values_at_evaluation_points = np.zeros((self.num_vars, len(zeta_eval_arr)))
        for i, zeta_pt in enumerate(zeta_eval_arr):
            L_j_at_zeta_pt = _evaluate_lagrange_polynomial_at_point(self.nodes_array, self.bary_weights, zeta_pt)
            interpolated_values_at_evaluation_points[:, i] = np.dot(self.values_at_nodes, L_j_at_zeta_pt)

        if is_scalar_input:
            # If input was scalar, zeta_eval_arr has len 1.
            # interpolated_values_at_evaluation_points is shape (num_vars, 1).
            # We want to return a 1D array of shape (num_vars,).
            return interpolated_values_at_evaluation_points[:, 0]
        else:
            # If input was an array, interpolated_values_at_evaluation_points is (num_vars, len(zeta_eval_arr)).
            # This is the desired shape (features, samples).
            return interpolated_values_at_evaluation_points

def get_polynomial_interpolant(
    interpolation_definition_nodes_local_tau: np.ndarray,
    known_values_at_interpolation_definition_nodes: np.ndarray, # Expected shape (num_vars, num_nodes)
    barycentric_weights_for_definition_nodes: Optional[np.ndarray] = None
) -> PolynomialInterpolant:
    """
    Creates a Lagrange polynomial interpolant using barycentric formula.
    
    Args:
        interpolation_definition_nodes_local_tau: The points at which function values are known.
        known_values_at_interpolation_definition_nodes: The function values at the 'nodes'. Shape (num_vars, num_nodes).
        barycentric_weights_for_definition_nodes: Pre-computed barycentric weights for 'nodes'.

    Returns:
        A callable that evaluates the interpolant.
    """
    return PolynomialInterpolant(
        interpolation_definition_nodes_local_tau,
        known_values_at_interpolation_definition_nodes,
        barycentric_weights_for_definition_nodes
    )

# --- Section 4.1 (mesh.txt): Numerical Simulation in a Mesh Interval ---
def _simulate_dynamics_for_error_estimation(
    current_mesh_interval_index: int, 
    current_optimal_control_solution: OptimalControlSolution, 
    optimal_control_problem_definition: OptimalControlProblem,
    # NEW: Pass pre-computed polynomial evaluators
    state_polynomial_evaluator: Callable[[Union[float, np.ndarray]], np.ndarray],
    control_polynomial_evaluator: Callable[[Union[float, np.ndarray]], np.ndarray],
    ode_solver_callable_scipy: Callable = solve_ivp, 
    ode_solver_relative_tolerance: float = 1e-7,
    number_of_ode_simulation_evaluation_points: int = 50
) -> IntervalSimulationBundle:
    """
    Simulates dynamics forward (IVP, Eq. 21) and backward (TVP, Eq. 22)
    for error estimation in a given interval.
    Uses pre-computed polynomial interpolants for state and control.
    """
    interval_simulation_bundle = IntervalSimulationBundle(are_forward_and_backward_simulations_successful=False) # Default to failure

    if not current_optimal_control_solution.success or current_optimal_control_solution.raw_solution is None:
        print(f"    Warning: NLP solution unsuccessful or raw solution missing for interval {current_mesh_interval_index} in error simulation.")
        return interval_simulation_bundle

    num_states = optimal_control_problem_definition.num_states
    num_controls = optimal_control_problem_definition.num_controls
    dynamics_function = optimal_control_problem_definition.dynamics_function
    problem_parameters = optimal_control_problem_definition.problem_parameters
    
    # Time transformation parameters from the overall solution
    phase_initial_time_from_nlp_solution = current_optimal_control_solution.initial_time_variable
    phase_terminal_time_from_nlp_solution = current_optimal_control_solution.terminal_time_variable
    time_transformation_scaling_factor_alpha = (phase_terminal_time_from_nlp_solution - phase_initial_time_from_nlp_solution) / 2.0
    time_transformation_offset_alpha_0 = (phase_terminal_time_from_nlp_solution + phase_initial_time_from_nlp_solution) / 2.0
    
    global_normalized_mesh_nodes_from_nlp_solution = current_optimal_control_solution.global_normalized_mesh_nodes
    global_normalized_start_tau_of_current_interval = global_normalized_mesh_nodes_from_nlp_solution[current_mesh_interval_index]
    global_normalized_end_tau_of_current_interval = global_normalized_mesh_nodes_from_nlp_solution[current_mesh_interval_index+1]
    
    interval_transformation_scaling_factor_beta_k = (global_normalized_end_tau_of_current_interval - global_normalized_start_tau_of_current_interval) / 2.0
    if abs(interval_transformation_scaling_factor_beta_k) < 1e-12:
        print(f"    Warning: Interval {current_mesh_interval_index} has zero length. Skipping simulation.")
        return interval_simulation_bundle
    interval_transformation_offset_beta_k0 = (global_normalized_end_tau_of_current_interval + global_normalized_start_tau_of_current_interval) / 2.0
    overall_dynamics_scaling_factor_alpha_beta_k = time_transformation_scaling_factor_alpha * interval_transformation_scaling_factor_beta_k

    def dynamics_ode_right_hand_side_in_local_tau(current_local_tau_for_ode_integration: float, current_state_vector_for_ode: np.ndarray) -> np.ndarray:
        # Use pre-computed control_polynomial_evaluator
        control_vector_at_current_local_tau = control_polynomial_evaluator(current_local_tau_for_ode_integration)
        if control_vector_at_current_local_tau.ndim > 1 : control_vector_at_current_local_tau = control_vector_at_current_local_tau.flatten() # Ensure 1D for dynamics

        equivalent_global_normalized_tau_at_current_local_tau = interval_transformation_scaling_factor_beta_k * current_local_tau_for_ode_integration + interval_transformation_offset_beta_k0
        physical_time_at_current_local_tau = time_transformation_scaling_factor_alpha * equivalent_global_normalized_tau_at_current_local_tau + time_transformation_offset_alpha_0

        state_derivative_rhs_values_from_dynamics_function = dynamics_function(current_state_vector_for_ode, control_vector_at_current_local_tau, physical_time_at_current_local_tau, problem_parameters)
        state_derivative_rhs_numpy_array = np.array(state_derivative_rhs_values_from_dynamics_function, dtype=float).flatten()
        if state_derivative_rhs_numpy_array.shape[0] != num_states:
            raise ValueError(f"Dynamics function ODE RHS dimension mismatch. Expected {num_states}, got {state_derivative_rhs_numpy_array.shape[0]}.")
        return overall_dynamics_scaling_factor_alpha_beta_k * state_derivative_rhs_numpy_array

    # Use pre-computed state_polynomial_evaluator to get initial and terminal states
    initial_state_for_forward_simulation_at_interval_start = state_polynomial_evaluator(-1.0)
    if initial_state_for_forward_simulation_at_interval_start.ndim > 1:
        initial_state_for_forward_simulation_at_interval_start = initial_state_for_forward_simulation_at_interval_start.flatten()
    
    forward_simulation_local_tau_evaluation_points_input_to_ivp = np.linspace(-1, 1, number_of_ode_simulation_evaluation_points)
    forward_simulation_scipy_ivp_solution_object = ode_solver_callable_scipy(dynamics_ode_right_hand_side_in_local_tau, t_span=(-1, 1), y0=initial_state_for_forward_simulation_at_interval_start, t_eval=forward_simulation_local_tau_evaluation_points_input_to_ivp, method='RK45', rtol=ode_solver_relative_tolerance, atol=ode_solver_relative_tolerance*1e-2)
    
    interval_simulation_bundle.forward_simulation_local_tau_evaluation_points = forward_simulation_local_tau_evaluation_points_input_to_ivp
    interval_simulation_bundle.state_trajectory_from_forward_simulation = forward_simulation_scipy_ivp_solution_object.y if forward_simulation_scipy_ivp_solution_object.success else np.full((num_states, len(forward_simulation_local_tau_evaluation_points_input_to_ivp)), np.nan)
    if not forward_simulation_scipy_ivp_solution_object.success: print(f"    Fwd IVP fail int {current_mesh_interval_index}: {forward_simulation_scipy_ivp_solution_object.message}")
    # Use pre-computed state_polynomial_evaluator for NLP state evaluation
    interval_simulation_bundle.nlp_state_trajectory_evaluated_at_forward_simulation_points = state_polynomial_evaluator(forward_simulation_local_tau_evaluation_points_input_to_ivp)

    # Use pre-computed state_polynomial_evaluator to get terminal state
    terminal_state_for_backward_simulation_at_interval_end = state_polynomial_evaluator(1.0)
    if terminal_state_for_backward_simulation_at_interval_end.ndim > 1:
        terminal_state_for_backward_simulation_at_interval_end = terminal_state_for_backward_simulation_at_interval_end.flatten()
    
    backward_simulation_local_tau_integration_points_input_to_ivp = np.linspace(1, -1, number_of_ode_simulation_evaluation_points)
    backward_simulation_scipy_ivp_solution_object = ode_solver_callable_scipy(dynamics_ode_right_hand_side_in_local_tau, t_span=(1, -1), y0=terminal_state_for_backward_simulation_at_interval_end, t_eval=backward_simulation_local_tau_integration_points_input_to_ivp, method='RK45', rtol=ode_solver_relative_tolerance, atol=ode_solver_relative_tolerance*1e-2)
    
    interval_simulation_bundle.backward_simulation_local_tau_evaluation_points = np.flip(backward_simulation_local_tau_integration_points_input_to_ivp)
    interval_simulation_bundle.state_trajectory_from_backward_simulation = np.fliplr(backward_simulation_scipy_ivp_solution_object.y) if backward_simulation_scipy_ivp_solution_object.success else np.full((num_states, len(interval_simulation_bundle.backward_simulation_local_tau_evaluation_points)), np.nan)
    if not backward_simulation_scipy_ivp_solution_object.success: print(f"    Bwd TVP fail int {current_mesh_interval_index}: {backward_simulation_scipy_ivp_solution_object.message}")
    # Use pre-computed state_polynomial_evaluator for NLP state evaluation
    interval_simulation_bundle.nlp_state_trajectory_evaluated_at_backward_simulation_points = state_polynomial_evaluator(interval_simulation_bundle.backward_simulation_local_tau_evaluation_points)
    
    interval_simulation_bundle.are_forward_and_backward_simulations_successful = forward_simulation_scipy_ivp_solution_object.success and backward_simulation_scipy_ivp_solution_object.success
    return interval_simulation_bundle

# - -- Section 4.2 (mesh.txt): Relative Error Estimate in a Mesh Interval ---
def calculate_relative_error_estimate(
    mesh_interval_index_for_error_logging: int, # For logging
    interval_simulation_bundle_with_results: IntervalSimulationBundle,
    state_component_error_normalization_factors_gamma: np.ndarray # Pre-calculated normalization factors (num_states, 1)
) -> float:
    """
    Calculates the maximum relative error estimate for interval mesh_interval_index_for_error_logging.
    Follows Eq. 24, 25, 26 from mesh.txt.
    """
    if not interval_simulation_bundle_with_results.are_forward_and_backward_simulations_successful or \
       interval_simulation_bundle_with_results.state_trajectory_from_forward_simulation is None or interval_simulation_bundle_with_results.nlp_state_trajectory_evaluated_at_forward_simulation_points is None or \
       interval_simulation_bundle_with_results.state_trajectory_from_backward_simulation is None or interval_simulation_bundle_with_results.nlp_state_trajectory_evaluated_at_backward_simulation_points is None:
        print(f"    Interval {mesh_interval_index_for_error_logging}: Simulation results incomplete or failed for error calculation. Returning np.inf.")
        return np.inf

    num_states = interval_simulation_bundle_with_results.state_trajectory_from_forward_simulation.shape[0]
    if num_states == 0: return 0.0 # No states, no error

    # Calculate relative errors (Eq. 24 style)
    # Forward errors e_hat_i^(k)
    absolute_state_difference_from_forward_simulation = np.abs(interval_simulation_bundle_with_results.state_trajectory_from_forward_simulation - interval_simulation_bundle_with_results.nlp_state_trajectory_evaluated_at_forward_simulation_points)
    normalized_state_errors_all_points_forward_simulation = state_component_error_normalization_factors_gamma * absolute_state_difference_from_forward_simulation
    max_normalized_error_per_state_component_forward_simulation = np.nanmax(normalized_state_errors_all_points_forward_simulation, axis=1) if normalized_state_errors_all_points_forward_simulation.size > 0 else np.zeros(num_states)

    # Backward errors e_check_i^(k)
    absolute_state_difference_from_backward_simulation = np.abs(interval_simulation_bundle_with_results.state_trajectory_from_backward_simulation - interval_simulation_bundle_with_results.nlp_state_trajectory_evaluated_at_backward_simulation_points)
    normalized_state_errors_all_points_backward_simulation = state_component_error_normalization_factors_gamma * absolute_state_difference_from_backward_simulation
    max_normalized_error_per_state_component_backward_simulation = np.nanmax(normalized_state_errors_all_points_backward_simulation, axis=1) if normalized_state_errors_all_points_backward_simulation.size > 0 else np.zeros(num_states)

    max_normalized_error_per_state_component_in_interval = np.maximum(max_normalized_error_per_state_component_forward_simulation, max_normalized_error_per_state_component_backward_simulation)

    max_overall_normalized_relative_error_in_interval = np.nanmax(max_normalized_error_per_state_component_in_interval) if max_normalized_error_per_state_component_in_interval.size > 0 else np.inf

    if np.isnan(max_overall_normalized_relative_error_in_interval):
        print(f"    Interval {mesh_interval_index_for_error_logging}: Error calculation resulted in NaN. Treating as high error (np.inf).")
        return np.inf
    return float(max_overall_normalized_relative_error_in_interval)

# --- Section 4.3.1 (mesh.txt): p Refinement ---
class PRefineResult:
    """Class representing the result of p-refinement."""
    def __init__(self, actual_Nk_to_use, was_p_successful, unconstrained_target_Nk):
        self.actual_Nk_to_use = actual_Nk_to_use
        self.was_p_successful = was_p_successful
        self.unconstrained_target_Nk = unconstrained_target_Nk

def p_refine_interval(
    current_num_collocation_nodes_in_interval: int, 
    max_relative_error_in_interval_for_refinement: float, 
    error_tolerance_threshold_for_refinement: float, 
    max_allowable_collocation_nodes_per_interval_limit: int
) -> PRefineResult:  # Returns: (actual_Nk_to_use, was_p_successful, unconstrained_target_Nk)
    """
    Determines new polynomial degree for an interval using p-refinement.
    Follows Eq. 27 from mesh.txt.
    """
    if max_relative_error_in_interval_for_refinement <= error_tolerance_threshold_for_refinement:
        # Error is already within tolerance, no p-refinement needed from this function's perspective.
        # Return current Nk, indicate "not successful" for this specific call's purpose of *increasing* degree.
        # The unconstrained target would also be the current Nk or less.
        return PRefineResult(
            actual_Nk_to_use=current_num_collocation_nodes_in_interval,
            was_p_successful=False,
            unconstrained_target_Nk=current_num_collocation_nodes_in_interval
        )

    # Calculate P_k^+ (number_of_collocation_nodes_to_add) based on Eq. (27)
    if np.isinf(max_relative_error_in_interval_for_refinement):
        # If error is infinite, aim to increase significantly, up to N_max
        # This logic tries to make a substantial jump.
        number_of_collocation_nodes_to_add = max_allowable_collocation_nodes_per_interval_limit - current_num_collocation_nodes_in_interval
        number_of_collocation_nodes_to_add = max(1, number_of_collocation_nodes_to_add) # Ensure at least 1 if N_max is already met
    else:
        ratio = max_relative_error_in_interval_for_refinement / error_tolerance_threshold_for_refinement
        # Add at least 1 point if error > tolerance
        number_of_collocation_nodes_to_add = np.ceil(np.log10(ratio)) if ratio > 1.0 else 1.0
    
    number_of_collocation_nodes_to_add = max(1, int(number_of_collocation_nodes_to_add)) # Ensure P_k^+ >= 1
    
    # This is the N_k^[M+1] from Eq. (27) *before* applying the N_max constraint
    unconstrained_target_Nk = current_num_collocation_nodes_in_interval + number_of_collocation_nodes_to_add

    if unconstrained_target_Nk > max_allowable_collocation_nodes_per_interval_limit:
        # p-refinement "fails" because it would exceed N_max.
        # The actual Nk to use would be N_max if we were to apply p-refinement up to the limit,
        # but since it "fails", the caller (main loop) will trigger h-refinement.
        # We return N_max as the capped value, False for success, and the unconstrained target.
        return PRefineResult(
            actual_Nk_to_use=max_allowable_collocation_nodes_per_interval_limit,
            was_p_successful=False,
            unconstrained_target_Nk=unconstrained_target_Nk
        )
    
    # If unconstrained_target_Nk is <= current_num_collocation_nodes_in_interval,
    # it means P_k^+ was effectively zero (e.g. error very slightly above tolerance, log10(ratio) < 0 after ceil).
    # Since we forced number_of_collocation_nodes_to_add = max(1, ...), unconstrained_target_Nk will be > current_num_collocation_nodes_in_interval.
    # So, p-refinement is successful and within N_max.
    return PRefineResult(
        actual_Nk_to_use=unconstrained_target_Nk,
        was_p_successful=True,
        unconstrained_target_Nk=unconstrained_target_Nk
    )

# --- Section 4.3.2 (mesh.txt): h Refinement ---
class HRefineResult:
    """Class representing the result of h-refinement."""
    def __init__(self, collocation_nodes_for_new_subintervals, num_new_subintervals):
        self.collocation_nodes_for_new_subintervals = collocation_nodes_for_new_subintervals
        self.num_new_subintervals = num_new_subintervals

def h_refine_params(
    target_num_collocation_nodes_that_triggered_h_refinement: int,
    min_allowable_collocation_nodes_for_new_subintervals: int
) -> HRefineResult:
    """
    Determines parameters for h-refinement (splitting an interval).
    Follows Eq. 28 from mesh.txt.
    """
    number_of_new_subintervals_to_create_from_split = max(2, int(np.ceil(target_num_collocation_nodes_that_triggered_h_refinement / min_allowable_collocation_nodes_for_new_subintervals)))
    list_of_collocation_nodes_for_each_new_subinterval = [min_allowable_collocation_nodes_for_new_subintervals] * number_of_new_subintervals_to_create_from_split
    return HRefineResult(
        collocation_nodes_for_new_subintervals=list_of_collocation_nodes_for_each_new_subinterval,
        num_new_subintervals=number_of_new_subintervals_to_create_from_split
    )

# --- Transformation helpers for h-reduction (related to Eq. 7, 30 in mesh.txt) ---
def _map_global_normalized_tau_to_local_interval_tau(global_normalized_tau_point_to_map: float, global_normalized_start_tau_of_interval_definition: float, global_normalized_end_tau_of_interval_definition: float) -> float:
    """Maps tau in [global_normalized_start_tau_of_interval_definition, global_normalized_end_tau_of_interval_definition] to zeta in [-1, 1]. Inverse of Eq. 7 (mesh.txt)."""
    interval_definition_scaling_factor_beta = (global_normalized_end_tau_of_interval_definition - global_normalized_start_tau_of_interval_definition) / 2.0
    interval_definition_offset_beta0 = (global_normalized_end_tau_of_interval_definition + global_normalized_start_tau_of_interval_definition) / 2.0
    if abs(interval_definition_scaling_factor_beta) < 1e-12: return 0.0 
    return (global_normalized_tau_point_to_map - interval_definition_offset_beta0) / interval_definition_scaling_factor_beta

def _map_local_interval_tau_to_global_normalized_tau(local_interval_tau_point_to_map: float, global_normalized_start_tau_of_interval_definition: float, global_normalized_end_tau_of_interval_definition: float) -> float:
    """Maps zeta in [-1, 1] to tau in [global_normalized_start_tau_of_interval_definition, global_normalized_end_tau_of_interval_definition]. Eq. 7 (mesh.txt)."""
    interval_definition_scaling_factor_beta = (global_normalized_end_tau_of_interval_definition - global_normalized_start_tau_of_interval_definition) / 2.0
    interval_definition_offset_beta0 = (global_normalized_end_tau_of_interval_definition + global_normalized_start_tau_of_interval_definition) / 2.0
    return interval_definition_scaling_factor_beta * local_interval_tau_point_to_map + interval_definition_offset_beta0

def _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(local_tau_in_source_interval_k: float, global_normalized_start_tau_of_source_interval_k: float, global_normalized_shared_tau_node_between_intervals: float, global_normalized_end_tau_of_target_interval_k_plus_1: float) -> float:
    """Transforms zeta in interval k to equivalent zeta in interval k+1. Eq. 30 (mesh.txt)."""
    equivalent_global_normalized_tau_point_for_mapping = _map_local_interval_tau_to_global_normalized_tau(local_tau_in_source_interval_k, global_normalized_start_tau_of_source_interval_k, global_normalized_shared_tau_node_between_intervals)
    equivalent_local_tau_in_target_interval_k_plus_1 = _map_global_normalized_tau_to_local_interval_tau(equivalent_global_normalized_tau_point_for_mapping, global_normalized_shared_tau_node_between_intervals, global_normalized_end_tau_of_target_interval_k_plus_1)
    return equivalent_local_tau_in_target_interval_k_plus_1

def _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(local_tau_in_source_interval_k_plus_1: float, global_normalized_start_tau_of_source_interval_k: float, global_normalized_shared_tau_node_between_intervals: float, global_normalized_end_tau_of_target_interval_k_plus_1: float) -> float:
    """Transforms zeta in interval k+1 to equivalent zeta in interval k. Inverse of chi. Eq. 30 (mesh.txt)."""
    equivalent_global_normalized_tau_point_for_mapping = _map_local_interval_tau_to_global_normalized_tau(local_tau_in_source_interval_k_plus_1, global_normalized_shared_tau_node_between_intervals, global_normalized_end_tau_of_target_interval_k_plus_1)
    equivalent_local_tau_in_target_interval_k = _map_global_normalized_tau_to_local_interval_tau(equivalent_global_normalized_tau_point_for_mapping, global_normalized_start_tau_of_source_interval_k, global_normalized_shared_tau_node_between_intervals)
    return equivalent_local_tau_in_target_interval_k

# --- Section 4.3.3 (mesh.txt): h Reduction ---
def h_reduce_intervals(
    first_mesh_interval_index_for_merge_consideration: int, 
    nlp_solution: OptimalControlSolution, 
    problem_definition: OptimalControlProblem,
    adaptive_refinement_parameters_for_h_reduction: AdaptiveParameters, 
    state_component_error_normalization_factors_gamma: np.ndarray, # Pre-calculated
    state_polynomial_evaluator_for_first_interval: Callable, 
    control_polynomial_evaluator_for_first_interval: Optional[Callable],
    state_polynomial_evaluator_for_second_interval: Callable, 
    control_polynomial_evaluator_for_second_interval: Optional[Callable]
) -> bool:
    """
    Checks if two adjacent intervals first_mesh_interval_index_for_merge_consideration and first_mesh_interval_index_for_merge_consideration+1 can be merged.
    Follows logic of Section 4.3.3 in mesh.txt (Eq. 31-34).
    Returns True if merge is successful (error condition met), False otherwise.
    """
    print(f"    h-reduction check for intervals {first_mesh_interval_index_for_merge_consideration} and {first_mesh_interval_index_for_merge_consideration+1}.")
    error_tolerance_for_allowing_merge = adaptive_refinement_parameters_for_h_reduction.epsilon_tol
    ode_solver_tol = adaptive_refinement_parameters_for_h_reduction.ode_solver_tol
    ode_solver_absolute_tolerance_for_merge_simulation = ode_solver_tol * 1e-1 
    num_error_sim_points = adaptive_refinement_parameters_for_h_reduction.num_error_sim_points

    num_states = problem_definition.num_states
    num_controls = problem_definition.num_controls # num_controls not directly used here, but for context
    dynamics_function = problem_definition.dynamics_function
    problem_parameters = problem_definition.problem_parameters
    
    if nlp_solution.raw_solution is None: # Should be caught by nlp_solution.success earlier
        print("      h-reduction failed: Raw solution missing.")
        return False

    global_normalized_mesh_nodes = nlp_solution.global_normalized_mesh_nodes # Assumed np.array
    global_normalized_start_tau_first_interval = global_normalized_mesh_nodes[first_mesh_interval_index_for_merge_consideration]
    global_normalized_shared_tau_node_for_merge = global_normalized_mesh_nodes[first_mesh_interval_index_for_merge_consideration + 1]
    global_normalized_end_tau_second_interval = global_normalized_mesh_nodes[first_mesh_interval_index_for_merge_consideration + 2]

    original_interval_scaling_factor_beta_first_interval = (global_normalized_shared_tau_node_for_merge - global_normalized_start_tau_first_interval) / 2.0
    original_interval_scaling_factor_beta_second_interval = (global_normalized_end_tau_second_interval - global_normalized_shared_tau_node_for_merge) / 2.0

    if abs(original_interval_scaling_factor_beta_first_interval) < 1e-12 or abs(original_interval_scaling_factor_beta_second_interval) < 1e-12:
        print("      h-reduction check: One of the intervals has zero length. Merge not possible.")
        return False

    solved_initial_time = nlp_solution.initial_time_variable
    solved_terminal_time = nlp_solution.terminal_time_variable
    alpha = (solved_terminal_time - solved_initial_time) / 2.0
    overall_dynamics_scaling_factor_first_interval = alpha * original_interval_scaling_factor_beta_first_interval
    overall_dynamics_scaling_factor_second_interval = alpha * original_interval_scaling_factor_beta_second_interval
    alpha_0 = (solved_terminal_time + solved_initial_time) / 2.0
    
    def _get_control_value_with_clipping_at_local_tau(control_polynomial_evaluator_for_interval: Optional[Callable], local_tau_point_for_control_evaluation: float) -> np.ndarray:
        if control_polynomial_evaluator_for_interval is None: return np.array([])
        clipped_local_tau_for_control_polynomial = np.clip(local_tau_point_for_control_evaluation, -1.0, 1.0)
        u_val = control_polynomial_evaluator_for_interval(clipped_local_tau_for_control_polynomial)
        return np.atleast_1d(u_val.squeeze())

    def merged_domain_forward_simulation_ode_rhs(current_local_tau_in_first_interval_basis: float, X_current_ode: np.ndarray) -> np.ndarray:
        U_val_ode = _get_control_value_with_clipping_at_local_tau(control_polynomial_evaluator_for_first_interval, current_local_tau_in_first_interval_basis)
        X_clipped = np.clip(X_current_ode, -1e6, 1e6)
        current_tau_global = _map_local_interval_tau_to_global_normalized_tau(current_local_tau_in_first_interval_basis, global_normalized_start_tau_first_interval, global_normalized_shared_tau_node_for_merge)
        t_actual = alpha * current_tau_global + alpha_0
        F_rhs_list_or_array = dynamics_function(X_clipped, U_val_ode, t_actual, problem_parameters)
        F_rhs_np = np.array(F_rhs_list_or_array, dtype=float).flatten()
        return overall_dynamics_scaling_factor_first_interval * F_rhs_np

    def merged_domain_backward_simulation_ode_rhs(current_local_tau_in_second_interval_basis: float, X_current_ode: np.ndarray) -> np.ndarray:
        U_val_ode = _get_control_value_with_clipping_at_local_tau(control_polynomial_evaluator_for_second_interval, current_local_tau_in_second_interval_basis)
        X_clipped = np.clip(X_current_ode, -1e6, 1e6)
        current_tau_global = _map_local_interval_tau_to_global_normalized_tau(current_local_tau_in_second_interval_basis, global_normalized_shared_tau_node_for_merge, global_normalized_end_tau_second_interval)
        t_actual = alpha * current_tau_global + alpha_0
        F_rhs_list_or_array = dynamics_function(X_clipped, U_val_ode, t_actual, problem_parameters)
        F_rhs_np = np.array(F_rhs_list_or_array, dtype=float).flatten()
        return overall_dynamics_scaling_factor_second_interval * F_rhs_np
    
    Y_solved_list = nlp_solution.solved_state_trajectories_per_interval
    Nk_k = problem_definition.collocation_points_per_interval[first_mesh_interval_index_for_merge_consideration]
    Nk_kp1 = problem_definition.collocation_points_per_interval[first_mesh_interval_index_for_merge_consideration+1]

    try:
        if Y_solved_list and first_mesh_interval_index_for_merge_consideration < len(Y_solved_list):
            Xk_nlp_discrete = Y_solved_list[first_mesh_interval_index_for_merge_consideration]
        else: # Fallback
            casadi_optimization_problem_object = nlp_solution.opti_object
            raw_sol = nlp_solution.raw_solution
            Xk_nlp_discrete_raw = raw_sol.value(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables[first_mesh_interval_index_for_merge_consideration])
            Xk_nlp_discrete = _extract_and_prepare_array(Xk_nlp_discrete_raw, num_states, Nk_k + 1)
        initial_state_for_merged_domain_forward_sim = Xk_nlp_discrete[:, 0].flatten()
    except Exception as e:
        print(f"      h-reduction failed: Error getting initial state Xk for merged IVP: {e}")
        return False

    target_end_local_tau_in_first_interval_basis_for_merged_fwd_sim = _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(1.0, global_normalized_start_tau_first_interval, global_normalized_shared_tau_node_for_merge, global_normalized_end_tau_second_interval)
    num_ivp_sim_pts = max(num_error_sim_points // 2, int(num_error_sim_points * (target_end_local_tau_in_first_interval_basis_for_merged_fwd_sim - (-1.0)) / 2.0) if target_end_local_tau_in_first_interval_basis_for_merged_fwd_sim > -1.0 else num_error_sim_points // 2)
    if num_ivp_sim_pts < 2: num_ivp_sim_pts = 2
    local_tau_evaluation_points_for_merged_fwd_sim = np.linspace(-1.0, target_end_local_tau_in_first_interval_basis_for_merged_fwd_sim, num_ivp_sim_pts)

    print(f"      h-reduction: Starting Merged IVP sim from zeta_k=-1 to {target_end_local_tau_in_first_interval_basis_for_merged_fwd_sim:.3f} ({num_ivp_sim_pts} pts)")
    try:
        merged_domain_forward_simulation_scipy_ivp_solution = solve_ivp(merged_domain_forward_simulation_ode_rhs, t_span=(-1.0, target_end_local_tau_in_first_interval_basis_for_merged_fwd_sim), y0=initial_state_for_merged_domain_forward_sim, t_eval=local_tau_evaluation_points_for_merged_fwd_sim, method='RK45', rtol=ode_solver_tol, atol=ode_solver_absolute_tolerance_for_merge_simulation)
        simulated_state_trajectory_merged_domain_fwd = merged_domain_forward_simulation_scipy_ivp_solution.y if merged_domain_forward_simulation_scipy_ivp_solution.success else np.full((num_states, len(local_tau_evaluation_points_for_merged_fwd_sim)), np.nan)
        if not merged_domain_forward_simulation_scipy_ivp_solution.success: print(f"      Merged IVP failed: {merged_domain_forward_simulation_scipy_ivp_solution.message}")
    except Exception as e:
        print(f"      Exception during merged IVP simulation: {e}")
        simulated_state_trajectory_merged_domain_fwd = np.full((num_states, len(local_tau_evaluation_points_for_merged_fwd_sim)), np.nan)

    try:
        if Y_solved_list and (first_mesh_interval_index_for_merge_consideration + 1) < len(Y_solved_list):
            Xkp1_nlp_discrete = Y_solved_list[first_mesh_interval_index_for_merge_consideration + 1]
        else: # Fallback
            casadi_optimization_problem_object = nlp_solution.opti_object
            raw_sol = nlp_solution.raw_solution
            Xkp1_nlp_discrete_raw = raw_sol.value(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables[first_mesh_interval_index_for_merge_consideration+1])
            Xkp1_nlp_discrete = _extract_and_prepare_array(Xkp1_nlp_discrete_raw, num_states, Nk_kp1 + 1)
        terminal_state_for_merged_domain_backward_sim = Xkp1_nlp_discrete[:, -1].flatten()
    except Exception as e:
         print(f"      h-reduction failed: Error getting terminal state Xkp1 for merged TVP: {e}")
         return False

    target_end_local_tau_in_second_interval_basis_for_merged_bwd_sim = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(-1.0, global_normalized_start_tau_first_interval, global_normalized_shared_tau_node_for_merge, global_normalized_end_tau_second_interval)
    num_tvp_sim_pts = max(num_error_sim_points // 2, int(num_error_sim_points * (1.0 - target_end_local_tau_in_second_interval_basis_for_merged_bwd_sim) / 2.0) if target_end_local_tau_in_second_interval_basis_for_merged_bwd_sim < 1.0 else num_error_sim_points // 2)
    if num_tvp_sim_pts < 2: num_tvp_sim_pts = 2
    local_tau_integration_points_for_merged_bwd_sim = np.linspace(1.0, target_end_local_tau_in_second_interval_basis_for_merged_bwd_sim, num_tvp_sim_pts)

    print(f"      h-reduction: Starting Merged TVP sim from zeta_kp1=1 to {target_end_local_tau_in_second_interval_basis_for_merged_bwd_sim:.3f} ({num_tvp_sim_pts} pts)")
    try:
        merged_domain_backward_simulation_scipy_ivp_solution = solve_ivp(merged_domain_backward_simulation_ode_rhs, t_span=(1.0, target_end_local_tau_in_second_interval_basis_for_merged_bwd_sim), y0=terminal_state_for_merged_domain_backward_sim, t_eval=local_tau_integration_points_for_merged_bwd_sim, method='RK45', rtol=ode_solver_tol, atol=ode_solver_absolute_tolerance_for_merge_simulation)
        sorted_local_tau_evaluation_points_for_merged_bwd_sim = np.flip(local_tau_integration_points_for_merged_bwd_sim)
        simulated_state_trajectory_merged_domain_bwd = np.fliplr(merged_domain_backward_simulation_scipy_ivp_solution.y) if merged_domain_backward_simulation_scipy_ivp_solution.success else np.full((num_states, len(sorted_local_tau_evaluation_points_for_merged_bwd_sim)), np.nan)
        if not merged_domain_backward_simulation_scipy_ivp_solution.success: print(f"      Merged TVP failed: {merged_domain_backward_simulation_scipy_ivp_solution.message}")
    except Exception as e:
        print(f"      Exception during merged TVP simulation: {e}")
        simulated_state_trajectory_merged_domain_bwd = np.full((num_states, len(sorted_local_tau_evaluation_points_for_merged_bwd_sim)), np.nan)
    
    if num_states == 0 : # No states, merge is fine if simulations didn't crash
        can_intervals_be_merged_based_on_estimated_error = (merged_domain_forward_simulation_scipy_ivp_solution.success if 'merged_domain_forward_simulation_scipy_ivp_solution' in locals() else False) and \
                   (merged_domain_backward_simulation_scipy_ivp_solution.success if 'merged_domain_backward_simulation_scipy_ivp_solution' in locals() else False)
        print(f"      h-reduction check (no states): can_intervals_be_merged_based_on_estimated_error = {can_intervals_be_merged_based_on_estimated_error}")
        return can_intervals_be_merged_based_on_estimated_error


    all_normalized_forward_errors_for_merged_domain_simulation = []
    for i_pt, zeta_k_pt_ivp in enumerate(local_tau_evaluation_points_for_merged_fwd_sim):
        X_hat_bar_pt = simulated_state_trajectory_merged_domain_fwd[:, i_pt]
        if np.any(np.isnan(X_hat_bar_pt)): continue

        if -1.0 <= zeta_k_pt_ivp <= 1.0 + 1e-9:
            target_nlp_state_for_comparison_in_merged_domain_simulation = state_polynomial_evaluator_for_first_interval(zeta_k_pt_ivp)
        else:
            zeta_kp1_equiv = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(zeta_k_pt_ivp, global_normalized_start_tau_first_interval, global_normalized_shared_tau_node_for_merge, global_normalized_end_tau_second_interval)
            target_nlp_state_for_comparison_in_merged_domain_simulation = state_polynomial_evaluator_for_second_interval(zeta_kp1_equiv)

        if target_nlp_state_for_comparison_in_merged_domain_simulation.ndim > 1: target_nlp_state_for_comparison_in_merged_domain_simulation = target_nlp_state_for_comparison_in_merged_domain_simulation.flatten()
        abs_diff = np.abs(X_hat_bar_pt - target_nlp_state_for_comparison_in_merged_domain_simulation)
        scaled_error_pt_components = state_component_error_normalization_factors_gamma.flatten() * abs_diff
        all_normalized_forward_errors_for_merged_domain_simulation.extend(list(scaled_error_pt_components))

    all_normalized_backward_errors_for_merged_domain_simulation = []
    for i_pt, zeta_kp1_pt_tvp in enumerate(sorted_local_tau_evaluation_points_for_merged_bwd_sim):
        X_check_bar_pt = simulated_state_trajectory_merged_domain_bwd[:, i_pt]
        if np.any(np.isnan(X_check_bar_pt)): continue

        if -1.0 - 1e-9 <= zeta_kp1_pt_tvp <= 1.0:
            target_nlp_state_for_comparison_in_merged_domain_simulation = state_polynomial_evaluator_for_second_interval(zeta_kp1_pt_tvp)
        else:
            zeta_k_equiv = _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(zeta_kp1_pt_tvp, global_normalized_start_tau_first_interval, global_normalized_shared_tau_node_for_merge, global_normalized_end_tau_second_interval)
            target_nlp_state_for_comparison_in_merged_domain_simulation = state_polynomial_evaluator_for_first_interval(zeta_k_equiv)

        if target_nlp_state_for_comparison_in_merged_domain_simulation.ndim > 1: target_nlp_state_for_comparison_in_merged_domain_simulation = target_nlp_state_for_comparison_in_merged_domain_simulation.flatten()
        abs_diff = np.abs(X_check_bar_pt - target_nlp_state_for_comparison_in_merged_domain_simulation)
        scaled_error_pt_components = state_component_error_normalization_factors_gamma.flatten() * abs_diff
        all_normalized_backward_errors_for_merged_domain_simulation.extend(list(scaled_error_pt_components))

    # Strictly following Eq. 34 from mesh.txt:
    # max_relative_error_for_potential_merged_interval = max(max(all_normalized_forward_errors_for_merged_domain_simulation), max(all_normalized_backward_errors_for_merged_domain_simulation))
    max_forward_error_across_merged_domain_simulation = np.nanmax(all_normalized_forward_errors_for_merged_domain_simulation) if all_normalized_forward_errors_for_merged_domain_simulation else np.inf
    max_backward_error_across_merged_domain_simulation = np.nanmax(all_normalized_backward_errors_for_merged_domain_simulation) if all_normalized_backward_errors_for_merged_domain_simulation else np.inf
    max_relative_error_for_potential_merged_interval = max(max_forward_error_across_merged_domain_simulation, max_backward_error_across_merged_domain_simulation)

    if np.isnan(max_relative_error_for_potential_merged_interval):
        print(f"      h-reduction check: max_relative_error_for_potential_merged_interval calculation resulted in NaN. Merge not approved.")
        max_relative_error_for_potential_merged_interval = np.inf 

    print(f"      h-reduction check result: max_relative_error_for_potential_merged_interval = {max_relative_error_for_potential_merged_interval:.4e}")
    can_intervals_be_merged_based_on_estimated_error = max_relative_error_for_potential_merged_interval <= error_tolerance_for_allowing_merge

    if can_intervals_be_merged_based_on_estimated_error: print(f"      h-reduction condition met. Merge approved for intervals {first_mesh_interval_index_for_merge_consideration}, {first_mesh_interval_index_for_merge_consideration+1}.")
    else: print(f"      h-reduction condition NOT met (error {max_relative_error_for_potential_merged_interval:.2e} > tol {error_tolerance_for_allowing_merge:.2e}). Merge failed.")

    return can_intervals_be_merged_based_on_estimated_error

# --- Section 4.3.4 (mesh.txt): p Reduction ---
class PReduceResult:
    """Class representing the result of p-reduction."""
    def __init__(self, new_num_collocation_nodes, was_reduction_applied):
        self.new_num_collocation_nodes = new_num_collocation_nodes
        self.was_reduction_applied = was_reduction_applied

def  p_reduce_interval(
    current_num_collocation_nodes_in_interval: int, 
    max_relative_error_in_interval_for_reduction: float, 
    error_tolerance_threshold_for_reduction: float,
    min_allowable_collocation_nodes_per_interval_after_reduction: int, 
    max_allowable_collocation_nodes_per_interval_for_delta_calc: int
) -> PReduceResult:
    """
    Determines new polynomial degree (new_num_collocation_nodes_after_p_reduction) for an interval using p-reduction.
    Follows Eq. 36 from mesh.txt.
    Returns new Nk and a boolean indicating if reduction was applied.
    """
    if max_relative_error_in_interval_for_reduction > error_tolerance_threshold_for_reduction:
        return PReduceResult(
            new_num_collocation_nodes=current_num_collocation_nodes_in_interval,
            was_reduction_applied=False
        )
    if current_num_collocation_nodes_in_interval <= min_allowable_collocation_nodes_per_interval_after_reduction:
        return PReduceResult(
            new_num_collocation_nodes=current_num_collocation_nodes_in_interval,
            was_reduction_applied=False
        )

    p_reduction_control_parameter_delta = float(min_allowable_collocation_nodes_per_interval_after_reduction + max_allowable_collocation_nodes_per_interval_for_delta_calc - current_num_collocation_nodes_in_interval)
    if abs(p_reduction_control_parameter_delta) < 1e-9: p_reduction_control_parameter_delta = 1.0 # Avoid division by zero; pragmatic choice

    if max_relative_error_in_interval_for_reduction < 1e-16: # Error is essentially zero
        number_of_collocation_nodes_to_remove = current_num_collocation_nodes_in_interval - min_allowable_collocation_nodes_per_interval_after_reduction
    else:
        ratio = error_tolerance_threshold_for_reduction / max_relative_error_in_interval_for_reduction # Should be >= 1
        if ratio < 1.0: number_of_collocation_nodes_to_remove = 0
        else:
            try:
                log_arg = np.power(ratio, 1.0 / p_reduction_control_parameter_delta)
                number_of_collocation_nodes_to_remove = np.floor(np.log10(log_arg)) if log_arg >= 1.0 else 0
            except (ValueError, OverflowError, ZeroDivisionError):
                number_of_collocation_nodes_to_remove = 0

    number_of_collocation_nodes_to_remove = max(0, int(number_of_collocation_nodes_to_remove))
    new_num_collocation_nodes_after_p_reduction = max(min_allowable_collocation_nodes_per_interval_after_reduction, current_num_collocation_nodes_in_interval - number_of_collocation_nodes_to_remove)
    was_collocation_nodes_p_reduction_applied = new_num_collocation_nodes_after_p_reduction < current_num_collocation_nodes_in_interval
    return PReduceResult(
        new_num_collocation_nodes=new_num_collocation_nodes_after_p_reduction,
        was_reduction_applied=was_collocation_nodes_p_reduction_applied
    )

# - -- Robust Default Initial Guess Generation ---
def  _generate_robust_default_initial_guess(
    base_optimal_control_problem_definition_for_defaults: OptimalControlProblem,
    current_list_of_collocation_nodes_per_interval_for_guess: List[int],
    propagated_phase_initial_time_guess: Optional[float] = None,
    propagated_phase_terminal_time_guess: Optional[float] = None,
    propagated_integral_values_guess: Optional[Union[float, np.ndarray]] = None
) -> InitialGuess:
    """
    Generates a robust, correctly dimensioned default initial guess.
    """
    num_intervals = len(current_list_of_collocation_nodes_per_interval_for_guess)
    num_states = base_optimal_control_problem_definition_for_defaults.num_states
    num_controls = base_optimal_control_problem_definition_for_defaults.num_controls
    num_integrals = base_optimal_control_problem_definition_for_defaults.num_integrals

    default_value_for_state_variable_guess = getattr(base_optimal_control_problem_definition_for_defaults.default_initial_guess_values, 'state', 0.0)
    default_value_for_control_variable_guess = getattr(base_optimal_control_problem_definition_for_defaults.default_initial_guess_values, 'control', 0.0)
    
    list_of_state_trajectory_guesses_for_each_interval = []
    list_of_control_trajectory_guesses_for_each_interval = []

    for mesh_interval_index in range(num_intervals):
        Nk_k = current_list_of_collocation_nodes_per_interval_for_guess[mesh_interval_index]

        state_trajectory_guess_for_current_interval = np.full((num_states, Nk_k + 1), default_value_for_state_variable_guess)
        list_of_state_trajectory_guesses_for_each_interval.append(state_trajectory_guess_for_current_interval)

        if num_controls > 0:
            control_trajectory_guess_for_current_interval = np.full((num_controls, Nk_k), default_value_for_control_variable_guess)
        else:
            control_trajectory_guess_for_current_interval = np.empty((0, Nk_k))
        list_of_control_trajectory_guesses_for_each_interval.append(control_trajectory_guess_for_current_interval)

    final_initial_time_guess_value = propagated_phase_initial_time_guess
    if final_initial_time_guess_value is None and base_optimal_control_problem_definition_for_defaults.initial_guess:
        final_initial_time_guess_value = base_optimal_control_problem_definition_for_defaults.initial_guess.initial_time_variable
    
    final_terminal_time_guess_value = propagated_phase_terminal_time_guess
    if final_terminal_time_guess_value is None and base_optimal_control_problem_definition_for_defaults.initial_guess:
        final_terminal_time_guess_value = base_optimal_control_problem_definition_for_defaults.initial_guess.terminal_time_variable

    final_integral_values_guess: Optional[Union[float, List[float], np.ndarray]] = None
    if num_integrals > 0:
        if propagated_integral_values_guess is not None:
            final_integral_values_guess = propagated_integral_values_guess
        else:
            default_value_for_integral_guess = getattr(base_optimal_control_problem_definition_for_defaults.default_initial_guess_values, 'integral', 0.0)
            if base_optimal_control_problem_definition_for_defaults.initial_guess and base_optimal_control_problem_definition_for_defaults.initial_guess.integrals is not None:
                raw_guess = base_optimal_control_problem_definition_for_defaults.initial_guess.integrals
            else:
                raw_guess = [default_value_for_integral_guess] * num_integrals if num_integrals > 1 else default_value_for_integral_guess

            if num_integrals == 1:
                 final_integral_values_guess = float(raw_guess) if not isinstance(raw_guess, (list, np.ndarray)) else float(raw_guess[0])
            elif isinstance(raw_guess, (list, np.ndarray)) and len(raw_guess) == num_integrals:
                 final_integral_values_guess = np.array(raw_guess)
            else: 
                 final_integral_values_guess = np.full(num_integrals, default_value_for_integral_guess)
    
    return InitialGuess(
        initial_time_variable=final_initial_time_guess_value,
        terminal_time_variable=final_terminal_time_guess_value,
        states=list_of_state_trajectory_guesses_for_each_interval,
        controls=list_of_control_trajectory_guesses_for_each_interval,
        integrals=final_integral_values_guess
    )

# --- Initial Guess Propagation from Previous Solution ---
def _propagate_guess_from_previous(
    previous_optimal_control_solution_to_propagate_from: OptimalControlSolution,
    base_optimal_control_problem_definition_for_propagation: OptimalControlProblem,
    target_list_of_collocation_nodes_per_interval_for_propagation: List[int],
    target_global_normalized_mesh_nodes_for_propagation: np.ndarray
) -> InitialGuess:
    """
    Creates an initial guess for the current NLP solve, propagating from previous_optimal_control_solution_to_propagate_from.
    """
    t0_prop = previous_optimal_control_solution_to_propagate_from.initial_time_variable
    tf_prop = previous_optimal_control_solution_to_propagate_from.terminal_time_variable
    integrals_prop = previous_optimal_control_solution_to_propagate_from.integrals

    initial_guess_propagated_from_previous_solution = _generate_robust_default_initial_guess(
        base_optimal_control_problem_definition_for_propagation, target_list_of_collocation_nodes_per_interval_for_propagation,
        propagated_phase_initial_time_guess=t0_prop, propagated_phase_terminal_time_guess=tf_prop, propagated_integral_values_guess=integrals_prop
    )

    num_collocation_nodes_list_from_previous_solution = previous_optimal_control_solution_to_propagate_from.num_collocation_nodes_list_at_solve_time
    global_normalized_mesh_nodes_from_previous_solution = previous_optimal_control_solution_to_propagate_from.global_mesh_nodes_at_solve_time

    can_state_control_trajectories_be_directly_propagated = False
    if num_collocation_nodes_list_from_previous_solution is not None and global_normalized_mesh_nodes_from_previous_solution is not None:
        if np.array_equal(target_list_of_collocation_nodes_per_interval_for_propagation, num_collocation_nodes_list_from_previous_solution) and \
           np.allclose(target_global_normalized_mesh_nodes_for_propagation, global_normalized_mesh_nodes_from_previous_solution):
            can_state_control_trajectories_be_directly_propagated = True

    if can_state_control_trajectories_be_directly_propagated:
        print("  Mesh structure identical to previous. Propagating state/control trajectories directly.")
        state_trajectories_list_from_previous_solution = previous_optimal_control_solution_to_propagate_from.solved_state_trajectories_per_interval
        control_trajectories_list_from_previous_solution = previous_optimal_control_solution_to_propagate_from.solved_control_trajectories_per_interval

        if state_trajectories_list_from_previous_solution and len(state_trajectories_list_from_previous_solution) == len(target_list_of_collocation_nodes_per_interval_for_propagation):
            initial_guess_propagated_from_previous_solution.states = state_trajectories_list_from_previous_solution
        else:
            print("    Warning: Previous Y_solved_values_list mismatch or missing. Using default states.")

        if control_trajectories_list_from_previous_solution and len(control_trajectories_list_from_previous_solution) == len(target_list_of_collocation_nodes_per_interval_for_propagation):
            initial_guess_propagated_from_previous_solution.controls = control_trajectories_list_from_previous_solution
        else:
             print("    Warning: Previous U_solved_values_list mismatch or missing. Using default controls.")
    else:
        print("  Mesh structure changed. Using robust default for state/control trajectories (times/integrals propagated).")

    return initial_guess_propagated_from_previous_solution

# --- Helper for Gamma Normalization Factor Calculation ---
def _calculate_gamma_normalizers(optimal_control_solution_for_gamma_calculation: OptimalControlSolution, optimal_control_problem_definition_for_gamma_calculation: OptimalControlProblem) -> Optional[np.ndarray]:
    """Calculates gamma_i normalization factors (Eq. 25) for error estimation."""
    if not optimal_control_solution_for_gamma_calculation.success or optimal_control_solution_for_gamma_calculation.raw_solution is None:
        return None

    num_states = optimal_control_problem_definition_for_gamma_calculation.num_states
    if num_states == 0: return np.array([]).reshape(0,1) # No states, no gamma

    Y_solved_list = optimal_control_solution_for_gamma_calculation.solved_state_trajectories_per_interval
    if not Y_solved_list: # Should have been populated if solve was successful
        print("    Warning: solved_state_trajectories_per_interval missing in optimal_control_solution_for_gamma_calculation for gamma calculation.")
        return None # Cannot calculate

    max_absolute_value_for_each_state_component_across_all_intervals = np.zeros(num_states)
    first_interval = True
    for Xk_nlp_discrete in Y_solved_list:
        if Xk_nlp_discrete.size == 0: continue # Skip if an interval somehow had no state points
        max_absolute_value_for_each_state_component_in_current_interval = np.max(np.abs(Xk_nlp_discrete), axis=1)
        if first_interval:
            max_absolute_value_for_each_state_component_across_all_intervals = max_absolute_value_for_each_state_component_in_current_interval
            first_interval = False
        else:
            max_absolute_value_for_each_state_component_across_all_intervals = np.maximum(max_absolute_value_for_each_state_component_across_all_intervals, max_absolute_value_for_each_state_component_in_current_interval)
    
    inverse_gamma_factors_plus_one = 1.0 + max_absolute_value_for_each_state_component_across_all_intervals
    state_component_error_normalization_factors_gamma_column_vector = 1.0 / np.maximum(inverse_gamma_factors_plus_one, 1e-12) # Avoid division by zero
    return state_component_error_normalization_factors_gamma_column_vector.reshape(-1, 1)


# --- Main Adaptive Loop (Algorithm 1 from mesh.txt) ---
def run_phs_adaptive_mesh_refinement(
    initial_optimal_control_problem_definition: OptimalControlProblem,
    adaptive_mesh_refinement_parameters: AdaptiveParameters
) -> OptimalControlSolution:
    """
    Implements Algorithm 1 from mesh.txt for PHS-Adaptive mesh refinement.
    """
    desired_mesh_error_tolerance_epsilon = adaptive_mesh_refinement_parameters.epsilon_tol
    max_allowable_adaptive_refinement_iterations = adaptive_mesh_refinement_parameters.M_max_iterations
    min_allowable_collocation_nodes_per_interval_setting = adaptive_mesh_refinement_parameters.N_min_poly_degree
    max_allowable_collocation_nodes_per_interval_setting = adaptive_mesh_refinement_parameters.N_max_poly_degree
    ode_solver_relative_tolerance_for_error_simulations = adaptive_mesh_refinement_parameters.ode_solver_tol
    number_of_evaluation_points_for_error_simulation_per_interval = adaptive_mesh_refinement_parameters.num_error_sim_points

    current_iteration_list_of_collocation_nodes_per_interval: List[int] = list(initial_optimal_control_problem_definition.collocation_points_per_interval)
    current_iteration_global_normalized_mesh_nodes_array: np.ndarray
    
    if initial_optimal_control_problem_definition.global_normalized_mesh_nodes is not None:
        current_iteration_global_normalized_mesh_nodes_array = np.array(initial_optimal_control_problem_definition.global_normalized_mesh_nodes)
    else:
        current_iteration_global_normalized_mesh_nodes_array = np.linspace(-1, 1, len(current_iteration_list_of_collocation_nodes_per_interval) + 1)

    for i in range(len(current_iteration_list_of_collocation_nodes_per_interval)):
        current_iteration_list_of_collocation_nodes_per_interval[i] = max(min_allowable_collocation_nodes_per_interval_setting, min(max_allowable_collocation_nodes_per_interval_setting, current_iteration_list_of_collocation_nodes_per_interval[i]))

    # Create a copy of the problem definition for current iteration
    problem_definition_for_current_iteration_nlp = OptimalControlProblem(
        num_states=initial_optimal_control_problem_definition.num_states,
        num_controls=initial_optimal_control_problem_definition.num_controls,
        num_integrals=initial_optimal_control_problem_definition.num_integrals,
        dynamics_function=initial_optimal_control_problem_definition.dynamics_function,
        objective_function=initial_optimal_control_problem_definition.objective_function,
        t0_bounds=initial_optimal_control_problem_definition.t0_bounds,
        tf_bounds=initial_optimal_control_problem_definition.tf_bounds,
        integral_integrand_function=initial_optimal_control_problem_definition.integral_integrand_function,
        path_constraints_function=initial_optimal_control_problem_definition.path_constraints_function,
        event_constraints_function=initial_optimal_control_problem_definition.event_constraints_function,
        problem_parameters=initial_optimal_control_problem_definition.problem_parameters,
        default_initial_guess_values=initial_optimal_control_problem_definition.default_initial_guess_values,
        solver_options=initial_optimal_control_problem_definition.solver_options
    )
    
    num_states = problem_definition_for_current_iteration_nlp.num_states
    num_controls = problem_definition_for_current_iteration_nlp.num_controls

    most_recent_successful_optimal_control_solution_details: Optional[OptimalControlSolution] = None

    for current_adaptive_refinement_iteration_number in range(max_allowable_adaptive_refinement_iterations):
        print(f"\n--- Adaptive Iteration M = {current_adaptive_refinement_iteration_number} ---")
        number_of_mesh_intervals_in_current_iteration = len(current_iteration_list_of_collocation_nodes_per_interval)

        problem_definition_for_current_iteration_nlp.collocation_points_per_interval = list(current_iteration_list_of_collocation_nodes_per_interval)
        problem_definition_for_current_iteration_nlp.global_normalized_mesh_nodes = list(current_iteration_global_normalized_mesh_nodes_array) # Ensure it's a list for JSON later if problem_definition is saved

        if current_adaptive_refinement_iteration_number == 0 or not most_recent_successful_optimal_control_solution_details or not most_recent_successful_optimal_control_solution_details.success:
            print("  First iteration or previous NLP failed/unavailable. Using robust default initial guess.")
            initial_guess_data_for_current_nlp_solve = _generate_robust_default_initial_guess(
                initial_optimal_control_problem_definition, current_iteration_list_of_collocation_nodes_per_interval
            )
        else:
            initial_guess_data_for_current_nlp_solve = _propagate_guess_from_previous(
                most_recent_successful_optimal_control_solution_details, initial_optimal_control_problem_definition,
                current_iteration_list_of_collocation_nodes_per_interval, current_iteration_global_normalized_mesh_nodes_array
            )
        problem_definition_for_current_iteration_nlp.initial_guess = initial_guess_data_for_current_nlp_solve

        print(f"  Mesh K={number_of_mesh_intervals_in_current_iteration}, num_collocation_nodes_per_interval = {current_iteration_list_of_collocation_nodes_per_interval}")
        print(f"  Mesh nodes_tau_global = {np.array2string(current_iteration_global_normalized_mesh_nodes_array, precision=3)}")

        optimal_control_solution_from_current_nlp_solve: OptimalControlSolution = solve_single_phase_radau_collocation(problem_definition_for_current_iteration_nlp)

        if not optimal_control_solution_from_current_nlp_solve.success:
            error_or_status_message = f"NLP solver failed in adaptive iteration {current_adaptive_refinement_iteration_number}. " + (optimal_control_solution_from_current_nlp_solve.message or "Solver error.")
            print(f"  Error: {error_or_status_message} Stopping.")
            if most_recent_successful_optimal_control_solution_details: # Return the last good one, but mark as overall adaptive failure
                 most_recent_successful_optimal_control_solution_details.message = error_or_status_message
                 most_recent_successful_optimal_control_solution_details.success = False # Overall adaptive process failed
                 return most_recent_successful_optimal_control_solution_details
            else: # NLP failed on the very first attempt
                 optimal_control_solution_from_current_nlp_solve.message = error_or_status_message
                 return optimal_control_solution_from_current_nlp_solve # This already has success=False

        # Store solved values directly in the solution object for propagation and interpolation
        try:
            casadi_optimization_problem_object = optimal_control_solution_from_current_nlp_solve.opti_object
            raw_sol = optimal_control_solution_from_current_nlp_solve.raw_solution
            optimal_control_solution_from_current_nlp_solve.solved_state_trajectories_per_interval = [
                _extract_and_prepare_array(raw_sol.value(var), num_states, current_iteration_list_of_collocation_nodes_per_interval[i] + 1)
                for i, var in enumerate(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables)
            ]
            if num_controls > 0:
                optimal_control_solution_from_current_nlp_solve.solved_control_trajectories_per_interval = [
                    _extract_and_prepare_array(raw_sol.value(var), num_controls, current_iteration_list_of_collocation_nodes_per_interval[i])
                    for i, var in enumerate(casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables)
                ]
            else:
                optimal_control_solution_from_current_nlp_solve.solved_control_trajectories_per_interval = [np.empty((0, current_iteration_list_of_collocation_nodes_per_interval[i])) for i in range(number_of_mesh_intervals_in_current_iteration)]

        except Exception as e:
            error_or_status_message = f"Failed to extract solved trajectories from NLP solution at iter {current_adaptive_refinement_iteration_number}: {e}. Stopping."
            print(f"  Error: {error_or_status_message}")
            optimal_control_solution_from_current_nlp_solve.message = error_or_status_message
            optimal_control_solution_from_current_nlp_solve.success = False
            return optimal_control_solution_from_current_nlp_solve

        most_recent_successful_optimal_control_solution_details = optimal_control_solution_from_current_nlp_solve # This is now the latest successful one
        most_recent_successful_optimal_control_solution_details.num_collocation_nodes_list_at_solve_time = list(current_iteration_list_of_collocation_nodes_per_interval) # For initial guess propagation
        most_recent_successful_optimal_control_solution_details.global_mesh_nodes_at_solve_time = np.copy(current_iteration_global_normalized_mesh_nodes_array) # For initial guess propagation

        state_component_error_normalization_factors_gamma_for_iteration = _calculate_gamma_normalizers(optimal_control_solution_from_current_nlp_solve, problem_definition_for_current_iteration_nlp)
        if state_component_error_normalization_factors_gamma_for_iteration is None and num_states > 0: # num_states > 0 check because gamma_i is empty if num_states=0
            error_or_status_message = f"Failed to calculate gamma_i normalizers at iter {current_adaptive_refinement_iteration_number}. Stopping."
            print(f"  Error: {error_or_status_message}")
            optimal_control_solution_from_current_nlp_solve.message = error_or_status_message
            optimal_control_solution_from_current_nlp_solve.success = False # Mark as failure
            return optimal_control_solution_from_current_nlp_solve

        # Pre-compute basis components and polynomial interpolants for all intervals
        list_of_state_polynomial_evaluators_for_all_intervals: List[Optional[PolynomialInterpolant]] = [None] * number_of_mesh_intervals_in_current_iteration
        list_of_control_polynomial_evaluators_for_all_intervals: List[Optional[PolynomialInterpolant]] = [None] * number_of_mesh_intervals_in_current_iteration

        # Get solved state/control trajectories for the current iteration
        current_iteration_solved_state_trajectories_list = optimal_control_solution_from_current_nlp_solve.solved_state_trajectories_per_interval
        current_iteration_solved_control_trajectories_list = optimal_control_solution_from_current_nlp_solve.solved_control_trajectories_per_interval

        # Caches to avoid redundant computation of basis components and barycentric weights
        # These depend only on Nk, not on interval-specific data
        basis_components_cache = {}
        control_barycentric_weights_cache = {}

        # Create polynomial interpolants once per interval
        for k_idx_interp in range(number_of_mesh_intervals_in_current_iteration):
            try:
                # Compute LGR basis components ONCE per unique Nk value
                Nk_interp = current_iteration_list_of_collocation_nodes_per_interval[k_idx_interp]

                # Use cache for basis components
                if Nk_interp not in basis_components_cache:
                    # Only compute if we haven't seen this Nk before
                    basis_components_cache[Nk_interp] = compute_radau_collocation_components(Nk_interp)

                # Get the basis components from cache
                basis_comps = basis_components_cache[Nk_interp]

                # Get state data for the interval
                state_data_for_interpolant = current_iteration_solved_state_trajectories_list[k_idx_interp]

                # Create state polynomial interpolant ONCE using pre-computed barycentric weights
                list_of_state_polynomial_evaluators_for_all_intervals[k_idx_interp] = get_polynomial_interpolant(
                    basis_comps.state_approximation_nodes, 
                    state_data_for_interpolant,
                    basis_comps.barycentric_weights_for_state_nodes
                )

                # Create control polynomial interpolant or empty function if no controls
                if num_controls > 0:
                    control_data_for_interpolant = current_iteration_solved_control_trajectories_list[k_idx_interp]

                    # Use cache for control barycentric weights
                    if Nk_interp not in control_barycentric_weights_cache:
                        # Only compute if we haven't seen this Nk before
                        control_barycentric_weights_cache[Nk_interp] = _compute_barycentric_weights(basis_comps.collocation_nodes)

                    # Get the control barycentric weights from cache
                    control_bary_weights = control_barycentric_weights_cache[Nk_interp]

                    # Create control interpolant ONCE
                    list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = get_polynomial_interpolant(
                        basis_comps.collocation_nodes,
                        control_data_for_interpolant,
                        control_bary_weights
                    )
                else:
                    # No controls - create empty function
                    list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = get_polynomial_interpolant(
                        np.array([-1.0, 1.0]),
                        np.empty((0, 2)),
                        None
                    )

            except Exception as e:
                print(f"  Warning: Error creating interpolant for interval {k_idx_interp}: {e}. Error estimation might be affected.")
                # Create fallback interpolants that return NaN values
                if list_of_state_polynomial_evaluators_for_all_intervals[k_idx_interp] is None:
                    list_of_state_polynomial_evaluators_for_all_intervals[k_idx_interp] = get_polynomial_interpolant(
                        np.array([-1.0, 1.0]),
                        np.full((num_states, 2), np.nan),
                        None
                    )
                if list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] is None:
                    list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = get_polynomial_interpolant(
                        np.array([-1.0, 1.0]),
                        np.full((num_controls if num_controls > 0 else 0, 2), np.nan),
                        None
                    )

        list_of_max_relative_errors_for_each_interval_this_iteration: List[float] = [np.inf] * number_of_mesh_intervals_in_current_iteration

        # Use pre-computed interpolants for error estimation
        for mesh_interval_index in range(number_of_mesh_intervals_in_current_iteration):
            print(f"  Starting error simulation for interval {mesh_interval_index}...")

            # Get pre-computed interpolants for this interval
            current_state_polynomial_evaluator = list_of_state_polynomial_evaluators_for_all_intervals[mesh_interval_index]
            current_control_polynomial_evaluator = list_of_control_polynomial_evaluators_for_all_intervals[mesh_interval_index]

            if current_state_polynomial_evaluator is None or current_control_polynomial_evaluator is None:
                print(f"    Warning: Missing interpolants for interval {mesh_interval_index}. Assigning high error.")
                list_of_max_relative_errors_for_each_interval_this_iteration[mesh_interval_index] = np.inf
                continue

            # Pass pre-computed interpolants to _simulate_dynamics_for_error_estimation
            error_estimation_simulation_bundle_for_current_interval = _simulate_dynamics_for_error_estimation(
                mesh_interval_index, 
                optimal_control_solution_from_current_nlp_solve, 
                problem_definition_for_current_iteration_nlp,
                current_state_polynomial_evaluator,  # Pass pre-computed state interpolant
                current_control_polynomial_evaluator,  # Pass pre-computed control interpolant
                ode_solver_relative_tolerance=ode_solver_relative_tolerance_for_error_simulations, 
                number_of_ode_simulation_evaluation_points=number_of_evaluation_points_for_error_simulation_per_interval
            )
            # state_component_error_normalization_factors_gamma_for_iteration can be empty if num_states=0
            estimated_max_relative_error_for_current_processed_interval = calculate_relative_error_estimate(
                mesh_interval_index, 
                error_estimation_simulation_bundle_for_current_interval, 
                state_component_error_normalization_factors_gamma_for_iteration if num_states > 0 else np.array([])
            )
            list_of_max_relative_errors_for_each_interval_this_iteration[mesh_interval_index] = estimated_max_relative_error_for_current_processed_interval
            print(f"    Interval {mesh_interval_index}: Nk={current_iteration_list_of_collocation_nodes_per_interval[mesh_interval_index]}, Error={estimated_max_relative_error_for_current_processed_interval:.4e}")

        print(f"  Overall errors list_of_max_relative_errors_for_each_interval_this_iteration: {[f'{e:.2e}' for e in list_of_max_relative_errors_for_each_interval_this_iteration]}")

        are_all_interval_errors_within_tolerance_and_valid = True
        if number_of_mesh_intervals_in_current_iteration == 0: # No intervals, considered converged
            are_all_interval_errors_within_tolerance_and_valid = True
        elif not list_of_max_relative_errors_for_each_interval_this_iteration : # No errors calculated but intervals exist
            are_all_interval_errors_within_tolerance_and_valid = False
        else:
            for err_val_conv in list_of_max_relative_errors_for_each_interval_this_iteration:
                if np.isnan(err_val_conv) or np.isinf(err_val_conv) or err_val_conv > desired_mesh_error_tolerance_epsilon:
                    are_all_interval_errors_within_tolerance_and_valid = False
                    break
                   
        if are_all_interval_errors_within_tolerance_and_valid:
            print(f"Mesh converged after {current_adaptive_refinement_iteration_number+1} iterations.")
            optimal_control_solution_from_current_nlp_solve.num_collocation_nodes_per_interval = current_iteration_list_of_collocation_nodes_per_interval.copy() # Final mesh config
            optimal_control_solution_from_current_nlp_solve.global_normalized_mesh_nodes = np.copy(current_iteration_global_normalized_mesh_nodes_array)
            optimal_control_solution_from_current_nlp_solve.message = f"Adaptive mesh converged to tolerance {desired_mesh_error_tolerance_epsilon:.1e} in {current_adaptive_refinement_iteration_number+1} iterations."
            # optimal_control_solution_from_current_nlp_solve.success is already True from the solver
            return optimal_control_solution_from_current_nlp_solve

        # Refine mesh M
        list_of_collocation_nodes_for_next_iteration_being_built = []
        global_normalized_mesh_nodes_for_next_iteration_being_built = [current_iteration_global_normalized_mesh_nodes_array[0]] 

        current_original_mesh_interval_index_being_processed = 0 
        while current_original_mesh_interval_index_being_processed < number_of_mesh_intervals_in_current_iteration:
            max_relative_error_in_current_original_interval_being_processed = list_of_max_relative_errors_for_each_interval_this_iteration[current_original_mesh_interval_index_being_processed]
            num_collocation_nodes_in_current_original_interval_being_processed = current_iteration_list_of_collocation_nodes_per_interval[current_original_mesh_interval_index_being_processed]
            print(f"    Processing original interval {current_original_mesh_interval_index_being_processed}: Nk={num_collocation_nodes_in_current_original_interval_being_processed}, Error={max_relative_error_in_current_original_interval_being_processed:.2e}")

            if np.isnan(max_relative_error_in_current_original_interval_being_processed) or \
               np.isinf(max_relative_error_in_current_original_interval_being_processed) or \
               max_relative_error_in_current_original_interval_being_processed > desired_mesh_error_tolerance_epsilon:

                print(f"      Interval {current_original_mesh_interval_index_being_processed} error > tol (or invalid). Attempting p-refinement.")

                # Call p_refine_interval
                p_refine_result = p_refine_interval(
                    num_collocation_nodes_in_current_original_interval_being_processed,
                    max_relative_error_in_current_original_interval_being_processed,
                    desired_mesh_error_tolerance_epsilon,
                    max_allowable_collocation_nodes_per_interval_setting
                )

                if p_refine_result.was_p_successful:
                    print(f"        p-refinement applied: Nk {num_collocation_nodes_in_current_original_interval_being_processed} -> {p_refine_result.actual_Nk_to_use}")
                    list_of_collocation_nodes_for_next_iteration_being_built.append(p_refine_result.actual_Nk_to_use)
                    global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1])
                    current_original_mesh_interval_index_being_processed += 1
                else: 
                    # p-refinement "failed" because unconstrained_target_Nk_for_h > N_max (or no increase was possible)
                    # The actual_Nk_from_p_attempt would be N_max in the case of exceeding N_max.
                    # unconstrained_target_Nk_for_h holds the value from Eq. 27 needed for h-refinement.
                    print(f"        p-refinement failed (target Nk {p_refine_result.unconstrained_target_Nk} would exceed N_max {max_allowable_collocation_nodes_per_interval_setting}, or no increase possible). Attempting h-refinement.")

                    # Directly use unconstrained_target_Nk_for_h for h-refinement parameter calculation
                    h_refine_result = h_refine_params(
                        p_refine_result.unconstrained_target_Nk, # Use the unconstrained target from p_refine_interval
                        min_allowable_collocation_nodes_per_interval_setting
                    )
                    print(f"          h-refinement: Splitting interval {current_original_mesh_interval_index_being_processed} into {h_refine_result.num_new_subintervals} subintervals, each Nk={h_refine_result.collocation_nodes_for_new_subintervals[0]}.")
                    list_of_collocation_nodes_for_next_iteration_being_built.extend(h_refine_result.collocation_nodes_for_new_subintervals)

                    global_normalized_start_tau_of_original_interval_being_split = current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed]
                    global_normalized_end_tau_of_original_interval_being_split = current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1]
                    newly_created_global_normalized_mesh_nodes_for_subintervals = np.linspace(
                        global_normalized_start_tau_of_original_interval_being_split, 
                        global_normalized_end_tau_of_original_interval_being_split, 
                        h_refine_result.num_new_subintervals + 1
                    )
                    global_normalized_mesh_nodes_for_next_iteration_being_built.extend(list(newly_created_global_normalized_mesh_nodes_for_subintervals[1:]))
                    current_original_mesh_interval_index_being_processed += 1
            else: # e_max(k) <= epsilon
                print(f"    Interval {current_original_mesh_interval_index_being_processed} error <= tol.")
                is_current_original_interval_pair_eligible_for_h_reduction_check = False
                if current_original_mesh_interval_index_being_processed < number_of_mesh_intervals_in_current_iteration - 1:
                    max_relative_error_in_next_adjacent_original_interval = list_of_max_relative_errors_for_each_interval_this_iteration[current_original_mesh_interval_index_being_processed + 1]
                    if not (np.isnan(max_relative_error_in_next_adjacent_original_interval) or np.isinf(max_relative_error_in_next_adjacent_original_interval)) and max_relative_error_in_next_adjacent_original_interval <= desired_mesh_error_tolerance_epsilon:
                        is_current_original_interval_pair_eligible_for_h_reduction_check = True
                        print(f"      Interval {current_original_mesh_interval_index_being_processed+1} also has low error ({max_relative_error_in_next_adjacent_original_interval:.2e}). Eligible for h-reduction.")

                if is_current_original_interval_pair_eligible_for_h_reduction_check:
                    are_polynomial_evaluators_available_for_h_reduction_check = (
                        list_of_state_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed] is not None and
                        list_of_state_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed+1] is not None and
                        (num_controls == 0 or (list_of_control_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed] is not None and list_of_control_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed+1] is not None))
                    )
                    was_h_reduction_merge_successful_for_current_original_interval_pair = False
                    if are_polynomial_evaluators_available_for_h_reduction_check:
                        # Pass problem_definition_for_current_iteration_nlp because it contains the correct num_collocation_nodes_per_interval for this iteration
                        was_h_reduction_merge_successful_for_current_original_interval_pair = h_reduce_intervals(
                            current_original_mesh_interval_index_being_processed, 
                            optimal_control_solution_from_current_nlp_solve, 
                            problem_definition_for_current_iteration_nlp, 
                            adaptive_mesh_refinement_parameters,
                            state_component_error_normalization_factors_gamma_for_iteration if num_states > 0 else np.array([]), # Pass calculated gamma
                            list_of_state_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed], 
                            list_of_control_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed],
                            list_of_state_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed+1], 
                            list_of_control_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed+1]
                        )
                    else:
                        print("      Skipping h-reduction attempt due to missing interpolants (should be rare).")

                    if was_h_reduction_merge_successful_for_current_original_interval_pair:
                        print(f"      h-reduction applied to merge interval {current_original_mesh_interval_index_being_processed} and {current_original_mesh_interval_index_being_processed+1}.")
                        # Use the maximum Nk from the two intervals being merged, with bounds
                        num_collocation_nodes_for_newly_merged_interval_from_h_reduction = max(
                            current_iteration_list_of_collocation_nodes_per_interval[current_original_mesh_interval_index_being_processed], 
                            current_iteration_list_of_collocation_nodes_per_interval[current_original_mesh_interval_index_being_processed+1]
                        )
                        num_collocation_nodes_for_newly_merged_interval_from_h_reduction = max(
                            min_allowable_collocation_nodes_per_interval_setting, 
                            min(max_allowable_collocation_nodes_per_interval_setting, num_collocation_nodes_for_newly_merged_interval_from_h_reduction)
                        )
                        list_of_collocation_nodes_for_next_iteration_being_built.append(num_collocation_nodes_for_newly_merged_interval_from_h_reduction)
                        global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 2])
                        current_original_mesh_interval_index_being_processed += 2 
                    else: 
                        print(f"      h-reduction failed or condition not met. Attempting p-reduction for interval {current_original_mesh_interval_index_being_processed}.")
                        p_reduce_result = p_reduce_interval(
                            num_collocation_nodes_in_current_original_interval_being_processed, 
                            max_relative_error_in_current_original_interval_being_processed, 
                            desired_mesh_error_tolerance_epsilon, 
                            min_allowable_collocation_nodes_per_interval_setting, 
                            max_allowable_collocation_nodes_per_interval_setting
                        )
                        if p_reduce_result.was_reduction_applied: 
                            print(f"        p-reduction applied: Nk {num_collocation_nodes_in_current_original_interval_being_processed} -> {p_reduce_result.new_num_collocation_nodes}")
                        else: 
                            print(f"        p-reduction not applied for Nk {num_collocation_nodes_in_current_original_interval_being_processed}.")
                        list_of_collocation_nodes_for_next_iteration_being_built.append(p_reduce_result.new_num_collocation_nodes)
                        global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1])
                        current_original_mesh_interval_index_being_processed += 1
                else: 
                    print(f"      Not eligible for h-reduction with next interval. Attempting p-reduction for interval {current_original_mesh_interval_index_being_processed}.")
                    p_reduce_result = p_reduce_interval(
                        num_collocation_nodes_in_current_original_interval_being_processed, 
                        max_relative_error_in_current_original_interval_being_processed, 
                        desired_mesh_error_tolerance_epsilon, 
                        min_allowable_collocation_nodes_per_interval_setting, 
                        max_allowable_collocation_nodes_per_interval_setting
                    )
                    if p_reduce_result.was_reduction_applied: 
                        print(f"        p-reduction applied: Nk {num_collocation_nodes_in_current_original_interval_being_processed} -> {p_reduce_result.new_num_collocation_nodes}")
                    else: 
                        print(f"        p-reduction not applied for Nk {num_collocation_nodes_in_current_original_interval_being_processed}.")
                    list_of_collocation_nodes_for_next_iteration_being_built.append(p_reduce_result.new_num_collocation_nodes)
                    global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1])
                    current_original_mesh_interval_index_being_processed += 1

        current_iteration_list_of_collocation_nodes_per_interval = list_of_collocation_nodes_for_next_iteration_being_built
        current_iteration_global_normalized_mesh_nodes_array = np.array(global_normalized_mesh_nodes_for_next_iteration_being_built) # Ensure it's an array for next iter

        # --- Mesh Sanity Checks ---
        solution_object_for_early_return_due_to_mesh_inconsistency = most_recent_successful_optimal_control_solution_details if most_recent_successful_optimal_control_solution_details else OptimalControlSolution()

        if solution_object_for_early_return_due_to_mesh_inconsistency is None:
            solution_object_for_early_return_due_to_mesh_inconsistency = OptimalControlSolution()

        solution_object_for_early_return_due_to_mesh_inconsistency.num_collocation_nodes_per_interval = current_iteration_list_of_collocation_nodes_per_interval
        solution_object_for_early_return_due_to_mesh_inconsistency.global_normalized_mesh_nodes = current_iteration_global_normalized_mesh_nodes_array

        if not current_iteration_list_of_collocation_nodes_per_interval and len(current_iteration_global_normalized_mesh_nodes_array) > 1 :
            error_or_status_message = "Stopped due to mesh inconsistency (empty num_collocation_nodes_per_interval but mesh_nodes exist)."
            print(f"  Error: {error_or_status_message} Stopping.")
            solution_object_for_early_return_due_to_mesh_inconsistency.message = error_or_status_message
            solution_object_for_early_return_due_to_mesh_inconsistency.success = False
            return solution_object_for_early_return_due_to_mesh_inconsistency

        if current_iteration_list_of_collocation_nodes_per_interval and len(current_iteration_list_of_collocation_nodes_per_interval) != (len(current_iteration_global_normalized_mesh_nodes_array) -1):
            error_or_status_message = f"Mesh structure inconsistent after refinement. num_collocation_nodes_per_interval len: {len(current_iteration_list_of_collocation_nodes_per_interval)}, Nodes len-1: {len(current_iteration_global_normalized_mesh_nodes_array)-1}."
            print(f"  Error: {error_or_status_message} Stopping.")
            solution_object_for_early_return_due_to_mesh_inconsistency.message = error_or_status_message
            solution_object_for_early_return_due_to_mesh_inconsistency.success = False
            return solution_object_for_early_return_due_to_mesh_inconsistency

        if len(current_iteration_list_of_collocation_nodes_per_interval) > 0 :
             unique_global_mesh_nodes_after_refinement, counts_of_unique_global_mesh_nodes_after_refinement = np.unique(np.round(current_iteration_global_normalized_mesh_nodes_array, decimals=12), return_counts=True)
             if np.any(counts_of_unique_global_mesh_nodes_after_refinement > 1):
                  detected_duplicate_global_mesh_nodes_after_refinement = unique_global_mesh_nodes_after_refinement[counts_of_unique_global_mesh_nodes_after_refinement > 1]
                  error_or_status_message = f"Duplicate mesh nodes found (after rounding): {detected_duplicate_global_mesh_nodes_after_refinement}. Original nodes: {current_iteration_global_normalized_mesh_nodes_array}."
                  print(f"  Error: {error_or_status_message} Stopping.")
                  solution_object_for_early_return_due_to_mesh_inconsistency.message = error_or_status_message
                  solution_object_for_early_return_due_to_mesh_inconsistency.success = False
                  return solution_object_for_early_return_due_to_mesh_inconsistency
             if len(unique_global_mesh_nodes_after_refinement) > 1 and not np.all(np.diff(unique_global_mesh_nodes_after_refinement) > 1e-9): 
                  # sorted_global_mesh_nodes_for_consistency_check = np.sort(current_iteration_global_normalized_mesh_nodes_array) # unique_global_mesh_nodes_after_refinement is already sorted
                  differences_between_sorted_unique_global_mesh_nodes_after_refinement = np.diff(unique_global_mesh_nodes_after_refinement) # Check differences_between_sorted_unique_global_mesh_nodes_after_refinement of unique, sorted nodes
                  indices_of_problematic_small_or_non_positive_node_differences = np.where(differences_between_sorted_unique_global_mesh_nodes_after_refinement <= 1e-9)[0]
                  string_representation_of_problematic_global_mesh_node_pairs = ", ".join([f"({unique_global_mesh_nodes_after_refinement[i]:.3f}, {unique_global_mesh_nodes_after_refinement[i+1]:.3f})" for i in indices_of_problematic_small_or_non_positive_node_differences]) if indices_of_problematic_small_or_non_positive_node_differences.size > 0 else "N/A"
                  error_or_status_message = f"Mesh nodes not strictly increasing or interval too small. Problem pairs: {string_representation_of_problematic_global_mesh_node_pairs}. All nodes: {current_iteration_global_normalized_mesh_nodes_array}."
                  print(f"  Error: {error_or_status_message} Stopping.")
                  solution_object_for_early_return_due_to_mesh_inconsistency.message = error_or_status_message
                  solution_object_for_early_return_due_to_mesh_inconsistency.success = False
                  return solution_object_for_early_return_due_to_mesh_inconsistency

    # --- Max Iterations Reached ---
    message_indicating_max_iterations_reached_without_full_convergence = f"Adaptive mesh refinement reached max iterations ({max_allowable_adaptive_refinement_iterations}) without full convergence to tolerance {desired_mesh_error_tolerance_epsilon:.1e}."
    print(message_indicating_max_iterations_reached_without_full_convergence)
    if most_recent_successful_optimal_control_solution_details: # This means at least one NLP solve was successful
        most_recent_successful_optimal_control_solution_details.message = message_indicating_max_iterations_reached_without_full_convergence
        # most_recent_successful_optimal_control_solution_details.success remains True (NLP solved) but overall adaptive process did not converge to tolerance.
        # The calling function should check the message if it needs to distinguish.
        most_recent_successful_optimal_control_solution_details.num_collocation_nodes_per_interval = current_iteration_list_of_collocation_nodes_per_interval.copy()
        most_recent_successful_optimal_control_solution_details.global_normalized_mesh_nodes = np.copy(current_iteration_global_normalized_mesh_nodes_array)
        return most_recent_successful_optimal_control_solution_details
    else: # No NLP was ever successful
        failed_solution = OptimalControlSolution()
        failed_solution.success = False
        failed_solution.message = message_indicating_max_iterations_reached_without_full_convergence + " No successful NLP solution obtained throughout iterations."
        failed_solution.num_collocation_nodes_per_interval = current_iteration_list_of_collocation_nodes_per_interval
        failed_solution.global_normalized_mesh_nodes = current_iteration_global_normalized_mesh_nodes_array.tolist() if isinstance(current_iteration_global_normalized_mesh_nodes_array, np.ndarray) else current_iteration_global_normalized_mesh_nodes_array
        return failed_solution