"""
Error estimation for PHS adaptive mesh refinement.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Union, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp

from ...core.problem import ProblemDefinition, Solution
from ...core.basis import _compute_barycentric_weights, _evaluate_lagrange_polynomial_at_point

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
                    setattr(self, field_name, current_val.reshape(1, -1) if current_val.shape[0] != 0 else np.empty((0,0)))
                elif current_val.ndim > 2:
                    raise ValueError(
                        f"Field {field_name} must be 1D or 2D array, "
                        f"got {current_val.ndim}D."
                    )

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

def get_polynomial_interpolant(
    interpolation_definition_nodes_local_tau: np.ndarray,
    known_values_at_interpolation_definition_nodes: np.ndarray, # Expected shape (num_vars, num_nodes)
    barycentric_weights_for_definition_nodes: Optional[np.ndarray] = None
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Creates a Lagrange polynomial interpolant using barycentric formula.
    Args:
        interpolation_definition_nodes_local_tau: The points at which function values are known.
        known_values_at_interpolation_definition_nodes: The function values at the 'nodes'. Shape (num_vars, num_nodes).
        barycentric_weights_for_definition_nodes: Pre-computed barycentric weights for 'nodes'.
    Returns:
        A callable function that evaluates the interpolant.
    """
    if not isinstance(known_values_at_interpolation_definition_nodes, np.ndarray):
        values_at_nodes_arr = np.array(known_values_at_interpolation_definition_nodes)
    else:
        values_at_nodes_arr = known_values_at_interpolation_definition_nodes

    if values_at_nodes_arr.ndim == 1: # Ensure known_values_at_interpolation_definition_nodes is 2D (num_vars, num_nodes)
        values_at_nodes_arr = values_at_nodes_arr.reshape(1, -1)
    num_vars, num_nodes_val = values_at_nodes_arr.shape

    if not isinstance(interpolation_definition_nodes_local_tau, np.ndarray):
        nodes_array = np.array(interpolation_definition_nodes_local_tau)
    else:
        nodes_array = interpolation_definition_nodes_local_tau
    num_nodes_pts = len(nodes_array)

    if num_nodes_pts != num_nodes_val:
        raise ValueError(f"Mismatch in number of nodes ({num_nodes_pts}) and number of values columns ({num_nodes_val})")

    if barycentric_weights_for_definition_nodes is None:
        bary_weights_internal = _compute_barycentric_weights(nodes_array)
    else:
        bary_weights_internal = np.array(barycentric_weights_for_definition_nodes) if not isinstance(barycentric_weights_for_definition_nodes, np.ndarray) else barycentric_weights_for_definition_nodes
        if len(bary_weights_internal) != num_nodes_pts:
            raise ValueError("Provided barycentric_weights_for_definition_nodes length does not match nodes length.")

    def evaluate_lagrange_polynomial_at_local_tau(local_tau_evaluation_point_or_points: Union[float, np.ndarray]) -> np.ndarray:
        is_scalar_input = np.isscalar(local_tau_evaluation_point_or_points)
        zeta_eval_arr = np.atleast_1d(local_tau_evaluation_point_or_points)
        # values_at_nodes_arr is shape (num_vars, num_nodes_val)
        # L_j_at_zeta_pt (from _evaluate_lagrange_polynomial_at_point) is shape (num_nodes_val,)
        # np.dot(values_at_nodes_arr, L_j_at_zeta_pt) results in shape (num_vars,)
        # interpolated_values_at_evaluation_points will be shape (num_vars, len(zeta_eval_arr))
        interpolated_values_at_evaluation_points = np.zeros((num_vars, len(zeta_eval_arr)))
        for i, zeta_pt in enumerate(zeta_eval_arr):
            L_j_at_zeta_pt = _evaluate_lagrange_polynomial_at_point(nodes_array, bary_weights_internal, zeta_pt)
            interpolated_values_at_evaluation_points[:, i] = np.dot(values_at_nodes_arr, L_j_at_zeta_pt)
        
        if is_scalar_input:
            # If input was scalar, zeta_eval_arr has len 1.
            # interpolated_values_at_evaluation_points is shape (num_vars, 1).
            # We want to return a 1D array of shape (num_vars,).
            return interpolated_values_at_evaluation_points[:, 0]
        else:
            # If input was an array, interpolated_values_at_evaluation_points is (num_vars, len(zeta_eval_arr)).
            # This is the desired shape (features, samples).
            return interpolated_values_at_evaluation_points
    return evaluate_lagrange_polynomial_at_local_tau

def _simulate_dynamics_for_error_estimation(
    current_mesh_interval_index: int, 
    current_optimal_control_solution: Solution, 
    optimal_control_problem_definition: ProblemDefinition,
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

    if not current_optimal_control_solution.get("success", False) or current_optimal_control_solution.get("raw_solution") is None:
        print(f"    Warning: NLP solution unsuccessful or raw solution missing for interval {current_mesh_interval_index} in error simulation.")
        return interval_simulation_bundle

    num_states = optimal_control_problem_definition['num_states']; num_controls = optimal_control_problem_definition['num_controls']
    dynamics_function = optimal_control_problem_definition['dynamics_function']; problem_parameters = optimal_control_problem_definition.get('problem_parameters', {})
    
    # Time transformation parameters from the overall solution
    phase_initial_time_from_nlp_solution = current_optimal_control_solution['initial_time_variable']
    phase_terminal_time_from_nlp_solution = current_optimal_control_solution['terminal_time_variable']
    time_transformation_scaling_factor_alpha = (phase_terminal_time_from_nlp_solution - phase_initial_time_from_nlp_solution) / 2.0
    time_transformation_offset_alpha_0 = (phase_terminal_time_from_nlp_solution + phase_initial_time_from_nlp_solution) / 2.0
    
    global_normalized_mesh_nodes_from_nlp_solution = current_optimal_control_solution['global_normalized_mesh_nodes']
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

def _calculate_gamma_normalizers(optimal_control_solution_for_gamma_calculation: Solution, optimal_control_problem_definition_for_gamma_calculation: ProblemDefinition) -> Optional[np.ndarray]:
    """Calculates gamma_i normalization factors (Eq. 25) for error estimation."""
    if not optimal_control_solution_for_gamma_calculation.get("success") or optimal_control_solution_for_gamma_calculation.get("raw_solution") is None:
        return None
        
    num_states = optimal_control_problem_definition_for_gamma_calculation['num_states']
    if num_states == 0: return np.array([]).reshape(0,1) # No states, no gamma

    Y_solved_list = optimal_control_solution_for_gamma_calculation.get('solved_state_trajectories_per_interval')
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