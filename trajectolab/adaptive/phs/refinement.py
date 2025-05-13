"""
Refinement strategies for PHS adaptive mesh.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Union, Optional
from scipy.integrate import solve_ivp

from ...core.problem import ProblemDefinition, Solution
from ...core.basis import compute_radau_collocation_components

def p_refine_interval(
    current_num_collocation_nodes_in_interval: int, 
    max_relative_error_in_interval_for_refinement: float, 
    error_tolerance_threshold_for_refinement: float, 
    max_allowable_collocation_nodes_per_interval_limit: int
) -> Tuple[int, bool, int]:  # Returns: (actual_Nk_to_use, was_p_successful, unconstrained_target_Nk)
    """
    Determines new polynomial degree for an interval using p-refinement.
    Follows Eq. 27 from mesh.txt.
    
    Returns:
        actual_Nk_to_use (int): The number of collocation points to use for the next iteration.
                                This will be capped by max_allowable_collocation_nodes_per_interval_limit.
        was_p_successful (bool): True if p-refinement resulted in a valid new degree within N_max.
                                 False if the unconstrained target degree exceeded N_max.
        unconstrained_target_Nk (int): The target number of collocation points calculated from
                                       Eq. 27, *before* applying the N_max constraint. This is
                                       needed for h-refinement if p-refinement fails.
    """
    if max_relative_error_in_interval_for_refinement <= error_tolerance_threshold_for_refinement:
        # Error is already within tolerance, no p-refinement needed from this function's perspective.
        # Return current Nk, indicate "not successful" for this specific call's purpose of *increasing* degree.
        # The unconstrained target would also be the current Nk or less.
        return current_num_collocation_nodes_in_interval, False, current_num_collocation_nodes_in_interval

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
        return max_allowable_collocation_nodes_per_interval_limit, False, unconstrained_target_Nk
    
    # If unconstrained_target_Nk is <= current_num_collocation_nodes_in_interval,
    # it means P_k^+ was effectively zero (e.g. error very slightly above tolerance, log10(ratio) < 0 after ceil).
    # Since we forced number_of_collocation_nodes_to_add = max(1, ...), unconstrained_target_Nk will be > current_num_collocation_nodes_in_interval.
    # So, p-refinement is successful and within N_max.
    return unconstrained_target_Nk, True, unconstrained_target_Nk

def h_refine_params(
    target_num_collocation_nodes_that_triggered_h_refinement: int,
    min_allowable_collocation_nodes_for_new_subintervals: int
) -> Tuple[List[int], int]:
    """
    Determines parameters for h-refinement (splitting an interval).
    Follows Eq. 28 from mesh.txt.
    """
    number_of_new_subintervals_to_create_from_split = max(2, int(np.ceil(target_num_collocation_nodes_that_triggered_h_refinement / min_allowable_collocation_nodes_for_new_subintervals)))
    list_of_collocation_nodes_for_each_new_subinterval = [min_allowable_collocation_nodes_for_new_subintervals] * number_of_new_subintervals_to_create_from_split
    return list_of_collocation_nodes_for_each_new_subinterval, number_of_new_subintervals_to_create_from_split

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

def h_reduce_intervals(
    first_mesh_interval_index_for_merge_consideration: int, nlp_solution: Solution, problem_definition: ProblemDefinition,
    adaptive_refinement_parameters_for_h_reduction: Dict[str, Any], state_component_error_normalization_factors_gamma: np.ndarray, # Pre-calculated
    state_polynomial_evaluator_for_first_interval: Callable, control_polynomial_evaluator_for_first_interval: Optional[Callable],
    state_polynomial_evaluator_for_second_interval: Callable, control_polynomial_evaluator_for_second_interval: Optional[Callable]
) -> bool:
    """
    Checks if two adjacent intervals first_mesh_interval_index_for_merge_consideration and first_mesh_interval_index_for_merge_consideration+1 can be merged.
    Follows logic of Section 4.3.3 in mesh.txt (Eq. 31-34).
    Returns True if merge is successful (error condition met), False otherwise.
    """
    print(f"    h-reduction check for intervals {first_mesh_interval_index_for_merge_consideration} and {first_mesh_interval_index_for_merge_consideration+1}.")
    error_tolerance_for_allowing_merge = adaptive_refinement_parameters_for_h_reduction['epsilon_tol']
    ode_solver_tol = adaptive_refinement_parameters_for_h_reduction.get('ode_solver_tol', 1e-7)
    ode_solver_absolute_tolerance_for_merge_simulation = ode_solver_tol * 1e-1 
    num_error_sim_points = adaptive_refinement_parameters_for_h_reduction.get('num_error_sim_points', 50)

    num_states = problem_definition['num_states']; num_controls = problem_definition['num_controls'] # num_controls not directly used here, but for context
    dynamics_function = problem_definition['dynamics_function']; problem_parameters = problem_definition.get('problem_parameters', {})
    
    if nlp_solution.get("raw_solution") is None: # Should be caught by nlp_solution["success"] earlier
        print("      h-reduction failed: Raw solution missing.")
        return False

    global_normalized_mesh_nodes = nlp_solution['global_normalized_mesh_nodes'] # Assumed np.array
    global_normalized_start_tau_first_interval = global_normalized_mesh_nodes[first_mesh_interval_index_for_merge_consideration]
    global_normalized_shared_tau_node_for_merge = global_normalized_mesh_nodes[first_mesh_interval_index_for_merge_consideration + 1]
    global_normalized_end_tau_second_interval = global_normalized_mesh_nodes[first_mesh_interval_index_for_merge_consideration + 2]

    original_interval_scaling_factor_beta_first_interval = (global_normalized_shared_tau_node_for_merge - global_normalized_start_tau_first_interval) / 2.0
    original_interval_scaling_factor_beta_second_interval = (global_normalized_end_tau_second_interval - global_normalized_shared_tau_node_for_merge) / 2.0

    if abs(original_interval_scaling_factor_beta_first_interval) < 1e-12 or abs(original_interval_scaling_factor_beta_second_interval) < 1e-12:
        print("      h-reduction check: One of the intervals has zero length. Merge not possible.")
        return False

    solved_initial_time = nlp_solution['initial_time_variable']; solved_terminal_time = nlp_solution['terminal_time_variable']
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
    
    Y_solved_list = nlp_solution.get('solved_state_trajectories_per_interval')
    Nk_k = problem_definition['collocation_points_per_interval'][first_mesh_interval_index_for_merge_consideration]
    Nk_kp1 = problem_definition['collocation_points_per_interval'][first_mesh_interval_index_for_merge_consideration+1]

    try:
        if Y_solved_list and first_mesh_interval_index_for_merge_consideration < len(Y_solved_list):
            Xk_nlp_discrete = Y_solved_list[first_mesh_interval_index_for_merge_consideration]
        else: # Fallback
            from .error import _extract_and_prepare_array
            casadi_optimization_problem_object = nlp_solution['opti_object']; raw_sol = nlp_solution['raw_solution']
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
            from .error import _extract_and_prepare_array
            casadi_optimization_problem_object = nlp_solution['opti_object']; raw_sol = nlp_solution['raw_solution']
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

def p_reduce_interval(
    current_num_collocation_nodes_in_interval: int, max_relative_error_in_interval_for_reduction: float, error_tolerance_threshold_for_reduction: float,
    min_allowable_collocation_nodes_per_interval_after_reduction: int, max_allowable_collocation_nodes_per_interval_for_delta_calc: int
) -> Tuple[int, bool]:
    """
    Determines new polynomial degree (new_num_collocation_nodes_after_p_reduction) for an interval using p-reduction.
    Follows Eq. 36 from mesh.txt.
    Returns new Nk and a boolean indicating if reduction was applied.
    """
    if max_relative_error_in_interval_for_reduction > error_tolerance_threshold_for_reduction: return current_num_collocation_nodes_in_interval, False
    if current_num_collocation_nodes_in_interval <= min_allowable_collocation_nodes_per_interval_after_reduction: return current_num_collocation_nodes_in_interval, False

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
    return new_num_collocation_nodes_after_p_reduction, was_collocation_nodes_p_reduction_applied

def _generate_robust_default_initial_guess(
    base_optimal_control_problem_definition_for_defaults: ProblemDefinition,
    current_list_of_collocation_nodes_per_interval_for_guess: List[int],
    propagated_phase_initial_time_guess: Optional[float] = None,
    propagated_phase_terminal_time_guess: Optional[float] = None,
    propagated_integral_values_guess: Optional[Union[float, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Generates a robust, correctly dimensioned default initial initial_guess_data.
    """
    num_intervals = len(current_list_of_collocation_nodes_per_interval_for_guess)
    num_states = base_optimal_control_problem_definition_for_defaults['num_states']
    num_controls = base_optimal_control_problem_definition_for_defaults['num_controls']
    num_integrals = base_optimal_control_problem_definition_for_defaults.get('num_integrals', 0)

    default_value_for_state_variable_guess = base_optimal_control_problem_definition_for_defaults.get('default_initial_guess_values', {}).get('state', 0.0)
    default_value_for_control_variable_guess = base_optimal_control_problem_definition_for_defaults.get('default_initial_guess_values', {}).get('control', 0.0)
    
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

    final_initial_time_guess_value = propagated_phase_initial_time_guess if propagated_phase_initial_time_guess is not None else base_optimal_control_problem_definition_for_defaults.get('initial_guess', {}).get('initial_time_variable', 0.0)
    final_terminal_time_guess_value = propagated_phase_terminal_time_guess if propagated_phase_terminal_time_guess is not None else base_optimal_control_problem_definition_for_defaults.get('initial_guess', {}).get('terminal_time_variable', 1.0)

    final_integral_values_guess: Optional[Union[float, List[float], np.ndarray]] = None
    if num_integrals > 0:
        if propagated_integral_values_guess is not None:
            final_integral_values_guess = propagated_integral_values_guess
        else:
            default_value_for_integral_guess = base_optimal_control_problem_definition_for_defaults.get('default_initial_guess_values', {}).get('integral', 0.0)
            raw_guess = base_optimal_control_problem_definition_for_defaults.get('initial_guess', {}).get('integrals', [default_value_for_integral_guess] * num_integrals if num_integrals > 1 else default_value_for_integral_guess)
            if num_integrals == 1:
                 final_integral_values_guess = float(raw_guess) if not isinstance(raw_guess, (list, np.ndarray)) else float(raw_guess[0])
            elif isinstance(raw_guess, (list, np.ndarray)) and len(raw_guess) == num_integrals:
                 final_integral_values_guess = np.array(raw_guess)
            else: 
                 final_integral_values_guess = np.full(num_integrals, default_value_for_integral_guess)
    
    return {
        "states": list_of_state_trajectory_guesses_for_each_interval,
        "controls": list_of_control_trajectory_guesses_for_each_interval,
        "initial_time_variable": final_initial_time_guess_value,
        "terminal_time_variable": final_terminal_time_guess_value,
        "integrals": final_integral_values_guess
    }

def _propagate_guess_from_previous(
    previous_optimal_control_solution_to_propagate_from: Solution,
    base_optimal_control_problem_definition_for_propagation: ProblemDefinition,
    target_list_of_collocation_nodes_per_interval_for_propagation: List[int],
    target_global_normalized_mesh_nodes_for_propagation: np.ndarray
) -> Dict[str, Any]:
    """
    Creates an initial initial_guess_data for the current NLP solve, propagating from previous_optimal_control_solution_to_propagate_from.
    """
    t0_prop = previous_optimal_control_solution_to_propagate_from.get('initial_time_variable')
    tf_prop = previous_optimal_control_solution_to_propagate_from.get('terminal_time_variable')
    integrals_prop = previous_optimal_control_solution_to_propagate_from.get('integrals')

    initial_guess_dictionary_propagated_from_previous_solution = _generate_robust_default_initial_guess(
        base_optimal_control_problem_definition_for_propagation, target_list_of_collocation_nodes_per_interval_for_propagation,
        propagated_phase_initial_time_guess=t0_prop, propagated_phase_terminal_time_guess=tf_prop, propagated_integral_values_guess=integrals_prop
    )

    num_collocation_nodes_list_from_previous_solution = previous_optimal_control_solution_to_propagate_from.get("Nk_list_at_solve_time")
    global_normalized_mesh_nodes_from_previous_solution = previous_optimal_control_solution_to_propagate_from.get("mesh_nodes_tau_global_at_solve_time")

    can_state_control_trajectories_be_directly_propagated = False
    if num_collocation_nodes_list_from_previous_solution is not None and global_normalized_mesh_nodes_from_previous_solution is not None:
        if np.array_equal(target_list_of_collocation_nodes_per_interval_for_propagation, num_collocation_nodes_list_from_previous_solution) and \
           np.allclose(target_global_normalized_mesh_nodes_for_propagation, global_normalized_mesh_nodes_from_previous_solution):
            can_state_control_trajectories_be_directly_propagated = True

    if can_state_control_trajectories_be_directly_propagated:
        print("  Mesh structure identical to previous. Propagating state/control trajectories directly.")
        state_trajectories_list_from_previous_solution = previous_optimal_control_solution_to_propagate_from.get('solved_state_trajectories_per_interval')
        control_trajectories_list_from_previous_solution = previous_optimal_control_solution_to_propagate_from.get('solved_control_trajectories_per_interval')

        if state_trajectories_list_from_previous_solution and len(state_trajectories_list_from_previous_solution) == len(target_list_of_collocation_nodes_per_interval_for_propagation):
            initial_guess_dictionary_propagated_from_previous_solution['states'] = state_trajectories_list_from_previous_solution
        else:
            print("    Warning: Previous Y_solved_values_list mismatch or missing. Using default states.")
        
        if control_trajectories_list_from_previous_solution and len(control_trajectories_list_from_previous_solution) == len(target_list_of_collocation_nodes_per_interval_for_propagation):
            initial_guess_dictionary_propagated_from_previous_solution['controls'] = control_trajectories_list_from_previous_solution
        else:
             print("    Warning: Previous U_solved_values_list mismatch or missing. Using default controls.")
    else:
        print("  Mesh structure changed. Using robust default for state/control trajectories (times/integrals propagated).")

    return initial_guess_dictionary_propagated_from_previous_solution