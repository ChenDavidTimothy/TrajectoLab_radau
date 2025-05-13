"""
PHS-Adaptive mesh refinement method implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from scipy.integrate import solve_ivp

from ...core.problem import ProblemDefinition, Solution
from ...core.solver import solve_single_phase_radau_collocation
from ...core.basis import compute_radau_collocation_components, _compute_barycentric_weights
from ..base.base import AdaptiveMethod, AdaptiveParams
from .error import (IntervalSimulationBundle, _extract_and_prepare_array,
                   get_polynomial_interpolant, _simulate_dynamics_for_error_estimation, 
                   calculate_relative_error_estimate, _calculate_gamma_normalizers)
from .refinement import (p_refine_interval, h_refine_params, h_reduce_intervals, p_reduce_interval,
                        _map_global_normalized_tau_to_local_interval_tau,
                        _map_local_interval_tau_to_global_normalized_tau,
                        _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1,
                        _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k,
                        _generate_robust_default_initial_guess,
                        _propagate_guess_from_previous)

class PHSMethod(AdaptiveMethod):
    """
    Implementation of the PHS (p-refinement, h-refinement, and h/p-reduction) adaptive method.
    """
    
    def __init__(self, problem_definition: ProblemDefinition, params: Optional[AdaptiveParams] = None):
        """
        Initialize the PHS adaptive method.
        
        Args:
            problem_definition: The optimal control problem definition
            params: PHS-specific parameters, or None to use defaults
        """
        if params is None:
            params = self.get_default_params()
        super().__init__(problem_definition, params)
    
    def solve(self) -> Solution:
        """
        Solve using PHS-Adaptive mesh refinement.
        
        Returns:
            Solution dictionary
        """
        # Call the original function with our parameters
        return run_phs_adaptive_mesh_refinement(self.problem_definition, self.params)
    
    @classmethod
    def get_default_params(cls) -> AdaptiveParams:
        """
        Get default parameters for the PHS method.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'epsilon_tol': 1e-3,
            'M_max_iterations': 30,
            'N_min_poly_degree': 4,
            'N_max_poly_degree': 10,
            'ode_solver_tol': 1e-7,
            'num_error_sim_points': 50
        }


# Copy the original run_phs_adaptive_mesh_refinement function here
def run_phs_adaptive_mesh_refinement(
    initial_optimal_control_problem_definition: ProblemDefinition,
    adaptive_mesh_refinement_parameters: AdaptiveParams
) -> Solution:
    """
    Implements Algorithm 1 from mesh.txt for PHS-Adaptive mesh refinement.
    """
    desired_mesh_error_tolerance_epsilon = adaptive_mesh_refinement_parameters['epsilon_tol']
    max_allowable_adaptive_refinement_iterations = adaptive_mesh_refinement_parameters['M_max_iterations']
    min_allowable_collocation_nodes_per_interval_setting = adaptive_mesh_refinement_parameters['N_min_poly_degree']
    max_allowable_collocation_nodes_per_interval_setting = adaptive_mesh_refinement_parameters['N_max_poly_degree']
    ode_solver_relative_tolerance_for_error_simulations = adaptive_mesh_refinement_parameters.get('ode_solver_tol', 1e-7)
    number_of_evaluation_points_for_error_simulation_per_interval = adaptive_mesh_refinement_parameters.get('num_error_sim_points', 50)

    current_iteration_list_of_collocation_nodes_per_interval: List[int] = list(initial_optimal_control_problem_definition['collocation_points_per_interval'])
    current_iteration_global_normalized_mesh_nodes_array: np.ndarray = np.array(
        initial_optimal_control_problem_definition.get('global_normalized_mesh_nodes',
                                np.linspace(-1, 1, len(current_iteration_list_of_collocation_nodes_per_interval) + 1))
    )

    for i in range(len(current_iteration_list_of_collocation_nodes_per_interval)):
        current_iteration_list_of_collocation_nodes_per_interval[i] = max(min_allowable_collocation_nodes_per_interval_setting, min(max_allowable_collocation_nodes_per_interval_setting, current_iteration_list_of_collocation_nodes_per_interval[i]))

    problem_definition_for_current_iteration_nlp = {key: (val.copy() if isinstance(val, dict) else val)
                                for key, val in initial_optimal_control_problem_definition.items()}
    num_states = problem_definition_for_current_iteration_nlp['num_states']
    num_controls = problem_definition_for_current_iteration_nlp['num_controls']

    most_recent_successful_optimal_control_solution_details: Optional[Solution] = None

    for current_adaptive_refinement_iteration_number in range(max_allowable_adaptive_refinement_iterations):
        print(f"\n--- Adaptive Iteration M = {current_adaptive_refinement_iteration_number} ---")
        number_of_mesh_intervals_in_current_iteration = len(current_iteration_list_of_collocation_nodes_per_interval)

        problem_definition_for_current_iteration_nlp['collocation_points_per_interval'] = list(current_iteration_list_of_collocation_nodes_per_interval)
        problem_definition_for_current_iteration_nlp['global_normalized_mesh_nodes'] = list(current_iteration_global_normalized_mesh_nodes_array) # Ensure it's a list for JSON later if problem_definition is saved

        initial_guess_data_for_current_nlp_solve: Dict[str, Any]
        if current_adaptive_refinement_iteration_number == 0 or not most_recent_successful_optimal_control_solution_details or not most_recent_successful_optimal_control_solution_details.get("success", False):
            print("  First iteration or previous NLP failed/unavailable. Using robust default initial initial_guess_data.")
            initial_guess_data_for_current_nlp_solve = _generate_robust_default_initial_guess(
                initial_optimal_control_problem_definition, current_iteration_list_of_collocation_nodes_per_interval
            )
        else:
            initial_guess_data_for_current_nlp_solve = _propagate_guess_from_previous(
                most_recent_successful_optimal_control_solution_details, initial_optimal_control_problem_definition,
                current_iteration_list_of_collocation_nodes_per_interval, current_iteration_global_normalized_mesh_nodes_array
            )
        problem_definition_for_current_iteration_nlp['initial_guess'] = initial_guess_data_for_current_nlp_solve
        
        print(f"  Mesh K={number_of_mesh_intervals_in_current_iteration}, num_collocation_nodes_per_interval = {current_iteration_list_of_collocation_nodes_per_interval}")
        print(f"  Mesh nodes_tau_global = {np.array2string(current_iteration_global_normalized_mesh_nodes_array, precision=3)}")

        optimal_control_solution_from_current_nlp_solve: Solution = solve_single_phase_radau_collocation(problem_definition_for_current_iteration_nlp)

        if not optimal_control_solution_from_current_nlp_solve.get("success", False):
            error_or_status_message = f"NLP solver failed in adaptive iteration {current_adaptive_refinement_iteration_number}. " + (optimal_control_solution_from_current_nlp_solve.get("message", "") or "Solver error.")
            print(f"  Error: {error_or_status_message} Stopping.")
            if most_recent_successful_optimal_control_solution_details: # Return the last good one, but mark as overall adaptive failure
                 most_recent_successful_optimal_control_solution_details["message"] = error_or_status_message
                 most_recent_successful_optimal_control_solution_details["success"] = False # Overall adaptive process failed
                 return most_recent_successful_optimal_control_solution_details
            else: # NLP failed on the very first attempt
                 optimal_control_solution_from_current_nlp_solve["message"] = error_or_status_message
                 return optimal_control_solution_from_current_nlp_solve # This already has success=False

        # Store solved values directly in the solution dictionary for propagation and interpolation
        try:
            casadi_optimization_problem_object = optimal_control_solution_from_current_nlp_solve['opti_object']; raw_sol = optimal_control_solution_from_current_nlp_solve['raw_solution']
            optimal_control_solution_from_current_nlp_solve['solved_state_trajectories_per_interval'] = [
                _extract_and_prepare_array(raw_sol.value(var), num_states, current_iteration_list_of_collocation_nodes_per_interval[i] + 1)
                for i, var in enumerate(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables)
            ]
            if num_controls > 0:
                optimal_control_solution_from_current_nlp_solve['solved_control_trajectories_per_interval'] = [
                    _extract_and_prepare_array(raw_sol.value(var), num_controls, current_iteration_list_of_collocation_nodes_per_interval[i])
                    for i, var in enumerate(casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables)
                ]
            else:
                optimal_control_solution_from_current_nlp_solve['solved_control_trajectories_per_interval'] = [np.empty((0, current_iteration_list_of_collocation_nodes_per_interval[i])) for i in range(number_of_mesh_intervals_in_current_iteration)]

        except Exception as e:
            error_or_status_message = f"Failed to extract solved trajectories from NLP solution at iter {current_adaptive_refinement_iteration_number}: {e}. Stopping."
            print(f"  Error: {error_or_status_message}")
            optimal_control_solution_from_current_nlp_solve["message"] = error_or_status_message
            optimal_control_solution_from_current_nlp_solve["success"] = False
            return optimal_control_solution_from_current_nlp_solve

        most_recent_successful_optimal_control_solution_details = optimal_control_solution_from_current_nlp_solve # This is now the latest successful one
        most_recent_successful_optimal_control_solution_details["num_collocation_nodes_list_at_solve_time"] = list(current_iteration_list_of_collocation_nodes_per_interval) # For initial_guess_data propagation
        most_recent_successful_optimal_control_solution_details["mesh_nodes_tau_global_at_solve_time"] = np.copy(current_iteration_global_normalized_mesh_nodes_array) # For initial_guess_data propagation

        state_component_error_normalization_factors_gamma_for_iteration = _calculate_gamma_normalizers(optimal_control_solution_from_current_nlp_solve, problem_definition_for_current_iteration_nlp)
        if state_component_error_normalization_factors_gamma_for_iteration is None and num_states > 0: # num_states > 0 check because gamma_i is empty if num_states=0
            error_or_status_message = f"Failed to calculate gamma_i normalizers at iter {current_adaptive_refinement_iteration_number}. Stopping."
            print(f"  Error: {error_or_status_message}")
            optimal_control_solution_from_current_nlp_solve["message"] = error_or_status_message
            optimal_control_solution_from_current_nlp_solve["success"] = False # Mark as failure
            return optimal_control_solution_from_current_nlp_solve
        
        # Pre-compute basis components and polynomial interpolants for all intervals
        list_of_state_polynomial_evaluators_for_all_intervals: List[Optional[Callable]] = [None] * number_of_mesh_intervals_in_current_iteration
        list_of_control_polynomial_evaluators_for_all_intervals: List[Optional[Callable]] = [None] * number_of_mesh_intervals_in_current_iteration

        # Get solved state/control trajectories for the current iteration
        current_iteration_solved_state_trajectories_list = optimal_control_solution_from_current_nlp_solve['solved_state_trajectories_per_interval']
        current_iteration_solved_control_trajectories_list = optimal_control_solution_from_current_nlp_solve.get('solved_control_trajectories_per_interval', [])

        # Caches to avoid redundant computation of basis components and barycentric weights
        # These depend only on Nk, not on interval-specific data
        basis_components_cache: Dict[int, Dict[str, Any]] = {}
        control_barycentric_weights_cache: Dict[int, np.ndarray] = {}

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
                    basis_comps['state_approximation_nodes'], 
                    state_data_for_interpolant,
                    basis_comps['barycentric_weights_for_state_nodes']
                )
                
                # Create control polynomial interpolant or empty function if no controls
                if num_controls > 0:
                    control_data_for_interpolant = current_iteration_solved_control_trajectories_list[k_idx_interp]
                    
                    # Use cache for control barycentric weights
                    if Nk_interp not in control_barycentric_weights_cache:
                        # Only compute if we haven't seen this Nk before
                        control_barycentric_weights_cache[Nk_interp] = _compute_barycentric_weights(basis_comps['collocation_nodes'])
                    
                    # Get the control barycentric weights from cache
                    control_bary_weights = control_barycentric_weights_cache[Nk_interp]
                    
                    # Create control interpolant ONCE
                    list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = get_polynomial_interpolant(
                        basis_comps['collocation_nodes'],
                        control_data_for_interpolant,
                        control_bary_weights
                    )
                else:
                    # No controls - create empty function
                    list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = lambda zeta: np.array([])
                
            except Exception as e:
                print(f"  Warning: Error creating interpolant for interval {k_idx_interp}: {e}. Error estimation might be affected.")
                # Create fallback interpolants that return NaN values
                if list_of_state_polynomial_evaluators_for_all_intervals[k_idx_interp] is None:
                    list_of_state_polynomial_evaluators_for_all_intervals[k_idx_interp] = lambda zeta: np.full((num_states, np.atleast_1d(zeta).shape[0]), np.nan)
                if list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] is None:
                    if num_controls > 0:
                        list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = lambda zeta: np.full((num_controls, np.atleast_1d(zeta).shape[0]), np.nan)
                    else:
                        list_of_control_polynomial_evaluators_for_all_intervals[k_idx_interp] = lambda zeta: np.array([])
        
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
            estimated_max_relative_error_for_current_processed_interval = calculate_relative_error_estimate(mesh_interval_index, error_estimation_simulation_bundle_for_current_interval, state_component_error_normalization_factors_gamma_for_iteration if num_states > 0 else np.array([]))
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
                    are_all_interval_errors_within_tolerance_and_valid = False; break
        
        if are_all_interval_errors_within_tolerance_and_valid:
            print(f"Mesh converged after {current_adaptive_refinement_iteration_number+1} iterations.")
            optimal_control_solution_from_current_nlp_solve["num_collocation_nodes_per_interval"] = current_iteration_list_of_collocation_nodes_per_interval.copy() # Final mesh config
            optimal_control_solution_from_current_nlp_solve["global_normalized_mesh_nodes"] = np.copy(current_iteration_global_normalized_mesh_nodes_array)
            optimal_control_solution_from_current_nlp_solve["message"] = f"Adaptive mesh converged to tolerance {desired_mesh_error_tolerance_epsilon:.1e} in {current_adaptive_refinement_iteration_number+1} iterations."
            # optimal_control_solution_from_current_nlp_solve["success"] is already True from the solver
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
                
                # Call MODIFIED p_refine_interval
                actual_Nk_from_p_attempt, p_success_flag, unconstrained_target_Nk_for_h = p_refine_interval(
                    num_collocation_nodes_in_current_original_interval_being_processed,
                    max_relative_error_in_current_original_interval_being_processed,
                    desired_mesh_error_tolerance_epsilon,
                    max_allowable_collocation_nodes_per_interval_setting
                )
                
                if p_success_flag:
                    print(f"        p-refinement applied: Nk {num_collocation_nodes_in_current_original_interval_being_processed} -> {actual_Nk_from_p_attempt}")
                    list_of_collocation_nodes_for_next_iteration_being_built.append(actual_Nk_from_p_attempt)
                    global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1])
                    current_original_mesh_interval_index_being_processed += 1
                else: 
                    # p-refinement "failed" because unconstrained_target_Nk_for_h > N_max (or no increase was possible)
                    # The actual_Nk_from_p_attempt would be N_max in the case of exceeding N_max.
                    # unconstrained_target_Nk_for_h holds the value from Eq. 27 needed for h-refinement.
                    print(f"        p-refinement failed (target Nk {unconstrained_target_Nk_for_h} would exceed N_max {max_allowable_collocation_nodes_per_interval_setting}, or no increase possible). Attempting h-refinement.")
                    
                    # Directly use unconstrained_target_Nk_for_h for h-refinement parameter calculation
                    list_of_collocation_nodes_for_newly_created_subintervals_from_h_refinement, number_of_subintervals_created_by_h_refinement_for_current_original_interval = h_refine_params(
                        unconstrained_target_Nk_for_h, # Use the unconstrained target from p_refine_interval
                        min_allowable_collocation_nodes_per_interval_setting
                    )
                    print(f"          h-refinement: Splitting interval {current_original_mesh_interval_index_being_processed} into {number_of_subintervals_created_by_h_refinement_for_current_original_interval} subintervals, each Nk={list_of_collocation_nodes_for_newly_created_subintervals_from_h_refinement[0]}.")
                    list_of_collocation_nodes_for_next_iteration_being_built.extend(list_of_collocation_nodes_for_newly_created_subintervals_from_h_refinement)
                    
                    global_normalized_start_tau_of_original_interval_being_split = current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed]
                    global_normalized_end_tau_of_original_interval_being_split = current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1]
                    newly_created_global_normalized_mesh_nodes_for_subintervals = np.linspace(
                        global_normalized_start_tau_of_original_interval_being_split, 
                        global_normalized_end_tau_of_original_interval_being_split, 
                        number_of_subintervals_created_by_h_refinement_for_current_original_interval + 1
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
                            current_original_mesh_interval_index_being_processed, optimal_control_solution_from_current_nlp_solve, problem_definition_for_current_iteration_nlp, adaptive_mesh_refinement_parameters,
                            state_component_error_normalization_factors_gamma_for_iteration if num_states > 0 else np.array([]), # Pass calculated gamma
                            list_of_state_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed], list_of_control_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed],
                            list_of_state_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed+1], list_of_control_polynomial_evaluators_for_all_intervals[current_original_mesh_interval_index_being_processed+1]
                        )
                    else:
                        print("      Skipping h-reduction attempt due to missing interpolants (should be rare).")

                    if was_h_reduction_merge_successful_for_current_original_interval_pair:
                        print(f"      h-reduction applied to merge interval {current_original_mesh_interval_index_being_processed} and {current_original_mesh_interval_index_being_processed+1}.")
                        num_collocation_nodes_for_newly_merged_interval_from_h_reduction = max(current_iteration_list_of_collocation_nodes_per_interval[current_original_mesh_interval_index_being_processed], current_iteration_list_of_collocation_nodes_per_interval[current_original_mesh_interval_index_being_processed+1])
                        num_collocation_nodes_for_newly_merged_interval_from_h_reduction = max(min_allowable_collocation_nodes_per_interval_setting, min(max_allowable_collocation_nodes_per_interval_setting, num_collocation_nodes_for_newly_merged_interval_from_h_reduction))
                        list_of_collocation_nodes_for_next_iteration_being_built.append(num_collocation_nodes_for_newly_merged_interval_from_h_reduction)
                        global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 2])
                        current_original_mesh_interval_index_being_processed += 2 
                    else: 
                        print(f"      h-reduction failed or condition not met. Attempting p-reduction for interval {current_original_mesh_interval_index_being_processed}.")
                        num_collocation_nodes_after_p_reduction_attempt_for_current_original_interval, was_p_reduction_applied_to_current_original_interval = p_reduce_interval(
                            num_collocation_nodes_in_current_original_interval_being_processed, max_relative_error_in_current_original_interval_being_processed, desired_mesh_error_tolerance_epsilon, min_allowable_collocation_nodes_per_interval_setting, max_allowable_collocation_nodes_per_interval_setting
                        )
                        if was_p_reduction_applied_to_current_original_interval: print(f"        p-reduction applied: Nk {num_collocation_nodes_in_current_original_interval_being_processed} -> {num_collocation_nodes_after_p_reduction_attempt_for_current_original_interval}")
                        else: print(f"        p-reduction not applied for Nk {num_collocation_nodes_in_current_original_interval_being_processed}.")
                        list_of_collocation_nodes_for_next_iteration_being_built.append(num_collocation_nodes_after_p_reduction_attempt_for_current_original_interval)
                        global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1])
                        current_original_mesh_interval_index_being_processed += 1
                else: 
                    print(f"      Not eligible for h-reduction with next interval. Attempting p-reduction for interval {current_original_mesh_interval_index_being_processed}.")
                    num_collocation_nodes_after_p_reduction_attempt_for_current_original_interval, was_p_reduction_applied_to_current_original_interval = p_reduce_interval(
                        num_collocation_nodes_in_current_original_interval_being_processed, max_relative_error_in_current_original_interval_being_processed, desired_mesh_error_tolerance_epsilon, min_allowable_collocation_nodes_per_interval_setting, max_allowable_collocation_nodes_per_interval_setting
                    )
                    if was_p_reduction_applied_to_current_original_interval: print(f"        p-reduction applied: Nk {num_collocation_nodes_in_current_original_interval_being_processed} -> {num_collocation_nodes_after_p_reduction_attempt_for_current_original_interval}")
                    else: print(f"        p-reduction not applied for Nk {num_collocation_nodes_in_current_original_interval_being_processed}.")
                    list_of_collocation_nodes_for_next_iteration_being_built.append(num_collocation_nodes_after_p_reduction_attempt_for_current_original_interval)
                    global_normalized_mesh_nodes_for_next_iteration_being_built.append(current_iteration_global_normalized_mesh_nodes_array[current_original_mesh_interval_index_being_processed + 1])
                    current_original_mesh_interval_index_being_processed += 1
        
        current_iteration_list_of_collocation_nodes_per_interval = list_of_collocation_nodes_for_next_iteration_being_built
        current_iteration_global_normalized_mesh_nodes_array = np.array(global_normalized_mesh_nodes_for_next_iteration_being_built) # Ensure it's an array for next iter
        
        # --- Mesh Sanity Checks ---
        solution_object_for_early_return_due_to_mesh_inconsistency = most_recent_successful_optimal_control_solution_details if most_recent_successful_optimal_control_solution_details else \
                                 {"success": False, "message": "", "num_collocation_nodes_per_interval": current_iteration_list_of_collocation_nodes_per_interval, "global_normalized_mesh_nodes": current_iteration_global_normalized_mesh_nodes_array}

        if not current_iteration_list_of_collocation_nodes_per_interval and len(current_iteration_global_normalized_mesh_nodes_array) > 1 :
            error_or_status_message = "Stopped due to mesh inconsistency (empty num_collocation_nodes_per_interval but mesh_nodes exist)."
            print(f"  Error: {error_or_status_message} Stopping.")
            solution_object_for_early_return_due_to_mesh_inconsistency["message"] = error_or_status_message; solution_object_for_early_return_due_to_mesh_inconsistency["success"] = False
            return solution_object_for_early_return_due_to_mesh_inconsistency

        if current_iteration_list_of_collocation_nodes_per_interval and len(current_iteration_list_of_collocation_nodes_per_interval) != (len(current_iteration_global_normalized_mesh_nodes_array) -1):
            error_or_status_message = f"Mesh structure inconsistent after refinement. num_collocation_nodes_per_interval len: {len(current_iteration_list_of_collocation_nodes_per_interval)}, Nodes len-1: {len(current_iteration_global_normalized_mesh_nodes_array)-1}."
            print(f"  Error: {error_or_status_message} Stopping.")
            solution_object_for_early_return_due_to_mesh_inconsistency["message"] = error_or_status_message; solution_object_for_early_return_due_to_mesh_inconsistency["success"] = False
            return solution_object_for_early_return_due_to_mesh_inconsistency
        
        if len(current_iteration_list_of_collocation_nodes_per_interval) > 0 :
             unique_global_mesh_nodes_after_refinement, counts_of_unique_global_mesh_nodes_after_refinement = np.unique(np.round(current_iteration_global_normalized_mesh_nodes_array, decimals=12), return_counts=True)
             if np.any(counts_of_unique_global_mesh_nodes_after_refinement > 1):
                  detected_duplicate_global_mesh_nodes_after_refinement = unique_global_mesh_nodes_after_refinement[counts_of_unique_global_mesh_nodes_after_refinement > 1]
                  error_or_status_message = f"Duplicate mesh nodes found (after rounding): {detected_duplicate_global_mesh_nodes_after_refinement}. Original nodes: {current_iteration_global_normalized_mesh_nodes_array}."
                  print(f"  Error: {error_or_status_message} Stopping.")
                  solution_object_for_early_return_due_to_mesh_inconsistency["message"] = error_or_status_message; solution_object_for_early_return_due_to_mesh_inconsistency["success"] = False
                  return solution_object_for_early_return_due_to_mesh_inconsistency
             if len(unique_global_mesh_nodes_after_refinement) > 1 and not np.all(np.diff(unique_global_mesh_nodes_after_refinement) > 1e-9): 
                  # sorted_global_mesh_nodes_for_consistency_check = np.sort(current_iteration_global_normalized_mesh_nodes_array) # unique_global_mesh_nodes_after_refinement is already sorted
                  differences_between_sorted_unique_global_mesh_nodes_after_refinement = np.diff(unique_global_mesh_nodes_after_refinement) # Check differences_between_sorted_unique_global_mesh_nodes_after_refinement of unique, sorted nodes
                  indices_of_problematic_small_or_non_positive_node_differences = np.where(differences_between_sorted_unique_global_mesh_nodes_after_refinement <= 1e-9)[0]
                  string_representation_of_problematic_global_mesh_node_pairs = ", ".join([f"({unique_global_mesh_nodes_after_refinement[i]:.3f}, {unique_global_mesh_nodes_after_refinement[i+1]:.3f})" for i in indices_of_problematic_small_or_non_positive_node_differences]) if indices_of_problematic_small_or_non_positive_node_differences.size > 0 else "N/A"
                  error_or_status_message = f"Mesh nodes not strictly increasing or interval too small. Problem pairs: {string_representation_of_problematic_global_mesh_node_pairs}. All nodes: {current_iteration_global_normalized_mesh_nodes_array}."
                  print(f"  Error: {error_or_status_message} Stopping.")
                  solution_object_for_early_return_due_to_mesh_inconsistency["message"] = error_or_status_message; solution_object_for_early_return_due_to_mesh_inconsistency["success"] = False
                  return solution_object_for_early_return_due_to_mesh_inconsistency

    # --- Max Iterations Reached ---
    message_indicating_max_iterations_reached_without_full_convergence = f"Adaptive mesh refinement reached max iterations ({max_allowable_adaptive_refinement_iterations}) without full convergence to tolerance {desired_mesh_error_tolerance_epsilon:.1e}."
    print(message_indicating_max_iterations_reached_without_full_convergence)
    if most_recent_successful_optimal_control_solution_details: # This means at least one NLP solve was successful
        most_recent_successful_optimal_control_solution_details["message"] = message_indicating_max_iterations_reached_without_full_convergence
        # most_recent_successful_optimal_control_solution_details["success"] remains True (NLP solved) but overall adaptive process did not converge to tolerance.
        # The calling function should check the message if it needs to distinguish.
        most_recent_successful_optimal_control_solution_details["num_collocation_nodes_per_interval"] = current_iteration_list_of_collocation_nodes_per_interval.copy()
        most_recent_successful_optimal_control_solution_details["global_normalized_mesh_nodes"] = np.copy(current_iteration_global_normalized_mesh_nodes_array)
        return most_recent_successful_optimal_control_solution_details
    else: # No NLP was ever successful
        return {
            "success": False, "message": message_indicating_max_iterations_reached_without_full_convergence + " No successful NLP solution obtained throughout iterations.",
            "initial_time_variable": None, "terminal_time_variable": None, "objective": None, "integrals": None,
            "time_states": np.array([]), "states": [], "time_controls": np.array([]), "controls": [],
            "raw_solution": None, "opti_object": None,
            "num_collocation_nodes_per_interval": current_iteration_list_of_collocation_nodes_per_interval, "global_normalized_mesh_nodes": current_iteration_global_normalized_mesh_nodes_array.tolist() if isinstance(current_iteration_global_normalized_mesh_nodes_array, np.ndarray) else current_iteration_global_normalized_mesh_nodes_array
        }