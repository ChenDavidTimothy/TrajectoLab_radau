# solver_radau.py (Production Ready - Variable Sharing Implementation)

import casadi as ca
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

# Assuming radau_pseudospectral_basis.py is in the same directory or accessible in PYTHONPATH
import rpm_basis # Ensure this import works

# Define a type alias for the problem definition dictionary for clarity
ProblemDefinition = Dict[str, Any]
# Type alias for the solution dictionary
Solution = Dict[str, Any]


def _extract_and_format_solution(
    casadi_solution_object: Optional[ca.OptiSol], # CasADi solution object, can be None if solver fails
    casadi_optimization_problem_object: ca.Opti,         # CasADi Opti object with attached problem variables
    problem_definition: ProblemDefinition,
    num_collocation_nodes_per_interval: List[int], # List of collocation points per interval
    global_normalized_mesh_nodes: np.ndarray # Actual global mesh nodes used
) -> Solution:
    """
    Extracts and formats the solution from the CasADi solution object.

    Args:
        casadi_solution_object: The solution object returned by opti.solve(). None if solver failed.
        casadi_optimization_problem_object: The CasADi Opti stack object containing the problem formulation
                  and solved variables.
        problem_definition: The dictionary defining the optimal control problem.
        num_collocation_nodes_per_interval: List of the number of collocation points in each interval.
        global_normalized_mesh_nodes: The global tau mesh nodes used for this solution.

    Returns:
        A dictionary containing the solution results (times, states, controls,
        objective, success status, etc.).
    """
    if casadi_solution_object is None:
        return {
            "success": False,
            "message": "Solver did not find a solution or was not run.",
            "initial_time_variable": None, "terminal_time_variable": None, "objective": None, "integrals": None,
            "time_states": np.array([]), "states": [],
            "time_controls": np.array([]), "controls": [],
            "raw_solution": None, "opti_object": casadi_optimization_problem_object,
            "num_collocation_nodes_per_interval": num_collocation_nodes_per_interval,
            "global_normalized_mesh_nodes": global_normalized_mesh_nodes
        }

    num_mesh_intervals: int = len(num_collocation_nodes_per_interval)
    num_states: int = problem_definition['num_states']
    num_controls: int = problem_definition['num_controls']
    num_integrals: int = problem_definition.get('num_integrals', 0)

    try:
        solved_initial_time: float = casadi_solution_object.value(casadi_optimization_problem_object.initial_time_variable_reference)
        solved_terminal_time: float = casadi_solution_object.value(casadi_optimization_problem_object.terminal_time_variable_reference)
        solved_objective_value: float = casadi_solution_object.value(casadi_optimization_problem_object.symbolic_objective_function_reference)
    except Exception as e: # Handle cases where variables might not be available in a failed solution
        return {
            "success": False,
            "message": f"Failed to extract core solution values: {e}",
            "initial_time_variable": None, "terminal_time_variable": None, "objective": None, "integrals": None,
            "time_states": np.array([]), "states": [],
            "time_controls": np.array([]), "controls": [],
            "raw_solution": casadi_solution_object, "opti_object": casadi_optimization_problem_object,
            "num_collocation_nodes_per_interval": num_collocation_nodes_per_interval,
            "global_normalized_mesh_nodes": global_normalized_mesh_nodes
        }


    solved_integral_values: Optional[Union[float, np.ndarray]] = None
    if num_integrals > 0 and hasattr(casadi_optimization_problem_object, 'integral_variables_object_reference') and casadi_optimization_problem_object.integral_variables_object_reference is not None:
        try:
            raw_solved_integral_value_from_casadi = casadi_solution_object.value(casadi_optimization_problem_object.integral_variables_object_reference)
            if isinstance(raw_solved_integral_value_from_casadi, ca.DM):
                if num_integrals == 1 and raw_solved_integral_value_from_casadi.shape == (1, 1): # Single integral scalar
                    solved_integral_values = float(raw_solved_integral_value_from_casadi[0,0])
                else: # Multiple integrals or single integral as (1,) DM
                    solved_integral_values = raw_solved_integral_value_from_casadi.toarray().flatten()
            else: # Assumed scalar for num_integrals=1 if not DM
                solved_integral_values = float(raw_solved_integral_value_from_casadi)
        except Exception as e:
            print(f"Warning: Could not extract integral values: {e}")
            solved_integral_values = np.full(num_integrals, np.nan) if num_integrals > 1 else np.nan


    state_trajectory_times: List[float] = []
    state_trajectory_values_list_of_lists: List[List[float]] = [[] for _ in range(num_states)]
    control_trajectory_times: List[float] = []
    control_trajectory_values_list_of_lists: List[List[float]] = [[] for _ in range(num_controls)]

    # Use the global_normalized_mesh_nodes that was actually used in the solver
    # (passed as global_normalized_mesh_nodes)
    # instead of casadi_optimization_problem_object.metadata_global_normalized_mesh_nodes if there's any doubt
    # about when metadata_global_normalized_mesh_nodes is set vs. used.
    # However, they should be the same if set correctly in solve_single_phase_radau_collocation.

    last_time_point_added_to_state_trajectory: float = -np.inf

    for mesh_interval_index in range(num_mesh_intervals):
        # solved_state_at_local_approximation_nodes_current_interval: np.ndarray = casadi_solution_object.value(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables[mesh_interval_index])
        # Ensure state_at_local_approximation_nodes_all_intervals_variables is correctly populated and accessible
        if not hasattr(casadi_optimization_problem_object, 'state_at_local_approximation_nodes_all_intervals_variables') or mesh_interval_index >= len(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables):
            print(f"Error: state_at_local_approximation_nodes_all_intervals_variables not found or incomplete in casadi_optimization_problem_object for interval {mesh_interval_index}.")
            # Add dummy values or handle error appropriately
            continue # Skip this interval if data is missing
        solved_state_at_local_approximation_nodes_current_interval: np.ndarray = casadi_solution_object.value(casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables[mesh_interval_index])


        local_state_approximation_nodes_tau_current_interval: np.ndarray = casadi_optimization_problem_object.metadata_local_state_approximation_nodes_tau[mesh_interval_index] # (num_collocation_nodes_in_current_interval+1,)
        num_collocation_nodes_current_interval: int = num_collocation_nodes_per_interval[mesh_interval_index]

        if num_states == 1 and solved_state_at_local_approximation_nodes_current_interval.ndim == 1:
            solved_state_at_local_approximation_nodes_current_interval = solved_state_at_local_approximation_nodes_current_interval.reshape(1, -1)

        for local_state_approximation_node_index in range(num_collocation_nodes_current_interval + 1):
            current_local_tau_value_at_node: float = local_state_approximation_nodes_tau_current_interval[local_state_approximation_node_index]
            
            current_global_normalized_segment_start_tau: float = global_normalized_mesh_nodes[mesh_interval_index]
            current_global_normalized_segment_end_tau: float = global_normalized_mesh_nodes[mesh_interval_index+1]
            current_global_normalized_tau_value_at_node: float = (current_global_normalized_segment_end_tau - current_global_normalized_segment_start_tau) / 2 * current_local_tau_value_at_node + \
                                   (current_global_normalized_segment_end_tau + current_global_normalized_segment_start_tau) / 2
            
            physical_time_at_current_node: float = (solved_terminal_time - solved_initial_time) / 2 * current_global_normalized_tau_value_at_node + (solved_terminal_time + solved_initial_time) / 2

            is_overall_last_state_trajectory_point = (mesh_interval_index == num_mesh_intervals - 1 and local_state_approximation_node_index == num_collocation_nodes_current_interval)
            if abs(physical_time_at_current_node - last_time_point_added_to_state_trajectory) > 1e-9 or is_overall_last_state_trajectory_point or not state_trajectory_times:
                state_trajectory_times.append(physical_time_at_current_node)
                for state_variable_index in range(num_states):
                    state_trajectory_values_list_of_lists[state_variable_index].append(solved_state_at_local_approximation_nodes_current_interval[state_variable_index, local_state_approximation_node_index])
                last_time_point_added_to_state_trajectory = physical_time_at_current_node

    last_time_point_added_to_control_trajectory: float = -np.inf
    for mesh_interval_index in range(num_mesh_intervals):
        # solved_control_at_local_collocation_nodes_current_interval: np.ndarray = casadi_solution_object.value(casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index])
        if not hasattr(casadi_optimization_problem_object, 'control_at_local_collocation_nodes_all_intervals_variables') or mesh_interval_index >= len(casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables):
            print(f"Error: control_at_local_collocation_nodes_all_intervals_variables not found or incomplete in casadi_optimization_problem_object for interval {mesh_interval_index}.")
            continue
        solved_control_at_local_collocation_nodes_current_interval: np.ndarray = casadi_solution_object.value(casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index])

        local_collocation_nodes_tau_current_interval: np.ndarray = casadi_optimization_problem_object.metadata_local_collocation_nodes_tau[mesh_interval_index] # (num_collocation_nodes_in_current_interval,)
        num_collocation_nodes_current_interval: int = num_collocation_nodes_per_interval[mesh_interval_index]

        if num_controls == 1 and solved_control_at_local_collocation_nodes_current_interval.ndim == 1:
            solved_control_at_local_collocation_nodes_current_interval = solved_control_at_local_collocation_nodes_current_interval.reshape(1, -1)

        for local_collocation_node_index in range(num_collocation_nodes_current_interval):
            current_local_tau_value_at_node: float = local_collocation_nodes_tau_current_interval[local_collocation_node_index]

            current_global_normalized_segment_start_tau: float = global_normalized_mesh_nodes[mesh_interval_index]
            current_global_normalized_segment_end_tau: float = global_normalized_mesh_nodes[mesh_interval_index+1]
            current_global_normalized_tau_value_at_node: float = (current_global_normalized_segment_end_tau - current_global_normalized_segment_start_tau) / 2 * current_local_tau_value_at_node + \
                                   (current_global_normalized_segment_end_tau + current_global_normalized_segment_start_tau) / 2
            physical_time_at_current_node: float = (solved_terminal_time - solved_initial_time) / 2 * current_global_normalized_tau_value_at_node + (solved_terminal_time + solved_initial_time) / 2

            is_overall_last_control_trajectory_point = (mesh_interval_index == num_mesh_intervals - 1 and local_collocation_node_index == num_collocation_nodes_current_interval - 1)
            if abs(physical_time_at_current_node - last_time_point_added_to_control_trajectory) > 1e-9 or is_overall_last_control_trajectory_point or not control_trajectory_times:
                control_trajectory_times.append(physical_time_at_current_node)
                for control_variable_index in range(num_controls):
                    control_trajectory_values_list_of_lists[control_variable_index].append(solved_control_at_local_collocation_nodes_current_interval[control_variable_index, local_collocation_node_index])
                last_time_point_added_to_control_trajectory = physical_time_at_current_node
                
    return {
        "success": True,
        "message": "NLP solved successfully.",
        "initial_time_variable": solved_initial_time,
        "terminal_time_variable": solved_terminal_time,
        "objective": solved_objective_value,
        "integrals": solved_integral_values,
        "time_states": np.array(state_trajectory_times),
        "states": [np.array(s_traj) for s_traj in state_trajectory_values_list_of_lists], # List of arrays (one per state)
        "time_controls": np.array(control_trajectory_times),
        "controls": [np.array(c_traj) for c_traj in control_trajectory_values_list_of_lists], # List of arrays (one per control)
        "raw_solution": casadi_solution_object,
        "opti_object": casadi_optimization_problem_object,
        "num_collocation_nodes_per_interval": num_collocation_nodes_per_interval, # Number of collocation points in each interval
        "global_normalized_mesh_nodes": global_normalized_mesh_nodes # Global tau mesh points
    }


def solve_single_phase_radau_collocation(problem_definition: ProblemDefinition) -> Solution:
    """
    Solves a single-phase optimal control problem using a multiple-interval
    Radau Pseudospectral Method (RPM) based on GPOPS-II conventions.
    This implementation uses VARIABLE SHARING for state continuity across mesh intervals.

    Args:
        problem_definition: A dictionary containing the problem definition. 
                     (See original docstring for full list of keys)
                     Key modification: Can now include 'global_normalized_mesh_nodes'.
    Returns:
        A dictionary containing the solution results.
    """
    opti = ca.Opti()

    # --- Extract Problem Parameters ---
    num_states: int = problem_definition['num_states']
    num_controls: int = problem_definition['num_controls']
    num_integrals: int = problem_definition.get('num_integrals', 0)

    if 'collocation_points_per_interval' not in problem_definition:
        raise ValueError("problem_definition must include 'collocation_points_per_interval'.")
    num_collocation_nodes_per_interval: List[int] = problem_definition['collocation_points_per_interval']
    if not isinstance(num_collocation_nodes_per_interval, list) or not all(isinstance(n, int) and n > 0 for n in num_collocation_nodes_per_interval):
        raise ValueError("'collocation_points_per_interval' must be a list of positive integers.")
    num_mesh_intervals: int = len(num_collocation_nodes_per_interval) # Number of mesh intervals

    # User-defined functions
    dynamics_function = problem_definition['dynamics_function']
    objective_function = problem_definition['objective_function']
    path_constraints_function = problem_definition.get('path_constraints_function', None)
    event_constraints_function = problem_definition.get('event_constraints_function', None)
    integral_integrand_function = problem_definition.get('integral_integrand_function', None)
    problem_parameters: Dict[str, Any] = problem_definition.get('problem_parameters', {})

    # --- Decision Variables ---
    initial_time_variable: ca.MX = opti.variable()
    terminal_time_variable: ca.MX = opti.variable()
    opti.subject_to(initial_time_variable >= problem_definition['t0_bounds'][0])
    opti.subject_to(initial_time_variable <= problem_definition['t0_bounds'][1])
    opti.subject_to(terminal_time_variable >= problem_definition['tf_bounds'][0])
    opti.subject_to(terminal_time_variable <= problem_definition['tf_bounds'][1])
    opti.subject_to(terminal_time_variable > initial_time_variable + 1e-6) 

    # --- Mesh Definition ---
    # Global mesh nodes in the computational domain tau_global in [-1, 1]
    # There are num_mesh_intervals+1 such global mesh points.
    # Allow user to specify global_normalized_mesh_nodes for adaptive meshing.
    user_specified_global_normalized_mesh_nodes_input = problem_definition.get('global_normalized_mesh_nodes', None)
    global_normalized_mesh_nodes: np.ndarray

    if user_specified_global_normalized_mesh_nodes_input is not None:
        global_normalized_mesh_nodes = np.array(user_specified_global_normalized_mesh_nodes_input, dtype=float)
        if not (len(global_normalized_mesh_nodes) == num_mesh_intervals + 1 and \
                np.all(np.diff(global_normalized_mesh_nodes) > 1e-9) and \
                np.isclose(global_normalized_mesh_nodes[0], -1.0) and \
                np.isclose(global_normalized_mesh_nodes[-1], 1.0)):
            raise ValueError(
                "Provided 'global_normalized_mesh_nodes' must be sorted, have num_mesh_intervals+1 elements, "
                "start at -1.0, and end at +1.0, with positive interval lengths."
            )
    else:
        # Default to uniform distribution if not provided
        global_normalized_mesh_nodes = np.linspace(-1, 1, num_mesh_intervals + 1)
    
    # State Variables (Shared at global mesh points)
    state_at_global_mesh_nodes_variables: List[ca.MX] = [opti.variable(num_states) for _ in range(num_mesh_intervals + 1)]
    state_at_local_approximation_nodes_all_intervals_variables: List[ca.MX] = []
    state_at_interior_local_approximation_nodes_all_intervals_variables: List[Optional[ca.MX]] = []

    # Control Variables
    control_at_local_collocation_nodes_all_intervals_variables: List[ca.MX] = []
    for mesh_interval_index in range(num_mesh_intervals):
        Nk_current_interval: int = num_collocation_nodes_per_interval[mesh_interval_index]
        U_k: ca.MX = opti.variable(num_controls, Nk_current_interval)
        control_at_local_collocation_nodes_all_intervals_variables.append(U_k)

    # Integral Variables
    integral_decision_variables: Optional[ca.MX] = None
    if num_integrals > 0:
        integral_decision_variables = opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
    
    accumulated_integral_expressions: List[ca.MX] = [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    local_state_approximation_nodes_tau_all_intervals: List[np.ndarray] = []
    local_collocation_nodes_tau_all_intervals: List[np.ndarray] = []

    # --- Assemble State Representations & Define Interval Constraints ---
    for mesh_interval_index in range(num_mesh_intervals):
        num_collocation_nodes_in_current_interval: int = num_collocation_nodes_per_interval[mesh_interval_index]
        current_interval_state_at_local_approximation_nodes_columns: List[ca.MX] = [ca.MX(num_states, 1) for _ in range(num_collocation_nodes_in_current_interval + 1)]
        current_interval_state_at_local_approximation_nodes_columns[0] = state_at_global_mesh_nodes_variables[mesh_interval_index]

        interior_local_state_approximation_nodes_variables_current_interval: Optional[ca.MX] = None
        if num_collocation_nodes_in_current_interval > 1:
            num_interior_local_state_approximation_nodes: int = num_collocation_nodes_in_current_interval - 1
            if num_interior_local_state_approximation_nodes > 0:
                interior_local_state_approximation_nodes_variables_current_interval = opti.variable(num_states, num_interior_local_state_approximation_nodes)
                for interior_node_index in range(num_interior_local_state_approximation_nodes):
                    current_interval_state_at_local_approximation_nodes_columns[interior_node_index + 1] = interior_local_state_approximation_nodes_variables_current_interval[:, interior_node_index]
        state_at_interior_local_approximation_nodes_all_intervals_variables.append(interior_local_state_approximation_nodes_variables_current_interval)
        current_interval_state_at_local_approximation_nodes_columns[num_collocation_nodes_in_current_interval] = state_at_global_mesh_nodes_variables[mesh_interval_index + 1]
        
        state_at_local_approximation_nodes_current_interval_variable: ca.MX = ca.horzcat(*current_interval_state_at_local_approximation_nodes_columns)
        state_at_local_approximation_nodes_all_intervals_variables.append(state_at_local_approximation_nodes_current_interval_variable)

        radau_basis_components: Dict[str, np.ndarray] = rpm_basis.compute_radau_collocation_components(num_collocation_nodes_in_current_interval)
        local_state_approximation_nodes_tau_current_interval: np.ndarray = radau_basis_components['state_approximation_nodes'].flatten()
        local_collocation_nodes_tau_current_interval: np.ndarray = radau_basis_components['collocation_nodes'].flatten()
        local_quadrature_weights_current_interval: np.ndarray = radau_basis_components['quadrature_weights'].flatten()
        differentiation_matrix_current_interval: ca.DM = ca.DM(radau_basis_components['differentiation_matrix'])

        local_state_approximation_nodes_tau_all_intervals.append(local_state_approximation_nodes_tau_current_interval)
        local_collocation_nodes_tau_all_intervals.append(local_collocation_nodes_tau_current_interval)

        state_at_local_approximation_nodes_current_interval_variable: ca.MX = state_at_local_approximation_nodes_all_intervals_variables[mesh_interval_index]
        control_at_local_collocation_nodes_current_interval_variable: ca.MX = control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index]
        state_derivative_with_respect_to_local_tau_at_collocation_nodes: ca.MX = ca.mtimes(state_at_local_approximation_nodes_current_interval_variable, differentiation_matrix_current_interval.T)

        current_global_normalized_segment_length: float = global_normalized_mesh_nodes[mesh_interval_index+1] - global_normalized_mesh_nodes[mesh_interval_index]
        # Ensure current_global_normalized_segment_length is positive, which should be guaranteed by the check above
        if current_global_normalized_segment_length <= 1e-9: # Add a small tolerance
             raise ValueError(f"Mesh interval {mesh_interval_index} has zero or negative length in global tau: {current_global_normalized_segment_length}")

        local_tau_to_time_scaling_factor: ca.MX = (terminal_time_variable - initial_time_variable) * current_global_normalized_segment_length / 4.0

        for i_colloc in range(num_collocation_nodes_in_current_interval):
            state_at_current_collocation_node: ca.MX = state_at_local_approximation_nodes_current_interval_variable[:, i_colloc]
            control_at_current_collocation_node: ca.MX = control_at_local_collocation_nodes_current_interval_variable[:, i_colloc]

            current_local_tau_value_at_collocation_node: float = local_collocation_nodes_tau_current_interval[i_colloc]
            current_global_normalized_tau_value_at_collocation_node: float = current_global_normalized_segment_length / 2 * current_local_tau_value_at_collocation_node + \
                                       (global_normalized_mesh_nodes[mesh_interval_index+1] + global_normalized_mesh_nodes[mesh_interval_index]) / 2
            physical_time_at_current_collocation_node: ca.MX = (terminal_time_variable - initial_time_variable) / 2 * current_global_normalized_tau_value_at_collocation_node + (terminal_time_variable + initial_time_variable) / 2

            state_derivative_right_hand_side_values_list: List[ca.MX] = dynamics_function(state_at_current_collocation_node, control_at_current_collocation_node, physical_time_at_current_collocation_node, problem_parameters)
            # Ensure state_derivative_right_hand_side_casadi_vector is a CasADi MX vector (column vector)
            if isinstance(state_derivative_right_hand_side_values_list, list):
                state_derivative_right_hand_side_casadi_vector = ca.vertcat(*state_derivative_right_hand_side_values_list) if state_derivative_right_hand_side_values_list else ca.MX(num_states,1) # Handle empty list for num_states=0 (unlikely)
            elif isinstance(state_derivative_right_hand_side_values_list, ca.MX) and state_derivative_right_hand_side_values_list.shape[1] == 1: # Already a column vector
                state_derivative_right_hand_side_casadi_vector = state_derivative_right_hand_side_values_list
            elif isinstance(state_derivative_right_hand_side_values_list, ca.MX) and state_derivative_right_hand_side_values_list.shape[0] == 1 and num_states > 1 : # Row vector, needs transpose for num_states > 1
                 state_derivative_right_hand_side_casadi_vector = state_derivative_right_hand_side_values_list.T
            elif isinstance(state_derivative_right_hand_side_values_list, ca.MX) and num_states == 1: # Scalar MX for num_states=1 is fine
                 state_derivative_right_hand_side_casadi_vector = state_derivative_right_hand_side_values_list
            else:
                raise TypeError(f"Dynamics function output type not supported: {type(state_derivative_right_hand_side_values_list)}")

            if state_derivative_right_hand_side_casadi_vector.shape[0] != num_states:
                 raise ValueError(f"Dynamics function output dimensions mismatch. Expected {num_states} states, got {state_derivative_right_hand_side_casadi_vector.shape[0]}.")


            opti.subject_to( state_derivative_with_respect_to_local_tau_at_collocation_nodes[:, i_colloc] == local_tau_to_time_scaling_factor * state_derivative_right_hand_side_casadi_vector )

            if path_constraints_function:
                path_constraint_evaluations_list: List[Dict[str,Any]] = path_constraints_function(state_at_current_collocation_node, control_at_current_collocation_node, physical_time_at_current_collocation_node, problem_parameters)
                if not isinstance(path_constraint_evaluations_list, list): path_constraint_evaluations_list = [path_constraint_evaluations_list]
                for path_constraint_item in path_constraint_evaluations_list:
                    if 'val' not in path_constraint_item: raise ValueError("Path constraint dict missing 'val' key.")
                    constraint_value_expression = path_constraint_item['val']
                    if 'min' in path_constraint_item: opti.subject_to(constraint_value_expression >= path_constraint_item['min'])
                    if 'max' in path_constraint_item: opti.subject_to(constraint_value_expression <= path_constraint_item['max'])
                    if 'equals' in path_constraint_item: opti.subject_to(constraint_value_expression == path_constraint_item['equals'])
        
        if num_integrals > 0 and integral_integrand_function:
            for integral_index in range(num_integrals):
                quadrature_sum_for_integral_j_current_interval: ca.MX = ca.MX(0)
                for i_colloc in range(num_collocation_nodes_in_current_interval):
                    state_at_current_collocation_node = state_at_local_approximation_nodes_current_interval_variable[:, i_colloc]
                    control_at_current_collocation_node = control_at_local_collocation_nodes_current_interval_variable[:, i_colloc]
                    
                    current_local_tau_value_at_collocation_node = local_collocation_nodes_tau_current_interval[i_colloc]
                    current_global_normalized_tau_value_at_collocation_node = current_global_normalized_segment_length / 2 * current_local_tau_value_at_collocation_node + \
                                        (global_normalized_mesh_nodes[mesh_interval_index+1] + global_normalized_mesh_nodes[mesh_interval_index]) / 2
                    physical_time_at_current_collocation_node = (terminal_time_variable - initial_time_variable) / 2 * current_global_normalized_tau_value_at_collocation_node + (terminal_time_variable + initial_time_variable) / 2
                    
                    current_quadrature_weight: float = local_quadrature_weights_current_interval[i_colloc]
                    integrand_value_at_collocation_node: ca.MX = integral_integrand_function(
                        state_at_current_collocation_node, control_at_current_collocation_node, physical_time_at_current_collocation_node, integral_index, problem_parameters
                    )
                    quadrature_sum_for_integral_j_current_interval += current_quadrature_weight * integrand_value_at_collocation_node
                accumulated_integral_expressions[integral_index] += local_tau_to_time_scaling_factor * quadrature_sum_for_integral_j_current_interval

    if num_integrals > 0 and integral_integrand_function:
        if num_integrals == 1:
            opti.subject_to(integral_decision_variables == accumulated_integral_expressions[0])
        else:
            for integral_index in range(num_integrals):
                opti.subject_to(integral_decision_variables[integral_index] == accumulated_integral_expressions[integral_index])

    initial_phase_state_variables: ca.MX = state_at_global_mesh_nodes_variables[0]
    terminal_phase_state_variables: ca.MX = state_at_global_mesh_nodes_variables[num_mesh_intervals]

    symbolic_objective_function_value: ca.MX = objective_function(initial_time_variable, terminal_time_variable, initial_phase_state_variables, terminal_phase_state_variables, integral_decision_variables, problem_parameters)
    opti.minimize(symbolic_objective_function_value)

    if event_constraints_function:
        event_constraint_evaluations_list: List[Dict[str,Any]] = event_constraints_function(initial_time_variable, terminal_time_variable, initial_phase_state_variables, terminal_phase_state_variables, integral_decision_variables, problem_parameters)
        if not isinstance(event_constraint_evaluations_list, list): event_constraint_evaluations_list = [event_constraint_evaluations_list]
        for event_constraint_item in event_constraint_evaluations_list:
            if 'val' not in event_constraint_item: raise ValueError("Event constraint dict missing 'val' key.")
            constraint_value_expression = event_constraint_item['val']
            if 'min' in event_constraint_item: opti.subject_to(constraint_value_expression >= event_constraint_item['min'])
            if 'max' in event_constraint_item: opti.subject_to(constraint_value_expression <= event_constraint_item['max'])
            if 'equals' in event_constraint_item: opti.subject_to(constraint_value_expression == event_constraint_item['equals'])

    # --- Initial Guess ---
    if 'initial_guess' in problem_definition:
        initial_guess_data: Dict[str, Any] = problem_definition['initial_guess']
        if 'initial_time_variable' in initial_guess_data and initial_guess_data['initial_time_variable'] is not None: opti.set_initial(initial_time_variable, initial_guess_data['initial_time_variable'])
        if 'terminal_time_variable' in initial_guess_data and initial_guess_data['terminal_time_variable'] is not None: opti.set_initial(terminal_time_variable, initial_guess_data['terminal_time_variable'])

        # Initial initial_guess_data for states
        if 'states' in initial_guess_data and initial_guess_data['states'] is not None:
            # initial_guess_data['states'] should be a list of arrays, one for each interval.
            # Each array initial_guess_data['states'][mesh_interval_index] is (num_states, num_collocation_nodes_per_interval[mesh_interval_index]+1)
            # OR it can be a single array for the whole trajectory, which we'd need to map.
            # For now, assume it's per interval as constructed by _extract_and_format_solution.

            # 1. Guess for SHARED state_at_global_mesh_nodes_variables
            if num_mesh_intervals > 0 and len(initial_guess_data['states']) > 0:
                # Guess for state at the very start of the phase (global mesh point 0)
                # This comes from the start of the initial_guess_data for the first interval.
                if isinstance(initial_guess_data['states'][0], np.ndarray) and initial_guess_data['states'][0].shape[0] == num_states:
                     opti.set_initial(state_at_global_mesh_nodes_variables[0], initial_guess_data['states'][0][:, 0])
                else:
                    print(f"Warning: Initial initial_guess_data format for states[0] not as expected. Skipping initial_guess_data for state_at_global_mesh_nodes_variables[0].")

                # Guess for states at global mesh points 1 to num_mesh_intervals.
                # state_at_global_mesh_nodes_variables[mesh_interval_index+1] gets its initial_guess_data from the END of interval mesh_interval_index.
                for mesh_interval_index in range(num_mesh_intervals):
                    if mesh_interval_index < len(initial_guess_data['states']) and isinstance(initial_guess_data['states'][mesh_interval_index], np.ndarray):
                        num_collocation_nodes_in_current_guess_interval: int = initial_guess_data['states'][mesh_interval_index].shape[1] -1 # num_collocation_nodes_in_current_interval from initial_guess_data
                        if initial_guess_data['states'][mesh_interval_index].shape[0] == num_states and num_collocation_nodes_in_current_guess_interval >=0 :
                            opti.set_initial(state_at_global_mesh_nodes_variables[mesh_interval_index + 1], initial_guess_data['states'][mesh_interval_index][:, num_collocation_nodes_in_current_guess_interval]) # Use last column
                        else:
                             print(f"Warning: Initial initial_guess_data for states[{mesh_interval_index}] has incorrect shape or num_collocation_nodes_in_current_interval. Skipping initial_guess_data for state_at_global_mesh_nodes_variables[{mesh_interval_index+1}].")
                    else:
                        print(f"Warning: Not enough state guesses or incorrect format for interval {mesh_interval_index}. Skipping initial_guess_data for state_at_global_mesh_nodes_variables[{mesh_interval_index+1}].")
            
            # 2. Guess for INTERVAL-INTERIOR state variables
            for mesh_interval_index in range(num_mesh_intervals):
                num_collocation_nodes_for_problem_interval: int = num_collocation_nodes_per_interval[mesh_interval_index] # num_collocation_nodes_in_current_interval for the current problem setup
                if num_collocation_nodes_for_problem_interval > 1: # Interior states exist only if num_collocation_nodes_for_problem_interval > 1
                    num_interior_local_state_approximation_nodes_for_problem: int = num_collocation_nodes_for_problem_interval - 1
                    if num_interior_local_state_approximation_nodes_for_problem > 0 and \
                       mesh_interval_index < len(state_at_interior_local_approximation_nodes_all_intervals_variables) and \
                       state_at_interior_local_approximation_nodes_all_intervals_variables[mesh_interval_index] is not None and \
                       mesh_interval_index < len(initial_guess_data['states']) and \
                       isinstance(initial_guess_data['states'][mesh_interval_index], np.ndarray) and \
                       initial_guess_data['states'][mesh_interval_index].shape[0] == num_states:
                        
                        # Guess for interior states are columns 1 to num_collocation_nodes_for_problem_interval-1 from initial_guess_data['states'][mesh_interval_index]
                        # The initial_guess_data interval might have a different num_collocation_nodes_in_current_interval, so we take what's available.
                        num_collocation_nodes_in_guess_interval = initial_guess_data['states'][mesh_interval_index].shape[1] - 1
                        
                        # We need to provide a initial_guess_data of shape (num_states, num_interior_local_state_approximation_nodes_for_problem)
                        # We take columns 1 to min(num_collocation_nodes_for_problem_interval, num_collocation_nodes_in_guess_interval) from the initial_guess_data
                        num_columns_to_take_from_guess = min(num_interior_local_state_approximation_nodes_for_problem, num_collocation_nodes_in_guess_interval -1 if num_collocation_nodes_in_guess_interval > 0 else 0)

                        if num_columns_to_take_from_guess > 0:
                            actual_guess_data_slice = initial_guess_data['states'][mesh_interval_index][:, 1 : 1 + num_columns_to_take_from_guess]
                            
                            # If num_columns_to_take_from_guess < num_interior_local_state_approximation_nodes_for_problem, we need to pad
                            if actual_guess_data_slice.shape[1] < num_interior_local_state_approximation_nodes_for_problem:
                                padding_for_guess_data = np.tile(actual_guess_data_slice[:, -1:], (1, num_interior_local_state_approximation_nodes_for_problem - actual_guess_data_slice.shape[1]))
                                final_guess_data_slice_for_initialization = np.hstack((actual_guess_data_slice, padding_for_guess_data))
                            else:
                                final_guess_data_slice_for_initialization = actual_guess_data_slice[:, :num_interior_local_state_approximation_nodes_for_problem]

                            if final_guess_data_slice_for_initialization.shape == (num_states, num_interior_local_state_approximation_nodes_for_problem):
                                opti.set_initial(state_at_interior_local_approximation_nodes_all_intervals_variables[mesh_interval_index], final_guess_data_slice_for_initialization)
                            else:
                                print(f"Warning: Shape mismatch after processing INTERIOR state initial_guess_data for interval {mesh_interval_index}. Skipping.")
                        elif num_interior_local_state_approximation_nodes_for_problem > 0 : # Need initial_guess_data but can't form one
                             print(f"Warning: Cannot form INTERIOR state initial_guess_data for interval {mesh_interval_index} from provided initial_guess_data['states'][{mesh_interval_index}]. Skipping.")


        if 'controls' in initial_guess_data and initial_guess_data['controls'] is not None:
            for mesh_interval_index in range(num_mesh_intervals):
                if mesh_interval_index < len(initial_guess_data['controls']) and isinstance(initial_guess_data['controls'][mesh_interval_index], np.ndarray):
                    num_collocation_nodes_for_problem_current_interval: int = num_collocation_nodes_per_interval[mesh_interval_index]
                    expected_shape_for_guess: Tuple[int, int] = (num_controls, num_collocation_nodes_for_problem_current_interval)
                    
                    current_raw_control_guess_for_interval_k: np.ndarray = initial_guess_data['controls'][mesh_interval_index]
                    
                    # Ensure initial_guess_data is 2D (num_controls, Nk_guess)
                    if num_controls == 1 and current_raw_control_guess_for_interval_k.ndim == 1:
                        current_raw_control_guess_for_interval_k = current_raw_control_guess_for_interval_k.reshape(1, -1)
                    
                    if current_raw_control_guess_for_interval_k.shape[0] == num_controls:
                        num_collocation_nodes_in_guess_interval = current_raw_control_guess_for_interval_k.shape[1]
                        
                        # Take min(num_collocation_nodes_for_problem_current_interval, num_collocation_nodes_in_guess_interval) columns
                        num_columns_to_take_from_guess = min(num_collocation_nodes_for_problem_current_interval, num_collocation_nodes_in_guess_interval)
                        
                        if num_columns_to_take_from_guess > 0:
                            final_guess_data_slice_for_initialization = current_raw_control_guess_for_interval_k[:, :num_columns_to_take_from_guess]
                            
                            # If num_columns_to_take_from_guess < num_collocation_nodes_for_problem_current_interval, pad by repeating the last column
                            if final_guess_data_slice_for_initialization.shape[1] < num_collocation_nodes_for_problem_current_interval:
                                padding_for_guess_data = np.tile(final_guess_data_slice_for_initialization[:, -1:], (1, num_collocation_nodes_for_problem_current_interval - final_guess_data_slice_for_initialization.shape[1]))
                                final_guess_data_slice_for_initialization = np.hstack((final_guess_data_slice_for_initialization, padding_for_guess_data))

                            if final_guess_data_slice_for_initialization.shape == expected_shape_for_guess:
                                opti.set_initial(control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index], final_guess_data_slice_for_initialization)
                            else:
                                print(f"Warning: Shape mismatch after processing control initial_guess_data for interval {mesh_interval_index}. Skipping.")
                        elif num_collocation_nodes_for_problem_current_interval > 0:
                             print(f"Warning: Cannot form control initial_guess_data for interval {mesh_interval_index} from provided initial_guess_data['controls'][{mesh_interval_index}]. Skipping.")
                    else:
                        print(f"Warning: Initial initial_guess_data for controls in interval {mesh_interval_index} has incorrect num_states rows. Skipping.")
        
        if 'integrals' in initial_guess_data and initial_guess_data['integrals'] is not None and integral_decision_variables is not None:
            # (Original logic for integral initial_guess_data can be kept or refined if needed)
            initial_guess_for_integrals = initial_guess_data['integrals']
            if num_integrals == 1:
                value_to_set_for_initial_guess: Optional[float] = None
                if isinstance(initial_guess_for_integrals, (int, float)): value_to_set_for_initial_guess = float(initial_guess_for_integrals)
                elif isinstance(initial_guess_for_integrals, (list, np.ndarray)) and np.array(initial_guess_for_integrals).size == 1:
                    value_to_set_for_initial_guess = float(np.array(initial_guess_for_integrals).item())
                if value_to_set_for_initial_guess is not None: opti.set_initial(integral_decision_variables, value_to_set_for_initial_guess)
                else: print(f"Warning: Initial initial_guess_data for single integral format. Got {initial_guess_for_integrals}. Skipping.")
            elif isinstance(initial_guess_for_integrals, (list, np.ndarray)) and np.array(initial_guess_for_integrals).size == num_integrals:
                opti.set_initial(integral_decision_variables, np.array(initial_guess_for_integrals).flatten())
            else: print(f"Warning: Initial initial_guess_data for integrals format/length. Got {initial_guess_for_integrals}. Skipping.")


    # --- Solve the NLP ---
    solver_options: Dict[str, Any] = problem_definition.get("solver_options", {})
    if solver_options is None: solver_options = {} 
    opti.solver("ipopt", solver_options)

    opti.initial_time_variable_reference = initial_time_variable
    opti.terminal_time_variable_reference = terminal_time_variable
    if integral_decision_variables is not None: opti.integral_variables_object_reference = integral_decision_variables
    opti.state_at_local_approximation_nodes_all_intervals_variables = state_at_local_approximation_nodes_all_intervals_variables 
    opti.control_at_local_collocation_nodes_all_intervals_variables = control_at_local_collocation_nodes_all_intervals_variables
    opti.metadata_local_state_approximation_nodes_tau = local_state_approximation_nodes_tau_all_intervals
    opti.metadata_local_collocation_nodes_tau = local_collocation_nodes_tau_all_intervals
    opti.metadata_global_normalized_mesh_nodes = global_normalized_mesh_nodes # Store the actual mesh used
    opti.symbolic_objective_function_reference = symbolic_objective_function_value

    solver_solution_status_object: Optional[ca.OptiSol] = None
    try:
        solver_solution_status_object = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        return _extract_and_format_solution(solver_solution_status_object, opti, problem_definition, num_collocation_nodes_per_interval, global_normalized_mesh_nodes)
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        solution_failure_dictionary = _extract_and_format_solution(solver_solution_status_object, opti, problem_definition, num_collocation_nodes_per_interval, global_normalized_mesh_nodes)
        solution_failure_dictionary["success"] = False 
        solution_failure_dictionary["message"] = f"Solver runtime error: {e}"
        try:
            if hasattr(opti, 'debug') and opti.debug is not None:
                if initial_time_variable is not None: solution_failure_dictionary["initial_time_variable"] = opti.debug.value(initial_time_variable)
                if terminal_time_variable is not None: solution_failure_dictionary["terminal_time_variable"] = opti.debug.value(terminal_time_variable)
        except Exception as debug_e:
            print(f"  Could not retrieve debug values after solver error: {debug_e}")
        return solution_failure_dictionary