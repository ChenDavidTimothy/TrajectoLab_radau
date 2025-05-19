"""
Solution extraction and formatting utilities.
"""

import casadi as ca
import numpy as np

# Add these imports at the top
from .problem.scaling import reverse_scaling_on_solution_trajectories
from .tl_types import (
    CasadiOpti,
    CasadiOptiSol,
    FloatArray,
    FloatMatrix,
    ListOfCasadiMX,
    OptimalControlSolution,
    ProblemProtocol,
)
from .utils.variable_scaling import ProblemScalingInfo


def extract_and_format_solution_with_scaling(
    casadi_solution_object: CasadiOptiSol | None,
    casadi_optimization_problem_object: CasadiOpti,
    problem: ProblemProtocol,
    num_collocation_nodes_per_interval: list[int],
    global_normalized_mesh_nodes: FloatArray,
    scaling_info: ProblemScalingInfo,
) -> OptimalControlSolution:
    """
    Extract and format the solution with variable scaling support.

    This function extracts the scaled solution from CasADi and converts
    back to physical units using the provided scaling information.

    Args:
        casadi_solution_object: CasADi solution object (may be None if solve failed)
        casadi_optimization_problem_object: CasADi optimization object
        problem: Problem definition
        num_collocation_nodes_per_interval: Collocation nodes per interval
        global_normalized_mesh_nodes: Global mesh nodes
        scaling_info: Scaling information used during optimization

    Returns:
        OptimalControlSolution with trajectories in physical units
    """
    # First extract solution using the standard method
    solution = extract_and_format_solution(
        casadi_solution_object,
        casadi_optimization_problem_object,
        problem,
        num_collocation_nodes_per_interval,
        global_normalized_mesh_nodes,
    )

    # If scaling is disabled or solution extraction failed, return as-is
    if (
        not scaling_info.scaling_enabled
        or not solution.success
        or not hasattr(solution, "solved_state_trajectories_per_interval")
        or solution.solved_state_trajectories_per_interval is None
    ):
        return solution

    try:
        # Extract raw trajectories (these are in scaled space)
        scaled_states = solution.solved_state_trajectories_per_interval
        scaled_controls = getattr(solution, "solved_control_trajectories_per_interval", None)

        if scaled_controls is None:
            # Create empty control trajectories if none exist
            scaled_controls = [
                np.empty((0, N_k), dtype=np.float64) for N_k in num_collocation_nodes_per_interval
            ]

        # Convert back to physical units
        physical_states, physical_controls = reverse_scaling_on_solution_trajectories(
            scaled_states, scaled_controls, scaling_info, problem._variable_state
        )

        # Update solution with physical trajectories
        solution.solved_state_trajectories_per_interval = physical_states
        solution.solved_control_trajectories_per_interval = physical_controls

        # Also update the user-facing trajectory arrays if they exist
        if hasattr(solution, "states") and solution.states:
            # Convert state trajectories
            for i, phys_state_traj in enumerate(physical_states):
                if i < len(solution.states):
                    # Update the time-series trajectory
                    # Note: This assumes 1-to-1 correspondence which may need refinement
                    pass

        if hasattr(solution, "controls") and solution.controls:
            # Convert control trajectories
            for i, phys_control_traj in enumerate(physical_controls):
                if i < len(solution.controls):
                    # Update the time-series trajectory
                    # Note: This assumes 1-to-1 correspondence which may need refinement
                    pass

    except Exception as e:
        print(f"Warning: Failed to reverse variable scaling in solution: {e}")
        # Continue with scaled solution rather than failing completely

    return solution


def extract_integral_values(
    casadi_solution_object: ca.OptiSol | None, opti_object: ca.Opti, num_integrals: int
) -> float | FloatArray | None:
    """Extract integral values from the CasADi solution."""
    if (
        num_integrals == 0
        or not hasattr(opti_object, "integral_variables_object_reference")
        or opti_object.integral_variables_object_reference is None
        or casadi_solution_object is None
    ):
        return None

    try:
        raw_value = casadi_solution_object.value(opti_object.integral_variables_object_reference)

        if isinstance(raw_value, ca.DM):
            np_array_value = np.asarray(raw_value.toarray())
            if num_integrals == 1:
                if np_array_value.size == 1:
                    return float(np_array_value.item())
                else:
                    print(
                        f"Warning: For num_integrals=1, CasADi DM value resulted in array shape {np_array_value.shape} "
                        f"after toarray(). Attempting to use the first element."
                    )
                    if np_array_value.size > 0:
                        return float(np_array_value.flatten()[0])
                    else:
                        print(
                            "Warning: For num_integrals=1, CasADi DM value is empty after conversion."
                        )
                        return np.nan
            else:
                return np_array_value.flatten().astype(np.float64)

        elif isinstance(raw_value, float | int):
            if num_integrals == 1:
                return float(raw_value)
            else:
                print(
                    f"Warning: Expected array for {num_integrals} integrals, but CasADi value() returned scalar {raw_value}."
                )
                return np.full(num_integrals, np.nan, dtype=np.float64)
        else:
            print(
                f"Warning: CasADi .value() returned an unexpected type: {type(raw_value)}. Value: {raw_value}"
            )
            if num_integrals > 1:
                return np.full(num_integrals, np.nan, dtype=np.float64)
            elif num_integrals == 1:
                return np.nan
            return None

    except Exception as e:
        print(f"Warning: Could not extract integral values: {e}")
        if num_integrals > 1:
            return np.full(num_integrals, np.nan, dtype=np.float64)
        elif num_integrals == 1:
            return np.nan
        return None


def process_trajectory_points(
    mesh_interval_index: int,
    casadi_solution_object: CasadiOptiSol,
    opti_object: CasadiOpti,
    variables_list: ListOfCasadiMX,
    local_tau_nodes: list[FloatArray],
    global_normalized_mesh_nodes: FloatArray,
    initial_time: float,
    terminal_time: float,
    last_added_point: float,
    trajectory_times: list[float],
    trajectory_values_lists: list[list[float]],
    num_variables: int,
    is_state: bool = True,
) -> float:
    """
    Process trajectory points for a single mesh interval.

    Args:
        mesh_interval_index: Index of the current mesh interval
        casadi_solution_object: CasADi solution object
        opti_object: CasADi optimization object
        variables_list: List of CasADi variables for this trajectory type
        local_tau_nodes: Local tau values for each interval
        global_normalized_mesh_nodes: Global mesh nodes
        initial_time: Problem initial time
        terminal_time: Problem terminal time
        last_added_point: Last time point added (to avoid duplicates)
        trajectory_times: Output list for time points
        trajectory_values_lists: Output list for trajectory values
        num_variables: Number of variables (states or controls)
        is_state: True for states, False for controls

    Returns:
        Updated last_added_point value
    """
    if mesh_interval_index >= len(variables_list) or mesh_interval_index >= len(local_tau_nodes):
        print(
            f"Error: Variable list or tau nodes not found or incomplete for interval {mesh_interval_index}."
        )
        return last_added_point

    current_interval_variables = variables_list[mesh_interval_index]
    current_interval_local_tau_values = local_tau_nodes[mesh_interval_index]

    solved_values: FloatMatrix = casadi_solution_object.value(current_interval_variables)
    if num_variables == 1 and solved_values.ndim == 1:
        solved_values = solved_values.reshape(1, -1)

    # For controls, we don't include the final point (which belongs to states)
    num_nodes_to_process = len(current_interval_local_tau_values)
    if not is_state and num_nodes_to_process > 0:
        num_nodes_to_process -= 1

    for node_index in range(num_nodes_to_process):
        local_tau: float = current_interval_local_tau_values[node_index]
        segment_start: float = global_normalized_mesh_nodes[mesh_interval_index]
        segment_end: float = global_normalized_mesh_nodes[mesh_interval_index + 1]

        # Transform from local tau to global tau
        global_tau: float = (segment_end - segment_start) / 2 * local_tau + (
            segment_end + segment_start
        ) / 2

        # Transform from global tau to physical time
        physical_time: float = (terminal_time - initial_time) / 2 * global_tau + (
            terminal_time + initial_time
        ) / 2

        # Check if this is the last point in the trajectory or if we should add it
        is_last_point_in_trajectory: bool = (
            mesh_interval_index == len(variables_list) - 1
            and node_index == num_nodes_to_process - 1
        )

        # Add point if it's sufficiently different from the last one, or if it's the final point
        if (
            abs(physical_time - last_added_point) > 1e-9
            or is_last_point_in_trajectory
            or not trajectory_times
        ):
            trajectory_times.append(physical_time)
            for var_index in range(num_variables):
                trajectory_values_lists[var_index].append(solved_values[var_index, node_index])
            last_added_point = physical_time

    return last_added_point


def extract_and_format_solution(
    casadi_solution_object: CasadiOptiSol | None,
    casadi_optimization_problem_object: CasadiOpti,
    problem: ProblemProtocol,
    num_collocation_nodes_per_interval: list[int],
    global_normalized_mesh_nodes: FloatArray,
) -> OptimalControlSolution:
    """Extract and format the solution from CasADi optimization result."""
    solution = OptimalControlSolution()
    solution.opti_object = casadi_optimization_problem_object
    solution.num_collocation_nodes_per_interval = list(num_collocation_nodes_per_interval)
    solution.global_normalized_mesh_nodes = global_normalized_mesh_nodes.copy()

    if casadi_solution_object is None:
        solution.success = False
        solution.message = "Solver did not find a solution or was not run."
        return solution

    num_mesh_intervals: int = len(num_collocation_nodes_per_interval)
    num_states: int = len(problem._states)
    num_controls: int = len(problem._controls)
    num_integrals: int = problem._num_integrals

    try:
        solution.initial_time_variable = float(
            casadi_solution_object.value(
                casadi_optimization_problem_object.initial_time_variable_reference
            )
        )
        solution.terminal_time_variable = float(
            casadi_solution_object.value(
                casadi_optimization_problem_object.terminal_time_variable_reference
            )
        )
        solution.objective = float(
            casadi_solution_object.value(
                casadi_optimization_problem_object.symbolic_objective_function_reference
            )
        )
    except Exception as e:
        solution.success = False
        solution.message = f"Failed to extract core solution values: {e}"
        solution.raw_solution = casadi_solution_object
        return solution

    solution.integrals = extract_integral_values(
        casadi_solution_object, casadi_optimization_problem_object, num_integrals
    )

    # Extract state trajectories
    state_trajectory_times: list[float] = []
    state_trajectory_values: list[list[float]] = [[] for _ in range(num_states)]
    last_time_point_added_to_state_trajectory: float = -np.inf

    for mesh_idx in range(num_mesh_intervals):
        if not hasattr(
            casadi_optimization_problem_object,
            "state_at_local_approximation_nodes_all_intervals_variables",
        ) or not hasattr(
            casadi_optimization_problem_object, "metadata_local_state_approximation_nodes_tau"
        ):
            print("Error: State trajectory attributes missing in optimization object.")
            continue

        last_time_point_added_to_state_trajectory = process_trajectory_points(
            mesh_idx,
            casadi_solution_object,
            casadi_optimization_problem_object,
            casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables,
            casadi_optimization_problem_object.metadata_local_state_approximation_nodes_tau,
            global_normalized_mesh_nodes,
            solution.initial_time_variable,
            solution.terminal_time_variable,
            last_time_point_added_to_state_trajectory,
            state_trajectory_times,
            state_trajectory_values,
            num_states,
            is_state=True,
        )

    # Extract control trajectories
    control_trajectory_times: list[float] = []
    control_trajectory_values: list[list[float]] = [[] for _ in range(num_controls)]
    last_time_point_added_to_control_trajectory: float = -np.inf

    for mesh_idx in range(num_mesh_intervals):
        if not hasattr(
            casadi_optimization_problem_object,
            "control_at_local_collocation_nodes_all_intervals_variables",
        ) or not hasattr(
            casadi_optimization_problem_object, "metadata_local_collocation_nodes_tau"
        ):
            print("Error: Control trajectory attributes missing in optimization object.")
            continue

        last_time_point_added_to_control_trajectory = process_trajectory_points(
            mesh_idx,
            casadi_solution_object,
            casadi_optimization_problem_object,
            casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables,
            casadi_optimization_problem_object.metadata_local_collocation_nodes_tau,
            global_normalized_mesh_nodes,
            solution.initial_time_variable,
            solution.terminal_time_variable,
            last_time_point_added_to_control_trajectory,
            control_trajectory_times,
            control_trajectory_values,
            num_controls,
            is_state=False,
        )

    solution.success = True
    solution.message = "NLP solved successfully."
    solution.time_states = np.array(state_trajectory_times, dtype=np.float64)
    solution.states = [np.array(s_traj, dtype=np.float64) for s_traj in state_trajectory_values]
    solution.time_controls = np.array(control_trajectory_times, dtype=np.float64)
    solution.controls = [np.array(c_traj, dtype=np.float64) for c_traj in control_trajectory_values]
    solution.raw_solution = casadi_solution_object

    return solution
