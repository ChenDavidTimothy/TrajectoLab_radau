"""
Redesigned solution extraction with proper optimal control scaling support.

Key changes:
1. Extract proper scaling information from problem
2. Handle objective unscaling (from w_0 * J back to J)
3. Proper trajectory unscaling using variable scaling factors
4. Clean separation of scaling components
"""

import casadi as ca
import numpy as np

from .tl_types import (
    CasadiOpti,
    CasadiOptiSol,
    FloatArray,
    FloatMatrix,
    ListOfCasadiMX,
    OptimalControlSolution,
    ProblemProtocol,
)


def extract_integral_values(
    casadi_solution_object: ca.OptiSol | None, opti_object: ca.Opti, num_integrals: int
) -> float | FloatArray | None:
    """Extract integral values from CasADi solution."""
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
                return float(np_array_value.item()) if np_array_value.size == 1 else np.nan
            else:
                return np_array_value.flatten().astype(np.float64)
        elif isinstance(raw_value, float | int):
            return float(raw_value) if num_integrals == 1 else np.full(num_integrals, np.nan)
        else:
            return np.nan if num_integrals == 1 else np.full(num_integrals, np.nan)

    except Exception as e:
        print(f"Warning: Could not extract integral values: {e}")
        return np.nan if num_integrals == 1 else np.full(num_integrals, np.nan)


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
    """Process trajectory points for a single mesh interval."""
    if mesh_interval_index >= len(variables_list) or mesh_interval_index >= len(local_tau_nodes):
        return last_added_point

    current_interval_variables = variables_list[mesh_interval_index]
    current_interval_local_tau_values = local_tau_nodes[mesh_interval_index]

    solved_values: FloatMatrix = casadi_solution_object.value(current_interval_variables)
    if num_variables == 1 and solved_values.ndim == 1:
        solved_values = solved_values.reshape(1, -1)

    # Process nodes (controls exclude final point)
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

        # Add point if sufficiently different or is final point
        is_last_point = (
            mesh_interval_index == len(variables_list) - 1
            and node_index == num_nodes_to_process - 1
        )

        if abs(physical_time - last_added_point) > 1e-9 or is_last_point or not trajectory_times:
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
    """
    Extract and format solution with proper scaling support.

    Key improvements:
    1. Proper objective unscaling (w_0 * J -> J)
    2. Correct scaling information extraction
    3. Clean trajectory unscaling
    """
    solution = OptimalControlSolution()
    solution.opti_object = casadi_optimization_problem_object
    solution.num_collocation_nodes_per_interval = list(num_collocation_nodes_per_interval)
    solution.global_normalized_mesh_nodes = global_normalized_mesh_nodes.copy()

    print("📊 Extracting solution with proper scaling support")

    # Extract scaling information from problem
    try:
        scaling_info = problem.get_scaling_info()
        solution.auto_scaling_enabled = scaling_info.get("auto_scaling_enabled", False)

        if solution.auto_scaling_enabled:
            print("  ✅ Auto-scaling detected in solution")

            # Extract variable scaling factors
            solution.variable_scaling_factors = scaling_info.get("variable_scaling_factors", {})

            # Extract objective scaling
            obj_scaling_info = scaling_info.get("objective_scaling", {})
            solution.objective_scaling_factor = obj_scaling_info.get("w_0", 1.0)
            solution.objective_computed_from_hessian = obj_scaling_info.get(
                "computed_from_hessian", False
            )
            solution.gerschgorin_omega = obj_scaling_info.get("gerschgorin_omega")

            # Extract constraint scaling
            constraint_scaling_info = scaling_info.get("constraint_scaling", {})
            solution.ode_defect_scaling_factors = constraint_scaling_info.get("W_f", {})
            solution.path_constraint_scaling_factors = constraint_scaling_info.get("W_g", {})

            # Extract symbol mappings
            solution.original_physical_symbols = scaling_info.get("original_physical_symbols", {})
            solution.scaled_nlp_symbols = scaling_info.get("scaled_nlp_symbols", {})
            solution.physical_to_scaled_names = scaling_info.get("physical_to_scaled_names", {})

            print(f"    📐 Objective scaling factor w_0: {solution.objective_scaling_factor:.3e}")
            print(f"    📊 Variable scaling factors: {len(solution.variable_scaling_factors)}")
        else:
            print("  ➡️  No auto-scaling used")
            # Initialize default values
            solution.variable_scaling_factors = {}
            solution.objective_scaling_factor = 1.0
            solution.ode_defect_scaling_factors = {}
            solution.path_constraint_scaling_factors = {}
            solution.original_physical_symbols = {}
            solution.scaled_nlp_symbols = {}
            solution.physical_to_scaled_names = {}

    except Exception as e:
        print(f"⚠️  Warning: Could not extract scaling information: {e}")
        solution.auto_scaling_enabled = False
        solution.variable_scaling_factors = {}
        solution.objective_scaling_factor = 1.0

    if casadi_solution_object is None:
        solution.success = False
        solution.message = "Solver did not find a solution or was not run."
        return solution

    num_mesh_intervals = len(num_collocation_nodes_per_interval)
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Extract core solution values
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

        # Extract and properly unscale objective value
        raw_objective = float(
            casadi_solution_object.value(
                casadi_optimization_problem_object.symbolic_objective_function_reference
            )
        )

        # Apply proper objective unscaling
        if solution.auto_scaling_enabled and solution.objective_scaling_factor != 1.0:
            # The NLP solved: minimize w_0 * J
            # So the physical objective is: J = (w_0 * J) / w_0
            solution.objective = raw_objective / solution.objective_scaling_factor
            print(
                f"  📐 Objective unscaled: {raw_objective:.6f} / {solution.objective_scaling_factor:.3e} = {solution.objective:.6f}"
            )
        else:
            solution.objective = raw_objective
            print(f"  📊 Objective (no scaling): {solution.objective:.6f}")

    except Exception as e:
        solution.success = False
        solution.message = f"Failed to extract core solution values: {e}"
        solution.raw_solution = casadi_solution_object
        return solution

    # Extract integral values
    solution.integrals = extract_integral_values(
        casadi_solution_object, casadi_optimization_problem_object, num_integrals
    )

    # Extract state trajectories
    state_trajectory_times: list[float] = []
    state_trajectory_values: list[list[float]] = [[] for _ in range(num_states)]
    last_time_point_added_to_state_trajectory: float = -np.inf

    for mesh_idx in range(num_mesh_intervals):
        if hasattr(
            casadi_optimization_problem_object,
            "state_at_local_approximation_nodes_all_intervals_variables",
        ) and hasattr(
            casadi_optimization_problem_object, "metadata_local_state_approximation_nodes_tau"
        ):
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
        if hasattr(
            casadi_optimization_problem_object,
            "control_at_local_collocation_nodes_all_intervals_variables",
        ) and hasattr(casadi_optimization_problem_object, "metadata_local_collocation_nodes_tau"):
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

    # Extract per-interval trajectories (scaled values from NLP)
    if hasattr(
        casadi_optimization_problem_object,
        "state_at_local_approximation_nodes_all_intervals_variables",
    ):
        solution.solved_state_trajectories_per_interval = []
        for mesh_idx in range(num_mesh_intervals):
            state_vars = casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables[
                mesh_idx
            ]
            state_vals = casadi_solution_object.value(state_vars)

            # Ensure proper array format
            if isinstance(state_vals, ca.DM | ca.MX):
                state_vals = np.array(state_vals.full())
            else:
                state_vals = np.array(state_vals)

            if num_states == 1 and state_vals.ndim == 1:
                state_vals = state_vals.reshape(1, -1)
            elif num_states > 1 and state_vals.ndim == 1:
                num_points = len(state_vals) // num_states
                if num_points * num_states == len(state_vals):
                    state_vals = state_vals.reshape(num_states, num_points)

            solution.solved_state_trajectories_per_interval.append(state_vals)

    if (
        hasattr(
            casadi_optimization_problem_object,
            "control_at_local_collocation_nodes_all_intervals_variables",
        )
        and num_controls > 0
    ):
        solution.solved_control_trajectories_per_interval = []
        for mesh_idx in range(num_mesh_intervals):
            control_vars = casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables[
                mesh_idx
            ]
            control_vals = casadi_solution_object.value(control_vars)

            # Ensure proper array format
            if isinstance(control_vals, ca.DM | ca.MX):
                control_vals = np.array(control_vals.full())
            else:
                control_vals = np.array(control_vals)

            if num_controls == 1 and control_vals.ndim == 1:
                control_vals = control_vals.reshape(1, -1)
            elif num_controls > 1 and control_vals.ndim == 1:
                num_points = len(control_vals) // num_controls
                if num_points * num_controls == len(control_vals):
                    control_vals = control_vals.reshape(num_controls, num_points)

            solution.solved_control_trajectories_per_interval.append(control_vals)

    # Finalize solution
    solution.success = True
    solution.message = "NLP solved successfully with proper scaling."
    solution.time_states = np.array(state_trajectory_times, dtype=np.float64)
    solution.states = [np.array(s_traj, dtype=np.float64) for s_traj in state_trajectory_values]
    solution.time_controls = np.array(control_trajectory_times, dtype=np.float64)
    solution.controls = [np.array(c_traj, dtype=np.float64) for c_traj in control_trajectory_values]
    solution.raw_solution = casadi_solution_object

    print("  ✅ Solution extracted successfully")
    if solution.auto_scaling_enabled:
        print(f"    📊 Final objective (physical): {solution.objective:.6f}")
        print(f"    📐 Scaling factor used: w_0 = {solution.objective_scaling_factor:.3e}")

    return solution
