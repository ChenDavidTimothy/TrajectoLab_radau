# trajectolab/solution_extraction.py
"""
Solution data extraction and formatting from raw CasADi multiphase optimization results - PURGED.
All redundancy eliminated, using centralized validation.
"""

import logging

import casadi as ca
import numpy as np

from .exceptions import DataIntegrityError, SolutionExtractionError
from .input_validation import validate_array_numerical_integrity
from .tl_types import FloatArray, OptimalControlSolution, PhaseID, ProblemProtocol


logger = logging.getLogger(__name__)


def extract_multiphase_integral_values(
    casadi_solution_object: ca.OptiSol | None,
    opti_object: ca.Opti,
    phase_id: PhaseID,
    num_integrals: int,
) -> float | FloatArray | None:
    """SINGLE SOURCE for extracting integral values for a specific phase from the CasADi solution."""
    if (
        num_integrals == 0
        or not hasattr(opti_object, "multiphase_variables_reference")
        or opti_object.multiphase_variables_reference is None
        or casadi_solution_object is None
    ):
        return None

    try:
        variables = opti_object.multiphase_variables_reference
        if phase_id not in variables.phase_variables:
            return None

        phase_vars = variables.phase_variables[phase_id]
        if phase_vars.integral_variables is None:
            return None

        raw_value = casadi_solution_object.value(phase_vars.integral_variables)

        if isinstance(raw_value, ca.DM):
            np_array_value = np.asarray(raw_value.toarray())
            if num_integrals == 1:
                if np_array_value.size == 1:
                    result = float(np_array_value.item())
                    # SINGLE validation call
                    validate_array_numerical_integrity(
                        np.array([result]),
                        f"Phase {phase_id} integral value",
                        "integral extraction",
                    )
                    return result
                else:
                    logger.warning(
                        f"Phase {phase_id}: For num_integrals=1, unexpected array shape {np_array_value.shape}"
                    )
                    if np_array_value.size > 0:
                        return float(np_array_value.flatten()[0])
                    else:
                        logger.warning(f"Phase {phase_id}: Empty integral value array")
                        return np.nan
            else:
                result_array = np_array_value.flatten().astype(np.float64)
                # SINGLE validation call
                validate_array_numerical_integrity(
                    result_array, f"Phase {phase_id} integral array", "integral extraction"
                )
                return result_array

        elif isinstance(raw_value, (float, int)):
            if num_integrals == 1:
                result = float(raw_value)
                validate_array_numerical_integrity(
                    np.array([result]), f"Phase {phase_id} integral value", "integral extraction"
                )
                return result
            else:
                logger.warning(
                    f"Phase {phase_id}: Expected array for {num_integrals} integrals, got scalar {raw_value}"
                )
                return np.full(num_integrals, np.nan, dtype=np.float64)
        else:
            logger.warning(
                f"Phase {phase_id}: Unexpected CasADi value type: {type(raw_value)}, value: {raw_value}"
            )
            if num_integrals > 1:
                return np.full(num_integrals, np.nan, dtype=np.float64)
            elif num_integrals == 1:
                return np.nan
            return None

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        logger.warning(f"Could not extract integral values for phase {phase_id}: {e}")
        if num_integrals > 1:
            return np.full(num_integrals, np.nan, dtype=np.float64)
        elif num_integrals == 1:
            return np.nan
        return None


def process_phase_trajectory_points(
    phase_id: PhaseID,
    mesh_interval_index: int,
    casadi_solution_object: ca.OptiSol,
    variables_list: list[ca.MX],
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
    """SINGLE SOURCE for processing trajectory points for a single mesh interval within a phase."""
    # Guard clause validation
    if mesh_interval_index >= len(variables_list) or mesh_interval_index >= len(local_tau_nodes):
        raise SolutionExtractionError(
            f"Phase {phase_id}: Variable list or tau nodes incomplete for interval {mesh_interval_index}",
            "Solution data structure inconsistency",
        )

    current_interval_variables = variables_list[mesh_interval_index]
    current_interval_local_tau_values = local_tau_nodes[mesh_interval_index]

    try:
        solved_values = casadi_solution_object.value(current_interval_variables)
    except Exception as e:
        raise SolutionExtractionError(
            f"Failed to extract values for phase {phase_id} interval {mesh_interval_index}: {e}",
            "CasADi value extraction error",
        ) from e

    # Shape validation
    if num_variables == 1 and solved_values.ndim == 1:
        solved_values = solved_values.reshape(1, -1)

    # SINGLE validation call
    validate_array_numerical_integrity(
        solved_values,
        f"Phase {phase_id} solution values for interval {mesh_interval_index}",
        "solution data extraction",
    )

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
                value = solved_values[var_index, node_index]
                # SINGLE validation call
                validate_array_numerical_integrity(
                    np.array([value]),
                    f"Phase {phase_id} trajectory value at interval {mesh_interval_index}, node {node_index}",
                    "trajectory data extraction",
                )
                trajectory_values_lists[var_index].append(value)
            last_added_point = physical_time

    return last_added_point


def extract_and_format_multiphase_solution(
    casadi_solution_object: ca.OptiSol | None,
    casadi_optimization_problem_object: ca.Opti,
    problem: ProblemProtocol,
) -> OptimalControlSolution:
    """SINGLE SOURCE for extracting and formatting the solution from multiphase CasADi optimization result."""
    solution = OptimalControlSolution()
    solution.opti_object = casadi_optimization_problem_object

    if casadi_solution_object is None:
        solution.success = False
        solution.message = "Multiphase solver did not find a solution or was not run."
        return solution

    # Get multiphase structure
    phase_ids = problem.get_phase_ids()
    total_states, total_controls, num_static_params = problem.get_total_variable_counts()

    if not hasattr(casadi_optimization_problem_object, "multiphase_variables_reference"):
        solution.success = False
        solution.message = "Missing multiphase variables reference in optimization object."
        return solution

    variables = casadi_optimization_problem_object.multiphase_variables_reference
    metadata = casadi_optimization_problem_object.multiphase_metadata_reference

    # Extract core solution values with validation
    try:
        # Extract objective
        objective = float(
            casadi_solution_object.value(
                casadi_optimization_problem_object.multiphase_objective_expression_reference
            )
        )

        # SINGLE validation call
        validate_array_numerical_integrity(
            np.array([objective]), "multiphase objective value", "core solution extraction"
        )
        solution.objective = objective

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        solution.success = False
        solution.message = f"Failed to extract core multiphase solution values: {e}"
        solution.raw_solution = casadi_solution_object
        raise SolutionExtractionError(
            f"Critical failure in multiphase core value extraction: {e}",
            "Core solution processing error",
        ) from e

    # Extract static parameters
    if num_static_params > 0 and variables.static_parameters is not None:
        try:
            static_params_raw = casadi_solution_object.value(variables.static_parameters)
            if isinstance(static_params_raw, ca.DM):
                static_params_array = np.asarray(static_params_raw.toarray()).flatten()
            else:
                static_params_array = np.array(static_params_raw).flatten()

            # SINGLE validation call
            validate_array_numerical_integrity(
                static_params_array, "static parameters", "static parameter extraction"
            )
            solution.static_parameters = static_params_array.astype(np.float64)
        except Exception as e:
            if isinstance(e, DataIntegrityError):
                raise
            logger.warning(f"Failed to extract static parameters: {e}")
            solution.static_parameters = None

    # Extract solution for each phase
    for phase_id in phase_ids:
        if phase_id not in variables.phase_variables:
            continue

        phase_vars = variables.phase_variables[phase_id]
        phase_def = problem._phases[phase_id]
        num_states, num_controls = problem.get_phase_variable_counts(phase_id)
        num_mesh_intervals = len(phase_def.collocation_points_per_interval)
        num_integrals = phase_def.num_integrals

        try:
            # Extract phase times
            initial_time = float(casadi_solution_object.value(phase_vars.initial_time))
            terminal_time = float(casadi_solution_object.value(phase_vars.terminal_time))

            # SINGLE validation call for times
            validate_array_numerical_integrity(
                np.array([initial_time, terminal_time]),
                f"Phase {phase_id} times",
                "phase time extraction",
            )

            solution.phase_initial_times[phase_id] = initial_time
            solution.phase_terminal_times[phase_id] = terminal_time

        except Exception as e:
            if isinstance(e, DataIntegrityError):
                raise
            logger.error(f"Failed to extract phase {phase_id} times: {e}")
            solution.phase_initial_times[phase_id] = float("nan")
            solution.phase_terminal_times[phase_id] = float("nan")
            continue

        # Extract phase integrals
        solution.phase_integrals[phase_id] = extract_multiphase_integral_values(
            casadi_solution_object, casadi_optimization_problem_object, phase_id, num_integrals
        )

        # Extract phase state trajectories
        state_trajectory_times: list[float] = []
        state_trajectory_values: list[list[float]] = [[] for _ in range(num_states)]
        last_time_point_added_to_state_trajectory: float = -np.inf

        try:
            for mesh_idx in range(num_mesh_intervals):
                last_time_point_added_to_state_trajectory = process_phase_trajectory_points(
                    phase_id,
                    mesh_idx,
                    casadi_solution_object,
                    phase_vars.state_matrices,
                    metadata.phase_local_state_tau[phase_id],
                    metadata.phase_global_mesh_nodes[phase_id],
                    initial_time,
                    terminal_time,
                    last_time_point_added_to_state_trajectory,
                    state_trajectory_times,
                    state_trajectory_values,
                    num_states,
                    is_state=True,
                )
        except Exception as e:
            if isinstance(e, (SolutionExtractionError, DataIntegrityError)):
                raise
            raise SolutionExtractionError(
                f"Failed to extract phase {phase_id} state trajectories: {e}",
                "State trajectory processing error",
            ) from e

        solution.phase_time_states[phase_id] = np.array(state_trajectory_times, dtype=np.float64)
        solution.phase_states[phase_id] = [
            np.array(s_traj, dtype=np.float64) for s_traj in state_trajectory_values
        ]

        # Extract phase control trajectories
        control_trajectory_times: list[float] = []
        control_trajectory_values: list[list[float]] = [[] for _ in range(num_controls)]
        last_time_point_added_to_control_trajectory: float = -np.inf

        try:
            for mesh_idx in range(num_mesh_intervals):
                last_time_point_added_to_control_trajectory = process_phase_trajectory_points(
                    phase_id,
                    mesh_idx,
                    casadi_solution_object,
                    phase_vars.control_variables,
                    metadata.phase_local_control_tau[phase_id],
                    metadata.phase_global_mesh_nodes[phase_id],
                    initial_time,
                    terminal_time,
                    last_time_point_added_to_control_trajectory,
                    control_trajectory_times,
                    control_trajectory_values,
                    num_controls,
                    is_state=False,
                )
        except Exception as e:
            if isinstance(e, (SolutionExtractionError, DataIntegrityError)):
                raise
            raise SolutionExtractionError(
                f"Failed to extract phase {phase_id} control trajectories: {e}",
                "Control trajectory processing error",
            ) from e

        solution.phase_time_controls[phase_id] = np.array(
            control_trajectory_times, dtype=np.float64
        )
        solution.phase_controls[phase_id] = [
            np.array(c_traj, dtype=np.float64) for c_traj in control_trajectory_values
        ]

        # Store mesh information for this phase
        solution.phase_mesh_intervals[phase_id] = phase_def.collocation_points_per_interval.copy()
        solution.phase_mesh_nodes[phase_id] = metadata.phase_global_mesh_nodes[phase_id].copy()

        # Extract per-interval trajectories for this phase
        try:
            solution.phase_solved_state_trajectories_per_interval[phase_id] = []
            for mesh_idx in range(num_mesh_intervals):
                state_vars = phase_vars.state_matrices[mesh_idx]
                state_vals = casadi_solution_object.value(state_vars)

                # Ensure proper dimensionality
                if isinstance(state_vals, (ca.DM, ca.MX)):
                    state_vals = np.array(state_vals.full())
                else:
                    state_vals = np.array(state_vals)

                # SINGLE validation call
                validate_array_numerical_integrity(
                    state_vals,
                    f"Phase {phase_id} state values for interval {mesh_idx}",
                    "per-interval state data extraction",
                )

                # Ensure it's 2D for consistent processing
                if num_states == 1 and state_vals.ndim == 1:
                    state_vals = state_vals.reshape(1, -1)
                elif num_states > 1 and state_vals.ndim == 1:
                    num_points = len(state_vals) // num_states
                    if num_points * num_states == len(state_vals):
                        state_vals = state_vals.reshape(num_states, num_points)

                solution.phase_solved_state_trajectories_per_interval[phase_id].append(state_vals)

            # Extract per-interval control trajectories for this phase
            if num_controls > 0:
                solution.phase_solved_control_trajectories_per_interval[phase_id] = []
                for mesh_idx in range(num_mesh_intervals):
                    control_vars = phase_vars.control_variables[mesh_idx]
                    control_vals = casadi_solution_object.value(control_vars)

                    # Ensure proper dimensionality
                    if isinstance(control_vals, (ca.DM, ca.MX)):
                        control_vals = np.array(control_vals.full())
                    else:
                        control_vals = np.array(control_vals)

                    # SINGLE validation call
                    validate_array_numerical_integrity(
                        control_vals,
                        f"Phase {phase_id} control values for interval {mesh_idx}",
                        "per-interval control data extraction",
                    )

                    # Ensure it's 2D for consistent processing
                    if num_controls == 1 and control_vals.ndim == 1:
                        control_vals = control_vals.reshape(1, -1)
                    elif num_controls > 1 and control_vals.ndim == 1:
                        num_points = len(control_vals) // num_controls
                        if num_points * num_controls == len(control_vals):
                            control_vals = control_vals.reshape(num_controls, num_points)

                    solution.phase_solved_control_trajectories_per_interval[phase_id].append(
                        control_vals
                    )

        except Exception as e:
            if isinstance(e, DataIntegrityError):
                raise
            raise SolutionExtractionError(
                f"Failed to extract per-interval trajectories for phase {phase_id}: {e}",
                "Per-interval trajectory processing error",
            ) from e

    # Finalize solution
    solution.success = True
    solution.message = "Multiphase NLP solved successfully."
    solution.raw_solution = casadi_solution_object

    return solution
