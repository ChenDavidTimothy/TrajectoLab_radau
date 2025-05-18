from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import casadi as ca
import numpy as np

from .radau import RadauBasisComponents, compute_radau_collocation_components
from .tl_types import (
    CasadiDM,
    CasadiMatrix,
    CasadiMX,
    CasadiOpti,
    CasadiOptiSol,
    DynamicsCallable,
    EventConstraint,
    EventConstraintsCallable,
    FloatArray,
    FloatMatrix,
    InitialGuessIntegrals,
    InitialGuessTrajectory,
    IntegralIntegrandCallable,
    ListOfCasadiMX,
    ObjectiveCallable,
    PathConstraint,
    PathConstraintsCallable,
    ProblemParameters,
    ProblemProtocol,
    TrajectoryData,
)


class InitialGuess:
    """
    Initial guess for the optimal control problem.
    """

    def __init__(
        self,
        initial_time_variable: float | None = None,
        terminal_time_variable: float | None = None,
        states: InitialGuessTrajectory | None = None,
        controls: InitialGuessTrajectory | None = None,
        integrals: InitialGuessIntegrals | None = None,
    ) -> None:
        self.initial_time_variable = initial_time_variable
        self.terminal_time_variable = terminal_time_variable
        self.states = states or []
        self.controls = controls or []
        self.integrals = integrals


class OptimalControlSolution:
    """
    Solution to an optimal control problem.
    """

    def __init__(self) -> None:
        self.success: bool = False
        self.message: str = "Solver not run yet."
        self.initial_time_variable: float | None = None
        self.terminal_time_variable: float | None = None
        self.objective: float | None = None
        self.integrals: float | FloatArray | None = None
        self.time_states: FloatArray = np.array([], dtype=np.float64)
        self.states: TrajectoryData = []
        self.time_controls: FloatArray = np.array([], dtype=np.float64)
        self.controls: TrajectoryData = []
        self.raw_solution: CasadiOptiSol | None = None
        self.opti_object: CasadiOpti | None = None
        self.num_collocation_nodes_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: FloatArray | None = None
        self.num_collocation_nodes_list_at_solve_time: list[int] | None = None
        self.global_mesh_nodes_at_solve_time: FloatArray | None = None
        self.solved_state_trajectories_per_interval: list[FloatMatrix] | None = None
        self.solved_control_trajectories_per_interval: list[FloatMatrix] | None = None


def _extract_integral_values(
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
                return cast(FloatArray, np_array_value.flatten())

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


def _process_trajectory_points(
    mesh_interval_index: int,
    casadi_solution_object: CasadiOptiSol,
    opti_object: CasadiOpti,  # Contains metadata and variable references
    variables_list_attr_name: str,  # Name of the attribute on opti_object holding variable list
    local_tau_nodes_attr_name: str,  # Name of the attribute on opti_object for local tau nodes
    global_normalized_mesh_nodes: FloatArray,
    initial_time: float,
    terminal_time: float,
    last_added_point: float,
    trajectory_times: list[float],
    trajectory_values_lists: list[list[float]],  # Outer list for variables, inner for time points
    num_variables: int,
    is_state: bool = True,
) -> float:
    variables_list: ListOfCasadiMX = getattr(opti_object, variables_list_attr_name, [])
    local_tau_values_all_intervals: list[FloatArray] = getattr(
        opti_object, local_tau_nodes_attr_name, []
    )

    if mesh_interval_index >= len(variables_list) or mesh_interval_index >= len(
        local_tau_values_all_intervals
    ):
        print(
            f"Error: Variable list or tau nodes not found or incomplete for interval {mesh_interval_index}."
        )
        return last_added_point

    current_interval_variables: CasadiMX = variables_list[mesh_interval_index]
    current_interval_local_tau_values: FloatArray = local_tau_values_all_intervals[
        mesh_interval_index
    ]

    solved_values: FloatMatrix = casadi_solution_object.value(current_interval_variables)
    if num_variables == 1 and solved_values.ndim == 1:  # Ensure 2D for consistency
        solved_values = solved_values.reshape(1, -1)

    num_nodes_to_process = len(current_interval_local_tau_values)
    if (
        not is_state and num_nodes_to_process > 0
    ):  # Controls don't use the interval end point for values
        num_nodes_to_process -= 1

    for node_index in range(num_nodes_to_process):
        local_tau: float = current_interval_local_tau_values[node_index]
        segment_start: float = global_normalized_mesh_nodes[mesh_interval_index]
        segment_end: float = global_normalized_mesh_nodes[mesh_interval_index + 1]

        global_tau: float = (segment_end - segment_start) / 2 * local_tau + (
            segment_end + segment_start
        ) / 2
        physical_time: float = (terminal_time - initial_time) / 2 * global_tau + (
            terminal_time + initial_time
        ) / 2

        is_last_point_in_trajectory: bool = (
            mesh_interval_index == len(variables_list) - 1
            and node_index == num_nodes_to_process - 1
        )
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


def _extract_and_format_solution(
    casadi_solution_object: CasadiOptiSol | None,
    casadi_optimization_problem_object: CasadiOpti,
    problem: ProblemProtocol,
    num_collocation_nodes_per_interval: list[int],
    global_normalized_mesh_nodes: FloatArray,
) -> OptimalControlSolution:
    solution = OptimalControlSolution()
    solution.opti_object = casadi_optimization_problem_object  # Store opti object early
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

    solution.integrals = _extract_integral_values(
        casadi_solution_object, casadi_optimization_problem_object, num_integrals
    )

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
        last_time_point_added_to_state_trajectory = _process_trajectory_points(
            mesh_idx,
            casadi_solution_object,
            casadi_optimization_problem_object,
            "state_at_local_approximation_nodes_all_intervals_variables",
            "metadata_local_state_approximation_nodes_tau",
            global_normalized_mesh_nodes,
            solution.initial_time_variable,
            solution.terminal_time_variable,
            last_time_point_added_to_state_trajectory,
            state_trajectory_times,
            state_trajectory_values,
            num_states,
            is_state=True,
        )

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

        last_time_point_added_to_control_trajectory = _process_trajectory_points(
            mesh_idx,
            casadi_solution_object,
            casadi_optimization_problem_object,
            "control_at_local_collocation_nodes_all_intervals_variables",
            "metadata_local_collocation_nodes_tau",
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


def _apply_constraint(opti: CasadiOpti, constraint: PathConstraint | EventConstraint) -> None:
    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)


def _validate_dynamics_output(
    output: list[CasadiMX] | CasadiMatrix | Sequence[CasadiMX], num_states: int
) -> CasadiMX:
    """Validates and converts dynamics function output to the expected CasadiMX format."""
    if isinstance(output, list):
        # First convert result to MX if it's a DM
        result = ca.vertcat(*output) if output else ca.MX(num_states, 1)
        # Ensure result is MX type
        return ca.MX(result) if isinstance(result, ca.DM) else result
    elif isinstance(output, ca.MX):
        if output.shape[1] == 1:
            return output
        elif output.shape[0] == 1 and num_states > 1:
            return output.T
        elif num_states == 1:  # Scalar MX for num_states=1
            return output
    elif isinstance(output, ca.DM):
        # Explicitly convert DM to MX
        result = ca.MX(output)
        if result.shape[1] == 1:
            return result
        elif result.shape[0] == 1 and num_states > 1:
            return result.T
        else:
            return result
    elif isinstance(output, Sequence):
        # Convert sequence to list and then process
        return _validate_dynamics_output(list(output), num_states)

    raise TypeError(f"Dynamics function output type not supported: {type(output)}")


def _validate_and_set_integral_guess(
    opti: CasadiOpti,
    integral_vars: CasadiMX,
    guess: float | FloatArray | list[float] | None,
    num_integrals: int,
) -> None:
    """
    Validate and set initial guess for integrals with strict dimension checking.

    Args:
        opti: CasADi optimization object
        integral_vars: CasADi integral variables
        guess: Initial guess for integrals
        num_integrals: Expected number of integrals

    Raises:
        ValueError: If guess dimensions don't match requirements exactly
    """
    if guess is None:
        raise ValueError(f"Integral guess required for {num_integrals} integrals")

    if num_integrals == 1:
        if not isinstance(guess, int | float):
            raise ValueError(
                f"For single integral, guess must be scalar (int or float), "
                f"got {type(guess)} with value {guess}"
            )
        opti.set_initial(integral_vars, float(guess))

    elif num_integrals > 1:
        if isinstance(guess, int | float):
            raise ValueError(
                f"For {num_integrals} integrals, guess must be array-like, got scalar {guess}"
            )

        # Convert to numpy array and validate
        guess_array = np.array(guess, dtype=np.float64)
        if guess_array.size != num_integrals:
            raise ValueError(
                f"Integral guess must have exactly {num_integrals} elements, got {guess_array.size}"
            )

        opti.set_initial(integral_vars, guess_array.flatten())


def solve_single_phase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """
    Solves a single-phase optimal control problem using Radau pseudospectral collocation.

    Args:
        problem: The optimal control problem definition

    Returns:
        An OptimalControlSolution object containing the solution

    Raises:
        ValueError: If problem configuration is invalid or initial guess is missing/invalid
    """
    # Validate problem is properly configured
    if not hasattr(problem, "_mesh_configured") or not problem._mesh_configured:
        raise ValueError(
            "Problem mesh must be explicitly configured before solving. "
            "Call problem.set_mesh(polynomial_degrees, mesh_points)"
        )

    # Validate initial guess is provided and complete
    problem.validate_initial_guess()

    opti: CasadiOpti = ca.Opti()

    # Extract necessary problem data
    num_states: int = len(problem._states)
    num_controls: int = len(problem._controls)
    num_integrals: int = problem._num_integrals

    if not problem.collocation_points_per_interval:
        raise ValueError("Problem must include 'collocation_points_per_interval'.")

    num_collocation_nodes_per_interval: list[int] = problem.collocation_points_per_interval
    if not (
        isinstance(num_collocation_nodes_per_interval, list)
        and all(isinstance(n, int) and n > 0 for n in num_collocation_nodes_per_interval)
    ):
        raise ValueError("'collocation_points_per_interval' must be a list of positive integers.")

    num_mesh_intervals: int = len(num_collocation_nodes_per_interval)

    # Get vectorized functions directly from problem
    dynamics_function: DynamicsCallable = problem.get_dynamics_function()
    objective_function: ObjectiveCallable = problem.get_objective_function()
    path_constraints_function: PathConstraintsCallable | None = (
        problem.get_path_constraints_function()
    )
    event_constraints_function: EventConstraintsCallable | None = (
        problem.get_event_constraints_function()
    )
    integral_integrand_function: IntegralIntegrandCallable | None = problem.get_integrand_function()
    problem_parameters: ProblemParameters = problem._parameters

    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()
    opti.subject_to(initial_time_variable >= problem._t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem._t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem._tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem._tf_bounds[1])
    opti.subject_to(
        terminal_time_variable > initial_time_variable + 1e-6
    )  # Ensure non-zero positive duration

    # Validate and use mesh
    if problem.global_normalized_mesh_nodes is None:
        raise ValueError("Global normalized mesh nodes must be set")

    global_normalized_mesh_nodes = problem.global_normalized_mesh_nodes

    # Additional mesh validation
    if not (
        len(global_normalized_mesh_nodes) == num_mesh_intervals + 1
        and np.all(np.diff(global_normalized_mesh_nodes) > 1e-9)
        and np.isclose(global_normalized_mesh_nodes[0], -1.0)
        and np.isclose(global_normalized_mesh_nodes[-1], 1.0)
    ):
        raise ValueError(
            "Global mesh nodes must be sorted, have num_mesh_intervals+1 elements, "
            "start at -1.0, end at +1.0, with positive interval lengths."
        )

    state_at_global_mesh_nodes_variables: ListOfCasadiMX = [
        opti.variable(num_states) for _ in range(num_mesh_intervals + 1)
    ]
    state_at_local_approximation_nodes_all_intervals_variables: list[CasadiMatrix] = []
    state_at_interior_local_approximation_nodes_all_intervals_variables: list[CasadiMX | None] = []

    control_at_local_collocation_nodes_all_intervals_variables: ListOfCasadiMX = [
        opti.variable(num_controls, num_collocation_nodes_per_interval[k])
        for k in range(num_mesh_intervals)
    ]

    integral_decision_variables: CasadiMX | None = None
    if num_integrals > 0:
        integral_decision_variables = (
            opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        )

    accumulated_integral_expressions: list[CasadiMX] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )
    local_state_approximation_nodes_tau_all_intervals: list[FloatArray] = []
    local_collocation_nodes_tau_all_intervals: list[FloatArray] = []

    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes: int = num_collocation_nodes_per_interval[mesh_interval_index]
        current_interval_state_columns: list[CasadiMX] = [
            ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)
        ]
        current_interval_state_columns[0] = state_at_global_mesh_nodes_variables[
            mesh_interval_index
        ]

        interior_nodes_var: CasadiMX | None = None
        if num_colloc_nodes > 1:  # Only relevant if there are interior collocation points
            num_interior_nodes: int = (
                num_colloc_nodes - 1
            )  # num_state_approx_nodes = num_colloc_nodes (LGR) + 1 (endpoint)
            # interior approx nodes = num_colloc_nodes - 1
            if num_interior_nodes > 0:
                interior_nodes_var = opti.variable(num_states, num_interior_nodes)
                if interior_nodes_var is None:
                    raise ValueError("Failed to create interior_nodes_var")
                for i in range(num_interior_nodes):
                    current_interval_state_columns[i + 1] = interior_nodes_var[:, i]
        state_at_interior_local_approximation_nodes_all_intervals_variables.append(
            interior_nodes_var
        )
        current_interval_state_columns[num_colloc_nodes] = state_at_global_mesh_nodes_variables[
            mesh_interval_index + 1
        ]

        state_at_nodes: CasadiMatrix = ca.horzcat(*current_interval_state_columns)
        state_at_local_approximation_nodes_all_intervals_variables.append(state_at_nodes)

        basis_components: RadauBasisComponents = compute_radau_collocation_components(
            num_colloc_nodes
        )
        state_nodes_tau: FloatArray = basis_components.state_approximation_nodes.flatten()
        colloc_nodes_tau: FloatArray = basis_components.collocation_nodes.flatten()
        quad_weights: FloatArray = basis_components.quadrature_weights.flatten()
        diff_matrix: CasadiDM = ca.DM(basis_components.differentiation_matrix)  # Ensure it's DM

        local_state_approximation_nodes_tau_all_intervals.append(state_nodes_tau)
        local_collocation_nodes_tau_all_intervals.append(colloc_nodes_tau)

        state_derivative_at_colloc: CasadiMX = ca.mtimes(state_at_nodes, diff_matrix.T)

        global_segment_length: float = (
            global_normalized_mesh_nodes[mesh_interval_index + 1]
            - global_normalized_mesh_nodes[mesh_interval_index]
        )
        if global_segment_length <= 1e-9:  # Using a small epsilon
            raise ValueError(
                f"Mesh interval {mesh_interval_index} has zero or negative length: {global_segment_length}"
            )

        # Scaling factor for derivative: dt/d(local_tau), where physical_time = time_transform(global_tau)
        # and global_tau = global_tau_transform(local_tau)
        # d(physical_time)/d(local_tau) = ( (tf-t0)/2 ) * ( (global_tau_end - global_tau_start)/2 )
        # This original (tf-t0)*global_segment_length/4.0 is correct for mapping from local tau [-1,1] to physical time.
        tau_to_time_scaling: CasadiMX = (
            (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
        )

        for i_colloc in range(num_colloc_nodes):
            state_at_colloc: CasadiMX = state_at_nodes[
                :, i_colloc
            ]  # This uses state approx nodes indexing
            control_at_colloc: CasadiMX = (
                control_at_local_collocation_nodes_all_intervals_variables[mesh_interval_index][
                    :, i_colloc
                ]
            )

            local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
            global_colloc_tau_val: CasadiMX = (
                global_segment_length / 2 * local_colloc_tau_val
                + (
                    global_normalized_mesh_nodes[mesh_interval_index + 1]
                    + global_normalized_mesh_nodes[mesh_interval_index]
                )
                / 2
            )
            physical_time_at_colloc: CasadiMX = (
                terminal_time_variable - initial_time_variable
            ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

            state_derivative_rhs: list[CasadiMX] | CasadiMX | Sequence[CasadiMX] = (
                dynamics_function(
                    state_at_colloc, control_at_colloc, physical_time_at_colloc, problem_parameters
                )
            )
            state_derivative_rhs_vector: CasadiMX = _validate_dynamics_output(
                state_derivative_rhs, num_states
            )

            opti.subject_to(
                state_derivative_at_colloc[:, i_colloc]
                == tau_to_time_scaling * state_derivative_rhs_vector
            )

            if path_constraints_function:
                path_constraints_result: list[PathConstraint] | PathConstraint = (
                    path_constraints_function(
                        state_at_colloc,
                        control_at_colloc,
                        physical_time_at_colloc,
                        problem_parameters,
                    )
                )
                constraints_to_apply = (
                    path_constraints_result
                    if isinstance(path_constraints_result, list)
                    else [path_constraints_result]
                )
                for constraint in constraints_to_apply:
                    _apply_constraint(opti, constraint)

        if num_integrals > 0 and integral_integrand_function:
            for integral_index in range(num_integrals):
                quad_sum: CasadiMX = ca.MX(0)
                for i_colloc in range(num_colloc_nodes):  # Sum over collocation points
                    state_at_colloc_for_integral: CasadiMX = state_at_nodes[
                        :, i_colloc
                    ]  # State at LGR node
                    control_at_colloc_for_integral: CasadiMX = (
                        control_at_local_collocation_nodes_all_intervals_variables[
                            mesh_interval_index
                        ][:, i_colloc]
                    )

                    local_colloc_tau_val_for_integral: float = colloc_nodes_tau[i_colloc]
                    global_colloc_tau_val_for_integral: CasadiMX = (
                        global_segment_length / 2 * local_colloc_tau_val_for_integral
                        + (
                            global_normalized_mesh_nodes[mesh_interval_index + 1]
                            + global_normalized_mesh_nodes[mesh_interval_index]
                        )
                        / 2
                    )
                    physical_time_at_colloc_for_integral: CasadiMX = (
                        terminal_time_variable - initial_time_variable
                    ) / 2 * global_colloc_tau_val_for_integral + (
                        terminal_time_variable + initial_time_variable
                    ) / 2

                    weight: float = quad_weights[i_colloc]
                    integrand_value: CasadiMX = integral_integrand_function(
                        state_at_colloc_for_integral,
                        control_at_colloc_for_integral,
                        physical_time_at_colloc_for_integral,
                        integral_index,
                        problem_parameters,
                    )
                    quad_sum += weight * integrand_value
                accumulated_integral_expressions[integral_index] += tau_to_time_scaling * quad_sum

    if (
        num_integrals > 0
        and integral_integrand_function
        and integral_decision_variables is not None
    ):
        if num_integrals == 1:
            opti.subject_to(integral_decision_variables == accumulated_integral_expressions[0])
        else:
            for i in range(num_integrals):
                opti.subject_to(
                    integral_decision_variables[i] == accumulated_integral_expressions[i]
                )

    initial_state: CasadiMX = state_at_global_mesh_nodes_variables[0]
    terminal_state: CasadiMX = state_at_global_mesh_nodes_variables[num_mesh_intervals]

    objective_value: CasadiMX = objective_function(
        initial_time_variable,
        terminal_time_variable,
        initial_state,
        terminal_state,
        integral_decision_variables,
        problem_parameters,
    )
    opti.minimize(objective_value)

    if event_constraints_function:
        event_constraints_result: list[EventConstraint] | EventConstraint = (
            event_constraints_function(
                initial_time_variable,
                terminal_time_variable,
                initial_state,
                terminal_state,
                integral_decision_variables,
                problem_parameters,
            )
        )
        event_constraints_to_apply = (
            event_constraints_result
            if isinstance(event_constraints_result, list)
            else [event_constraints_result]
        )
        for event_constraint in event_constraints_to_apply:
            _apply_constraint(opti, event_constraint)

    # Apply initial guess with strict validation
    if problem.initial_guess is None:
        raise ValueError("Initial guess must be provided")

    ig: InitialGuess = problem.initial_guess

    # Time variables
    if ig.initial_time_variable is not None:
        opti.set_initial(initial_time_variable, ig.initial_time_variable)
    if ig.terminal_time_variable is not None:
        opti.set_initial(terminal_time_variable, ig.terminal_time_variable)

    # States - strict dimension validation
    if not ig.states:
        raise ValueError("States initial guess must be provided")

    if len(ig.states) != num_mesh_intervals:
        raise ValueError(
            f"States guess must have {num_mesh_intervals} arrays, got {len(ig.states)}"
        )

    # Global mesh nodes (interval endpoints)
    for k in range(num_mesh_intervals):
        state_guess_k = ig.states[k]

        # Validate dimensions strictly
        expected_shape = (num_states, num_collocation_nodes_per_interval[k] + 1)
        if state_guess_k.shape != expected_shape:
            raise ValueError(
                f"State guess for interval {k} has shape {state_guess_k.shape}, "
                f"expected {expected_shape}"
            )

        # Set initial mesh node (start of interval k)
        if k == 0:
            opti.set_initial(state_at_global_mesh_nodes_variables[0], state_guess_k[:, 0])

        # Set final mesh node (end of interval k)
        opti.set_initial(state_at_global_mesh_nodes_variables[k + 1], state_guess_k[:, -1])

    # Interior state approximation nodes
    for k in range(num_mesh_intervals):
        interior_var = state_at_interior_local_approximation_nodes_all_intervals_variables[k]
        if interior_var is not None:
            state_guess_k = ig.states[k]
            num_interior_nodes = interior_var.shape[1]

            # Extract interior columns (excluding first and last)
            if state_guess_k.shape[1] >= num_interior_nodes + 2:
                interior_guess = state_guess_k[:, 1 : 1 + num_interior_nodes]
                opti.set_initial(interior_var, interior_guess)
            else:
                raise ValueError(
                    f"State guess for interval {k} has {state_guess_k.shape[1]} nodes, "
                    f"but needs at least {num_interior_nodes + 2} for interior nodes"
                )

    # Controls - strict dimension validation
    if not ig.controls:
        raise ValueError("Controls initial guess must be provided")

    if len(ig.controls) != num_mesh_intervals:
        raise ValueError(
            f"Controls guess must have {num_mesh_intervals} arrays, got {len(ig.controls)}"
        )

    for k in range(num_mesh_intervals):
        control_guess_k = ig.controls[k]
        expected_shape = (num_controls, num_collocation_nodes_per_interval[k])

        if control_guess_k.shape != expected_shape:
            raise ValueError(
                f"Control guess for interval {k} has shape {control_guess_k.shape}, "
                f"expected {expected_shape}"
            )

        opti.set_initial(
            control_at_local_collocation_nodes_all_intervals_variables[k], control_guess_k
        )

    # Integrals - strict validation
    if num_integrals > 0 and integral_decision_variables is not None:
        _validate_and_set_integral_guess(
            opti, integral_decision_variables, ig.integrals, num_integrals
        )

    # Set solver options
    solver_options_to_use: dict[str, object] = problem.solver_options or {}
    opti.solver("ipopt", solver_options_to_use)

    # Store references for solution extraction
    opti.initial_time_variable_reference = initial_time_variable
    opti.terminal_time_variable_reference = terminal_time_variable
    if integral_decision_variables is not None:
        opti.integral_variables_object_reference = integral_decision_variables
    else:
        opti.integral_variables_object_reference = None  # Ensure attribute exists

    opti.state_at_local_approximation_nodes_all_intervals_variables = (
        state_at_local_approximation_nodes_all_intervals_variables
    )
    opti.control_at_local_collocation_nodes_all_intervals_variables = (
        control_at_local_collocation_nodes_all_intervals_variables
    )
    opti.metadata_local_state_approximation_nodes_tau = (
        local_state_approximation_nodes_tau_all_intervals
    )
    opti.metadata_local_collocation_nodes_tau = local_collocation_nodes_tau_all_intervals
    opti.metadata_global_normalized_mesh_nodes = global_normalized_mesh_nodes
    opti.symbolic_objective_function_reference = objective_value

    solution_obj: OptimalControlSolution
    try:
        solver_solution: CasadiOptiSol = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        solution_obj = _extract_and_format_solution(
            solver_solution,
            opti,
            problem,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        solution_obj = _extract_and_format_solution(
            None,  # Indicates solver failure or no solution
            opti,
            problem,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
        solution_obj.success = (
            False  # Ensure this is set even if _extract_and_format_solution did it
        )
        solution_obj.message = f"Solver runtime error: {e}"
        try:
            # Attempt to get debug values if solver failed after variables were set
            if hasattr(opti, "debug") and opti.debug is not None:
                # Ensure variables are not None before trying to access .debug.value()
                if initial_time_variable is not None:
                    solution_obj.initial_time_variable = float(
                        opti.debug.value(initial_time_variable)
                    )
                if terminal_time_variable is not None:
                    solution_obj.terminal_time_variable = float(
                        opti.debug.value(terminal_time_variable)
                    )
        except Exception as debug_e:
            print(f"  Could not retrieve debug values after solver error: {debug_e}")

    # These were assigned to solution inside _extract_and_format_solution for success case
    # For failure case, they are also set to the input parameters.
    # Storing them here ensures they are always part of the returned solution object.
    solution_obj.num_collocation_nodes_list_at_solve_time = list(num_collocation_nodes_per_interval)
    solution_obj.global_mesh_nodes_at_solve_time = global_normalized_mesh_nodes.copy()
    return solution_obj
