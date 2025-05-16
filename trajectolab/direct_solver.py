from __future__ import annotations

from typing import Any

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
    TrajectoryData,
)


class InitialGuess:
    initial_time_variable: float | None
    terminal_time_variable: float | None
    states: InitialGuessTrajectory | None
    controls: InitialGuessTrajectory | None
    integrals: InitialGuessIntegrals | None

    def __init__(
        self,
        initial_time_variable: float | None = None,
        terminal_time_variable: float | None = None,
        states: InitialGuessTrajectory | None = None,
        controls: InitialGuessTrajectory | None = None,
        integrals: InitialGuessIntegrals | None = None,
    ):
        self.initial_time_variable = initial_time_variable
        self.terminal_time_variable = terminal_time_variable
        self.states = states or []
        self.controls = controls or []
        self.integrals = integrals


class DefaultGuessValues:
    state: float
    control: float
    integral: float

    def __init__(self, state: float = 0.0, control: float = 0.0, integral: float = 0.0):
        self.state = state
        self.control = control
        self.integral = integral


class OptimalControlProblem:
    num_states: int
    num_controls: int
    dynamics_function: DynamicsCallable
    objective_function: ObjectiveCallable
    t0_bounds: tuple[float, float]
    tf_bounds: tuple[float, float]
    num_integrals: int
    collocation_points_per_interval: list[int] | None
    global_normalized_mesh_nodes: FloatArray | None
    integral_integrand_function: IntegralIntegrandCallable | None
    path_constraints_function: PathConstraintsCallable | None
    event_constraints_function: EventConstraintsCallable | None
    problem_parameters: ProblemParameters | None
    initial_guess: InitialGuess | None
    default_initial_guess_values: DefaultGuessValues | None
    solver_options: dict[str, Any] | None  # Any is used for diverse solver options

    def __init__(
        self,
        num_states: int,
        num_controls: int,
        dynamics_function: DynamicsCallable,
        objective_function: ObjectiveCallable,
        t0_bounds: tuple[float, float],
        tf_bounds: tuple[float, float],
        num_integrals: int = 0,
        collocation_points_per_interval: list[int] | None = None,
        global_normalized_mesh_nodes: FloatArray | None = None,
        integral_integrand_function: IntegralIntegrandCallable | None = None,
        path_constraints_function: PathConstraintsCallable | None = None,
        event_constraints_function: EventConstraintsCallable | None = None,
        problem_parameters: ProblemParameters | None = None,
        initial_guess: InitialGuess | None = None,
        default_initial_guess_values: DefaultGuessValues | None = None,
        solver_options: dict[str, Any] | None = None,
    ):
        self.num_states = num_states
        self.num_controls = num_controls
        self.num_integrals = num_integrals
        self.collocation_points_per_interval = collocation_points_per_interval or []
        self.global_normalized_mesh_nodes = global_normalized_mesh_nodes

        self.dynamics_function = dynamics_function
        self.objective_function = objective_function
        self.integral_integrand_function = integral_integrand_function
        self.path_constraints_function = path_constraints_function
        self.event_constraints_function = event_constraints_function

        self.t0_bounds = t0_bounds
        self.tf_bounds = tf_bounds
        self.problem_parameters = problem_parameters or {}

        self.initial_guess = initial_guess
        self.default_initial_guess_values = default_initial_guess_values or DefaultGuessValues()
        self.solver_options = solver_options or {}


class OptimalControlSolution:
    success: bool
    message: str
    initial_time_variable: float | None
    terminal_time_variable: float | None
    objective: float | None
    integrals: float | FloatArray | None
    time_states: FloatArray
    states: TrajectoryData
    time_controls: FloatArray
    controls: TrajectoryData
    raw_solution: CasadiOptiSol | None
    opti_object: CasadiOpti | None
    num_collocation_nodes_per_interval: list[int]
    global_normalized_mesh_nodes: FloatArray | None
    num_collocation_nodes_list_at_solve_time: list[int] | None
    global_mesh_nodes_at_solve_time: FloatArray | None
    # These were not previously typed, adding example types
    solved_state_trajectories_per_interval: list[FloatMatrix] | None  # Example
    solved_control_trajectories_per_interval: list[FloatMatrix] | None  # Example

    def __init__(self) -> None:
        self.success = False
        self.message = "Solver not run yet."
        self.initial_time_variable = None
        self.terminal_time_variable = None
        self.objective = None
        self.integrals = None
        self.time_states = np.array([], dtype=np.float64)
        self.states = []
        self.time_controls = np.array([], dtype=np.float64)
        self.controls = []
        self.raw_solution = None
        self.opti_object = None
        self.num_collocation_nodes_per_interval = []
        self.global_normalized_mesh_nodes = None
        self.num_collocation_nodes_list_at_solve_time = None
        self.global_mesh_nodes_at_solve_time = None
        self.solved_state_trajectories_per_interval = None
        self.solved_control_trajectories_per_interval = None


def _extract_integral_values(
    casadi_solution_object: CasadiOptiSol,
    opti_object: CasadiOpti,
    num_integrals: int,
) -> float | FloatArray | None:
    if (
        num_integrals == 0
        or not hasattr(opti_object, "integral_variables_object_reference")
        or opti_object.integral_variables_object_reference is None
    ):
        return None

    try:
        raw_value: CasadiDM | float = casadi_solution_object.value(
            opti_object.integral_variables_object_reference
        )

        if isinstance(raw_value, ca.DM):
            if num_integrals == 1 and raw_value.shape == (1, 1):
                return float(raw_value[0, 0])
            return raw_value.toarray().flatten()  # type: ignore[no-any-return]
        return float(raw_value)
    except Exception as e:
        print(f"Warning: Could not extract integral values: {e}")
        return np.full(num_integrals, np.nan) if num_integrals > 1 else np.nan


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
    problem_definition: OptimalControlProblem,
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
    num_states: int = problem_definition.num_states
    num_controls: int = problem_definition.num_controls
    num_integrals: int = problem_definition.num_integrals

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


def _validate_dynamics_output(output: list[CasadiMX] | CasadiMatrix, num_states: int) -> CasadiMX:
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

    raise TypeError(f"Dynamics function output type not supported: {type(output)}")


def _set_initial_value_for_integrals(
    opti: CasadiOpti,
    integral_vars: CasadiMX,
    guess: float | FloatArray | list[float] | None,  # Broadened guess type
    num_integrals: int,
) -> None:
    if guess is None:
        return

    if num_integrals == 1:
        if isinstance(guess, (int, float)):  # int included for convenience
            opti.set_initial(integral_vars, float(guess))
        elif isinstance(guess, (list, np.ndarray)) and np.array(guess).size == 1:
            opti.set_initial(integral_vars, float(np.array(guess).item()))
        else:
            print(f"Warning: Invalid format for single integral guess: {guess}")
    elif isinstance(guess, (list, np.ndarray)) and np.array(guess).size == num_integrals:
        # Ensure flat array for CasADi if it's a list of numbers or a 1D array
        flat_guess = np.array(guess, dtype=np.float64).flatten()
        opti.set_initial(integral_vars, flat_guess)
    else:
        print(f"Warning: Invalid format/length for multiple integrals guess: {guess}")


def solve_single_phase_radau_collocation(
    problem_definition: OptimalControlProblem,
) -> OptimalControlSolution:
    opti: CasadiOpti = ca.Opti()

    num_states: int = problem_definition.num_states
    num_controls: int = problem_definition.num_controls
    num_integrals: int = problem_definition.num_integrals

    if not problem_definition.collocation_points_per_interval:
        raise ValueError("problem_definition must include 'collocation_points_per_interval'.")

    num_collocation_nodes_per_interval: list[int] = (
        problem_definition.collocation_points_per_interval
    )
    if not (
        isinstance(num_collocation_nodes_per_interval, list)
        and all(isinstance(n, int) and n > 0 for n in num_collocation_nodes_per_interval)
    ):
        raise ValueError("'collocation_points_per_interval' must be a list of positive integers.")

    num_mesh_intervals: int = len(num_collocation_nodes_per_interval)

    dynamics_function: DynamicsCallable = problem_definition.dynamics_function
    objective_function: ObjectiveCallable = problem_definition.objective_function
    path_constraints_function: PathConstraintsCallable | None = (
        problem_definition.path_constraints_function
    )
    event_constraints_function: EventConstraintsCallable | None = (
        problem_definition.event_constraints_function
    )
    integral_integrand_function: IntegralIntegrandCallable | None = (
        problem_definition.integral_integrand_function
    )
    problem_parameters: ProblemParameters | None = problem_definition.problem_parameters

    initial_time_variable: CasadiMX = opti.variable()
    terminal_time_variable: CasadiMX = opti.variable()
    opti.subject_to(initial_time_variable >= problem_definition.t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem_definition.t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem_definition.tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem_definition.tf_bounds[1])
    opti.subject_to(
        terminal_time_variable > initial_time_variable + 1e-6
    )  # Ensure non-zero positive duration

    user_mesh: FloatArray | None = problem_definition.global_normalized_mesh_nodes
    global_normalized_mesh_nodes: FloatArray
    if user_mesh is not None:
        global_normalized_mesh_nodes = np.array(user_mesh, dtype=np.float64)
        if not (
            len(global_normalized_mesh_nodes) == num_mesh_intervals + 1
            and np.all(np.diff(global_normalized_mesh_nodes) > 1e-9)
            and np.isclose(global_normalized_mesh_nodes[0], -1.0)
            and np.isclose(global_normalized_mesh_nodes[-1], 1.0)
        ):
            raise ValueError(
                "Provided 'global_normalized_mesh_nodes' must be sorted, "
                "have num_mesh_intervals+1 elements, start at -1.0, and end at +1.0, "
                "with positive interval lengths."
            )
    else:
        global_normalized_mesh_nodes = np.linspace(-1, 1, num_mesh_intervals + 1, dtype=np.float64)

    state_at_global_mesh_nodes_variables: ListOfCasadiMX = [
        opti.variable(num_states) for _ in range(num_mesh_intervals + 1)
    ]
    state_at_local_approximation_nodes_all_intervals_variables: ListOfCasadiMX = []
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
                for i in range(num_interior_nodes):
                    current_interval_state_columns[i + 1] = interior_nodes_var[:, i]
        state_at_interior_local_approximation_nodes_all_intervals_variables.append(
            interior_nodes_var
        )
        current_interval_state_columns[num_colloc_nodes] = state_at_global_mesh_nodes_variables[
            mesh_interval_index + 1
        ]

        state_at_nodes: CasadiMX = ca.horzcat(*current_interval_state_columns)
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

            state_derivative_rhs: list[CasadiMX] | CasadiMX = dynamics_function(
                state_at_colloc, control_at_colloc, physical_time_at_colloc, problem_parameters
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
        constraints_to_apply = (
            event_constraints_result
            if isinstance(event_constraints_result, list)
            else [event_constraints_result]
        )
        for constraint in constraints_to_apply:
            _apply_constraint(opti, constraint)

    if problem_definition.initial_guess:
        ig: InitialGuess = problem_definition.initial_guess

        if ig.initial_time_variable is not None:
            opti.set_initial(initial_time_variable, ig.initial_time_variable)
        if ig.terminal_time_variable is not None:
            opti.set_initial(terminal_time_variable, ig.terminal_time_variable)

        if ig.states:  # Check if list is not empty
            # Initial state at global mesh node 0
            if (
                len(ig.states) > 0 and ig.states[0].shape[0] == num_states
            ):  # ig.states[0] is FloatMatrix
                opti.set_initial(state_at_global_mesh_nodes_variables[0], ig.states[0][:, 0])

            # States at subsequent global mesh nodes (interval end points)
            for k in range(num_mesh_intervals):
                if k < len(ig.states) and ig.states[k].shape[0] == num_states:
                    # The guess for interval k applies to states up to the END of interval k
                    # The last column of ig.states[k] is the state at the end of the interval k
                    # which corresponds to state_at_global_mesh_nodes_variables[k+1]
                    cols_in_guess_k = ig.states[k].shape[1]
                    if cols_in_guess_k > 0:
                        opti.set_initial(
                            state_at_global_mesh_nodes_variables[k + 1], ig.states[k][:, -1]
                        )

            # Interior state approximation nodes
            for k in range(num_mesh_intervals):
                interior_approx_nodes_var_k = (
                    state_at_interior_local_approximation_nodes_all_intervals_variables[k]
                )
                if (
                    interior_approx_nodes_var_k is not None
                ):  # if num_interior_nodes > 0 for this interval
                    num_interior_nodes_k = interior_approx_nodes_var_k.shape[
                        1
                    ]  # num_colloc_nodes - 1
                    if k < len(ig.states) and ig.states[k].shape[0] == num_states:
                        guess_matrix_k = ig.states[
                            k
                        ]  # FloatMatrix [num_states, num_nodes_in_guess_interval_k]
                        # Guessed nodes for interval k: [start_node, interior_node_1, ..., interior_node_M, end_node]
                        # We need to provide guesses for the interior_nodes_var (num_interior_nodes_k columns)
                        # These correspond to guess_matrix_k[:, 1:-1] if available
                        num_guessed_interior_nodes = guess_matrix_k.shape[1] - 2
                        if (
                            num_guessed_interior_nodes >= num_interior_nodes_k
                            and num_interior_nodes_k > 0
                        ):
                            opti.set_initial(
                                interior_approx_nodes_var_k,
                                guess_matrix_k[:, 1 : 1 + num_interior_nodes_k],
                            )
                        elif (
                            num_guessed_interior_nodes > 0 and num_interior_nodes_k > 0
                        ):  # Pad if guess is shorter
                            padded_guess = np.tile(
                                guess_matrix_k[:, 1 : 1 + num_guessed_interior_nodes],
                                (
                                    1,
                                    int(np.ceil(num_interior_nodes_k / num_guessed_interior_nodes)),
                                ),
                            )
                            opti.set_initial(
                                interior_approx_nodes_var_k, padded_guess[:, :num_interior_nodes_k]
                            )
                        # else: no good guess for interior points from this format, CasADi will use default 0.0

        if ig.controls:  # Check if list is not empty
            for k in range(num_mesh_intervals):
                if k < len(ig.controls) and isinstance(
                    ig.controls[k], np.ndarray
                ):  # ig.controls[k] is FloatMatrix
                    control_guess_k: FloatMatrix = ig.controls[k]
                    target_var_k: CasadiMX = (
                        control_at_local_collocation_nodes_all_intervals_variables[k]
                    )
                    nodes_needed_k: int = target_var_k.shape[
                        1
                    ]  # num_collocation_nodes_per_interval[k]

                    if control_guess_k.shape[0] == num_controls and control_guess_k.shape[1] > 0:
                        if control_guess_k.shape[1] == nodes_needed_k:
                            opti.set_initial(target_var_k, control_guess_k)
                        elif control_guess_k.shape[1] < nodes_needed_k:  # Pad
                            padding = np.tile(
                                control_guess_k[:, -1:],
                                (1, nodes_needed_k - control_guess_k.shape[1]),
                            )
                            padded_guess = np.hstack((control_guess_k, padding))
                            opti.set_initial(target_var_k, padded_guess)
                        else:  # Truncate
                            opti.set_initial(target_var_k, control_guess_k[:, :nodes_needed_k])

        if ig.integrals is not None and integral_decision_variables is not None:
            _set_initial_value_for_integrals(
                opti, integral_decision_variables, ig.integrals, num_integrals
            )

    solver_options_to_use: dict[str, Any] = problem_definition.solver_options or {}
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
    opti.metadata_global_normalized_mesh_nodes = (
        global_normalized_mesh_nodes  # This is already a FloatArray
    )
    opti.symbolic_objective_function_reference = objective_value

    solution_obj: OptimalControlSolution
    try:
        solver_solution: CasadiOptiSol = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        solution_obj = _extract_and_format_solution(
            solver_solution,
            opti,
            problem_definition,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        solution_obj = _extract_and_format_solution(
            None,  # Indicates solver failure or no solution
            opti,
            problem_definition,
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
                # (they are defined, but good practice)
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
