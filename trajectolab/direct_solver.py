"""
Direct solver implementation using Radau pseudospectral method.

This module provides the core optimization solver that transcribes optimal control
problems into nonlinear programming problems using Radau pseudospectral collocation.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import casadi as ca
import numpy as np

from trajectolab.radau import compute_radau_collocation_components
from trajectolab.trajectolab_types import (  # Constants; Basic types; Dictionary types; Function protocols
    _FloatArray,
    _ParamDict,
    _Vector,
)

# --- Type Aliases ---
# CasADi-specific types
_CasadiScalar: TypeAlias = ca.MX | ca.SX | ca.DM
_CasadiVector: TypeAlias = ca.MX | ca.SX | ca.DM
_CasadiMatrix: TypeAlias = ca.MX | ca.SX | ca.DM
_CasadiSolution: TypeAlias = ca.OptiSol

# Solver-specific types
_SolverOptionsDict: TypeAlias = dict[str, Any]
_CollocationPointsList: TypeAlias = list[int]
_MeshNodesList: TypeAlias = list[float] | np.ndarray
_StateTrajectory: TypeAlias = list[np.ndarray]
_ControlTrajectory: TypeAlias = list[np.ndarray]


@dataclass
class PathConstraint:
    """Represents a constraint along the trajectory."""

    val: _CasadiScalar
    min_val: float | None = None
    max_val: float | None = None
    equals: float | None = None


@dataclass
class EventConstraint:
    """Represents a constraint at endpoints or events."""

    val: _CasadiScalar
    min_val: float | None = None
    max_val: float | None = None
    equals: float | None = None


@dataclass
class InitialGuess:
    """Initial guess for the solver."""

    initial_time_variable: float | None = None
    terminal_time_variable: float | None = None
    states: list[np.ndarray] | None = field(default_factory=list)
    controls: list[np.ndarray] | None = field(default_factory=list)
    integrals: float | Sequence[float] | None = None


@dataclass
class DefaultGuessValues:
    """Default values for solver initial guesses."""

    state: float = 0.0
    control: float = 0.0
    integral: float = 0.0


@dataclass
class OptimalControlProblem:
    """Defines an optimal control problem for the direct solver."""

    num_states: int
    num_controls: int
    dynamics_function: Callable[
        [_CasadiVector, _CasadiVector, float, _ParamDict], list[float] | _CasadiVector
    ]
    objective_function: Callable[
        [float, float, _CasadiVector, _CasadiVector, _CasadiScalar | None, _ParamDict], float
    ]
    t0_bounds: list[float]
    tf_bounds: list[float]
    num_integrals: int = 0
    collocation_points_per_interval: _CollocationPointsList | None = None
    global_normalized_mesh_nodes: _MeshNodesList | None = None
    integral_integrand_function: (
        Callable[[_CasadiVector, _CasadiVector, float, int, _ParamDict], float] | None
    ) = None
    path_constraints_function: (
        Callable[[_CasadiVector, _CasadiVector, float, _ParamDict], list[PathConstraint]] | None
    ) = None
    event_constraints_function: (
        Callable[
            [
                float,
                float,
                _CasadiVector,
                _CasadiVector,
                _CasadiScalar | Sequence[float] | None,
                _ParamDict,
            ],
            list[EventConstraint],
        ]
        | None
    ) = None
    problem_parameters: _ParamDict = field(default_factory=dict)
    initial_guess: InitialGuess | None = None
    default_initial_guess_values: DefaultGuessValues = field(default_factory=DefaultGuessValues)
    solver_options: _SolverOptionsDict = field(default_factory=dict)


class OptimalControlSolution:
    """Solution container for an optimal control problem."""

    def __init__(self) -> None:
        self.success: bool = False
        self.message: str = "Solver not run yet."
        self.initial_time_variable: float | None = None
        self.terminal_time_variable: float | None = None
        self.objective: float | None = None
        self.integrals: float | _FloatArray | None = None
        self.time_states: _FloatArray = np.array([], dtype=np.float64)
        self.states: list[_FloatArray] = []
        self.time_controls: _FloatArray = np.array([], dtype=np.float64)
        self.controls: list[_FloatArray] = []
        self.raw_solution: _CasadiSolution | None = None
        self.opti_object: ca.Opti | None = None
        self.num_collocation_nodes_per_interval: _CollocationPointsList = []
        self.global_normalized_mesh_nodes: _MeshNodesList | None = None
        # Additional attributes
        self.num_collocation_nodes_list_at_solve_time: _CollocationPointsList | None = None
        self.global_mesh_nodes_at_solve_time: _MeshNodesList | None = None
        self.solved_state_trajectories_per_interval: _StateTrajectory | None = None
        self.solved_control_trajectories_per_interval: _ControlTrajectory | None = None


def _extract_integral_values(
    casadi_solution_object: _CasadiSolution | None, opti_object: ca.Opti, num_integrals: int
) -> float | _FloatArray | None:
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

        # Handle different possible return formats
        if isinstance(raw_value, ca.DM):
            if num_integrals == 1 and raw_value.shape == (1, 1):
                return float(raw_value[0, 0])
            return cast(_FloatArray, raw_value.toarray().flatten())
        return float(raw_value)  # Scalar case
    except Exception as e:
        print(f"Warning: Could not extract integral values: {e}")
        return np.full(num_integrals, np.nan, dtype=np.float64) if num_integrals > 1 else np.nan


def _process_trajectory_points(
    mesh_interval_index: int,
    casadi_solution_object: _CasadiSolution,
    opti_object: ca.Opti,
    variables_list: list[_CasadiMatrix],
    local_tau_values: _MeshNodesList,
    global_normalized_mesh_nodes: _MeshNodesList,
    initial_time: float,
    terminal_time: float,
    last_added_point: float,
    trajectory_times: list[float],
    trajectory_values_lists: list[list[float]],
    num_variables: int,
    is_state: bool = True,
) -> float:
    """Process trajectory points from CasADi solution."""

    if mesh_interval_index >= len(variables_list):
        print(f"Error: Variable list not found or incomplete for interval {mesh_interval_index}.")
        return last_added_point

    solved_values = casadi_solution_object.value(variables_list[mesh_interval_index])
    if num_variables == 1 and solved_values.ndim == 1:
        solved_values = solved_values.reshape(1, -1)

    # Determine number of nodes to process (states include end point, controls don't)
    num_nodes = len(local_tau_values)
    if not is_state and num_nodes > 0:
        num_nodes -= 1  # Controls don't have values at the end point

    for node_index in range(num_nodes):
        # Map from local tau to global tau to physical time
        local_tau = local_tau_values[node_index]
        segment_start = global_normalized_mesh_nodes[mesh_interval_index]
        segment_end = global_normalized_mesh_nodes[mesh_interval_index + 1]

        global_tau = (segment_end - segment_start) / 2 * local_tau + (
            segment_end + segment_start
        ) / 2
        physical_time = (terminal_time - initial_time) / 2 * global_tau + (
            terminal_time + initial_time
        ) / 2

        # Determine if we should add this point
        is_last_point = (
            mesh_interval_index == len(variables_list) - 1 and node_index == num_nodes - 1
        )
        if abs(physical_time - last_added_point) > 1e-9 or is_last_point or not trajectory_times:
            trajectory_times.append(physical_time)
            for var_index in range(num_variables):
                trajectory_values_lists[var_index].append(
                    float(solved_values[var_index, node_index])
                )
            last_added_point = physical_time

    return last_added_point


def _extract_and_format_solution(
    casadi_solution_object: _CasadiSolution | None,
    casadi_optimization_problem_object: ca.Opti,
    problem_definition: OptimalControlProblem,
    num_collocation_nodes_per_interval: _CollocationPointsList,
    global_normalized_mesh_nodes: _MeshNodesList,
) -> OptimalControlSolution:
    """Extract and format the solution from CasADi."""

    solution = OptimalControlSolution()

    if casadi_solution_object is None:
        solution.success = False
        solution.message = "Solver did not find a solution or was not run."
        solution.opti_object = casadi_optimization_problem_object
        solution.num_collocation_nodes_per_interval = num_collocation_nodes_per_interval
        solution.global_normalized_mesh_nodes = global_normalized_mesh_nodes
        return solution

    num_mesh_intervals = len(num_collocation_nodes_per_interval)
    num_states = problem_definition.num_states
    num_controls = problem_definition.num_controls
    num_integrals = problem_definition.num_integrals

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
        solution.opti_object = casadi_optimization_problem_object
        solution.num_collocation_nodes_per_interval = num_collocation_nodes_per_interval
        solution.global_normalized_mesh_nodes = global_normalized_mesh_nodes
        return solution

    # Extract integral values
    solution.integrals = _extract_integral_values(
        casadi_solution_object, casadi_optimization_problem_object, num_integrals
    )

    # Process state trajectories
    state_trajectory_times: list[float] = []
    state_trajectory_values: list[list[float]] = [[] for _ in range(num_states)]
    last_time_point_added_to_state_trajectory: float = -np.inf

    for mesh_interval_index in range(num_mesh_intervals):
        if not hasattr(
            casadi_optimization_problem_object,
            "state_at_local_approximation_nodes_all_intervals_variables",
        ):
            print(
                "Error: state_at_local_approximation_nodes_all_intervals_variables not found in optimization object"
            )
            continue

        last_time_point_added_to_state_trajectory = _process_trajectory_points(
            mesh_interval_index,
            casadi_solution_object,
            casadi_optimization_problem_object,
            casadi_optimization_problem_object.state_at_local_approximation_nodes_all_intervals_variables,
            casadi_optimization_problem_object.metadata_local_state_approximation_nodes_tau[
                mesh_interval_index
            ],
            global_normalized_mesh_nodes,
            solution.initial_time_variable,
            solution.terminal_time_variable,
            last_time_point_added_to_state_trajectory,
            state_trajectory_times,
            state_trajectory_values,
            num_states,
            is_state=True,
        )

    # Process control trajectories
    control_trajectory_times: list[float] = []
    control_trajectory_values: list[list[float]] = [[] for _ in range(num_controls)]
    last_time_point_added_to_control_trajectory: float = -np.inf

    for mesh_interval_index in range(num_mesh_intervals):
        if not hasattr(
            casadi_optimization_problem_object,
            "control_at_local_collocation_nodes_all_intervals_variables",
        ):
            print(
                "Error: control_at_local_collocation_nodes_all_intervals_variables not found in optimization object"
            )
            continue

        last_time_point_added_to_control_trajectory = _process_trajectory_points(
            mesh_interval_index,
            casadi_solution_object,
            casadi_optimization_problem_object,
            casadi_optimization_problem_object.control_at_local_collocation_nodes_all_intervals_variables,
            casadi_optimization_problem_object.metadata_local_collocation_nodes_tau[
                mesh_interval_index
            ],
            global_normalized_mesh_nodes,
            solution.initial_time_variable,
            solution.terminal_time_variable,
            last_time_point_added_to_control_trajectory,
            control_trajectory_times,
            control_trajectory_values,
            num_controls,
            is_state=False,
        )

    # Set solution values
    solution.success = True
    solution.message = "NLP solved successfully."
    solution.time_states = np.array(state_trajectory_times, dtype=np.float64)
    solution.states = [np.array(s_traj, dtype=np.float64) for s_traj in state_trajectory_values]
    solution.time_controls = np.array(control_trajectory_times, dtype=np.float64)
    solution.controls = [np.array(c_traj, dtype=np.float64) for c_traj in control_trajectory_values]
    solution.raw_solution = casadi_solution_object
    solution.opti_object = casadi_optimization_problem_object
    solution.num_collocation_nodes_per_interval = num_collocation_nodes_per_interval
    solution.global_normalized_mesh_nodes = global_normalized_mesh_nodes

    return solution


def _apply_constraint(opti: ca.Opti, constraint: PathConstraint | EventConstraint) -> None:
    """Apply a constraint to the optimization problem."""

    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)


def _validate_dynamics_output(output: Any, num_states: int) -> _CasadiVector:
    """Validate and standardize the output from dynamics functions."""

    if isinstance(output, list):
        return ca.vertcat(*output) if output else ca.MX(num_states, 1)
    elif isinstance(output, ca.MX):
        if output.shape[1] == 1:  # Already a column vector
            return output
        elif output.shape[0] == 1 and num_states > 1:  # Row vector, needs transpose
            return output.T
        elif num_states == 1:  # Scalar MX for num_states=1 is fine
            return output

    raise TypeError(f"Dynamics function output type not supported: {type(output)}")


def _set_initial_value_for_integrals(
    opti: ca.Opti,
    integral_vars: _CasadiScalar | _CasadiVector,
    guess: float | Sequence[float] | None,
    num_integrals: int,
) -> None:
    """Set initial values for integral variables."""

    if guess is None:
        return

    if num_integrals == 1:
        if isinstance(guess, (int, float)):
            opti.set_initial(integral_vars, float(guess))
        elif isinstance(guess, (list, np.ndarray)) and np.array(guess).size == 1:
            opti.set_initial(integral_vars, float(np.array(guess).item()))
        else:
            print(f"Warning: Invalid format for single integral guess: {guess}")
    elif isinstance(guess, (list, np.ndarray)) and np.array(guess).size == num_integrals:
        opti.set_initial(integral_vars, np.array(guess, dtype=np.float64).flatten())
    else:
        print(f"Warning: Invalid format/length for multiple integrals guess: {guess}")


def solve_single_phase_radau_collocation(
    problem_definition: OptimalControlProblem,
) -> OptimalControlSolution:
    """
    Solve a single-phase optimal control problem using Radau pseudospectral collocation.

    Args:
        problem_definition: Complete definition of the optimal control problem

    Returns:
        Solution object containing trajectories and metadata

    Raises:
        ValueError: If problem definition is incomplete or invalid
    """

    opti = ca.Opti()

    # --- Extract Problem Parameters ---
    num_states = problem_definition.num_states
    num_controls = problem_definition.num_controls
    num_integrals = problem_definition.num_integrals

    if not problem_definition.collocation_points_per_interval:
        raise ValueError("problem_definition must include 'collocation_points_per_interval'.")

    num_collocation_nodes_per_interval = problem_definition.collocation_points_per_interval
    if not isinstance(num_collocation_nodes_per_interval, list) or not all(
        isinstance(n, int) and n > 0 for n in num_collocation_nodes_per_interval
    ):
        raise ValueError("'collocation_points_per_interval' must be a list of positive integers.")

    num_mesh_intervals = len(num_collocation_nodes_per_interval)

    # User-defined functions
    dynamics_function = problem_definition.dynamics_function
    objective_function = problem_definition.objective_function
    path_constraints_function = problem_definition.path_constraints_function
    event_constraints_function = problem_definition.event_constraints_function
    integral_integrand_function = problem_definition.integral_integrand_function
    problem_parameters = problem_definition.problem_parameters

    # --- Decision Variables ---
    initial_time_variable = opti.variable()
    terminal_time_variable = opti.variable()
    opti.subject_to(initial_time_variable >= problem_definition.t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem_definition.t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem_definition.tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem_definition.tf_bounds[1])
    opti.subject_to(terminal_time_variable > initial_time_variable + 1e-6)

    # --- Mesh Definition ---
    # Allow user to specify global_normalized_mesh_nodes for adaptive meshing
    user_mesh = problem_definition.global_normalized_mesh_nodes

    if user_mesh is not None:
        global_normalized_mesh_nodes = np.array(user_mesh, dtype=np.float64)
        if not (
            len(global_normalized_mesh_nodes) == num_mesh_intervals + 1
            and np.all(np.diff(global_normalized_mesh_nodes) > 1e-9)
            and np.isclose(global_normalized_mesh_nodes[0], -1.0)
            and np.isclose(global_normalized_mesh_nodes[-1], 1.0)
        ):
            raise ValueError(
                "Provided 'global_normalized_mesh_nodes' must be sorted, have num_mesh_intervals+1 elements, "
                "start at -1.0, and end at +1.0, with positive interval lengths."
            )
    else:
        # Default to uniform distribution if not provided
        global_normalized_mesh_nodes = np.linspace(-1, 1, num_mesh_intervals + 1, dtype=np.float64)

    # State Variables (Shared at global mesh points)
    state_at_global_mesh_nodes_variables: list[_CasadiVector] = [
        opti.variable(num_states) for _ in range(num_mesh_intervals + 1)
    ]
    state_at_local_approximation_nodes_all_intervals_variables: list[_CasadiMatrix] = []
    state_at_interior_local_approximation_nodes_all_intervals_variables: list[
        _CasadiMatrix | None
    ] = []

    # Control Variables
    control_at_local_collocation_nodes_all_intervals_variables: list[_CasadiMatrix] = [
        opti.variable(num_controls, num_collocation_nodes_per_interval[k])
        for k in range(num_mesh_intervals)
    ]

    # Integral Variables
    integral_decision_variables: _CasadiScalar | _CasadiVector | None = None
    if num_integrals > 0:
        integral_decision_variables = (
            opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        )

    accumulated_integral_expressions: list[_CasadiScalar] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )
    local_state_approximation_nodes_tau_all_intervals: list[_Vector] = []
    local_collocation_nodes_tau_all_intervals: list[_Vector] = []

    # --- Assemble State Representations & Define Interval Constraints ---
    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = num_collocation_nodes_per_interval[mesh_interval_index]

        # Setup state variables for this interval
        current_interval_state_columns: list[_CasadiVector] = [
            ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)
        ]
        current_interval_state_columns[0] = state_at_global_mesh_nodes_variables[
            mesh_interval_index
        ]

        # Handle interior state approximation nodes
        interior_nodes_var = None
        if num_colloc_nodes > 1:
            num_interior_nodes = num_colloc_nodes - 1
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

        state_at_nodes = ca.horzcat(*current_interval_state_columns)
        state_at_local_approximation_nodes_all_intervals_variables.append(state_at_nodes)

        # Get Radau basis components
        basis_components = compute_radau_collocation_components(num_colloc_nodes)
        state_nodes_tau = basis_components.state_approximation_nodes.flatten()
        colloc_nodes_tau = basis_components.collocation_nodes.flatten()
        quad_weights = basis_components.quadrature_weights.flatten()
        diff_matrix = ca.DM(basis_components.differentiation_matrix)

        local_state_approximation_nodes_tau_all_intervals.append(state_nodes_tau)
        local_collocation_nodes_tau_all_intervals.append(colloc_nodes_tau)

        # Calculate state derivatives
        state_derivative_at_colloc = ca.mtimes(state_at_nodes, diff_matrix.T)

        # Calculate scaling factor for this interval
        global_segment_length = (
            global_normalized_mesh_nodes[mesh_interval_index + 1]
            - global_normalized_mesh_nodes[mesh_interval_index]
        )
        if global_segment_length <= 1e-9:
            raise ValueError(
                f"Mesh interval {mesh_interval_index} has zero or negative length in global tau: {global_segment_length}"
            )

        tau_to_time_scaling = (
            (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
        )

        # Apply dynamics constraints at each collocation point
        for i_colloc in range(num_colloc_nodes):
            state_at_colloc = state_at_nodes[:, i_colloc]
            control_at_colloc = control_at_local_collocation_nodes_all_intervals_variables[
                mesh_interval_index
            ][:, i_colloc]

            # Map from local tau to global tau to physical time
            local_tau = colloc_nodes_tau[i_colloc]
            global_tau = (
                global_segment_length / 2 * local_tau
                + (
                    global_normalized_mesh_nodes[mesh_interval_index + 1]
                    + global_normalized_mesh_nodes[mesh_interval_index]
                )
                / 2
            )
            physical_time = (terminal_time_variable - initial_time_variable) / 2 * global_tau + (
                terminal_time_variable + initial_time_variable
            ) / 2

            # Apply dynamics constraint
            state_derivative_rhs = dynamics_function(
                state_at_colloc, control_at_colloc, physical_time, problem_parameters
            )
            state_derivative_rhs_vector = _validate_dynamics_output(
                state_derivative_rhs, num_states
            )

            opti.subject_to(
                state_derivative_at_colloc[:, i_colloc]
                == tau_to_time_scaling * state_derivative_rhs_vector
            )

            # Apply path constraints if provided
            if path_constraints_function:
                path_constraints = path_constraints_function(
                    state_at_colloc,
                    control_at_colloc,
                    physical_time,
                    problem_parameters,
                )
                if not isinstance(path_constraints, list):
                    path_constraints = [path_constraints]

                for constraint in path_constraints:
                    if isinstance(constraint, PathConstraint):
                        _apply_constraint(opti, constraint)
                    else:
                        raise ValueError(
                            "Path constraint function must return a list of PathConstraint objects"
                        )

        # Handle integral calculation if needed
        if num_integrals > 0 and integral_integrand_function:
            for integral_index in range(num_integrals):
                quad_sum = ca.MX(0)
                for i_colloc in range(num_colloc_nodes):
                    state_at_colloc = state_at_nodes[:, i_colloc]
                    control_at_colloc = control_at_local_collocation_nodes_all_intervals_variables[
                        mesh_interval_index
                    ][:, i_colloc]

                    local_tau = colloc_nodes_tau[i_colloc]
                    global_tau = (
                        global_segment_length / 2 * local_tau
                        + (
                            global_normalized_mesh_nodes[mesh_interval_index + 1]
                            + global_normalized_mesh_nodes[mesh_interval_index]
                        )
                        / 2
                    )
                    physical_time = (
                        terminal_time_variable - initial_time_variable
                    ) / 2 * global_tau + (terminal_time_variable + initial_time_variable) / 2

                    weight = quad_weights[i_colloc]
                    integrand_value = integral_integrand_function(
                        state_at_colloc,
                        control_at_colloc,
                        physical_time,
                        integral_index,
                        problem_parameters,
                    )
                    quad_sum += weight * integrand_value

                accumulated_integral_expressions[integral_index] += tau_to_time_scaling * quad_sum

    # Link integral decision variables to their computed values
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

    # Set up objective function
    initial_state = state_at_global_mesh_nodes_variables[0]
    terminal_state = state_at_global_mesh_nodes_variables[num_mesh_intervals]

    objective_value = objective_function(
        initial_time_variable,
        terminal_time_variable,
        initial_state,
        terminal_state,
        integral_decision_variables,
        problem_parameters,
    )
    opti.minimize(objective_value)

    # Apply event constraints if provided
    if event_constraints_function:
        event_constraints = event_constraints_function(
            initial_time_variable,
            terminal_time_variable,
            initial_state,
            terminal_state,
            integral_decision_variables,
            problem_parameters,
        )

        if not isinstance(event_constraints, list):
            event_constraints = [event_constraints]

        for constraint in event_constraints:
            if isinstance(constraint, EventConstraint):
                _apply_constraint(opti, constraint)
            else:
                raise ValueError(
                    "Event constraint function must return a list of EventConstraint objects"
                )

    # --- Set Initial Guesses ---
    if problem_definition.initial_guess:
        ig = problem_definition.initial_guess

        # Time variables
        if ig.initial_time_variable is not None:
            opti.set_initial(initial_time_variable, ig.initial_time_variable)
        if ig.terminal_time_variable is not None:
            opti.set_initial(terminal_time_variable, ig.terminal_time_variable)

        # States at global mesh points
        if ig.states and len(ig.states) > 0:
            # First point
            if isinstance(ig.states[0], np.ndarray) and ig.states[0].shape[0] == num_states:
                opti.set_initial(state_at_global_mesh_nodes_variables[0], ig.states[0][:, 0])

            # End points of each interval
            for k in range(num_mesh_intervals):
                if k < len(ig.states) and isinstance(ig.states[k], np.ndarray):
                    cols = ig.states[k].shape[1] - 1
                    if ig.states[k].shape[0] == num_states and cols >= 0:
                        opti.set_initial(
                            state_at_global_mesh_nodes_variables[k + 1],
                            ig.states[k][:, cols],
                        )

            # Interior points
            for k in range(num_mesh_intervals):
                interior_nodes = num_collocation_nodes_per_interval[k] - 1
                if (
                    interior_nodes > 0
                    and k < len(ig.states)
                    and isinstance(ig.states[k], np.ndarray)
                    and ig.states[k].shape[0] == num_states
                ):
                    guess_cols = ig.states[k].shape[1] - 1
                    if guess_cols > 1:  # At least some interior points available
                        cols_to_use = min(interior_nodes, guess_cols - 1)
                        if (
                            cols_to_use > 0
                            and state_at_interior_local_approximation_nodes_all_intervals_variables[
                                k
                            ]
                            is not None
                        ):
                            guess_slice = ig.states[k][:, 1 : 1 + cols_to_use]

                            # Pad if needed
                            if cols_to_use < interior_nodes:
                                padding = np.tile(
                                    guess_slice[:, -1:],
                                    (1, interior_nodes - cols_to_use),
                                )
                                guess_slice = np.hstack((guess_slice, padding))

                            opti.set_initial(
                                state_at_interior_local_approximation_nodes_all_intervals_variables[
                                    k
                                ],
                                guess_slice,
                            )

        # Controls
        if ig.controls and len(ig.controls) > 0:
            for k in range(num_mesh_intervals):
                if k < len(ig.controls) and isinstance(ig.controls[k], np.ndarray):
                    nodes_needed = num_collocation_nodes_per_interval[k]
                    control_guess = ig.controls[k]

                    # Ensure 2D array
                    if num_controls == 1 and control_guess.ndim == 1:
                        control_guess = control_guess.reshape(1, -1)

                    if control_guess.shape[0] == num_controls:
                        cols_available = control_guess.shape[1]
                        cols_to_use = min(nodes_needed, cols_available)

                        if cols_to_use > 0:
                            guess_slice = control_guess[:, :cols_to_use]

                            # Pad if needed
                            if cols_to_use < nodes_needed:
                                padding = np.tile(
                                    guess_slice[:, -1:], (1, nodes_needed - cols_to_use)
                                )
                                guess_slice = np.hstack((guess_slice, padding))

                            opti.set_initial(
                                control_at_local_collocation_nodes_all_intervals_variables[k],
                                guess_slice,
                            )

        # Integrals
        if ig.integrals is not None and integral_decision_variables is not None:
            _set_initial_value_for_integrals(
                opti, integral_decision_variables, ig.integrals, num_integrals
            )

    # --- Configure Solver and Solve ---
    solver_options = problem_definition.solver_options or {}
    opti.solver("ipopt", solver_options)

    # Store references for solution extraction
    opti.initial_time_variable_reference = initial_time_variable
    opti.terminal_time_variable_reference = terminal_time_variable
    if integral_decision_variables is not None:
        opti.integral_variables_object_reference = integral_decision_variables
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

    # Solve and extract solution
    try:
        solver_solution: _CasadiSolution = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        solution = _extract_and_format_solution(
            solver_solution,
            opti,
            problem_definition,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
        solution.num_collocation_nodes_list_at_solve_time = list(num_collocation_nodes_per_interval)
        solution.global_mesh_nodes_at_solve_time = np.array(
            global_normalized_mesh_nodes, dtype=np.float64
        )
        return solution
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        solution = _extract_and_format_solution(
            None,
            opti,
            problem_definition,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes,
        )
        solution.success = False
        solution.message = f"Solver runtime error: {e}"
        try:
            if hasattr(opti, "debug") and opti.debug is not None:
                if initial_time_variable is not None:
                    solution.initial_time_variable = float(opti.debug.value(initial_time_variable))
                if terminal_time_variable is not None:
                    solution.terminal_time_variable = float(
                        opti.debug.value(terminal_time_variable)
                    )
        except Exception as debug_e:
            print(f"  Could not retrieve debug values after solver error: {debug_e}")
        return solution
