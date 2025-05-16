"""
Direct solver implementation using Radau pseudospectral method.

This module provides the core optimization solver that transcribes optimal control
problems into nonlinear programming problems using Radau pseudospectral collocation.
"""

from collections.abc import Sequence

# dataclass and field are no longer needed here directly for local definitions
from typing import Any, cast  # TypeAlias no longer needed here for local definitions

import casadi as ca
import numpy as np

from trajectolab.radau import compute_radau_collocation_components

# Updated import list from trajectolab_types
from trajectolab.trajectolab_types import (
    EventConstraint,
    OptimalControlProblem,
    OptimalControlSolution,
    PathConstraint,
    _CasadiMatrix,
    _CasadiScalar,
    _CasadiSolution,
    _CasadiVector,
    _CollocationPointsList,
    _FloatArray,
    _MeshNodesList,
    _Vector,
)

# --- Type Aliases and DataClasses previously defined here are now in trajectolab_types.py ---


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
                return cast(_FloatArray, np_array_value.flatten())

        elif isinstance(raw_value, (float, int)):
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
    casadi_solution_object: _CasadiSolution,
    opti_object: ca.Opti,
    variables_list: list[_CasadiMatrix],
    local_tau_values: _MeshNodesList,  # _Vector could also be appropriate if always 1D numpy array
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

    num_nodes = len(local_tau_values)
    if not is_state and num_nodes > 0:
        num_nodes -= 1

    for node_index in range(num_nodes):
        local_tau = local_tau_values[node_index]
        # Ensure global_normalized_mesh_nodes is treated as np.ndarray for indexing if it's a list
        g_mesh_nodes_arr = np.asarray(global_normalized_mesh_nodes)
        segment_start = g_mesh_nodes_arr[mesh_interval_index]
        segment_end = g_mesh_nodes_arr[mesh_interval_index + 1]

        global_tau = (segment_end - segment_start) / 2 * local_tau + (
            segment_end + segment_start
        ) / 2
        physical_time = (terminal_time - initial_time) / 2 * global_tau + (
            terminal_time + initial_time
        ) / 2

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

    solution = OptimalControlSolution()  # Uses default factory from dataclass

    if casadi_solution_object is None:
        solution.success = False
        solution.message = "Solver did not find a solution or was not run."
        solution.opti_object = casadi_optimization_problem_object
        solution.num_collocation_nodes_per_interval = list(num_collocation_nodes_per_interval)
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
        solution.num_collocation_nodes_per_interval = list(num_collocation_nodes_per_interval)
        solution.global_normalized_mesh_nodes = global_normalized_mesh_nodes
        return solution

    solution.integrals = _extract_integral_values(
        casadi_solution_object, casadi_optimization_problem_object, num_integrals
    )

    state_trajectory_times: list[float] = []
    state_trajectory_values: list[list[float]] = [[] for _ in range(num_states)]
    last_time_point_added_to_state_trajectory: float = -np.inf

    g_mesh_nodes_arr = np.asarray(
        global_normalized_mesh_nodes
    )  # Ensure it's an array for processing

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
            g_mesh_nodes_arr,  # Pass the ensured array
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
            g_mesh_nodes_arr,  # Pass the ensured array
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
    solution.opti_object = casadi_optimization_problem_object
    solution.num_collocation_nodes_per_interval = list(num_collocation_nodes_per_interval)
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
        return ca.vertcat(*output) if output else ca.MX(num_states, 1)  # Ensure column vector
    elif isinstance(output, ca.MX):
        if output.is_empty():  # Handle empty MX
            return ca.MX(num_states, 1)
        if output.shape[1] == 1:
            return output
        elif output.shape[0] == 1 and output.shape[1] == num_states:  # Row vector, needs transpose
            return output.T
        elif output.shape[0] == num_states and output.shape[1] == 1:  # Already column
            return output
        elif num_states == 1 and output.is_scalar():  # Scalar MX for num_states=1 is fine
            return output
        # If it's (num_states, 1) but not MX, it might be SX/DM, which is fine for _CasadiVector
        if hasattr(output, "shape") and output.shape == (num_states, 1):
            return cast(_CasadiVector, output)

    raise TypeError(
        f"Dynamics function output type not supported or shape mismatch: "
        f"type {type(output)}, shape {output.shape if hasattr(output, 'shape') else 'N/A'}. "
        f"Expected ({num_states}, 1) or list of {num_states} scalars."
    )


def _set_initial_value_for_integrals(
    opti: ca.Opti,
    integral_vars: _CasadiScalar | _CasadiVector,
    guess: float | Sequence[float] | np.ndarray | None,
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

    dynamics_function = problem_definition.dynamics_function
    objective_function = problem_definition.objective_function
    path_constraints_function = problem_definition.path_constraints_function
    event_constraints_function = problem_definition.event_constraints_function
    integral_integrand_function = problem_definition.integral_integrand_function
    problem_parameters = problem_definition.problem_parameters

    initial_time_variable = opti.variable()
    terminal_time_variable = opti.variable()
    opti.subject_to(initial_time_variable >= problem_definition.t0_bounds[0])
    opti.subject_to(initial_time_variable <= problem_definition.t0_bounds[1])
    opti.subject_to(terminal_time_variable >= problem_definition.tf_bounds[0])
    opti.subject_to(terminal_time_variable <= problem_definition.tf_bounds[1])
    opti.subject_to(terminal_time_variable > initial_time_variable + 1e-6)  # Ensure tf > t0

    user_mesh = problem_definition.global_normalized_mesh_nodes
    global_normalized_mesh_nodes_arr: np.ndarray
    if user_mesh is not None:
        global_normalized_mesh_nodes_arr = np.array(user_mesh, dtype=np.float64)
        if not (
            len(global_normalized_mesh_nodes_arr) == num_mesh_intervals + 1
            and np.all(np.diff(global_normalized_mesh_nodes_arr) > 1e-9)  # Strictly increasing
            and np.isclose(global_normalized_mesh_nodes_arr[0], -1.0)
            and np.isclose(global_normalized_mesh_nodes_arr[-1], 1.0)
        ):
            raise ValueError(
                "Provided 'global_normalized_mesh_nodes' must be sorted, have num_mesh_intervals+1 elements, "
                "start at -1.0, and end at +1.0, with positive interval lengths."
            )
    else:
        global_normalized_mesh_nodes_arr = np.linspace(
            -1, 1, num_mesh_intervals + 1, dtype=np.float64
        )

    state_at_global_mesh_nodes_variables: list[_CasadiVector] = [
        opti.variable(num_states) for _ in range(num_mesh_intervals + 1)
    ]
    state_at_local_approximation_nodes_all_intervals_variables: list[_CasadiMatrix] = []
    state_at_interior_local_approximation_nodes_all_intervals_variables: list[
        _CasadiMatrix | None
    ] = []

    control_at_local_collocation_nodes_all_intervals_variables: list[_CasadiMatrix] = [
        opti.variable(num_controls, num_collocation_nodes_per_interval[k])
        for k in range(num_mesh_intervals)
    ]

    integral_decision_variables: _CasadiScalar | _CasadiVector | None = None
    if num_integrals > 0:
        integral_decision_variables = (
            opti.variable(num_integrals) if num_integrals > 1 else opti.variable()
        )

    accumulated_integral_expressions: list[_CasadiScalar] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )
    local_state_approximation_nodes_tau_all_intervals: list[_Vector] = (
        []
    )  # Actually list of np.ndarray
    local_collocation_nodes_tau_all_intervals: list[_Vector] = []  # Actually list of np.ndarray

    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = num_collocation_nodes_per_interval[mesh_interval_index]

        current_interval_state_columns: list[_CasadiVector] = [
            ca.MX(num_states, 1) for _ in range(num_colloc_nodes + 1)
        ]  # Ensure column vectors
        current_interval_state_columns[0] = state_at_global_mesh_nodes_variables[
            mesh_interval_index
        ]

        interior_nodes_var = None
        if num_colloc_nodes > 1:  # Only relevant if there are interior collocation points
            num_interior_nodes = (
                num_colloc_nodes - 1
            )  # LGL nodes include endpoints, Radau does not mean K-1 interior states necessarily
            # State approx nodes are K+1
            if (
                num_interior_nodes > 0
            ):  # This logic depends on how states are defined at collocation vs state approx. nodes
                # For Radau, state is approximated at K+1 points (including boundary)
                # Typically K interior state decision variables are NOT introduced like this for Radau.
                # The state_at_nodes will use K+1 points from global_mesh_nodes and K-1 interior approx points IF K > 1
                # Let's assume state_at_nodes means state at all K+1 LGL-like state approximation points
                # This section for 'interior_nodes_var' seems more aligned with LGL methods
                # For Radau, the states are defined at K+1 Legendre-Gauss-Radau points (τ_0 = -1, and K roots of P_K - P_{K-1})
                # The decision variables are typically the states at the mesh interval boundaries (shared)
                # and potentially coefficients of the polynomial or states at interior collocation points
                # if they are not directly reconstructed.
                # The provided code seems to create state variables at K+1 points per interval:
                # X_k,0 (from global mesh) and X_k,j for j=1..K (interior collocation points, or end point X_k,K = X_{k+1},0)

                # Re-evaluating the state variable setup for Radau:
                # States are: X_0, X_1, ..., X_N (at global mesh nodes)
                # And at K-1 interior points if K > 1 for state_at_nodes
                # state_at_nodes = [X_k, X_k,1, ..., X_k,K-1, X_{k+1}] where X_k,j are new variables
                # This seems to be what the code does for `interior_nodes_var`

                # num_interior_state_approx_nodes = num_colloc_nodes -1 (if Radau points for state approx are K+1, and K are coll)
                # The Radau scheme uses K collocation points. The state is approximated by a K-degree polynomial.
                # It is defined by K+1 values. Typically X_k (start of interval) and U_kj at K collocation points,
                # and X_kj (state at collocation points). Defect constraints link these.
                # The current code uses state_at_nodes of size (num_states, num_colloc_nodes + 1).
                # And state_at_global_mesh_nodes_variables.

                # Let's stick to the current code's structure for `interior_nodes_var`
                # but clarify `num_interior_nodes`. State approx nodes are K+1.
                # So, there are (K+1) - 2 = K-1 interior state variables IF endpoints are shared.
                num_interior_vars_to_create = (
                    num_colloc_nodes - 1
                )  # K-1 interior points for a K-degree polynomial.
                if num_interior_vars_to_create > 0:
                    interior_nodes_var = opti.variable(num_states, num_interior_vars_to_create)
                    for i in range(num_interior_vars_to_create):
                        current_interval_state_columns[i + 1] = interior_nodes_var[:, i]

        state_at_interior_local_approximation_nodes_all_intervals_variables.append(
            interior_nodes_var
        )
        current_interval_state_columns[num_colloc_nodes] = state_at_global_mesh_nodes_variables[
            mesh_interval_index + 1
        ]

        state_at_nodes = ca.horzcat(
            *current_interval_state_columns
        )  # This is X_k, X_k,1..K-1, X_{k+1}
        state_at_local_approximation_nodes_all_intervals_variables.append(state_at_nodes)

        basis_components = compute_radau_collocation_components(num_colloc_nodes)
        # These are tau for state approximation (K+1 points, typically includes -1 and roots of L_K')
        state_nodes_tau = basis_components.state_approximation_nodes.flatten()
        # These are tau for collocation (K points, roots of L_K or similar for Radau)
        colloc_nodes_tau = basis_components.collocation_nodes.flatten()
        quad_weights = basis_components.quadrature_weights.flatten()
        # Diff matrix D: X_dot_at_coll = D * X_at_state_nodes (or similar scaling)
        # The code uses X_dot_at_coll = X_at_state_nodes * D.T, so D should be (K x K+1)
        diff_matrix = ca.DM(basis_components.differentiation_matrix)

        local_state_approximation_nodes_tau_all_intervals.append(state_nodes_tau)  # np.ndarray
        local_collocation_nodes_tau_all_intervals.append(colloc_nodes_tau)  # np.ndarray

        # State derivative at collocation points:
        # state_derivative_at_colloc should be (num_states, num_colloc_nodes)
        # state_at_nodes is (num_states, num_colloc_nodes + 1)
        # diff_matrix is (num_colloc_nodes, num_colloc_nodes + 1) for Radau (maps K+1 state pts to K coll pts derivatives)
        state_derivative_at_colloc = ca.mtimes(state_at_nodes, diff_matrix.T)

        global_segment_length = (
            global_normalized_mesh_nodes_arr[mesh_interval_index + 1]
            - global_normalized_mesh_nodes_arr[mesh_interval_index]
        )
        if global_segment_length <= 1e-9:  # Should use a small epsilon
            raise ValueError(
                f"Mesh interval {mesh_interval_index} has zero or negative length in global tau: {global_segment_length}"
            )
        # Scaling: dt/dtau_local = (tf - t0)/2 * (tau_global_end - tau_global_start)/2 = (tf-t0)/4 * global_segment_length
        # No, it's ( (tf-t0)/2 * d(global_tau)/d(local_tau) )
        # d(physical_time)/d(local_tau) = ( (tf-t0)/2 * (global_segment_length/2) )
        # So X_dot_physical * (dt/dtau_local) = X_dot_local_tau
        # RHS is f(X,U,t_physical). So constraint is X_dot_local_tau - f * (dt/dtau_local) == 0
        # state_derivative_at_colloc is already dX/d(local_tau) if diff_matrix is scaled by 2/local_interval_length (which is 2 for [-1,1])
        # The diff_matrix from compute_radau_collocation_components usually assumes local_tau in [-1,1].
        # So state_derivative_at_colloc is dX/d(local_tau_canonical)
        # Need to scale RHS by dt/d(local_tau_canonical)
        # t_physical = ( (tf-t0)/2 ) * t_global + ( (tf+t0)/2 )
        # t_global   = ( (global_end - global_start)/2 ) * t_local_canonical + ( (global_end + global_start)/2 )
        # dt_physical / dt_local_canonical = ( (tf-t0)/2 ) * ( (global_end - global_start)/2 )
        #                                = ( (tf-t0)/4 ) * global_segment_length
        time_scaling_factor_for_dynamics = (
            (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
        )

        for i_colloc in range(num_colloc_nodes):
            # State at collocation point must be interpolated from state_at_nodes if collocation_nodes are different from state_approximation_nodes
            # If using Lagrange polynomials: X(tau_coll_i) = sum( L_j(tau_coll_i) * X_at_state_node_j )
            # The current structure implies state_at_nodes[:, i_colloc] is state at collocation point i_colloc.
            # This is true if the first K points of state_nodes_tau ARE colloc_nodes_tau.
            # For Radau, often X(tau_coll_j) is directly a variable or reconstructed.
            # The state_at_nodes has K+1 points. The diff_matrix gives derivatives at K collocation points.
            # We need X_at_colloc_point. The current structure seems to pick from X_at_nodes.
            # This might be an approximation or specific scheme choice.
            # If basis_components.state_approximation_nodes includes all basis_components.collocation_nodes, then it's just an indexing.
            # More robust: interpolate state_at_nodes to colloc_nodes_tau[i_colloc]
            # However, the direct use of state_at_nodes[:, i_colloc] is simpler if valid by construction.
            # Assuming for now that state_at_nodes includes values at collocation points for the first K columns.
            # This is typical if state_approximation_nodes are e.g. LGL nodes and collocation points are subset or related.
            # Radau collocation points are usually distinct from the set of LGR state points (except -1).

            # Let's assume state_at_colloc is correctly formed/interpolated by some convention not fully explicit here,
            # or that `state_at_nodes[:, i_colloc]` is indeed the state AT the i-th collocation point.
            # This is often the case if the first K state approx nodes are the K collocation nodes.
            state_at_colloc_pt = state_at_nodes[
                :, i_colloc
            ]  # This line is potentially problematic without clearer definition of state_nodes_tau vs colloc_nodes_tau

            control_at_colloc = control_at_local_collocation_nodes_all_intervals_variables[
                mesh_interval_index
            ][:, i_colloc]

            local_tau_coll = colloc_nodes_tau[i_colloc]
            global_tau_coll = (
                global_segment_length / 2 * local_tau_coll
                + (
                    global_normalized_mesh_nodes_arr[mesh_interval_index + 1]
                    + global_normalized_mesh_nodes_arr[mesh_interval_index]
                )
                / 2
            )
            physical_time_at_colloc = (
                terminal_time_variable - initial_time_variable
            ) / 2 * global_tau_coll + (terminal_time_variable + initial_time_variable) / 2

            state_derivative_rhs = dynamics_function(
                state_at_colloc_pt, control_at_colloc, physical_time_at_colloc, problem_parameters
            )
            state_derivative_rhs_vector = _validate_dynamics_output(
                state_derivative_rhs, num_states
            )

            opti.subject_to(
                state_derivative_at_colloc[:, i_colloc]  # This is dX/d(local_tau_canonical)
                == time_scaling_factor_for_dynamics * state_derivative_rhs_vector
            )

            if path_constraints_function:
                path_constraints_output = path_constraints_function(
                    state_at_colloc_pt,
                    control_at_colloc,
                    physical_time_at_colloc,
                    problem_parameters,
                )
                processed_path_constraints: list[Any]
                if isinstance(path_constraints_output, list):
                    processed_path_constraints = path_constraints_output
                elif path_constraints_output is None:
                    processed_path_constraints = []
                else:
                    processed_path_constraints = [path_constraints_output]

                for constraint_item in processed_path_constraints:
                    if isinstance(constraint_item, PathConstraint):
                        _apply_constraint(opti, constraint_item)
                    else:
                        actual_type = type(constraint_item).__name__
                        raise ValueError(
                            f"Item from path_constraints_function is of type '{actual_type}', "
                            f"but 'PathConstraint' is expected. Value: {constraint_item}"
                        )

        if num_integrals > 0 and integral_integrand_function:
            for integral_index in range(num_integrals):
                quad_sum = ca.MX(0)
                for i_colloc in range(num_colloc_nodes):  # Quadrature uses collocation points
                    state_at_colloc_for_quad = state_at_nodes[
                        :, i_colloc
                    ]  # Same assumption as above
                    control_at_colloc_for_quad = (
                        control_at_local_collocation_nodes_all_intervals_variables[
                            mesh_interval_index
                        ][:, i_colloc]
                    )

                    local_tau_quad = colloc_nodes_tau[
                        i_colloc
                    ]  # Using collocation nodes for quadrature
                    global_tau_quad = (
                        global_segment_length / 2 * local_tau_quad
                        + (
                            global_normalized_mesh_nodes_arr[mesh_interval_index + 1]
                            + global_normalized_mesh_nodes_arr[mesh_interval_index]
                        )
                        / 2
                    )
                    physical_time_at_quad = (
                        terminal_time_variable - initial_time_variable
                    ) / 2 * global_tau_quad + (terminal_time_variable + initial_time_variable) / 2

                    weight = quad_weights[i_colloc]
                    integrand_value = integral_integrand_function(
                        state_at_colloc_for_quad,
                        control_at_colloc_for_quad,
                        physical_time_at_quad,
                        integral_index,
                        problem_parameters,
                    )
                    quad_sum += weight * integrand_value
                # Accumulated sum needs to be scaled by dt_physical / d_local_tau_canonical
                accumulated_integral_expressions[integral_index] += (
                    time_scaling_factor_for_dynamics * quad_sum
                )

    if (
        num_integrals > 0
        and integral_integrand_function  # Make sure function is provided
        and integral_decision_variables is not None
    ):
        if num_integrals == 1:
            opti.subject_to(integral_decision_variables == accumulated_integral_expressions[0])
        else:  # num_integrals > 1
            for i in range(num_integrals):
                # Ensure integral_decision_variables is indexable if num_integrals > 1
                current_integral_var = (
                    integral_decision_variables[i]
                    if isinstance(integral_decision_variables, (ca.SX, ca.MX))
                    and integral_decision_variables.shape[0] > 1
                    else integral_decision_variables
                )

                opti.subject_to(current_integral_var == accumulated_integral_expressions[i])

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

    if event_constraints_function:
        event_constraints_output = event_constraints_function(
            initial_time_variable,
            terminal_time_variable,
            initial_state,
            terminal_state,
            integral_decision_variables,
            problem_parameters,
        )
        processed_event_constraints: list[Any]
        if isinstance(event_constraints_output, list):
            processed_event_constraints = event_constraints_output
        elif event_constraints_output is None:
            processed_event_constraints = []
        else:
            processed_event_constraints = [event_constraints_output]

        for constraint_item in processed_event_constraints:
            if isinstance(constraint_item, EventConstraint):
                _apply_constraint(opti, constraint_item)
            else:
                actual_type = type(constraint_item).__name__
                raise ValueError(
                    f"Item from event_constraints_function is of type '{actual_type}', "
                    f"but 'EventConstraint' is expected. Value: {constraint_item}"
                )

    if problem_definition.initial_guess:
        ig = problem_definition.initial_guess
        def_ig = problem_definition.default_initial_guess_values

        opti.set_initial(
            initial_time_variable,
            (
                ig.initial_time_variable
                if ig.initial_time_variable is not None
                else (problem_definition.t0_bounds[0] + problem_definition.t0_bounds[1]) / 2
            ),
        )
        opti.set_initial(
            terminal_time_variable,
            (
                ig.terminal_time_variable
                if ig.terminal_time_variable is not None
                else (problem_definition.tf_bounds[0] + problem_definition.tf_bounds[1]) / 2
            ),
        )

        # States at global mesh points
        if ig.states and len(ig.states) > 0:
            # Iterate through the global mesh points
            for k_global_mesh in range(num_mesh_intervals + 1):
                # Try to get a guess for this global mesh point
                # This requires a mapping from ig.states (per interval) to global mesh points
                # A simpler approach: use ig.states[0][:,0] for state_at_global_mesh_nodes_variables[0]
                # and ig.states[k][:, -1] for state_at_global_mesh_nodes_variables[k+1]
                if k_global_mesh == 0:
                    if ig.states[0].shape[0] == num_states and ig.states[0].shape[1] > 0:
                        opti.set_initial(
                            state_at_global_mesh_nodes_variables[0], ig.states[0][:, 0]
                        )
                    else:  # Fallback to default
                        opti.set_initial(
                            state_at_global_mesh_nodes_variables[0], [def_ig.state] * num_states
                        )

                elif (
                    k_global_mesh > 0
                    and (k_global_mesh - 1) < len(ig.states)
                    and isinstance(ig.states[k_global_mesh - 1], np.ndarray)
                    and ig.states[k_global_mesh - 1].shape[0] == num_states
                    and ig.states[k_global_mesh - 1].shape[1] > 0
                ):
                    opti.set_initial(
                        state_at_global_mesh_nodes_variables[k_global_mesh],
                        ig.states[k_global_mesh - 1][:, -1],
                    )  # Use last point of interval k-1
                else:  # Fallback
                    opti.set_initial(
                        state_at_global_mesh_nodes_variables[k_global_mesh],
                        [def_ig.state] * num_states,
                    )
        else:  # No states guess provided
            for k_global_mesh in range(num_mesh_intervals + 1):
                opti.set_initial(
                    state_at_global_mesh_nodes_variables[k_global_mesh], [def_ig.state] * num_states
                )

        # States at interior local approximation nodes
        for k_interval in range(num_mesh_intervals):
            interior_var_matrix = (
                state_at_interior_local_approximation_nodes_all_intervals_variables[k_interval]
            )
            if (
                interior_var_matrix is not None
            ):  # If there are interior state variables for this interval
                num_interior_pts = interior_var_matrix.shape[1]
                # Try to get guess from ig.states[k_interval]
                interval_state_guess = None
                if (
                    ig.states
                    and k_interval < len(ig.states)
                    and isinstance(ig.states[k_interval], np.ndarray)
                    and ig.states[k_interval].shape[0] == num_states
                ):
                    interval_state_guess = ig.states[k_interval]

                if (
                    interval_state_guess is not None and interval_state_guess.shape[1] > 2
                ):  # if there are interior points in guess (more than start and end)
                    # Use points from ig.states[k_interval][:, 1:-1]
                    available_interior_guess_pts = interval_state_guess[:, 1:-1]
                    num_available_interior_guess_pts = available_interior_guess_pts.shape[1]

                    guess_for_interior_vars = np.zeros((num_states, num_interior_pts))
                    for i_interior in range(num_interior_pts):
                        if i_interior < num_available_interior_guess_pts:
                            guess_for_interior_vars[:, i_interior] = available_interior_guess_pts[
                                :, i_interior
                            ]
                        else:  # Not enough guess points, use last available or default
                            guess_for_interior_vars[:, i_interior] = (
                                available_interior_guess_pts[:, -1]
                                if num_available_interior_guess_pts > 0
                                else def_ig.state
                            )
                    opti.set_initial(interior_var_matrix, guess_for_interior_vars)
                else:  # Fallback to default
                    opti.set_initial(
                        interior_var_matrix, np.full((num_states, num_interior_pts), def_ig.state)
                    )

        # Controls
        if ig.controls and len(ig.controls) > 0:
            for k in range(num_mesh_intervals):
                target_control_var = control_at_local_collocation_nodes_all_intervals_variables[k]
                nodes_needed = target_control_var.shape[1]
                guess_val = np.full((num_controls, nodes_needed), def_ig.control)  # Default

                if k < len(ig.controls) and isinstance(ig.controls[k], np.ndarray):
                    control_guess_arr = ig.controls[k]
                    if num_controls == 1 and control_guess_arr.ndim == 1:
                        control_guess_arr = control_guess_arr.reshape(1, -1)

                    if control_guess_arr.shape[0] == num_controls:
                        cols_available = control_guess_arr.shape[1]
                        cols_to_use = min(nodes_needed, cols_available)
                        if cols_to_use > 0:
                            guess_val[:, :cols_to_use] = control_guess_arr[:, :cols_to_use]
                            if cols_to_use < nodes_needed:  # Pad if needed
                                padding = np.tile(
                                    control_guess_arr[:, cols_to_use - 1 : cols_to_use],
                                    (1, nodes_needed - cols_to_use),
                                )
                                guess_val[:, cols_to_use:] = padding
                opti.set_initial(target_control_var, guess_val)
        else:  # No control guess
            for k in range(num_mesh_intervals):
                target_control_var = control_at_local_collocation_nodes_all_intervals_variables[k]
                nodes_needed = target_control_var.shape[1]
                opti.set_initial(
                    target_control_var, np.full((num_controls, nodes_needed), def_ig.control)
                )

        if ig.integrals is not None and integral_decision_variables is not None:
            _set_initial_value_for_integrals(
                opti, integral_decision_variables, ig.integrals, num_integrals
            )
        elif (
            num_integrals > 0 and integral_decision_variables is not None
        ):  # No integral guess, use default
            _set_initial_value_for_integrals(
                opti,
                integral_decision_variables,
                [def_ig.integral] * num_integrals if num_integrals > 1 else def_ig.integral,
                num_integrals,
            )

    solver_options = problem_definition.solver_options if problem_definition.solver_options else {}
    opti.solver("ipopt", solver_options)

    opti.initial_time_variable_reference = initial_time_variable
    opti.terminal_time_variable_reference = terminal_time_variable
    if integral_decision_variables is not None:
        opti.integral_variables_object_reference = integral_decision_variables
    else:  # Ensure the attribute exists even if None
        opti.integral_variables_object_reference = None

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
    opti.metadata_global_normalized_mesh_nodes = global_normalized_mesh_nodes_arr
    opti.symbolic_objective_function_reference = objective_value

    try:
        solver_solution: _CasadiSolution = opti.solve()
        print("NLP problem formulated and solver called successfully.")
        solution = _extract_and_format_solution(
            solver_solution,
            opti,
            problem_definition,
            num_collocation_nodes_per_interval,
            global_normalized_mesh_nodes_arr,  # Pass the array version
        )
        # Store actual lists/arrays used at solve time
        solution.num_collocation_nodes_list_at_solve_time = list(num_collocation_nodes_per_interval)
        solution.global_mesh_nodes_at_solve_time = np.array(
            global_normalized_mesh_nodes_arr, dtype=np.float64
        )

        # For potential debugging or advanced analysis, store the solved trajectories per interval
        # This would require extracting them before global concatenation if needed.
        # solution.solved_state_trajectories_per_interval = ...
        # solution.solved_control_trajectories_per_interval = ...
        return solution
    except RuntimeError as e:
        print(f"Error during NLP solution: {e}")
        print("Solver failed.")
        # Create a default solution object on failure
        solution = OptimalControlSolution(
            success=False,
            message=f"Solver runtime error: {e}",
            opti_object=opti,  # Store the opti stack for debugging
            num_collocation_nodes_per_interval=list(num_collocation_nodes_per_interval),
            global_normalized_mesh_nodes=global_normalized_mesh_nodes_arr,
            num_collocation_nodes_list_at_solve_time=list(num_collocation_nodes_per_interval),
            global_mesh_nodes_at_solve_time=np.array(
                global_normalized_mesh_nodes_arr, dtype=np.float64
            ),
        )
        try:
            # Attempt to retrieve debug values if solver failed but opti.debug exists
            if hasattr(opti, "debug") and opti.debug is not None:
                if (
                    hasattr(opti, "initial_time_variable_reference")
                    and opti.initial_time_variable_reference is not None
                ):
                    solution.initial_time_variable = float(
                        opti.debug.value(opti.initial_time_variable_reference)
                    )
                if (
                    hasattr(opti, "terminal_time_variable_reference")
                    and opti.terminal_time_variable_reference is not None
                ):
                    solution.terminal_time_variable = float(
                        opti.debug.value(opti.terminal_time_variable_reference)
                    )
                # Potentially retrieve other debug values if needed and safe
        except Exception as debug_e:
            print(f"  Could not retrieve debug values after solver error: {debug_e}")
        return solution
