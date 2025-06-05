import logging

import numpy as np

from trajectolab.adaptive.phs.numerical import (
    PolynomialInterpolant,
    _map_global_normalized_tau_to_local_interval_tau,
)
from trajectolab.exceptions import ConfigurationError, DataIntegrityError, InterpolationError
from trajectolab.radau import _compute_radau_collocation_components
from trajectolab.tl_types import (
    FloatArray,
    MultiPhaseInitialGuess,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


__all__ = ["_propagate_multiphase_solution_to_new_meshes"]

logger = logging.getLogger(__name__)


def _find_containing_interval_index(global_tau: float, mesh_points: FloatArray) -> int | None:
    """Find interval containing global_tau."""
    if mesh_points.ndim != 1 or mesh_points.size < 2:
        return None

    tolerance = 1e-10
    if global_tau < mesh_points[0] - tolerance or global_tau > mesh_points[-1] + tolerance:
        return (
            0
            if abs(global_tau - mesh_points[0]) < tolerance
            else (len(mesh_points) - 2 if abs(global_tau - mesh_points[-1]) < tolerance else None)
        )

    if abs(global_tau - mesh_points[-1]) < tolerance:
        return len(mesh_points) - 2

    return min(
        max(0, int(np.searchsorted(mesh_points, global_tau, side="right")) - 1),
        len(mesh_points) - 2,
    )


def _determine_interpolation_parameters(
    global_tau: float, prev_mesh_points: FloatArray
) -> tuple[int, float]:
    """Pure interpolation parameter calculation."""
    prev_interval_idx = _find_containing_interval_index(global_tau, prev_mesh_points)

    if prev_interval_idx is None:
        if global_tau < prev_mesh_points[0]:
            return 0, -1.0
        elif global_tau > prev_mesh_points[-1]:
            return len(prev_mesh_points) - 2, 1.0
        else:
            raise InterpolationError(
                f"Could not locate global_tau {global_tau} in mesh boundaries",
                "Mesh boundary mapping error",
            )

    prev_tau_start, prev_tau_end = (
        prev_mesh_points[prev_interval_idx],
        prev_mesh_points[prev_interval_idx + 1],
    )
    return prev_interval_idx, _map_global_normalized_tau_to_local_interval_tau(
        global_tau, prev_tau_start, prev_tau_end
    )


def _validate_interpolation_inputs(
    prev_trajectory_per_interval: list[FloatArray],
    prev_polynomial_degrees: list[int],
    target_polynomial_degrees: list[int],
    phase_id: PhaseID,
) -> None:
    """Validate interpolation input data consistency."""
    if (
        len(prev_trajectory_per_interval) != len(prev_polynomial_degrees)
        or not prev_trajectory_per_interval
        or not target_polynomial_degrees
    ):
        raise InterpolationError(
            f"Phase {phase_id} trajectory/degree data inconsistency or empty",
            "Input data inconsistency",
        )


def _create_phase_interpolants(
    prev_trajectory_per_interval: list[FloatArray],
    prev_polynomial_degrees: list[int],
    is_state_trajectory: bool,
) -> list[PolynomialInterpolant]:
    """Create polynomial interpolants for previous trajectory data."""
    prev_interpolants = []

    for N_k, traj_k in zip(prev_polynomial_degrees, prev_trajectory_per_interval, strict=False):
        basis_components = _compute_radau_collocation_components(N_k)

        if is_state_trajectory:
            local_nodes = basis_components.state_approximation_nodes
            barycentric_weights = basis_components.barycentric_weights_for_state_nodes
        else:
            from trajectolab.radau import _compute_barycentric_weights

            local_nodes = basis_components.collocation_nodes
            barycentric_weights = _compute_barycentric_weights(basis_components.collocation_nodes)

        prev_interpolants.append(PolynomialInterpolant(local_nodes, traj_k, barycentric_weights))

    return prev_interpolants


def _compute_target_interval_nodes(
    target_mesh_points: FloatArray, k: int, target_local_nodes: FloatArray
) -> FloatArray:
    """Compute global tau points for target interval."""
    target_tau_start, target_tau_end = target_mesh_points[k], target_mesh_points[k + 1]
    tau_range = target_tau_end - target_tau_start
    tau_offset = target_tau_start + target_tau_end
    return np.asarray((tau_range * target_local_nodes + tau_offset) / 2.0, dtype=np.float64)


def _get_target_basis_nodes(N_k_target: int, is_state_trajectory: bool) -> tuple[FloatArray, int]:
    """Get basis nodes and count for target interval."""
    target_basis = _compute_radau_collocation_components(N_k_target)

    if is_state_trajectory:
        target_local_nodes = target_basis.state_approximation_nodes
        num_target_nodes = N_k_target + 1
    else:
        target_local_nodes = target_basis.collocation_nodes
        num_target_nodes = N_k_target

    return target_local_nodes, num_target_nodes


def _interpolate_at_points(
    global_tau_points: FloatArray,
    prev_mesh_points: FloatArray,
    prev_interpolants: list[PolynomialInterpolant],
    num_variables: int,
    phase_id: PhaseID,
) -> FloatArray:
    """Interpolate trajectory values at given global tau points."""
    num_points = len(global_tau_points)
    target_values = np.zeros((num_variables, num_points), dtype=np.float64)

    for j, global_tau in enumerate(global_tau_points):
        prev_interval_idx, prev_local_tau = _determine_interpolation_parameters(
            global_tau, prev_mesh_points
        )
        interpolated_values = prev_interpolants[prev_interval_idx](prev_local_tau)

        if interpolated_values.ndim > 1:
            interpolated_values = interpolated_values.flatten()

        # Validate only at first point to catch early issues
        if j == 0 and (
            np.any(np.isnan(interpolated_values)) or np.any(np.isinf(interpolated_values))
        ):
            raise DataIntegrityError(
                f"Numerical corruption in interpolation result for phase {phase_id}",
                "Interpolation result validation",
            )

        if len(interpolated_values) == num_variables:
            target_values[:, j] = interpolated_values
        elif num_variables != 0:
            raise InterpolationError(
                f"Phase {phase_id} dimension mismatch: interpolated {len(interpolated_values)} values, expected {num_variables}",
                "Interpolation output dimension error",
            )

    return target_values


def _process_single_target_interval(
    k: int,
    N_k_target: int,
    target_mesh_points: FloatArray,
    prev_mesh_points: FloatArray,
    prev_interpolants: list[PolynomialInterpolant],
    num_variables: int,
    phase_id: PhaseID,
    is_state_trajectory: bool,
) -> FloatArray:
    """Process a single target interval for interpolation."""
    target_local_nodes, num_target_nodes = _get_target_basis_nodes(N_k_target, is_state_trajectory)
    global_tau_points = _compute_target_interval_nodes(target_mesh_points, k, target_local_nodes)

    return _interpolate_at_points(
        global_tau_points, prev_mesh_points, prev_interpolants, num_variables, phase_id
    )


def _interpolate_phase_trajectory_to_new_mesh_streamlined(
    prev_trajectory_per_interval: list[FloatArray],
    prev_mesh_points: FloatArray,
    prev_polynomial_degrees: list[int],
    target_mesh_points: FloatArray,
    target_polynomial_degrees: list[int],
    num_variables: int,
    phase_id: PhaseID,
    is_state_trajectory: bool = True,
) -> list[FloatArray]:
    """VECTORIZED interpolation with direct allocation."""
    _validate_interpolation_inputs(
        prev_trajectory_per_interval, prev_polynomial_degrees, target_polynomial_degrees, phase_id
    )

    prev_interpolants = _create_phase_interpolants(
        prev_trajectory_per_interval, prev_polynomial_degrees, is_state_trajectory
    )

    target_trajectories = []
    for k, N_k_target in enumerate(target_polynomial_degrees):
        target_traj_k = _process_single_target_interval(
            k,
            N_k_target,
            target_mesh_points,
            prev_mesh_points,
            prev_interpolants,
            num_variables,
            phase_id,
            is_state_trajectory,
        )
        target_trajectories.append(target_traj_k)

    return target_trajectories


def _validate_propagation_preconditions(prev_solution: OptimalControlSolution) -> None:
    """Validate solution is suitable for propagation."""
    if not prev_solution.success:
        raise InterpolationError(
            "Cannot propagate from unsuccessful previous unified solution",
            "Invalid source solution for propagation",
        )


def _validate_target_configuration(
    phase_id: PhaseID,
    target_phase_polynomial_degrees: dict[PhaseID, list[int]],
    target_phase_mesh_points: dict[PhaseID, FloatArray],
) -> tuple[list[int], FloatArray]:
    """Validate and extract target configuration for phase."""
    if phase_id not in target_phase_polynomial_degrees or phase_id not in target_phase_mesh_points:
        raise ConfigurationError(
            f"Missing target mesh configuration for phase {phase_id}",
            "Target mesh configuration error",
        )

    target_degrees = target_phase_polynomial_degrees[phase_id]
    target_mesh = target_phase_mesh_points[phase_id]

    if len(target_degrees) != len(target_mesh) - 1:
        raise ConfigurationError(
            f"Phase {phase_id} target polynomial degrees count ({len(target_degrees)}) != target mesh intervals ({len(target_mesh) - 1})",
            "Target mesh configuration error",
        )

    return target_degrees, target_mesh


def _validate_previous_solution_data(
    prev_solution: OptimalControlSolution, phase_id: PhaseID
) -> tuple[list[FloatArray], list[FloatArray], FloatArray, list[int]]:
    """Validate and extract previous solution data for phase."""
    required_keys = [
        "phase_solved_state_trajectories_per_interval",
        "phase_solved_control_trajectories_per_interval",
        "phase_mesh_nodes",
        "phase_mesh_intervals",
    ]
    if not all(phase_id in getattr(prev_solution, key) for key in required_keys):
        raise InterpolationError(
            f"Previous solution missing required data for phase {phase_id}",
            "Missing source data",
        )

    prev_states = prev_solution.phase_solved_state_trajectories_per_interval[phase_id]
    prev_controls = prev_solution.phase_solved_control_trajectories_per_interval[phase_id]
    prev_mesh = prev_solution.phase_mesh_nodes[phase_id]
    prev_degrees = prev_solution.phase_mesh_intervals[phase_id]

    if len(prev_states) != len(prev_degrees) or len(prev_controls) != len(prev_degrees):
        raise InterpolationError(
            f"Phase {phase_id} previous data inconsistency",
            "Previous solution data inconsistency",
        )

    return prev_states, prev_controls, prev_mesh, prev_degrees


def _interpolate_phase_data(
    prev_states: list[FloatArray],
    prev_controls: list[FloatArray],
    prev_mesh: FloatArray,
    prev_degrees: list[int],
    target_mesh: FloatArray,
    target_degrees: list[int],
    num_states: int,
    num_controls: int,
    phase_id: PhaseID,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Interpolate both state and control data for phase."""
    phase_states = _interpolate_phase_trajectory_to_new_mesh_streamlined(
        prev_states,
        prev_mesh,
        prev_degrees,
        target_mesh,
        target_degrees,
        num_states,
        phase_id,
        True,
    )

    phase_controls = _interpolate_phase_trajectory_to_new_mesh_streamlined(
        prev_controls,
        prev_mesh,
        prev_degrees,
        target_mesh,
        target_degrees,
        num_controls,
        phase_id,
        False,
    )

    return phase_states, phase_controls


def _extract_preserved_variables(
    prev_solution: OptimalControlSolution,
) -> tuple[dict, dict, dict, FloatArray | None]:
    """Extract variables that are preserved during propagation."""
    return (
        prev_solution.phase_initial_times.copy(),
        prev_solution.phase_terminal_times.copy(),
        prev_solution.phase_integrals.copy(),
        prev_solution.static_parameters,
    )


def _validate_final_guess(initial_guess: MultiPhaseInitialGuess, problem: ProblemProtocol) -> None:
    """Validate final initial guess has all required phase data."""
    for phase_id in problem._get_phase_ids():
        if (
            phase_id not in initial_guess.phase_states
            or phase_id not in initial_guess.phase_controls
        ):
            raise InterpolationError(f"Missing interpolated data for phase {phase_id}")


def _propagate_multiphase_solution_to_new_meshes(
    prev_solution: OptimalControlSolution,
    problem: ProblemProtocol,
    target_phase_polynomial_degrees: dict[PhaseID, list[int]],
    target_phase_mesh_points: dict[PhaseID, FloatArray],
) -> MultiPhaseInitialGuess:
    """VECTORIZED multiphase solution propagation."""
    _validate_propagation_preconditions(prev_solution)

    # Extract preserved variables
    (
        phase_initial_times,
        phase_terminal_times,
        phase_integrals,
        static_parameters,
    ) = _extract_preserved_variables(prev_solution)

    phase_states, phase_controls = {}, {}

    for phase_id in problem._get_phase_ids():
        # Validate target configuration
        target_degrees, target_mesh = _validate_target_configuration(
            phase_id, target_phase_polynomial_degrees, target_phase_mesh_points
        )

        # Validate previous data
        prev_states, prev_controls, prev_mesh, prev_degrees = _validate_previous_solution_data(
            prev_solution, phase_id
        )

        num_states, num_controls = problem._get_phase_variable_counts(phase_id)

        # Interpolate phase data
        phase_states[phase_id], phase_controls[phase_id] = _interpolate_phase_data(
            prev_states,
            prev_controls,
            prev_mesh,
            prev_degrees,
            target_mesh,
            target_degrees,
            num_states,
            num_controls,
            phase_id,
        )

    initial_guess = MultiPhaseInitialGuess(
        phase_states=phase_states,
        phase_controls=phase_controls,
        phase_initial_times=phase_initial_times,
        phase_terminal_times=phase_terminal_times,
        phase_integrals=phase_integrals,
        static_parameters=static_parameters,
    )

    _validate_final_guess(initial_guess, problem)
    return initial_guess
