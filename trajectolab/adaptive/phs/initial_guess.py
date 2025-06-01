"""
Streamlined initial guess propagation and interpolation for multiphase adaptive mesh refinement.
BLOAT ELIMINATED: Removed memory pooling, simplified validation, direct array allocation.
DEAD CODE REMOVED: Eliminated verified dead code from Phase 1 verification.
"""

import logging
from typing import cast

import numpy as np

from trajectolab.adaptive.phs.numerical import (
    PolynomialInterpolant,
    map_global_normalized_tau_to_local_interval_tau,
    map_local_interval_tau_to_global_normalized_tau,
)
from trajectolab.exceptions import ConfigurationError, DataIntegrityError, InterpolationError
from trajectolab.radau import compute_radau_collocation_components
from trajectolab.tl_types import (
    FloatArray,
    MultiPhaseInitialGuess,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)


__all__ = [
    "propagate_multiphase_solution_to_new_meshes",
]

logger = logging.getLogger(__name__)


# ========================================================================
# MATHEMATICAL CORE FUNCTIONS - Pure calculations (UNCHANGED)
# ========================================================================


def _interpolate_polynomial_at_evaluation_points(
    nodes: FloatArray,
    values: FloatArray,
    evaluation_points: FloatArray,
    barycentric_weights: FloatArray,
) -> FloatArray:
    """Pure polynomial interpolation calculation - easily testable."""
    from trajectolab.radau import evaluate_lagrange_polynomial_at_point

    num_variables = values.shape[0]
    num_eval_points = len(evaluation_points)
    result = np.zeros((num_variables, num_eval_points), dtype=np.float64)

    for i, point in enumerate(evaluation_points):
        lagrange_coeffs = evaluate_lagrange_polynomial_at_point(nodes, barycentric_weights, point)
        result[:, i] = np.dot(values, lagrange_coeffs)

    return result


def _find_containing_interval_index(global_tau: float, mesh_points: FloatArray) -> int | None:
    """
    Finds the 0-indexed interval 'k' such that mesh_points[k] <= global_tau < mesh_points[k+1].
    """
    if not isinstance(mesh_points, np.ndarray) or mesh_points.ndim != 1 or mesh_points.size < 2:
        logger.error("Invalid mesh_points provided to _find_containing_interval_index.")
        return None

    tolerance = 1e-10

    # Handle points outside the mesh boundaries
    if global_tau < mesh_points[0]:
        if abs(global_tau - mesh_points[0]) < tolerance:
            return 0
        return None

    if global_tau > mesh_points[-1]:
        if abs(global_tau - mesh_points[-1]) < tolerance:
            return len(mesh_points) - 2
        return None

    # If global_tau is exactly the last point of the mesh
    if abs(global_tau - mesh_points[-1]) < tolerance:
        return len(mesh_points) - 2

    idx = np.searchsorted(mesh_points, global_tau, side="right")
    found_interval_idx = max(0, int(idx) - 1)

    # Safeguard for upper bound
    if found_interval_idx >= len(mesh_points) - 1:
        found_interval_idx = len(mesh_points) - 2

    return int(found_interval_idx)


def _calculate_global_tau_points_for_interval(
    target_local_nodes: FloatArray, target_tau_start: float, target_tau_end: float
) -> FloatArray:
    """Pure coordinate transformation calculation - easily testable."""
    return np.array(
        [
            map_local_interval_tau_to_global_normalized_tau(
                local_tau, target_tau_start, target_tau_end
            )
            for local_tau in target_local_nodes
        ],
        dtype=np.float64,
    )


def _determine_interpolation_parameters(
    global_tau: float, prev_mesh_points: FloatArray
) -> tuple[int, float]:
    """Pure interpolation parameter calculation - easily testable."""
    # Find which previous interval contains this global tau
    prev_interval_idx = _find_containing_interval_index(global_tau, prev_mesh_points)

    if prev_interval_idx is None:
        # Point is outside previous mesh - use boundary values
        if global_tau < prev_mesh_points[0]:
            return 0, -1.0
        elif global_tau > prev_mesh_points[-1]:
            return len(prev_mesh_points) - 2, 1.0
        else:
            raise InterpolationError(
                f"Could not locate global_tau {global_tau} in mesh boundaries",
                "Mesh boundary mapping error",
            )
    else:
        # Convert global tau to local tau for the containing previous interval
        prev_tau_start = prev_mesh_points[prev_interval_idx]
        prev_tau_end = prev_mesh_points[prev_interval_idx + 1]
        prev_local_tau = map_global_normalized_tau_to_local_interval_tau(
            global_tau, prev_tau_start, prev_tau_end
        )
        return prev_interval_idx, prev_local_tau


# ========================================================================
# STREAMLINED INTERPOLATION - Memory pooling removed
# ========================================================================


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
    """
    STREAMLINED interpolation with direct allocation - NO MEMORY POOLING.
    """
    trajectory_type = "state" if is_state_trajectory else "control"
    logger.info(
        f"    STREAMLINED interpolation of phase {phase_id} {trajectory_type} trajectories:"
    )

    # Validate input consistency
    if len(prev_trajectory_per_interval) != len(prev_polynomial_degrees):
        raise InterpolationError(
            f"Phase {phase_id} previous trajectory count ({len(prev_trajectory_per_interval)}) doesn't match polynomial degrees count ({len(prev_polynomial_degrees)})",
            "Input data inconsistency in interpolation",
        )

    if not prev_trajectory_per_interval or not target_polynomial_degrees:
        raise InterpolationError(
            f"Cannot interpolate phase {phase_id} with empty trajectory or polynomial degree data",
            "Missing interpolation input data",
        )

    # Create polynomial interpolants for each interval in previous solution
    prev_interpolants = []

    for k, (N_k, traj_k) in enumerate(
        zip(prev_polynomial_degrees, prev_trajectory_per_interval, strict=False)
    ):
        # Get the appropriate nodes for this interval type
        if is_state_trajectory:
            basis_components = compute_radau_collocation_components(N_k)
            local_nodes = basis_components.state_approximation_nodes
            barycentric_weights = basis_components.barycentric_weights_for_state_nodes
        else:
            basis_components = compute_radau_collocation_components(N_k)
            local_nodes = basis_components.collocation_nodes
            from trajectolab.radau import compute_barycentric_weights

            barycentric_weights = compute_barycentric_weights(local_nodes)

        # Create interpolant for this interval
        interpolant = PolynomialInterpolant(local_nodes, traj_k, barycentric_weights)
        prev_interpolants.append(interpolant)

    # Interpolate trajectory values for each target interval - DIRECT ALLOCATION
    target_trajectories = []

    for k, N_k_target in enumerate(target_polynomial_degrees):
        logger.debug(
            f"      Processing phase {phase_id} target interval {k} (degree {N_k_target})..."
        )

        # Get target nodes for this interval type
        if is_state_trajectory:
            target_basis = compute_radau_collocation_components(N_k_target)
            target_local_nodes = target_basis.state_approximation_nodes
            num_target_nodes = N_k_target + 1
        else:
            target_basis = compute_radau_collocation_components(N_k_target)
            target_local_nodes = target_basis.collocation_nodes
            num_target_nodes = N_k_target

        # STREAMLINED: Direct allocation instead of memory pooling
        target_traj_k = np.zeros((num_variables, num_target_nodes), dtype=np.float64)

        # Global interval boundaries for target interval k
        target_tau_start = target_mesh_points[k]
        target_tau_end = target_mesh_points[k + 1]

        # Use MATHEMATICAL CORE for coordinate transformation
        global_tau_points = _calculate_global_tau_points_for_interval(
            target_local_nodes, target_tau_start, target_tau_end
        )

        # Process all nodes for this interval
        for j, global_tau in enumerate(global_tau_points):
            # Use MATHEMATICAL CORE for interpolation parameter determination
            prev_interval_idx, prev_local_tau = _determine_interpolation_parameters(
                global_tau, prev_mesh_points
            )

            # Evaluate the interpolant at this local tau
            interpolated_values = prev_interpolants[prev_interval_idx](prev_local_tau)
            if interpolated_values.ndim > 1:
                interpolated_values = interpolated_values.flatten()

            # STREAMLINED: Simple NaN check instead of complex validation
            if np.any(np.isnan(interpolated_values)) or np.any(np.isinf(interpolated_values)):
                raise DataIntegrityError(
                    f"Numerical corruption in interpolation result for phase {phase_id}",
                    "Interpolation result validation",
                )

            # Store the interpolated values
            if len(interpolated_values) == num_variables:
                target_traj_k[:, j] = interpolated_values
            elif num_variables == 0:
                # Handle empty variable case
                pass
            else:
                raise InterpolationError(
                    f"Phase {phase_id} dimension mismatch: interpolated {len(interpolated_values)} values, expected {num_variables}",
                    "Interpolation output dimension error",
                )

        # STREAMLINED: No copy needed since we allocated directly
        target_trajectories.append(target_traj_k)

    logger.info(
        f"    ✓ Successfully interpolated phase {phase_id} {trajectory_type} trajectories (STREAMLINED)"
    )
    return cast(list[FloatArray], target_trajectories)


def propagate_multiphase_solution_to_new_meshes(
    prev_solution: OptimalControlSolution,
    problem: ProblemProtocol,
    target_phase_polynomial_degrees: dict[PhaseID, list[int]],
    target_phase_mesh_points: dict[PhaseID, FloatArray],
) -> MultiPhaseInitialGuess:
    """
    STREAMLINED multiphase solution propagation with direct allocation.
    DEAD CODE REMOVED: Eliminated verified conditional extraction logic.
    """
    logger.info("  Starting STREAMLINED multiphase interpolation-based propagation...")

    # Validate previous solution
    if not prev_solution.success:
        raise InterpolationError(
            "Cannot propagate from unsuccessful previous unified solution",
            "Invalid source solution for propagation",
        )

    # Extract time variables (preserved across mesh changes)
    phase_initial_times = prev_solution.phase_initial_times.copy()
    phase_terminal_times = prev_solution.phase_terminal_times.copy()
    phase_integrals = prev_solution.phase_integrals.copy()
    static_parameters = prev_solution.static_parameters

    # Interpolate trajectories for each phase
    phase_states = {}
    phase_controls = {}

    for phase_id in problem.get_phase_ids():
        logger.info(f"    Processing phase {phase_id} interpolation...")

        # Validate target mesh configuration for this phase
        if (
            phase_id not in target_phase_polynomial_degrees
            or phase_id not in target_phase_mesh_points
        ):
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

        # Get previous mesh information correctly for this phase
        if (
            phase_id not in prev_solution.phase_solved_state_trajectories_per_interval
            or phase_id not in prev_solution.phase_solved_control_trajectories_per_interval
            or phase_id not in prev_solution.phase_mesh_nodes
            or phase_id not in prev_solution.phase_mesh_intervals
        ):
            raise InterpolationError(
                f"Previous solution missing required data for phase {phase_id}",
                "Missing source data",
            )

        prev_states = prev_solution.phase_solved_state_trajectories_per_interval[phase_id]
        prev_controls = prev_solution.phase_solved_control_trajectories_per_interval[phase_id]
        prev_mesh = prev_solution.phase_mesh_nodes[phase_id]
        prev_degrees = prev_solution.phase_mesh_intervals[phase_id]

        # Validate consistency
        if len(prev_states) != len(prev_degrees) or len(prev_controls) != len(prev_degrees):
            raise InterpolationError(
                f"Phase {phase_id} previous data inconsistency",
                "Previous solution data inconsistency",
            )

        # Get problem dimensions for this phase
        num_states, num_controls = problem.get_phase_variable_counts(phase_id)

        # Use streamlined interpolation for this phase
        phase_states[phase_id] = _interpolate_phase_trajectory_to_new_mesh_streamlined(
            prev_trajectory_per_interval=prev_states,
            prev_mesh_points=prev_mesh,
            prev_polynomial_degrees=prev_degrees,
            target_mesh_points=target_mesh,
            target_polynomial_degrees=target_degrees,
            num_variables=num_states,
            phase_id=phase_id,
            is_state_trajectory=True,
        )

        phase_controls[phase_id] = _interpolate_phase_trajectory_to_new_mesh_streamlined(
            prev_trajectory_per_interval=prev_controls,
            prev_mesh_points=prev_mesh,
            prev_polynomial_degrees=prev_degrees,
            target_mesh_points=target_mesh,
            target_polynomial_degrees=target_degrees,
            num_variables=num_controls,
            phase_id=phase_id,
            is_state_trajectory=False,
        )

    # Create multiphase initial guess object
    initial_guess = MultiPhaseInitialGuess(
        phase_states=phase_states,
        phase_controls=phase_controls,
        phase_initial_times=phase_initial_times,
        phase_terminal_times=phase_terminal_times,
        phase_integrals=phase_integrals,
        static_parameters=static_parameters,
    )

    # STREAMLINED: Simple validation instead of complex validation infrastructure
    logger.info("  Validating interpolated multiphase initial guess...")
    _validate_interpolated_guess(initial_guess, problem)
    logger.info("  ✓ Completed STREAMLINED multiphase interpolation-based propagation successfully")

    return initial_guess


def _validate_interpolated_guess(
    initial_guess: MultiPhaseInitialGuess, problem: ProblemProtocol
) -> None:
    """STREAMLINED validation - essential checks only."""
    # Check that all required phases have data
    for phase_id in problem.get_phase_ids():
        if (
            phase_id not in initial_guess.phase_states
            or phase_id not in initial_guess.phase_controls
        ):
            raise InterpolationError(f"Missing interpolated data for phase {phase_id}")

        # Basic NaN check on first trajectory of each phase
        if (
            initial_guess.phase_states[phase_id]
            and initial_guess.phase_states[phase_id][0].size > 0
        ):
            if np.any(np.isnan(initial_guess.phase_states[phase_id][0])):
                raise DataIntegrityError(f"NaN values in interpolated states for phase {phase_id}")
