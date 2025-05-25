"""
Initial guess propagation and interpolation for adaptive mesh refinement.
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
from trajectolab.input_validation import validate_initial_guess_structure
from trajectolab.radau import compute_radau_collocation_components
from trajectolab.tl_types import FloatArray, InitialGuess, OptimalControlSolution, ProblemProtocol
from trajectolab.utils.memory_pool import create_buffer_context


__all__ = [
    "propagate_solution_to_new_mesh",
]

logger = logging.getLogger(__name__)


def _interpolate_trajectory_to_new_mesh_optimized(
    prev_trajectory_per_interval: list[FloatArray],
    prev_mesh_points: FloatArray,
    prev_polynomial_degrees: list[int],
    target_mesh_points: FloatArray,
    target_polynomial_degrees: list[int],
    num_variables: int,
    is_state_trajectory: bool = True,
) -> list[FloatArray]:
    """
    OPTIMIZED interpolation using memory pooling - ENHANCED WITH FAIL-FAST.
    """
    trajectory_type = "state" if is_state_trajectory else "control"
    logger.info(f"    OPTIMIZED interpolation of {trajectory_type} trajectories:")
    logger.info(
        f"      Previous mesh: {len(prev_mesh_points) - 1} intervals, degrees {prev_polynomial_degrees}"
    )
    logger.info(
        f"      Target mesh: {len(target_mesh_points) - 1} intervals, degrees {target_polynomial_degrees}"
    )

    # Guard clause: Validate input consistency
    if len(prev_trajectory_per_interval) != len(prev_polynomial_degrees):
        raise InterpolationError(
            f"Previous trajectory count ({len(prev_trajectory_per_interval)}) doesn't match polynomial degrees count ({len(prev_polynomial_degrees)})",
            "Input data inconsistency in interpolation",
        )

    # Guard clause: Check for empty inputs
    if not prev_trajectory_per_interval or not target_polynomial_degrees:
        raise InterpolationError(
            "Cannot interpolate with empty trajectory or polynomial degree data",
            "Missing interpolation input data",
        )

    try:
        # Use memory pool context for automatic buffer management
        with create_buffer_context() as buffer_pool:
            # Create polynomial interpolants for each interval in previous solution
            prev_interpolants = []

            for k, (N_k, traj_k) in enumerate(
                zip(prev_polynomial_degrees, prev_trajectory_per_interval, strict=False)
            ):
                # Critical: Validate trajectory shape
                expected_nodes = N_k + 1 if is_state_trajectory else N_k

                if traj_k.shape != (num_variables, expected_nodes):
                    raise InterpolationError(
                        f"Previous {trajectory_type} trajectory for interval {k} has shape {traj_k.shape}, expected ({num_variables}, {expected_nodes})",
                        "Shape mismatch in interpolation input",
                    )

                # Critical: Check for NaN/Inf in trajectory data
                if np.any(np.isnan(traj_k)) or np.any(np.isinf(traj_k)):
                    raise DataIntegrityError(
                        f"Previous {trajectory_type} trajectory for interval {k} contains NaN or Inf values",
                        "Numerical corruption in interpolation input",
                    )

                # Get the appropriate nodes for this interval type (cached via Radau cache)
                try:
                    if is_state_trajectory:
                        basis_components = compute_radau_collocation_components(N_k)
                        local_nodes = basis_components.state_approximation_nodes
                        barycentric_weights = basis_components.barycentric_weights_for_state_nodes
                    else:
                        basis_components = compute_radau_collocation_components(N_k)
                        local_nodes = basis_components.collocation_nodes
                        from trajectolab.radau import compute_barycentric_weights

                        barycentric_weights = compute_barycentric_weights(local_nodes)
                except Exception as e:
                    raise InterpolationError(
                        f"Failed to compute basis components for interval {k} with degree {N_k}: {e}",
                        "Radau basis computation error",
                    ) from e

                # Create interpolant for this interval
                try:
                    interpolant = PolynomialInterpolant(local_nodes, traj_k, barycentric_weights)
                    prev_interpolants.append(interpolant)
                    logger.debug(
                        f"        Interval {k}: Created interpolant for {traj_k.shape} trajectory"
                    )
                except Exception as e:
                    raise InterpolationError(
                        f"Failed to create interpolant for interval {k}: {e}",
                        "Polynomial interpolant construction error",
                    ) from e

            # Interpolate trajectory values for each target interval using pooled memory
            target_trajectories = []

            for k, N_k_target in enumerate(target_polynomial_degrees):
                logger.debug(f"      Processing target interval {k} (degree {N_k_target})...")

                # Get target nodes for this interval type (cached via Radau cache)
                try:
                    if is_state_trajectory:
                        target_basis = compute_radau_collocation_components(N_k_target)
                        target_local_nodes = target_basis.state_approximation_nodes
                        num_target_nodes = N_k_target + 1
                    else:
                        target_basis = compute_radau_collocation_components(N_k_target)
                        target_local_nodes = target_basis.collocation_nodes
                        num_target_nodes = N_k_target
                except Exception as e:
                    raise InterpolationError(
                        f"Failed to compute target basis for interval {k} with degree {N_k_target}: {e}",
                        "Target basis computation error",
                    ) from e

                # Get pooled buffer instead of allocating new memory
                target_traj_k = buffer_pool.get_buffer((num_variables, num_target_nodes))

                # Global interval boundaries for target interval k
                target_tau_start = target_mesh_points[k]
                target_tau_end = target_mesh_points[k + 1]

                # Vectorized evaluation where possible
                global_tau_points = np.array(
                    [
                        map_local_interval_tau_to_global_normalized_tau(
                            local_tau, target_tau_start, target_tau_end
                        )
                        for local_tau in target_local_nodes
                    ],
                    dtype=np.float64,
                )

                # Process all nodes for this interval
                for j, (_, global_tau) in enumerate(
                    zip(target_local_nodes, global_tau_points, strict=False)
                ):
                    # Find which previous interval contains this global tau
                    prev_interval_idx = _find_containing_interval(global_tau, prev_mesh_points)

                    if prev_interval_idx is None:
                        # Point is outside previous mesh - use boundary values
                        if global_tau < prev_mesh_points[0]:
                            prev_interval_idx = 0
                            prev_local_tau = -1.0
                        elif global_tau > prev_mesh_points[-1]:
                            prev_interval_idx = len(prev_interpolants) - 1
                            prev_local_tau = 1.0
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

                    # Evaluate the interpolant at this local tau
                    try:
                        interpolated_values = prev_interpolants[prev_interval_idx](prev_local_tau)
                        if interpolated_values.ndim > 1:
                            interpolated_values = interpolated_values.flatten()

                        # Critical: Validate interpolated values
                        if np.any(np.isnan(interpolated_values)) or np.any(
                            np.isinf(interpolated_values)
                        ):
                            raise DataIntegrityError(
                                f"Invalid interpolated values at tau={prev_local_tau}: contains NaN or Inf",
                                "Numerical corruption in interpolation result",
                            )

                        # Store the interpolated values
                        if len(interpolated_values) == num_variables:
                            target_traj_k[:, j] = interpolated_values
                        elif num_variables == 0:
                            # Handle empty variable case
                            pass
                        else:
                            raise InterpolationError(
                                f"Dimension mismatch: interpolated {len(interpolated_values)} values, expected {num_variables}",
                                "Interpolation output dimension error",
                            )

                    except Exception as e:
                        if isinstance(e, InterpolationError | DataIntegrityError):
                            raise  # Re-raise TrajectoLab-specific errors
                        raise InterpolationError(
                            f"Failed to evaluate interpolant: {e}", "Interpolant evaluation error"
                        ) from e

                # Copy result from pooled buffer to permanent storage
                result_array = np.array(target_traj_k, dtype=np.float64, copy=True)

                # Final validation of result
                if np.any(np.isnan(result_array)) or np.any(np.isinf(result_array)):
                    raise DataIntegrityError(
                        f"Final interpolated trajectory for interval {k} contains NaN or Inf",
                        "Corruption in final interpolation result",
                    )

                target_trajectories.append(result_array)
                logger.debug(
                    f"        Interval {k}: Created trajectory of shape {result_array.shape}"
                )

        logger.info(f"    ✓ Successfully interpolated {trajectory_type} trajectories (OPTIMIZED)")
        return cast(list[FloatArray], target_trajectories)

    except Exception as e:
        if isinstance(e, InterpolationError | DataIntegrityError):
            raise  # Re-raise TrajectoLab-specific errors
        raise InterpolationError(
            f"Failed to interpolate {trajectory_type} trajectories: {e}",
            "Critical interpolation failure",
        ) from e


def _find_containing_interval(global_tau: float, mesh_points: FloatArray) -> int | None:
    """Find which mesh interval contains the given global tau value."""
    tolerance = 1e-10

    if global_tau < mesh_points[0] - tolerance:
        return None
    if global_tau > mesh_points[-1] + tolerance:
        return None

    # Use binary search for large meshes
    if len(mesh_points) > 10:
        idx = int(np.searchsorted(mesh_points, global_tau)) - 1
        if 0 <= idx < len(mesh_points) - 1:
            return idx

    # Fallback: linear search for small meshes
    for k in range(len(mesh_points) - 1):
        if mesh_points[k] - tolerance <= global_tau <= mesh_points[k + 1] + tolerance:
            return k

    return None


def propagate_solution_to_new_mesh(
    prev_solution: OptimalControlSolution,
    problem: ProblemProtocol,
    target_polynomial_degrees: list[int],
    target_mesh_points: FloatArray,
) -> InitialGuess:
    """
    OPTIMIZED solution propagation using unified storage and memory pooling - ENHANCED WITH FAIL-FAST.
    """
    logger.info("  Starting OPTIMIZED aggressive interpolation-based propagation...")

    # Guard clause: Validate previous solution
    if not prev_solution.success:
        raise InterpolationError(
            "Cannot propagate from unsuccessful previous solution",
            "Invalid source solution for propagation",
        )

    # Guard clause: Get previous mesh information and validate it exists
    prev_states = prev_solution.solved_state_trajectories_per_interval
    prev_controls = prev_solution.solved_control_trajectories_per_interval
    prev_mesh = prev_solution.global_mesh_nodes_at_solve_time
    prev_degrees = prev_solution.num_collocation_nodes_list_at_solve_time

    if prev_states is None:
        raise InterpolationError(
            "Previous solution missing state trajectory data for interpolation",
            "Missing source state data",
        )
    if prev_controls is None:
        raise InterpolationError(
            "Previous solution missing control trajectory data for interpolation",
            "Missing source control data",
        )
    if prev_mesh is None:
        raise InterpolationError(
            "Previous solution missing mesh information for interpolation",
            "Missing source mesh data",
        )
    if prev_degrees is None:
        raise InterpolationError(
            "Previous solution missing polynomial degree information for interpolation",
            "Missing source degree data",
        )

    # Guard clause: Validate target mesh configuration
    if len(target_polynomial_degrees) != len(target_mesh_points) - 1:
        raise ConfigurationError(
            f"Target polynomial degrees count ({len(target_polynomial_degrees)}) doesn't match target mesh intervals ({len(target_mesh_points) - 1})",
            "Target mesh configuration error",
        )

    try:
        # ALWAYS propagate time variables and integrals (mesh-independent)
        t0_guess = prev_solution.initial_time_variable
        tf_guess = prev_solution.terminal_time_variable
        integrals_guess = prev_solution.integrals

        # Critical: Validate time variables
        if t0_guess is None or tf_guess is None:
            raise InterpolationError(
                "Previous solution missing time variables", "Invalid time data for propagation"
            )

        if np.isnan(t0_guess) or np.isinf(t0_guess) or np.isnan(tf_guess) or np.isinf(tf_guess):
            raise DataIntegrityError(
                f"Invalid time variables: t0={t0_guess}, tf={tf_guess}",
                "Numerical corruption in time data",
            )

        logger.info(f"    Propagated time variables: t0={t0_guess}, tf={tf_guess}")
        if integrals_guess is not None:
            if isinstance(integrals_guess, int | float):
                logger.info(f"    Propagated integral: {integrals_guess}")
            else:
                logger.info(f"    Propagated integrals: {len(integrals_guess)} values")

        # Get problem dimensions using unified storage
        num_states, num_controls = problem.get_variable_counts()

        logger.info(f"    Problem dimensions: {num_states} states, {num_controls} controls")
        logger.info(
            f"    Mesh transition: {len(prev_degrees)} → {len(target_polynomial_degrees)} intervals"
        )

        # Use memory-pooled interpolation for massive performance gains
        states_guess = _interpolate_trajectory_to_new_mesh_optimized(
            prev_trajectory_per_interval=prev_states,
            prev_mesh_points=prev_mesh,
            prev_polynomial_degrees=prev_degrees,
            target_mesh_points=target_mesh_points,
            target_polynomial_degrees=target_polynomial_degrees,
            num_variables=num_states,
            is_state_trajectory=True,
        )

        controls_guess = _interpolate_trajectory_to_new_mesh_optimized(
            prev_trajectory_per_interval=prev_controls,
            prev_mesh_points=prev_mesh,
            prev_polynomial_degrees=prev_degrees,
            target_mesh_points=target_mesh_points,
            target_polynomial_degrees=target_polynomial_degrees,
            num_variables=num_controls,
            is_state_trajectory=False,
        )

        # Create initial guess object
        initial_guess = InitialGuess(
            initial_time_variable=t0_guess,
            terminal_time_variable=tf_guess,
            states=states_guess,
            controls=controls_guess,
            integrals=integrals_guess,
        )

        # Validate the interpolated initial guess using existing validation infrastructure
        logger.info("  Validating interpolated initial guess...")
        validate_initial_guess_structure(
            initial_guess=initial_guess,
            num_states=num_states,
            num_controls=num_controls,
            num_integrals=problem._num_integrals,
            polynomial_degrees=target_polynomial_degrees,
        )
        logger.info("  ✓ All interpolated initial guess validations passed")

        logger.info(
            "  ✓ Completed OPTIMIZED aggressive interpolation-based propagation successfully"
        )
        return initial_guess

    except Exception as e:
        if isinstance(e, InterpolationError | DataIntegrityError | ConfigurationError):
            raise  # Re-raise TrajectoLab-specific errors
        raise InterpolationError(
            f"OPTIMIZED aggressive interpolation propagation failed: {e}",
            "Critical propagation failure",
        ) from e
