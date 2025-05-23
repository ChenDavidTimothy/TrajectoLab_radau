"""
Input validation utilities for the direct solver - SIMPLIFIED.
Removed redundant validation patterns, consolidated validation logic.
"""

import logging
from collections.abc import Sequence
from typing import TypeVar

import casadi as ca
import numpy as np

from .tl_types import (
    CasadiMX,
    CasadiOpti,
    FloatArray,
    InitialGuess,
)
from .utils.constants import MESH_TOLERANCE, MINIMUM_TIME_INTERVAL, ZERO_TOLERANCE


logger = logging.getLogger(__name__)

T = TypeVar("T")


def validate_dynamics_output(
    output: list[CasadiMX] | CasadiMX | Sequence[CasadiMX], num_states: int
) -> CasadiMX:
    """Validates and converts dynamics function output to the expected CasadiMX format."""
    if isinstance(output, list):
        result = ca.vertcat(*output) if output else ca.MX(num_states, 1)
        return ca.MX(result) if isinstance(result, ca.DM) else result
    elif isinstance(output, ca.MX):
        if output.shape[1] == 1:
            return output
        elif output.shape[0] == 1 and num_states > 1:
            return output.T
        elif num_states == 1:
            return output
    elif isinstance(output, ca.DM):
        result = ca.MX(output)
        if result.shape[1] == 1:
            return result
        elif result.shape[0] == 1 and num_states > 1:
            return result.T
        else:
            return result
    elif isinstance(output, Sequence):
        return validate_dynamics_output(list(output), num_states)

    raise TypeError(f"Dynamics function output type not supported: {type(output)}")


def _validate_numeric_value(value: float | int, name: str) -> None:
    """Validate a single numeric value."""
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be scalar, got {type(value)}")
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"{name} has invalid value: {value}")


def _validate_array_values(array: FloatArray, name: str) -> None:
    """Validate array for NaN and infinite values."""
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains infinite values")
    if array.dtype != np.float64:
        raise ValueError(f"{name} has dtype {array.dtype}, expected float64")


def validate_integral_values(integrals: float | FloatArray | None, num_integrals: int) -> None:
    """Validate integral values - UNIFIED validation."""
    if integrals is None:
        return

    if num_integrals == 1:
        if not isinstance(integrals, int | float):
            raise ValueError(f"For single integral, expected scalar, got {type(integrals)}")
        _validate_numeric_value(integrals, "Integral")
    elif num_integrals > 1:
        if isinstance(integrals, int | float):
            raise ValueError(
                f"For {num_integrals} integrals, guess must be array-like, got scalar {integrals}"
            )

        integrals_array = np.array(integrals, dtype=np.float64)
        if integrals_array.size != num_integrals:
            raise ValueError(
                f"Integral guess must have exactly {num_integrals} elements, got {integrals_array.size}"
            )
        _validate_array_values(integrals_array, "Integrals")


def validate_time_values(initial_time: float | None, terminal_time: float | None) -> None:
    """Validate actual time values (not bounds)."""
    if initial_time is not None:
        _validate_numeric_value(initial_time, "Initial time")

    if terminal_time is not None:
        _validate_numeric_value(terminal_time, "Terminal time")

    # Check time ordering if both are present
    if initial_time is not None and terminal_time is not None:
        if terminal_time <= initial_time:
            raise ValueError(
                f"Terminal time ({terminal_time}) must be greater than initial time ({initial_time})"
            )


def validate_trajectory_arrays(
    trajectories: list[FloatArray],
    expected_shapes: list[tuple[int, int]],
    trajectory_type: str,
) -> None:
    """Validate trajectory arrays against expected shapes."""
    if len(trajectories) != len(expected_shapes):
        raise ValueError(
            f"{trajectory_type.capitalize()} trajectory count mismatch: "
            f"got {len(trajectories)} arrays, expected {len(expected_shapes)}"
        )

    for k, (traj, expected_shape) in enumerate(zip(trajectories, expected_shapes, strict=False)):
        # Check shape
        if traj.shape != expected_shape:
            raise ValueError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} has shape {traj.shape}, "
                f"expected {expected_shape}"
            )

        # Check for invalid values and data type
        _validate_array_values(traj, f"{trajectory_type.capitalize()} trajectory for interval {k}")


def validate_initial_guess_structure(
    initial_guess: InitialGuess,
    num_states: int,
    num_controls: int,
    num_integrals: int,
    polynomial_degrees: list[int],
) -> None:
    """UNIFIED initial guess validation."""
    num_intervals = len(polynomial_degrees)

    # Validate time values
    validate_time_values(initial_guess.initial_time_variable, initial_guess.terminal_time_variable)

    # Validate integrals
    validate_integral_values(initial_guess.integrals, num_integrals)

    # Validate state trajectories
    if initial_guess.states is not None:
        expected_state_shapes = [(num_states, N + 1) for N in polynomial_degrees]
        validate_trajectory_arrays(initial_guess.states, expected_state_shapes, "state")

        if len(initial_guess.states) != num_intervals:
            raise ValueError(
                f"State trajectories count ({len(initial_guess.states)}) doesn't match "
                f"number of intervals ({num_intervals})"
            )

    # Validate control trajectories
    if initial_guess.controls is not None:
        expected_control_shapes = [(num_controls, N) for N in polynomial_degrees]
        validate_trajectory_arrays(initial_guess.controls, expected_control_shapes, "control")

        if len(initial_guess.controls) != num_intervals:
            raise ValueError(
                f"Control trajectories count ({len(initial_guess.controls)}) doesn't match "
                f"number of intervals ({num_intervals})"
            )


def set_integral_guess_values(
    opti: CasadiOpti,
    integral_vars: CasadiMX,
    guess: float | FloatArray | None,
    num_integrals: int,
) -> None:
    """Set initial guess values for integrals in CasADi optimization object."""
    if guess is None:
        return

    if num_integrals == 1:
        if isinstance(guess, int | float):
            opti.set_initial(integral_vars, float(guess))
        else:
            raise ValueError(f"Expected scalar for single integral, got {type(guess)}")
    elif num_integrals > 1:
        guess_array = np.array(guess, dtype=np.float64)
        opti.set_initial(integral_vars, guess_array.flatten())


def validate_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    num_mesh_intervals: int,
) -> None:
    """UNIFIED mesh configuration validation."""
    # Check polynomial degrees count
    if len(polynomial_degrees) != num_mesh_intervals:
        raise ValueError(
            f"Number of polynomial degrees ({len(polynomial_degrees)}) must equal "
            f"number of mesh intervals ({num_mesh_intervals})"
        )

    # Check mesh points count
    if len(mesh_points) != num_mesh_intervals + 1:
        raise ValueError(
            f"Number of mesh points ({len(mesh_points)}) must be exactly "
            f"one more than number of intervals ({num_mesh_intervals})"
        )

    # Validate mesh points boundaries
    if not np.isclose(mesh_points[0], -1.0, atol=ZERO_TOLERANCE):
        raise ValueError(f"First mesh point must be -1.0, got {mesh_points[0]}")

    if not np.isclose(mesh_points[-1], 1.0, atol=ZERO_TOLERANCE):
        raise ValueError(f"Last mesh point must be 1.0, got {mesh_points[-1]}")

    # Check mesh points are strictly increasing with minimum spacing
    mesh_diffs = np.diff(mesh_points)
    if not np.all(mesh_diffs > MESH_TOLERANCE):
        min_diff = np.min(mesh_diffs)
        raise ValueError(
            f"Mesh points must be strictly increasing with minimum spacing of {MESH_TOLERANCE}, "
            f"but found minimum difference of {min_diff}"
        )

    # Validate polynomial degrees
    for i, degree in enumerate(polynomial_degrees):
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError(
                f"Polynomial degree for interval {i} must be positive integer, got {degree}"
            )


def validate_time_bounds(
    t0_bounds: tuple[float, float],
    tf_bounds: tuple[float, float],
) -> None:
    """Validate time bound constraints."""
    # Check bound ordering
    if t0_bounds[0] > t0_bounds[1]:
        raise ValueError(f"Initial time bounds are invalid: {t0_bounds}")
    if tf_bounds[0] > tf_bounds[1]:
        raise ValueError(f"Final time bounds are invalid: {tf_bounds}")

    # Check if valid time duration is possible
    max_possible_duration = tf_bounds[1] - t0_bounds[0]
    if max_possible_duration < MINIMUM_TIME_INTERVAL:
        raise ValueError(
            f"Maximum possible time duration ({max_possible_duration}) is below "
            f"threshold ({MINIMUM_TIME_INTERVAL}). "
            f"Time bounds: t0 ∈ {t0_bounds}, tf ∈ {tf_bounds}"
        )

    # If both times are fixed (bounds are equal), check the fixed duration
    t0_is_fixed = abs(t0_bounds[1] - t0_bounds[0]) < ZERO_TOLERANCE
    tf_is_fixed = abs(tf_bounds[1] - tf_bounds[0]) < ZERO_TOLERANCE

    if t0_is_fixed and tf_is_fixed:
        fixed_duration = tf_bounds[0] - t0_bounds[0]
        if fixed_duration < MINIMUM_TIME_INTERVAL:
            raise ValueError(
                f"Fixed time duration ({fixed_duration}) is below "
                f"threshold ({MINIMUM_TIME_INTERVAL}). "
                f"Fixed t0 = {t0_bounds[0]}, fixed tf = {tf_bounds[0]}"
            )


def validate_interval_length(
    interval_start: float,
    interval_end: float,
    interval_index: int,
) -> None:
    """Validate that an interval has sufficient length."""
    interval_length = interval_end - interval_start
    if interval_length <= MESH_TOLERANCE:
        raise ValueError(
            f"Mesh interval {interval_index} has insufficient length: {interval_length}. "
            f"Minimum length required: {MESH_TOLERANCE}"
        )


def validate_problem_dimensions(
    num_states: int,
    num_controls: int,
    num_integrals: int,
) -> None:
    """Validate problem dimension parameters."""
    if num_states < 0:
        raise ValueError(f"Number of states must be non-negative, got {num_states}")

    if num_controls < 0:
        raise ValueError(f"Number of controls must be non-negative, got {num_controls}")

    if num_integrals < 0:
        raise ValueError(f"Number of integrals must be non-negative, got {num_integrals}")

    # At least one state is typically required for meaningful optimal control
    if num_states == 0:
        import warnings

        warnings.warn(
            "Problem has no state variables. This may not be a meaningful optimal control problem.",
            stacklevel=2,
        )
