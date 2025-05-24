"""
Input validation utilities for the direct solver - ENHANCED WITH FAIL-FAST.
Added targeted TrajectoLab-specific error handling for critical validation failures.
"""

import logging
from collections.abc import Sequence
from typing import TypeVar

import casadi as ca
import numpy as np

from .exceptions import ConfigurationError, DataIntegrityError
from .tl_types import CasadiMX, CasadiOpti, FloatArray, InitialGuess
from .utils.constants import MESH_TOLERANCE, MINIMUM_TIME_INTERVAL, ZERO_TOLERANCE


logger = logging.getLogger(__name__)
T = TypeVar("T")


def validate_dynamics_output(
    output: list[CasadiMX] | CasadiMX | Sequence[CasadiMX], num_states: int
) -> CasadiMX:
    """
    Validates and converts dynamics function output to the expected CasadiMX format.

    Raises DataIntegrityError for TrajectoLab-specific shape mismatches.
    """
    # Guard clause: Check for None output (TrajectoLab logic error)
    if output is None:
        raise DataIntegrityError(
            "Dynamics function returned None",
            "This indicates a TrajectoLab internal error in dynamics function construction",
        )

    if isinstance(output, list):
        result = ca.vertcat(*output) if output else ca.MX(num_states, 1)
        result = ca.MX(result) if isinstance(result, ca.DM) else result
    elif isinstance(output, ca.MX):
        result = output
        if output.shape[1] == 1:
            result = output
        elif output.shape[0] == 1 and num_states > 1:
            result = output.T
        elif num_states == 1:
            result = output
    elif isinstance(output, ca.DM):
        result = ca.MX(output)
        if result.shape[1] == 1:
            pass
        elif result.shape[0] == 1 and num_states > 1:
            result = result.T
    elif isinstance(output, Sequence):
        result = validate_dynamics_output(list(output), num_states)
    else:
        raise DataIntegrityError(
            f"Dynamics function output has unsupported type: {type(output)}",
            "TrajectoLab failed to process dynamics function return value",
        )

    # Critical shape validation - TrajectoLab's responsibility
    if result.shape[0] != num_states:
        raise DataIntegrityError(
            f"Dynamics output has {result.shape[0]} rows, expected {num_states}",
            "Shape mismatch in TrajectoLab dynamics processing",
        )

    return result


def validate_initial_guess_structure(
    initial_guess: InitialGuess,
    num_states: int,
    num_controls: int,
    num_integrals: int,
    polynomial_degrees: list[int],
) -> None:
    """
    UNIFIED initial guess validation with fail-fast for critical errors.
    """
    # Guard clause: Essential configuration must be present
    if not polynomial_degrees:
        raise ConfigurationError(
            "Cannot validate initial guess: polynomial degrees not configured",
            "Call problem.set_mesh() before validation",
        )

    num_intervals = len(polynomial_degrees)

    # Validate time values
    validate_time_values(initial_guess.initial_time_variable, initial_guess.terminal_time_variable)

    # Validate integrals
    validate_integral_values(initial_guess.integrals, num_integrals)

    # Critical state trajectory validation
    if initial_guess.states is not None:
        expected_state_shapes = [(num_states, N + 1) for N in polynomial_degrees]
        validate_trajectory_arrays(initial_guess.states, expected_state_shapes, "state")

        if len(initial_guess.states) != num_intervals:
            raise DataIntegrityError(
                f"State trajectories count ({len(initial_guess.states)}) doesn't match intervals ({num_intervals})",
                "Initial guess structure inconsistency",
            )

    # Critical control trajectory validation
    if initial_guess.controls is not None:
        expected_control_shapes = [(num_controls, N) for N in polynomial_degrees]
        validate_trajectory_arrays(initial_guess.controls, expected_control_shapes, "control")

        if len(initial_guess.controls) != num_intervals:
            raise DataIntegrityError(
                f"Control trajectories count ({len(initial_guess.controls)}) doesn't match intervals ({num_intervals})",
                "Initial guess structure inconsistency",
            )


def validate_trajectory_arrays(
    trajectories: list[FloatArray],
    expected_shapes: list[tuple[int, int]],
    trajectory_type: str,
) -> None:
    """
    Validate trajectory arrays against expected shapes with fail-fast.
    """
    if len(trajectories) != len(expected_shapes):
        raise DataIntegrityError(
            f"{trajectory_type.capitalize()} trajectory count mismatch: got {len(trajectories)}, expected {len(expected_shapes)}",
            f"TrajectoLab {trajectory_type} data preparation error",
        )

    for k, (traj, expected_shape) in enumerate(zip(trajectories, expected_shapes, strict=False)):
        # Critical shape validation
        if traj.shape != expected_shape:
            raise DataIntegrityError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} has shape {traj.shape}, expected {expected_shape}",
                "Shape mismatch in TrajectoLab trajectory data",
            )

        # Critical NaN/Inf validation - TrajectoLab's responsibility to catch this
        if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
            raise DataIntegrityError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} contains NaN or Inf values",
                "Invalid numerical data in TrajectoLab trajectory",
            )


def validate_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    num_mesh_intervals: int,
) -> None:
    """
    UNIFIED mesh configuration validation with fail-fast.
    """
    # Guard clause: Basic requirements
    if not polynomial_degrees:
        raise ConfigurationError(
            "Polynomial degrees list is empty", "Mesh configuration incomplete"
        )

    if len(polynomial_degrees) != num_mesh_intervals:
        raise ConfigurationError(
            f"Number of polynomial degrees ({len(polynomial_degrees)}) must equal intervals ({num_mesh_intervals})",
            "Mesh configuration mismatch",
        )

    if len(mesh_points) != num_mesh_intervals + 1:
        raise ConfigurationError(
            f"Number of mesh points ({len(mesh_points)}) must be exactly one more than intervals ({num_mesh_intervals})",
            "Mesh points configuration error",
        )

    # Critical boundary validation
    if not np.isclose(mesh_points[0], -1.0, atol=ZERO_TOLERANCE):
        raise ConfigurationError(
            f"First mesh point must be -1.0, got {mesh_points[0]}", "Invalid mesh boundary"
        )

    if not np.isclose(mesh_points[-1], 1.0, atol=ZERO_TOLERANCE):
        raise ConfigurationError(
            f"Last mesh point must be 1.0, got {mesh_points[-1]}", "Invalid mesh boundary"
        )

    # Critical spacing validation
    mesh_diffs = np.diff(mesh_points)
    if not np.all(mesh_diffs > MESH_TOLERANCE):
        min_diff = np.min(mesh_diffs)
        raise ConfigurationError(
            f"Mesh points must be strictly increasing with minimum spacing of {MESH_TOLERANCE}, found minimum difference of {min_diff}",
            "Invalid mesh spacing",
        )

    # Validate polynomial degrees
    for i, degree in enumerate(polynomial_degrees):
        if not isinstance(degree, int) or degree <= 0:
            raise ConfigurationError(
                f"Polynomial degree for interval {i} must be positive integer, got {degree}",
                "Invalid polynomial degree specification",
            )


def validate_time_bounds(
    t0_bounds: tuple[float, float],
    tf_bounds: tuple[float, float],
) -> None:
    """
    Validate time bound constraints with fail-fast.
    """
    # Guard clause: Check bound ordering
    if t0_bounds[0] > t0_bounds[1]:
        raise ConfigurationError(
            f"Initial time bounds are invalid: {t0_bounds}", "Lower bound cannot exceed upper bound"
        )
    if tf_bounds[0] > tf_bounds[1]:
        raise ConfigurationError(
            f"Final time bounds are invalid: {tf_bounds}", "Lower bound cannot exceed upper bound"
        )

    # Check if valid time duration is possible
    max_possible_duration = tf_bounds[1] - t0_bounds[0]
    if max_possible_duration < MINIMUM_TIME_INTERVAL:
        raise ConfigurationError(
            f"Maximum possible time duration ({max_possible_duration}) is below minimum ({MINIMUM_TIME_INTERVAL})",
            f"Time bounds: t0 ∈ {t0_bounds}, tf ∈ {tf_bounds}",
        )


# Helper validation functions (keeping existing implementation but with targeted error handling)
def _validate_numeric_value(value: float | int, name: str) -> None:
    """Validate a single numeric value."""
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be scalar, got {type(value)}")
    if np.isnan(value) or np.isinf(value):
        raise DataIntegrityError(f"{name} has invalid value: {value}", "Numerical data corruption")


def _validate_array_values(array: FloatArray, name: str) -> None:
    """Validate array for NaN and infinite values."""
    if np.any(np.isnan(array)):
        raise DataIntegrityError(f"{name} contains NaN values", "Numerical data corruption")
    if np.any(np.isinf(array)):
        raise DataIntegrityError(f"{name} contains infinite values", "Numerical data corruption")
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


def validate_interval_length(
    interval_start: float, interval_end: float, interval_index: int
) -> None:
    """Validate that an interval has sufficient length."""
    interval_length = interval_end - interval_start
    if interval_length <= MESH_TOLERANCE:
        raise ConfigurationError(
            f"Mesh interval {interval_index} has insufficient length: {interval_length}",
            f"Minimum length required: {MESH_TOLERANCE}",
        )


def validate_problem_dimensions(num_states: int, num_controls: int, num_integrals: int) -> None:
    """Validate problem dimension parameters."""
    if num_states < 0:
        raise ConfigurationError(
            f"Number of states must be non-negative, got {num_states}", "Invalid problem dimensions"
        )

    if num_controls < 0:
        raise ConfigurationError(
            f"Number of controls must be non-negative, got {num_controls}",
            "Invalid problem dimensions",
        )

    if num_integrals < 0:
        raise ConfigurationError(
            f"Number of integrals must be non-negative, got {num_integrals}",
            "Invalid problem dimensions",
        )

    # Warning for edge case
    if num_states == 0:
        import warnings

        warnings.warn(
            "Problem has no state variables. This may not be a meaningful optimal control problem.",
            stacklevel=2,
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
