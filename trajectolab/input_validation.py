"""
Input validation utilities for the direct solver.
"""

import logging
from collections.abc import Sequence

import casadi as ca
import numpy as np

from .tl_types import (
    CasadiMatrix,
    CasadiMX,
    CasadiOpti,
    FloatArray,
    InitialGuess,
    InitialGuessIntegrals,
)
from .utils.constants import MESH_TOLERANCE, MINIMUM_TIME_INTERVAL, ZERO_TOLERANCE


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s  - %(message)s",
    handlers=[
        logging.StreamHandler(),  # This outputs to console
    ],
)


def validate_dynamics_output(
    output: list[CasadiMX] | CasadiMatrix | Sequence[CasadiMX], num_states: int
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


def validate_integral_values(
    integrals: InitialGuessIntegrals,
    num_integrals: int,
) -> None:
    """
    Validate integral values without setting them in CasADi.
    Generic version that can be used by both direct solver and adaptive algorithm.

    Args:
        integrals: Integral values to validate
        num_integrals: Expected number of integrals

    Raises:
        ValueError: If validation fails
    """
    if integrals is None:
        return

    if num_integrals == 1:
        if not isinstance(integrals, int | float):
            raise ValueError(
                f"For single integral, guess must be scalar (int or float), "
                f"got {type(integrals)} with value {integrals}"
            )
        if np.isnan(float(integrals)) or np.isinf(float(integrals)):
            raise ValueError(f"Integral has invalid value: {integrals}")

    elif num_integrals > 1:
        if isinstance(integrals, int | float):
            raise ValueError(
                f"For {num_integrals} integrals, guess must be array-like, got scalar {integrals}"
            )

        if isinstance(integrals, list):
            integrals_array = np.array(integrals, dtype=np.float64)
        else:
            integrals_array = np.array(integrals, dtype=np.float64)

        if integrals_array.size != num_integrals:
            raise ValueError(
                f"Integral guess must have exactly {num_integrals} elements, got {integrals_array.size}"
            )
        if np.any(np.isnan(integrals_array)) or np.any(np.isinf(integrals_array)):
            raise ValueError("Integrals contain invalid values (NaN or inf)")


def validate_time_values(
    initial_time: float | None,
    terminal_time: float | None,
) -> None:
    """
    Validate actual time values (not bounds).

    Args:
        initial_time: Initial time value
        terminal_time: Terminal time value

    Raises:
        ValueError: If validation fails
    """
    if initial_time is not None:
        if not isinstance(initial_time, int | float):
            raise ValueError(f"Initial time must be scalar, got {type(initial_time)}")
        if np.isnan(initial_time) or np.isinf(initial_time):
            raise ValueError(f"Initial time has invalid value: {initial_time}")

    if terminal_time is not None:
        if not isinstance(terminal_time, int | float):
            raise ValueError(f"Terminal time must be scalar, got {type(terminal_time)}")
        if np.isnan(terminal_time) or np.isinf(terminal_time):
            raise ValueError(f"Terminal time has invalid value: {terminal_time}")

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
    """
    Validate trajectory arrays against expected shapes.

    Args:
        trajectories: List of trajectory arrays
        expected_shapes: List of expected (num_vars, num_nodes) shapes
        trajectory_type: "state" or "control" for error messages

    Raises:
        ValueError: If validation fails
    """
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

        # Check for invalid values
        if np.any(np.isnan(traj)):
            raise ValueError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} contains NaN values"
            )

        if np.any(np.isinf(traj)):
            raise ValueError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} contains infinite values"
            )

        # Check data type
        if traj.dtype != np.float64:
            raise ValueError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} has dtype {traj.dtype}, "
                f"expected float64"
            )


def validate_initial_guess_structure(
    initial_guess: InitialGuess,
    num_states: int,
    num_controls: int,
    num_integrals: int,
    polynomial_degrees: list[int],
) -> None:
    """
    Validate the structure of an InitialGuess object.
    Generic version that can be used by both direct solver and adaptive algorithm.

    Args:
        initial_guess: Initial guess to validate
        num_states: Expected number of states
        num_controls: Expected number of controls
        num_integrals: Expected number of integrals
        polynomial_degrees: Polynomial degrees per interval

    Raises:
        ValueError: If validation fails
    """
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


def validate_and_set_integral_guess(
    opti: CasadiOpti,
    integral_vars: CasadiMX,
    guess: InitialGuessIntegrals,
    num_integrals: int,
) -> None:
    """
    Validate and set initial guess for integrals with strict dimension checking.
    Now uses the generic validation function.

    Args:
        opti: CasADi optimization object
        integral_vars: CasADi integral variables
        guess: Initial guess for integrals (should not be None here)
        num_integrals: Expected number of integrals

    Raises:
        ValueError: If guess dimensions don't match requirements exactly
    """
    # First validate using the generic function
    validate_integral_values(guess, num_integrals)

    # Then set the values in CasADi
    if guess is None:
        return

    if num_integrals == 1:
        # After validation, guess should be a scalar (int or float) for single integrals
        if isinstance(guess, int | float):
            opti.set_initial(integral_vars, float(guess))
        else:
            # This should not happen after validation, but handle gracefully
            raise ValueError(
                f"Expected scalar for single integral after validation, got {type(guess)}"
            )
    elif num_integrals > 1:
        if isinstance(guess, list):
            guess_array = np.array(guess, dtype=np.float64)
        else:
            guess_array = np.array(guess, dtype=np.float64)
        opti.set_initial(integral_vars, guess_array.flatten())


def validate_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    num_mesh_intervals: int,
) -> None:
    """
    Comprehensive mesh configuration validation.

    Validates:
    - Polynomial degrees count matches intervals
    - Mesh points count is intervals + 1
    - Mesh points are strictly increasing
    - Mesh starts at -1.0 and ends at 1.0
    - Minimum spacing between points
    - All polynomial degrees are positive integers

    Args:
        polynomial_degrees: Polynomial degrees per interval
        mesh_points: Normalized mesh points in [-1, 1]
        num_mesh_intervals: Expected number of intervals

    Raises:
        ValueError: If any validation check fails
    """
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


def validate_mesh_for_adaptive_algorithm(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    min_degree: int,
    max_degree: int,
) -> None:
    """
    Validate mesh configuration for adaptive algorithms with degree bounds.

    Extends basic mesh validation with polynomial degree bounds checking.

    Args:
        polynomial_degrees: Polynomial degrees per interval
        mesh_points: Normalized mesh points in [-1, 1]
        min_degree: Minimum allowed polynomial degree
        max_degree: Maximum allowed polynomial degree

    Raises:
        ValueError: If mesh configuration is invalid or degrees outside bounds
    """
    # First do basic mesh validation
    validate_mesh_configuration(polynomial_degrees, mesh_points, len(polynomial_degrees))

    # Additional degree bounds validation for adaptive algorithm
    for i, degree in enumerate(polynomial_degrees):
        if degree < min_degree:
            raise ValueError(
                f"Polynomial degree {degree} for interval {i} is below minimum {min_degree}"
            )
        if degree > max_degree:
            raise ValueError(
                f"Polynomial degree {degree} for interval {i} is above maximum {max_degree}"
            )


def validate_time_bounds(
    t0_bounds: tuple[float, float],
    tf_bounds: tuple[float, float],
) -> None:
    """
    Validate time bound constraints.

    Handles both fixed and free time cases appropriately.

    Args:
        t0_bounds: (min_t0, max_t0) bounds for initial time
        tf_bounds: (min_tf, max_tf) bounds for final time

    Raises:
        ValueError: If time bounds are invalid
    """
    # Check bound ordering
    if t0_bounds[0] > t0_bounds[1]:
        raise ValueError(f"Initial time bounds are invalid: {t0_bounds}")

    if tf_bounds[0] > tf_bounds[1]:
        raise ValueError(f"Final time bounds are invalid: {tf_bounds}")

    # For free final time problems, it's acceptable for min_tf to equal max_t0
    # The optimization constraint will ensure tf > t0 + epsilon
    # Only check that it's not impossible to have a valid time duration
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
    """
    Validate that an interval has sufficient length.

    Args:
        interval_start: Start of interval
        interval_end: End of interval
        interval_index: Index of interval for error reporting

    Raises:
        ValueError: If interval length is insufficient
    """
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
    """
    Validate problem dimension parameters.

    Args:
        num_states: Number of state variables
        num_controls: Number of control variables
        num_integrals: Number of integral variables

    Raises:
        ValueError: If dimensions are invalid
    """
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
