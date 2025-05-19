"""
Input validation utilities for the direct solver.
"""

from collections.abc import Sequence

import casadi as ca
import numpy as np

from .tl_types import CasadiMatrix, CasadiMX, CasadiOpti, FloatArray


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


def validate_and_set_integral_guess(
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
        guess: Initial guess for integrals (should not be None here)
        num_integrals: Expected number of integrals

    Raises:
        ValueError: If guess dimensions don't match requirements exactly
    """
    if guess is None:
        return

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

        guess_array = np.array(guess, dtype=np.float64)
        if guess_array.size != num_integrals:
            raise ValueError(
                f"Integral guess must have exactly {num_integrals} elements, got {guess_array.size}"
            )

        opti.set_initial(integral_vars, guess_array.flatten())


def validate_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    num_mesh_intervals: int,
) -> None:
    """
    Validate mesh configuration parameters.

    Args:
        polynomial_degrees: Polynomial degrees per interval
        mesh_points: Normalized mesh points
        num_mesh_intervals: Expected number of intervals

    Raises:
        ValueError: If mesh configuration is invalid
    """
    if len(polynomial_degrees) != num_mesh_intervals:
        raise ValueError(
            f"Number of polynomial degrees ({len(polynomial_degrees)}) must equal "
            f"number of mesh intervals ({num_mesh_intervals})"
        )

    if len(mesh_points) != num_mesh_intervals + 1:
        raise ValueError(
            f"Number of mesh points ({len(mesh_points)}) must be exactly "
            f"one more than number of intervals ({num_mesh_intervals})"
        )

    if not (
        np.all(np.diff(mesh_points) > 1e-9)
        and np.isclose(mesh_points[0], -1.0)
        and np.isclose(mesh_points[-1], 1.0)
    ):
        raise ValueError(
            "Global mesh nodes must be sorted, have positive interval lengths, "
            "start at -1.0, and end at +1.0"
        )

    for i, degree in enumerate(polynomial_degrees):
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError(
                f"Polynomial degree for interval {i} must be positive integer, got {degree}"
            )
