"""
Utility functions for trajectory optimization and numerical operations.
"""

from typing import cast, overload

import numpy as np

from trajectolab.trajectolab_types import ZERO_TOLERANCE, _ArrayLike, _ErrorTuple, _FloatArray


@overload
def linear_interpolation(t: _ArrayLike, x: _ArrayLike, t_eval: float) -> float: ...


@overload
def linear_interpolation(t: _ArrayLike, x: _ArrayLike, t_eval: _ArrayLike) -> _FloatArray: ...


def linear_interpolation(
    t: _ArrayLike, x: _ArrayLike, t_eval: _ArrayLike | float
) -> _FloatArray | float:
    """Perform linear interpolation of values.

    Args:
        t: Independent variable points
        x: Dependent variable values
        t_eval: Points at which to evaluate the interpolation

    Returns:
        Interpolated values at t_eval points (scalar or array depending on input)
    """
    # Ensure all inputs are float64 arrays for numerical stability
    t_arr = np.asarray(t, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    if isinstance(t_eval, (float, int)):
        # Use a different variable name to avoid type conflicts
        scalar_result: float = float(np.interp(float(t_eval), t_arr, x_arr))
        return scalar_result
    else:
        t_eval_arr = np.asarray(t_eval, dtype=np.float64)
        # Use a different variable name with explicit type annotation
        array_result: _FloatArray = np.interp(t_eval_arr, t_arr, x_arr)
        return array_result  # Removed redundant cast


def uniform_mesh(num_intervals: int) -> _FloatArray:
    """Create a uniform mesh from -1 to 1.

    Args:
        num_intervals: Number of intervals in the mesh

    Returns:
        Array of mesh points
    """
    return cast(_FloatArray, np.linspace(-1, 1, num_intervals + 1, dtype=np.float64))


def refine_around_point(mesh: _FloatArray, point: float, num_intervals: int = 2) -> _FloatArray:
    """Refine a mesh by adding points around a specified point.

    Args:
        mesh: The existing mesh points
        point: The point around which to refine
        num_intervals: Number of intervals to create in the refined region

    Returns:
        The refined mesh with additional points

    Raises:
        ValueError: If the point is outside the mesh range
    """
    # Find interval containing the point
    if point < mesh[0] or point > mesh[-1]:
        raise ValueError(f"Point {point} is outside the mesh range [{mesh[0]}, {mesh[-1]}]")

    # Convert to Python int to avoid NumPy integer typing issues
    interval_idx_np = np.searchsorted(mesh, point) - 1
    interval_idx = int(interval_idx_np)
    interval_idx = max(0, min(interval_idx, len(mesh) - 2))

    # Create refined mesh points
    interval_start = mesh[interval_idx]
    interval_end = mesh[interval_idx + 1]
    new_points = np.linspace(interval_start, interval_end, num_intervals + 1, dtype=np.float64)

    # Create new mesh
    refined_mesh = np.concatenate([mesh[:interval_idx], new_points, mesh[interval_idx + 2 :]])

    # Ensure the result is float64
    return cast(_FloatArray, refined_mesh)


def estimate_error(
    x_approx: _FloatArray, x_ref: _FloatArray, absolute: bool = False
) -> _ErrorTuple:
    """Calculate error metrics between approximate and reference values.

    Args:
        x_approx: Approximate values
        x_ref: Reference values
        absolute: If True, compute absolute errors; otherwise, compute relative errors

    Returns:
        Tuple of (max_error, mean_error, rms_error)

    Raises:
        ValueError: If shapes of inputs don't match
    """
    if x_approx.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: x_approx {x_approx.shape}, x_ref {x_ref.shape}")

    if absolute:
        errors = np.abs(x_approx - x_ref)
    else:
        # Relative error with protection against division by zero
        denominator = np.maximum(np.abs(x_ref), ZERO_TOLERANCE)
        errors = np.abs(x_approx - x_ref) / denominator

    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))
    rms_error = float(np.sqrt(np.mean(np.square(errors))))

    return max_error, mean_error, rms_error


@overload
def map_normalized_to_physical_time(tau: float, t0: float, tf: float) -> float: ...


@overload
def map_normalized_to_physical_time(tau: _FloatArray, t0: float, tf: float) -> _FloatArray: ...


def map_normalized_to_physical_time(
    tau: float | _FloatArray, t0: float, tf: float
) -> float | _FloatArray:
    """Map normalized time tau in [-1,1] to physical time t in [t0,tf].

    Args:
        tau: Normalized time value(s) in the range [-1, 1]
        t0: Initial physical time
        tf: Final physical time

    Returns:
        Physical time value(s) in the range [t0, tf]
    """
    if isinstance(tau, np.ndarray):
        array_result: _FloatArray = 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)
        return array_result  # Removed redundant cast
    else:
        scalar_result: float = 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)
        return float(scalar_result)


@overload
def map_physical_to_normalized_time(t: float, t0: float, tf: float) -> float: ...


@overload
def map_physical_to_normalized_time(t: _FloatArray, t0: float, tf: float) -> _FloatArray: ...


def map_physical_to_normalized_time(
    t: float | _FloatArray, t0: float, tf: float
) -> float | _FloatArray:
    """Map physical time t in [t0,tf] to normalized time tau in [-1,1].

    Args:
        t: Physical time value(s) in the range [t0, tf]
        t0: Initial physical time
        tf: Final physical time

    Returns:
        Normalized time value(s) in the range [-1, 1]
    """
    if isinstance(t, np.ndarray):
        array_result: _FloatArray = 2 * (t - t0) / (tf - t0) - 1
        return array_result  # Removed redundant cast
    else:
        scalar_result: float = 2 * (t - t0) / (tf - t0) - 1
        return float(scalar_result)
