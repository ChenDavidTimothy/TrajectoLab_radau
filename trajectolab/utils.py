from typing import cast  # Retain cast for type checker hints

import numpy as np

# Import the centralized FloatArray type definition from tl_types.py
from .tl_types import FloatArray  # Assuming tl_types.py is in the same directory/package


def linear_interpolation(t: FloatArray, x: FloatArray, t_eval: float | FloatArray) -> FloatArray:
    """
    Performs 1-D linear interpolation.

    Args:
        t: Array of x-coordinates of the data points (e.g., time). Must be sorted.
        x: Array of y-coordinates of the data points (e.g., function values).
           Must be the same size as t.
        t_eval: The x-coordinate(s) at which to evaluate the interpolated values.
                Can be a single float or a NumPy array of floats.

    Returns:
        A NumPy array containing the interpolated values corresponding to t_eval.

    Raises:
        ValueError: If t and x have different shapes or lengths.
    """
    if t.shape != x.shape:
        raise ValueError(
            f"Input arrays t and x must have the same shape. "
            f"Got t.shape={t.shape}, x.shape={x.shape}"
        )
    # np.interp is efficient and handles various cases, including t_eval outside t's range.
    result = np.interp(t_eval, t, x)
    return cast(FloatArray, result)


def uniform_mesh(num_intervals: int) -> FloatArray:
    """
    Creates a 1-D array (mesh) of evenly spaced points in the interval [-1, 1].

    Args:
        num_intervals: The number of intervals the mesh should have.
                       The number of points generated will be num_intervals + 1.

    Returns:
        A NumPy array of (num_intervals + 1) equally spaced float64 points
        between -1 and 1, inclusive.

    Raises:
        ValueError: If num_intervals is not a positive integer.
    """
    if not isinstance(num_intervals, int) or num_intervals <= 0:
        raise ValueError("num_intervals must be a positive integer.")

    mesh_points = np.linspace(-1.0, 1.0, num_intervals + 1, dtype=np.float64)
    return cast(FloatArray, mesh_points)


def refine_around_point(mesh: FloatArray, point: float, num_new_intervals: int = 2) -> FloatArray:
    """
    Refines a mesh by adding more points within the interval that contains a specific point.

    The original interval containing the point is replaced by `num_new_intervals`
    new sub-intervals (i.e., `num_new_intervals + 1` points).

    Args:
        mesh: The initial sorted 1-D NumPy array of mesh points. Must have at least 2 points.
        point: The point around which the mesh should be refined.
        num_new_intervals: The number of new sub-intervals to create within the
                           interval containing the point. Defaults to 2.

    Returns:
        A new, refined mesh array (NumPy array of float64).

    Raises:
        ValueError: If the point is outside the mesh range, if the mesh is invalid,
                    or if num_new_intervals is not positive.
    """
    if not (isinstance(mesh, np.ndarray) and mesh.ndim == 1 and mesh.size >= 2):
        raise ValueError("Mesh must be a 1D NumPy array with at least two points.")
    if point < mesh[0] or point > mesh[-1]:
        raise ValueError(f"Point {point} is outside the mesh range [{mesh[0]}, {mesh[-1]}].")
    if not isinstance(num_new_intervals, int) or num_new_intervals <= 0:
        raise ValueError("num_new_intervals must be a positive integer.")

    # Find the index of the interval containing the point.
    # np.searchsorted returns the insertion point. Subtracting 1 gives the left boundary.
    # Explicitly cast to int for use with Python's min/max and array indexing.
    interval_idx = int(np.searchsorted(mesh, point, side="right") - 1)

    # Handle edge case where point might be exactly mesh[0].
    # searchsorted with side='right' on mesh[0] would give 1, so idx becomes 0.
    # Clamp interval_idx to ensure it's a valid starting index for an interval.
    interval_idx = max(0, min(interval_idx, len(mesh) - 2))

    interval_start = mesh[interval_idx]
    interval_end = mesh[interval_idx + 1]

    # Create refined points for the identified interval.
    new_points = np.linspace(interval_start, interval_end, num_new_intervals + 1, dtype=np.float64)

    # Construct the new mesh by concatenating parts of the old mesh with new_points.
    refined_mesh = np.concatenate([mesh[:interval_idx], new_points, mesh[interval_idx + 2 :]])

    return cast(FloatArray, refined_mesh)


def estimate_error(
    x_approx: FloatArray, x_ref: FloatArray, absolute: bool = False
) -> tuple[float, float, float]:
    """
    Estimates error metrics (max, mean, RMS) between an approximate and a reference solution.

    Args:
        x_approx: The NumPy array of approximate values.
        x_ref: The NumPy array of reference (true or more accurate) values.
               Must have the same shape as x_approx.
        absolute: If True, calculates absolute error: |approx - ref|.
                  If False (default), calculates relative error:
                  |approx - ref| / max(|ref|, epsilon), where epsilon is machine epsilon
                  for float64 to prevent division by zero and stabilize small denominators.

    Returns:
        A tuple containing three float values: (max_error, mean_error, rms_error).

    Raises:
        ValueError: If x_approx and x_ref have different shapes.
    """
    if x_approx.shape != x_ref.shape:
        raise ValueError(
            f"Shape mismatch: x_approx {x_approx.shape}, x_ref {x_ref.shape}. "
            "Arrays must have the same dimensions for error estimation."
        )

    if absolute:
        errors = np.abs(x_approx - x_ref)
    else:
        # Relative error with protection against division by zero or very small numbers.
        epsilon = np.finfo(np.float64).eps
        denominator = np.maximum(np.abs(x_ref), epsilon)
        errors = np.abs(x_approx - x_ref) / denominator

    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))
    rms_error = float(np.sqrt(np.mean(np.square(errors))))

    return max_error, mean_error, rms_error


def map_normalized_to_physical_time(
    tau: float | FloatArray, t0: float, tf: float
) -> float | FloatArray:
    """
    Maps time `tau` from a normalized domain (typically [-1, 1]) to a physical time [t0, tf].

    Args:
        tau: The normalized time value(s) (float or NumPy array).
        t0: The start of the physical time interval.
        tf: The end of the physical time interval.

    Returns:
        The corresponding physical time value(s). Returns a Python float if `tau`
        is a float, otherwise returns a NumPy array (FloatArray).

    Raises:
        ValueError: If t0 and tf are the same, as this defines a zero-length interval.
    """
    if t0 == tf:
        raise ValueError(
            "Start time t0 and end time tf must be different for a valid mapping interval."
        )

    physical_time = 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)

    if isinstance(tau, float):
        return float(physical_time)
    return cast(FloatArray, physical_time)


def map_physical_to_normalized_time(
    t: float | FloatArray, t0: float, tf: float
) -> float | FloatArray:
    """
    Maps physical time `t` from the interval [t0, tf] to a normalized time `tau`
    (typically in the interval [-1, 1]).

    Args:
        t: The physical time value(s) (float or NumPy array).
        t0: The start of the physical time interval.
        tf: The end of the physical time interval.

    Returns:
        The corresponding normalized time value(s). Returns a Python float if `t`
        is a float, otherwise returns a NumPy array (FloatArray).

    Raises:
        ValueError: If t0 and tf are the same, leading to division by zero.
    """
    if tf - t0 == 0:
        raise ValueError(
            "Physical time interval [t0, tf] must have non-zero length (tf - t0 != 0) "
            "for normalization."
        )

    normalized_time = (2.0 * (t - t0) / (tf - t0)) - 1.0

    if isinstance(t, float):
        return float(normalized_time)
    return cast(FloatArray, normalized_time)
