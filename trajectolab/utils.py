from collections.abc import Sequence
from typing import TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray

# Type aliases
_ErrorTuple: TypeAlias = tuple[float, float, float]  # max_error, mean_error, rms_error
_FloatArray: TypeAlias = NDArray[np.float64]
_ArrayLike: TypeAlias = Sequence[float] | _FloatArray


@overload
def linear_interpolation(t: _ArrayLike, x: _ArrayLike, t_eval: float) -> float: ...


@overload
def linear_interpolation(t: _ArrayLike, x: _ArrayLike, t_eval: _ArrayLike) -> _FloatArray: ...


def linear_interpolation(
    t: _ArrayLike, x: _ArrayLike, t_eval: _ArrayLike | float
) -> _FloatArray | float:
    result = np.interp(t_eval, t, x)
    # Cast the result to ensure type checker knows the return type
    if isinstance(t_eval, (float, int)):
        return float(result)
    return cast(_FloatArray, result)


def uniform_mesh(num_intervals: int) -> _FloatArray:
    return np.linspace(-1, 1, num_intervals + 1, dtype=np.float64)


def refine_around_point(mesh: _FloatArray, point: float, num_intervals: int = 2) -> _FloatArray:
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
    new_points = np.linspace(interval_start, interval_end, num_intervals + 1)

    # Create new mesh
    refined_mesh = np.concatenate([mesh[:interval_idx], new_points, mesh[interval_idx + 2 :]])

    # Ensure the result is float64
    return cast(_FloatArray, refined_mesh.astype(np.float64))


def estimate_error(
    x_approx: _FloatArray, x_ref: _FloatArray, absolute: bool = False
) -> _ErrorTuple:
    if x_approx.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: x_approx {x_approx.shape}, x_ref {x_ref.shape}")

    if absolute:
        errors = np.abs(x_approx - x_ref)
    else:
        # Relative error with protection against division by zero
        denominator = np.maximum(np.abs(x_ref), 1e-12)
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
    result = 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)
    return cast(_FloatArray, result) if isinstance(tau, np.ndarray) else float(result)


@overload
def map_physical_to_normalized_time(t: float, t0: float, tf: float) -> float: ...


@overload
def map_physical_to_normalized_time(t: _FloatArray, t0: float, tf: float) -> _FloatArray: ...


def map_physical_to_normalized_time(
    t: float | _FloatArray, t0: float, tf: float
) -> float | _FloatArray:
    result = 2 * (t - t0) / (tf - t0) - 1
    return cast(_FloatArray, result) if isinstance(t, np.ndarray) else float(result)
