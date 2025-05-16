from collections.abc import Sequence
from typing import Tuple, TypeAlias, Union, cast

import numpy as np
from numpy.typing import NDArray

# Type aliases
_FloatArray: TypeAlias = NDArray[np.float64]
_ArrayLike: TypeAlias = Union[Sequence[float], _FloatArray]


def linear_interpolation(
    t: _FloatArray, x: _FloatArray, t_eval: Union[float, _FloatArray]
) -> _FloatArray:
    result = np.interp(t_eval, t, x)
    return cast(_FloatArray, result)


def uniform_mesh(num_intervals: int) -> _FloatArray:
    return cast(_FloatArray, np.linspace(-1, 1, num_intervals + 1, dtype=np.float64))


def refine_around_point(mesh: _FloatArray, point: float, num_intervals: int = 2) -> _FloatArray:
    # Find interval containing the point
    if point < mesh[0] or point > mesh[-1]:
        raise ValueError(f"Point {point} is outside the mesh range [{mesh[0]}, {mesh[-1]}]")

    interval_idx = int(np.searchsorted(mesh, point) - 1)
    interval_idx = max(0, min(interval_idx, len(mesh) - 2))

    # Create refined mesh points
    interval_start = mesh[interval_idx]
    interval_end = mesh[interval_idx + 1]
    new_points = np.linspace(interval_start, interval_end, num_intervals + 1, dtype=np.float64)

    # Create new mesh
    refined_mesh = np.concatenate([mesh[:interval_idx], new_points, mesh[interval_idx + 2 :]])

    return cast(_FloatArray, refined_mesh)


def estimate_error(
    x_approx: _FloatArray, x_ref: _FloatArray, absolute: bool = False
) -> Tuple[float, float, float]:
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


def map_normalized_to_physical_time(
    tau: Union[float, _FloatArray], t0: float, tf: float
) -> Union[float, _FloatArray]:
    result = 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)
    if isinstance(tau, float):
        return float(result)
    return cast(_FloatArray, result)


def map_physical_to_normalized_time(
    t: Union[float, _FloatArray], t0: float, tf: float
) -> Union[float, _FloatArray]:
    result = 2 * (t - t0) / (tf - t0) - 1
    if isinstance(t, float):
        return float(result)
    return cast(_FloatArray, result)
