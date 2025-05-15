import numpy as np


def linear_interpolation(t, x, t_eval):
    return np.interp(t_eval, t, x)


def uniform_mesh(num_intervals):
    return np.linspace(-1, 1, num_intervals + 1)


def refine_around_point(mesh, point, num_intervals=2):
    # Find interval containing the point
    if point < mesh[0] or point > mesh[-1]:
        raise ValueError(f"Point {point} is outside the mesh range [{mesh[0]}, {mesh[-1]}]")

    interval_idx = np.searchsorted(mesh, point) - 1
    interval_idx = max(0, min(interval_idx, len(mesh) - 2))

    # Create refined mesh points
    interval_start = mesh[interval_idx]
    interval_end = mesh[interval_idx + 1]
    new_points = np.linspace(interval_start, interval_end, num_intervals + 1)

    # Create new mesh
    refined_mesh = np.concatenate([mesh[:interval_idx], new_points, mesh[interval_idx + 2 :]])

    return refined_mesh


def estimate_error(x_approx, x_ref, absolute=False):
    if x_approx.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: x_approx {x_approx.shape}, x_ref {x_ref.shape}")

    if absolute:
        errors = np.abs(x_approx - x_ref)
    else:
        # Relative error with protection against division by zero
        denominator = np.maximum(np.abs(x_ref), 1e-12)
        errors = np.abs(x_approx - x_ref) / denominator

    max_error = np.max(errors)
    mean_error = np.mean(errors)
    rms_error = np.sqrt(np.mean(np.square(errors)))

    return max_error, mean_error, rms_error


def map_normalized_to_physical_time(tau, t0, tf):
    return 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)


def map_physical_to_normalized_time(t, t0, tf):
    return 2 * (t - t0) / (tf - t0) - 1
