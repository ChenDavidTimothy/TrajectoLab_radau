import numpy as np
from typing import List, Tuple, Optional, Union

def linear_interpolation(t: np.ndarray, 
                         x: np.ndarray, 
                         t_eval: Union[float, np.ndarray]) -> np.ndarray:
    """
    Linear interpolation of 1D data.
    
    Args:
        t: Time/independent variable points
        x: Function values at t
        t_eval: Point(s) at which to evaluate the interpolant
        
    Returns:
        Interpolated values at t_eval
    """
    return np.interp(t_eval, t, x)

def uniform_mesh(num_intervals: int) -> np.ndarray:
    """
    Create a uniform mesh in normalized time domain [-1, 1].
    
    Args:
        num_intervals: Number of mesh intervals
        
    Returns:
        Array of mesh points
    """
    return np.linspace(-1, 1, num_intervals + 1)

def refine_around_point(mesh: np.ndarray, 
                         point: float, 
                         num_intervals: int = 2) -> np.ndarray:
    """
    Refine a mesh around a specific point.
    
    Args:
        mesh: The original mesh
        point: The point around which to refine
        num_intervals: Number of intervals to replace the original interval with
        
    Returns:
        The refined mesh
    """
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
    refined_mesh = np.concatenate([
        mesh[:interval_idx],
        new_points,
        mesh[interval_idx + 2:]
    ])
    
    return refined_mesh

def estimate_error(x_approx: np.ndarray, 
                   x_ref: np.ndarray, 
                   absolute: bool = False) -> Tuple[float, float, float]:
    """
    Estimate error between approximate and reference solutions.
    
    Args:
        x_approx: Approximate solution
        x_ref: Reference solution
        absolute: Whether to compute absolute error
        
    Returns:
        Tuple of (max_error, mean_error, rms_error)
    """
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

def map_normalized_to_physical_time(tau: Union[float, np.ndarray], 
                                    t0: float, 
                                    tf: float) -> Union[float, np.ndarray]:
    """
    Map from normalized time domain [-1, 1] to physical time domain [t0, tf].
    
    Args:
        tau: Normalized time point(s)
        t0: Initial physical time
        tf: Final physical time
        
    Returns:
        Physical time point(s)
    """
    return 0.5 * (tf - t0) * tau + 0.5 * (tf + t0)

def map_physical_to_normalized_time(t: Union[float, np.ndarray], 
                                    t0: float, 
                                    tf: float) -> Union[float, np.ndarray]:
    """
    Map from physical time domain [t0, tf] to normalized time domain [-1, 1].
    
    Args:
        t: Physical time point(s)
        t0: Initial physical time
        tf: Final physical time
        
    Returns:
        Normalized time point(s)
    """
    return 2 * (t - t0) / (tf - t0) - 1 