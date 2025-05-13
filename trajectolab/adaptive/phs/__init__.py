"""
PHS-Adaptive mesh refinement method.

This module provides the PHS (p-refinement, h-refinement, and h/p-reduction)
adaptive mesh refinement method for the Radau Pseudospectral Method.
"""

from .method import PHSMethod, run_phs_adaptive_mesh_refinement
from .error import (IntervalSimulationBundle, calculate_relative_error_estimate)
from .refinement import (p_refine_interval, h_refine_params, 
                        h_reduce_intervals, p_reduce_interval)

__all__ = [
    'PHSMethod',
    'run_phs_adaptive_mesh_refinement',
    'IntervalSimulationBundle',
    'calculate_relative_error_estimate',
    'p_refine_interval',
    'h_refine_params', 
    'h_reduce_intervals',
    'p_reduce_interval'
]