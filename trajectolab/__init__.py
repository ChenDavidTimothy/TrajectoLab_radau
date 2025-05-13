"""
TrajectoLab: A Python library for trajectory optimization using the Radau Pseudospectral Method.

This library provides tools for solving optimal control problems using 
direct transcription with the Radau Pseudospectral Method and adaptive mesh refinement.
"""

__version__ = '0.1.0'

# Import and expose main components for easy access
from .core import solve_single_phase_radau_collocation, compute_radau_collocation_components
from .core import ProblemDefinition, Solution
from .adaptive import AdaptiveMethod, AdaptiveParams, PHSMethod, create_method
from .adaptive import run_phs_adaptive_mesh_refinement  # For backward compatibility
from .utils import SolutionProcessor, plot_solution_with_processor

__all__ = [
    # Core components
    'solve_single_phase_radau_collocation',
    'compute_radau_collocation_components',
    'ProblemDefinition',
    'Solution',
    
    # Adaptive components
    'AdaptiveMethod',
    'AdaptiveParams',
    'PHSMethod',
    'create_method',
    'run_phs_adaptive_mesh_refinement',
    
    # Utilities
    'SolutionProcessor',
    'plot_solution_with_processor',
]