"""
Core module for TrajectoLab - Radau Pseudospectral Method implementation.

This module provides the fundamental solving capabilities for optimal control
problems using the Radau Pseudospectral Method.
"""

from .solver import solve_single_phase_radau_collocation
from .basis import compute_radau_collocation_components
from .problem import ProblemDefinition, Solution

__all__ = [
    'solve_single_phase_radau_collocation',
    'compute_radau_collocation_components',
    'ProblemDefinition',
    'Solution'
]