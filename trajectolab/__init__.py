"""
TrajectoLab: A Python framework for optimal trajectory generation - SIMPLIFIED

This package provides a unified interface for solving optimal control problems
using the Radau Pseudospectral Method for direct collocation.
Removed ALL legacy code and redundancies while preserving identical functionality.
"""

from trajectolab.problem import Problem
from trajectolab.solver import solve_adaptive, solve_fixed_mesh


__all__ = [
    "Problem",
    "solve_adaptive",
    "solve_fixed_mesh",
]

__version__ = "0.2.0"  # Incremented for simplified version
