"""
TrajectoLab: A Python framework for optimal trajectory generation

This package provides a functional interface for solving optimal control problems
using the Radau Pseudospectral Method for direct collocation.
"""

from trajectolab.problem import Problem
from trajectolab.solution import Solution
from trajectolab.solver import solve, solve_adaptive, solve_fixed_mesh


__all__ = [
    "Problem",
    "Solution",
    "solve",
    "solve_adaptive",
    "solve_fixed_mesh",
]

__version__ = "0.1.0"
