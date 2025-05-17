"""
TrajectoLab: A Python framework for optimal trajectory generation

This package provides a flexible and extensible framework for solving
optimal control problems, with a focus on trajectory optimization.
It implements the Radau Pseudospectral Method for direct collocation.
"""

from trajectolab.adaptive import FixedMesh, PHSAdaptive
from trajectolab.direct_solver import InitialGuess
from trajectolab.problem import Constraint, Problem
from trajectolab.solution import Solution
from trajectolab.solver import RadauDirectSolver, solve

__all__ = [
    "Problem",
    "Solution",
    "solve",
    "RadauDirectSolver",
    "PHSAdaptive",
    "FixedMesh",
    "Constraint",
    "InitialGuess",
]

__version__ = "0.1.0"
