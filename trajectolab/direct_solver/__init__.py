"""
Direct solver package for optimal control problems using Radau pseudospectral method.
"""

from ..tl_types import OptimalControlSolution
from .core import solve_single_phase_radau_collocation


__all__ = [
    "OptimalControlSolution",
    "solve_single_phase_radau_collocation",
]
