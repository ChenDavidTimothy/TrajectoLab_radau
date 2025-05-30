# trajectolab/direct_solver/__init__.py
"""
Direct solver package for multiphase optimal control problems using Radau pseudospectral method.
"""

from ..tl_types import OptimalControlSolution
from .core_solver import solve_multiphase_radau_collocation


__all__ = [
    "OptimalControlSolution",
    "solve_multiphase_radau_collocation",
]
