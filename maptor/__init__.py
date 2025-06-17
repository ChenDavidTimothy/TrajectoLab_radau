"""
MAPTOR: Multiphase Adaptive Trajectory Optimizer

A Python framework for solving optimal control problems using the Radau
Pseudospectral Method. MAPTOR transforms continuous trajectory optimization
problems into solvable nonlinear programming problems through spectral
collocation methods.

Key Features:
    - Intuitive problem definition API
    - Adaptive mesh refinement for high-precision solutions
    - Multiphase trajectory support with automatic phase linking
    - Built-in plotting and solution analysis tools
    - Full type safety with comprehensive type hints

Quick Start:
    >>> import maptor as mtor
    >>> problem = mtor.Problem("Minimum Time Problem")
    >>> phase = problem.set_phase(1)
    >>> t = phase.time(initial=0.0)
    >>> x = phase.state("position", initial=0.0, final=1.0)
    >>> u = phase.control("force", boundary=(-1.0, 1.0))
    >>> phase.dynamics({x: u})
    >>> problem.minimize(t.final)
    >>> phase.mesh([8], [-1.0, 1.0])
    >>> solution = mtor.solve_adaptive(problem)
    >>> if solution.status["success"]:
    ...     solution.plot()

Documentation:
    https://maptor.github.io/maptor/

Repository:
    https://github.com/maptor/maptor

Logging:
    import logging
    logging.getLogger('maptor').setLevel(logging.INFO)  # Major operations
    logging.getLogger('maptor').setLevel(logging.DEBUG)  # Detailed debugging
"""

from __future__ import annotations

import logging

# Import exceptions first - foundational error handling
from maptor.exceptions import (
    ConfigurationError,
    DataIntegrityError,
    InterpolationError,
    MAPTORBaseError,
    SolutionExtractionError,
)

# Core problem definition interface
from maptor.problem import Problem

# Solver functions - primary user interface
from maptor.solver import solve_adaptive, solve_fixed_mesh


# Version and metadata
__version__ = "0.1.0"
__author__ = "David Timothy"
__description__ = "Multiphase Adaptive Trajectory Optimizer"

# Public API - Only these should be used by external code
__all__ = [
    "ConfigurationError",
    "DataIntegrityError",
    "InterpolationError",
    # Exception Hierarchy
    "MAPTORBaseError",
    # Core Classes
    "Problem",
    "SolutionExtractionError",
    # Solver Functions
    "solve_adaptive",
    "solve_fixed_mesh",
]

# Configure logging - no handlers, let user control output
logging.getLogger(__name__).addHandler(logging.NullHandler())


def _get_config() -> dict[str, str]:
    """Get MAPTOR configuration information."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "repository": "https://github.com/maptor/maptor",
        "documentation": "https://maptor.github.io/maptor/",
        "license": "LGPL v3",
    }


# Development and debugging utilities (not in __all__)
def _show_config() -> None:
    """Print MAPTOR configuration (internal use)."""
    config = _get_config()
    print("MAPTOR Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
