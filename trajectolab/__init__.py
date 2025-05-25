# trajectolab/__init__.py
"""
TrajectoLab: A Python framework for optimal trajectory generation

This package provides a unified interface for solving optimal control problems
using the Radau Pseudospectral Method for direct collocation.

Logging:
By default, TrajectoLab produces no output. To enable logging::

    import logging
    logging.getLogger('trajectolab').setLevel(logging.INFO)  # Major operations
    logging.getLogger('trajectolab').setLevel(logging.DEBUG)  # Detailed debugging
"""

import logging

# Import TrajectoLab-specific exceptions for user access
from trajectolab.exceptions import (
    ConfigurationError,
    DataIntegrityError,
    InterpolationError,
    SolutionExtractionError,
    TrajectoLabBaseError,
)
from trajectolab.problem import Problem
from trajectolab.solver import solve_adaptive, solve_fixed_mesh


__all__ = [
    "ConfigurationError",
    "DataIntegrityError",
    "InterpolationError",
    "Problem",
    "SolutionExtractionError",
    "TrajectoLabBaseError",
    "solve_adaptive",
    "solve_fixed_mesh",
]

__version__ = "0.2.1"

# PRODUCTION LOGGING: Silent by default, user controls everything
logging.getLogger(__name__).addHandler(logging.NullHandler())
