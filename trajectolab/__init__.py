# trajectolab/__init__.py
"""
TrajectoLab: A Python framework for multiphase optimal trajectory generation

This package provides a unified interface for solving multiphase optimal control problems
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

__version__ = "0.3.0"  # Updated for multiphase support


logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to see everything
    format="%(name)s  - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ],
)
# PRODUCTION LOGGING: Silent by default, user controls everything
logging.getLogger(__name__).setLevel(logging.INFO)
