"""
MAPTOR: A Python framework for multiphase optimal trajectory generation

This package provides a unified interface for solving multiphase optimal control problems
using the Radau Pseudospectral Method for direct collocation.

Logging:
By default, MAPTOR produces no output. To enable logging::

    import logging
    logging.getLogger('maptor').setLevel(logging.INFO)  # Major operations
    logging.getLogger('maptor').setLevel(logging.DEBUG)  # Detailed debugging
"""

import logging

# Import MAPTOR-specific exceptions for user access
from maptor.exceptions import (
    ConfigurationError,
    DataIntegrityError,
    InterpolationError,
    MAPTORBaseError,
    SolutionExtractionError,
)
from maptor.problem import Problem
from maptor.solver import solve_adaptive, solve_fixed_mesh


__all__ = [
    "ConfigurationError",
    "DataIntegrityError",
    "InterpolationError",
    "MAPTORBaseError",
    "Problem",
    "SolutionExtractionError",
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
