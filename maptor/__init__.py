"""
Logging:
    import logging
    logging.getLogger('maptor').setLevel(logging.INFO)  # Major operations
    logging.getLogger('maptor').setLevel(logging.DEBUG)  # Detailed debugging
"""

import logging

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

__version__ = "0.1.0"


logging.getLogger(__name__).addHandler(logging.NullHandler())
