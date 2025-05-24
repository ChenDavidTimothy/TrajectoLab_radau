"""
TrajectoLab: A Python framework for optimal trajectory generation - ENHANCED WITH FAIL-FAST ERROR HANDLING

This package provides a unified interface for solving optimal control problems
using the Radau Pseudospectral Method for direct collocation.

Enhanced with targeted, fail-fast error handling for critical TrajectoLab operations
while maintaining lean, readable code focused on real failure modes.
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
    # Exceptions for advanced users
    "TrajectoLabBaseError",
    "solve_adaptive",
    "solve_fixed_mesh",
]

__version__ = "0.2.1"  # Incremented for enhanced error handling

# Configure TrajectoLab root logger with appropriate level
_logger = logging.getLogger("trajectolab")
if not _logger.handlers:
    # Only add handler if none exists to avoid duplicate logging
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)  # Default to INFO level
