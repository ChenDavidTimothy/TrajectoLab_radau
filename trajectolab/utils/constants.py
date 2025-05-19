"""
Numerical constants for TrajectoLab computations.

This module defines tolerances, thresholds, and physical constants used
throughout the codebase for numerical stability and convergence criteria.
All constants are typed for better type safety.
"""

from typing import TypeAlias


# Type aliases for different kinds of constants
_Tolerance: TypeAlias = float
_Duration: TypeAlias = float
_Factor: TypeAlias = float

# --- Floating Point Precision ---
ZERO_TOLERANCE: _Tolerance = 1e-12
"""Tolerance for considering floating point values as zero."""

MESH_TOLERANCE: _Tolerance = 1e-9
"""Minimum spacing required between mesh points."""

# --- Time Integration ---
MINIMUM_TIME_INTERVAL: _Duration = 1e-6
"""Minimum allowed time interval for optimal control problems."""

DEFAULT_ODE_RTOL: _Tolerance = 1e-7
"""Default relative tolerance for ODE solvers."""

DEFAULT_ODE_ATOL_FACTOR: _Factor = 1e-1
"""Factor for computing absolute tolerance from relative tolerance."""

# --- Iteration Limits ---
MAX_MESH_REFINEMENT_ITERATIONS: int = 100
"""Maximum iterations for adaptive mesh refinement."""

MAX_NLP_SOLVER_ITERATIONS: int = 3000
"""Default maximum iterations for NLP solver."""

# --- Physical Constants (if needed)
# GRAVITY_ACCELERATION: float = 9.80665  # m/s^2

# --- Validation Ranges ---
MIN_POLYNOMIAL_DEGREE: int = 1
"""Minimum allowed polynomial degree for discretization."""

MAX_POLYNOMIAL_DEGREE: int = 50
"""Maximum allowed polynomial degree for discretization."""
