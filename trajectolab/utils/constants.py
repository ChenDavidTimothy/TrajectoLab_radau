"""
Numerical constants and tolerances used throughout TrajectoLab computations.
"""

from typing import TypeAlias


# Type aliases for different kinds of constants
_Tolerance: TypeAlias = float
_Duration: TypeAlias = float
_Factor: TypeAlias = float

# --- Floating Point Precision ---
ZERO_TOLERANCE: _Tolerance = 1e-17
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
