from typing import TypeAlias


_Tolerance: TypeAlias = float
_Duration: TypeAlias = float
_Factor: TypeAlias = float

ZERO_TOLERANCE: _Tolerance = 1e-18
"""Tolerance for considering floating point values as zero."""

MESH_TOLERANCE: _Tolerance = 1e-9
"""Minimum spacing required between mesh points."""

MINIMUM_TIME_INTERVAL: _Duration = 1e-6
"""Minimum allowed time interval for optimal control problems."""

# ODE Solver Defaults - SINGLE SOURCE OF TRUTH
DEFAULT_ODE_RTOL: _Tolerance = 1e-7
"""Default relative tolerance for ODE solvers."""

DEFAULT_ODE_ATOL_FACTOR: _Factor = 1e-2
"""Factor for computing absolute tolerance from relative tolerance (atol = rtol * factor)."""

DEFAULT_ODE_METHOD: str = "RK45"
"""Default ODE integration method."""

DEFAULT_ODE_MAX_STEP: float | None = None
"""Default maximum step size for ODE solver (None = no limit)."""

DEFAULT_ERROR_SIM_POINTS: int = 50
"""Default number of points for error simulation."""
