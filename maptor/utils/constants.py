from typing import TypeAlias


_Tolerance: TypeAlias = float
_Duration: TypeAlias = float
_Factor: TypeAlias = float

# Tolerance for considering floating point values as zero.
ZERO_TOLERANCE: _Tolerance = 1e-18

# Minimum spacing required between mesh points.
MESH_TOLERANCE: _Tolerance = 1e-9

# Minimum allowed time interval for optimal control problems.
MINIMUM_TIME_INTERVAL: _Duration = 1e-6


# Default relative tolerance for ODE solvers.
DEFAULT_ODE_RTOL: _Tolerance = 1e-7

# Factor for computing absolute tolerance from relative tolerance (atol = rtol * factor).
DEFAULT_ODE_ATOL_FACTOR: _Factor = 1e-2

# Default ODE integration method.
DEFAULT_ODE_METHOD: str = "RK45"

# Default maximum step size for ODE solver (None = no limit).
DEFAULT_ODE_MAX_STEP: float | None = None

# Default number of points for error simulation.
DEFAULT_ERROR_SIM_POINTS: int = 50
