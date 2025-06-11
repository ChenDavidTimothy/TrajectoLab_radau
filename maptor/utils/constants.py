from typing import TypeAlias


_Tolerance: TypeAlias = float
_Duration: TypeAlias = float
_Factor: TypeAlias = float
_Count: TypeAlias = int
_Limit: TypeAlias = int
_Bound: TypeAlias = float

# ===========================
# CORE NUMERICAL TOLERANCES
# ===========================

# Tolerance for considering floating point values as zero.
ZERO_TOLERANCE: _Tolerance = 1e-18

# Minimum spacing required between mesh points.
MESH_TOLERANCE: _Tolerance = 1e-9

# Minimum allowed time interval for optimal control problems.
MINIMUM_TIME_INTERVAL: _Duration = 1e-6

# ===========================
# ARBITRARY TOLERANCES - CONFIGURABLE
# ===========================

# Tolerance for interpolation boundary checking.
INTERPOLATION_TOLERANCE: _Tolerance = 1e-10

# Tolerance for considering interval width as zero.
INTERVAL_WIDTH_TOLERANCE: _Tolerance = 1e-12

# General near-zero tolerance for numerical comparisons.
NEAR_ZERO_TOLERANCE: _Tolerance = 1e-12

# ===========================
# DEFAULT BOUNDS - ARBITRARY
# ===========================

# Default lower bound for unconstrained variables.
DEFAULT_VARIABLE_LOWER_BOUND: _Bound = -1e5

# Default upper bound for unconstrained variables.
DEFAULT_VARIABLE_UPPER_BOUND: _Bound = 1e5

# Default lower bound for time variables.
DEFAULT_TIME_LOWER_BOUND: _Bound = -1e6

# Default upper bound for time variables.
DEFAULT_TIME_UPPER_BOUND: _Bound = 1e6

# ===========================
# CACHE AND PERFORMANCE
# ===========================

# Default LRU cache size for expensive computations.
DEFAULT_LRU_CACHE_SIZE: _Count = 32

# Default maximum NLP iterations.
DEFAULT_NLP_MAX_ITERATIONS: _Limit = 3000

# ===========================
# ODE SOLVER DEFAULTS
# ===========================

# Default relative tolerance for ODE solvers.
DEFAULT_ODE_RTOL: _Tolerance = 1e-7

# Factor for computing absolute tolerance from relative tolerance (atol = rtol * factor).
DEFAULT_ODE_ATOL_FACTOR: _Factor = 1e-2

# Default ODE integration method.
DEFAULT_ODE_METHOD: str = "RK45"

# Default maximum step size for ODE solver (None = no limit).
DEFAULT_ODE_MAX_STEP: float | None = None

# ===========================
# ADAPTIVE ALGORITHM DEFAULTS
# ===========================

# Default number of points for error simulation.
DEFAULT_ERROR_SIM_POINTS: int = 50

# Default maximum adaptive refinement iterations.
DEFAULT_ADAPTIVE_MAX_ITERATIONS: _Count = 10

# Default minimum polynomial degree.
DEFAULT_MIN_POLYNOMIAL_DEGREE: _Count = 3

# Default maximum polynomial degree.
DEFAULT_MAX_POLYNOMIAL_DEGREE: _Count = 10

# ===========================
# ERROR BOUNDS - SAFETY
# ===========================

# Minimum error value to prevent underflow.
MINIMUM_ERROR_VALUE: _Tolerance = 1e-15
