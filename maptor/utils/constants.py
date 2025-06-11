from typing import TypeAlias


_Tolerance: TypeAlias = float
_Duration: TypeAlias = float
_Factor: TypeAlias = float

# IEEE 754 fundamental constants
MACHINE_EPSILON: _Tolerance = 2.22e-16

# Primary tolerance hierarchy (powers of 100 scaling for clear separation)
NUMERICAL_ZERO: _Tolerance = 1e-14  # 100x machine epsilon - accumulated error safety
COORDINATE_PRECISION: _Tolerance = 1e-12  # Spatial/mesh operations on normalized coordinates
TIME_PRECISION: _Tolerance = 1e-10  # Temporal operations and comparisons
ALGORITHM_PRECISION: _Tolerance = 1e-8  # Solver convergence and iteration control

# Large value constants (well below IEEE 754 overflow ≈ 1.8e308)
LARGE_VALUE: float = 1e10  # General "infinity" replacement
VERY_LARGE_VALUE: float = 1e12  # When larger bounds needed

# Derived constants (reusing primary hierarchy)
ZERO_TOLERANCE: _Tolerance = NUMERICAL_ZERO
MESH_TOLERANCE: _Tolerance = COORDINATE_PRECISION
MINIMUM_TIME_INTERVAL: _Duration = TIME_PRECISION
TRAJECTORY_TIME_TOLERANCE: _Tolerance = TIME_PRECISION
INTERPOLATION_TOLERANCE: _Tolerance = COORDINATE_PRECISION
BOUNDARY_MATCHING_TOLERANCE: _Tolerance = COORDINATE_PRECISION

# Bound constants (reusing large values)
DEFAULT_BOUND_INFINITY: float = LARGE_VALUE
DEFAULT_TIME_BOUND_INFINITY: float = LARGE_VALUE

# ODE solver constants
DEFAULT_ODE_RTOL: _Tolerance = ALGORITHM_PRECISION
DEFAULT_ODE_ATOL_FACTOR: _Factor = 1e-2  # Standard 100x ratio
DEFAULT_ODE_METHOD: str = "RK45"
DEFAULT_ODE_MAX_STEP: float | None = None
DEFAULT_ERROR_SIM_POINTS: int = 50

# NLP solver defaults
DEFAULT_NLP_MAX_ITERATIONS: int = 3000
DEFAULT_NLP_TOLERANCE: float = ALGORITHM_PRECISION

# Adaptive algorithm defaults
DEFAULT_ADAPTIVE_ERROR_TOLERANCE: float = 1e-6
DEFAULT_ADAPTIVE_MAX_ITERATIONS: int = 10
DEFAULT_MIN_POLYNOMIAL_DEGREE: int = 3
DEFAULT_MAX_POLYNOMIAL_DEGREE: int = 10

# Plotting constants (aesthetic, no numerical impact)
DEFAULT_FIGURE_SIZE: tuple[float, float] = (12.0, 8.0)
DEFAULT_GRID_ALPHA: float = 0.3
DEFAULT_PHASE_BOUNDARY_ALPHA: float = 0.7
DEFAULT_PHASE_BOUNDARY_LINEWIDTH: float = 2.0
