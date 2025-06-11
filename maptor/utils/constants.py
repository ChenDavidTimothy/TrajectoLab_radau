import numpy as np


# ===========================
# MACHINE PRECISION FOUNDATION
# ===========================


# Fundamental machine precision constants
MACHINE_EPS = float(np.finfo(np.float64).eps)  # ~2.22e-16
SQRT_MACHINE_EPS = float(np.sqrt(MACHINE_EPS))  # ~1.49e-8
CBRT_MACHINE_EPS = float(np.cbrt(MACHINE_EPS))  # ~6.06e-6
FOURTH_ROOT_MACHINE_EPS = float(np.sqrt(SQRT_MACHINE_EPS))  # ~1.22e-4


MAX_MATHEMATICAL_CONDITION = 1.0 / SQRT_MACHINE_EPS
RELATIVE_PRECISION = float(np.sqrt(MACHINE_EPS))
# ===========================
# CORE NUMERICAL TOLERANCES - MACHINE PRECISION BASED
# ===========================


# Minimum physical time interval
MINIMUM_TIME_INTERVAL = SQRT_MACHINE_EPS  # ~1.49e-8

# ===========================
# ADAPTIVE ALGORITHM CONSTANTS - EXTRACTED MAGIC NUMBERS
# ===========================

# Minimum refinement increment (extracted from refinement.py)
MIN_REFINEMENT_NODES = 1

# Minimum h-refinement subintervals (extracted from refinement.py)
MIN_H_SUBINTERVALS = 2


# ===========================
# ODE SOLVER DEFAULTS - MACHINE PRECISION BASED
# ===========================

# ODE relative tolerance (cube root for derivative accuracy)
DEFAULT_ODE_RTOL = CBRT_MACHINE_EPS  # ~6.06e-6

# Absolute tolerance factor
DEFAULT_ODE_ATOL_FACTOR = 1e-2

# Default ODE method
DEFAULT_ODE_METHOD = "RK45"

# Default maximum step size
DEFAULT_ODE_MAX_STEP = None

# ===========================
# CACHE AND PERFORMANCE - UNCHANGED
# ===========================

DEFAULT_LRU_CACHE_SIZE = 32
DEFAULT_NLP_MAX_ITERATIONS = 3000

# ===========================
# ADAPTIVE ALGORITHM DEFAULTS - UNCHANGED
# ===========================

DEFAULT_ERROR_SIM_POINTS = 50
DEFAULT_ADAPTIVE_MAX_ITERATIONS = 10
DEFAULT_MIN_POLYNOMIAL_DEGREE = 3
DEFAULT_MAX_POLYNOMIAL_DEGREE = 10
