"""
tl_types.py

Core type definitions for the TrajectoLab project.
This module centralizes custom type aliases, constants, and potentially
more complex type structures as the project grows.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# --- Core Numerical Type Aliases ---

# Type alias for a 1D NumPy array of float64.
# Represents vectors or sequences of floating-point numbers.
FloatArray: TypeAlias = NDArray[np.float64]

# Type alias for a 2D NumPy array of float64.
# Represents matrices or other 2D grids of floating-point numbers.
FloatMatrix: TypeAlias = NDArray[np.float64]  # NDArray[np.float64] can be N-dimensional,
# but this alias clarifies intent for 2D.

# --- Core Numerical Constants ---

# Standard tolerance for floating-point comparisons to zero.
# Used to handle precision issues in numerical algorithms.
ZERO_TOLERANCE: float = 1e-12


# --- Future type definitions can be added below ---
# Example:
# StateVector: TypeAlias = FloatArray
# TimeVector: TypeAlias = FloatArray
# ParameterDict: TypeAlias = dict[str, float | int | str]
