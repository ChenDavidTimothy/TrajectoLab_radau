"""
Core type definitions for TrajectoLab.

This module contains all type aliases and protocols used throughout the library,
with a focus on scientific computing patterns and trajectory optimization types.
"""

from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

# ---- Numeric Array Types ----
_FloatArray: TypeAlias = NDArray[np.float64]
_IntArray: TypeAlias = NDArray[np.int32]
_BoolArray: TypeAlias = NDArray[np.bool_]

# ---- Mathematical Shapes ----
_Vector: TypeAlias = NDArray[np.float64]  # 1D array
_Matrix: TypeAlias = NDArray[np.float64]  # 2D array

# ---- Array-Like Input Types ----
_ArrayLike: TypeAlias = Sequence[float] | _FloatArray
_IntArrayLike: TypeAlias = Sequence[int] | _IntArray

# ---- Time and Mesh Types ----
_TimePoint: TypeAlias = float
_NormalizedTimePoint: TypeAlias = float  # tau ∈ [-1, 1]
_MeshPoints: TypeAlias = _FloatArray

# ---- Result Types ----
_ErrorTuple: TypeAlias = tuple[float, float, float]  # max_error, mean_error, rms_error

# ---- Optimization Problem Types ----
_StateDict: TypeAlias = dict[str, float | _FloatArray]
_ControlDict: TypeAlias = dict[str, float | _FloatArray]
_ParamDict: TypeAlias = dict[str, Any]  # Parameters can be any type


# ---- Protocol Classes for Callback Functions ----
class DynamicsFunction(Protocol):
    """Protocol for system dynamics functions."""

    def __call__(
        self, states: _StateDict, controls: _ControlDict, time: float, params: _ParamDict
    ) -> _StateDict: ...


class ObjectiveFunction(Protocol):
    """Protocol for objective functions."""

    def __call__(
        self,
        t0: float,
        tf: float,
        x0: _StateDict,
        xf: _StateDict,
        integrals: Sequence[float] | float | None,
        params: _ParamDict,
    ) -> float: ...


class IntegrandFunction(Protocol):
    """Protocol for integrand functions used in cost integration."""

    def __call__(
        self, states: _StateDict, controls: _ControlDict, time: float, params: _ParamDict
    ) -> float: ...


# ---- Constants ----
ZERO_TOLERANCE: float = 1e-12  # Tolerance for floating point comparisons
