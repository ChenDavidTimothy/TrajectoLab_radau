"""
Core type definitions for TrajectoLab.

This module contains all type aliases and protocols used throughout the library,
with a focus on scientific computing patterns and trajectory optimization types.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
_MeshPoints: TypeAlias = NDArray[np.float64]

# ---- Result Types ----
_ErrorTuple: TypeAlias = tuple[float, float, float]  # max_error, mean_error, rms_error

# ---- Optimization Problem Types ----
_StateDict: TypeAlias = dict[str, float | _FloatArray]
_ControlDict: TypeAlias = dict[str, float | _FloatArray]
_ParamDict: TypeAlias = dict[str, Any]  # Parameters can be any type

# ---- Collocation-Specific Types ----
_CollocationNodes: TypeAlias = _Vector  # Nodes used for enforcing dynamics
_StateNodes: TypeAlias = _Vector  # Nodes used for state approximation
_QuadratureWeights: TypeAlias = _Vector  # Weights for numerical integration
_DifferentiationMatrix: TypeAlias = _Matrix  # Maps state values to derivatives
_BarycentricWeights: TypeAlias = _Vector  # Weights for barycentric interpolation

# ---- CasADi Specific Types ----
# Using Any as a placeholder for CasADi types to avoid direct dependency
_CasadiMX: TypeAlias = Any  # Placeholder for ca.MX
_CasadiDM: TypeAlias = Any  # Placeholder for ca.DM
_CasadiSolution: TypeAlias = Any  # Placeholder for casadi solution object
_CasadiOpti: TypeAlias = Any  # Placeholder for casadi.Opti object

# ---- Direct Solver Types ----
_SolverOptions: TypeAlias = dict[str, Any]
_TrajectoryTimePoints: TypeAlias = list[_TimePoint]
_TrajectoryStateValues: TypeAlias = list[list[float]]
_TrajectoryControlValues: TypeAlias = list[list[float]]
_IntegralValue: TypeAlias = float | list[float] | None


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


# ---- Vectorized Function Protocols for Direct Solver ----
class VectorizedDynamicsFunction(Protocol):
    """Protocol for vectorized dynamics function used by direct solvers."""

    def __call__(
        self, states: _FloatArray, controls: _FloatArray, time: _TimePoint, params: _ParamDict
    ) -> list[float] | _Vector | _CasadiMX: ...


class VectorizedObjectiveFunction(Protocol):
    """Protocol for vectorized objective function used by direct solvers."""

    def __call__(
        self,
        t0: _TimePoint,
        tf: _TimePoint,
        x0: _FloatArray,
        xf: _FloatArray,
        q: Sequence[float] | float | None,
        params: _ParamDict,
    ) -> float: ...


class VectorizedIntegrandFunction(Protocol):
    """Protocol for vectorized integrand function used by direct solvers."""

    def __call__(
        self,
        states: _FloatArray,
        controls: _FloatArray,
        time: _TimePoint,
        integral_idx: int,
        params: _ParamDict,
    ) -> float: ...


# ---- Constraint Class and Related Types ----
class Constraint:
    """Represents a constraint with bounds or equality condition."""

    def __init__(
        self,
        val: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        equals: float | None = None,
    ):
        self.val = val
        self.lower = lower
        self.upper = upper
        self.equals = equals

        if equals is not None:
            self.lower = equals
            self.upper = equals


# ---- Direct Solver Constraint Classes ----
class PathConstraint:
    """Path constraint applied at points along the trajectory."""

    def __init__(
        self,
        val: float | _FloatArray | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        equals: float | None = None,
    ):
        self.val = val
        self.min = min_val
        self.max = max_val
        self.equals = equals


class EventConstraint:
    """Event constraint applied at specific events (initial/final points)."""

    def __init__(
        self,
        val: float | _FloatArray | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        equals: float | None = None,
    ):
        self.val = val
        self.min = min_val
        self.max = max_val
        self.equals = equals


# ---- Direct Solver Function Types ----
_VectorizedPathConstraintFunc: TypeAlias = Callable[
    [_FloatArray, _FloatArray, _TimePoint, _ParamDict], list[PathConstraint]
]

_VectorizedEventConstraintFunc: TypeAlias = Callable[
    [_TimePoint, _TimePoint, _FloatArray, _FloatArray, Sequence[float] | float | None, _ParamDict],
    list[EventConstraint],
]


# ---- Collocation Data Structures ----
@dataclass
class RadauNodesAndWeights:
    """Basic nodes and weights for Radau collocation."""

    state_approximation_nodes: _StateNodes
    collocation_nodes: _CollocationNodes
    quadrature_weights: _QuadratureWeights


@dataclass
class RadauBasisComponents:
    """Complete set of components for Radau collocation."""

    state_approximation_nodes: _StateNodes
    collocation_nodes: _CollocationNodes
    quadrature_weights: _QuadratureWeights
    differentiation_matrix: _DifferentiationMatrix
    barycentric_weights_for_state_nodes: _BarycentricWeights
    lagrange_at_tau_plus_one: _Vector


# ---- Initial Guess Classes ----
@dataclass
class DefaultGuessValues:
    """Default values for initial guesses when not provided."""

    state: float = 0.0
    control: float = 0.0
    integral: float = 0.0


@dataclass
class InitialGuess:
    """Initial guess for optimization variables."""

    initial_time_variable: float | None = None
    terminal_time_variable: float | None = None
    states: list[_FloatArray] | None = None
    controls: list[_FloatArray] | None = None
    integrals: Sequence[float] | float | None = None


# ---- Constraint Function Types ----
_ConstraintFunc: TypeAlias = Callable[
    [_StateDict, _ControlDict, float, _ParamDict], "Constraint | list[Constraint]"
]

_EventConstraintFunc: TypeAlias = Callable[
    [float, float, _StateDict, _StateDict, float | Sequence[float] | None, _ParamDict],
    "Constraint | list[Constraint]",
]


# ---- Constants ----
ZERO_TOLERANCE: float = 1e-12  # Tolerance for floating point comparisons
