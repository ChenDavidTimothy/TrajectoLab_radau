"""
tl_types.py

Core type definitions for the TrajectoLab project.
This module centralizes custom type aliases, constants, and potentially
more complex type structures as the project grows.
"""

from __future__ import (
    annotations,
)

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TypeVar

import casadi as ca
import numpy as np
from numpy import int_ as np_int_
from numpy.typing import NDArray

# --- Core Numerical Type Aliases ---

# Type alias for a 1D NumPy array of float64.
# Represents vectors or sequences of floating-point numbers.
_FloatArray: TypeAlias = NDArray[np.float64]

# Type alias for a 2D NumPy array of float64.
# Represents matrices or other 2D grids of floating-point numbers.
_FloatMatrix: TypeAlias = NDArray[np.float64]

# Type alias for a 1D NumPy array of integers.
# Useful for indices, counts, or integer-valued parameters.
_IntArray: TypeAlias = NDArray[np_int_]


# --- Core Numerical Constants ---

# Standard tolerance for floating-point comparisons to zero.
# Used to handle precision issues in numerical algorithms.
ZERO_TOLERANCE: float = 1e-12


# --- CasADi Type Aliases ---
_CasadiMX: TypeAlias = ca.MX
_CasadiDM: TypeAlias = ca.DM
_CasadiMatrix: TypeAlias = _CasadiMX | _CasadiDM  # Union type for any CasADi matrix
_CasadiOpti: TypeAlias = ca.Opti
_CasadiOptiSol: TypeAlias = ca.OptiSol


# --- Problem Structure Data Classes and Type Aliases ---
_ProblemParameters: TypeAlias = dict[str, object]

# Dictionary type aliases for Problem class
_StateDictType: TypeAlias = dict[str, float | _CasadiMX]
_ControlDictType: TypeAlias = dict[str, float | _CasadiMX]

# Forward reference for Problem module's Constraint class
# (without creating a duplicate class)
ConstraintType: TypeAlias = Any

# User-facing function types for Problem class using ConstraintType as placeholder
_DynamicsFuncType: TypeAlias = Callable[
    [_StateDictType, _ControlDictType, float | _CasadiMX, _ProblemParameters | None], _StateDictType
]
_ObjectiveFuncType: TypeAlias = Callable[
    [
        float | _CasadiMX,
        float | _CasadiMX,
        _StateDictType,
        _StateDictType,
        float | _CasadiMX | None,
        _ProblemParameters | None,
    ],
    float | _CasadiMX,
]
_IntegrandFuncType: TypeAlias = Callable[
    [_StateDictType, _ControlDictType, float | _CasadiMX, _ProblemParameters | None],
    float | _CasadiMX,
]
_ConstraintFuncType: TypeAlias = Callable[
    [_StateDictType, _ControlDictType, float | _CasadiMX, _ProblemParameters | None],
    "ConstraintType | list[ConstraintType]",
]
_EventConstraintFuncType: TypeAlias = Callable[
    [
        float | _CasadiMX,
        float | _CasadiMX,
        _StateDictType,
        _StateDictType,
        float | _CasadiMX | None,
        _ProblemParameters | None,
    ],
    "ConstraintType | list[ConstraintType]",
]


@dataclass
class PathConstraint:
    val: _CasadiMX
    min_val: float | None = None
    max_val: float | None = None
    equals: float | None = None


@dataclass
class EventConstraint:
    val: _CasadiMX
    min_val: float | None = None
    max_val: float | None = None
    equals: float | None = None


# --- Callable Types for Solver ---
# (state, control, time, params) -> state_derivative
_DynamicsCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _ProblemParameters],
    list[_CasadiMX] | _CasadiMX | Sequence[_CasadiMX],
]

# (t0, tf, x0, xf, integrals, params) -> objective_value
_ObjectiveCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX | None, _ProblemParameters],
    _CasadiMX,
]

# (state, control, time, integral_idx, params) -> integrand_value
_IntegralIntegrandCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, int, _ProblemParameters],
    _CasadiMX,
]

_PathConstraintsCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _ProblemParameters],
    list[PathConstraint] | PathConstraint,
]

_EventConstraintsCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX | None, _ProblemParameters],
    list[EventConstraint] | EventConstraint,
]

# --- Helper Type Aliases for Trajectories and Guesses ---
_ListOfCasadiMX: TypeAlias = list[_CasadiMX]
_TrajectoryData: TypeAlias = list[_FloatArray]  # List of 1D arrays for each state/control component

# For initial guesses, states/controls are typically a list of 2D matrices,
# one per mesh interval: [num_variables, num_nodes_in_interval]
_InitialGuessTrajectory: TypeAlias = list[_FloatMatrix]
_InitialGuessIntegrals: TypeAlias = float | _FloatArray | list[float] | None

# --- Types for Adaptive Mesh Refinement ---
# For evaluator callables
_StateEvaluator: TypeAlias = Callable[[float | _FloatArray], _FloatArray]
_ControlEvaluator: TypeAlias = Callable[[float | _FloatArray], _FloatArray]

# ODE solver related types
_DynamicsRHSCallable: TypeAlias = Callable[[float, _FloatArray], _FloatArray]


# Define a protocol for the ODE solver result
class ODESolverResult(Protocol):
    """Protocol for the result of ODE solvers like solve_ivp."""

    y: _FloatMatrix
    t: _FloatArray
    success: bool
    message: str


# Make the ODESolverCallable more flexible with optional kwargs
_ODESolverCallable: TypeAlias = Callable[..., ODESolverResult]

# Gamma normalization factors type
_GammaFactors: TypeAlias = _FloatArray

# Type variable for generic functions
T = TypeVar("T")


# Protocol for Problem to avoid circular imports
class ProblemProtocol(Protocol):
    """Protocol defining the expected interface of a Problem object."""

    name: str
    _states: dict[str, dict[str, Any]]
    _controls: dict[str, dict[str, Any]]
    _parameters: _ProblemParameters
    _t0_bounds: tuple[float, float]
    _tf_bounds: tuple[float, float]
    _dynamics_func: _DynamicsFuncType | None
    _objective_type: str | None
    _objective_func: _ObjectiveFuncType | None
    _path_constraints: list[_ConstraintFuncType]
    _event_constraints: list[_EventConstraintFuncType]
    _num_integrals: int
    _integral_functions: list[_IntegrandFuncType]
    collocation_points_per_interval: list[int]
    global_normalized_mesh_nodes: _FloatArray | None
    initial_guess: Any
    solver_options: dict[str, object]

    def get_dynamics_function(self) -> _DynamicsCallable: ...
    def get_objective_function(self) -> _ObjectiveCallable: ...
    def get_integrand_function(self) -> _IntegralIntegrandCallable | None: ...
    def get_path_constraints_function(self) -> _PathConstraintsCallable | None: ...
    def get_event_constraints_function(self) -> _EventConstraintsCallable: ...
