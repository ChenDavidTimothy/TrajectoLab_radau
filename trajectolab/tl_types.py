"""
tl_types.py

Core type definitions for the TrajectoLab project.
This module centralizes custom type aliases, constants, and potentially
more complex type structures as the project grows.
"""

from __future__ import (  # Ensures PathConstraint/EventConstraint resolve correctly in Callables
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
# We'll use typing.Any initially and then import the actual Constraint class
# when needed to avoid circular imports
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


# --- Callable Types for OptimalControlProblem ---
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
_TrajectoryData: TypeAlias = list[
    _FloatArray
]  # e.g., list of 1D arrays for each state/control component

# For initial guesses, states/controls are typically a list of 2D matrices,
# one per mesh interval: [num_variables, num_nodes_in_interval]
_InitialGuessTrajectory: TypeAlias = list[_FloatMatrix]
_InitialGuessIntegrals: TypeAlias = float | _FloatArray | list[float] | None

# --- PHSAdaptive Types ---
# Solution and legacy problem references without circular imports
_LegacySolutionType: TypeAlias = Any  # Will be OptimalControlSolution
_LegacyProblemType: TypeAlias = Any  # Will be OptimalControlProblem

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
