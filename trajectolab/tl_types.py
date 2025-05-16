"""
tl_types.py

Core type definitions for the TrajectoLab project.
This module centralizes custom type aliases, constants, and potentially
more complex type structures as the project grows.
"""

from __future__ import (  # Ensures PathConstraint/EventConstraint resolve correctly in Callables
    annotations,
)

from dataclasses import dataclass
from typing import Callable, TypeAlias

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
    [_CasadiMX, _CasadiMX, _CasadiMX, _ProblemParameters | None],
    list[_CasadiMX] | _CasadiMX,
]

# (t0, tf, x0, xf, integrals, params) -> objective_value
_ObjectiveCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX | None, _ProblemParameters | None],
    _CasadiMX,
]

# (state, control, time, integral_idx, params) -> integrand_value
_IntegralIntegrandCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, int, _ProblemParameters | None],
    _CasadiMX,
]

_PathConstraintsCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _ProblemParameters | None],
    list[PathConstraint] | PathConstraint,
]

_EventConstraintsCallable: TypeAlias = Callable[
    [_CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX, _CasadiMX | None, _ProblemParameters | None],
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
