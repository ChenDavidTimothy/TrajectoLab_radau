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
from typing import Any, Callable, TypeAlias

import casadi as ca
import numpy as np
from numpy import int_ as np_int_
from numpy.typing import NDArray

# --- Core Numerical Type Aliases ---

# Type alias for a 1D NumPy array of float64.
# Represents vectors or sequences of floating-point numbers.
FloatArray: TypeAlias = NDArray[np.float64]

# Type alias for a 2D NumPy array of float64.
# Represents matrices or other 2D grids of floating-point numbers.
FloatMatrix: TypeAlias = NDArray[np.float64]

# Type alias for a 1D NumPy array of integers.
# Useful for indices, counts, or integer-valued parameters.
IntArray: TypeAlias = NDArray[np_int_]


# --- Core Numerical Constants ---

# Standard tolerance for floating-point comparisons to zero.
# Used to handle precision issues in numerical algorithms.
ZERO_TOLERANCE: float = 1e-12


# --- CasADi Type Aliases ---
CasadiMX: TypeAlias = ca.MX
CasadiDM: TypeAlias = ca.DM
CasadiOpti: TypeAlias = ca.Opti
CasadiOptiSol: TypeAlias = ca.OptiSol


# --- Problem Structure Data Classes and Type Aliases ---
ProblemParameters: TypeAlias = dict[str, Any]


@dataclass
class PathConstraint:
    val: CasadiMX
    min_val: float | None = None
    max_val: float | None = None
    equals: float | None = None


@dataclass
class EventConstraint:
    val: CasadiMX
    min_val: float | None = None
    max_val: float | None = None
    equals: float | None = None


# --- Callable Types for OptimalControlProblem ---
# (state, control, time, params) -> state_derivative
DynamicsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters | None],
    list[CasadiMX] | CasadiMX,
]

# (t0, tf, x0, xf, integrals, params) -> objective_value
ObjectiveCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters | None],
    CasadiMX,
]

# (state, control, time, integral_idx, params) -> integrand_value
IntegralIntegrandCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, int, ProblemParameters | None],
    CasadiMX,
]

PathConstraintsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters | None],
    list[PathConstraint] | PathConstraint,
]

EventConstraintsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters | None],
    list[EventConstraint] | EventConstraint,
]

# --- Helper Type Aliases for Trajectories and Guesses ---
ListOfCasadiMX: TypeAlias = list[CasadiMX]
TrajectoryData: TypeAlias = list[
    FloatArray
]  # e.g., list of 1D arrays for each state/control component

# For initial guesses, states/controls are typically a list of 2D matrices,
# one per mesh interval: [num_variables, num_nodes_in_interval]
InitialGuessTrajectory: TypeAlias = list[FloatMatrix]
InitialGuessIntegrals: TypeAlias = float | FloatArray
