"""
Core type definitions for the TrajectoLab project with symbolic support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import casadi as ca
import numpy as np
from numpy import int_ as np_int_
from numpy.typing import NDArray

# --- Core Numerical Type Aliases ---
_FloatArray: TypeAlias = NDArray[np.float64]
_FloatMatrix: TypeAlias = NDArray[np.float64]
_IntArray: TypeAlias = NDArray[np_int_]

# --- Core Symbolic Type Aliases ---
_SymType: TypeAlias = ca.MX
_SymExpr: TypeAlias = ca.MX | float | int

# --- Core Numerical Constants ---
ZERO_TOLERANCE: float = 1e-12

# --- CasADi Type Aliases ---
_CasadiMX: TypeAlias = ca.MX
_CasadiDM: TypeAlias = ca.DM
_CasadiMatrix: TypeAlias = _CasadiMX | _CasadiDM
_CasadiOpti: TypeAlias = ca.Opti
_CasadiOptiSol: TypeAlias = ca.OptiSol
_CasadiFunction: TypeAlias = ca.Function

# --- Problem Structure Data Classes and Type Aliases ---
_ProblemParameters: TypeAlias = dict[str, object]

# Dictionary type aliases for Problem class - keep for compatibility
_StateDictType: TypeAlias = dict[str, float | _CasadiMX]
_ControlDictType: TypeAlias = dict[str, float | _CasadiMX]

# Forward reference for Problem module's Constraint class
ConstraintType: TypeAlias = Any

_ConstraintValue: TypeAlias = _CasadiMX | float | _FloatArray


# --- Constraints ---
class PathConstraint:
    """Path constraint for the solver."""

    def __init__(
        self,
        val: _CasadiMX | float,
        min_val: float | None = None,
        max_val: float | None = None,
        equals: float | None = None,
    ) -> None:
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.equals = equals


class EventConstraint:
    """Event constraint for the solver."""

    def __init__(
        self,
        val: _CasadiMX | float,
        min_val: float | None = None,
        max_val: float | None = None,
        equals: float | None = None,
    ) -> None:
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.equals = equals


# --- Function Protocol Classes for Symbolic Problem Description ---
class TimeVariable(Protocol):
    """Protocol for time variable with initial/final properties."""

    @property
    def initial(self) -> _SymType: ...

    @property
    def final(self) -> _SymType: ...

    def __call__(self) -> _SymType: ...


# --- Function Protocol Classes for User-Facing APIs ---
# These protocols allow for flexible parameter signatures while enforcing return types


class DynamicsFuncProtocol(Protocol):
    """Protocol for user-defined dynamics functions."""

    def __call__(
        self, states: _StateDictType, controls: _ControlDictType, *args: Any, **kwargs: Any
    ) -> _StateDictType:
        """
        Defines the dynamics of the system.

        Args:
            states: Dictionary of state variables
            controls: Dictionary of control variables
            *args: Additional positional arguments (e.g., time)
            **kwargs: Additional keyword arguments (e.g., parameters)

        Returns:
            Dictionary of state derivatives
        """
        ...


class ObjectiveFuncProtocol(Protocol):
    """Protocol for user-defined objective functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> float | _CasadiMX:
        """
        Defines the objective function to minimize.

        Args can include:
            initial_time: Initial time
            final_time: Final time
            initial_states: Dictionary of initial state values
            final_states: Dictionary of final state values
            integrals: Integral values (if using integral objective)
            params: Problem parameters

        Returns:
            Objective value to minimize
        """
        ...


class IntegrandFuncProtocol(Protocol):
    """Protocol for user-defined integral functions."""

    def __call__(
        self, states: _StateDictType, controls: _ControlDictType, *args: Any, **kwargs: Any
    ) -> float | _CasadiMX:
        """
        Defines the integrand for an integral cost term.

        Args:
            states: Dictionary of state variables
            controls: Dictionary of control variables
            *args: Additional positional arguments (e.g., time)
            **kwargs: Additional keyword arguments (e.g., parameters)

        Returns:
            Integrand value
        """
        ...


class ConstraintFuncProtocol(Protocol):
    """Protocol for user-defined path constraint functions."""

    def __call__(
        self, states: _StateDictType, controls: _ControlDictType, *args: Any, **kwargs: Any
    ) -> ConstraintType | list[ConstraintType]:
        """
        Defines path constraints.

        Args:
            states: Dictionary of state variables
            controls: Dictionary of control variables
            *args: Additional positional arguments (e.g., time)
            **kwargs: Additional keyword arguments (e.g., parameters)

        Returns:
            Constraint or list of constraints
        """
        ...


class EventConstraintFuncProtocol(Protocol):
    """Protocol for user-defined event constraint functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> ConstraintType | list[ConstraintType]:
        """
        Defines event (boundary) constraints.

        Args can include:
            initial_time: Initial time
            final_time: Final time
            initial_states: Dictionary of initial state values
            final_states: Dictionary of final state values
            integrals: Integral values
            params: Problem parameters

        Returns:
            Constraint or list of constraints
        """
        ...


# User-facing function types using Protocol classes
_DynamicsFuncType: TypeAlias = DynamicsFuncProtocol
_ObjectiveFuncType: TypeAlias = ObjectiveFuncProtocol
_IntegrandFuncType: TypeAlias = IntegrandFuncProtocol
_ConstraintFuncType: TypeAlias = ConstraintFuncProtocol
_EventConstraintFuncType: TypeAlias = EventConstraintFuncProtocol


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
    _num_integrals: int
    collocation_points_per_interval: list[int]
    global_normalized_mesh_nodes: _FloatArray | None
    initial_guess: Any
    solver_options: dict[str, object]

    # Symbolic attributes
    _sym_states: dict[str, _SymType]
    _sym_controls: dict[str, _SymType]
    _sym_parameters: dict[str, _SymType]
    _sym_time: _SymType | None
    _sym_time_initial: _SymType | None
    _sym_time_final: _SymType | None
    _dynamics_expressions: dict[_SymType, _SymExpr]
    _objective_expression: _SymExpr | None
    _objective_type: str | None
    _constraints: list[_SymExpr]
    _integral_expressions: list[_SymExpr]
    _integral_symbols: list[_SymType]

    def get_dynamics_function(self) -> _DynamicsCallable: ...
    def get_objective_function(self) -> _ObjectiveCallable: ...
    def get_integrand_function(self) -> _IntegralIntegrandCallable | None: ...
    def get_path_constraints_function(self) -> _PathConstraintsCallable | None: ...
    def get_event_constraints_function(self) -> _EventConstraintsCallable | None: ...
