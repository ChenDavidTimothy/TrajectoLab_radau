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


# --- Core Numerical Type Aliases (Public) ---
FloatArray: TypeAlias = NDArray[np.float64]
FloatMatrix: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np_int_]

# --- Core Symbolic Type Aliases (Public) ---
SymType: TypeAlias = ca.MX
SymExpr: TypeAlias = ca.MX | float | int

# --- Core Numerical Constants ---
ZERO_TOLERANCE: float = 1e-12

# --- CasADi Type Aliases (Public) ---
CasadiMX: TypeAlias = ca.MX
CasadiDM: TypeAlias = ca.DM
CasadiMatrix: TypeAlias = CasadiMX | CasadiDM
CasadiOpti: TypeAlias = ca.Opti
CasadiOptiSol: TypeAlias = ca.OptiSol
CasadiFunction: TypeAlias = ca.Function

# --- Problem Structure Data Classes and Type Aliases ---
ProblemParameters: TypeAlias = dict[str, object]

# Dictionary type aliases for Problem class - keep for compatibility
StateDictType: TypeAlias = dict[str, float | CasadiMX]
ControlDictType: TypeAlias = dict[str, float | CasadiMX]

# Forward reference for Problem module's Constraint class
ConstraintType: TypeAlias = Any

ConstraintValue: TypeAlias = CasadiMX | float | FloatArray


# --- Constraints ---
class PathConstraint:
    """Path constraint for the solver."""

    def __init__(
        self,
        val: CasadiMX | float,
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
        val: CasadiMX | float,
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
    def initial(self) -> SymType: ...

    @property
    def final(self) -> SymType: ...

    def __call__(self) -> SymType: ...


# --- Function Protocol Classes for User-Facing APIs ---
# These protocols allow for flexible parameter signatures while enforcing return types


class DynamicsFuncProtocol(Protocol):
    """Protocol for user-defined dynamics functions."""

    def __call__(
        self, states: StateDictType, controls: ControlDictType, *args: Any, **kwargs: Any
    ) -> StateDictType:
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

    def __call__(self, *args: Any, **kwargs: Any) -> float | CasadiMX:
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
        self, states: StateDictType, controls: ControlDictType, *args: Any, **kwargs: Any
    ) -> float | CasadiMX:
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
        self, states: StateDictType, controls: ControlDictType, *args: Any, **kwargs: Any
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
DynamicsFuncType: TypeAlias = DynamicsFuncProtocol
ObjectiveFuncType: TypeAlias = ObjectiveFuncProtocol
IntegrandFuncType: TypeAlias = IntegrandFuncProtocol
ConstraintFuncType: TypeAlias = ConstraintFuncProtocol
EventConstraintFuncType: TypeAlias = EventConstraintFuncProtocol


# --- Callable Types for Solver ---
# (state, control, time, params) -> state_derivative
DynamicsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters],
    list[CasadiMX] | CasadiMX | Sequence[CasadiMX],
]

# (t0, tf, x0, xf, integrals, params) -> objective_value
ObjectiveCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters],
    CasadiMX,
]

# (state, control, time, integral_idx, params) -> integrand_value
IntegralIntegrandCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, int, ProblemParameters],
    CasadiMX,
]

PathConstraintsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters],
    list[PathConstraint] | PathConstraint,
]

EventConstraintsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters],
    list[EventConstraint] | EventConstraint,
]

# --- Helper Type Aliases for Trajectories and Guesses ---
ListOfCasadiMX: TypeAlias = list[CasadiMX]
TrajectoryData: TypeAlias = list[FloatArray]  # List of 1D arrays for each state/control component

# For initial guesses, states/controls are typically a list of 2D matrices,
# one per mesh interval: [num_variables, num_nodes_in_interval]
InitialGuessTrajectory: TypeAlias = list[FloatMatrix]
InitialGuessIntegrals: TypeAlias = float | FloatArray | list[float] | None

# --- Types for Adaptive Mesh Refinement ---
# For evaluator callables
StateEvaluator: TypeAlias = Callable[[float | FloatArray], FloatArray]
ControlEvaluator: TypeAlias = Callable[[float | FloatArray], FloatArray]

# ODE solver related types
DynamicsRHSCallable: TypeAlias = Callable[[float, FloatArray], FloatArray]


# Define a protocol for the ODE solver result
class ODESolverResult(Protocol):
    """Protocol for the result of ODE solvers like solve_ivp."""

    y: FloatMatrix
    t: FloatArray
    success: bool
    message: str


# Make the ODESolverCallable more flexible with optional kwargs
ODESolverCallable: TypeAlias = Callable[..., ODESolverResult]

# Gamma normalization factors type
GammaFactors: TypeAlias = FloatArray

# Type variable for generic functions
T = TypeVar("T")


# Protocol for Problem to avoid circular imports
class ProblemProtocol(Protocol):
    """Protocol defining the expected interface of a Problem object."""

    name: str
    _states: dict[str, dict[str, Any]]
    _controls: dict[str, dict[str, Any]]
    _parameters: ProblemParameters
    _t0_bounds: tuple[float, float]
    _tf_bounds: tuple[float, float]
    _num_integrals: int
    collocation_points_per_interval: list[int]
    global_normalized_mesh_nodes: FloatArray | None
    initial_guess: Any
    solver_options: dict[str, object]

    # Symbolic attributes
    _sym_states: dict[str, SymType]
    _sym_controls: dict[str, SymType]
    _sym_parameters: dict[str, SymType]
    _sym_time: SymType | None
    _sym_time_initial: SymType | None
    _sym_time_final: SymType | None
    _dynamics_expressions: dict[SymType, SymExpr]
    _objective_expression: SymExpr | None
    _objective_type: str | None
    _constraints: list[SymExpr]
    _integral_expressions: list[SymExpr]
    _integral_symbols: list[SymType]

    def get_dynamics_function(self) -> DynamicsCallable: ...
    def get_objective_function(self) -> ObjectiveCallable: ...
    def get_integrand_function(self) -> IntegralIntegrandCallable | None: ...
    def get_path_constraints_function(self) -> PathConstraintsCallable | None: ...
    def get_event_constraints_function(self) -> EventConstraintsCallable | None: ...
