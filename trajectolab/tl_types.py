"""
Core type definitions for the TrajectoLab project with unified constraint system.
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
NumericArrayLike: TypeAlias = (
    NDArray[np.floating[Any]]
    | NDArray[np.integer[Any]]
    | Sequence[float]
    | Sequence[int]
    | list[float]
    | list[int]
)
# --- Core Symbolic Type Aliases (Public) ---
SymType: TypeAlias = ca.MX
SymExpr: TypeAlias = ca.MX | float | int

# --- CasADi Type Aliases (Public) ---
CasadiMX: TypeAlias = ca.MX
CasadiDM: TypeAlias = ca.DM
CasadiMatrix: TypeAlias = CasadiMX | CasadiDM
CasadiOpti: TypeAlias = ca.Opti
CasadiOptiSol: TypeAlias = ca.OptiSol
CasadiFunction: TypeAlias = ca.Function
ListOfCasadiMX: TypeAlias = list[CasadiMX]

# --- Problem Structure Data Classes and Type Aliases ---
ProblemParameters: TypeAlias = dict[str, float | int | str]


# --- Unified Constraint System ---
class Constraint:
    """
    Unified constraint class for optimal control problems.

    Supports equality and inequality constraints:
    - Equality: val == equals
    - Lower bound: val >= min_val
    - Upper bound: val <= max_val
    - Box constraint: min_val <= val <= max_val

    Used for both path constraints (applied at collocation points)
    and event constraints (applied at boundaries).
    """

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

        # Validation
        if equals is not None and (min_val is not None or max_val is not None):
            raise ValueError("Cannot specify equality constraint with bound constraints")
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

    def __repr__(self) -> str:
        if self.equals is not None:
            return f"Constraint(val == {self.equals})"

        bounds = []
        if self.min_val is not None:
            bounds.append(f"{self.min_val} <=")
        bounds.append("val")
        if self.max_val is not None:
            bounds.append(f"<= {self.max_val}")

        return f"Constraint({' '.join(bounds)})"


# --- Function Protocol Classes for Time Variable ---
class TimeVariable(Protocol):
    """Protocol for time variable with initial/final properties."""

    @property
    def initial(self) -> SymType: ...

    @property
    def final(self) -> SymType: ...

    def __call__(self) -> SymType: ...


# --- Function Protocol Classes for User-Facing APIs ---
class DynamicsFuncProtocol(Protocol):
    """Protocol for user-defined dynamics functions."""

    def __call__(
        self,
        states: dict[str, float | CasadiMX],
        controls: dict[str, float | CasadiMX],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float | CasadiMX]:
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
        self,
        states: dict[str, float | CasadiMX],
        controls: dict[str, float | CasadiMX],
        *args: Any,
        **kwargs: Any,
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
        self,
        states: dict[str, float | CasadiMX],
        controls: dict[str, float | CasadiMX],
        *args: Any,
        **kwargs: Any,
    ) -> Constraint | list[Constraint]:
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

    def __call__(self, *args: Any, **kwargs: Any) -> Constraint | list[Constraint]:
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


class InitialGuess:
    """
    Initial guess for the optimal control problem.
    All components are optional.
    """

    def __init__(
        self,
        initial_time_variable: float | None = None,
        terminal_time_variable: float | None = None,
        states: list[FloatMatrix] | None = None,
        controls: list[FloatMatrix] | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        self.initial_time_variable = initial_time_variable
        self.terminal_time_variable = terminal_time_variable
        self.states = states
        self.controls = controls
        self.integrals = integrals


class OptimalControlSolution:
    """
    Solution to an optimal control problem.
    """

    def __init__(self) -> None:
        self.success: bool = False
        self.message: str = "Solver not run yet."
        self.initial_time_variable: float | None = None
        self.terminal_time_variable: float | None = None
        self.objective: float | None = None
        self.integrals: float | FloatArray | None = None
        self.time_states: FloatArray = np.array([], dtype=np.float64)
        self.states: list[FloatArray] = []
        self.time_controls: FloatArray = np.array([], dtype=np.float64)
        self.controls: list[FloatArray] = []
        self.raw_solution: CasadiOptiSol | None = None
        self.opti_object: CasadiOpti | None = None
        self.num_collocation_nodes_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: FloatArray | None = None
        self.num_collocation_nodes_list_at_solve_time: list[int] | None = None
        self.global_mesh_nodes_at_solve_time: FloatArray | None = None
        self.solved_state_trajectories_per_interval: list[FloatMatrix] | None = None
        self.solved_control_trajectories_per_interval: list[FloatMatrix] | None = None


# User-facing function types using Protocol classes
DynamicsFuncType: TypeAlias = DynamicsFuncProtocol
ObjectiveFuncType: TypeAlias = ObjectiveFuncProtocol
IntegrandFuncType: TypeAlias = IntegrandFuncProtocol
ConstraintFuncType: TypeAlias = ConstraintFuncProtocol
EventConstraintFuncType: TypeAlias = EventConstraintFuncProtocol

# --- Callable Types for Solver (Internal) ---
DynamicsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters],
    list[CasadiMX] | CasadiMX | Sequence[CasadiMX],
]

ObjectiveCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters],
    CasadiMX,
]

IntegralIntegrandCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, int, ProblemParameters],
    CasadiMX,
]

PathConstraintsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters],
    list[Constraint] | Constraint,
]

EventConstraintsCallable: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters],
    list[Constraint] | Constraint,
]

# --- Helper Type Aliases for Trajectories and Guesses ---
TrajectoryData: TypeAlias = list[FloatArray]

# For initial guesses, states/controls are typically a list of 2D matrices
InitialGuessTrajectory: TypeAlias = list[FloatMatrix]
InitialGuessIntegrals: TypeAlias = float | FloatArray | list[float] | None

# --- Types for Adaptive Mesh Refinement ---
StateEvaluator: TypeAlias = Callable[[float | FloatArray], FloatArray]
ControlEvaluator: TypeAlias = Callable[[float | FloatArray], FloatArray]

# ODE solver related types
DynamicsRHSCallable: TypeAlias = Callable[[float, FloatArray], FloatArray]


class ODESolverResult(Protocol):
    """Protocol for the result of ODE solvers like solve_ivp."""

    y: FloatMatrix
    t: FloatArray
    success: bool
    message: str


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

    _mesh_configured: bool

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

    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None: ...
    def validate_initial_guess(self) -> None: ...
    def get_dynamics_function(self) -> DynamicsCallable: ...
    def get_objective_function(self) -> ObjectiveCallable: ...
    def get_integrand_function(self) -> IntegralIntegrandCallable | None: ...
    def get_path_constraints_function(self) -> PathConstraintsCallable | None: ...
    def get_event_constraints_function(self) -> EventConstraintsCallable | None: ...
