"""
Core type definitions and protocols for the TrajectoLab optimal control framework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import casadi as ca
import numpy as np
from numpy import int_ as np_int_
from numpy.typing import NDArray


# --- Core Numerical Types (Consolidated) ---
FloatArray: TypeAlias = NDArray[np.float64]  # Single array type for all float arrays
IntArray: TypeAlias = NDArray[np_int_]
NumericArrayLike: TypeAlias = (
    NDArray[np.floating[Any]]
    | NDArray[np.integer[Any]]
    | Sequence[float]
    | Sequence[int]
    | list[float]
    | list[int]
)

# --- Core Symbolic Types ---
SymExpr: TypeAlias = ca.MX | float | int

# --- CasADi Types (Simplified) ---
CasadiMX: TypeAlias = ca.MX
CasadiDM: TypeAlias = ca.DM
CasadiOpti: TypeAlias = ca.Opti
CasadiOptiSol: TypeAlias = ca.OptiSol
CasadiFunction: TypeAlias = ca.Function
ListOfCasadiMX: TypeAlias = list[CasadiMX]

# --- Problem Structure ---
ProblemParameters: TypeAlias = dict[str, float | int | str]

# --- Unified Constraint API Types ---
ConstraintInput: TypeAlias = float | int | tuple[float | int | None, float | int | None] | None
"""
Type alias for unified constraint specification.

Supported input types:
- float/int: Equality constraint (variable = value)
- tuple(lower, upper): Range constraint with None for unbounded sides
- None: No constraint specified
"""


# --- Unified Constraint System ---
class Constraint:
    """Unified constraint class for optimal control problems."""

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


# --- Time Variable Protocol ---
class TimeVariable(Protocol):
    """Protocol for time variable with initial/final properties."""

    @property
    def initial(self) -> CasadiMX: ...

    @property
    def final(self) -> CasadiMX: ...

    def __call__(self) -> CasadiMX: ...


# --- Initial Guess Classes ---
class InitialGuess:
    """Initial guess for the optimal control problem."""

    def __init__(
        self,
        initial_time_variable: float | None = None,
        terminal_time_variable: float | None = None,
        states: list[FloatArray] | None = None,
        controls: list[FloatArray] | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        self.initial_time_variable = initial_time_variable
        self.terminal_time_variable = terminal_time_variable
        self.states = states
        self.controls = controls
        self.integrals = integrals


class OptimalControlSolution:
    """Solution to an optimal control problem."""

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
        self.solved_state_trajectories_per_interval: list[FloatArray] | None = None
        self.solved_control_trajectories_per_interval: list[FloatArray] | None = None


# --- Solver Callable Types ---
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

# --- Adaptive Algorithm Types ---
StateEvaluator: TypeAlias = Callable[[float | FloatArray], FloatArray]
ControlEvaluator: TypeAlias = Callable[[float | FloatArray], FloatArray]
DynamicsRHSCallable: TypeAlias = Callable[[float, FloatArray], FloatArray]
GammaFactors: TypeAlias = FloatArray


# --- ODE Solver Types ---
class ODESolverResult(Protocol):
    """Protocol for the result of ODE solvers like solve_ivp."""

    y: FloatArray
    t: FloatArray
    success: bool
    message: str


ODESolverCallable: TypeAlias = Callable[..., ODESolverResult]

# --- Type Variable ---
T = TypeVar("T")


# --- Problem Protocol (Updated for Unified API) ---
class ProblemProtocol(Protocol):
    """Protocol defining the expected interface of a Problem object."""

    name: str
    _num_integrals: int
    collocation_points_per_interval: list[int]
    global_normalized_mesh_nodes: FloatArray | None
    initial_guess: Any
    solver_options: dict[str, object]
    _mesh_configured: bool

    # Time bounds (compatibility)
    _t0_bounds: tuple[float, float]
    _tf_bounds: tuple[float, float]

    # Symbolic variables
    _sym_time: CasadiMX | None
    _sym_time_initial: CasadiMX | None
    _sym_time_final: CasadiMX | None
    _dynamics_expressions: dict[CasadiMX, SymExpr]
    _objective_expression: SymExpr | None
    _constraints: list[SymExpr]
    _integral_expressions: list[SymExpr]
    _integral_symbols: list[CasadiMX]
    _parameters: ProblemParameters

    # Methods that return variable info
    def get_variable_counts(self) -> tuple[int, int]:
        """Return (num_states, num_controls)"""
        ...

    def get_ordered_state_symbols(self) -> list[CasadiMX]:
        """Get state symbols in order"""
        ...

    def get_ordered_control_symbols(self) -> list[CasadiMX]:
        """Get control symbols in order"""
        ...

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order"""
        ...

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order"""
        ...

    def get_state_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get state bounds in order (compatibility)"""
        ...

    def get_control_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get control bounds in order (compatibility)"""
        ...

    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None: ...
    def validate_initial_guess(self) -> None: ...
    def get_dynamics_function(self) -> DynamicsCallable: ...
    def get_objective_function(self) -> ObjectiveCallable: ...
    def get_integrand_function(self) -> IntegralIntegrandCallable | None: ...
    def get_path_constraints_function(self) -> PathConstraintsCallable | None: ...
    def get_event_constraints_function(self) -> EventConstraintsCallable | None: ...
