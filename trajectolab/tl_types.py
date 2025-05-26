"""
Core type definitions for the TrajectoLab optimal control framework - SIMPLIFIED.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias

import casadi as ca
import numpy as np
from numpy.typing import NDArray


# --- NUMERICAL SAFETY TYPES (Non-negotiable) ---
FloatArray: TypeAlias = NDArray[np.float64]  # Critical for numerical precision
NumericArrayLike: TypeAlias = (
    NDArray[np.floating[Any]]
    | NDArray[np.integer[Any]]
    | Sequence[float]
    | Sequence[int]
    | list[float]
    | list[int]
)

# --- USER API TYPES (High value) ---
ConstraintInput: TypeAlias = float | int | tuple[float | int | None, float | int | None] | None
"""
Type alias for unified constraint specification.

Supported input types:
- float/int: Equality constraint (variable = value)
- tuple(lower, upper): Range constraint with None for unbounded sides
- None: No constraint specified
"""

ProblemParameters: TypeAlias = dict[str, float | int | str]


# --- EXTERNAL INTERFACE PROTOCOLS (Required) ---
class ODESolverResult(Protocol):
    """Protocol for the result of ODE solvers like solve_ivp."""

    y: FloatArray
    t: FloatArray
    success: bool
    message: str


ODESolverCallable: TypeAlias = Callable[..., ODESolverResult]


class ProblemProtocol(Protocol):
    """Protocol defining the expected interface of a Problem object for solver."""

    # Essential solver properties (RESTORED - these are actually needed)
    _num_integrals: int
    _parameters: ProblemParameters
    initial_guess: Any
    solver_options: dict[str, object]

    # Mesh properties (RESTORED - solver needs these)
    _mesh_configured: bool
    collocation_points_per_interval: list[int]
    global_normalized_mesh_nodes: FloatArray

    # Time bounds (RESTORED - solver needs these)
    _t0_bounds: tuple[float, float]
    _tf_bounds: tuple[float, float]

    # Expression storage (RESTORED - validation needs these)
    _dynamics_expressions: dict[ca.MX, ca.MX]
    _objective_expression: ca.MX | None

    # Essential solver methods
    def get_variable_counts(self) -> tuple[int, int]:
        """Return (num_states, num_controls)"""
        ...

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order"""
        ...

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order"""
        ...

    def get_dynamics_function(self) -> Callable[..., list[ca.MX]]:
        """Get dynamics function for solver"""
        ...

    def get_objective_function(self) -> Callable[..., ca.MX]:
        """Get objective function for solver"""
        ...

    def get_integrand_function(self) -> Callable[..., ca.MX] | None:
        """Get integrand function for solver"""
        ...

    def get_path_constraints_function(self) -> Callable[..., list[Constraint]] | None:
        """Get path constraints function for solver"""
        ...

    def get_event_constraints_function(self) -> Callable[..., list[Constraint]] | None:
        """Get event constraints function for solver"""
        ...

    def validate_initial_guess(self) -> None:
        """Validate the current initial guess"""
        ...

    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None: ...


# --- UNIFIED CONSTRAINT SYSTEM ---
class Constraint:
    """Unified constraint class for optimal control problems."""

    def __init__(
        self,
        val: ca.MX | float,
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


# --- DATA CONTAINERS ---
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
        self.raw_solution: ca.OptiSol | None = None
        self.opti_object: ca.Opti | None = None
        self.num_collocation_nodes_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: FloatArray | None = None
        self.num_collocation_nodes_list_at_solve_time: list[int] | None = None
        self.global_mesh_nodes_at_solve_time: FloatArray | None = None
        self.solved_state_trajectories_per_interval: list[FloatArray] | None = None
        self.solved_control_trajectories_per_interval: list[FloatArray] | None = None


# --- TIME VARIABLE PROTOCOL ---
class TimeVariable(Protocol):
    """Protocol for time variable with initial/final properties."""

    @property
    def initial(self) -> ca.MX: ...

    @property
    def final(self) -> ca.MX: ...

    def __call__(self) -> ca.MX: ...
