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


"""
Updated OptimalControlSolution class to support proper scaling.
"""


class OptimalControlSolution:
    """
    Solution to an optimal control problem with proper scaling support.

    Enhanced to support the proper scaling methodology from scale.txt:
    - Variable scaling factors (Rule 2)
    - Objective scaling factors (Rule 5)
    - Constraint scaling factors (Rules 3 & 4)
    - Symbol mappings for proper unscaling
    """

    def __init__(self) -> None:
        # Core solution data
        self.success: bool = False
        self.message: str = "Solver not run yet."
        self.initial_time_variable: float | None = None
        self.terminal_time_variable: float | None = None
        self.objective: float | None = None
        self.integrals: float | FloatArray | None = None

        # Trajectory data
        self.time_states: FloatArray = np.array([], dtype=np.float64)
        self.states: list[FloatArray] = []
        self.time_controls: FloatArray = np.array([], dtype=np.float64)
        self.controls: list[FloatArray] = []

        # Raw solver data
        self.raw_solution: CasadiOptiSol | None = None
        self.opti_object: CasadiOpti | None = None

        # Mesh information
        self.num_collocation_nodes_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: FloatArray | None = None
        self.num_collocation_nodes_list_at_solve_time: list[int] | None = None
        self.global_mesh_nodes_at_solve_time: FloatArray | None = None
        self.solved_state_trajectories_per_interval: list[FloatMatrix] | None = None
        self.solved_control_trajectories_per_interval: list[FloatMatrix] | None = None

        # === PROPER SCALING FIELDS ===

        # Auto-scaling status
        self.auto_scaling_enabled: bool = False

        # Variable scaling factors (Rule 2: ỹ = V_y * y + r_y)
        self.variable_scaling_factors: dict[str, dict[str, float]] = {}
        # Format: {"var_name": {"v": scale_factor, "r": shift_factor, "rule": "2.1.a"}}

        # Objective scaling (Rule 5: w_0 * J)
        self.objective_scaling_factor: float = 1.0
        self.objective_computed_from_hessian: bool = False
        self.gerschgorin_omega: float | None = None

        # Constraint scaling (Rules 3 & 4)
        self.ode_defect_scaling_factors: dict[str, float] = {}  # W_f factors (Rule 3)
        self.path_constraint_scaling_factors: dict[str, float] = {}  # W_g factors (Rule 4)

        # Symbol mappings for proper scaling
        self.original_physical_symbols: dict[
            str, SymType
        ] = {}  # Original symbols (never corrupted)
        self.scaled_nlp_symbols: dict[str, SymType] = {}  # Scaled symbols for NLP
        self.physical_to_scaled_names: dict[str, str] = {}  # Name mappings

    @property
    def final_time(self) -> float | None:
        """Get final time (alias for terminal_time_variable)."""
        return self.terminal_time_variable

    def get_variable_scaling_info(self, var_name: str) -> dict[str, float] | None:
        """Get scaling information for a specific variable."""
        return self.variable_scaling_factors.get(var_name)

    def get_objective_scaling_info(self) -> dict[str, Any]:
        """Get objective scaling information."""
        return {
            "w_0": self.objective_scaling_factor,
            "computed_from_hessian": self.objective_computed_from_hessian,
            "gerschgorin_omega": self.gerschgorin_omega,
        }

    def get_constraint_scaling_info(self) -> dict[str, dict[str, float]]:
        """Get constraint scaling information."""
        return {
            "ode_defect_scaling": self.ode_defect_scaling_factors.copy(),
            "path_constraint_scaling": self.path_constraint_scaling_factors.copy(),
        }

    def print_scaling_summary(self) -> None:
        """Print summary of scaling information used in solution."""
        if not self.auto_scaling_enabled:
            print("❌ No auto-scaling was used for this solution")
            return

        print("\n" + "=" * 60)
        print("📊 SOLUTION SCALING SUMMARY")
        print("=" * 60)

        print("✅ Proper auto-scaling was applied")

        # Variable scaling
        if self.variable_scaling_factors:
            print(f"\n📐 VARIABLE SCALING (Rule 2): {len(self.variable_scaling_factors)} variables")
            print(f"{'Variable':<12} | {'v (scale)':<12} | {'r (shift)':<12} | {'Rule'}")
            print("-" * 60)
            for var_name, factors in sorted(self.variable_scaling_factors.items()):
                v = factors.get("v", 1.0)
                r = factors.get("r", 0.0)
                rule = factors.get("rule", "Unknown")
                print(f"{var_name:<12} | {v:<12.3e} | {r:<12.3f} | {rule}")

        # Objective scaling
        print("\n📊 OBJECTIVE SCALING (Rule 5):")
        print(f"  w_0 = {self.objective_scaling_factor:.3e}")
        if self.objective_computed_from_hessian and self.gerschgorin_omega is not None:
            print(f"  ϖ (Gerschgorin) = {self.gerschgorin_omega:.3e}")
        else:
            print("  Using default scaling")

        # Constraint scaling
        if self.ode_defect_scaling_factors:
            print("\n🔧 ODE DEFECT SCALING (Rule 3: W_f = V_y):")
            for var_name, w_f in self.ode_defect_scaling_factors.items():
                print(f"  {var_name}: W_f = {w_f:.3e}")

        if self.path_constraint_scaling_factors:
            print("\n🔧 PATH CONSTRAINT SCALING (Rule 4):")
            for constraint_name, w_g in self.path_constraint_scaling_factors.items():
                print(f"  {constraint_name}: W_g = {w_g:.3e}")

        print("\n📈 SOLUTION VALUES:")
        print(f"  Raw NLP objective: {(self.objective * self.objective_scaling_factor):.6f}")
        print(f"  Physical objective: {self.objective:.6f}")
        print(f"  Unscaling factor: 1/{self.objective_scaling_factor:.3e}")

        print("=" * 60)


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

    # Auto-scaling related attributes (Added missing attributes)
    _auto_scaling_enabled: bool

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
    def get_scaling_info(self) -> dict[str, Any]: ...  # Added missing method
