"""
Core type definitions for TrajectoLab.

This module contains all type aliases and protocols used throughout the library,
with a focus on scientific computing patterns and trajectory optimization types.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import casadi as ca  # Added import for CasADi types
import numpy as np
from numpy.typing import NDArray

# ---- Numeric Array Types ----
_FloatArray: TypeAlias = NDArray[np.float64]
_IntArray: TypeAlias = NDArray[np.int32]
_BoolArray: TypeAlias = NDArray[np.bool_]

# ---- Mathematical Shapes ----
_Vector: TypeAlias = NDArray[np.float64]  # 1D array
_Matrix: TypeAlias = NDArray[np.float64]  # 2D array

# ---- Array-Like Input Types ----
_ArrayLike: TypeAlias = Sequence[float] | _FloatArray
_IntArrayLike: TypeAlias = Sequence[int] | _IntArray

# ---- Time and Mesh Types ----
_TimePoint: TypeAlias = float
_NormalizedTimePoint: TypeAlias = float  # tau ∈ [-1, 1]
_MeshPoints: TypeAlias = _FloatArray
_MeshNodesList: TypeAlias = list[float] | np.ndarray  # Moved from direct_solver.py

# ---- Result Types ----
_ErrorTuple: TypeAlias = tuple[float, float, float]  # max_error, mean_error, rms_error

# ---- Optimization Problem Types ----
_StateDict: TypeAlias = dict[str, float | _FloatArray]
_ControlDict: TypeAlias = dict[str, float | _FloatArray]
_ParamDict: TypeAlias = dict[str, Any]  # Parameters can be any type

# ---- CasADi-specific Types ---- (Moved from direct_solver.py)
_CasadiScalar: TypeAlias = ca.MX | ca.SX | ca.DM
_CasadiVector: TypeAlias = ca.MX | ca.SX | ca.DM
_CasadiMatrix: TypeAlias = ca.MX | ca.SX | ca.DM
_CasadiSolution: TypeAlias = ca.OptiSol  # Type for CasADi's solver solution object

# ---- Solver-Specific and Trajectory Types ---- (Moved from direct_solver.py)
_SolverOptionsDict: TypeAlias = dict[str, Any]
_CollocationPointsList: TypeAlias = list[int]
_StateTrajectory: TypeAlias = list[np.ndarray]  # List of state vectors over time
_ControlTrajectory: TypeAlias = list[np.ndarray]  # List of control vectors over time


# ---- Constraint Data Structures ---- (Moved from direct_solver.py)
@dataclass
class PathConstraint:
    """Represents a constraint along the trajectory."""

    val: _CasadiScalar  # The symbolic value of the constraint
    min_val: float | None = None  # Lower bound for the constraint
    max_val: float | None = None  # Upper bound for the constraint
    equals: float | None = None  # Exact value the constraint must take


@dataclass
class EventConstraint:
    """Represents a constraint at endpoints or specific events."""

    val: _CasadiScalar  # The symbolic value of the constraint
    min_val: float | None = None  # Lower bound for the constraint
    max_val: float | None = None  # Upper bound for the constraint
    equals: float | None = None  # Exact value the constraint must take


# ---- Initial Guess Data Structures ---- (Moved from direct_solver.py)
@dataclass
class InitialGuess:
    """Initial guess for the solver's decision variables."""

    initial_time_variable: float | None = None
    terminal_time_variable: float | None = None
    states: list[np.ndarray] | None = field(default_factory=list)  # Guess for state trajectory
    controls: list[np.ndarray] | None = field(default_factory=list)  # Guess for control trajectory
    integrals: float | Sequence[float] | None = None  # Guess for integral values


@dataclass
class DefaultGuessValues:
    """Default values for solver initial guesses if not otherwise specified."""

    state: float = 0.0
    control: float = 0.0
    integral: float = 0.0


# ---- Optimal Control Problem Definition ---- (Moved from direct_solver.py)
@dataclass
class OptimalControlProblem:
    """Defines an optimal control problem for the direct solver."""

    num_states: int
    num_controls: int
    dynamics_function: Callable[
        [_CasadiVector, _CasadiVector, float, _ParamDict], list[float] | _CasadiVector
    ]
    objective_function: Callable[
        [float, float, _CasadiVector, _CasadiVector, _CasadiScalar | None, _ParamDict], float
    ]
    t0_bounds: list[float]  # [min_t0, max_t0]
    tf_bounds: list[float]  # [min_tf, max_tf]
    num_integrals: int = 0
    collocation_points_per_interval: _CollocationPointsList | None = None
    global_normalized_mesh_nodes: _MeshNodesList | None = None
    integral_integrand_function: (
        Callable[[_CasadiVector, _CasadiVector, float, int, _ParamDict], float] | None
    ) = None
    path_constraints_function: (
        Callable[
            [_CasadiVector, _CasadiVector, float, _ParamDict],
            list[PathConstraint] | PathConstraint | None,
        ]
        | None
    ) = None
    event_constraints_function: (
        Callable[
            [
                float,
                float,
                _CasadiVector,
                _CasadiVector,
                _CasadiScalar | Sequence[float] | None,
                _ParamDict,
            ],
            list[EventConstraint] | EventConstraint | None,
        ]
        | None
    ) = None
    problem_parameters: _ParamDict = field(default_factory=dict)
    initial_guess: InitialGuess | None = None
    default_initial_guess_values: DefaultGuessValues = field(default_factory=DefaultGuessValues)
    solver_options: _SolverOptionsDict = field(default_factory=dict)


# ---- Optimal Control Solution Structure ---- (Moved from direct_solver.py)
@dataclass
class OptimalControlSolution:  # Changed to dataclass for consistency and default init
    """Solution container for an optimal control problem."""

    success: bool = False
    message: str = "Solver not run yet."
    initial_time_variable: float | None = None
    terminal_time_variable: float | None = None
    objective: float | None = None
    integrals: float | _FloatArray | None = None
    time_states: _FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    states: list[_FloatArray] = field(default_factory=list)
    time_controls: _FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    controls: list[_FloatArray] = field(default_factory=list)
    raw_solution: _CasadiSolution | None = None  # Raw solution object from CasADi
    opti_object: ca.Opti | None = None  # CasADi Opti stack object
    num_collocation_nodes_per_interval: _CollocationPointsList = field(default_factory=list)
    global_normalized_mesh_nodes: _MeshNodesList | None = None
    # Additional attributes for solver run details
    num_collocation_nodes_list_at_solve_time: _CollocationPointsList | None = None
    global_mesh_nodes_at_solve_time: _MeshNodesList | None = None
    solved_state_trajectories_per_interval: _StateTrajectory | None = None
    solved_control_trajectories_per_interval: _ControlTrajectory | None = None


# ---- Collocation-Specific Types (Original to this file) ----
_CollocationNodes: TypeAlias = _Vector  # Nodes used for enforcing dynamics
_StateNodes: TypeAlias = _Vector  # Nodes used for state approximation
_QuadratureWeights: TypeAlias = _Vector  # Weights for numerical integration
_DifferentiationMatrix: TypeAlias = _Matrix  # Maps state values to derivatives
_BarycentricWeights: TypeAlias = _Vector  # Weights for barycentric interpolation


# ---- Protocol Classes for Callback Functions (Original to this file) ----
class DynamicsFunction(Protocol):
    """Protocol for system dynamics functions."""

    def __call__(
        self, states: _StateDict, controls: _ControlDict, time: float, params: _ParamDict
    ) -> _StateDict: ...


class ObjectiveFunction(Protocol):
    """Protocol for objective functions."""

    def __call__(
        self,
        t0: float,
        tf: float,
        x0: _StateDict,
        xf: _StateDict,
        integrals: Sequence[float] | float | None,
        params: _ParamDict,
    ) -> float: ...


class IntegrandFunction(Protocol):
    """Protocol for integrand functions used in cost integration."""

    def __call__(
        self, states: _StateDict, controls: _ControlDict, time: float, params: _ParamDict
    ) -> float: ...


# ---- Constraint Class and Related Types (Original to this file - generic Constraint) ----
# Note: PathConstraint and EventConstraint (now above) are more specific for the solver
# This generic Constraint might be for other uses or a base.
class Constraint:
    """Represents a generic constraint with bounds or equality condition."""

    def __init__(
        self,
        val: float | None = None,  # This is a float, unlike Path/EventConstraint's _CasadiScalar
        lower: float | None = None,
        upper: float | None = None,
        equals: float | None = None,
    ):
        self.val = val
        self.lower = lower
        self.upper = upper
        self.equals = equals

        if equals is not None:
            self.lower = equals
            self.upper = equals


# ---- Collocation Data Structures (Original to this file) ----
@dataclass
class RadauNodesAndWeights:
    """Basic nodes and weights for Radau collocation."""

    state_approximation_nodes: _StateNodes
    collocation_nodes: _CollocationNodes
    quadrature_weights: _QuadratureWeights


@dataclass
class RadauBasisComponents:
    """Complete set of components for Radau collocation."""

    state_approximation_nodes: _StateNodes
    collocation_nodes: _CollocationNodes
    quadrature_weights: _QuadratureWeights
    differentiation_matrix: _DifferentiationMatrix
    barycentric_weights_for_state_nodes: _BarycentricWeights
    lagrange_at_tau_plus_one: _Vector


# ---- Constraint Function Types (Original to this file) ----
_ConstraintFunc: TypeAlias = Callable[
    [_StateDict, _ControlDict, float, _ParamDict], "Constraint | list[Constraint]"
]

_EventConstraintFunc: TypeAlias = Callable[
    [float, float, _StateDict, _StateDict, float | Sequence[float] | None, _ParamDict],
    "Constraint | list[Constraint]",
]

# ---- Constants (Original to this file) ----
ZERO_TOLERANCE: float = 1e-12  # Tolerance for floating point comparisons
