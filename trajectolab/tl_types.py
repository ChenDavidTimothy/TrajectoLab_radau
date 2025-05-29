"""
Core type definitions for the TrajectoLab optimal control framework with multi-phase support.

This module defines all type aliases, protocols, and data structures for both single-phase
and multi-phase optimal control problems, faithfully implementing the CGPOPS mathematical
structure for multi-phase problems.
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

    # Essential solver properties
    _num_integrals: int
    _parameters: ProblemParameters
    initial_guess: Any
    solver_options: dict[str, object]

    # Mesh properties
    _mesh_configured: bool
    collocation_points_per_interval: list[int]
    global_normalized_mesh_nodes: FloatArray

    # Time bounds
    _t0_bounds: tuple[float, float]
    _tf_bounds: tuple[float, float]

    # Expression storage
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


class MultiPhaseProblemProtocol(Protocol):
    """
    Protocol defining the expected interface of a multi-phase Problem object for solver.

    Implements the general multiple-phase optimal control problem structure from CGPOPS
    Section 2, with phases linked through event constraints and shared static parameters.
    """

    # Multi-phase structure - faithful to CGPOPS Section 2
    phases: list[ProblemProtocol]
    global_parameters: dict[str, float]
    inter_phase_constraints: list[ca.MX]
    global_objective_expression: ca.MX | None
    solver_options: dict[str, object]

    # Multi-phase methods
    def get_phase_count(self) -> int:
        """Get total number of phases P"""
        ...

    def get_phase_endpoint_vectors(self) -> list[ca.MX]:
        """
        Get endpoint vectors E^(p) for all phases.

        Each endpoint vector E^(p) = [Y_1^(p), t_0^(p), Y_{N^(p)+1}^(p), t_f^(p), Q^(p)]
        as defined in CGPOPS Equation (15).
        """
        ...

    def get_inter_phase_constraints_function(self) -> Callable[..., list[Constraint]] | None:
        """
        Get inter-phase event constraints function.

        Returns function that evaluates event constraints linking phases:
        b_min ≤ b(E^(1), ..., E^(P), s) ≤ b_max
        as defined in CGPOPS Equation (14).
        """
        ...

    def get_global_objective_function(self) -> Callable[..., ca.MX]:
        """
        Get global objective function over all phase endpoints.

        Returns function that evaluates:
        J = φ(E^(1), ..., E^(P), s)
        as defined in CGPOPS Equation (17).
        """
        ...

    def validate_multi_phase_structure(self) -> None:
        """Validate multi-phase problem structure and phase consistency"""
        ...


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


# --- SINGLE-PHASE DATA CONTAINERS ---
class InitialGuess:
    """Initial guess for single-phase optimal control problem."""

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
    """Solution to single-phase optimal control problem."""

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


# --- MULTI-PHASE DATA CONTAINERS ---
class MultiPhaseInitialGuess:
    """
    Initial guess for multi-phase optimal control problem.

    Faithful to CGPOPS multi-phase structure where each phase has independent
    initial guess specification, plus global parameters shared across phases.
    """

    def __init__(
        self,
        phase_initial_guesses: list[InitialGuess] | None = None,
        global_parameters: dict[str, float] | None = None,
    ) -> None:
        self.phase_initial_guesses = phase_initial_guesses or []
        self.global_parameters = global_parameters or {}


class PhaseEndpointVector:
    """
    Phase endpoint vector E^(p) as defined in CGPOPS Equation (15).

    E^(p) = [Y_1^(p), t_0^(p), Y_{N^(p)+1}^(p), t_f^(p), Q^(p)]

    Contains initial state, initial time, final state, final time, and integrals
    for phase p, used in event constraints and global objective function.
    """

    def __init__(
        self,
        phase_index: int,
        initial_state: FloatArray | None = None,
        initial_time: float | None = None,
        final_state: FloatArray | None = None,
        final_time: float | None = None,
        integrals: FloatArray | None = None,
    ) -> None:
        self.phase_index = phase_index
        self.initial_state = initial_state
        self.initial_time = initial_time
        self.final_state = final_state
        self.final_time = final_time
        self.integrals = integrals

    def to_vector(self) -> FloatArray:
        """Convert endpoint data to vector format for constraint evaluation."""
        components = []

        if self.initial_state is not None:
            components.append(self.initial_state.flatten())
        if self.initial_time is not None:
            components.append(np.array([self.initial_time]))
        if self.final_state is not None:
            components.append(self.final_state.flatten())
        if self.final_time is not None:
            components.append(np.array([self.final_time]))
        if self.integrals is not None:
            components.append(self.integrals.flatten())

        return np.concatenate(components) if components else np.array([])


class MultiPhaseOptimalControlSolution:
    """
    Solution to multi-phase optimal control problem.

    Faithful implementation of CGPOPS multi-phase solution structure,
    containing phase-specific solutions plus global optimization results.
    """

    def __init__(self) -> None:
        # Overall solution status
        self.success: bool = False
        self.message: str = "Multi-phase solver not run yet."
        self.objective: float | None = None

        # Phase-specific solutions - each phase has independent solution
        self.phase_solutions: list[OptimalControlSolution] = []
        self.phase_count: int = 0

        # Global static parameters s - shared across all phases (CGPOPS Equation 6)
        self.global_parameters: dict[str, float] = {}

        # Phase endpoint vectors E^(p) - used in event constraints (CGPOPS Equation 15)
        self.phase_endpoints: list[PhaseEndpointVector] = []

        # Inter-phase constraint information
        self.inter_phase_constraint_violations: FloatArray | None = None
        self.inter_phase_constraint_multipliers: FloatArray | None = None
        self.max_inter_phase_constraint_violation: float | None = None

        # Phase transition analysis
        self.phase_continuity_errors: dict[tuple[int, int], FloatArray] = {}
        self.phase_jump_magnitudes: dict[tuple[int, int], float] = {}

        # Solver metadata
        self.raw_solution: ca.OptiSol | None = None
        self.opti_object: ca.Opti | None = None
        self.solve_time: float | None = None
        self.nlp_iterations: int | None = None

        # Multi-phase mesh information
        self.phase_mesh_configurations: list[dict[str, Any]] = []
        self.total_collocation_points: int = 0

    def get_phase_solution(self, phase_index: int) -> OptimalControlSolution:
        """Get solution for specific phase with bounds checking."""
        if not (0 <= phase_index < len(self.phase_solutions)):
            raise IndexError(
                f"Phase index {phase_index} out of range [0, {len(self.phase_solutions)})"
            )
        return self.phase_solutions[phase_index]

    def get_phase_endpoint(self, phase_index: int) -> PhaseEndpointVector:
        """Get endpoint vector for specific phase with bounds checking."""
        if not (0 <= phase_index < len(self.phase_endpoints)):
            raise IndexError(
                f"Phase index {phase_index} out of range [0, {len(self.phase_endpoints)})"
            )
        return self.phase_endpoints[phase_index]

    def analyze_phase_continuity(self) -> dict[str, Any]:
        """
        Analyze continuity between phases.

        Returns comprehensive analysis of state continuity, time continuity,
        and constraint violations at phase boundaries.
        """
        analysis = {
            "continuous_phases": [],
            "discontinuous_phases": [],
            "max_state_discontinuity": 0.0,
            "max_time_gap": 0.0,
            "phase_transition_summary": {},
        }

        for i in range(len(self.phase_solutions) - 1):
            phase_i = self.phase_solutions[i]
            phase_j = self.phase_solutions[i + 1]

            # Analyze state continuity
            if (
                phase_i.states
                and phase_j.states
                and len(phase_i.states) > 0
                and len(phase_j.states) > 0
            ):
                final_state_i = np.array([state[-1] for state in phase_i.states])
                initial_state_j = np.array([state[0] for state in phase_j.states])
                state_discontinuity = np.linalg.norm(final_state_i - initial_state_j)

                analysis["max_state_discontinuity"] = max(
                    analysis["max_state_discontinuity"], state_discontinuity
                )

                # Analyze time continuity
                time_gap = 0.0
                if (
                    phase_i.terminal_time_variable is not None
                    and phase_j.initial_time_variable is not None
                ):
                    time_gap = abs(phase_j.initial_time_variable - phase_i.terminal_time_variable)
                    analysis["max_time_gap"] = max(analysis["max_time_gap"], time_gap)

                # Classify phase transition
                transition_key = (i, i + 1)
                analysis["phase_transition_summary"][transition_key] = {
                    "state_discontinuity": state_discontinuity,
                    "time_gap": time_gap,
                    "is_continuous": state_discontinuity < 1e-6 and time_gap < 1e-6,
                }

                if state_discontinuity < 1e-6 and time_gap < 1e-6:
                    analysis["continuous_phases"].append(transition_key)
                else:
                    analysis["discontinuous_phases"].append(transition_key)

        return analysis


# --- MULTI-PHASE TYPE ALIASES ---
MultiPhaseEndpointVectors: TypeAlias = list[PhaseEndpointVector]
"""Type alias for collection of phase endpoint vectors E^(1), ..., E^(P)"""

InterPhaseConstraintFunction: TypeAlias = Callable[
    [MultiPhaseEndpointVectors, dict[str, float]], list[Constraint]
]
"""Type alias for inter-phase event constraint function b(E^(1), ..., E^(P), s)"""

GlobalObjectiveFunction: TypeAlias = Callable[[MultiPhaseEndpointVectors, dict[str, float]], ca.MX]
"""Type alias for global objective function φ(E^(1), ..., E^(P), s)"""
