"""
Core problem definition - UNIFIED CONSTRAINT API - FIXED ORDERING DEPENDENCY.
Users can now call set_mesh() and set_initial_guess() in any order.
Validation is deferred until solve time when all information is available.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..tl_types import FloatArray, NumericArrayLike, SymExpr, SymType
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .state import ConstraintInput, ConstraintState, MeshState, VariableState


# Configure problem-specific logger
problem_logger = logging.getLogger("trajectolab.problem")
if not problem_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    problem_logger.addHandler(handler)
    problem_logger.setLevel(logging.INFO)


class Problem:
    """Main class for defining optimal control problems - UNIFIED CONSTRAINT API - FIXED ORDERING."""

    def __init__(self, name: str = "Unnamed Problem") -> None:
        """Initialize a new problem instance."""
        self.name = name
        problem_logger.info(f"Creating problem '{name}'")

        # State containers
        self._variable_state = VariableState()
        self._constraint_state = ConstraintState()
        self._mesh_state = MeshState()

        # Initial guess container
        self._initial_guess_container = [None]

        # Solver options
        self.solver_options: dict[str, Any] = {}

    # ========================================================================
    # UNIFIED PROPERTIES - Direct access to optimized storage
    # ========================================================================

    @property
    def _parameters(self) -> dict[str, Any]:
        return self._variable_state.parameters

    @property
    def _sym_time(self) -> SymType | None:
        return self._variable_state.sym_time

    @property
    def _sym_time_initial(self) -> SymType | None:
        return self._variable_state.sym_time_initial

    @property
    def _sym_time_final(self) -> SymType | None:
        return self._variable_state.sym_time_final

    @property
    def _t0_bounds(self) -> tuple[float, float]:
        return self._variable_state.t0_bounds

    @property
    def _tf_bounds(self) -> tuple[float, float]:
        return self._variable_state.tf_bounds

    @property
    def _dynamics_expressions(self) -> dict[SymType, SymExpr]:
        return self._variable_state.dynamics_expressions

    @property
    def _objective_expression(self) -> SymExpr | None:
        return self._variable_state.objective_expression

    @property
    def _constraints(self) -> list[SymExpr]:
        return self._constraint_state.constraints

    @property
    def _integral_expressions(self) -> list[SymExpr]:
        return self._variable_state.integral_expressions

    @property
    def _integral_symbols(self) -> list[SymType]:
        return self._variable_state.integral_symbols

    @property
    def _num_integrals(self) -> int:
        return self._variable_state.num_integrals

    @property
    def collocation_points_per_interval(self) -> list[int]:
        return self._mesh_state.collocation_points_per_interval

    @property
    def global_normalized_mesh_nodes(self) -> FloatArray | None:
        return self._mesh_state.global_normalized_mesh_nodes

    @property
    def _mesh_configured(self) -> bool:
        return self._mesh_state.configured

    @property
    def initial_guess(self):
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        self._initial_guess_container[0] = value

    # ========================================================================
    # PROTOCOL INTERFACE METHODS - Required by ProblemProtocol
    # ========================================================================

    def get_variable_counts(self) -> tuple[int, int]:
        """Return (num_states, num_controls)."""
        return self._variable_state.get_variable_counts()

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols in order."""
        return self._variable_state.get_ordered_state_symbols()

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols in order."""
        return self._variable_state.get_ordered_control_symbols()

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order."""
        return self._variable_state.get_ordered_state_names()

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order."""
        return self._variable_state.get_ordered_control_names()

    def get_state_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get state bounds in order (compatibility method)."""
        # Convert boundary constraints to bounds for compatibility
        boundary_constraints = self._variable_state.get_state_boundary_constraints()
        bounds = []
        for constraint in boundary_constraints:
            if constraint is None or not constraint.has_constraint():
                bounds.append((None, None))
            elif constraint.equals is not None:
                bounds.append((constraint.equals, constraint.equals))
            else:
                bounds.append((constraint.lower, constraint.upper))
        return bounds

    def get_control_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get control bounds in order (compatibility method)."""
        # Convert boundary constraints to bounds for compatibility
        boundary_constraints = self._variable_state.get_control_boundary_constraints()
        bounds = []
        for constraint in boundary_constraints:
            if constraint is None or not constraint.has_constraint():
                bounds.append((None, None))
            elif constraint.equals is not None:
                bounds.append((constraint.equals, constraint.equals))
            else:
                bounds.append((constraint.lower, constraint.upper))
        return bounds

    # ========================================================================
    # UNIFIED CONSTRAINT API METHODS - New API
    # ========================================================================

    def time(
        self,
        initial: ConstraintInput = 0.0,
        final: ConstraintInput = None,
    ) -> variables_problem.TimeVariableImpl:
        """
        Define time variable with unified constraint specification.

        Args:
            initial: Initial time constraint (default: fixed at 0.0)
                - float/int: Fixed initial time t0 = value
                - tuple(lower, upper): Range constraint lower ≤ t0 ≤ upper
                - None: Treated as fixed at 0.0
            final: Final time constraint (default: free)
                - float/int: Fixed final time tf = value
                - tuple(lower, upper): Range constraint lower ≤ tf ≤ upper
                - None: Fully free (subject to tf > t0 + ε)

        Returns:
            TimeVariableImpl object with .initial and .final properties

        Examples:
            problem.time()  # t0=0 (fixed), tf free
            problem.time(final=10.0)  # t0=0 (fixed), tf=10.0 (fixed)
            problem.time(initial=1.0, final=(5.0, 10.0))  # t0=1.0, tf ∈ [5.0, 10.0]
            problem.time(initial=(0.0, 1.0), final=(10.0, None))  # t0 ∈ [0.0, 1.0], tf ≥ 10.0
        """
        return variables_problem.create_time_variable(self._variable_state, initial, final)

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> SymType:
        """
        Define a state variable with unified constraint specification.

        Args:
            name: Variable name
            initial: Initial condition constraint (event constraint at t0)
                - float/int: Fixed value s(t0) = value
                - tuple(lower, upper): Range constraint lower ≤ s(t0) ≤ upper
                - None: No initial constraint
            final: Final condition constraint (event constraint at tf)
                - float/int: Fixed value s(tf) = value
                - tuple(lower, upper): Range constraint lower ≤ s(tf) ≤ upper
                - None: No final constraint
            boundary: Path constraint (applies throughout trajectory)
                - float/int: Fixed value s(t) = value for all t ∈ [t0, tf]
                - tuple(lower, upper): Range constraint lower ≤ s(t) ≤ upper for all t
                - None: No path constraint

        Returns:
            CasADi symbolic variable

        Examples:
            x = problem.state("x", initial=0.0, boundary=(-10.0, 10.0))
            y = problem.state("y", initial=(-1.0, 1.0), final=0.0)
            z = problem.state("z", final=(None, 100.0), boundary=(0.0, None))
        """
        return variables_problem.create_state_variable(
            self._variable_state, name, initial, final, boundary
        )

    def control(
        self,
        name: str,
        boundary: ConstraintInput = None,  # Keep only boundary (path constraints)
    ) -> SymType:
        """
        Define a control variable with path constraints.

        Args:
            name: Variable name
            boundary: Path constraint (applies throughout trajectory)
                - float/int: Fixed value u(t) = value for all t ∈ [t0, tf]
                - tuple(lower, upper): Range constraint lower ≤ u(t) ≤ upper for all t
                - None: No path constraint (strongly recommended to set bounds)

        Returns:
            CasADi symbolic variable

        Examples:
            throttle = problem.control("throttle", boundary=(0.0, 1.0))
            steer = problem.control("steer", boundary=(-1.0, 1.0))
        """
        return variables_problem.create_control_variable(self._variable_state, name, boundary)

    def parameter(self, name: str, value: Any) -> SymType:
        """Define a parameter variable."""
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """Define system dynamics."""
        variables_problem.set_dynamics(self._variable_state, dynamics_dict)

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        """Add an integral expression."""
        return variables_problem.add_integral(self._variable_state, integrand_expr)

    def minimize(self, objective_expr: SymExpr) -> None:
        """Define the objective function to minimize."""
        variables_problem.set_objective(self._variable_state, objective_expr)

    def subject_to(self, constraint_expr: SymExpr) -> None:
        """Add a constraint to the problem."""
        constraints_problem.add_constraint(self._constraint_state, constraint_expr)

    # ========================================================================
    # MESH MANAGEMENT METHODS
    # ========================================================================

    def set_mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """
        Configure mesh structure for the problem.

        FIXED: Can now be called before or after set_initial_guess().
        """
        print("\n=== SETTING MESH ===")
        print(f"Polynomial degrees: {polynomial_degrees}")
        print(f"Mesh points: {mesh_points}")

        mesh.configure_mesh(self._mesh_state, polynomial_degrees, mesh_points)
        print("Mesh configured successfully")

        # NOTE: We no longer clear the initial guess when mesh changes!
        # This allows users to set initial guess before mesh configuration.
        print("Initial guess preserved (can be set before or after mesh)")

    # ========================================================================
    # INITIAL GUESS METHODS - FIXED ORDERING DEPENDENCY
    # ========================================================================

    def set_initial_guess(
        self,
        states: Sequence[FloatArray] | None = None,
        controls: Sequence[FloatArray] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """
        Set initial guess for the problem.

        FIXED: Can now be called before or after set_mesh().
        Validation is deferred until solve time when all information is available.
        """
        print("\n=== SETTING INITIAL GUESS ===")

        initial_guess_problem.set_initial_guess(
            self._initial_guess_container,
            self._mesh_state,
            self._variable_state,
            states=states,
            controls=controls,
            initial_time=initial_time,
            terminal_time=terminal_time,
            integrals=integrals,
        )

        # Provide helpful feedback
        if self._mesh_configured:
            print("Initial guess set successfully (mesh already configured)")
            # Try to validate now that we have both pieces
            try:
                self.validate_initial_guess()
                print("Initial guess validated successfully")
            except Exception as e:
                print(f"Initial guess validation failed: {e}")
        else:
            print("Initial guess stored successfully (mesh not yet configured)")
            print("Validation will occur when mesh is set or solver runs")

    def can_validate_initial_guess(self) -> bool:
        """Check if we have enough information to validate the initial guess."""
        return initial_guess_problem.can_validate_initial_guess(
            self._mesh_state, self._variable_state
        )

    def get_initial_guess_requirements(self):
        """
        Get initial guess requirements.

        FIXED: Now handles case where mesh isn't configured yet.
        """
        requirements = initial_guess_problem.get_initial_guess_requirements(
            self._mesh_state, self._variable_state
        )

        if not self._mesh_configured:
            print("Note: Mesh must be configured to get specific shape requirements")

        return requirements

    def validate_initial_guess(self) -> None:
        """
        Validate the current initial guess.

        FIXED: Provides clear error message if mesh not configured.
        This is called automatically when the solver runs.
        """
        initial_guess_problem.validate_initial_guess(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    def get_solver_input_summary(self):
        """Get solver input summary."""
        return initial_guess_problem.get_solver_input_summary(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    # ========================================================================
    # SOLVER INTERFACE METHODS
    # ========================================================================

    def get_dynamics_function(self):
        """Get dynamics function for solver."""
        return solver_interface.get_dynamics_function(self._variable_state)

    def get_objective_function(self):
        """Get objective function for solver."""
        return solver_interface.get_objective_function(self._variable_state)

    def get_integrand_function(self):
        """Get integrand function for solver."""
        return solver_interface.get_integrand_function(self._variable_state)

    def get_path_constraints_function(self):
        """Get path constraints function for solver."""
        return solver_interface.get_path_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )

    def get_event_constraints_function(self):
        """Get event constraints function for solver."""
        return solver_interface.get_event_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )
