"""
Core problem definition - SIMPLIFIED.
Removed ALL legacy compatibility layers, uses only unified storage.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..tl_types import FloatArray, NumericArrayLike, SymExpr, SymType
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .state import ConstraintState, MeshState, VariableState


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
    """Main class for defining optimal control problems - SIMPLIFIED."""

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
        """Get state bounds in order."""
        return self._variable_state.get_state_bounds()

    def get_control_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get control bounds in order."""
        return self._variable_state.get_control_bounds()

    # ========================================================================
    # VARIABLE CREATION METHODS
    # ========================================================================

    def time(self, initial: float = 0.0, final: float | None = None, free_final: bool = False):
        return variables_problem.create_time_variable(
            self._variable_state, initial, final, free_final
        )

    def state(
        self,
        name: str,
        initial: float | None = None,
        final: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> SymType:
        """Define a state variable."""
        return variables_problem.create_state_variable(
            self._variable_state, name, initial, final, lower, upper
        )

    def control(
        self,
        name: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> SymType:
        """Define a control variable."""
        return variables_problem.create_control_variable(self._variable_state, name, lower, upper)

    def parameter(self, name: str, value: Any) -> SymType:
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """Define system dynamics."""
        variables_problem.set_dynamics(self._variable_state, dynamics_dict)

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
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
        """Configure mesh structure for the problem."""
        print("\n=== SETTING MESH ===")
        print(f"Polynomial degrees: {polynomial_degrees}")
        print(f"Mesh points: {mesh_points}")

        mesh.configure_mesh(self._mesh_state, polynomial_degrees, mesh_points)
        print("Mesh configured successfully")

        # Clear initial guess when mesh changes
        initial_guess_problem.clear_initial_guess(self._initial_guess_container)
        print("Initial guess cleared")

    # ========================================================================
    # INITIAL GUESS METHODS
    # ========================================================================

    def set_initial_guess(
        self,
        states: Sequence[FloatArray] | None = None,
        controls: Sequence[FloatArray] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """Set initial guess for the problem."""
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

    def get_initial_guess_requirements(self):
        return initial_guess_problem.get_initial_guess_requirements(
            self._mesh_state, self._variable_state
        )

    def validate_initial_guess(self) -> None:
        initial_guess_problem.validate_initial_guess(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    def get_solver_input_summary(self):
        return initial_guess_problem.get_solver_input_summary(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    # ========================================================================
    # SOLVER INTERFACE METHODS
    # ========================================================================

    def get_dynamics_function(self):
        return solver_interface.get_dynamics_function(self._variable_state)

    def get_objective_function(self):
        return solver_interface.get_objective_function(self._variable_state)

    def get_integrand_function(self):
        return solver_interface.get_integrand_function(self._variable_state)

    def get_path_constraints_function(self):
        return solver_interface.get_path_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )

    def get_event_constraints_function(self):
        return solver_interface.get_event_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )
