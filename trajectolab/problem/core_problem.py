"""
Core Problem class for optimal control problem definition.
"""

from __future__ import annotations

from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .state import ConstraintState, MeshState, VariableState


class Problem:
    """Main class for defining optimal control problems."""

    def __init__(self, name: str = "Unnamed Problem", use_scaling: bool = True) -> None:
        self.name = name

        # State containers
        self._variable_state = VariableState()
        self._constraint_state = ConstraintState()
        self._mesh_state = MeshState()

        # Initial guess is stored as a mutable container to allow modification by functions
        self._initial_guess_container = [None]

        # Solver options
        self.solver_options: dict[str, Any] = {}

        # Create scaling object ONCE with the desired initial state
        from trajectolab.scaling import Scaling

        self._scaling = Scaling(enabled=use_scaling)
        print(f"Scaling object created with enabled={self._scaling.enabled}")

    # SIMPLIFIED - Direct delegation to _scaling object as single source of truth
    @property
    def use_scaling(self) -> bool:
        """Get the current scaling state directly from the scaling object."""
        return self._scaling.enabled if hasattr(self, "_scaling") else False

    @use_scaling.setter
    def use_scaling(self, value: bool) -> None:
        """Set scaling state directly on the scaling object."""
        if hasattr(self, "_scaling"):
            self._scaling.enabled = value
            print(f"Scaling enabled set to: {self._scaling.enabled}")

    # Property access to state attributes for backward compatibility

    @property
    def _states(self) -> dict[str, dict[str, Any]]:
        return self._variable_state.states

    @property
    def _controls(self) -> dict[str, dict[str, Any]]:
        return self._variable_state.controls

    @property
    def _parameters(self) -> dict[str, Any]:
        return self._variable_state.parameters

    @property
    def _sym_states(self) -> dict[str, SymType]:
        return self._variable_state.sym_states

    @property
    def _sym_controls(self) -> dict[str, SymType]:
        return self._variable_state.sym_controls

    @property
    def _sym_parameters(self) -> dict[str, SymType]:
        return self._variable_state.sym_parameters

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

    # Variable creation methods
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
        return variables_problem.create_state_variable(
            self._variable_state, name, initial, final, lower, upper
        )

    def control(self, name: str, lower: float | None = None, upper: float | None = None) -> SymType:
        return variables_problem.create_control_variable(self._variable_state, name, lower, upper)

    def parameter(self, name: str, value: Any) -> SymType:
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        variables_problem.set_dynamics(self._variable_state, dynamics_dict)

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        return variables_problem.add_integral(self._variable_state, integrand_expr)

    def minimize(self, objective_expr: SymExpr) -> None:
        variables_problem.set_objective(self._variable_state, objective_expr)

    def subject_to(self, constraint_expr: SymExpr) -> None:
        constraints_problem.add_constraint(self._constraint_state, constraint_expr)

    # Mesh management methods
    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None:
        """Configure mesh structure for the problem.

        This method clears any existing initial guess, as mesh changes require
        a new guess that matches the new mesh structure. After setting the mesh,
        call set_initial_guess() to provide a starting point for the solver.

        Note: When using automatic scaling, scaling factors will be computed
        right before solving, using both bounds and any initial guess provided.
        """
        print("\n=== SETTING MESH ===")
        print(f"Polynomial degrees: {polynomial_degrees}")
        print(f"Mesh points: {mesh_points}")

        mesh.configure_mesh(self._mesh_state, polynomial_degrees, mesh_points)
        print("Mesh configured successfully")

        # Clear initial guess when mesh changes
        initial_guess_problem.clear_initial_guess(self._initial_guess_container)
        print("Initial guess cleared")

        # NScaling computation removed from here - will happen at solve time
        print("Scaling will be computed right before solving")

    # Initial guess methods
    def set_initial_guess(self, **kwargs) -> None:
        initial_guess_problem.set_initial_guess(
            self._initial_guess_container, self._mesh_state, self._variable_state, **kwargs
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

    # Solver interface methods
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

    def get_scaling(self) -> object:
        """Get the scaling object."""
        return self._scaling if hasattr(self, "_scaling") else None
