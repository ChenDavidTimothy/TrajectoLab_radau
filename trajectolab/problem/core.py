"""
Core Problem class for optimal control problem definition.
"""

from __future__ import annotations

from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType
from .constraints import ConstraintManager
from .initial_guess import InitialGuessManager
from .mesh import MeshManager
from .solver_interface import SolverInterface
from .variables import VariableManager


class Problem:
    """Main class for defining optimal control problems."""

    def __init__(self, name: str = "Unnamed Problem") -> None:
        self.name = name

        # Component managers - use different names to avoid conflicts with properties
        self._variable_manager = VariableManager()
        self._constraint_manager = ConstraintManager()
        self._mesh_manager = MeshManager()
        self._initial_guess_manager = InitialGuessManager()
        self._solver_interface = SolverInterface()

        # Solver options
        self.solver_options: dict[str, Any] = {}

    # Delegate property access to managers
    @property
    def _states(self) -> dict[str, dict[str, Any]]:
        return self._variable_manager.states

    @property
    def _controls(self) -> dict[str, dict[str, Any]]:
        return self._variable_manager.controls

    @property
    def _parameters(self) -> dict[str, Any]:
        return self._variable_manager.parameters

    @property
    def _sym_states(self) -> dict[str, SymType]:
        return self._variable_manager.sym_states

    @property
    def _sym_controls(self) -> dict[str, SymType]:
        return self._variable_manager.sym_controls

    @property
    def _sym_parameters(self) -> dict[str, SymType]:
        return self._variable_manager.sym_parameters

    @property
    def _sym_time(self) -> SymType | None:
        return self._variable_manager.sym_time

    @property
    def _sym_time_initial(self) -> SymType | None:
        return self._variable_manager.sym_time_initial

    @property
    def _sym_time_final(self) -> SymType | None:
        return self._variable_manager.sym_time_final

    @property
    def _t0_bounds(self) -> tuple[float, float]:
        return self._variable_manager.t0_bounds

    @property
    def _tf_bounds(self) -> tuple[float, float]:
        return self._variable_manager.tf_bounds

    @property
    def _dynamics_expressions(self) -> dict[SymType, SymExpr]:
        return self._variable_manager.dynamics_expressions

    @property
    def _objective_expression(self) -> SymExpr | None:
        return self._variable_manager.objective_expression

    @property
    def _constraints(self) -> list[SymExpr]:
        return self._constraint_manager.constraints

    @property
    def _integral_expressions(self) -> list[SymExpr]:
        return self._variable_manager.integral_expressions

    @property
    def _integral_symbols(self) -> list[SymType]:
        return self._variable_manager.integral_symbols

    @property
    def _num_integrals(self) -> int:
        return self._variable_manager.num_integrals

    @property
    def collocation_points_per_interval(self) -> list[int]:
        return self._mesh_manager.collocation_points_per_interval

    @property
    def global_normalized_mesh_nodes(self) -> FloatArray | None:
        return self._mesh_manager.global_normalized_mesh_nodes

    @property
    def _mesh_configured(self) -> bool:
        return self._mesh_manager.configured

    @property
    def initial_guess(self):
        return self._initial_guess_manager.current_guess

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        self._initial_guess_manager.current_guess = value

    # Delegate method calls to appropriate managers
    def time(self, initial: float = 0.0, final: float | None = None, free_final: bool = False):
        return self._variable_manager.create_time_variable(initial, final, free_final)

    def state(
        self,
        name: str,
        initial: float | None = None,
        final: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> SymType:
        return self._variable_manager.create_state_variable(name, initial, final, lower, upper)

    def control(self, name: str, lower: float | None = None, upper: float | None = None) -> SymType:
        return self._variable_manager.create_control_variable(name, lower, upper)

    def parameter(self, name: str, value: Any) -> SymType:
        return self._variable_manager.create_parameter_variable(name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        self._variable_manager.set_dynamics(dynamics_dict)

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        return self._variable_manager.add_integral(integrand_expr)

    def minimize(self, objective_expr: SymExpr) -> None:
        self._variable_manager.set_objective(objective_expr)

    def subject_to(self, constraint_expr: SymExpr) -> None:
        self._constraint_manager.add_constraint(constraint_expr)

    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None:
        self._mesh_manager.configure_mesh(polynomial_degrees, mesh_points)
        # Clear initial guess when mesh changes
        self._initial_guess_manager.clear_guess()

    def set_initial_guess(self, **kwargs) -> None:
        self._initial_guess_manager.set_guess(
            mesh_manager=self._mesh_manager, variable_manager=self._variable_manager, **kwargs
        )

    def get_initial_guess_requirements(self):
        return self._initial_guess_manager.get_requirements(
            self._mesh_manager, self._variable_manager
        )

    def validate_initial_guess(self) -> None:
        self._initial_guess_manager.validate_guess(self._mesh_manager, self._variable_manager)

    def get_solver_input_summary(self):
        return self._initial_guess_manager.get_solver_input_summary(
            self._mesh_manager, self._variable_manager
        )

    # Solver interface methods
    def get_dynamics_function(self):
        return self._solver_interface.get_dynamics_function(self._variable_manager)

    def get_objective_function(self):
        return self._solver_interface.get_objective_function(self._variable_manager)

    def get_integrand_function(self):
        return self._solver_interface.get_integrand_function(self._variable_manager)

    def get_path_constraints_function(self):
        return self._solver_interface.get_path_constraints_function(
            self._variable_manager, self._constraint_manager
        )

    def get_event_constraints_function(self):
        return self._solver_interface.get_event_constraints_function(
            self._variable_manager, self._constraint_manager
        )
