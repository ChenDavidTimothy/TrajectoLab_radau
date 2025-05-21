from __future__ import annotations

import logging
from typing import Any

from ..scaling import ScalingManager
from ..tl_types import FloatArray, SymExpr, SymType
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .constraints_problem import Constraint
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
    """Main class for defining optimal control problems."""

    def __init__(self, name: str = "Unnamed Problem", auto_scaling: bool = True) -> None:
        """
        Initialize a new problem instance.

        Args:
            name: Name of the problem
            auto_scaling: Whether to enable automatic scaling
        """
        self.name = name
        problem_logger.info(f"Creating problem '{name}'")

        # State containers
        self._variable_state = VariableState()
        self._constraint_state = ConstraintState()
        self._mesh_state = MeshState()

        # Initialize scaling manager
        self._scaling_manager = ScalingManager(enabled=auto_scaling)

        # Initial guess is stored as a mutable container to allow modification by functions
        self._initial_guess_container = [None]

        # Solver options
        self.solver_options: dict[str, Any] = {}

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
        time_var = variables_problem.create_time_variable(
            self._variable_state, initial, final, free_final
        )

        # Register time bounds with scaling manager
        if free_final and final is not None:
            self._scaling_manager.register_time_bounds(initial, final)
        elif free_final:
            # Rough estimate for time bounds with free final time
            self._scaling_manager.register_time_bounds(initial, initial + 1000.0)
        elif final is not None:
            self._scaling_manager.register_time_bounds(initial, final)

        return time_var

    def state(
        self,
        name: str,
        initial: float | None = None,
        final: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> SymType:
        sym_var = variables_problem.create_state_variable(
            self._variable_state, name, initial, final, lower, upper
        )
        # Register with scaling manager - use actual physical bounds
        self._scaling_manager.register_state(name, sym_var, lower, upper)
        return sym_var

    def control(self, name: str, lower: float | None = None, upper: float | None = None) -> SymType:
        sym_var = variables_problem.create_control_variable(
            self._variable_state, name, lower, upper
        )
        # Register with scaling manager - use actual physical bounds
        self._scaling_manager.register_control(name, sym_var, lower, upper)
        return sym_var

    def parameter(self, name: str, value: Any) -> SymType:
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """
        Set dynamics expressions with automatic scaling.

        Args:
            dynamics_dict: Mapping from state symbols to dynamics expressions
        """
        if self._scaling_manager.enabled:
            # Apply Rule 3 for ODE defect scaling
            scaled_dict = {}
            for state_sym, expr in dynamics_dict.items():
                # Scale each dynamics expression using Rule 3
                scaled_expr = self._scaling_manager.scale_dynamics(state_sym, expr)
                scaled_dict[state_sym] = scaled_expr

            variables_problem.set_dynamics(self._variable_state, scaled_dict)
        else:
            variables_problem.set_dynamics(self._variable_state, dynamics_dict)

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        return variables_problem.add_integral(self._variable_state, integrand_expr)

    def minimize(self, objective_expr: SymExpr) -> None:
        """
        Set the objective function to minimize.

        Args:
            objective_expr: Objective function expression to minimize
        """
        # We don't scale the objective expression directly
        # Scaling will happen at the numerical level during optimization
        variables_problem.set_objective(self._variable_state, objective_expr)

    def subject_to(self, constraint_expr: SymExpr | Constraint) -> None:
        """
        Add a constraint with automatic scaling.

        Args:
            constraint_expr: Constraint expression or Constraint object (physical units)
        """
        # Convert SymExpr to Constraint if needed
        if not isinstance(constraint_expr, Constraint):
            constraint = Constraint(val=constraint_expr, equals=0.0)
        else:
            constraint = constraint_expr

        # Apply scaling if enabled
        if self._scaling_manager.enabled:
            scaled_constraint = self._scaling_manager.scale_constraint(constraint)
            constraints_problem.add_constraint(self._constraint_state, scaled_constraint)
        else:
            # If scaling disabled, use constraint as-is
            constraints_problem.add_constraint(self._constraint_state, constraint)

    # Mesh management methods
    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None:
        """Configure mesh structure for the problem.

        This method clears any existing initial guess, as mesh changes require
        a new guess that matches the new mesh structure. After setting the mesh,
        call set_initial_guess() to provide a starting point for the solver.
        """
        print("\n=== SETTING MESH ===")
        print(f"Polynomial degrees: {polynomial_degrees}")
        print(f"Mesh points: {mesh_points}")

        mesh.configure_mesh(self._mesh_state, polynomial_degrees, mesh_points)
        print("Mesh configured successfully")

        # Clear initial guess when mesh changes
        initial_guess_problem.clear_initial_guess(self._initial_guess_container)
        print("Initial guess cleared")

    # Initial guess methods
    def set_initial_guess(self, **kwargs) -> None:
        """
        Set initial guess with automatic scaling.

        Args:
            **kwargs: Initial guess components (states, controls, times, etc.)
        """
        # First, create the initial guess with physical values
        initial_guess_problem.set_initial_guess(
            self._initial_guess_container, self._mesh_state, self._variable_state, **kwargs
        )

        # Scale the initial guess if scaling is enabled
        if self._scaling_manager.enabled and self._initial_guess_container[0] is not None:
            original_guess = self._initial_guess_container[0]
            scaled_guess = self._scaling_manager.scale_initial_guess(original_guess)
            self._initial_guess_container[0] = scaled_guess

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
