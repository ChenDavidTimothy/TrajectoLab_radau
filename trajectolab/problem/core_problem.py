"""
Redesigned Problem class implementing proper optimal control scaling.

Key architectural changes:
1. Clean separation between physical and scaled symbols
2. Original expressions never corrupted
3. Scaling applied only at solver interface level
4. Proper objective/constraint scaling following scale.txt
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..scaling.core_scale import AutoScalingManager
from ..tl_types import FloatArray, FloatMatrix, NumericArrayLike, SymExpr, SymType
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
    """
    Main class for defining optimal control problems with proper scaling.

    Architecture:
    - Physical symbols: Original, never corrupted, used in user expressions
    - Scaled symbols: For NLP optimization only
    - Scaling applied at solver interface, not at definition time
    """

    def __init__(self, name: str = "Unnamed Problem", auto_scaling: bool = False) -> None:
        """
        Initialize optimal control problem.

        Args:
            name: Problem name
            auto_scaling: Enable proper optimal control scaling
        """
        self.name = name
        problem_logger.info(f"Creating problem '{name}' with auto_scaling={auto_scaling}")

        # Auto-scaling manager
        self._auto_scaling_enabled = auto_scaling
        self._scaling_manager: AutoScalingManager | None = (
            AutoScalingManager() if auto_scaling else None
        )

        # Problem state containers
        self._variable_state = VariableState()
        self._constraint_state = ConstraintState()
        self._mesh_state = MeshState()

        # Initial guess storage
        self._initial_guess_container = [None]

        # Solver options
        self.solver_options: dict[str, Any] = {}

        # Physical variable names tracking (for scaling)
        self._physical_state_names: list[str] = []
        self._physical_control_names: list[str] = []

    # Property access for state attributes
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

    # Variable creation methods - REDESIGNED
    def time(self, initial: float = 0.0, final: float | None = None, free_final: bool = False):
        """Create time variable (no scaling applied to time)."""
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
        scale_guide_lower: float | None = None,
        scale_guide_upper: float | None = None,
    ) -> SymType:
        """
        Define a state variable with proper scaling architecture.

        Key change: Returns ORIGINAL physical symbol, not scaled expression.
        Scaling handled internally for NLP optimization.
        """
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            # No scaling - standard implementation
            symbol = variables_problem.create_state_variable(
                self._variable_state, name, initial, final, lower, upper
            )
            self._physical_state_names.append(name)
            return symbol

        # With auto-scaling - proper implementation
        print(f"🔧 Creating state '{name}' with proper scaling")

        # Update scaling manager with variable information
        self._scaling_manager.update_variable_range(name, initial, final, lower, upper)

        # Setup variable scaling
        self._scaling_manager.setup_variable_scaling(
            name, lower, upper, scale_guide_lower, scale_guide_upper
        )

        # Get scaled bounds for NLP variable
        physical_bounds = {"initial": initial, "final": final, "lower": lower, "upper": upper}
        scaled_bounds = self._scaling_manager.get_scaled_variable_bounds(name, physical_bounds)

        # Create scaled NLP variable and original physical symbol
        def create_scaled_state(scaled_name: str, **bounds):
            return variables_problem.create_state_variable(
                self._variable_state, scaled_name, **bounds
            )

        # KEY: This returns the ORIGINAL symbol, stores scaled symbol internally
        original_symbol = self._scaling_manager.create_variable_symbols(
            name, create_scaled_state, scaled_bounds
        )

        # Track physical name
        self._physical_state_names.append(name)

        print("  ✅ Created original symbol for user expressions")
        print("  📊 Scaled NLP variable created internally")

        return original_symbol

    def control(
        self,
        name: str,
        lower: float | None = None,
        upper: float | None = None,
        scale_guide_lower: float | None = None,
        scale_guide_upper: float | None = None,
    ) -> SymType:
        """
        Define a control variable with proper scaling architecture.

        Key change: Returns ORIGINAL physical symbol, not scaled expression.
        """
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            # No scaling - standard implementation
            symbol = variables_problem.create_control_variable(
                self._variable_state, name, lower, upper
            )
            self._physical_control_names.append(name)
            return symbol

        # With auto-scaling - proper implementation
        print(f"🔧 Creating control '{name}' with proper scaling")

        # Update scaling manager
        self._scaling_manager.update_variable_range(name, None, None, lower, upper)

        # Setup variable scaling
        self._scaling_manager.setup_variable_scaling(
            name, lower, upper, scale_guide_lower, scale_guide_upper
        )

        # Get scaled bounds
        physical_bounds = {"lower": lower, "upper": upper}
        scaled_bounds = self._scaling_manager.get_scaled_variable_bounds(name, physical_bounds)

        # Create scaled NLP variable and original physical symbol
        def create_scaled_control(scaled_name: str, **bounds):
            return variables_problem.create_control_variable(
                self._variable_state, scaled_name, **bounds
            )

        # KEY: Returns ORIGINAL symbol
        original_symbol = self._scaling_manager.create_variable_symbols(
            name, create_scaled_control, scaled_bounds
        )

        # Track physical name
        self._physical_control_names.append(name)

        print("  ✅ Created original symbol for user expressions")

        return original_symbol

    def parameter(self, name: str, value: Any) -> SymType:
        """Create parameter variable (no scaling for parameters)."""
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    # Expression definition methods - REDESIGNED
    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """
        Define system dynamics with proper scaling handling.

        Key change: Store expressions with ORIGINAL symbols.
        Transformation happens at solver interface level.
        """
        print("\n🎯 DYNAMICS DEFINITION:")
        print(f"  📥 Received dynamics for {len(dynamics_dict)} state variables")

        if not self._auto_scaling_enabled or self._scaling_manager is None:
            print("  ⏭️  Auto-scaling disabled, storing original dynamics")
            variables_problem.set_dynamics(self._variable_state, dynamics_dict)
            return

        print("  ✅ Auto-scaling enabled")
        print("  📝 Storing dynamics with ORIGINAL symbols (no corruption)")
        print("  🔄 Transformation will occur at solver interface level")

        # Store dynamics expressions with original symbols - NO TRANSFORMATION
        variables_problem.set_dynamics(self._variable_state, dynamics_dict)

        # Setup ODE defect scaling (Rule 3: W_f = V_y)
        self._scaling_manager.setup_ode_defect_scaling()

        print("  📐 ODE defect scaling (W_f) configured per Rule 3")

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        """
        Add integral cost term with proper scaling handling.

        Key change: Store integrand with ORIGINAL symbols.
        No corruption of integrand structure.
        """
        print("  📊 Adding integral with ORIGINAL symbols (no corruption)")
        print("  ✅ Integrand structure preserved: ∫ f(x, u) dt")

        # Store integrand with original symbols - NO TRANSFORMATION
        return variables_problem.add_integral(self._variable_state, integrand_expr)

    def minimize(self, objective_expr: SymExpr) -> None:
        """
        Define objective function with proper scaling.

        Key change: Store objective with ORIGINAL symbols.
        Multiplicative scaling (w_0) applied at solver level.
        """
        print("\n📊 OBJECTIVE DEFINITION:")

        if not self._auto_scaling_enabled or self._scaling_manager is None:
            print("  ⏭️  Auto-scaling disabled, storing original objective")
            variables_problem.set_objective(self._variable_state, objective_expr)
            return

        print("  ✅ Auto-scaling enabled")
        print("  📝 Storing objective with ORIGINAL symbols (no substitution)")
        print("  📐 Multiplicative scaling (w_0) will be applied per Rule 5")

        # Store objective with original symbols - NO TRANSFORMATION
        variables_problem.set_objective(self._variable_state, objective_expr)

    def subject_to(self, constraint_expr: SymExpr) -> None:
        """
        Add constraint with proper scaling handling.

        Key change: Store constraint with ORIGINAL symbols.
        Constraint scaling (W_g) applied at solver level.
        """
        print("  📝 Adding constraint with ORIGINAL symbols")

        # Store constraint with original symbols - NO TRANSFORMATION
        constraints_problem.add_constraint(self._constraint_state, constraint_expr)

    # Mesh and initial guess methods
    def set_mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """Configure mesh structure."""
        print("\n=== MESH CONFIGURATION ===")
        print(f"Polynomial degrees: {polynomial_degrees}")
        print(f"Mesh points: {mesh_points}")

        mesh.configure_mesh(self._mesh_state, polynomial_degrees, mesh_points)
        print("✅ Mesh configured successfully")

        # Clear initial guess
        initial_guess_problem.clear_initial_guess(self._initial_guess_container)
        print("🔄 Initial guess cleared (mesh changed)")

        # Setup constraint scaling if auto-scaling enabled
        if self._auto_scaling_enabled and self._scaling_manager:
            self._scaling_manager.setup_path_constraint_scaling()
            print("📐 Path constraint scaling (W_g) configured per Rule 4")

    def set_initial_guess(
        self,
        states: Sequence[FloatMatrix] | None = None,
        controls: Sequence[FloatMatrix] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """Set initial guess with proper scaling support."""
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            # No scaling
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
            return

        print("\n🔧 INITIAL GUESS SCALING:")

        # Scale trajectories for NLP
        scaled_states = None
        scaled_controls = None

        if states is not None:
            print(f"  📊 Scaling {len(states)} state trajectory arrays")
            scaled_states = self._scaling_manager.scale_trajectory_arrays(
                states, self._physical_state_names
            )

        if controls is not None:
            print(f"  📊 Scaling {len(controls)} control trajectory arrays")
            scaled_controls = self._scaling_manager.scale_trajectory_arrays(
                controls, self._physical_control_names
            )

        # Set scaled initial guess
        initial_guess_problem.set_initial_guess(
            self._initial_guess_container,
            self._mesh_state,
            self._variable_state,
            states=scaled_states,
            controls=scaled_controls,
            initial_time=initial_time,
            terminal_time=terminal_time,
            integrals=integrals,
        )

        print("  ✅ Initial guess scaled and set for NLP")

    def get_initial_guess_requirements(self):
        """Get initial guess requirements."""
        return initial_guess_problem.get_initial_guess_requirements(
            self._mesh_state, self._variable_state
        )

    def validate_initial_guess(self) -> None:
        """Validate initial guess."""
        initial_guess_problem.validate_initial_guess(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    def get_solver_input_summary(self):
        """Get solver input summary."""
        return initial_guess_problem.get_solver_input_summary(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    # Solver interface methods - UPDATED for proper scaling
    def get_dynamics_function(self):
        """Get dynamics function with proper scaling transformation."""
        return solver_interface.get_dynamics_function(
            self._variable_state,
            self._scaling_manager,
            self._physical_state_names,
            self._physical_control_names,
        )

    def get_objective_function(self):
        """Get objective function with proper scaling (Rule 5: w_0 * J)."""
        return solver_interface.get_objective_function(
            self._variable_state, self._scaling_manager, self._physical_state_names
        )

    def get_integrand_function(self):
        """Get integrand function with proper scaling (preserve structure)."""
        return solver_interface.get_integrand_function(
            self._variable_state,
            self._scaling_manager,
            self._physical_state_names,
            self._physical_control_names,
        )

    def get_path_constraints_function(self):
        """Get path constraints function with proper scaling (Rule 4: W_g)."""
        return solver_interface.get_path_constraints_function(
            self._constraint_state,
            self._variable_state,
            self._scaling_manager,
            self._physical_state_names,
            self._physical_control_names,
        )

    def get_event_constraints_function(self):
        """Get event constraints function with proper scaling."""
        return solver_interface.get_event_constraints_function(
            self._constraint_state,
            self._variable_state,
            self._scaling_manager,
            self._physical_state_names,
        )

    # Scaling information and summary
    def get_scaling_info(self) -> dict[str, Any]:
        """Get scaling information for solution storage."""
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            return {"auto_scaling_enabled": False}

        return self._scaling_manager.get_scaling_info()

    def print_scaling_summary(self) -> None:
        """Print scaling configuration summary."""
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            print("❌ Auto-scaling is DISABLED")
            return

        self._scaling_manager.print_scaling_summary()

    def compute_objective_scaling_at_solution(self, hessian: FloatArray | None = None) -> None:
        """Compute objective scaling at solution (Rule 5)."""
        if self._scaling_manager:
            self._scaling_manager.compute_objective_scaling(hessian)
            print(
                f"📐 Objective scaling updated: w_0 = {self._scaling_manager.objective_scaling_factor:.3e}"
            )
