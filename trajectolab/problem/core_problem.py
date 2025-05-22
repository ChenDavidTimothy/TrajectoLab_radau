from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..scaling import AutoScalingManager
from ..tl_types import FloatArray, FloatMatrix, SymExpr, SymType
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
    """Main class for defining optimal control problems."""

    def __init__(self, name: str = "Unnamed Problem", auto_scaling: bool = False) -> None:
        """
        Initialize a new problem instance.

        Args:
            name: Name of the problem
            auto_scaling: Whether to enable automatic variable scaling
        """
        self.name = name
        problem_logger.info(f"Creating problem '{name}'")

        # Auto-scaling support
        self._auto_scaling_enabled = auto_scaling
        self._scaling_manager: AutoScalingManager | None = (
            AutoScalingManager() if auto_scaling else None
        )

        # State containers (unchanged)
        self._variable_state = VariableState()
        self._constraint_state = ConstraintState()
        self._mesh_state = MeshState()

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
        Define a state variable with optional scaling.

        Args:
            name: Variable name
            initial: Initial value constraint
            final: Final value constraint
            lower: Lower bound
            upper: Upper bound
            scale_guide_lower: Optional lower bound used for scaling (overrides automatic scaling)
            scale_guide_upper: Optional upper bound used for scaling (overrides automatic scaling)

        Returns:
            Symbolic variable (in physical space if auto-scaling is enabled)
        """
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            # Original implementation
            return variables_problem.create_state_variable(
                self._variable_state, name, initial, final, lower, upper
            )

        # Update initial guess range info for this variable
        self._scaling_manager.update_initial_guess_range(name, initial, final, lower, upper)

        # Set up scaling for this variable
        scaling_factors = self._scaling_manager.setup_variable_scaling(
            name, lower, upper, scale_guide_lower, scale_guide_upper
        )

        # Get scaled properties
        scaled_props = self._scaling_manager.get_scaled_properties(
            name, initial, final, lower, upper
        )

        # Create internal scaled variable
        scaled_name = f"{name}_scaled"
        scaled_symbol = variables_problem.create_state_variable(
            self._variable_state, scaled_name, **scaled_props
        )

        # Create physical symbol
        physical_symbol = self._scaling_manager.create_physical_symbol(name, scaled_symbol)

        # Store mappings
        self._scaling_manager.variable_mappings.add_mapping(name, scaled_name, physical_symbol)

        return physical_symbol

    def control(
        self,
        name: str,
        lower: float | None = None,
        upper: float | None = None,
        scale_guide_lower: float | None = None,
        scale_guide_upper: float | None = None,
    ) -> SymType:
        """
        Define a control variable with optional scaling.

        Args:
            name: Variable name
            lower: Lower bound
            upper: Upper bound
            scale_guide_lower: Optional lower bound used for scaling (overrides automatic scaling)
            scale_guide_upper: Optional upper bound used for scaling (overrides automatic scaling)

        Returns:
            Symbolic variable (in physical space if auto-scaling is enabled)
        """
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            # Original implementation
            return variables_problem.create_control_variable(
                self._variable_state, name, lower, upper
            )

        # Update initial guess range info for this variable
        self._scaling_manager.update_initial_guess_range(name, None, None, lower, upper)

        # Set up scaling for this variable
        scaling_factors = self._scaling_manager.setup_variable_scaling(
            name, lower, upper, scale_guide_lower, scale_guide_upper
        )

        # Get scaled properties
        scaled_props = self._scaling_manager.get_scaled_properties(name, None, None, lower, upper)

        # Create internal scaled variable
        scaled_name = f"{name}_scaled"
        scaled_symbol = variables_problem.create_control_variable(
            self._variable_state,
            scaled_name,
            lower=scaled_props.get("lower"),
            upper=scaled_props.get("upper"),
        )

        # Create physical symbol
        physical_symbol = self._scaling_manager.create_physical_symbol(name, scaled_symbol)

        # Store mappings
        self._scaling_manager.variable_mappings.add_mapping(name, scaled_name, physical_symbol)

        return physical_symbol

    def parameter(self, name: str, value: Any) -> SymType:
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """
        Define system dynamics with auto-scaling support and comprehensive logging.
        """
        print("\n🎯 DYNAMICS SCALING ANALYSIS:")
        print(f"  📥 Received dynamics for {len(dynamics_dict)} state variables")

        if not self._auto_scaling_enabled or self._scaling_manager is None:
            print("  ⏭️  Auto-scaling disabled, using original dynamics")
            variables_problem.set_dynamics(self._variable_state, dynamics_dict)
            return

        print("  🔄 Auto-scaling enabled, transforming dynamics...")

        # Get scaled symbols for transformation
        scaled_symbols = {}
        for name in self._scaling_manager.variable_mappings.physical_to_scaled:
            scaled_name = self._scaling_manager.variable_mappings.physical_to_scaled[name]
            for sym_name, sym in self._variable_state.sym_states.items():
                if sym_name == scaled_name:
                    scaled_symbols[name] = sym
                    break

        # Transform dynamics using scaling manager
        scaled_dynamics = self._scaling_manager.transform_dynamics(
            dynamics_dict,
            self._scaling_manager.variable_mappings.physical_symbols,
            scaled_symbols,
        )

        print(f"  ✅ Successfully transformed {len(scaled_dynamics)} dynamics equations")
        variables_problem.set_dynamics(self._variable_state, scaled_dynamics)

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        return variables_problem.add_integral(self._variable_state, integrand_expr)

    def minimize(self, objective_expr: SymExpr) -> None:
        """
        Define the objective function to minimize.

        Args:
            objective_expr: Expression to minimize
        """
        if not self._auto_scaling_enabled:
            # Original implementation
            variables_problem.set_objective(self._variable_state, objective_expr)
            return

        # For auto-scaling, we pass the objective directly
        # since expressions using physical variables are automatically
        # converted to expressions using scaled variables
        variables_problem.set_objective(self._variable_state, objective_expr)

    def subject_to(self, constraint_expr: SymExpr) -> None:
        """
        Add a constraint to the problem.

        Args:
            constraint_expr: Constraint expression
        """
        if not self._auto_scaling_enabled:
            # Original implementation
            constraints_problem.add_constraint(self._constraint_state, constraint_expr)
            return

        # For auto-scaling, we pass the constraint directly
        # since expressions using physical variables are automatically
        # converted to expressions using scaled variables
        constraints_problem.add_constraint(self._constraint_state, constraint_expr)

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
    def set_initial_guess(
        self,
        states: Sequence[FloatMatrix] | None = None,
        controls: Sequence[FloatMatrix] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """
        Set initial guess with auto-scaling support.

        Args:
            states: State trajectories in physical space
            controls: Control trajectories in physical space
            initial_time: Initial time
            terminal_time: Terminal time
            integrals: Integral values
        """
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            # Original implementation
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

        # For auto-scaling, convert physical initial guess to scaled space
        scaled_states = None
        scaled_controls = None

        if states is not None:
            state_names = self._get_ordered_physical_names(is_state=True)
            scaled_states = self._scaling_manager.scale_trajectories(states, state_names)

        if controls is not None:
            control_names = self._get_ordered_physical_names(is_state=False)
            scaled_controls = self._scaling_manager.scale_trajectories(controls, control_names)

        # Time and integrals remain unchanged
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

    def _get_ordered_physical_names(self, is_state: bool) -> list[str]:
        """Get ordered list of physical variable names."""
        if is_state:
            state_items = sorted(self._variable_state.states.items(), key=lambda x: x[1]["index"])
            scaled_names = [name for name, _ in state_items if name.endswith("_scaled")]
        else:
            control_items = sorted(
                self._variable_state.controls.items(), key=lambda x: x[1]["index"]
            )
            scaled_names = [name for name, _ in control_items if name.endswith("_scaled")]

        # Convert scaled names to physical names
        physical_names = []
        if self._scaling_manager is not None:
            for scaled_name in scaled_names:
                physical_name = self._scaling_manager.variable_mappings.scaled_to_physical.get(
                    scaled_name
                )
                if physical_name:
                    physical_names.append(physical_name)

        return physical_names

    def get_scaling_info(self) -> dict[str, Any]:
        """
        Get scaling information for analysis.

        Returns:
            Dictionary with scaling factors and variable mappings
        """
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            return {"auto_scaling_enabled": False}

        return self._scaling_manager.get_scaling_info_for_solution()

    def print_scaling_summary(self) -> None:
        """Print comprehensive scaling configuration summary."""
        if not self._auto_scaling_enabled or self._scaling_manager is None:
            print("❌ Auto-scaling is DISABLED")
            return

        self._scaling_manager.print_scaling_summary()
