from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

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
        self._scaling_factors: dict[str, dict[str, float]] = {}
        self._physical_to_tilde_map: dict[str, str] = {}
        self._tilde_to_physical_map: dict[str, str] = {}
        self._physical_symbols: dict[str, SymType] = {}
        self._initial_guess_ranges: dict[str, dict[str, float | None]] = {}

        # State containers
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
        if not self._auto_scaling_enabled:
            # Original implementation
            return variables_problem.create_state_variable(
                self._variable_state, name, initial, final, lower, upper
            )

        # Update initial guess range info for this variable
        self._update_initial_guess_range(name, initial, final, lower, upper)

        # Determine scaling factors - prefer scale guides if provided
        explicit_lower = scale_guide_lower if scale_guide_lower is not None else lower
        explicit_upper = scale_guide_upper if scale_guide_upper is not None else upper
        vk, rk = self._determine_scaling_factors(name, explicit_lower, explicit_upper)

        # Transform bounds and initial/final values for the tilde (scaled) variable
        scaled_props = self._get_scaled_props(name, vk, rk, initial, final, lower, upper)

        # Create internal scaled variable
        tilde_name = f"{name}_tilde"
        tilde_symbol = variables_problem.create_state_variable(
            self._variable_state, tilde_name, **scaled_props
        )

        # Store mappings
        self._physical_to_tilde_map[name] = tilde_name
        self._tilde_to_physical_map[tilde_name] = name

        # Create physical view of the variable
        physical_symbol = self._create_physical_symbol(name, tilde_symbol, vk, rk)
        self._physical_symbols[name] = physical_symbol

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
        if not self._auto_scaling_enabled:
            # Original implementation
            return variables_problem.create_control_variable(
                self._variable_state, name, lower, upper
            )

        # Update initial guess range info for this variable
        self._update_initial_guess_range(name, None, None, lower, upper)

        # Determine scaling factors - prefer scale guides if provided
        explicit_lower = scale_guide_lower if scale_guide_lower is not None else lower
        explicit_upper = scale_guide_upper if scale_guide_upper is not None else upper
        vk, rk = self._determine_scaling_factors(name, explicit_lower, explicit_upper)

        # Transform bounds for the tilde (scaled) variable
        scaled_props = self._get_scaled_props(name, vk, rk, None, None, lower, upper)

        # Create internal scaled variable
        tilde_name = f"{name}_tilde"
        tilde_symbol = variables_problem.create_control_variable(
            self._variable_state,
            tilde_name,
            lower=scaled_props.get("lower"),
            upper=scaled_props.get("upper"),
        )

        # Store mappings
        self._physical_to_tilde_map[name] = tilde_name
        self._tilde_to_physical_map[tilde_name] = name

        # Create physical view of the variable
        physical_symbol = self._create_physical_symbol(name, tilde_symbol, vk, rk)
        self._physical_symbols[name] = physical_symbol

        return physical_symbol

    def parameter(self, name: str, value: Any) -> SymType:
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """
        Define system dynamics with auto-scaling support.

        Args:
            dynamics_dict: Dictionary mapping state variables to their derivatives
        """
        if not self._auto_scaling_enabled:
            # Original implementation
            variables_problem.set_dynamics(self._variable_state, dynamics_dict)
            return

        # Auto-scaling implementation
        scaled_dynamics_dict = {}

        for state_sym, rhs_expr in dynamics_dict.items():
            # Find the physical variable name
            physical_name = None
            for name, sym in self._physical_symbols.items():
                if sym is state_sym:
                    physical_name = name
                    break

            if physical_name is None:
                raise ValueError("Physical variable not found in dynamics definition")

            # Get the corresponding tilde symbol
            tilde_name = self._physical_to_tilde_map.get(physical_name)
            if tilde_name is None:
                raise ValueError(f"Tilde variable not found for physical variable {physical_name}")

            tilde_sym = None
            for name, sym in self._variable_state.sym_states.items():
                if name == tilde_name:
                    tilde_sym = sym
                    break

            if tilde_sym is None:
                raise ValueError(f"Tilde symbol not found for {tilde_name}")

            # Apply scaling to the dynamics: dx_tilde/dt = v * dx/dt
            vk = self._scaling_factors[physical_name]["v"]
            scaled_dynamics_dict[tilde_sym] = vk * rhs_expr

        variables_problem.set_dynamics(self._variable_state, scaled_dynamics_dict)

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
        if not self._auto_scaling_enabled:
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
            scaled_states = self._scale_trajectories(states, is_state=True)

        if controls is not None:
            scaled_controls = self._scale_trajectories(controls, is_state=False)

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

    def _scale_trajectories(
        self,
        trajectories: Sequence[FloatMatrix],
        is_state: bool,
    ) -> list[FloatMatrix]:
        """
        Scale trajectories from physical to scaled space.

        Args:
            trajectories: List of trajectory arrays in physical space
            is_state: True if scaling state trajectories, False for controls

        Returns:
            List of trajectory arrays in scaled space
        """
        scaled_trajectories = []

        # Get ordered list of variable names
        if is_state:
            variables = [
                name
                for name, meta in sorted(
                    self._variable_state.states.items(), key=lambda x: x[1]["index"]
                )
                if not name.endswith("_tilde")
            ]
        else:
            variables = [
                name
                for name, meta in sorted(
                    self._variable_state.controls.items(), key=lambda x: x[1]["index"]
                )
                if not name.endswith("_tilde")
            ]

        # Get corresponding physical names
        physical_names = []
        for var in variables:
            if var.endswith("_tilde"):
                physical_name = self._tilde_to_physical_map.get(var)
                if physical_name:
                    physical_names.append(physical_name)
            else:
                # No physical name found - this should never happen with auto_scaling=True
                pass

        for traj_array in trajectories:
            # Create scaled array of same shape
            scaled_array = np.zeros_like(traj_array)

            # Scale each row
            for i, name in enumerate(physical_names):
                if name in self._scaling_factors:
                    vk = self._scaling_factors[name]["v"]
                    rk = self._scaling_factors[name]["r"]
                    scaled_array[i, :] = vk * traj_array[i, :] + rk
                else:
                    # Fallback if scaling factors not found
                    scaled_array[i, :] = traj_array[i, :]

            scaled_trajectories.append(scaled_array)

        return scaled_trajectories

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

    def _determine_scaling_factors(
        self,
        var_name: str,
        explicit_lower: float | None,
        explicit_upper: float | None,
    ) -> tuple[float, float]:
        """
        Determine appropriate scaling factors for a variable.

        Args:
            var_name: Variable name
            explicit_lower: Lower bound for scaling
            explicit_upper: Upper bound for scaling

        Returns:
            Tuple of (vk, rk) scaling factors
        """
        vk = 1.0
        rk = 0.0
        rule_applied = "2.4 (Default)"

        # Rule 2.1.a: Use explicit bounds if provided and not equal
        if (
            explicit_lower is not None
            and explicit_upper is not None
            and not np.isclose(explicit_upper, explicit_lower)
        ):
            vk = 1.0 / (explicit_upper - explicit_lower)
            rk = 0.5 - explicit_upper / (explicit_upper - explicit_lower)
            rule_applied = "2.1.a (Explicit Bounds)"

        # Rule 2.1.b: Use initial guess range if available
        elif var_name in self._initial_guess_ranges:
            guess_min = self._initial_guess_ranges[var_name].get("min")
            guess_max = self._initial_guess_ranges[var_name].get("max")

            if (
                guess_min is not None
                and guess_max is not None
                and not np.isclose(guess_max, guess_min)
            ):
                vk = 1.0 / (guess_max - guess_min)
                rk = 0.5 - guess_max / (guess_max - guess_min)
                rule_applied = "2.1.b (Initial Guess Range)"

        # Store the scaling factors
        self._scaling_factors[var_name] = {"v": vk, "r": rk, "rule": rule_applied}

        return vk, rk

    def _get_scaled_props(
        self,
        var_name: str,
        vk: float,
        rk: float,
        initial: float | None,
        final: float | None,
        lower: float | None,
        upper: float | None,
    ) -> dict[str, float | None]:
        """
        Get scaled properties for variable creation.

        Args:
            var_name: Variable name
            vk: Scaling factor v
            rk: Scaling factor r
            initial: Initial value in physical space
            final: Final value in physical space
            lower: Lower bound in physical space
            upper: Upper bound in physical space

        Returns:
            Dictionary with scaled properties
        """
        scaled_def: dict[str, float | None] = {}

        # Transform initial/final values
        if initial is not None:
            scaled_def["initial"] = vk * initial + rk
        if final is not None:
            scaled_def["final"] = vk * final + rk

        # Transform bounds - for variables with explicit bounds
        # we normalize to [-0.5, 0.5]
        if self._scaling_factors[var_name]["rule"] != "2.4 (Default)":
            scaled_def["lower"] = -0.5
            scaled_def["upper"] = 0.5
        else:
            # For default scaling, we still transform the bounds
            scaled_def["lower"] = vk * lower + rk if lower is not None else None
            scaled_def["upper"] = vk * upper + rk if upper is not None else None

        return scaled_def

    def _create_physical_symbol(
        self,
        name: str,
        tilde_symbol: SymType,
        vk: float,
        rk: float,
    ) -> SymType:
        """
        Create a physical space symbolic variable that maps to the scaled variable.

        Args:
            name: Physical variable name
            tilde_symbol: Scaled symbolic variable
            vk: Scaling factor v
            rk: Scaling factor r

        Returns:
            Physical space symbolic variable
        """
        if np.isclose(vk, 0):
            raise ValueError(
                f"Scaling factor 'v' for {name} is zero, cannot create physical symbol"
            )

        # Create physical view as (tilde - r) / v
        physical_symbol = (tilde_symbol - rk) / vk
        return physical_symbol

    def _update_initial_guess_range(
        self,
        var_name: str,
        initial: float | None,
        final: float | None,
        lower: float | None,
        upper: float | None,
    ) -> None:
        """
        Update initial guess range information for a variable.

        Args:
            var_name: Variable name
            initial: Initial value
            final: Final value
            lower: Lower bound
            upper: Upper bound
        """
        # Create entry if not exists
        if var_name not in self._initial_guess_ranges:
            self._initial_guess_ranges[var_name] = {"min": None, "max": None}

        # Update min/max from initial/final if available
        if initial is not None and final is not None:
            self._initial_guess_ranges[var_name]["min"] = min(initial, final)
            self._initial_guess_ranges[var_name]["max"] = max(initial, final)
        elif initial is not None:
            if self._initial_guess_ranges[var_name]["min"] is None:
                self._initial_guess_ranges[var_name]["min"] = initial
            else:
                self._initial_guess_ranges[var_name]["min"] = min(
                    self._initial_guess_ranges[var_name]["min"], initial
                )

            if self._initial_guess_ranges[var_name]["max"] is None:
                self._initial_guess_ranges[var_name]["max"] = initial
            else:
                self._initial_guess_ranges[var_name]["max"] = max(
                    self._initial_guess_ranges[var_name]["max"], initial
                )
        elif final is not None:
            if self._initial_guess_ranges[var_name]["min"] is None:
                self._initial_guess_ranges[var_name]["min"] = final
            else:
                self._initial_guess_ranges[var_name]["min"] = min(
                    self._initial_guess_ranges[var_name]["min"], final
                )

            if self._initial_guess_ranges[var_name]["max"] is None:
                self._initial_guess_ranges[var_name]["max"] = final
            else:
                self._initial_guess_ranges[var_name]["max"] = max(
                    self._initial_guess_ranges[var_name]["max"], final
                )

        # Also consider bounds if available
        if lower is not None:
            if self._initial_guess_ranges[var_name]["min"] is None:
                self._initial_guess_ranges[var_name]["min"] = lower
            else:
                self._initial_guess_ranges[var_name]["min"] = min(
                    self._initial_guess_ranges[var_name]["min"], lower
                )

        if upper is not None:
            if self._initial_guess_ranges[var_name]["max"] is None:
                self._initial_guess_ranges[var_name]["max"] = upper
            else:
                self._initial_guess_ranges[var_name]["max"] = max(
                    self._initial_guess_ranges[var_name]["max"], upper
                )

    def get_scaling_info(self) -> dict[str, Any]:
        """
        Get scaling information for analysis.

        Returns:
            Dictionary with scaling factors and variable mappings
        """
        if not self._auto_scaling_enabled:
            return {"auto_scaling_enabled": False}

        return {
            "auto_scaling_enabled": True,
            "scaling_factors": self._scaling_factors,
            "physical_to_tilde_map": self._physical_to_tilde_map,
            "tilde_to_physical_map": self._tilde_to_physical_map,
        }
