from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import casadi as ca
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
        self._physical_to_scaled_map: dict[str, str] = {}
        self._scaled_to_physical_map: dict[str, str] = {}
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

        # Transform bounds and initial/final values for the scaled (scaled) variable
        scaled_props = self._get_scaled_props(name, vk, rk, initial, final, lower, upper)

        # Create internal scaled variable
        scaled_name = f"{name}_scaled"
        scaled_symbol = variables_problem.create_state_variable(
            self._variable_state, scaled_name, **scaled_props
        )

        # Store mappings
        self._physical_to_scaled_map[name] = scaled_name
        self._scaled_to_physical_map[scaled_name] = name

        # Create physical view of the variable
        physical_symbol = self._create_physical_symbol(name, scaled_symbol, vk, rk)
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

        # Transform bounds for the scaled (scaled) variable
        scaled_props = self._get_scaled_props(name, vk, rk, None, None, lower, upper)

        # Create internal scaled variable
        scaled_name = f"{name}_scaled"
        scaled_symbol = variables_problem.create_control_variable(
            self._variable_state,
            scaled_name,
            lower=scaled_props.get("lower"),
            upper=scaled_props.get("upper"),
        )

        # Store mappings
        self._physical_to_scaled_map[name] = scaled_name
        self._scaled_to_physical_map[scaled_name] = name

        # Create physical view of the variable
        physical_symbol = self._create_physical_symbol(name, scaled_symbol, vk, rk)
        self._physical_symbols[name] = physical_symbol

        return physical_symbol

    def parameter(self, name: str, value: Any) -> SymType:
        return variables_problem.create_parameter_variable(self._variable_state, name, value)

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """
        Define system dynamics with auto-scaling support and comprehensive logging.
        """
        print("\n🎯 DYNAMICS SCALING ANALYSIS:")
        print(f"  📥 Received dynamics for {len(dynamics_dict)} state variables")

        if not self._auto_scaling_enabled:
            print("  ⏭️  Auto-scaling disabled, using original dynamics")
            variables_problem.set_dynamics(self._variable_state, dynamics_dict)
            return

        print("  🔄 Auto-scaling enabled, transforming dynamics...")
        scaled_dynamics_dict = {}

        for state_sym, rhs_expr in dynamics_dict.items():
            # Find the physical variable name
            physical_name = None
            for name, sym in self._physical_symbols.items():
                if ca.is_equal(sym, state_sym):  # Use CasADi equality check
                    physical_name = name
                    break

            if physical_name is None:
                print("  🚨 ERROR: Physical variable not found for symbolic state")
                raise ValueError("Physical variable not found in dynamics definition")

            print(f"\n  📊 Processing state '{physical_name}':")

            # Get the corresponding scaled symbol
            scaled_name = self._physical_to_scaled_map.get(physical_name)
            if scaled_name is None:
                print("    🚨 ERROR: No scaled mapping found")
                raise ValueError(f"scaled variable not found for physical variable {physical_name}")

            print(f"    🔗 Physical '{physical_name}' → scaled '{scaled_name}'")

            scaled_sym = None
            for name, sym in self._variable_state.sym_states.items():
                if name == scaled_name:
                    scaled_sym = sym
                    break

            if scaled_sym is None:
                print("    🚨 ERROR: scaled symbol not found")
                raise ValueError(f"scaled symbol not found for {scaled_name}")

            # Get scaling factor and apply to dynamics
            if physical_name not in self._scaling_factors:
                print("    🚨 ERROR: No scaling factors found")
                raise ValueError(f"Scaling factors not found for {physical_name}")

            vk = self._scaling_factors[physical_name]["v"]
            scaling_rule = self._scaling_factors[physical_name]["rule"]

            print(f"    📐 Scaling factor: vk = {vk:.6e}")
            print(f"    📋 Scaling rule: {scaling_rule}")
            print(f"    🔄 Transformation: d(scaled)/dt = {vk:.6e} * d(physical)/dt")
            print("    ✅ Rule 3 compliance: W_f = V_y (ODE defect scaling = state scaling)")

            # Apply scaling to the dynamics: dx_scaled/dt = v * dx/dt
            scaled_rhs = vk * rhs_expr
            scaled_dynamics_dict[scaled_sym] = scaled_rhs

            print("    ✔️  Successfully scaled dynamics equation")

        print("\n  ✅ All dynamics successfully scaled and stored")
        print(f"  📤 Passing {len(scaled_dynamics_dict)} scaled dynamics to solver")

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
        Scale trajectories from physical to scaled space with comprehensive logging.
        """
        trajectory_type = "STATE" if is_state else "CONTROL"
        print(f"\n🎯 {trajectory_type} TRAJECTORY SCALING ANALYSIS:")
        print(f"  📥 Received {len(trajectories)} trajectory arrays")

        if not trajectories:
            print("  ⏭️  No trajectories to scale")
            return []

        scaled_trajectories = []

        # Get ordered list of scaled variable names
        if is_state:
            scaled_variables = [
                name
                for name, meta in sorted(
                    self._variable_state.states.items(), key=lambda x: x[1]["index"]
                )
                if name.endswith("_scaled")
            ]
        else:
            scaled_variables = [
                name
                for name, meta in sorted(
                    self._variable_state.controls.items(), key=lambda x: x[1]["index"]
                )
                if name.endswith("_scaled")
            ]

        print(f"  📋 Found {len(scaled_variables)} scaled variables: {scaled_variables}")

        # Get corresponding physical names in the correct order
        physical_names = []
        for scaled_var in scaled_variables:
            physical_name = self._scaled_to_physical_map.get(scaled_var)
            if physical_name is None:
                print(f"  🚨 CRITICAL ERROR: No physical mapping for '{scaled_var}'")
                raise ValueError(f"Physical name not found for scaled variable '{scaled_var}'")
            physical_names.append(physical_name)

        # Safety validation: Check that we have the expected number of variables
        expected_num_vars = len(scaled_variables)
        if len(physical_names) != expected_num_vars:
            raise ValueError(
                f"CRITICAL ERROR: Variable count mismatch for {'states' if is_state else 'controls'}: "
                f"expected {expected_num_vars}, got {len(physical_names)}. "
                f"Auto-scaling configuration is inconsistent."
            )

        # Additional safety check: Ensure we have at least one variable when auto-scaling is enabled
        if expected_num_vars == 0:
            raise ValueError(
                f"CRITICAL ERROR: No {'state' if is_state else 'control'} variables found "
                f"for auto-scaling. This should not happen when auto_scaling=True."
            )

        print("  🔗 Variable mapping:")
        for i, (phys, scaled) in enumerate(zip(physical_names, scaled_variables, strict=False)):
            print(f"    [{i}] {phys} ↔ {scaled}")

        # Process each trajectory array
        for traj_idx, traj_array in enumerate(trajectories):
            print(f"\n  📊 Processing trajectory array {traj_idx}:")
            print(f"    📏 Shape: {traj_array.shape}")
            print(f"    📈 Value range: [{np.min(traj_array):.6e}, {np.max(traj_array):.6e}]")

            # Validate trajectory array
            if traj_array.shape[0] != len(physical_names):
                print("    🚨 SHAPE MISMATCH ERROR")
                raise ValueError(f"Expected {len(physical_names)} rows, got {traj_array.shape[0]}")

            if not np.all(np.isfinite(traj_array)):
                print("    🚨 NON-FINITE VALUES ERROR")
                raise ValueError("Trajectory contains non-finite values")

            # Create scaled array
            scaled_array = np.zeros_like(traj_array, dtype=np.float64)

            # Scale each row
            for i, physical_name in enumerate(physical_names):
                scaling_info = self._scaling_factors[physical_name]
                vk = scaling_info["v"]
                rk = scaling_info["r"]
                rule = scaling_info["rule"]

                print(f"\n    🔄 Row {i} - Variable '{physical_name}':")
                print(f"      📐 Scaling: vk={vk:.6e}, rk={rk:.6e}")
                print(f"      📋 Rule: {rule}")

                # Show original values
                orig_row = traj_array[i, :]
                orig_min, orig_max = np.min(orig_row), np.max(orig_row)
                print(f"      📥 Physical range: [{orig_min:.6e}, {orig_max:.6e}]")

                # Apply scaling: scaled = v * physical + r
                scaled_row = vk * orig_row + rk
                scaled_min, scaled_max = np.min(scaled_row), np.max(scaled_row)
                print(f"      📤 Scaled range: [{scaled_min:.6e}, {scaled_max:.6e}]")

                # Check if scaled values are reasonable
                if rule != "2.4 (Default)":
                    if not (-0.6 <= scaled_min <= 0.6 and -0.6 <= scaled_max <= 0.6):
                        print("      ⚠️  WARNING: Scaled values outside expected [-0.5, 0.5] range!")

                # Validate finite results
                if not np.all(np.isfinite(scaled_row)):
                    print("      🚨 SCALING PRODUCED NON-FINITE VALUES")
                    raise ValueError(f"Scaling failed for {physical_name}")

                scaled_array[i, :] = scaled_row
                print(f"      ✅ Successfully scaled {orig_row.shape[0]} points")

            print(f"    ✔️  Trajectory {traj_idx} fully scaled")
            print(
                f"    📤 Final scaled range: [{np.min(scaled_array):.6e}, {np.max(scaled_array):.6e}]"
            )
            scaled_trajectories.append(scaled_array)

        print(f"\n  ✅ ALL {trajectory_type} TRAJECTORIES SUCCESSFULLY SCALED")
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
        Determine appropriate scaling factors for a variable with comprehensive logging.
        """
        print(f"\n🔍 SCALING ANALYSIS for variable '{var_name}':")
        print(f"  📊 Input bounds: lower={explicit_lower}, upper={explicit_upper}")

        # Check initial guess ranges
        guess_info = self._initial_guess_ranges.get(var_name, {})
        guess_min = guess_info.get("min")
        guess_max = guess_info.get("max")
        print(f"  📈 Initial guess range: min={guess_min}, max={guess_max}")

        vk = 1.0
        rk = 0.0
        rule_applied = "2.4 (Default)"

        # Rule 2.1.a: Use explicit bounds if provided and not equal
        if (
            explicit_lower is not None
            and explicit_upper is not None
            and not np.isclose(explicit_upper, explicit_lower)
        ):
            range_val = explicit_upper - explicit_lower
            vk = 1.0 / range_val
            rk = 0.5 - explicit_upper / range_val
            rule_applied = "2.1.a (Explicit Bounds)"

            print("  ✅ Applied Rule 2.1.a (Explicit Bounds)")
            print(f"     Range = {explicit_upper} - {explicit_lower} = {range_val}")
            print(f"     vk = 1/{range_val} = {vk}")
            print(f"     rk = 0.5 - {explicit_upper}/{range_val} = {rk}")

            # Verify transformation: [lower, upper] → [-0.5, 0.5]
            scaled_lower = vk * explicit_lower + rk
            scaled_upper = vk * explicit_upper + rk
            print("     🔄 Transformation check:")
            print(
                f"       Physical [{explicit_lower}, {explicit_upper}] → Scaled [{scaled_lower:.6f}, {scaled_upper:.6f}]"
            )

            if not (
                np.isclose(scaled_lower, -0.5, atol=1e-10)
                and np.isclose(scaled_upper, 0.5, atol=1e-10)
            ):
                print("     ⚠️  WARNING: Scaling transformation doesn't map to [-0.5, 0.5] exactly!")

        # Rule 2.1.b: Use initial guess range if available
        elif (
            guess_min is not None and guess_max is not None and not np.isclose(guess_max, guess_min)
        ):
            range_val = guess_max - guess_min
            vk = 1.0 / range_val
            rk = 0.5 - guess_max / range_val
            rule_applied = "2.1.b (Initial Guess Range)"

            print("  ✅ Applied Rule 2.1.b (Initial Guess Range)")
            print(f"     Range = {guess_max} - {guess_min} = {range_val}")
            print(f"     vk = 1/{range_val} = {vk}")
            print(f"     rk = 0.5 - {guess_max}/{range_val} = {rk}")

            # Verify transformation
            scaled_min = vk * guess_min + rk
            scaled_max = vk * guess_max + rk
            print("     🔄 Transformation check:")
            print(
                f"       Physical [{guess_min}, {guess_max}] → Scaled [{scaled_min:.6f}, {scaled_max:.6f}]"
            )

        else:
            print("  ✅ Applied Rule 2.4 (Default): vk=1.0, rk=0.0")
            print("     Reason: No valid bounds or guess range available")

        # Safety checks
        if not np.isfinite(vk) or not np.isfinite(rk):
            print("  🚨 CRITICAL ERROR: Non-finite scaling factors!")
            raise ValueError(f"Non-finite scaling factors for {var_name}: vk={vk}, rk={rk}")

        if np.abs(vk) < 1e-15:
            print(f"  🚨 CRITICAL ERROR: Scaling factor vk={vk} is too small!")
            raise ValueError(f"Scaling factor too small for {var_name}")

        if np.abs(vk) > 1e15:
            print(f"  🚨 CRITICAL ERROR: Scaling factor vk={vk} is too large!")
            raise ValueError(f"Scaling factor too large for {var_name}")

        # Store the scaling factors
        self._scaling_factors[var_name] = {"v": vk, "r": rk, "rule": rule_applied}

        print(f"  📝 Final scaling factors: vk={vk:.6e}, rk={rk:.6e}")
        print(f"  📋 Rule applied: {rule_applied}")

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
        scaled_symbol: SymType,
        vk: float,
        rk: float,
    ) -> SymType:
        """
        Create a physical space symbolic variable that maps to the scaled variable.

        Args:
            name: Physical variable name
            scaled_symbol: Scaled symbolic variable
            vk: Scaling factor v
            rk: Scaling factor r

        Returns:
            Physical space symbolic variable
        """
        if np.isclose(vk, 0):
            raise ValueError(
                f"Scaling factor 'v' for {name} is zero, cannot create physical symbol"
            )

        # Create physical view as (scaled - r) / v
        physical_symbol = (scaled_symbol - rk) / vk
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
            "physical_to_scaled_map": self._physical_to_scaled_map,
            "scaled_to_physical_map": self._scaled_to_physical_map,
        }

    def print_scaling_summary(self) -> None:
        """Print comprehensive scaling configuration summary."""
        print(f"\n{'=' * 80}")
        print("🎯 AUTO-SCALING CONFIGURATION SUMMARY")
        print(f"{'=' * 80}")

        if not self._auto_scaling_enabled:
            print("❌ Auto-scaling is DISABLED")
            return

        print("✅ Auto-scaling is ENABLED")
        print(f"📊 Total variables with scaling: {len(self._scaling_factors)}")

        print("\n📋 SCALING FACTORS BY RULE:")
        rules_count = {}
        for var_info in self._scaling_factors.values():
            rule = var_info["rule"]
            rules_count[rule] = rules_count.get(rule, 0) + 1

        for rule, count in rules_count.items():
            print(f"  {rule}: {count} variables")

        print("\n📐 DETAILED SCALING FACTORS:")
        print(
            f"{'Variable':<15} | {'Rule':<30} | {'v (scale)':<12} | {'r (shift)':<12} | {'Mapped Range'}"
        )
        print(f"{'-' * 15}-+-{'-' * 30}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 20}")

        for var_name, sf_info in sorted(self._scaling_factors.items()):
            rule = sf_info["rule"]
            v_factor = sf_info["v"]
            r_factor = sf_info["r"]

            # Calculate what physical range maps to [-0.5, 0.5]
            if v_factor != 0:
                phys_min = (-0.5 - r_factor) / v_factor
                phys_max = (0.5 - r_factor) / v_factor
                range_str = f"[{phys_min:.2e}, {phys_max:.2e}]"
            else:
                range_str = "N/A"

            print(
                f"{var_name:<15} | {rule:<30} | {v_factor:<12.3e} | {r_factor:<12.3e} | {range_str}"
            )

        print("\n🔗 VARIABLE MAPPINGS:")
        print(f"{'Physical':<15} ↔ {'scaled'}")
        print(f"{'-' * 15}---{'-' * 15}")
        for phys, scaled in sorted(self._physical_to_scaled_map.items()):
            print(f"{phys:<15} ↔ {scaled}")

        print("\n✅ Scaling configuration is consistent and ready for use")
        print(f"{'=' * 80}")
