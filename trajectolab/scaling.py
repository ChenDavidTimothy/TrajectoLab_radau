"""
Automatic scaling functionality for optimal control problems.

Implements the three fundamental scaling rules:
1. Scale from a Control Perspective: Consistent scaling across time
2. Variable Scaling: Transform variables to O(1)
3. ODE Defect Scaling: Scale constraints based on state scaling
"""

from __future__ import annotations

import logging
from typing import TypeAlias

import casadi as ca
import numpy as np

from trajectolab.tl_types import (
    Constraint,
    FloatArray,
    InitialGuess,
    OptimalControlSolution,
    SymExpr,
    SymType,
)


# Configure scaling-specific logger
scaling_logger = logging.getLogger("trajectolab.scaling")
if not scaling_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    scaling_logger.addHandler(handler)
    scaling_logger.setLevel(logging.INFO)

# Type alias for scale factors
_ScaleFactor: TypeAlias = tuple[float, float]  # (weight, shift)


class ScalingManager:
    """
    Manages automatic scaling for optimal control problems.

    Implements the three scaling rules:
    1. Scale from a Control Perspective: Consistent scaling across time
    2. Variable Scaling: Transform variables to O(1)
    3. ODE Defect Scaling: Scale constraints based on state scaling
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialize scaling manager.

        Args:
            enabled: Whether automatic scaling is enabled
        """
        self.enabled = enabled
        self._state_scale_factors: dict[str, _ScaleFactor] = {}
        self._control_scale_factors: dict[str, _ScaleFactor] = {}
        self._sym_state_map: dict[SymType, str] = {}
        self._sym_control_map: dict[SymType, str] = {}
        self._param_scale_factors: dict[str, _ScaleFactor] = {}
        self._time_scale_factor: _ScaleFactor = (1.0, 0.0)  # Default: no scaling
        self._original_state_syms: dict[str, SymType] = {}
        self._original_control_syms: dict[str, SymType] = {}
        self._scaled_state_syms: dict[str, SymType] = {}
        self._scaled_control_syms: dict[str, SymType] = {}

        scaling_logger.info(f"Automatic scaling {'enabled' if enabled else 'disabled'}")

    def register_state(
        self, name: str, sym: SymType, lower: float | None = None, upper: float | None = None
    ) -> None:
        """
        Register a state variable for scaling.

        Args:
            name: Variable name
            sym: Symbolic variable
            lower: Lower bound or None
            upper: Upper bound or None
        """
        # Check if this symbolic variable is already registered
        if sym in self._sym_state_map:
            return

        if not self.enabled:
            self._state_scale_factors[name] = (1.0, 0.0)  # No scaling
            self._sym_state_map[sym] = name
            self._original_state_syms[name] = sym
            return

        # Calculate scale factors using Rule 2
        scale_weight, scale_shift = self._calculate_scale_factors(name, lower, upper)
        self._state_scale_factors[name] = (scale_weight, scale_shift)
        self._sym_state_map[sym] = name
        self._original_state_syms[name] = sym

        # Create a new symbolic variable for the scaled version
        scaled_sym = ca.MX.sym(f"{name}_scaled", sym.shape)
        self._scaled_state_syms[name] = scaled_sym

        scaling_logger.debug(
            f"Registered state '{name}' with scale factor: weight={scale_weight}, shift={scale_shift}"
        )

    def register_control(
        self, name: str, sym: SymType, lower: float | None = None, upper: float | None = None
    ) -> None:
        """
        Register a control variable for scaling.

        Args:
            name: Variable name
            sym: Symbolic variable
            lower: Lower bound or None
            upper: Upper bound or None
        """
        if not self.enabled:
            self._control_scale_factors[name] = (1.0, 0.0)  # No scaling
            self._sym_control_map[sym] = name
            self._original_control_syms[name] = sym
            return

        # Calculate scale factors using Rule 2
        scale_weight, scale_shift = self._calculate_scale_factors(name, lower, upper)
        self._control_scale_factors[name] = (scale_weight, scale_shift)
        self._sym_control_map[sym] = name
        self._original_control_syms[name] = sym

        # Create a new symbolic variable for the scaled version
        scaled_sym = ca.MX.sym(f"{name}_scaled", sym.shape)
        self._scaled_control_syms[name] = scaled_sym

        scaling_logger.debug(
            f"Registered control '{name}' with scale factor: weight={scale_weight}, shift={scale_shift}"
        )

    def register_time_bounds(self, lower: float, upper: float) -> None:
        """
        Register time bounds for scaling.

        Args:
            lower: Lower bound on time
            upper: Upper bound on time
        """
        if not self.enabled:
            self._time_scale_factor = (1.0, 0.0)  # No scaling
            return

        # Calculate time scaling based on bounds using Rule 2
        scale_weight, scale_shift = self._calculate_scale_factors("time", lower, upper)
        self._time_scale_factor = (scale_weight, scale_shift)

        scaling_logger.debug(
            f"Registered time with scale factor: weight={scale_weight}, shift={scale_shift}"
        )

    def _calculate_scale_factors(
        self, name: str, lower: float | None, upper: float | None
    ) -> _ScaleFactor:
        """
        Calculate scale weight and shift based on Rule 2.

        Args:
            name: Variable name for logging
            lower: Lower bound or None
            upper: Upper bound or None

        Returns:
            Tuple of (scale_weight, scale_shift)
        """
        # Default: no scaling
        scale_weight = 1.0
        scale_shift = 0.0

        # Rule 2: If bounds available, calculate scale factors
        if lower is not None and upper is not None and upper != lower:
            # Weight: 1/(upper - lower)  (equation 4.250)
            scale_weight = 1.0 / (upper - lower)

            # Shift: 0.5 - upper/(upper - lower)  (equation 4.251)
            scale_shift = 0.5 - upper / (upper - lower)

            scaling_logger.debug(
                f"Variable '{name}' has bounds [{lower}, {upper}]: "
                f"weight={scale_weight}, shift={scale_shift}"
            )
        else:
            scaling_logger.debug(
                f"Variable '{name}' has insufficient bound information for scaling. "
                f"Using default: weight=1.0, shift=0.0"
            )

        return (scale_weight, scale_shift)

    def scale_state_value(self, name: str, value: float | FloatArray) -> float | FloatArray:
        """
        Scale a state variable value.

        Args:
            name: State variable name
            value: Physical value

        Returns:
            Scaled value
        """
        if not self.enabled or name not in self._state_scale_factors:
            return value

        scale_weight, scale_shift = self._state_scale_factors[name]

        # Apply scaling: ỹ = v_y * y + r_y
        return scale_weight * value + scale_shift

    def scale_control_value(self, name: str, value: float | FloatArray) -> float | FloatArray:
        """
        Scale a control variable value.

        Args:
            name: Control variable name
            value: Physical value

        Returns:
            Scaled value
        """
        if not self.enabled or name not in self._control_scale_factors:
            return value

        scale_weight, scale_shift = self._control_scale_factors[name]

        # Apply scaling: ũ = v_u * u + r_u
        return scale_weight * value + scale_shift

    def scale_time_value(self, value: float | FloatArray) -> float | FloatArray:
        """
        Scale a time value.

        Args:
            value: Physical value

        Returns:
            Scaled value
        """
        if not self.enabled:
            return value

        scale_weight, scale_shift = self._time_scale_factor

        # Apply scaling: t̃ = v_t * t + r_t
        return scale_weight * value + scale_shift

    def unscale_state_value(self, name: str, value: float | FloatArray) -> float | FloatArray:
        """
        Unscale a state variable value.

        Args:
            name: State variable name
            value: Scaled value

        Returns:
            Physical value
        """
        if not self.enabled or name not in self._state_scale_factors:
            return value

        scale_weight, scale_shift = self._state_scale_factors[name]

        # Invert scaling: y = (ỹ - r_y) / v_y
        return (value - scale_shift) / scale_weight

    def unscale_control_value(self, name: str, value: float | FloatArray) -> float | FloatArray:
        """
        Unscale a control variable value.

        Args:
            name: Control variable name
            value: Scaled value

        Returns:
            Physical value
        """
        if not self.enabled or name not in self._control_scale_factors:
            return value

        scale_weight, scale_shift = self._control_scale_factors[name]

        # Invert scaling: u = (ũ - r_u) / v_u
        return (value - scale_shift) / scale_weight

    def unscale_time_value(self, value: float | FloatArray) -> float | FloatArray:
        """
        Unscale a time value.

        Args:
            value: Scaled value

        Returns:
            Physical value
        """
        if not self.enabled:
            return value

        scale_weight, scale_shift = self._time_scale_factor

        # Invert scaling: t = (t̃ - r_t) / v_t
        return (value - scale_shift) / scale_weight

    def create_scaled_dynamics(
        self, dynamics_dict: dict[SymType, SymExpr]
    ) -> dict[SymType, SymExpr]:
        """
        Scale the dynamics equations.

        This implements Rules 1 and 3 by:
        1. Consistently scaling state variables
        2. Applying appropriate ODE defect scaling

        Args:
            dynamics_dict: Mapping from state symbols to dynamics expressions (in physical units)

        Returns:
            Mapping from state symbols to scaled dynamics expressions
        """
        if not self.enabled:
            return dynamics_dict

        # Create substitution maps for states and controls
        state_subs_map = {}
        control_subs_map = {}

        # For each state, create substitution: original_sym → unscaled expression of scaled_sym
        for name, orig_sym in self._original_state_syms.items():
            if name not in self._scaled_state_syms:
                continue

            scaled_sym = self._scaled_state_syms[name]
            scale_weight, scale_shift = self._state_scale_factors[name]

            # Unscaled expression: y = (ỹ - r_y) / v_y
            unscaled_expr = (scaled_sym - scale_shift) / scale_weight
            state_subs_map[orig_sym] = unscaled_expr

        # Same for controls
        for name, orig_sym in self._original_control_syms.items():
            if name not in self._scaled_control_syms:
                continue

            scaled_sym = self._scaled_control_syms[name]
            scale_weight, scale_shift = self._control_scale_factors[name]

            # Unscaled expression: u = (ũ - r_u) / v_u
            unscaled_expr = (scaled_sym - scale_shift) / scale_weight
            control_subs_map[orig_sym] = unscaled_expr

        # Combine substitution maps
        full_subs_map = {**state_subs_map, **control_subs_map}

        # Apply substitutions to dynamics expressions and scale appropriately
        scaled_dynamics_dict = {}

        for state_sym, dynamics_expr in dynamics_dict.items():
            if state_sym not in self._sym_state_map:
                # If state not registered, pass through unchanged
                scaled_dynamics_dict[state_sym] = dynamics_expr
                continue

            state_name = self._sym_state_map[state_sym]
            if state_name not in self._scaled_state_syms:
                # If scaled sym not created, pass through unchanged
                scaled_dynamics_dict[state_sym] = dynamics_expr
                continue

            scaled_state_sym = self._scaled_state_syms[state_name]
            state_scale_weight, _ = self._state_scale_factors[state_name]
            time_scale_weight, _ = self._time_scale_factor

            # Step 1: Substitute all variables with their unscaled expressions
            # This changes f(y,u,t) to f((ỹ-r_y)/v_y, (ũ-r_u)/v_u, (t̃-r_t)/v_t)
            substituted_expr = ca.substitute(
                [dynamics_expr], list(full_subs_map.keys()), list(full_subs_map.values())
            )[0]

            # Step 2: Apply scaling according to Rule 3
            # For ODE ẏ = f(y,u,t), the scaled equation is:
            # d(ỹ)/d(t̃) = (v_y/v_t) * f((ỹ-r_y)/v_y, (ũ-r_u)/v_u, (t̃-r_t)/v_t)
            scaled_expr = (state_scale_weight / time_scale_weight) * substituted_expr

            # Add to result with the scaled state symbol
            scaled_dynamics_dict[state_sym] = scaled_expr

        return scaled_dynamics_dict

    def scale_constraint(self, constraint: Constraint) -> Constraint:
        """
        Scale a constraint by applying scaling to its variables and bounds.

        Args:
            constraint: Original constraint in physical units

        Returns:
            Scaled constraint
        """
        if not self.enabled:
            return constraint

        # Create substitution maps for original to unscaled expressions
        subs_map = {}

        # Handle state variables
        for name, orig_sym in self._original_state_syms.items():
            if name in self._scaled_state_syms:
                scaled_sym = self._scaled_state_syms[name]
                scale_weight, scale_shift = self._state_scale_factors[name]
                unscaled_expr = (scaled_sym - scale_shift) / scale_weight
                subs_map[orig_sym] = unscaled_expr

        # Handle control variables
        for name, orig_sym in self._original_control_syms.items():
            if name in self._scaled_control_syms:
                scaled_sym = self._scaled_control_syms[name]
                scale_weight, scale_shift = self._control_scale_factors[name]
                unscaled_expr = (scaled_sym - scale_shift) / scale_weight
                subs_map[orig_sym] = unscaled_expr

        # Apply substitution to constraint expression
        original_val = constraint.val
        substituted_val = ca.substitute(
            [original_val], list(subs_map.keys()), list(subs_map.values())
        )[0]

        # Create scaled constraint with original bounds
        # The bounds will be applied to the scaled expression in the solver
        scaled_constraint = Constraint(
            val=substituted_val,
            min_val=constraint.min_val,
            max_val=constraint.max_val,
            equals=constraint.equals,
        )

        return scaled_constraint

    def scale_initial_guess(self, initial_guess: InitialGuess | None) -> InitialGuess | None:
        """
        Scale all components of an initial guess.

        Args:
            initial_guess: Original initial guess with physical values

        Returns:
            Scaled initial guess
        """
        if not self.enabled or initial_guess is None:
            return initial_guess

        # Scale time variables
        scaled_t0 = None
        if initial_guess.initial_time_variable is not None:
            scaled_t0 = self.scale_time_value(initial_guess.initial_time_variable)

        scaled_tf = None
        if initial_guess.terminal_time_variable is not None:
            scaled_tf = self.scale_time_value(initial_guess.terminal_time_variable)

        # Scale state trajectories
        scaled_states = None
        if initial_guess.states is not None:
            state_names = list(self._original_state_syms.keys())
            scaled_states = []

            # For each interval's state trajectory
            for interval_states in initial_guess.states:
                # Create scaled copy
                interval_scaled = np.copy(interval_states)

                # Scale each state variable
                for i, name in enumerate(state_names):
                    if i < interval_states.shape[0]:
                        interval_scaled[i, :] = self.scale_state_value(name, interval_states[i, :])

                scaled_states.append(interval_scaled)

        # Scale control trajectories
        scaled_controls = None
        if initial_guess.controls is not None:
            control_names = list(self._original_control_syms.keys())
            scaled_controls = []

            # For each interval's control trajectory
            for interval_controls in initial_guess.controls:
                # Create scaled copy
                interval_scaled = np.copy(interval_controls)

                # Scale each control variable
                for i, name in enumerate(control_names):
                    if i < interval_controls.shape[0]:
                        interval_scaled[i, :] = self.scale_control_value(
                            name, interval_controls[i, :]
                        )

                scaled_controls.append(interval_scaled)

        # Return scaled initial guess
        # Note: Integrals scaling would depend on problem formulation
        # For simplicity, we leave integrals unscaled here
        return InitialGuess(
            initial_time_variable=scaled_t0,
            terminal_time_variable=scaled_tf,
            states=scaled_states,
            controls=scaled_controls,
            integrals=initial_guess.integrals,
        )

    def unscale_solution(self, solution: OptimalControlSolution) -> OptimalControlSolution:
        """
        Unscale an optimal control solution.

        Args:
            solution: Solution with scaled values

        Returns:
            Solution with physical values
        """
        if not self.enabled:
            return solution

        # Unscale time variables
        if solution.initial_time_variable is not None:
            solution.initial_time_variable = self.unscale_time_value(solution.initial_time_variable)
        if solution.terminal_time_variable is not None:
            solution.terminal_time_variable = self.unscale_time_value(
                solution.terminal_time_variable
            )

        # Unscale state trajectories
        if solution.states is not None:
            state_names = list(self._original_state_syms.keys())
            for i, name in enumerate(state_names):
                if i < len(solution.states):
                    solution.states[i] = self.unscale_state_value(name, solution.states[i])

        # Unscale control trajectories
        if solution.controls is not None:
            control_names = list(self._original_control_syms.keys())
            for i, name in enumerate(control_names):
                if i < len(solution.controls):
                    solution.controls[i] = self.unscale_control_value(name, solution.controls[i])

        # Unscale per-interval trajectories if available
        if solution.solved_state_trajectories_per_interval is not None:
            state_names = list(self._original_state_syms.keys())
            for interval_idx, interval_states in enumerate(
                solution.solved_state_trajectories_per_interval
            ):
                for i, name in enumerate(state_names):
                    if i < interval_states.shape[0]:
                        solution.solved_state_trajectories_per_interval[interval_idx][i, :] = (
                            self.unscale_state_value(name, interval_states[i, :])
                        )

        if solution.solved_control_trajectories_per_interval is not None:
            control_names = list(self._original_control_syms.keys())
            for interval_idx, interval_controls in enumerate(
                solution.solved_control_trajectories_per_interval
            ):
                for i, name in enumerate(control_names):
                    if i < interval_controls.shape[0]:
                        solution.solved_control_trajectories_per_interval[interval_idx][i, :] = (
                            self.unscale_control_value(name, interval_controls[i, :])
                        )

        return solution

    def scale_objective(self, objective_expr: SymExpr) -> SymExpr:
        """
        Scale an objective function expression.

        Args:
            objective_expr: Objective expression (in physical units)

        Returns:
            Scaled objective expression
        """
        if not self.enabled:
            return objective_expr

        # Create substitution maps for states and controls
        subs_map = {}

        # For states: original_sym → unscaled expression of scaled_sym
        for name, orig_sym in self._original_state_syms.items():
            if name in self._scaled_state_syms:
                scaled_sym = self._scaled_state_syms[name]
                scale_weight, scale_shift = self._state_scale_factors[name]
                unscaled_expr = (scaled_sym - scale_shift) / scale_weight
                subs_map[orig_sym] = unscaled_expr

        # For controls: original_sym → unscaled expression of scaled_sym
        for name, orig_sym in self._original_control_syms.items():
            if name in self._scaled_control_syms:
                scaled_sym = self._scaled_control_syms[name]
                scale_weight, scale_shift = self._control_scale_factors[name]
                unscaled_expr = (scaled_sym - scale_shift) / scale_weight
                subs_map[orig_sym] = unscaled_expr

        # Apply substitution to objective expression
        # This replaces original symbols with expressions containing scaled symbols
        # J(y,u) → J((ỹ-r_y)/v_y, (ũ-r_u)/v_u)
        substituted_obj = ca.substitute(
            [objective_expr], list(subs_map.keys()), list(subs_map.values())
        )[0]

        return substituted_obj
