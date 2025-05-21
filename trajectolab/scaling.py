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

import numpy as np

from trajectolab.tl_types import (
    Constraint,
    FloatArray,
    InitialGuess,
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

    def __init__(self, enabled: bool = True, variable_state=None) -> None:
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
        self._time_scale_factor: _ScaleFactor = (1.0, 0.0)  # Default: no scaling
        self._variable_state = variable_state

        scaling_logger.info(f"Automatic scaling {'enabled' if enabled else 'disabled'}")

    def register_state(
        self,
        name: str,
        sym: SymType,
        lower: float | None = None,
        upper: float | None = None,
        initial_value: float | None = None,
    ) -> None:
        """
        Register a state variable for scaling.

        Args:
            name: Variable name
            sym: Symbolic variable
            lower: Lower bound or None
            upper: Upper bound or None
            initial_value: Initial value for fallback scaling
        """
        # Skip if scaling disabled or variable already registered
        if not self.enabled or sym in self._sym_state_map:
            if not self.enabled:
                self._state_scale_factors[name] = (1.0, 0.0, "disabled")
                self._sym_state_map[sym] = name
            return

        # Calculate scale factors using Rule 2
        scale_weight, scale_shift, method = self._calculate_scale_factors(
            name, lower, upper, initial_value
        )

        self._state_scale_factors[name] = (scale_weight, scale_shift, method)
        self._sym_state_map[sym] = name

        scaling_logger.info(
            f"Registered state '{name}' with {method} scaling: "
            f"weight={scale_weight:.6e}, shift={scale_shift:.6f}"
        )

    def register_control(
        self,
        name: str,
        sym: SymType,
        lower: float | None = None,
        upper: float | None = None,
        initial_value: float | None = None,
    ) -> None:
        """
        Register a control variable for scaling.

        Args:
            name: Variable name
            sym: Symbolic variable
            lower: Lower bound or None
            upper: Upper bound or None
            initial_value: Initial value for fallback scaling
        """
        # Skip if scaling disabled or variable already registered
        if not self.enabled or sym in self._sym_control_map:
            if not self.enabled:
                self._control_scale_factors[name] = (1.0, 0.0, "disabled")
                self._sym_control_map[sym] = name
            return

        # Calculate scale factors using Rule 2
        scale_weight, scale_shift, method = self._calculate_scale_factors(
            name, lower, upper, initial_value
        )

        self._control_scale_factors[name] = (scale_weight, scale_shift, method)
        self._sym_control_map[sym] = name

        scaling_logger.info(
            f"Registered control '{name}' with {method} scaling: "
            f"weight={scale_weight:.6e}, shift={scale_shift:.6f}"
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
        scale_info = self._calculate_scale_factors("time", lower, upper)
        scale_weight, scale_shift, method_used = scale_info  # Unpack all three values
        self._time_scale_factor = (scale_weight, scale_shift)

        scaling_logger.info(
            f"Registered time with {method_used} scaling: weight={scale_weight:.6e}, shift={scale_shift:.6f}"
        )

    def _calculate_scale_factors(
        self,
        name: str,
        lower: float | None,
        upper: float | None,
        initial_value: float | None = None,
    ) -> tuple[float, float, str]:
        """
        Calculate scale weight and shift based on Rule 2.

        Args:
            name: Variable name for logging
            lower: Lower bound or None
            upper: Upper bound or None
            initial_value: Initial value if available

        Returns:
            Tuple of (scale_weight, scale_shift, method_used)
        """
        # Default: no scaling
        scale_weight = 1.0
        scale_shift = 0.0

        method_used = "default"

        # Rule 2.a: If bounds available, calculate scale factors
        if lower is not None and upper is not None and abs(upper - lower) > 1e-10:
            # Weight: 1/(upper - lower)  (equation 4.250)
            scale_weight = 1.0 / (upper - lower)

            # Shift: 0.5 - upper/(upper - lower)  (equation 4.251)
            scale_shift = 0.5 - upper / (upper - lower)

            method_used = "bounds"
            scaling_logger.info(
                f"Variable '{name}' using bounds [{lower}, {upper}] for scaling: "
                f"weight={scale_weight:.6e}, shift={scale_shift:.6f}"
            )
        # Rule 2.b: If bounds not usable but initial guess available
        elif initial_value is not None:
            # Estimate a range based on the initial guess
            # A simple approach: use initial_guess ± 50%
            range_estimate = max(abs(initial_value) * 0.5, 1.0)  # Ensure minimum range
            est_lower = initial_value - range_estimate
            est_upper = initial_value + range_estimate

            scale_weight = 1.0 / (est_upper - est_lower)
            scale_shift = 0.5 - est_upper / (est_upper - est_lower)

            method_used = "initial_guess"
            scaling_logger.info(
                f"Variable '{name}' using initial guess {initial_value} for scaling: "
                f"estimated range [{est_lower}, {est_upper}], "
                f"weight={scale_weight:.6e}, shift={scale_shift:.6f}"
            )
        # Rule 2.c: Default case
        else:
            scaling_logger.info(
                f"Variable '{name}' has insufficient information for scaling. "
                f"Using default: weight=1.0, shift=0.0"
            )

        return (scale_weight, scale_shift, method_used)

    def get_state_scale_factors(self, name: str) -> _ScaleFactor:
        """Get scale factors for a state variable."""
        return self._state_scale_factors.get(name, (1.0, 0.0))

    def get_control_scale_factors(self, name: str) -> _ScaleFactor:
        """Get scale factors for a control variable."""
        return self._control_scale_factors.get(name, (1.0, 0.0))

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

        # Extract just the weight and shift, ignoring the method
        scale_info = self._state_scale_factors[name]
        scale_weight = scale_info[0]
        scale_shift = scale_info[1]

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

        # Extract just the weight and shift, ignoring the method
        scale_info = self._control_scale_factors[name]
        scale_weight = scale_info[0]
        scale_shift = scale_info[1]

        # Apply scaling: ũ = v_u * u + r_u
        return scale_weight * value + scale_shift

    def scale_time_value(self, value: float | FloatArray) -> float | FloatArray:
        """
        Scale a time value.

        Args:
            name: Time value name
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

        # Extract just the weight and shift, ignoring the method
        scale_info = self._state_scale_factors[name]
        scale_weight = scale_info[0]
        scale_shift = scale_info[1]

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

        # Extract just the weight and shift, ignoring the method
        scale_info = self._control_scale_factors[name]
        scale_weight = scale_info[0]
        scale_shift = scale_info[1]

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

    def scale_dynamics(self, state_sym: SymType, expr: SymExpr) -> SymExpr:
        """
        Properly scale a dynamics expression according to Rule 3.

        This transforms the dynamics from dx/dt = f(x,u,t) to dx̃/dt̃ correctly
        by substituting scaled variables and applying the appropriate scaling ratio.
        """
        if not self.enabled:
            return expr

        # Get state name and scale factors
        state_name = self._sym_state_map.get(state_sym)
        if state_name is None:
            return expr

        # Extract scaling factors
        state_info = self._state_scale_factors.get(state_name)
        if state_info is None:
            return expr

        state_weight, state_shift = state_info[0], state_info[1]
        time_weight, time_shift = self._time_scale_factor

        # Create substitution maps for scaled to unscaled variables
        scaled_to_unscaled = {}

        # For each state variable
        for sym, name in self._sym_state_map.items():
            if name in self._state_scale_factors:
                v_y, r_y = self._state_scale_factors[name][0:2]
                # Transform x̃ to x: x = (x̃ - r_y) / v_y
                scaled_to_unscaled[sym] = (sym - r_y) / v_y

        # For each control variable
        for sym, name in self._sym_control_map.items():
            if name in self._control_scale_factors:
                v_u, r_u = self._control_scale_factors[name][0:2]
                # Transform ũ to u: u = (ũ - r_u) / v_u
                scaled_to_unscaled[sym] = (sym - r_u) / v_u

        # For time variable
        if self._variable_state.sym_time is not None:
            # Transform t̃ to t: t = (t̃ - r_t) / v_t
            scaled_to_unscaled[self._variable_state.sym_time] = (
                self._variable_state.sym_time - time_shift
            ) / time_weight

        # First substitute all variables to work with unscaled expressions
        unscaled_expr = ca.substitute(
            [expr], list(scaled_to_unscaled.keys()), list(scaled_to_unscaled.values())
        )[0]

        # Then apply the proper scaling ratio as per rule 3
        scaling_ratio = state_weight / time_weight
        scaled_expr = scaling_ratio * unscaled_expr

        method = state_info[2] if len(state_info) > 2 else "unknown"
        scaling_logger.info(
            f"Scaling dynamics for state '{state_name}' ({method}): "
            f"scale_weight={state_weight:.6e}, time_weight={time_weight:.6e}, "
            f"ratio={scaling_ratio:.6e}"
        )

        return scaled_expr

    def scale_constraint(self, constraint: Constraint) -> Constraint:
        """
        Scale a constraint.

        Args:
            constraint: Original constraint

        Returns:
            Scaled constraint
        """
        if not self.enabled:
            return constraint

        # For now, we're keeping constraints unscaled
        # This is a simplification; a more complete implementation would
        # need to analyze the constraint expression and scale it appropriately
        return constraint

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
            state_names = list(self._sym_state_map.values())
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
            control_names = list(self._sym_control_map.values())
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
        # Note: Integrals scaling is not implemented here
        return InitialGuess(
            initial_time_variable=scaled_t0,
            terminal_time_variable=scaled_tf,
            states=scaled_states,
            controls=scaled_controls,
            integrals=initial_guess.integrals,
        )
