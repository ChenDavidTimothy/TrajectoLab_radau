"""
Core auto-scaling functionality for TrajectoLab.

This module implements the mathematical transformations and state management
for auto-scaling in optimal control problems. All scaling operations follow
the established rules for NASA safety-critical systems.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np

from ..tl_types import FloatArray, FloatMatrix, SymExpr, SymType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingFactors:
    """
    Immutable scaling transformation parameters.

    Transformation: scaled = v * physical + r
    Inverse: physical = (scaled - r) / v
    """

    v: float  # Scale factor
    r: float  # Shift factor
    rule: str  # Rule applied ("2.1.a", "2.1.b", "2.4")

    def __post_init__(self) -> None:
        """Validate scaling factors for safety-critical use."""
        if not np.isfinite(self.v) or not np.isfinite(self.r):
            raise ValueError(f"Non-finite scaling factors: v={self.v}, r={self.r}")

        if np.abs(self.v) < 1e-15:
            raise ValueError(f"Scaling factor v={self.v} too small - numerical instability risk")

        if np.abs(self.v) > 1e15:
            raise ValueError(f"Scaling factor v={self.v} too large - numerical instability risk")


@dataclass
class VariableMappings:
    """Bidirectional mappings between physical and scaled variable names."""

    physical_to_scaled: dict[str, str] = field(default_factory=dict)
    scaled_to_physical: dict[str, str] = field(default_factory=dict)
    physical_symbols: dict[str, SymType] = field(default_factory=dict)

    def add_mapping(self, physical_name: str, scaled_name: str, physical_symbol: SymType) -> None:
        """Add bidirectional mapping between physical and scaled variables."""
        self.physical_to_scaled[physical_name] = scaled_name
        self.scaled_to_physical[scaled_name] = physical_name
        self.physical_symbols[physical_name] = physical_symbol


def determine_scaling_factors(
    var_name: str,
    explicit_lower: float | None,
    explicit_upper: float | None,
    guess_min: float | None,
    guess_max: float | None,
) -> ScalingFactors:
    """
    Determine scaling factors using established rules for NASA safety applications.

    Rules:
    - 2.1.a: Use explicit bounds to scale to [-0.5, 0.5]
    - 2.1.b: Use initial guess range to scale to [-0.5, 0.5]
    - 2.4: Default (no scaling)

    Args:
        var_name: Variable name for error reporting
        explicit_lower: Explicit lower bound
        explicit_upper: Explicit upper bound
        guess_min: Minimum from initial guess range
        guess_max: Maximum from initial guess range

    Returns:
        ScalingFactors with transformation parameters

    Raises:
        ValueError: If scaling parameters are invalid
    """
    logger.debug(f"Determining scaling factors for variable '{var_name}'")
    logger.debug(f"  Bounds: [{explicit_lower}, {explicit_upper}]")
    logger.debug(f"  Guess range: [{guess_min}, {guess_max}]")

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

        logger.debug(f"  Applied Rule 2.1.a: range={range_val}, vk={vk}, rk={rk}")

        # Verify transformation maps to [-0.5, 0.5]
        scaled_lower = vk * explicit_lower + rk
        scaled_upper = vk * explicit_upper + rk
        if not (
            np.isclose(scaled_lower, -0.5, atol=1e-10) and np.isclose(scaled_upper, 0.5, atol=1e-10)
        ):
            logger.warning(
                f"  Scaling verification failed: [{scaled_lower}, {scaled_upper}] != [-0.5, 0.5]"
            )

    # Rule 2.1.b: Use initial guess range if available
    elif guess_min is not None and guess_max is not None and not np.isclose(guess_max, guess_min):
        range_val = guess_max - guess_min
        vk = 1.0 / range_val
        rk = 0.5 - guess_max / range_val
        rule_applied = "2.1.b (Initial Guess Range)"

        logger.debug(f"  Applied Rule 2.1.b: range={range_val}, vk={vk}, rk={rk}")

        # Verify transformation
        scaled_min = vk * guess_min + rk
        scaled_max = vk * guess_max + rk
        if not (
            np.isclose(scaled_min, -0.5, atol=1e-10) and np.isclose(scaled_max, 0.5, atol=1e-10)
        ):
            logger.warning(
                f"  Scaling verification failed: [{scaled_min}, {scaled_max}] != [-0.5, 0.5]"
            )

    else:
        logger.debug("  Applied Rule 2.4 (Default): no scaling")

    return ScalingFactors(v=vk, r=rk, rule=rule_applied)


def scale_values(values: FloatArray, factors: ScalingFactors) -> FloatArray:
    """
    Apply scaling transformation: scaled = v * physical + r

    Args:
        values: Physical space values
        factors: Scaling transformation parameters

    Returns:
        Values in scaled space

    Raises:
        ValueError: If transformation produces invalid results
    """
    if values.size == 0:
        return values

    scaled = factors.v * values + factors.r

    if not np.all(np.isfinite(scaled)):
        raise ValueError("Scaling transformation produced non-finite values")

    return scaled


def unscale_values(values: FloatArray, factors: ScalingFactors) -> FloatArray:
    """
    Apply inverse scaling transformation: physical = (scaled - r) / v

    Args:
        values: Scaled space values
        factors: Scaling transformation parameters

    Returns:
        Values in physical space

    Raises:
        ValueError: If transformation produces invalid results
    """
    if values.size == 0:
        return values

    physical = (values - factors.r) / factors.v

    if not np.all(np.isfinite(physical)):
        raise ValueError("Unscaling transformation produced non-finite values")

    return physical


def scale_trajectory_arrays(
    trajectories: Sequence[FloatMatrix],
    variable_names: list[str],
    scaling_factors_dict: dict[str, ScalingFactors],
) -> list[FloatMatrix]:
    """
    Scale trajectory arrays from physical to scaled space.

    Args:
        trajectories: List of trajectory arrays in physical space
        variable_names: Ordered list of variable names
        scaling_factors_dict: Scaling factors for each variable

    Returns:
        List of trajectory arrays in scaled space

    Raises:
        ValueError: If array dimensions don't match variable count
    """
    if not trajectories:
        return []

    logger.debug(f"Scaling {len(trajectories)} trajectory arrays")
    logger.debug(f"Variables: {variable_names}")

    scaled_trajectories = []

    for traj_idx, traj_array in enumerate(trajectories):
        logger.debug(f"  Processing trajectory {traj_idx}: shape {traj_array.shape}")

        # Validate dimensions
        if traj_array.shape[0] != len(variable_names):
            raise ValueError(
                f"Trajectory {traj_idx} has {traj_array.shape[0]} rows, "
                f"expected {len(variable_names)} for variables {variable_names}"
            )

        # Validate finite values
        if not np.all(np.isfinite(traj_array)):
            raise ValueError(f"Trajectory {traj_idx} contains non-finite values")

        # Scale each row (variable)
        scaled_array = np.zeros_like(traj_array, dtype=np.float64)
        for i, var_name in enumerate(variable_names):
            if var_name not in scaling_factors_dict:
                raise ValueError(f"No scaling factors found for variable '{var_name}'")

            factors = scaling_factors_dict[var_name]
            scaled_array[i, :] = scale_values(traj_array[i, :], factors)

            logger.debug(f"    Variable {var_name}: scaled with {factors.rule}")

        scaled_trajectories.append(scaled_array)

    return scaled_trajectories


def unscale_trajectory_arrays(
    trajectories: Sequence[FloatMatrix],
    variable_names: list[str],
    scaling_factors_dict: dict[str, ScalingFactors],
) -> list[FloatMatrix]:
    """
    Unscale trajectory arrays from scaled to physical space.

    Args:
        trajectories: List of trajectory arrays in scaled space
        variable_names: Ordered list of variable names
        scaling_factors_dict: Scaling factors for each variable

    Returns:
        List of trajectory arrays in physical space
    """
    if not trajectories:
        return []

    unscaled_trajectories = []

    for traj_array in trajectories:
        unscaled_array = np.zeros_like(traj_array, dtype=np.float64)
        for i, var_name in enumerate(variable_names):
            if var_name in scaling_factors_dict:
                factors = scaling_factors_dict[var_name]
                unscaled_array[i, :] = unscale_values(traj_array[i, :], factors)
            else:
                unscaled_array[i, :] = traj_array[i, :]  # No scaling applied

        unscaled_trajectories.append(unscaled_array)

    return unscaled_trajectories


def transform_dynamics_to_scaled_space(
    physical_dynamics: dict[SymType, SymExpr],
    physical_symbols: dict[str, SymType],
    scaled_symbols: dict[str, SymType],
    scaling_factors_dict: dict[str, ScalingFactors],
) -> dict[SymType, SymExpr]:
    """
    Transform dynamics equations from physical to scaled space using chain rule.

    If x_scaled = v * x_physical + r, then:
    dx_scaled/dt = v * dx_physical/dt

    Args:
        physical_dynamics: Dynamics in physical space
        physical_symbols: Physical symbolic variables
        scaled_symbols: Scaled symbolic variables
        scaling_factors_dict: Scaling factors

    Returns:
        Dynamics equations in scaled space

    Raises:
        ValueError: If symbol mapping fails
    """
    logger.debug("Transforming dynamics to scaled space")

    scaled_dynamics = {}

    for state_sym, rhs_expr in physical_dynamics.items():
        # Find physical variable name
        physical_name = None
        for name, sym in physical_symbols.items():
            if ca.is_equal(sym, state_sym):
                physical_name = name
                break

        if physical_name is None:
            raise ValueError("Physical variable not found in dynamics definition")

        # Get corresponding scaled symbol
        if physical_name not in scaled_symbols:
            raise ValueError(f"Scaled symbol not found for physical variable {physical_name}")

        scaled_sym = scaled_symbols[physical_name]

        # Get scaling factor and apply chain rule
        if physical_name not in scaling_factors_dict:
            raise ValueError(f"Scaling factors not found for {physical_name}")

        factors = scaling_factors_dict[physical_name]

        # Chain rule: d(scaled)/dt = v * d(physical)/dt
        scaled_rhs = factors.v * rhs_expr
        scaled_dynamics[scaled_sym] = scaled_rhs

        logger.debug(f"  {physical_name}: scaled with factor {factors.v}")

    logger.debug(f"Transformed {len(scaled_dynamics)} dynamics equations")
    return scaled_dynamics


class AutoScalingManager:
    """
    Manages auto-scaling state and integrates with Problem/Solution classes.

    This class maintains all scaling-related state and provides high-level
    operations for setting up and managing variable scaling throughout
    the optimal control problem solution process.
    """

    def __init__(self) -> None:
        """Initialize empty auto-scaling manager."""
        self._scaling_factors: dict[str, ScalingFactors] = {}
        self._mappings = VariableMappings()
        self._initial_guess_ranges: dict[str, dict[str, float | None]] = {}

    @property
    def scaling_factors(self) -> dict[str, ScalingFactors]:
        """Get read-only view of scaling factors."""
        return self._scaling_factors.copy()

    @property
    def variable_mappings(self) -> VariableMappings:
        """Get variable mappings."""
        return self._mappings

    def update_initial_guess_range(
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
            initial: Initial value constraint
            final: Final value constraint
            lower: Lower bound
            upper: Upper bound
        """
        if var_name not in self._initial_guess_ranges:
            self._initial_guess_ranges[var_name] = {"min": None, "max": None}

        range_info = self._initial_guess_ranges[var_name]

        # Update from initial/final values
        values_to_consider = [v for v in [initial, final, lower, upper] if v is not None]

        if values_to_consider:
            current_min = range_info["min"]
            current_max = range_info["max"]

            new_min = min(values_to_consider)
            new_max = max(values_to_consider)

            range_info["min"] = new_min if current_min is None else min(current_min, new_min)
            range_info["max"] = new_max if current_max is None else max(current_max, new_max)

    def setup_variable_scaling(
        self,
        var_name: str,
        explicit_lower: float | None,
        explicit_upper: float | None,
        scale_guide_lower: float | None = None,
        scale_guide_upper: float | None = None,
    ) -> ScalingFactors:
        """
        Set up scaling for a variable and store the scaling factors.

        Args:
            var_name: Physical variable name
            explicit_lower: Explicit lower bound
            explicit_upper: Explicit upper bound
            scale_guide_lower: Override lower bound for scaling
            scale_guide_upper: Override upper bound for scaling

        Returns:
            Computed scaling factors
        """
        # Use scale guides if provided, otherwise use explicit bounds
        bounds_lower = scale_guide_lower if scale_guide_lower is not None else explicit_lower
        bounds_upper = scale_guide_upper if scale_guide_upper is not None else explicit_upper

        # Get initial guess range
        guess_info = self._initial_guess_ranges.get(var_name, {})
        guess_min = guess_info.get("min")
        guess_max = guess_info.get("max")

        # Determine scaling factors
        factors = determine_scaling_factors(
            var_name, bounds_lower, bounds_upper, guess_min, guess_max
        )

        # Store scaling factors
        self._scaling_factors[var_name] = factors

        logger.info(
            f"Variable '{var_name}' scaling: {factors.rule}, v={factors.v:.3e}, r={factors.r:.3f}"
        )

        return factors

    def get_scaled_properties(
        self,
        var_name: str,
        initial: float | None,
        final: float | None,
        lower: float | None,
        upper: float | None,
    ) -> dict[str, float | None]:
        """
        Get scaled properties for variable creation.

        Args:
            var_name: Variable name
            initial: Initial value in physical space
            final: Final value in physical space
            lower: Lower bound in physical space
            upper: Upper bound in physical space

        Returns:
            Dictionary with scaled properties
        """
        if var_name not in self._scaling_factors:
            raise ValueError(f"No scaling factors found for variable '{var_name}'")

        factors = self._scaling_factors[var_name]
        scaled_props: dict[str, float | None] = {}

        # Transform initial/final values
        if initial is not None:
            scaled_props["initial"] = factors.v * initial + factors.r
        if final is not None:
            scaled_props["final"] = factors.v * final + factors.r

        # Transform bounds
        if factors.rule != "2.4 (Default)":
            # For variables with explicit bounds, normalize to [-0.5, 0.5]
            scaled_props["lower"] = -0.5
            scaled_props["upper"] = 0.5
        else:
            # For default scaling, transform bounds if present
            scaled_props["lower"] = factors.v * lower + factors.r if lower is not None else None
            scaled_props["upper"] = factors.v * upper + factors.r if upper is not None else None

        return scaled_props

    def create_physical_symbol(
        self,
        physical_name: str,
        scaled_symbol: SymType,
    ) -> SymType:
        """
        Create physical symbolic variable from scaled symbol.

        Args:
            physical_name: Physical variable name
            scaled_symbol: Scaled symbolic variable

        Returns:
            Physical symbolic variable
        """
        if physical_name not in self._scaling_factors:
            raise ValueError(f"No scaling factors found for variable '{physical_name}'")

        factors = self._scaling_factors[physical_name]

        if np.isclose(factors.v, 0):
            raise ValueError(
                f"Cannot create physical symbol - scaling factor v=0 for {physical_name}"
            )

        # Physical = (scaled - r) / v
        physical_symbol = (scaled_symbol - factors.r) / factors.v
        return physical_symbol

    def scale_trajectories(
        self,
        trajectories: Sequence[FloatMatrix],
        variable_names: list[str],
    ) -> list[FloatMatrix]:
        """
        Scale trajectory arrays using stored scaling factors.

        Args:
            trajectories: Physical space trajectories
            variable_names: Ordered variable names

        Returns:
            Scaled space trajectories
        """
        return scale_trajectory_arrays(trajectories, variable_names, self._scaling_factors)

    def transform_dynamics(
        self,
        physical_dynamics: dict[SymType, SymExpr],
        physical_symbols: dict[str, SymType],
        scaled_symbols: dict[str, SymType],
    ) -> dict[SymType, SymExpr]:
        """
        Transform dynamics to scaled space.

        Args:
            physical_dynamics: Physical space dynamics
            physical_symbols: Physical symbolic variables
            scaled_symbols: Scaled symbolic variables

        Returns:
            Scaled space dynamics
        """
        return transform_dynamics_to_scaled_space(
            physical_dynamics, physical_symbols, scaled_symbols, self._scaling_factors
        )

    def get_scaling_info_for_solution(self) -> dict[str, Any]:
        """
        Get scaling information for solution storage.

        Returns:
            Dictionary with all scaling information
        """
        return {
            "auto_scaling_enabled": True,
            "scaling_factors": {
                name: {"v": f.v, "r": f.r, "rule": f.rule}
                for name, f in self._scaling_factors.items()
            },
            "physical_to_scaled_map": self._mappings.physical_to_scaled.copy(),
            "scaled_to_physical_map": self._mappings.scaled_to_physical.copy(),
            "physical_symbols": self._mappings.physical_symbols.copy(),
        }

    def print_scaling_summary(self) -> None:
        """Print comprehensive scaling configuration summary."""
        print(f"\n{'=' * 80}")
        print("🎯 AUTO-SCALING CONFIGURATION SUMMARY")
        print(f"{'=' * 80}")

        print("✅ Auto-scaling is ENABLED")
        print(f"📊 Total variables with scaling: {len(self._scaling_factors)}")

        print("\n📋 SCALING FACTORS BY RULE:")
        rules_count = {}
        for factors in self._scaling_factors.values():
            rule = factors.rule
            rules_count[rule] = rules_count.get(rule, 0) + 1

        for rule, count in rules_count.items():
            print(f"  {rule}: {count} variables")

        print("\n📐 DETAILED SCALING FACTORS:")
        print(f"{'Variable':<15} | {'Rule':<30} | {'v (scale)':<12} | {'r (shift)':<12}")
        print(f"{'-' * 15}-+-{'-' * 30}-+-{'-' * 12}-+-{'-' * 12}")

        for var_name, factors in sorted(self._scaling_factors.items()):
            print(f"{var_name:<15} | {factors.rule:<30} | {factors.v:<12.3e} | {factors.r:<12.3e}")

        print("\n🔗 VARIABLE MAPPINGS:")
        print(f"{'Physical':<15} ↔ {'Scaled'}")
        print(f"{'-' * 15}---{'-' * 15}")
        for phys, scaled in sorted(self._mappings.physical_to_scaled.items()):
            print(f"{phys:<15} ↔ {scaled}")

        print(f"{'=' * 80}")
