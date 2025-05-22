"""
Enhanced auto-scaling for optimal control problems following scale.txt methodology.

This implements the correct scaling approach:
- Rule 2: Variable scaling ỹ = V_y * y + r_y
- Rule 3: ODE defect scaling W_f = V_y
- Rule 4: Constraint scaling W_g for unit row norms
- Rule 5: Objective scaling w_0 = 1/ϖ (multiplicative)

Key principle: Variable scaling is separate from objective/constraint scaling.
Variables are scaled for NLP conditioning, but expressions keep their structure.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np

from ..tl_types import FloatArray, FloatMatrix, SymType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingFactors:
    """Variable scaling transformation parameters following Rule 2."""

    v: float  # Scale factor: scaled = v * physical + r
    r: float  # Shift factor
    rule: str  # Rule applied ("2.1.a", "2.1.b", "2.4")

    def __post_init__(self) -> None:
        """Validate scaling factors for safety-critical use."""
        if not np.isfinite(self.v) or not np.isfinite(self.r):
            raise ValueError(f"Non-finite scaling factors: v={self.v}, r={self.r}")
        if np.abs(self.v) < 1e-15:
            raise ValueError(f"Scaling factor v={self.v} too small")
        if np.abs(self.v) > 1e15:
            raise ValueError(f"Scaling factor v={self.v} too large")


@dataclass
class ObjectiveScaling:
    """Objective scaling parameters following Rule 5."""

    w_0: float = 1.0  # Multiplicative objective scaling factor
    computed_from_hessian: bool = False
    gerschgorin_omega: float | None = None

    def apply_to_objective(self, objective_value: float) -> float:
        """Apply objective scaling: return w_0 * J."""
        return self.w_0 * objective_value

    def unscale_objective(self, scaled_objective: float) -> float:
        """Unscale objective: return J from w_0 * J."""
        return scaled_objective / self.w_0 if self.w_0 != 0 else scaled_objective


@dataclass
class ConstraintScaling:
    """Constraint scaling parameters following Rules 3 & 4."""

    W_f: dict[str, float] = field(default_factory=dict)  # ODE defect scaling (Rule 3)
    W_g: dict[str, float] = field(default_factory=dict)  # Path constraint scaling (Rule 4)


class AutoScalingManager:
    """
    Proper auto-scaling manager for optimal control problems.

    Implements scale.txt methodology with clear separation between:
    - Variable scaling (for NLP conditioning)
    - Objective scaling (multiplicative factors)
    - Constraint scaling (W_f, W_g factors)
    """

    def __init__(self) -> None:
        """Initialize proper auto-scaling manager."""
        # Variable scaling (Rule 2)
        self._variable_scaling: dict[str, ScalingFactors] = {}

        # Symbol management - KEY CHANGE: Keep original and scaled separate
        self._original_physical_symbols: dict[str, SymType] = {}  # Never corrupted
        self._scaled_nlp_symbols: dict[str, SymType] = {}  # For optimization
        self._physical_to_scaled_names: dict[str, str] = {}

        # Objective scaling (Rule 5)
        self._objective_scaling = ObjectiveScaling()

        # Constraint scaling (Rules 3 & 4)
        self._constraint_scaling = ConstraintScaling()

        # Initial guess ranges for determining variable scaling
        self._variable_ranges: dict[str, dict[str, float | None]] = {}

    # Properties
    @property
    def variable_scaling_factors(self) -> dict[str, ScalingFactors]:
        """Get variable scaling factors."""
        return self._variable_scaling.copy()

    @property
    def original_physical_symbols(self) -> dict[str, SymType]:
        """Get original physical symbols (never corrupted)."""
        return self._original_physical_symbols.copy()

    @property
    def scaled_nlp_symbols(self) -> dict[str, SymType]:
        """Get scaled symbols for NLP optimization."""
        return self._scaled_nlp_symbols.copy()

    @property
    def objective_scaling_factor(self) -> float:
        """Get objective scaling factor w_0."""
        return self._objective_scaling.w_0

    @property
    def ode_defect_scaling(self) -> dict[str, float]:
        """Get ODE defect scaling factors W_f."""
        return self._constraint_scaling.W_f.copy()

    # Variable range tracking
    def update_variable_range(
        self,
        var_name: str,
        initial: float | None = None,
        final: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """Update variable range information for scaling determination."""
        if var_name not in self._variable_ranges:
            self._variable_ranges[var_name] = {"min": None, "max": None}

        range_info = self._variable_ranges[var_name]
        values = [v for v in [initial, final, lower, upper] if v is not None]

        if values:
            current_min, current_max = range_info["min"], range_info["max"]
            new_min, new_max = min(values), max(values)
            range_info["min"] = new_min if current_min is None else min(current_min, new_min)
            range_info["max"] = new_max if current_max is None else max(current_max, new_max)

    # Variable scaling setup (Rule 2)
    def setup_variable_scaling(
        self,
        var_name: str,
        explicit_lower: float | None = None,
        explicit_upper: float | None = None,
        scale_guide_lower: float | None = None,
        scale_guide_upper: float | None = None,
    ) -> ScalingFactors:
        """
        Setup variable scaling following Rule 2 from scale.txt.

        Variable scaling: ỹ = V_y * y + r_y
        Goal: Scale variables to approximately [-0.5, 0.5] range.
        """
        # Determine bounds for scaling
        bounds_lower = scale_guide_lower if scale_guide_lower is not None else explicit_lower
        bounds_upper = scale_guide_upper if scale_guide_upper is not None else explicit_upper

        # Get initial guess range
        range_info = self._variable_ranges.get(var_name, {})
        guess_min, guess_max = range_info.get("min"), range_info.get("max")

        # Apply Rule 2 scaling logic
        factors = self._compute_variable_scaling_factors(
            var_name, bounds_lower, bounds_upper, guess_min, guess_max
        )

        # Store scaling factors
        self._variable_scaling[var_name] = factors

        logger.info(
            f"Variable '{var_name}' scaling: {factors.rule}, v={factors.v:.3e}, r={factors.r:.3f}"
        )
        return factors

    def _compute_variable_scaling_factors(
        self,
        var_name: str,
        explicit_lower: float | None,
        explicit_upper: float | None,
        guess_min: float | None,
        guess_max: float | None,
    ) -> ScalingFactors:
        """Compute variable scaling factors using Rule 2 hierarchy."""
        # Rule 2.1.a: Use explicit bounds if available
        if (
            explicit_lower is not None
            and explicit_upper is not None
            and not np.isclose(explicit_upper, explicit_lower)
        ):
            range_val = explicit_upper - explicit_lower
            v = 1.0 / range_val  # Equation 4.250
            r = 0.5 - explicit_upper / range_val  # Equation 4.251
            return ScalingFactors(v=v, r=r, rule="2.1.a (Explicit Bounds)")

        # Rule 2.1.b: Use initial guess range if available
        elif (
            guess_min is not None and guess_max is not None and not np.isclose(guess_max, guess_min)
        ):
            range_val = guess_max - guess_min
            v = 1.0 / range_val
            r = 0.5 - guess_max / range_val
            return ScalingFactors(v=v, r=r, rule="2.1.b (Initial Guess Range)")

        # Rule 2.4: Default (no scaling)
        else:
            return ScalingFactors(v=1.0, r=0.0, rule="2.4 (Default)")

    # Symbol management - KEY IMPROVEMENT
    def create_variable_symbols(
        self,
        physical_name: str,
        create_scaled_symbol_func,
        scaled_bounds: dict[str, float | None],
    ) -> SymType:
        """
        Create original physical symbol and scaled NLP symbol.

        Returns the ORIGINAL physical symbol for user expressions.
        Stores the scaled symbol internally for NLP optimization.

        This is the key fix: user gets original symbol, NLP gets scaled symbol.
        """
        # Create original physical symbol (independent, never corrupted)
        original_symbol = ca.MX.sym(physical_name, 1)
        self._original_physical_symbols[physical_name] = original_symbol

        # Create scaled NLP symbol for optimization
        scaled_name = f"{physical_name}_scaled"
        scaled_symbol = create_scaled_symbol_func(scaled_name, **scaled_bounds)
        self._scaled_nlp_symbols[physical_name] = scaled_symbol

        # Store name mapping
        self._physical_to_scaled_names[physical_name] = scaled_name

        # Return ORIGINAL symbol for user expressions
        return original_symbol

    def get_scaled_variable_bounds(self, var_name: str, physical_bounds: dict) -> dict:
        """Get bounds for scaled NLP variable."""
        if var_name not in self._variable_scaling:
            return physical_bounds

        factors = self._variable_scaling[var_name]

        # Transform bounds using scaling
        scaled_bounds = {}
        for key, value in physical_bounds.items():
            if value is not None and key in ["initial", "final", "lower", "upper"]:
                scaled_bounds[key] = factors.v * value + factors.r
            else:
                scaled_bounds[key] = value

        # For variables with explicit scaling, use normalized bounds
        if factors.rule.startswith("2.1"):
            if "lower" in scaled_bounds:
                scaled_bounds["lower"] = -0.5
            if "upper" in scaled_bounds:
                scaled_bounds["upper"] = 0.5

        return scaled_bounds

    # Constraint scaling setup (Rules 3 & 4)
    def setup_ode_defect_scaling(self) -> None:
        """Setup ODE defect scaling following Rule 3: W_f = V_y."""
        self._constraint_scaling.W_f.clear()
        for var_name, factors in self._variable_scaling.items():
            self._constraint_scaling.W_f[var_name] = factors.v
            logger.debug(f"ODE defect scaling W_f[{var_name}] = {factors.v}")

    def setup_path_constraint_scaling(self, constraint_jacobian: FloatMatrix | None = None) -> None:
        """Setup path constraint scaling following Rule 4."""
        # Simplified implementation - in full version, compute from Jacobian row norms
        # For now, use default scaling
        self._constraint_scaling.W_g.clear()
        if constraint_jacobian is not None:
            # TODO: Implement W_g = diag(1/||row_i||) for unit row norms
            pass
        logger.debug("Path constraint scaling W_g set to default")

    # Objective scaling (Rule 5)
    def compute_objective_scaling(self, hessian_at_solution: FloatArray | None = None) -> float:
        """
        Compute objective scaling following Rule 5: w_0 = 1/ϖ.

        Args:
            hessian_at_solution: Hessian matrix for Gerschgorin bounds computation

        Returns:
            Objective scaling factor w_0
        """
        if hessian_at_solution is not None and hessian_at_solution.size > 0:
            omega = self._compute_gerschgorin_omega(hessian_at_solution)
            w_0 = 1.0 / omega if omega > 1e-12 else 1.0

            self._objective_scaling = ObjectiveScaling(
                w_0=w_0, computed_from_hessian=True, gerschgorin_omega=omega
            )

            logger.info(f"Objective scaling: w_0 = {w_0:.3e} (ϖ = {omega:.3e})")
        else:
            self._objective_scaling = ObjectiveScaling(w_0=1.0)
            logger.info("Objective scaling: w_0 = 1.0 (default)")

        return self._objective_scaling.w_0

    def _compute_gerschgorin_omega(self, hessian: FloatArray) -> float:
        """Compute ϖ = max{|σ_L|, |σ_U|} using Gerschgorin bounds."""
        if hessian.size == 0:
            return 1.0

        # Gerschgorin circle theorem bounds
        diag_elements = np.diag(hessian)
        off_diag_sums = np.sum(np.abs(hessian), axis=1) - np.abs(diag_elements)

        sigma_L = np.min(diag_elements - off_diag_sums)  # Lower bound estimate
        sigma_U = np.max(diag_elements + off_diag_sums)  # Upper bound estimate

        return max(abs(sigma_L), abs(sigma_U))

    # Trajectory scaling
    def scale_trajectory_arrays(
        self,
        trajectories: Sequence[FloatMatrix],
        variable_names: list[str],
    ) -> list[FloatMatrix]:
        """Scale trajectory arrays from physical to scaled space."""
        if not trajectories:
            return []

        scaled_trajectories = []
        for traj_array in trajectories:
            scaled_array = np.zeros_like(traj_array, dtype=np.float64)
            for i, var_name in enumerate(variable_names):
                if var_name in self._variable_scaling:
                    factors = self._variable_scaling[var_name]
                    scaled_array[i, :] = factors.v * traj_array[i, :] + factors.r
                else:
                    scaled_array[i, :] = traj_array[i, :]  # No scaling
            scaled_trajectories.append(scaled_array)
        return scaled_trajectories

    def unscale_trajectory_arrays(
        self,
        trajectories: Sequence[FloatMatrix],
        variable_names: list[str],
    ) -> list[FloatMatrix]:
        """Unscale trajectory arrays from scaled to physical space."""
        if not trajectories:
            return []

        unscaled_trajectories = []
        for traj_array in trajectories:
            unscaled_array = np.zeros_like(traj_array, dtype=np.float64)
            for i, var_name in enumerate(variable_names):
                if var_name in self._variable_scaling:
                    factors = self._variable_scaling[var_name]
                    unscaled_array[i, :] = (traj_array[i, :] - factors.r) / factors.v
                else:
                    unscaled_array[i, :] = traj_array[i, :]  # No scaling
            unscaled_trajectories.append(unscaled_array)
        return unscaled_trajectories

    def unscale_objective_value(self, scaled_objective: float) -> float:
        """Unscale objective value from w_0 * J to J."""
        return self._objective_scaling.unscale_objective(scaled_objective)

    # Transformation for solver interface
    def create_physical_to_scaled_substitution_map(
        self,
        scaled_state_vector: CasadiMX,
        scaled_control_vector: CasadiMX,
        state_names: list[str],
        control_names: list[str],
    ) -> dict[SymType, CasadiMX]:
        """
        Create substitution map from original physical symbols to scaled variable expressions.

        This is used at the solver interface level to transform expressions.
        Key: Physical symbol -> (scaled_var - r) / v
        """
        substitution_map = {}

        # Map state symbols
        for i, var_name in enumerate(state_names):
            if var_name in self._original_physical_symbols and var_name in self._variable_scaling:
                original_symbol = self._original_physical_symbols[var_name]
                factors = self._variable_scaling[var_name]
                # Physical = (scaled - r) / v
                physical_expr = (scaled_state_vector[i] - factors.r) / factors.v
                substitution_map[original_symbol] = physical_expr

        # Map control symbols
        for i, var_name in enumerate(control_names):
            if var_name in self._original_physical_symbols and var_name in self._variable_scaling:
                original_symbol = self._original_physical_symbols[var_name]
                factors = self._variable_scaling[var_name]
                # Physical = (scaled - r) / v
                physical_expr = (scaled_control_vector[i] - factors.r) / factors.v
                substitution_map[original_symbol] = physical_expr

        return substitution_map

    # Information access
    def get_scaling_info(self) -> dict[str, Any]:
        """Get comprehensive scaling information."""
        return {
            "auto_scaling_enabled": True,
            "variable_scaling_factors": {
                name: {"v": f.v, "r": f.r, "rule": f.rule}
                for name, f in self._variable_scaling.items()
            },
            "objective_scaling": {
                "w_0": self._objective_scaling.w_0,
                "computed_from_hessian": self._objective_scaling.computed_from_hessian,
                "gerschgorin_omega": self._objective_scaling.gerschgorin_omega,
            },
            "constraint_scaling": {
                "W_f": self._constraint_scaling.W_f.copy(),
                "W_g": self._constraint_scaling.W_g.copy(),
            },
            "original_physical_symbols": self._original_physical_symbols.copy(),
            "scaled_nlp_symbols": self._scaled_nlp_symbols.copy(),
            "physical_to_scaled_names": self._physical_to_scaled_names.copy(),
        }

    def print_scaling_summary(self) -> None:
        """Print comprehensive scaling configuration summary."""
        print(f"\n{'=' * 80}")
        print("🎯 PROPER OPTIMAL CONTROL SCALING SUMMARY")
        print(f"{'=' * 80}")
        print("✅ Following scale.txt methodology")

        print(f"\n📊 VARIABLE SCALING (Rule 2): {len(self._variable_scaling)} variables")
        if self._variable_scaling:
            print(f"{'Variable':<15} | {'Rule':<30} | {'v (scale)':<12} | {'r (shift)':<12}")
            print(f"{'-' * 15}-+-{'-' * 30}-+-{'-' * 12}-+-{'-' * 12}")
            for name, factors in sorted(self._variable_scaling.items()):
                print(f"{name:<15} | {factors.rule:<30} | {factors.v:<12.3e} | {factors.r:<12.3e}")

        print("\n📐 OBJECTIVE SCALING (Rule 5):")
        print(f"  w_0 = {self._objective_scaling.w_0:.3e}")
        if self._objective_scaling.computed_from_hessian:
            print(f"  ϖ (Gerschgorin) = {self._objective_scaling.gerschgorin_omega:.3e}")
        else:
            print("  Using default scaling")

        print("\n🔧 CONSTRAINT SCALING (Rules 3 & 4):")
        if self._constraint_scaling.W_f:
            print("  ODE Defect Scaling W_f (Rule 3: W_f = V_y):")
            for var, scale in self._constraint_scaling.W_f.items():
                print(f"    {var}: {scale:.3e}")

        print("\n🔗 SYMBOL SEPARATION:")
        print(f"  Original physical symbols: {len(self._original_physical_symbols)}")
        print(f"  Scaled NLP symbols: {len(self._scaled_nlp_symbols)}")
        print("  ✅ No symbol corruption - expressions use original symbols")

        print(f"{'=' * 80}")


# Utility functions for backward compatibility
def scale_values(values: FloatArray, factors: ScalingFactors) -> FloatArray:
    """Apply scaling transformation: scaled = v * physical + r."""
    if values.size == 0:
        return values
    return factors.v * values + factors.r


def unscale_values(values: FloatArray, factors: ScalingFactors) -> FloatArray:
    """Apply inverse scaling transformation: physical = (scaled - r) / v."""
    if values.size == 0:
        return values
    return (values - factors.r) / factors.v
