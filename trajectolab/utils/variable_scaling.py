"""
Variable scaling utilities for optimal control problems.

Implements the scaling formulation from equation (4.243):
[tilde_y; tilde_u] = [V_y 0; 0 V_u] * [y; u] + [r_y; r_u]

This module provides functions for computing, applying, and reversing
variable scaling transformations to improve NLP numerical conditioning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np

from ..tl_types import FloatArray, FloatMatrix, ProblemProtocol
from .constants import ZERO_TOLERANCE


logger = logging.getLogger(__name__)

# Type aliases for scaling
_ScaleWeight: TypeAlias = float
_Shift: TypeAlias = float
_VariableRange: TypeAlias = tuple[float, float]


@dataclass
class VariableScalingInfo:
    """Scaling information for a single variable."""

    scale_weight: _ScaleWeight
    """Scale weight v_k from equation (4.250): v_k = 1/(y_U,k - y_L,k)"""

    shift: _Shift
    """Shift r_k from equation (4.251): r_k = 1/2 - y_U,k/(y_U,k - y_L,k)"""

    lower_bound: float | None = None
    """Original lower bound y_L,k (if available)"""

    upper_bound: float | None = None
    """Original upper bound y_U,k (if available)"""

    estimated_range: _VariableRange | None = None
    """Estimated range from initial guess (if bounds unavailable)"""

    scaling_method: str = "default"
    """Method used: 'bounds', 'initial_guess', or 'default'"""


@dataclass
class ProblemScalingInfo:
    """Complete scaling information for an optimal control problem."""

    state_scaling: dict[str, VariableScalingInfo] = field(default_factory=dict)
    """Scaling info for each state variable"""

    control_scaling: dict[str, VariableScalingInfo] = field(default_factory=dict)
    """Scaling info for each control variable"""

    time_scaling: VariableScalingInfo | None = None
    """Scaling info for time variables (if needed)"""

    scaling_enabled: bool = True
    """Whether scaling is enabled for this problem"""


def compute_scaling_from_bounds(
    lower_bound: float, upper_bound: float
) -> tuple[_ScaleWeight, _Shift]:
    """
    Compute scale weight and shift from variable bounds.

    Implements equations (4.250) and (4.251):
    v_k = 1/(y_U,k - y_L,k)
    r_k = 1/2 - y_U,k/(y_U,k - y_L,k)

    Args:
        lower_bound: Lower bound y_L,k
        upper_bound: Upper bound y_U,k

    Returns:
        Tuple of (scale_weight, shift)

    Raises:
        ValueError: If bounds are invalid
    """
    if upper_bound <= lower_bound:
        raise ValueError(
            f"Upper bound ({upper_bound}) must be greater than lower bound ({lower_bound})"
        )

    range_width = upper_bound - lower_bound

    if abs(range_width) < ZERO_TOLERANCE:
        raise ValueError(
            f"Variable range too small: {range_width}. Cannot compute meaningful scaling."
        )

    # Equation (4.250)
    scale_weight = 1.0 / range_width

    # Equation (4.251)
    shift = 0.5 - upper_bound / range_width

    return scale_weight, shift


def compute_scaling_from_range(estimated_range: _VariableRange) -> tuple[_ScaleWeight, _Shift]:
    """
    Compute scaling from estimated variable range.

    Used when explicit bounds are not available but we have
    estimates from initial guesses (Rule 2, step 1b).

    Args:
        estimated_range: Tuple of (min_estimate, max_estimate)

    Returns:
        Tuple of (scale_weight, shift)

    Raises:
        ValueError: If range estimate is invalid
    """
    min_val, max_val = estimated_range
    return compute_scaling_from_bounds(min_val, max_val)


def estimate_range_from_initial_guess(
    initial_guess_trajectory: FloatArray, safety_factor: float = 1.2
) -> _VariableRange:
    """
    Estimate variable range from initial guess trajectory.

    Used for Rule 2, step 1b when bounds are not available.

    Args:
        initial_guess_trajectory: Initial guess values for the variable
        safety_factor: Factor to expand estimated range for safety

    Returns:
        Estimated (min, max) range

    Raises:
        ValueError: If trajectory is empty or contains invalid values
    """
    if initial_guess_trajectory.size == 0:
        raise ValueError("Initial guess trajectory is empty")

    if np.any(np.isnan(initial_guess_trajectory)) or np.any(np.isinf(initial_guess_trajectory)):
        raise ValueError("Initial guess trajectory contains NaN or infinite values")

    min_val = float(np.min(initial_guess_trajectory))
    max_val = float(np.max(initial_guess_trajectory))

    # If all values are the same, create a small range around the value
    if abs(max_val - min_val) < ZERO_TOLERANCE:
        center = (max_val + min_val) / 2.0
        if abs(center) < ZERO_TOLERANCE:
            # Value is near zero, use symmetric range
            min_val = -1.0
            max_val = 1.0
        else:
            # Value is non-zero, use percentage-based range
            range_half = abs(center) * 0.1  # 10% of center value
            min_val = center - range_half
            max_val = center + range_half

    # Apply safety factor to expand range
    range_center = (max_val + min_val) / 2.0
    range_half = (max_val - min_val) / 2.0 * safety_factor

    return (range_center - range_half, range_center + range_half)


def create_default_scaling() -> VariableScalingInfo:
    """
    Create default scaling (no scaling applied).

    Used for Rule 2, step 3 when neither bounds nor initial guess
    provide useful information.

    Returns:
        VariableScalingInfo with identity scaling
    """
    return VariableScalingInfo(
        scale_weight=1.0,
        shift=0.0,
        lower_bound=None,
        upper_bound=None,
        estimated_range=None,
        scaling_method="default",
    )


def apply_forward_scaling(
    variables: FloatArray, scale_weights: FloatArray, shifts: FloatArray
) -> FloatArray:
    """
    Apply forward scaling transformation: tilde_y = V * y + r

    Implements the transformation from equation (4.243).

    Args:
        variables: Original variables y
        scale_weights: Diagonal elements of V matrix
        shifts: Shift vector r

    Returns:
        Scaled variables tilde_y

    Raises:
        ValueError: If array dimensions don't match
    """
    if variables.shape != scale_weights.shape or variables.shape != shifts.shape:
        raise ValueError(
            f"Array shape mismatch: variables={variables.shape}, "
            f"scale_weights={scale_weights.shape}, shifts={shifts.shape}"
        )

    return cast(FloatArray, scale_weights * variables + shifts)


def apply_reverse_scaling(
    scaled_variables: FloatArray, scale_weights: FloatArray, shifts: FloatArray
) -> FloatArray:
    """
    Apply reverse scaling transformation: y = V^(-1) * (tilde_y - r)

    Inverts the transformation from equation (4.243).

    Args:
        scaled_variables: Scaled variables tilde_y
        scale_weights: Diagonal elements of V matrix
        shifts: Shift vector r

    Returns:
        Original variables y

    Raises:
        ValueError: If array dimensions don't match or scale weights are zero
    """
    if scaled_variables.shape != scale_weights.shape or scaled_variables.shape != shifts.shape:
        raise ValueError(
            f"Array shape mismatch: scaled_variables={scaled_variables.shape}, "
            f"scale_weights={scale_weights.shape}, shifts={shifts.shape}"
        )

    if np.any(np.abs(scale_weights) < ZERO_TOLERANCE):
        raise ValueError("Scale weights contain values too close to zero")

    return cast(FloatArray, (scaled_variables - shifts) / scale_weights)


def create_scaling_matrices(
    scaling_info: ProblemScalingInfo, problem: ProblemProtocol
) -> tuple[FloatMatrix, FloatArray]:
    """
    Create block diagonal scaling matrix V and combined shift vector r.

    Implements the block structure from equation (4.243):
    V = [V_y  0 ]    r = [r_y]
        [0   V_u]        [r_u]

    Args:
        scaling_info: Complete problem scaling information
        problem: Problem definition for variable ordering

    Returns:
        Tuple of (V_matrix, r_vector) where:
        - V_matrix: Block diagonal scaling matrix
        - r_vector: Combined shift vector
    """
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    total_vars = num_states + num_controls

    # Initialize matrices
    V_matrix = np.eye(total_vars, dtype=np.float64)
    r_vector = np.zeros(total_vars, dtype=np.float64)

    # Fill state scaling (first num_states rows/columns)
    state_names = sorted(problem._states.keys(), key=lambda n: problem._states[n]["index"])
    for i, name in enumerate(state_names):
        if name in scaling_info.state_scaling:
            scaling = scaling_info.state_scaling[name]
            V_matrix[i, i] = scaling.scale_weight
            r_vector[i] = scaling.shift

    # Fill control scaling (next num_controls rows/columns)
    control_names = sorted(problem._controls.keys(), key=lambda n: problem._controls[n]["index"])
    for i, name in enumerate(control_names):
        idx = num_states + i
        if name in scaling_info.control_scaling:
            scaling = scaling_info.control_scaling[name]
            V_matrix[idx, idx] = scaling.scale_weight
            r_vector[idx] = scaling.shift

    return cast(FloatMatrix, V_matrix), cast(FloatArray, r_vector)


def validate_scaling_info(scaling_info: VariableScalingInfo) -> None:
    """
    Validate scaling information for consistency and numerical stability.

    Args:
        scaling_info: Scaling information to validate

    Raises:
        ValueError: If scaling info is invalid
    """
    if abs(scaling_info.scale_weight) < ZERO_TOLERANCE:
        raise ValueError(
            f"Scale weight too small: {scaling_info.scale_weight}. "
            f"This would cause numerical instability."
        )

    if not np.isfinite(scaling_info.scale_weight):
        raise ValueError(f"Scale weight is not finite: {scaling_info.scale_weight}")

    if not np.isfinite(scaling_info.shift):
        raise ValueError(f"Shift is not finite: {scaling_info.shift}")

    # Check that bounds are consistent if provided
    if scaling_info.lower_bound is not None and scaling_info.upper_bound is not None:
        if scaling_info.upper_bound <= scaling_info.lower_bound:
            raise ValueError(
                f"Invalid bounds: upper ({scaling_info.upper_bound}) <= "
                f"lower ({scaling_info.lower_bound})"
            )


def get_scaled_variable_bounds(scaling_info: VariableScalingInfo) -> tuple[float, float]:
    """
    Compute the bounds for a scaled variable.

    Given original bounds [y_L, y_U], computes scaled bounds:
    [tilde_y_L, tilde_y_U] = [V*y_L + r, V*y_U + r]

    With proper scaling, these should be approximately [-0.5, 0.5].

    Args:
        scaling_info: Scaling information for the variable

    Returns:
        Tuple of (scaled_lower_bound, scaled_upper_bound)

    Raises:
        ValueError: If bounds are not available
    """
    if scaling_info.lower_bound is None or scaling_info.upper_bound is None:
        raise ValueError("Cannot compute scaled bounds: original bounds not available")

    lower_scaled = scaling_info.scale_weight * scaling_info.lower_bound + scaling_info.shift
    upper_scaled = scaling_info.scale_weight * scaling_info.upper_bound + scaling_info.shift

    return (lower_scaled, upper_scaled)


def log_scaling_summary(scaling_info: ProblemScalingInfo) -> None:
    """
    Log a summary of the scaling configuration.

    Args:
        scaling_info: Complete problem scaling information
    """
    if not scaling_info.scaling_enabled:
        logger.info("Variable scaling is disabled")
        return

    logger.info("Variable scaling summary:")

    # State variables
    logger.info(f"  State variables ({len(scaling_info.state_scaling)}):")
    for name, scaling in scaling_info.state_scaling.items():
        scaled_range = "unknown"
        if scaling.lower_bound is not None and scaling.upper_bound is not None:
            try:
                lower_scaled, upper_scaled = get_scaled_variable_bounds(scaling)
                scaled_range = f"[{lower_scaled:.3f}, {upper_scaled:.3f}]"
            except ValueError:
                pass

        logger.info(
            f"    {name}: method={scaling.scaling_method}, "
            f"weight={scaling.scale_weight:.3e}, shift={scaling.shift:.3f}, "
            f"scaled_range={scaled_range}"
        )

    # Control variables
    logger.info(f"  Control variables ({len(scaling_info.control_scaling)}):")
    for name, scaling in scaling_info.control_scaling.items():
        scaled_range = "unknown"
        if scaling.lower_bound is not None and scaling.upper_bound is not None:
            try:
                lower_scaled, upper_scaled = get_scaled_variable_bounds(scaling)
                scaled_range = f"[{lower_scaled:.3f}, {upper_scaled:.3f}]"
            except ValueError:
                pass

        logger.info(
            f"    {name}: method={scaling.scaling_method}, "
            f"weight={scaling.scale_weight:.3e}, shift={scaling.shift:.3f}, "
            f"scaled_range={scaled_range}"
        )
