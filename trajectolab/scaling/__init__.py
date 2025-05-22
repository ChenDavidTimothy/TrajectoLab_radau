"""
Auto-scaling utilities for TrajectoLab.

This module provides comprehensive auto-scaling functionality for optimal control problems,
transforming variables from physical space to numerically well-conditioned scaled space.

For NASA safety-critical applications, all mathematical transformations are implemented
as pure functions that can be independently verified and tested.
"""

from .core_scale import (
    AutoScalingManager,
    ScalingFactors,
    VariableMappings,
    determine_scaling_factors,
    scale_trajectory_arrays,
    scale_values,
    transform_dynamics_to_scaled_space,
    unscale_trajectory_arrays,
    unscale_values,
)


__all__ = [
    "AutoScalingManager",
    "ScalingFactors",
    "VariableMappings",
    "determine_scaling_factors",
    "scale_trajectory_arrays",
    "scale_values",
    "transform_dynamics_to_scaled_space",
    "unscale_trajectory_arrays",
    "unscale_values",
]
