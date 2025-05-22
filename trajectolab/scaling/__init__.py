"""
Proper optimal control scaling module following scale.txt methodology.

This module implements the correct scaling approach:
- Rule 2: Variable scaling ỹ = V_y * y + r_y
- Rule 3: ODE defect scaling W_f = V_y
- Rule 4: Constraint scaling W_g for unit row norms
- Rule 5: Objective scaling w_0 = 1/ϖ (multiplicative)

Key principle: Separate variable scaling from objective/constraint scaling.
"""

# Import legacy functions for backward compatibility
from ..scaling.core_scale import (
    AutoScalingManager,
    ConstraintScaling,
    ObjectiveScaling,
    ScalingFactors,
)


__all__ = [
    # New proper scaling classes
    "AutoScalingManager",
    "ConstraintScaling",
    "ObjectiveScaling",
    "ScalingFactors",
]
