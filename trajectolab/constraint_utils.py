"""
Constraint application utilities.
"""

from .tl_types import CasadiOpti, EventConstraint, PathConstraint


def apply_constraint(opti: CasadiOpti, constraint: PathConstraint | EventConstraint) -> None:
    """Apply a constraint to the optimization problem."""
    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)
