"""
Streamlined PHS adaptive algorithm for multiphase problems.
BLOAT ELIMINATED: Simplified exports, removed over-engineered components.
"""

from trajectolab.adaptive.phs.algorithm import solve_multiphase_phs_adaptive_internal
from trajectolab.adaptive.phs.data_structures import AdaptiveParameters


__all__ = ["AdaptiveParameters", "solve_multiphase_phs_adaptive_internal"]
