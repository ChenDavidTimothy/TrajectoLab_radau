"""
Provides the PHS (P-refinement, H-refinement, S-refinement) adaptive algorithm for multiphase problems.
Updated to work with unified NLP multiphase optimization.
"""

from trajectolab.adaptive.phs.algorithm import solve_multiphase_phs_adaptive_internal
from trajectolab.adaptive.phs.data_structures import AdaptiveParameters


__all__ = ["AdaptiveParameters", "solve_multiphase_phs_adaptive_internal"]
