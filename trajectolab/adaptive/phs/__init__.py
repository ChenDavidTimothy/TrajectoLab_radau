"""
Provides the PHS (P-refinement, H-refinement, S-refinement) adaptive algorithm - SIMPLIFIED.
Updated to work with unified storage system.
"""

from trajectolab.adaptive.phs.algorithm import solve_phs_adaptive_internal
from trajectolab.adaptive.phs.data_structures import AdaptiveParameters


__all__ = ["AdaptiveParameters", "solve_phs_adaptive_internal"]
