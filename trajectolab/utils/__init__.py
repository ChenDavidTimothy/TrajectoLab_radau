"""
Utility functions and classes for TrajectoLab.
OPTIMIZED: Includes high-performance caching and memory management.
"""

from .casadi_utils import convert_casadi_to_numpy, extract_casadi_value, validate_casadi_expression
from .expression_cache import CasADiExpressionCache, get_global_expression_cache
from .memory_pool import InterpolationBufferPool, create_buffer_context, get_global_buffer_pool


__all__ = [
    "CasADiExpressionCache",
    "InterpolationBufferPool",
    "convert_casadi_to_numpy",
    "create_buffer_context",
    "extract_casadi_value",
    "get_global_buffer_pool",
    "get_global_expression_cache",
    "validate_casadi_expression",
]
