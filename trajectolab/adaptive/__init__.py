"""
Adaptive mesh refinement for TrajectoLab.

This module provides adaptive mesh refinement capabilities
for improving solution accuracy and efficiency.
"""

from typing import Optional, Dict, Any, Union

from .base.base import AdaptiveMethod, AdaptiveParams
from .phs.method import PHSMethod, run_phs_adaptive_mesh_refinement

def create_method(method_name: str, problem_definition: Dict[str, Any], 
                 params: Optional[Dict[str, Any]] = None) -> AdaptiveMethod:
    """
    Factory function to create an adaptive method.
    
    Args:
        method_name: Name of the adaptive method ('phs' currently supported)
        problem_definition: The optimal control problem definition
        params: Method-specific parameters, or None to use defaults
        
    Returns:
        An initialized adaptive method instance
        
    Raises:
        ValueError: If the method_name is not recognized
    """
    if method_name.lower() == 'phs':
        return PHSMethod(problem_definition, params)
    else:
        raise ValueError(f"Unknown adaptive method: {method_name}")

__all__ = [
    'AdaptiveMethod',
    'AdaptiveParams',
    'PHSMethod',
    'create_method',
    'run_phs_adaptive_mesh_refinement'
]