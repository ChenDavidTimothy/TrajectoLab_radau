"""
Base classes and interfaces for adaptive methods in TrajectoLab.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from ...core.problem import ProblemDefinition, Solution

# Type alias for parameters of any adaptive method
AdaptiveParams = Dict[str, Any]

class AdaptiveMethod(ABC):
    """Base class for all adaptive mesh refinement methods."""
    
    def __init__(self, problem_definition: ProblemDefinition, params: AdaptiveParams):
        """
        Initialize the adaptive method.
        
        Args:
            problem_definition: The optimal control problem definition
            params: Method-specific parameters
        """
        self.problem_definition = problem_definition
        self.params = params
    
    @abstractmethod
    def solve(self) -> Solution:
        """
        Solve the optimal control problem using this adaptive method.
        
        Returns:
            Solution dictionary
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_default_params(cls) -> AdaptiveParams:
        """
        Get default parameters for this adaptive method.
        
        Returns:
            Dictionary of default parameters
        """
        pass