"""
Problem definition types and utilities for TrajectoLab.
"""

from typing import Dict, Any, List, Callable, Union, Optional
import numpy as np
import casadi as ca

# Define type aliases from the original rpm_solver.py
ProblemDefinition = Dict[str, Any]
Solution = Dict[str, Any]

# No additional implementation - these are just type aliases