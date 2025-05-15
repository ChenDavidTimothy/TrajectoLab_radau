from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Dict, Any

class AdaptiveBase(ABC):    
    def __init__(self, initial_guess=None):
        self.initial_guess = initial_guess
    
    @abstractmethod
    def run(self, problem, legacy_problem, initial_solution=None):
        pass

class FixedMesh(AdaptiveBase):    
    def __init__(self, polynomial_degrees=None, mesh_points=None, initial_guess=None):
        super().__init__(initial_guess)  # Call parent constructor
        self.polynomial_degrees = polynomial_degrees or [4]
        
        if mesh_points is None:
            self.mesh_points = np.linspace(-1, 1, len(self.polynomial_degrees) + 1)
        else:
            self.mesh_points = np.array(mesh_points)
            
        # Validate mesh
        if len(self.polynomial_degrees) != len(self.mesh_points) - 1:
            raise ValueError("Number of polynomial degrees must be one less than number of mesh points.")
        if not np.isclose(self.mesh_points[0], -1.0) or not np.isclose(self.mesh_points[-1], 1.0):
            raise ValueError("Mesh points must start at -1.0 and end at 1.0.")
        if not np.all(np.diff(self.mesh_points) > 0):
            raise ValueError("Mesh points must be strictly increasing.")
    
    def run(self, problem, legacy_problem, initial_solution=None):
        from trajectolab.direct_solver import solve_single_phase_radau_collocation
        
        # Update legacy problem with our mesh configuration
        legacy_problem.collocation_points_per_interval = self.polynomial_degrees
        legacy_problem.global_normalized_mesh_nodes = self.mesh_points
        
        # Apply initial guess if provided
        if self.initial_guess is not None:
            legacy_problem.initial_guess = self.initial_guess
        
        # Solve the problem
        return solve_single_phase_radau_collocation(legacy_problem)