from abc import ABC, abstractmethod

import numpy as np

from ..direct_solver import OptimalControlSolution


class AdaptiveBase(ABC):
    """
    Base class for adaptive mesh refinement algorithms.
    """

    def __init__(self, initial_guess=None):
        self.initial_guess = initial_guess

    @abstractmethod
    def run(self, problem, initial_solution=None) -> OptimalControlSolution:
        """
        Run the adaptive mesh refinement algorithm.

        Args:
            problem: The optimal control problem
            initial_solution: Optional initial solution

        Returns:
            The optimal control solution
        """
        pass


class FixedMesh(AdaptiveBase):
    """
    Fixed mesh solver with specified polynomial degrees and mesh points.
    """

    def __init__(self, polynomial_degrees=None, mesh_points=None, initial_guess=None):
        super().__init__(initial_guess)
        self.polynomial_degrees = polynomial_degrees or [4]

        if mesh_points is None:
            self.mesh_points = np.linspace(-1, 1, len(self.polynomial_degrees) + 1)
        else:
            self.mesh_points = np.array(mesh_points)

        # Validate mesh
        if len(self.polynomial_degrees) != len(self.mesh_points) - 1:
            raise ValueError(
                "Number of polynomial degrees must be one less than number of mesh points."
            )
        if not np.isclose(self.mesh_points[0], -1.0) or not np.isclose(self.mesh_points[-1], 1.0):
            raise ValueError("Mesh points must start at -1.0 and end at 1.0.")
        if not np.all(np.diff(self.mesh_points) > 0):
            raise ValueError("Mesh points must be strictly increasing.")

    def run(self, problem, initial_solution=None) -> OptimalControlSolution:
        """
        Run the fixed mesh solver.

        Args:
            problem: The optimal control problem
            initial_solution: Optional initial solution

        Returns:
            The optimal control solution
        """
        from trajectolab.direct_solver import solve_single_phase_radau_collocation

        # Update problem with mesh configuration
        problem.collocation_points_per_interval = self.polynomial_degrees
        problem.global_normalized_mesh_nodes = self.mesh_points

        # Apply initial guess if provided
        if self.initial_guess is not None:
            problem.initial_guess = self.initial_guess

        # Solve the problem
        return solve_single_phase_radau_collocation(problem)
