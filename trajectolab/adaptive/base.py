from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np

from ..direct_solver import InitialGuess, OptimalControlSolution
from ..tl_types import FloatArray, ProblemProtocol


# Type alias for initial solutions
_InitialSolution: TypeAlias = OptimalControlSolution | None


class AdaptiveBase(ABC):
    """
    Base class for adaptive mesh refinement algorithms.
    """

    def __init__(self, initial_guess: InitialGuess | None = None) -> None:
        """
        Initialize adaptive algorithm.

        Args:
            initial_guess: REQUIRED initial guess for the first iteration.
                          Subsequent iterations will automatically use previous solutions.
        """
        self.initial_guess = initial_guess

    @abstractmethod
    def run(
        self, problem: ProblemProtocol, initial_solution: _InitialSolution = None
    ) -> OptimalControlSolution:
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
    Requires complete explicit specification - no defaults or assumptions.
    """

    def __init__(
        self,
        polynomial_degrees: list[int],
        mesh_points: FloatArray | list[float],
        initial_guess: InitialGuess | None = None,
    ) -> None:
        """
        Initialize fixed mesh solver.

        Args:
            polynomial_degrees: REQUIRED list of polynomial degrees per interval
            mesh_points: REQUIRED mesh points in [-1, 1]
            initial_guess: Optional initial guess (if not provided, user must set via problem)

        Raises:
            ValueError: If any parameter is invalid
        """
        super().__init__(initial_guess)

        # Validate required inputs
        if not polynomial_degrees:
            raise ValueError("polynomial_degrees must be provided and non-empty")

        if mesh_points is None:
            raise ValueError("mesh_points must be explicitly provided")

        # Convert to numpy array and validate
        mesh_array = np.array(mesh_points, dtype=np.float64)

        # Strict validation
        if len(polynomial_degrees) != len(mesh_array) - 1:
            raise ValueError(
                f"Number of polynomial degrees ({len(polynomial_degrees)}) must be exactly "
                f"one less than number of mesh points ({len(mesh_array)})"
            )

        # Validate polynomial degrees
        for k, degree in enumerate(polynomial_degrees):
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError(
                    f"Polynomial degree for interval {k} must be positive integer, got {degree}"
                )

        # Validate mesh points
        if not np.isclose(mesh_array[0], -1.0):
            raise ValueError(f"First mesh point must be -1.0, got {mesh_array[0]}")

        if not np.isclose(mesh_array[-1], 1.0):
            raise ValueError(f"Last mesh point must be 1.0, got {mesh_array[-1]}")

        if not np.all(np.diff(mesh_array) > 1e-9):
            raise ValueError("Mesh points must be strictly increasing with minimum spacing of 1e-9")

        self.polynomial_degrees = polynomial_degrees
        self.mesh_points = mesh_array

    def run(
        self, problem: ProblemProtocol, initial_solution: _InitialSolution = None
    ) -> OptimalControlSolution:
        """
        Run the fixed mesh solver.

        Args:
            problem: The optimal control problem
            initial_solution: Ignored for fixed mesh

        Returns:
            The optimal control solution

        Raises:
            ValueError: If problem is not properly configured
        """
        from trajectolab.direct_solver import solve_single_phase_radau_collocation

        # Set mesh configuration on problem (this clears any existing initial guess)
        problem.set_mesh(self.polynomial_degrees, self.mesh_points)

        # Apply initial guess from constructor if provided
        if self.initial_guess is not None:
            problem.initial_guess = self.initial_guess

        # Solve the problem - initial guess is optional
        return solve_single_phase_radau_collocation(problem)
