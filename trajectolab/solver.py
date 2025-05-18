from typing import cast

from trajectolab.adaptive.base import AdaptiveBase
from trajectolab.direct_solver import OptimalControlSolution
from trajectolab.problem import Problem
from trajectolab.solution import Solution
from trajectolab.tl_types import ProblemProtocol


class RadauDirectSolver:
    """
    Solver for optimal control problems using the Radau Pseudospectral Method.

    This solver provides interfaces to different mesh approaches:
    - Fixed mesh with complete explicit control
    - Adaptive mesh refinement with user-provided initial guess for first iteration
    """

    def __init__(
        self,
        mesh_method: AdaptiveBase | None = None,
        nlp_solver: str = "ipopt",
        nlp_options: dict[str, object] | None = None,
    ) -> None:
        """
        Initialize the Radau direct solver.

        Args:
            mesh_method: REQUIRED mesh method (FixedMesh or PHSAdaptive).
                        No default provided - user must explicitly choose.
            nlp_solver: Name of the NLP solver to use (default: "ipopt")
            nlp_options: Dictionary of options to pass to the NLP solver

        Raises:
            ValueError: If mesh_method is not provided
        """
        if mesh_method is None:
            raise ValueError(
                "mesh_method must be explicitly provided. Choose either:\n"
                "  - FixedMesh(polynomial_degrees=[...], mesh_points=[...])\n"
                "  - PHSAdaptive(initial_polynomial_degrees=[...], initial_mesh_points=[...], initial_guess=...)"
            )

        self.mesh_method: AdaptiveBase = mesh_method
        self.nlp_solver = nlp_solver
        self.nlp_options = nlp_options or {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

    def solve(self, problem: Problem) -> Solution:
        """
        Solve the optimal control problem.

        Args:
            problem: The optimal control problem to solve

        Returns:
            Solution object containing the solution data

        Raises:
            ValueError: If problem configuration is invalid
        """
        # Set solver options on problem
        problem.solver_options = self.nlp_options

        # Create protocol-compatible version of the problem
        protocol_problem = cast(ProblemProtocol, problem)

        # Run the mesh method (either fixed or adaptive)
        # The mesh method handles all mesh configuration and validation
        solution_data: OptimalControlSolution = self.mesh_method.run(protocol_problem)

        # Create and return Solution wrapper
        return Solution(solution_data, protocol_problem)


def solve(problem: Problem, solver: RadauDirectSolver | None = None) -> Solution:
    """
    Convenience function to solve an optimal control problem.

    Args:
        problem: The optimal control problem to solve
        solver: REQUIRED solver instance. No default provided.
                User must explicitly choose mesh method.

    Returns:
        Solution object containing the solution data

    Raises:
        ValueError: If solver is not provided
    """
    if solver is None:
        raise ValueError(
            "Solver must be explicitly provided. Example:\n"
            "  solver = RadauDirectSolver(mesh_method=FixedMesh(...))\n"
            "  solution = solve(problem, solver)"
        )

    return solver.solve(problem)
