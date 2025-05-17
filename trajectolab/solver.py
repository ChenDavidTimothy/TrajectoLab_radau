from typing import cast

import numpy as np

from trajectolab.adaptive.base import AdaptiveBase
from trajectolab.adaptive.phs import PHSAdaptive
from trajectolab.problem import Problem
from trajectolab.solution import Solution
from trajectolab.tl_types import ProblemProtocol


class RadauDirectSolver:
    """
    Solver for optimal control problems using the Radau Pseudospectral Method.

    This solver provides interfaces to different mesh refinement techniques,
    such as fixed-mesh collocation and adaptive mesh refinement.
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
            mesh_method: Mesh refinement method to use (default: PHSAdaptive)
            nlp_solver: Name of the NLP solver to use (default: "ipopt")
            nlp_options: Dictionary of options to pass to the NLP solver
        """
        self.mesh_method = mesh_method or PHSAdaptive()
        self.nlp_solver = nlp_solver
        self.nlp_options = nlp_options or {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

    def solve(self, problem: Problem) -> Solution:
        # Update problem with solver options
        problem.solver_options = self.nlp_options

        # Create a proper protocol-compatible version of the problem
        protocol_problem = problem

        # Handle mesh initialization if needed
        if (
            isinstance(problem.global_normalized_mesh_nodes, type(None))
            and problem.collocation_points_per_interval
        ):
            # Create default mesh from collocation points
            num_intervals = len(problem.collocation_points_per_interval)
            mesh_nodes = np.linspace(-1.0, 1.0, num_intervals + 1, dtype=np.float64)

            # Use setattr to bypass type checking constraints
            protocol_problem.global_normalized_mesh_nodes = mesh_nodes

        # Always use the mesh method (which is never None due to initialization)
        solution_data = self.mesh_method.run(cast(ProblemProtocol, protocol_problem))

        # Create Solution object
        return Solution(solution_data, cast(ProblemProtocol, protocol_problem))


def solve(problem: Problem, solver: RadauDirectSolver | None = None) -> Solution:
    """
    Convenience function to solve an optimal control problem.

    Args:
        problem: The optimal control problem to solve
        solver: Optional solver to use (default: new RadauDirectSolver instance)

    Returns:
        Solution object containing the solution data
    """
    if solver is None:
        solver = RadauDirectSolver()

    return solver.solve(problem)
