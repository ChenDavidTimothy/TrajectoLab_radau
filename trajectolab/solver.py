from typing import Any, Dict, Optional

from trajectolab.adaptive.base import AdaptiveBase
from trajectolab.adaptive.phs import PHSAdaptive
from trajectolab.problem import Problem
from trajectolab.solution import Solution


class RadauDirectSolver:
    def __init__(
        self,
        mesh_method: Optional[AdaptiveBase] = None,
        nlp_solver: str = "ipopt",
        nlp_options: Optional[Dict[str, Any]] = None,
    ):
        self.mesh_method = mesh_method or PHSAdaptive()
        self.nlp_solver = nlp_solver
        self.nlp_options = nlp_options or {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

    def solve(self, problem: Problem, initial_solution=None) -> Solution:
        # Convert the user-facing problem to the legacy format for the solver
        legacy_problem = problem._convert_to_legacy_problem()

        # Set solver options
        legacy_problem.solver_options = self.nlp_options

        # Use the mesh method to solve the problem
        legacy_solution = self.mesh_method.run(problem, legacy_problem, initial_solution)

        # Create Solution object from legacy solution
        return Solution(legacy_solution, problem)


def solve(problem: Problem, solver: Optional[RadauDirectSolver] = None) -> Solution:
    if solver is None:
        solver = RadauDirectSolver()

    return solver.solve(problem)
