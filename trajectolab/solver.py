from trajectolab.adaptive.phs import PHSAdaptive
from trajectolab.solution import Solution


class RadauDirectSolver:
    def __init__(
        self,
        mesh_method=None,
        nlp_solver="ipopt",
        nlp_options=None,
    ):
        self.mesh_method = mesh_method or PHSAdaptive()
        self.nlp_solver = nlp_solver
        self.nlp_options = nlp_options or {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

    def solve(self, problem, initial_solution=None):
        # Convert the user-facing problem to the legacy format for the solver
        legacy_problem = problem._convert_to_legacy_problem()

        # Set solver options
        legacy_problem.solver_options = self.nlp_options

        # Use the mesh method to solve the problem
        legacy_solution = self.mesh_method.run(legacy_problem, initial_solution)

        # Create Solution object from legacy solution
        return Solution(legacy_solution, problem)


def solve(problem, solver=None):
    if solver is None:
        solver = RadauDirectSolver()

    return solver.solve(problem)
