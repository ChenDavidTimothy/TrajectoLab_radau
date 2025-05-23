"""
Fixed solver functions with consistent scaling property access.
"""

import logging
from typing import cast

import numpy as np

from trajectolab.direct_solver import solve_single_phase_radau_collocation
from trajectolab.problem import Problem
from trajectolab.solution import _Solution
from trajectolab.tl_types import (
    InitialGuess,
    ODESolverCallable,
    OptimalControlSolution,
    ProblemProtocol,
)


# Configure solver-specific logger
solver_logger = logging.getLogger("trajectolab.solver")
if not solver_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    solver_logger.addHandler(handler)
    solver_logger.setLevel(logging.INFO)


def solve_fixed_mesh(
    problem: Problem,
    nlp_options: dict[str, object] | None = None,
) -> _Solution:
    """
    Solve optimal control problem with fixed mesh.
    """
    solver_logger.info(f"Starting fixed-mesh solve for problem '{problem.name}'")

    # Set solver options on problem
    problem.solver_options = nlp_options or {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }

    # Create protocol-compatible version and solve
    protocol_problem = cast(ProblemProtocol, problem)
    solution_data: OptimalControlSolution = solve_single_phase_radau_collocation(protocol_problem)

    return _Solution(solution_data, protocol_problem)


def solve_adaptive(
    problem: Problem,
    error_tolerance: float = 1e-6,
    max_iterations: int = 10,
    min_polynomial_degree: int = 3,
    max_polynomial_degree: int = 10,
    ode_solver_tolerance: float = 1e-7,  # Reuse existing parameter
    ode_method: str = "RK45",  # NEW: Simple method selection
    ode_max_step: float | None = None,  # NEW: Optional step size control
    num_error_sim_points: int = 50,
    ode_solver: ODESolverCallable | None = None,  # Advanced users only
    nlp_options: dict[str, object] | None = None,
    initial_guess: InitialGuess | None = None,
) -> _Solution:
    """
    Solve optimal control problem using adaptive mesh refinement.

    Args:
        ode_solver: ODE solver function (default: scipy.integrate.solve_ivp)
        ... other parameters
    """

    solver_logger.info(f"Starting adaptive solve for problem '{problem.name}'")
    solver_logger.info(f"Settings: error_tol={error_tolerance}, max_iter={max_iterations}")
    solver_logger.info(
        f"Polynomial degree range: [{min_polynomial_degree}, {max_polynomial_degree}]"
    )

    # Verify mesh is configured
    if not problem._mesh_configured:
        raise ValueError(
            "Initial mesh must be configured before solving. "
            "Call problem.set_mesh(polynomial_degrees, mesh_points) first."
        )

    if ode_solver is None:
        from scipy.integrate import solve_ivp

        ode_solver = solve_ivp

    # Extract initial mesh configuration from problem
    initial_polynomial_degrees = problem.collocation_points_per_interval
    initial_mesh_points = problem.global_normalized_mesh_nodes

    # Set solver options on problem
    problem.solver_options = nlp_options or {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }

    # Use whatever initial guess was set on the problem (if any)
    initial_guess = problem.initial_guess

    # Create protocol-compatible version
    protocol_problem = cast(ProblemProtocol, problem)

    # Call the internal adaptive solver
    from trajectolab.adaptive.phs.algorithm import solve_phs_adaptive_internal

    solution_data: OptimalControlSolution = solve_phs_adaptive_internal(
        problem=protocol_problem,
        initial_polynomial_degrees=initial_polynomial_degrees,
        initial_mesh_points=np.array(initial_mesh_points, dtype=np.float64),
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        ode_solver_tolerance=ode_solver_tolerance,
        ode_method=ode_method,  # NEW
        ode_max_step=ode_max_step,  # NEW
        ode_solver=ode_solver,  # NEW
        num_error_sim_points=num_error_sim_points,
        initial_guess=initial_guess,
    )

    return _Solution(solution_data, protocol_problem)
