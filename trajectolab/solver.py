"""
Fixed solver functions with consistent scaling property access.
"""

import logging
from typing import cast

import numpy as np

from trajectolab.direct_solver import solve_single_phase_radau_collocation
from trajectolab.problem import Problem
from trajectolab.solution import _Solution
from trajectolab.tl_types import OptimalControlSolution, ProblemProtocol


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
    error_tolerance: float = 1e-3,
    max_iterations: int = 30,
    min_polynomial_degree: int = 4,
    max_polynomial_degree: int = 16,
    ode_solver_tolerance: float = 1e-7,
    num_error_sim_points: int = 40,
    nlp_options: dict[str, object] | None = None,
) -> _Solution:
    """
    Solve optimal control problem with adaptive mesh refinement.
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
        num_error_sim_points=num_error_sim_points,
        initial_guess=initial_guess,
    )

    return _Solution(solution_data, protocol_problem)


def solve(
    problem: Problem,
    mesh_method: str = "fixed",
    **kwargs: object,
) -> _Solution:
    """
    General solve function that dispatches to specific solvers.

    Args:
        problem: The optimal control problem to solve
        mesh_method: Either "fixed" or "adaptive"
        **kwargs: Additional arguments passed to specific solver

    Returns:
        Solution object containing the results

    Note:
        To provide an initial guess, use problem.set_initial_guess() before calling this function.

    Raises:
        ValueError: If mesh_method is not recognized
    """
    solver_logger.info(f"Solving problem '{problem.name}' with {mesh_method} mesh")

    if mesh_method == "fixed":
        return solve_fixed_mesh(problem, **kwargs)  # type: ignore[arg-type]
    elif mesh_method == "adaptive":
        return solve_adaptive(problem, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown mesh_method: {mesh_method}. Use 'fixed' or 'adaptive'")
