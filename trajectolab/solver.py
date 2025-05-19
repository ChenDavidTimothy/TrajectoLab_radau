"""
Functional solver interface for optimal control problems using the Radau Pseudospectral Method.
"""

from typing import cast

import numpy as np

from trajectolab.adaptive.phs.algorithm import solve_phs_adaptive_internal
from trajectolab.direct_solver import OptimalControlSolution, solve_single_phase_radau_collocation
from trajectolab.problem import Problem
from trajectolab.solution import Solution
from trajectolab.tl_types import FloatArray, ProblemProtocol


def solve_fixed_mesh(
    problem: Problem,
    polynomial_degrees: list[int],
    mesh_points: FloatArray | list[float],
    nlp_options: dict[str, object] | None = None,
) -> Solution:
    """
    Solve optimal control problem with fixed mesh.

    Args:
        problem: The optimal control problem to solve
        polynomial_degrees: Number of collocation nodes per mesh interval
        mesh_points: Mesh points in normalized domain [-1, 1]
        nlp_options: Optional NLP solver options

    Returns:
        Solution object containing the results

    Note:
        To provide an initial guess, use problem.set_initial_guess() before calling this function.

    Raises:
        ValueError: If mesh specification is invalid
    """
    # Set solver options on problem
    problem.solver_options = nlp_options or {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }

    # Configure mesh on problem
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Initial guess is whatever was set on the problem (if any)
    # No need to do anything - it's already stored in problem.initial_guess

    # Create protocol-compatible version and solve
    protocol_problem = cast(ProblemProtocol, problem)
    solution_data: OptimalControlSolution = solve_single_phase_radau_collocation(protocol_problem)

    return Solution(solution_data, protocol_problem)


def solve_adaptive(
    problem: Problem,
    initial_polynomial_degrees: list[int],
    initial_mesh_points: FloatArray | list[float],
    error_tolerance: float = 1e-3,
    max_iterations: int = 30,
    min_polynomial_degree: int = 4,
    max_polynomial_degree: int = 16,
    ode_solver_tolerance: float = 1e-7,
    num_error_sim_points: int = 40,
    nlp_options: dict[str, object] | None = None,
) -> Solution:
    """
    Solve optimal control problem with adaptive mesh refinement.

    Args:
        problem: The optimal control problem to solve
        initial_polynomial_degrees: Initial polynomial degrees per interval
        initial_mesh_points: Initial mesh points in normalized domain [-1, 1]
        error_tolerance: Error tolerance threshold for mesh refinement
        max_iterations: Maximum number of refinement iterations
        min_polynomial_degree: Minimum polynomial degree allowed
        max_polynomial_degree: Maximum polynomial degree allowed
        ode_solver_tolerance: ODE solver tolerance for error estimation
        num_error_sim_points: Number of simulation points for error estimation
        nlp_options: Optional NLP solver options

    Returns:
        Solution object containing the results

    Note:
        To provide an initial guess, use problem.set_initial_guess() before calling this function.

    Raises:
        ValueError: If parameters are invalid
    """
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

    return Solution(solution_data, protocol_problem)


def solve(
    problem: Problem,
    mesh_method: str = "fixed",
    **kwargs: object,
) -> Solution:
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
    if mesh_method == "fixed":
        return solve_fixed_mesh(problem, **kwargs)  # type: ignore[arg-type]
    elif mesh_method == "adaptive":
        return solve_adaptive(problem, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown mesh_method: {mesh_method}. Use 'fixed' or 'adaptive'")
