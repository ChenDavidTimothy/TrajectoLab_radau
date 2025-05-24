# trajectolab/solver.py
"""
Solver interface with production logging.
"""

import logging
from typing import cast

import numpy as np

from trajectolab.direct_solver import solve_single_phase_radau_collocation
from trajectolab.input_validation import (
    validate_adaptive_solver_parameters,
    validate_problem_ready_for_solving,
)
from trajectolab.problem import Problem
from trajectolab.solution import Solution
from trajectolab.tl_types import (
    InitialGuess,
    ODESolverCallable,
    OptimalControlSolution,
    ProblemProtocol,
)


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


# Default solver options
DEFAULT_NLP_OPTIONS: dict[str, object] = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": 0,
}


def solve_fixed_mesh(
    problem: Problem,
    nlp_options: dict[str, object] | None = None,
) -> Solution:
    """Solve optimal control problem with fixed mesh."""

    # Log major operation start (INFO - user cares about this)
    logger.info("Starting fixed-mesh solve: problem='%s'", problem.name)

    # Log problem dimensions (DEBUG - developer info)
    if logger.isEnabledFor(logging.DEBUG):
        num_states, num_controls = problem.get_variable_counts()
        num_intervals = (
            len(problem.collocation_points_per_interval) if problem._mesh_configured else 0
        )
        logger.debug(
            "Problem dimensions: states=%d, controls=%d, intervals=%d",
            num_states,
            num_controls,
            num_intervals,
        )

    # Comprehensive validation
    validate_problem_ready_for_solving(problem)

    # Set solver options
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS

    # Log solver configuration (DEBUG)
    logger.debug("NLP solver options: %s", problem.solver_options)

    # Convert to protocol and solve
    protocol_problem = cast(ProblemProtocol, problem)
    solution_data: OptimalControlSolution = solve_single_phase_radau_collocation(protocol_problem)

    # Log solution status (INFO - user cares about success/failure)
    if solution_data.success:
        logger.info(
            "Fixed-mesh solve completed successfully: objective=%.6e",
            solution_data.objective or 0.0,
        )
    else:
        logger.warning("Fixed-mesh solve failed: %s", solution_data.message)

    return Solution(solution_data, protocol_problem)


def solve_adaptive(
    problem: Problem,
    error_tolerance: float = 1e-6,
    max_iterations: int = 10,
    min_polynomial_degree: int = 3,
    max_polynomial_degree: int = 10,
    ode_solver_tolerance: float = 1e-7,
    ode_method: str = "RK45",
    ode_max_step: float | None = None,
    num_error_sim_points: int = 50,
    ode_solver: ODESolverCallable | None = None,
    nlp_options: dict[str, object] | None = None,
    initial_guess: InitialGuess | None = None,
) -> Solution:
    """Solve optimal control problem using adaptive mesh refinement."""

    # Log major operation start with key parameters (INFO - user cares)
    logger.info(
        "Starting adaptive solve: problem='%s', tolerance=%.1e, max_iter=%d",
        problem.name,
        error_tolerance,
        max_iterations,
    )

    # Log detailed parameters (DEBUG - developer info)
    logger.debug(
        "Adaptive parameters: poly_degree=[%d,%d], ode_tol=%.1e, sim_points=%d",
        min_polynomial_degree,
        max_polynomial_degree,
        ode_solver_tolerance,
        num_error_sim_points,
    )

    # Validation
    validate_adaptive_solver_parameters(
        error_tolerance, max_iterations, min_polynomial_degree, max_polynomial_degree
    )
    validate_problem_ready_for_solving(problem)

    # Set default ODE solver if not provided
    if ode_solver is None:
        from scipy.integrate import solve_ivp

        ode_solver = solve_ivp
        logger.debug("Using default ODE solver: scipy.integrate.solve_ivp")

    # Extract initial mesh configuration
    initial_polynomial_degrees = problem.collocation_points_per_interval
    initial_mesh_points = problem.global_normalized_mesh_nodes

    # Log initial mesh (DEBUG)
    logger.debug(
        "Initial mesh: degrees=%s, points=%d", initial_polynomial_degrees, len(initial_mesh_points)
    )

    # Set solver options
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS
    protocol_problem = cast(ProblemProtocol, problem)
    initial_guess = problem.initial_guess

    # Call adaptive algorithm
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
        ode_method=ode_method,
        ode_max_step=ode_max_step,
        ode_solver=ode_solver,
        num_error_sim_points=num_error_sim_points,
        initial_guess=initial_guess,
    )

    # Log final result (INFO - user cares about convergence)
    if solution_data.success:
        final_intervals = len(solution_data.num_collocation_nodes_per_interval)
        logger.info(
            "Adaptive solve converged: objective=%.6e, final_intervals=%d",
            solution_data.objective or 0.0,
            final_intervals,
        )
    else:
        logger.warning("Adaptive solve failed: %s", solution_data.message)

    return Solution(solution_data, protocol_problem)
