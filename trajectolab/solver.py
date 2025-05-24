"""
Solver interface - SIMPLIFIED with ENHANCED ERROR HANDLING.
Unified common setup code, removed ALL redundant duplication.
Added targeted guard clauses for configuration validation.
"""

import logging
from typing import cast

import numpy as np

from trajectolab.direct_solver import solve_single_phase_radau_collocation
from trajectolab.exceptions import ConfigurationError
from trajectolab.problem import Problem
from trajectolab.solution import Solution
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


# Default solver options - single definition
DEFAULT_NLP_OPTIONS: dict[str, object] = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": 0,
}


def _validate_problem_for_solving(problem: Problem) -> None:
    """Validate that problem is ready for solving with fail-fast guard clauses."""
    # Guard clause: Mesh must be configured
    if not problem._mesh_configured:
        raise ConfigurationError(
            "Problem mesh must be configured before solving",
            "Call problem.set_mesh(polynomial_degrees, mesh_points) first",
        )

    # Guard clause: Must have dynamics
    if not problem._dynamics_expressions:
        raise ConfigurationError(
            "Problem dynamics must be defined before solving",
            "Call problem.dynamics({state: expression, ...}) first",
        )

    # Guard clause: Must have objective
    if problem._objective_expression is None:
        raise ConfigurationError(
            "Problem objective must be defined before solving",
            "Call problem.minimize(expression) first",
        )

    # Guard clause: Variable counts must be consistent
    num_states, num_controls = problem.get_variable_counts()
    if num_states == 0 and num_controls == 0:
        raise ConfigurationError(
            "Problem must have at least one state or control variable",
            "Define variables using problem.state() or problem.control()",
        )


def _setup_solver_common(
    problem: Problem, nlp_options: dict[str, object] | None
) -> ProblemProtocol:
    """UNIFIED setup logic for both fixed and adaptive solvers."""
    # Validate problem configuration first
    _validate_problem_for_solving(problem)

    # Set solver options - single point of control
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS

    # Return protocol-compatible version
    return cast(ProblemProtocol, problem)


def solve_fixed_mesh(
    problem: Problem,
    nlp_options: dict[str, object] | None = None,
) -> Solution:
    """Solve optimal control problem with fixed mesh - SIMPLIFIED with ENHANCED ERROR HANDLING."""
    solver_logger.info(f"Starting fixed-mesh solve for problem '{problem.name}'")

    # Unified setup with validation
    protocol_problem = _setup_solver_common(problem, nlp_options)

    # Solve
    solution_data: OptimalControlSolution = solve_single_phase_radau_collocation(protocol_problem)

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
    """Solve optimal control problem using adaptive mesh refinement - SIMPLIFIED with ENHANCED ERROR HANDLING."""

    solver_logger.info(f"Starting adaptive solve for problem '{problem.name}'")
    solver_logger.info(f"Settings: error_tol={error_tolerance}, max_iter={max_iterations}")
    solver_logger.info(
        f"Polynomial degree range: [{min_polynomial_degree}, {max_polynomial_degree}]"
    )

    # Guard clause: Validate adaptive parameters
    if error_tolerance <= 0:
        raise ConfigurationError(
            f"Error tolerance must be positive, got {error_tolerance}",
            "Provide a positive error tolerance value",
        )

    if max_iterations <= 0:
        raise ConfigurationError(
            f"Max iterations must be positive, got {max_iterations}",
            "Provide a positive max iterations value",
        )

    if min_polynomial_degree < 1 or max_polynomial_degree < min_polynomial_degree:
        raise ConfigurationError(
            f"Invalid polynomial degree range: [{min_polynomial_degree}, {max_polynomial_degree}]",
            "Min degree must be >= 1 and max degree must be >= min degree",
        )

    # Verify mesh is configured
    if not problem._mesh_configured:
        raise ConfigurationError(
            "Initial mesh must be configured before solving",
            "Call problem.set_mesh(polynomial_degrees, mesh_points) first",
        )

    # Set default ODE solver if not provided
    if ode_solver is None:
        from scipy.integrate import solve_ivp

        ode_solver = solve_ivp

    # Extract initial mesh configuration from problem
    initial_polynomial_degrees = problem.collocation_points_per_interval
    initial_mesh_points = problem.global_normalized_mesh_nodes

    # Unified setup with validation
    protocol_problem = _setup_solver_common(problem, nlp_options)

    # Use whatever initial guess was set on the problem (if any)
    initial_guess = problem.initial_guess

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
        ode_method=ode_method,
        ode_max_step=ode_max_step,
        ode_solver=ode_solver,
        num_error_sim_points=num_error_sim_points,
        initial_guess=initial_guess,
    )

    return Solution(solution_data, protocol_problem)
