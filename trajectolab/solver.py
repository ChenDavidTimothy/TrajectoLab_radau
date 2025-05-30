# trajectolab/solver.py
"""
Main solver interface for multiphase optimal control problems.

This module provides the primary solving functions that users call to solve
their multiphase optimal control problems using either fixed or adaptive mesh strategies.
"""

import logging
from typing import cast

from trajectolab.direct_solver import solve_multiphase_radau_collocation
from trajectolab.input_validation import (
    validate_adaptive_solver_parameters,
    validate_multiphase_problem_ready_for_solving,
)
from trajectolab.problem import Problem
from trajectolab.solution import Solution
from trajectolab.tl_types import (
    MultiPhaseInitialGuess,
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
    """
    Solve a multiphase optimal control problem using fixed pseudospectral meshes.

    This function solves the problem using the mesh configurations specified
    for each phase in problem.phase(p).set_mesh(). The meshes remain fixed during
    optimization, making this approach faster but potentially less accurate than
    adaptive methods.

    Args:
        problem: Multiphase Problem instance with configured meshes, dynamics, and objective
        nlp_options: Optional IPOPT solver options. Common options include:
            - "ipopt.max_iter": Maximum iterations (default: 3000)
            - "ipopt.tol": Convergence tolerance (default: 1e-8)
            - "ipopt.print_level": Output verbosity 0-12 (default: 0)

    Returns:
        Solution object containing optimization results, trajectories, and metadata.
        Check solution.success to verify if optimization succeeded.

    Raises:
        trajectolab.ConfigurationError: If problem is not properly configured

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> problem = tl.Problem("Multiphase Mission")
        >>>
        >>> # Phase 1: Ascent
        >>> with problem.phase(1) as ascent:
        >>>     t1 = ascent.time(initial=0.0, final=(100, 200))
        >>>     x1 = ascent.state("position", initial=0.0)
        >>>     u1 = ascent.control("thrust", boundary=(0, 1))
        >>>     ascent.dynamics({x1: u1})
        >>>     ascent.set_mesh([10], np.array([-1.0, 1.0]))
        >>>
        >>> # Phase 2: Coast
        >>> with problem.phase(2) as coast:
        >>>     t2 = coast.time(initial=t1.final)
        >>>     x2 = coast.state("position", initial=x1.final)
        >>>     coast.dynamics({x2: 0})
        >>>     coast.set_mesh([8], np.array([-1.0, 1.0]))
        >>>
        >>> # Cross-phase constraints and objective
        >>> problem.subject_to(x1.final == x2.initial)
        >>> problem.minimize(t2.final)
        >>>
        >>> solution = tl.solve_fixed_mesh(problem)
        >>> if solution.success:
        ...     print(f"Mission time: {solution.get_phase_final_time(2):.3f}")
        ...     solution.plot()
    """

    # Log major operation start
    logger.info("Starting multiphase fixed-mesh solve: problem='%s'", problem.name)

    # Log problem dimensions
    if logger.isEnabledFor(logging.DEBUG):
        phase_ids = problem.get_phase_ids()
        total_states, total_controls, num_static_params = problem.get_total_variable_counts()
        logger.debug(
            "Multiphase problem dimensions: phases=%d, total_states=%d, total_controls=%d, static_params=%d",
            len(phase_ids),
            total_states,
            total_controls,
            num_static_params,
        )

    # Comprehensive validation
    validate_multiphase_problem_ready_for_solving(cast(ProblemProtocol, problem))

    # Set solver options
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS

    # Log solver configuration
    logger.debug("NLP solver options: %s", problem.solver_options)

    # Convert to protocol and solve
    protocol_problem = cast(ProblemProtocol, problem)
    solution_data: OptimalControlSolution = solve_multiphase_radau_collocation(protocol_problem)

    # Log solution status
    if solution_data.success:
        logger.info(
            "Multiphase fixed-mesh solve completed successfully: objective=%.6e",
            solution_data.objective or 0.0,
        )
    else:
        logger.warning("Multiphase fixed-mesh solve failed: %s", solution_data.message)

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
    initial_guess: MultiPhaseInitialGuess | None = None,
) -> Solution:
    """
    Solve a multiphase optimal control problem using adaptive mesh refinement.

    This function automatically refines the mesh for each phase during optimization
    to achieve a specified error tolerance. It uses the PHS (p-refinement, h-refinement,
    s-refinement) algorithm to adaptively adjust polynomial degrees and mesh spacing
    for each phase independently.

    Args:
        problem: Multiphase Problem instance with initial mesh configurations for each phase
        error_tolerance: Target relative error tolerance (default: 1e-6)
        max_iterations: Maximum refinement iterations (default: 10)
        min_polynomial_degree: Minimum polynomial degree per interval (default: 3)
        max_polynomial_degree: Maximum polynomial degree per interval (default: 10)
        ode_solver_tolerance: Tolerance for error estimation ODE solver (default: 1e-7)
        ode_method: ODE integration method for error estimation (default: "RK45")
        ode_max_step: Maximum step size for ODE solver (default: None)
        num_error_sim_points: Number of points for error simulation (default: 50)
        ode_solver: Custom ODE solver function (default: scipy.integrate.solve_ivp)
        nlp_options: Optional IPOPT solver options for each NLP solve
        initial_guess: Initial guess for first iteration (overrides problem guess)

    Returns:
        Solution object with final refined meshes and high-accuracy results.
        The solution contains the final mesh configuration used for each phase.

    Raises:
        trajectolab.ConfigurationError: If problem is not properly configured or parameters are invalid

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> problem = tl.Problem("High Precision Multiphase")
        >>> # ... define multiphase problem ...
        >>>
        >>> # Set initial meshes for each phase
        >>> with problem.phase(1) as phase1:
        >>>     # ... define phase 1 ...
        >>>     phase1.set_mesh([5], np.array([-1.0, 1.0]))
        >>>
        >>> with problem.phase(2) as phase2:
        >>>     # ... define phase 2 ...
        >>>     phase2.set_mesh([5], np.array([-1.0, 1.0]))
        >>>
        >>> solution = tl.solve_adaptive(
        ...     problem,
        ...     error_tolerance=1e-8,
        ...     max_iterations=15
        ... )
        >>>
        >>> if solution.success:
        ...     print(f"Final meshes:")
        ...     for phase_id in problem.get_phase_ids():
        ...         intervals = solution.get_phase_mesh_intervals(phase_id)
        ...         print(f"  Phase {phase_id}: {len(intervals)} intervals")
        ...     solution.plot()

    Note:
        Adaptive solving typically takes longer than fixed mesh but provides
        higher accuracy and automatic mesh optimization for each phase. The initial
        meshes specified in each phase.set_mesh() are used as starting points.
    """

    # Log major operation start with key parameters
    logger.info(
        "Starting multiphase adaptive solve: problem='%s', tolerance=%.1e, max_iter=%d",
        problem.name,
        error_tolerance,
        max_iterations,
    )

    # Log detailed parameters
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
    validate_multiphase_problem_ready_for_solving(cast(ProblemProtocol, problem))

    # Set default ODE solver if not provided
    if ode_solver is None:
        from scipy.integrate import solve_ivp

        ode_solver = solve_ivp
        logger.debug("Using default ODE solver: scipy.integrate.solve_ivp")

    # Set solver options
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS
    protocol_problem = cast(ProblemProtocol, problem)

    # Use provided initial guess or problem's guess
    if initial_guess is not None:
        protocol_problem.initial_guess = initial_guess
    else:
        initial_guess = problem.initial_guess

    # Log initial mesh configurations
    if logger.isEnabledFor(logging.DEBUG):
        for phase_id in problem.get_phase_ids():
            phase_def = problem._phases[phase_id]
            if phase_def.mesh_configured:
                logger.debug(
                    "Phase %d initial mesh: degrees=%s, points=%d",
                    phase_id,
                    phase_def.collocation_points_per_interval,
                    len(phase_def.global_normalized_mesh_nodes)
                    if phase_def.global_normalized_mesh_nodes is not None
                    else 0,
                )

    # Call multiphase adaptive algorithm
    from trajectolab.adaptive.multiphase_phs.algorithm import solve_multiphase_phs_adaptive_internal

    solution_data: OptimalControlSolution = solve_multiphase_phs_adaptive_internal(
        problem=protocol_problem,
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

    # Log final result
    if solution_data.success:
        total_intervals = sum(
            len(intervals) for intervals in solution_data.phase_mesh_intervals.values()
        )
        logger.info(
            "Multiphase adaptive solve converged: objective=%.6e, total_intervals=%d",
            solution_data.objective or 0.0,
            total_intervals,
        )
    else:
        logger.warning("Multiphase adaptive solve failed: %s", solution_data.message)

    return Solution(solution_data, protocol_problem)
