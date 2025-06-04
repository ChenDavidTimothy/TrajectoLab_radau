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
from trajectolab.utils.constants import (
    DEFAULT_ERROR_SIM_POINTS,
    DEFAULT_ODE_ATOL_FACTOR,
    DEFAULT_ODE_MAX_STEP,
    DEFAULT_ODE_METHOD,
    DEFAULT_ODE_RTOL,
)


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
    show_summary: bool = True,
) -> Solution:
    """
    Solve a multiphase optimal control problem using fixed pseudospectral meshes.

    Args:
        problem: Multiphase Problem instance with configured meshes, dynamics, and objective
        nlp_options: Optional IPOPT solver options
        show_summary: Whether to display comprehensive solution summary (default: True)

    Returns:
        Solution object containing optimization results, trajectories, and metadata.

    Raises:
        trajectolab.ConfigurationError: If problem is not properly configured

    Examples:
        >>> # Create a simple rocket ascent problem
        >>> problem = Problem("Rocket Ascent")
        >>>
        >>> # Define phase with state variables
        >>> phase = problem.set_phase(1)
        >>> time = phase.time(initial=0, final=10)
        >>> altitude = phase.state("altitude", initial=0, final=1000)
        >>> velocity = phase.state("velocity", initial=0)
        >>> thrust = phase.control("thrust", boundary=(0, 1000))
        >>>
        >>> # Define dynamics
        >>> phase.dynamics({
        ...     altitude: velocity,
        ...     velocity: thrust - 9.81
        ... })
        >>>
        >>> # Add path constraints (applied at every collocation point)
        >>> phase.path_constraints(
        ...     altitude >= 0,           # Stay above ground
        ...     thrust <= 1000,          # Maximum thrust limit
        ...     velocity <= 100          # Speed limit
        ... )
        >>>
        >>> # Add event constraints (applied at boundaries)
        >>> phase.event_constraints(
        ...     altitude.initial == 0,   # Start at ground level
        ...     velocity.initial == 0,   # Start from rest
        ...     altitude.final >= 1000   # Reach target altitude
        ... )
        >>>
        >>> # Define objective and mesh
        >>> problem.minimize(time.final)
        >>> phase.mesh([4, 4], [0, 0.5, 1.0])
        >>>
        >>> # Solve the problem
        >>> solution = solve_fixed_mesh(problem)
    """
    logger.info("Starting multiphase fixed-mesh solve: problem='%s'", problem.name)

    # Log problem dimensions
    if logger.isEnabledFor(logging.DEBUG):
        phase_ids = problem._get_phase_ids()
        total_states, total_controls, num_static_params = problem.get_total_variable_counts()
        logger.debug(
            "Problem dimensions: phases=%d, total_states=%d, total_controls=%d, static_params=%d",
            len(phase_ids),
            total_states,
            total_controls,
            num_static_params,
        )

    # SINGLE comprehensive validation call
    validate_multiphase_problem_ready_for_solving(cast(ProblemProtocol, problem))

    # Configure solver
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS
    logger.debug("NLP solver options: %s", problem.solver_options)

    # Solve
    protocol_problem = cast(ProblemProtocol, problem)
    solution_data: OptimalControlSolution = solve_multiphase_radau_collocation(protocol_problem)

    # Log result
    if solution_data.success:
        logger.info(
            "Fixed-mesh solve completed successfully: objective=%.6e",
            solution_data.objective or 0.0,
        )
    else:
        logger.warning("Fixed-mesh solve failed: %s", solution_data.message)

    # Create solution object (will automatically show summary unless disabled)
    return Solution(solution_data, protocol_problem, auto_summary=show_summary)


def solve_adaptive(
    problem: Problem,
    error_tolerance: float = 1e-6,
    max_iterations: int = 10,
    min_polynomial_degree: int = 3,
    max_polynomial_degree: int = 10,
    ode_solver_tolerance: float = DEFAULT_ODE_RTOL,
    ode_method: str = DEFAULT_ODE_METHOD,
    ode_max_step: float | None = DEFAULT_ODE_MAX_STEP,
    ode_atol_factor: float = DEFAULT_ODE_ATOL_FACTOR,
    num_error_sim_points: int = DEFAULT_ERROR_SIM_POINTS,
    ode_solver: ODESolverCallable | None = None,
    nlp_options: dict[str, object] | None = None,
    initial_guess: MultiPhaseInitialGuess | None = None,
    show_summary: bool = True,
) -> Solution:
    """
    Solve a multiphase optimal control problem using adaptive mesh refinement.

    Args:
        problem: Multiphase Problem instance with initial mesh configurations for each phase
        error_tolerance: Target relative error tolerance (default: 1e-6)
        max_iterations: Maximum refinement iterations (default: 10)
        min_polynomial_degree: Minimum polynomial degree per interval (default: 3)
        max_polynomial_degree: Maximum polynomial degree per interval (default: 10)
        ode_solver_tolerance: Relative tolerance for error estimation ODE solver (default: 1e-7)
        ode_method: ODE integration method for error estimation (default: "RK45")
        ode_max_step: Maximum step size for ODE solver (default: None)
        ode_atol_factor: Factor for absolute tolerance calculation (atol = rtol * factor) (default: 1e-2)
        num_error_sim_points: Number of points for error simulation (default: 50)
        ode_solver: Custom ODE solver function (default: scipy.integrate.solve_ivp)
        nlp_options: Optional IPOPT solver options for each NLP solve
        initial_guess: Initial guess for first iteration (overrides problem guess)
        show_summary: Whether to display comprehensive solution summary (default: True)

    Returns:
        Solution object with final refined meshes and high-accuracy results.

    Raises:
        trajectolab.ConfigurationError: If problem is not properly configured or parameters are invalid

    Examples:
        >>> # Create a multiphase spacecraft trajectory problem
        >>> problem = Problem("Spacecraft Trajectory")
        >>>
        >>> # Phase 1: Launch ascent
        >>> p1 = problem.set_phase(1)
        >>> t1 = p1.time(initial=0, final=100)
        >>> altitude_p1 = p1.state("altitude", initial=0)
        >>> velocity_p1 = p1.state("velocity", initial=0)
        >>> mass_p1 = p1.state("mass", initial=1000)
        >>> thrust_p1 = p1.control("thrust")
        >>>
        >>> p1.dynamics({
        ...     altitude_p1: velocity_p1,
        ...     velocity_p1: thrust_p1 / mass_p1 - 9.81,
        ...     mass_p1: -thrust_p1 * 0.001
        ... })
        >>>
        >>> # Path constraints for phase 1
        >>> p1.path_constraints(
        ...     altitude_p1 >= 0,
        ...     thrust_p1 >= 0,
        ...     thrust_p1 <= 2000
        ... )
        >>>
        >>> # Event constraints for phase 1
        >>> p1.event_constraints(
        ...     altitude_p1.initial == 0,
        ...     velocity_p1.initial == 0,
        ...     mass_p1.initial == 1000
        ... )
        >>>
        >>> # Phase 2: Orbital insertion (linked via symbolic constraints)
        >>> p2 = problem.set_phase(2)
        >>> t2 = p2.time(initial=t1.final, final=200)  # Automatic phase linking
        >>> altitude_p2 = p2.state("altitude", initial=altitude_p1.final)  # Continuous altitude
        >>> velocity_p2 = p2.state("velocity", initial=velocity_p1.final)  # Continuous velocity
        >>> mass_p2 = p2.state("mass", initial=mass_p1.final)              # Continuous mass
        >>> thrust_p2 = p2.control("thrust")
        >>>
        >>> p2.dynamics({
        ...     altitude_p2: velocity_p2,
        ...     velocity_p2: thrust_p2 / mass_p2,
        ...     mass_p2: -thrust_p2 * 0.001
        ... })
        >>>
        >>> # Phase 2 constraints
        >>> p2.path_constraints(
        ...     thrust_p2 >= 0,
        ...     thrust_p2 <= 1000
        ... )
        >>>
        >>> p2.event_constraints(
        ...     altitude_p2.final >= 200000,  # Reach orbital altitude
        ...     velocity_p2.final >= 7800     # Achieve orbital velocity
        ... )
        >>>
        >>> # Configure meshes and solve adaptively
        >>> p1.mesh([3, 3], [0, 0.5, 1.0])
        >>> p2.mesh([3, 3], [0, 0.5, 1.0])
        >>> problem.minimize(mass_p1.initial - mass_p2.final)  # Maximize payload
        >>>
        >>> solution = solve_adaptive(problem, error_tolerance=1e-5)
    """
    logger.info(
        "Starting multiphase adaptive solve: problem='%s', tolerance=%.1e, max_iter=%d",
        problem.name,
        error_tolerance,
        max_iterations,
    )

    logger.debug(
        "Adaptive parameters: poly_degree=[%d,%d], ode_tol=%.1e, sim_points=%d",
        min_polynomial_degree,
        max_polynomial_degree,
        ode_solver_tolerance,
        num_error_sim_points,
    )

    # SINGLE comprehensive validation call
    validate_adaptive_solver_parameters(
        error_tolerance, max_iterations, min_polynomial_degree, max_polynomial_degree
    )
    validate_multiphase_problem_ready_for_solving(cast(ProblemProtocol, problem))

    # Configure solver
    problem.solver_options = nlp_options or DEFAULT_NLP_OPTIONS
    protocol_problem = cast(ProblemProtocol, problem)

    # Set initial guess
    if initial_guess is not None:
        protocol_problem.initial_guess = initial_guess

    # Log initial mesh configurations for all phases
    if logger.isEnabledFor(logging.DEBUG):
        for phase_id in problem._get_phase_ids():
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
    from trajectolab.adaptive.phs.algorithm import solve_multiphase_phs_adaptive_internal

    solution_data: OptimalControlSolution = solve_multiphase_phs_adaptive_internal(
        problem=protocol_problem,
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        ode_solver_tolerance=ode_solver_tolerance,
        ode_method=ode_method,
        ode_max_step=ode_max_step,
        ode_atol_factor=ode_atol_factor,
        ode_solver=ode_solver,  # Pass None if user didn't specify
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

    # Create solution object (will automatically show summary unless disabled)
    return Solution(solution_data, protocol_problem, auto_summary=show_summary)
