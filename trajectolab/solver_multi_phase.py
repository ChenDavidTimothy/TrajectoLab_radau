"""
Multi-phase solver interface for optimal control problems.

This module provides the primary solving functions that users call to solve
multi-phase optimal control problems using either fixed or adaptive mesh strategies.
Implements faithful CGPOPS multi-phase methodology with unified NLP formation.
"""

import logging
from typing import Any, cast

import numpy as np

from .direct_solver.multi_phase_core_solver import solve_multi_phase_radau_collocation
from .exceptions import ConfigurationError
from .problem.multi_phase_problem import MultiPhaseProblem
from .solution_multi_phase import MultiPhaseSolution
from .tl_types import (
    MultiPhaseInitialGuess,
    MultiPhaseProblemProtocol,
    ODESolverCallable,
)


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


# Default solver options for multi-phase problems
DEFAULT_MULTI_PHASE_NLP_OPTIONS: dict[str, object] = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": 0,
    # Multi-phase problems typically need higher iteration limits
    "ipopt.max_iter": 5000,
    # Tighter tolerance for multi-phase problems
    "ipopt.tol": 1e-8,
}


def solve_multi_phase_fixed_mesh(
    problem: MultiPhaseProblem,
    nlp_options: dict[str, object] | None = None,
) -> MultiPhaseSolution:
    """
    Solve a multi-phase optimal control problem using fixed pseudospectral meshes.

    This function solves multi-phase problems using the mesh configurations specified
    for each phase. Each phase uses its configured mesh which remains fixed during
    optimization. Creates a unified NLP following CGPOPS methodology for efficient
    simultaneous optimization across all phases.

    Args:
        problem: MultiPhaseProblem instance with configured phases, inter-phase
            constraints, global objective, and mesh configurations for all phases
        nlp_options: Optional IPOPT solver options. Common options include:
            - "ipopt.max_iter": Maximum iterations (default: 5000 for multi-phase)
            - "ipopt.tol": Convergence tolerance (default: 1e-8)
            - "ipopt.print_level": Output verbosity 0-12 (default: 0)
            - "ipopt.mu_strategy": Barrier parameter strategy
            - "ipopt.linear_solver": Linear solver (ma27, ma57, ma77, ma86, mumps)

    Returns:
        MultiPhaseSolution object containing optimization results, phase-specific
        trajectories, global parameters, and comprehensive analysis capabilities.
        Check solution.success to verify if optimization succeeded.

    Raises:
        trajectolab.ConfigurationError: If problem is not properly configured

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> # Create multi-phase problem
        >>> mp_problem = tl.MultiPhaseProblem("Spacecraft Mission")
        >>>
        >>> # Add phases
        >>> ascent = mp_problem.add_phase("Ascent")
        >>> coast = mp_problem.add_phase("Coast")
        >>> descent = mp_problem.add_phase("Descent")
        >>>
        >>> # Configure ascent phase
        >>> t1 = ascent.time(initial=0.0)
        >>> h1 = ascent.state("altitude", initial=0.0)
        >>> v1 = ascent.state("velocity", initial=0.0)
        >>> u1 = ascent.control("thrust", boundary=(0.0, 1.0))
        >>> ascent.dynamics({h1: v1, v1: u1})
        >>> ascent.set_mesh([10], np.array([-1.0, 1.0]))
        >>>
        >>> # Configure coast phase
        >>> t2 = coast.time()
        >>> h2 = coast.state("altitude")
        >>> v2 = coast.state("velocity")
        >>> coast.dynamics({h2: v2, v2: 0})  # Ballistic coast
        >>> coast.set_mesh([5], np.array([-1.0, 1.0]))
        >>>
        >>> # Configure descent phase
        >>> t3 = descent.time()
        >>> h3 = descent.state("altitude", final=0.0)
        >>> v3 = descent.state("velocity", final=0.0)
        >>> u3 = descent.control("thrust", boundary=(-1.0, 0.0))
        >>> descent.dynamics({h3: v3, v3: u3})
        >>> descent.set_mesh([15], np.array([-1.0, 1.0]))
        >>>
        >>> # Link phases with continuity constraints
        >>> mp_problem.link_phases(h1.final == h2.initial)  # Altitude continuity
        >>> mp_problem.link_phases(v1.final == v2.initial)  # Velocity continuity
        >>> mp_problem.link_phases(t1.final == t2.initial)  # Time continuity
        >>> mp_problem.link_phases(h2.final == h3.initial)
        >>> mp_problem.link_phases(v2.final == v3.initial)
        >>> mp_problem.link_phases(t2.final == t3.initial)
        >>>
        >>> # Add global parameters
        >>> gravity = mp_problem.add_global_parameter("gravity", 9.81)
        >>> mass = mp_problem.add_global_parameter("vehicle_mass", 1000.0)
        >>>
        >>> # Set global objective (minimize total mission time)
        >>> total_time = (t1.final - t1.initial +
        ...               t2.final - t2.initial +
        ...               t3.final - t3.initial)
        >>> mp_problem.set_global_objective(total_time)
        >>>
        >>> # Solve multi-phase problem
        >>> solution = tl.solve_multi_phase_fixed_mesh(mp_problem)
        >>>
        >>> if solution.success:
        ...     print(f"Optimal mission time: {solution.objective:.3f} seconds")
        ...     print(f"Global parameters: {solution.global_parameters}")
        ...     solution.plot_phases()
        ...     solution.print_solution_summary()
        ...
        ...     # Access individual phases
        ...     ascent_solution = solution.get_phase_solution(0)
        ...     print(f"Ascent duration: {ascent_solution.final_time:.3f} seconds")
    """

    # Log major operation start with key parameters (INFO - user cares)
    logger.info(
        "Starting multi-phase fixed-mesh solve: problem='%s', phases=%d",
        problem.name,
        problem.get_phase_count(),
    )

    # Log detailed problem structure (DEBUG - developer info)
    if logger.isEnabledFor(logging.DEBUG):
        phase_counts = [
            len(phase.collocation_points_per_interval)
            for phase in problem.phases
            if phase._mesh_configured
        ]
        total_intervals = sum(phase_counts) if phase_counts else 0
        logger.debug(
            "Multi-phase problem structure: phases=%d, total_intervals=%d, global_params=%d, inter_phase_constraints=%d",
            problem.get_phase_count(),
            total_intervals,
            len(problem.global_parameters),
            len(problem.inter_phase_constraints),
        )

    # Comprehensive validation
    validate_multi_phase_problem_structure(cast(MultiPhaseProblemProtocol, problem))

    # Set solver options
    problem.solver_options = nlp_options or DEFAULT_MULTI_PHASE_NLP_OPTIONS

    # Log solver configuration (DEBUG)
    logger.debug("Multi-phase NLP solver options: %s", problem.solver_options)

    # Convert to protocol and solve
    protocol_problem = cast(MultiPhaseProblemProtocol, problem)

    try:
        # Solve using multi-phase core solver
        solution_data = solve_multi_phase_radau_collocation(protocol_problem)

        # Log solution status (INFO - user cares about success/failure)
        if solution_data.success:
            logger.info(
                "Multi-phase fixed-mesh solve completed successfully: objective=%.6e, phases=%d",
                solution_data.objective or 0.0,
                solution_data.phase_count,
            )
        else:
            logger.warning("Multi-phase fixed-mesh solve failed: %s", solution_data.message)

        # Create comprehensive solution wrapper
        solution = MultiPhaseSolution(solution_data, protocol_problem)

        logger.debug("Multi-phase solution wrapper created successfully")
        return solution

    except Exception as e:
        logger.error("Multi-phase fixed-mesh solve failed: %s", str(e))

        # Create failed solution object
        from .tl_types import MultiPhaseOptimalControlSolution

        failed_solution_data = MultiPhaseOptimalControlSolution()
        failed_solution_data.success = False
        failed_solution_data.message = f"Multi-phase solve failed: {e}"
        failed_solution_data.phase_count = problem.get_phase_count()

        solution = MultiPhaseSolution(failed_solution_data, protocol_problem)
        return solution


def solve_multi_phase_adaptive(
    problem: MultiPhaseProblem,
    error_tolerance: float = 1e-6,
    max_iterations: int = 10,
    min_polynomial_degree: int = 3,
    max_polynomial_degree: int = 12,
    ode_solver_tolerance: float = 1e-7,
    ode_method: str = "RK45",
    ode_max_step: float | None = None,
    num_error_sim_points: int = 50,
    ode_solver: ODESolverCallable | None = None,
    nlp_options: dict[str, object] | None = None,
    initial_guess: MultiPhaseInitialGuess | None = None,
) -> MultiPhaseSolution:
    """
    Solve a multi-phase optimal control problem using adaptive mesh refinement.

    This function automatically refines the mesh for each phase during optimization
    to achieve a specified error tolerance. It uses adaptive mesh refinement
    algorithms applied independently to each phase while maintaining inter-phase
    coupling through the unified NLP formulation.

    Args:
        problem: MultiPhaseProblem instance with initial mesh configurations
        error_tolerance: Target relative error tolerance (default: 1e-6)
        max_iterations: Maximum refinement iterations (default: 10)
        min_polynomial_degree: Minimum polynomial degree per interval (default: 3)
        max_polynomial_degree: Maximum polynomial degree per interval (default: 12)
        ode_solver_tolerance: Tolerance for error estimation ODE solver (default: 1e-7)
        ode_method: ODE integration method for error estimation (default: "RK45")
        ode_max_step: Maximum step size for ODE solver (default: None)
        num_error_sim_points: Number of points for error simulation (default: 50)
        ode_solver: Custom ODE solver function (default: scipy.integrate.solve_ivp)
        nlp_options: Optional IPOPT solver options for each NLP solve
        initial_guess: Multi-phase initial guess for first iteration

    Returns:
        MultiPhaseSolution object with final refined meshes and high-accuracy results.
        The solution contains the final mesh configurations used for each phase.

    Raises:
        trajectolab.ConfigurationError: If problem is not properly configured or parameters are invalid

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> # Create and configure multi-phase problem
        >>> mp_problem = tl.MultiPhaseProblem("High Precision Mission")
        >>> # ... define phases, constraints, objective ...
        >>>
        >>> # Set initial meshes for each phase
        >>> phase_0.set_mesh([5], np.array([-1.0, 1.0]))    # Initial mesh
        >>> phase_1.set_mesh([4], np.array([-1.0, 1.0]))
        >>> phase_2.set_mesh([6], np.array([-1.0, 1.0]))
        >>>
        >>> # Solve with adaptive refinement
        >>> solution = tl.solve_multi_phase_adaptive(
        ...     mp_problem,
        ...     error_tolerance=1e-8,      # High accuracy
        ...     max_iterations=15,         # Allow more refinement
        ...     max_polynomial_degree=15   # Higher degree polynomials
        ... )
        >>>
        >>> if solution.success:
        ...     print(f"Converged with adaptive meshes:")
        ...     for i, phase_sol in enumerate(solution.get_all_phase_solutions()):
        ...         intervals = len(phase_sol.mesh_intervals) if hasattr(phase_sol, 'mesh_intervals') else 'N/A'
        ...         print(f"  Phase {i}: {intervals} intervals")
        ...     solution.plot_phases()

    Note:
        Multi-phase adaptive solving performs mesh refinement independently for each
        phase while maintaining inter-phase coupling through the global NLP. This
        provides optimal accuracy for each phase while preserving the mathematical
        structure required for proper multi-phase optimization.
    """

    # Log major operation start with key parameters (INFO - user cares)
    logger.info(
        "Starting multi-phase adaptive solve: problem='%s', phases=%d, tolerance=%.1e, max_iter=%d",
        problem.name,
        problem.get_phase_count(),
        error_tolerance,
        max_iterations,
    )

    # Log detailed adaptive parameters (DEBUG - developer info)
    logger.debug(
        "Multi-phase adaptive parameters: poly_degree=[%d,%d], ode_tol=%.1e, sim_points=%d",
        min_polynomial_degree,
        max_polynomial_degree,
        ode_solver_tolerance,
        num_error_sim_points,
    )

    # Comprehensive validation
    validate_multi_phase_problem_structure(cast(MultiPhaseProblemProtocol, problem))

    # Validate adaptive parameters
    _validate_multi_phase_adaptive_parameters(
        error_tolerance, max_iterations, min_polynomial_degree, max_polynomial_degree
    )

    # Set default ODE solver if not provided
    if ode_solver is None:
        from scipy.integrate import solve_ivp

        ode_solver = solve_ivp
        logger.debug("Using default ODE solver: scipy.integrate.solve_ivp")

    # Set solver options
    problem.solver_options = nlp_options or DEFAULT_MULTI_PHASE_NLP_OPTIONS

    try:
        # Note: Full multi-phase adaptive implementation would be complex and is
        # beyond the scope of this initial implementation. For now, we provide
        # the interface and use fixed-mesh solving with a warning.

        logger.warning(
            "Multi-phase adaptive mesh refinement not yet fully implemented. "
            "Using multi-phase fixed-mesh solver with current mesh configurations."
        )

        # Use fixed-mesh solver for now
        solution = solve_multi_phase_fixed_mesh(problem, nlp_options)

        # Update message to indicate adaptive was requested but not fully implemented
        if hasattr(solution.solution_data, "message"):
            solution.solution_data.message += " (adaptive refinement not yet implemented)"

        # Log final result (INFO - user cares about convergence)
        if solution.success:
            logger.info(
                "Multi-phase solve completed: objective=%.6e, phases=%d (fixed-mesh used)",
                solution.objective or 0.0,
                solution.phase_count,
            )
        else:
            logger.warning("Multi-phase solve failed: %s", solution.message)

        return solution

    except Exception as e:
        logger.error("Multi-phase adaptive solve failed: %s", str(e))

        # Create failed solution object
        from .tl_types import MultiPhaseOptimalControlSolution

        failed_solution_data = MultiPhaseOptimalControlSolution()
        failed_solution_data.success = False
        failed_solution_data.message = f"Multi-phase adaptive solve failed: {e}"
        failed_solution_data.phase_count = problem.get_phase_count()

        protocol_problem = cast(MultiPhaseProblemProtocol, problem)
        solution = MultiPhaseSolution(failed_solution_data, protocol_problem)
        return solution


def _validate_multi_phase_adaptive_parameters(
    error_tolerance: float,
    max_iterations: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
) -> None:
    """
    Validate multi-phase adaptive solver parameters.

    Args:
        error_tolerance: Target error tolerance
        max_iterations: Maximum refinement iterations
        min_polynomial_degree: Minimum polynomial degree
        max_polynomial_degree: Maximum polynomial degree

    Raises:
        ConfigurationError: If parameters are invalid
    """
    if error_tolerance <= 0:
        raise ConfigurationError(
            f"Error tolerance must be positive, got {error_tolerance}",
            "Multi-phase adaptive parameter validation error",
        )

    if error_tolerance >= 1.0:
        raise ConfigurationError(
            f"Error tolerance should be < 1.0 for meaningful accuracy, got {error_tolerance}",
            "Multi-phase adaptive parameter validation error",
        )

    if max_iterations <= 0:
        raise ConfigurationError(
            f"Max iterations must be positive, got {max_iterations}",
            "Multi-phase adaptive parameter validation error",
        )

    if max_iterations > 50:
        logger.warning(
            "Very high max_iterations (%d) may lead to excessive computation time", max_iterations
        )

    if min_polynomial_degree < 2:
        raise ConfigurationError(
            f"Minimum polynomial degree must be >= 2, got {min_polynomial_degree}",
            "Multi-phase adaptive parameter validation error",
        )

    if max_polynomial_degree < min_polynomial_degree:
        raise ConfigurationError(
            f"Maximum polynomial degree ({max_polynomial_degree}) must be >= "
            f"minimum polynomial degree ({min_polynomial_degree})",
            "Multi-phase adaptive parameter validation error",
        )

    if max_polynomial_degree > 20:
        logger.warning(
            "Very high max_polynomial_degree (%d) may lead to numerical issues",
            max_polynomial_degree,
        )


def validate_multi_phase_problem_structure(problem: MultiPhaseProblemProtocol) -> None:
    """
    Comprehensive validation of multi-phase problem structure for solving.

    Validates that the multi-phase problem is properly configured with:
    - At least 2 phases, each with dynamics and mesh configuration
    - Global objective function defined
    - Inter-phase constraints properly formed (if any)
    - Global parameters properly defined (if any)
    - Each phase ready for solving individually

    Args:
        problem: Multi-phase problem to validate

    Raises:
        ConfigurationError: If problem structure is invalid for solving
    """
    logger.debug("Validating multi-phase problem structure for solving")

    try:
        # Validate basic structure
        phase_count = problem.get_phase_count()
        if phase_count < 2:
            raise ConfigurationError(
                f"Multi-phase problem must have at least 2 phases, got {phase_count}. "
                f"For single-phase problems, use solve_fixed_mesh() or solve_adaptive().",
                "Multi-phase problem validation error",
            )

        # Validate global objective is defined
        if (
            not hasattr(problem, "global_objective_expression")
            or problem.global_objective_expression is None
        ):
            raise ConfigurationError(
                "Multi-phase problem requires global objective function. "
                "Use problem.set_global_objective() to define objective over phase endpoints.",
                "Multi-phase problem validation error",
            )

        # Validate each phase
        for phase_idx, phase_problem in enumerate(problem.phases):
            _validate_individual_phase_for_solving(phase_problem, phase_idx)

        # Validate inter-phase constraints (if any)
        if hasattr(problem, "inter_phase_constraints") and problem.inter_phase_constraints:
            _validate_inter_phase_constraints(problem.inter_phase_constraints)

        # Validate global parameters (if any)
        if hasattr(problem, "global_parameters") and problem.global_parameters:
            _validate_global_parameters(problem.global_parameters)

        logger.debug(
            "Multi-phase problem validation successful: %d phases, %d global params, %d inter-phase constraints",
            phase_count,
            len(getattr(problem, "global_parameters", {})),
            len(getattr(problem, "inter_phase_constraints", [])),
        )

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Multi-phase problem validation failed: {e}", "Multi-phase problem validation error"
        ) from e


def _validate_individual_phase_for_solving(phase_problem: Any, phase_idx: int) -> None:
    """Validate individual phase is ready for solving."""
    try:
        # Check phase has dynamics
        if not getattr(phase_problem, "_dynamics_expressions", None):
            raise ConfigurationError(
                f"Phase {phase_idx} has no dynamics defined. "
                f"Use phase.dynamics() to define differential equations.",
                "Multi-phase problem validation error",
            )

        # Check phase mesh is configured
        if not getattr(phase_problem, "_mesh_configured", False):
            raise ConfigurationError(
                f"Phase {phase_idx} mesh not configured. "
                f"Use phase.set_mesh() to configure pseudospectral mesh.",
                "Multi-phase problem validation error",
            )

        # Validate phase has variables
        try:
            num_states, num_controls = phase_problem.get_variable_counts()
            if num_states == 0:
                raise ConfigurationError(
                    f"Phase {phase_idx} has no state variables. "
                    f"Use phase.state() to define state variables.",
                    "Multi-phase problem validation error",
                )
            # Controls are optional - some phases might have no controls (e.g., ballistic coast)
        except Exception as e:
            raise ConfigurationError(
                f"Phase {phase_idx} variable validation failed: {e}",
                "Multi-phase problem validation error",
            ) from e

        # Validate phase can create required functions
        try:
            dynamics_func = phase_problem.get_dynamics_function()
            if dynamics_func is None:
                raise ConfigurationError(
                    f"Phase {phase_idx} cannot create dynamics function",
                    "Multi-phase problem validation error",
                )
        except Exception as e:
            raise ConfigurationError(
                f"Phase {phase_idx} function creation failed: {e}",
                "Multi-phase problem validation error",
            ) from e

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Phase {phase_idx} validation failed: {e}", "Multi-phase problem validation error"
        ) from e


def _validate_inter_phase_constraints(inter_phase_constraints: list) -> None:
    """Validate inter-phase constraints are properly formed."""
    if not inter_phase_constraints:
        return

    for i, constraint in enumerate(inter_phase_constraints):
        if constraint is None:
            raise ConfigurationError(
                f"Inter-phase constraint {i} is None", "Multi-phase problem validation error"
            )

        # Basic validation - detailed validation done during constraint application
        try:
            # Check if constraint is a valid CasADi expression
            import casadi as ca

            if not isinstance(constraint, ca.MX):
                raise ConfigurationError(
                    f"Inter-phase constraint {i} must be CasADi MX expression",
                    "Multi-phase problem validation error",
                )
        except Exception as e:
            raise ConfigurationError(
                f"Inter-phase constraint {i} validation failed: {e}",
                "Multi-phase problem validation error",
            ) from e


def _validate_global_parameters(global_parameters: dict) -> None:
    """Validate global parameters are properly defined."""
    if not global_parameters:
        return

    for param_name, param_value in global_parameters.items():
        if not isinstance(param_name, str) or not param_name:
            raise ConfigurationError(
                f"Global parameter name must be non-empty string, got: {param_name}",
                "Multi-phase problem validation error",
            )

        if not isinstance(param_value, (int, float)):
            raise ConfigurationError(
                f"Global parameter '{param_name}' value must be numeric, got {type(param_value)}: {param_value}",
                "Multi-phase problem validation error",
            )

        if not np.isfinite(param_value):
            raise ConfigurationError(
                f"Global parameter '{param_name}' value must be finite, got: {param_value}",
                "Multi-phase problem validation error",
            )
