"""
PHS adaptive algorithm implementation as a pure function.
"""

import logging
from typing import cast

import numpy as np

from trajectolab.adaptive.phs.data_structures import (
    AdaptiveParameters,
    extract_and_prepare_array,
)
from trajectolab.adaptive.phs.error_estimation import (
    calculate_gamma_normalizers,
    calculate_relative_error_estimate,
    simulate_dynamics_for_error_estimation,
)
from trajectolab.adaptive.phs.initial_guess import (
    propagate_solution_to_new_mesh,
)
from trajectolab.adaptive.phs.numerical import (
    get_polynomial_interpolant,
)
from trajectolab.adaptive.phs.refinement import (
    h_reduce_intervals,
    h_refine_params,
    p_reduce_interval,
    p_refine_interval,
)
from trajectolab.radau import (
    RadauBasisComponents,
    compute_barycentric_weights,
    compute_radau_collocation_components,
)
from trajectolab.tl_types import (
    ControlEvaluator,
    FloatArray,
    InitialGuess,
    OptimalControlSolution,
    ProblemProtocol,
    StateEvaluator,
)


logger = logging.getLogger(__name__)


def _validate_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    min_degree: int,
    max_degree: int,
) -> None:
    """Validate mesh configuration parameters."""
    if not polynomial_degrees:
        raise ValueError("polynomial_degrees must be provided and non-empty")

    if len(polynomial_degrees) != len(mesh_points) - 1:
        raise ValueError(
            f"Number of polynomial degrees ({len(polynomial_degrees)}) "
            f"must be one less than number of mesh points ({len(mesh_points)})"
        )

    if not np.isclose(mesh_points[0], -1.0):
        raise ValueError(f"First mesh point must be -1.0, got {mesh_points[0]}")

    if not np.isclose(mesh_points[-1], 1.0):
        raise ValueError(f"Last mesh point must be 1.0, got {mesh_points[-1]}")

    if not np.all(np.diff(mesh_points) > 1e-9):
        raise ValueError("Mesh points must be strictly increasing with minimum spacing of 1e-9")

    # Validate polynomial degree bounds
    for i, degree in enumerate(polynomial_degrees):
        if degree < min_degree:
            raise ValueError(
                f"Polynomial degree {degree} for interval {i} is below minimum {min_degree}"
            )
        if degree > max_degree:
            raise ValueError(
                f"Polynomial degree {degree} for interval {i} is above maximum {max_degree}"
            )


def _extract_solution_trajectories(
    solution: OptimalControlSolution, problem: ProblemProtocol, polynomial_degrees: list[int]
) -> None:
    """Extract state and control trajectories from the solution."""
    if solution.raw_solution is None or solution.opti_object is None:
        raise ValueError("Missing raw solution or opti object")

    opti = solution.opti_object
    raw_sol = solution.raw_solution

    # Extract state trajectories
    if hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables"):
        solution.solved_state_trajectories_per_interval = [
            extract_and_prepare_array(
                raw_sol.value(var),
                len(problem._states),
                polynomial_degrees[i] + 1,
            )
            for i, var in enumerate(opti.state_at_local_approximation_nodes_all_intervals_variables)
        ]

    # Extract control trajectories
    if len(problem._controls) > 0 and hasattr(
        opti, "control_at_local_collocation_nodes_all_intervals_variables"
    ):
        solution.solved_control_trajectories_per_interval = [
            extract_and_prepare_array(
                raw_sol.value(var),
                len(problem._controls),
                polynomial_degrees[i],
            )
            for i, var in enumerate(opti.control_at_local_collocation_nodes_all_intervals_variables)
        ]
    else:
        solution.solved_control_trajectories_per_interval = [
            np.empty((0, polynomial_degrees[i]), dtype=np.float64)
            for i in range(len(polynomial_degrees))
        ]


def _create_interpolants(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    polynomial_degrees: list[int],
) -> tuple[list[StateEvaluator | None], list[ControlEvaluator | None]]:
    """Create state and control interpolants for all intervals."""
    num_intervals = len(polynomial_degrees)
    basis_cache: dict[int, RadauBasisComponents] = {}
    control_weights_cache: dict[int, FloatArray] = {}
    state_evaluators: list[StateEvaluator | None] = [None] * num_intervals
    control_evaluators: list[ControlEvaluator | None] = [None] * num_intervals

    states_list = solution.solved_state_trajectories_per_interval
    controls_list = solution.solved_control_trajectories_per_interval

    if states_list is None:
        raise ValueError("Missing state trajectories for interpolant creation")

    if controls_list is None and len(problem._controls) > 0:
        raise ValueError("Missing control trajectories for interpolant creation")

    for k in range(num_intervals):
        try:
            N_k = polynomial_degrees[k]

            # Use cached basis components
            if N_k not in basis_cache:
                basis_cache[N_k] = compute_radau_collocation_components(N_k)

            basis = basis_cache[N_k]

            # Create state interpolant
            state_data = states_list[k]
            state_evaluators[k] = get_polynomial_interpolant(
                basis.state_approximation_nodes,
                state_data,
                basis.barycentric_weights_for_state_nodes,
            )

            # Create control interpolant
            if len(problem._controls) > 0 and controls_list is not None:
                control_data = controls_list[k]

                if N_k not in control_weights_cache:
                    control_weights_cache[N_k] = compute_barycentric_weights(
                        basis.collocation_nodes
                    )

                control_weights = control_weights_cache[N_k]
                control_evaluators[k] = get_polynomial_interpolant(
                    basis.collocation_nodes, control_data, control_weights
                )
            else:
                # Empty control interpolant
                control_evaluators[k] = get_polynomial_interpolant(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    np.empty((0, 2), dtype=np.float64),
                    None,
                )

        except Exception as e:
            logger.warning(f"Error creating interpolant for interval {k}: {e}")
            # Create fallback interpolants
            if state_evaluators[k] is None:
                state_evaluators[k] = get_polynomial_interpolant(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    np.full((len(problem._states), 2), np.nan, dtype=np.float64),
                    None,
                )
            if control_evaluators[k] is None:
                control_evaluators[k] = get_polynomial_interpolant(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    np.full(
                        (len(problem._controls) if len(problem._controls) > 0 else 0, 2),
                        np.nan,
                        dtype=np.float64,
                    ),
                    None,
                )

    return state_evaluators, control_evaluators


def _estimate_errors(
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    polynomial_degrees: list[int],
    state_evaluators: list[StateEvaluator | None],
    control_evaluators: list[ControlEvaluator | None],
    adaptive_params: AdaptiveParameters,
    gamma_factors: FloatArray,
) -> list[float]:
    """Estimate errors for all intervals."""
    num_intervals = len(polynomial_degrees)
    errors: list[float] = [np.inf] * num_intervals

    for k in range(num_intervals):
        state_eval = state_evaluators[k]
        control_eval = control_evaluators[k]

        if state_eval is None or control_eval is None:
            errors[k] = np.inf
            continue

        # Simulate dynamics for error estimation
        sim_bundle = simulate_dynamics_for_error_estimation(
            k,
            solution,
            problem,
            state_eval,
            control_eval,
            ode_rtol=adaptive_params.ode_solver_tolerance,
            n_eval_points=adaptive_params.num_error_sim_points,
        )

        # Calculate relative error
        error = calculate_relative_error_estimate(k, sim_bundle, gamma_factors)
        errors[k] = error

    return errors


def _refine_mesh(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    errors: list[float],
    adaptive_params: AdaptiveParameters,
    solution: OptimalControlSolution,
    problem: ProblemProtocol,
    state_evaluators: list[StateEvaluator | None],
    control_evaluators: list[ControlEvaluator | None],
    gamma_factors: FloatArray,
) -> tuple[list[int], FloatArray]:
    """Refine mesh for next iteration."""
    num_intervals = len(polynomial_degrees)
    next_polynomial_degrees: list[int] = []
    next_mesh_points = [mesh_points[0]]

    k = 0
    while k < num_intervals:
        error_k = errors[k]
        N_k = polynomial_degrees[k]

        if np.isnan(error_k) or np.isinf(error_k) or error_k > adaptive_params.error_tolerance:
            # Error above tolerance - try p-refinement
            p_result = p_refine_interval(
                error_k, N_k, adaptive_params.error_tolerance, adaptive_params.max_polynomial_degree
            )

            if p_result.was_p_successful:
                # p-refinement successful
                next_polynomial_degrees.append(p_result.actual_Nk_to_use)
                next_mesh_points.append(mesh_points[k + 1])
                k += 1
            else:
                # p-refinement failed - apply h-refinement
                h_result = h_refine_params(
                    p_result.unconstrained_target_Nk, adaptive_params.min_polynomial_degree
                )

                next_polynomial_degrees.extend(h_result.collocation_nodes_for_new_subintervals)

                # Create new mesh nodes for subintervals
                tau_start = mesh_points[k]
                tau_end = mesh_points[k + 1]
                new_nodes = np.linspace(
                    tau_start, tau_end, h_result.num_new_subintervals + 1, dtype=np.float64
                )
                next_mesh_points.extend(new_nodes[1:].tolist())
                k += 1
        else:
            # Error within tolerance - check for h-reduction or p-reduction
            can_merge = False
            if k < num_intervals - 1:
                next_error = errors[k + 1]
                if (
                    not (np.isnan(next_error) or np.isinf(next_error))
                    and next_error <= adaptive_params.error_tolerance
                ):
                    # Check if interpolants are available
                    if (
                        state_evaluators[k] is not None
                        and state_evaluators[k + 1] is not None
                        and (
                            len(problem._controls) == 0
                            or (
                                control_evaluators[k] is not None
                                and control_evaluators[k + 1] is not None
                            )
                        )
                    ):
                        state_eval_first = cast(StateEvaluator, state_evaluators[k])
                        state_eval_second = cast(StateEvaluator, state_evaluators[k + 1])
                        control_eval_first = control_evaluators[k]
                        control_eval_second = control_evaluators[k + 1]

                        can_merge = h_reduce_intervals(
                            k,
                            solution,
                            problem,
                            adaptive_params,
                            gamma_factors,
                            state_eval_first,
                            control_eval_first,
                            state_eval_second,
                            control_eval_second,
                        )

            if can_merge:
                # h-reduction successful
                merged_N = max(polynomial_degrees[k], polynomial_degrees[k + 1])
                merged_N = max(
                    adaptive_params.min_polynomial_degree,
                    min(adaptive_params.max_polynomial_degree, merged_N),
                )
                next_polynomial_degrees.append(merged_N)
                next_mesh_points.append(mesh_points[k + 2])
                k += 2
            else:
                # Try p-reduction
                p_reduce = p_reduce_interval(
                    N_k,
                    error_k,
                    adaptive_params.error_tolerance,
                    adaptive_params.min_polynomial_degree,
                    adaptive_params.max_polynomial_degree,
                )

                next_polynomial_degrees.append(p_reduce.new_num_collocation_nodes)
                next_mesh_points.append(mesh_points[k + 1])
                k += 1

    return next_polynomial_degrees, np.array(next_mesh_points, dtype=np.float64)


def solve_phs_adaptive_internal(
    problem: ProblemProtocol,
    initial_polynomial_degrees: list[int],
    initial_mesh_points: FloatArray,
    error_tolerance: float,
    max_iterations: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
    ode_solver_tolerance: float,
    num_error_sim_points: int,
    initial_guess: InitialGuess | None = None,
) -> OptimalControlSolution:
    """
    Internal PHS-Adaptive mesh refinement algorithm implementation.
    """
    # Validate parameters
    _validate_mesh_configuration(
        initial_polynomial_degrees,
        initial_mesh_points,
        min_polynomial_degree,
        max_polynomial_degree,
    )

    # Store adaptive parameters
    adaptive_params = AdaptiveParameters(
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        ode_solver_tolerance=ode_solver_tolerance,
        num_error_sim_points=num_error_sim_points,
    )

    # Initialize mesh configuration
    current_polynomial_degrees = list(initial_polynomial_degrees)
    current_mesh_points = initial_mesh_points.copy()

    # Initialize scaling if available
    use_scaling = hasattr(problem, "use_scaling") and problem.use_scaling
    if use_scaling:
        # Initial scaling computation based on problem definition
        problem._scaling.compute_from_problem(problem)

    # Track most recent successful solution
    most_recent_solution: OptimalControlSolution | None = None

    # Import solver function
    from trajectolab.direct_solver import solve_single_phase_radau_collocation

    # Main adaptive refinement loop
    for iteration in range(max_iterations):
        logger.info(f"Adaptive Iteration {iteration}")

        # Configure problem mesh
        problem.set_mesh(current_polynomial_degrees, current_mesh_points)

        if iteration == 0:
            # FIRST ITERATION: Use provided initial guess
            if problem.initial_guess is not None:
                try:
                    problem.validate_initial_guess()
                except ValueError as e:
                    raise ValueError(f"Initial guess invalid for mesh: {e}") from e
            elif initial_guess is not None:
                problem.initial_guess = initial_guess
                try:
                    problem.validate_initial_guess()
                except ValueError as e:
                    raise ValueError(f"Initial guess invalid for mesh: {e}") from e
            else:
                problem.initial_guess = None
        else:
            # SUBSEQUENT ITERATIONS: Always use aggressive interpolation propagation
            if most_recent_solution is None:
                raise ValueError("No previous solution available for propagation")

            propagated_guess = propagate_solution_to_new_mesh(
                most_recent_solution,
                problem,
                current_polynomial_degrees,
                current_mesh_points,
            )
            problem.initial_guess = propagated_guess

        # Solve optimal control problem
        solution = solve_single_phase_radau_collocation(problem)

        if not solution.success:
            error_msg = f"Solver failed in iteration {iteration}: {solution.message}"
            logger.error(error_msg)

            if most_recent_solution is not None:
                most_recent_solution.message = (
                    f"Adaptive stopped due to solver failure: {error_msg}"
                )
                most_recent_solution.success = False
                return most_recent_solution
            else:
                solution.message = error_msg
                return solution

        # Extract trajectories
        try:
            _extract_solution_trajectories(solution, problem, current_polynomial_degrees)
        except Exception as e:
            error_msg = f"Failed to extract trajectories: {e}"
            logger.error(error_msg)
            solution.message = error_msg
            solution.success = False
            return solution

        # Update most recent successful solution
        most_recent_solution = solution
        most_recent_solution.num_collocation_nodes_list_at_solve_time = (
            current_polynomial_degrees.copy()
        )
        most_recent_solution.global_mesh_nodes_at_solve_time = current_mesh_points.copy()

        # Update scaling for next iteration if needed
        if use_scaling and solution.success:
            from trajectolab.scaling import update_scaling_after_mesh_refinement

            update_scaling_after_mesh_refinement(problem._scaling, problem, solution)

        # Calculate gamma normalization factors
        gamma_factors = calculate_gamma_normalizers(solution, problem)
        if gamma_factors is None and len(problem._states) > 0:
            error_msg = f"Failed to calculate gamma normalizers at iteration {iteration}"
            logger.error(error_msg)
            solution.message = error_msg
            solution.success = False
            return solution

        # Use default gamma factors if none calculated
        safe_gamma = (
            gamma_factors
            if gamma_factors is not None
            else np.ones((len(problem._states), 1), dtype=np.float64)
        )

        # Create interpolants
        state_evaluators, control_evaluators = _create_interpolants(
            solution, problem, current_polynomial_degrees
        )

        # Calculate error estimates
        errors = _estimate_errors(
            solution,
            problem,
            current_polynomial_degrees,
            state_evaluators,
            control_evaluators,
            adaptive_params,
            safe_gamma,
        )

        # Check convergence
        all_errors_within_tolerance = all(
            not (np.isnan(error) or np.isinf(error)) and error <= adaptive_params.error_tolerance
            for error in errors
        )

        if all_errors_within_tolerance:
            logger.info(f"Converged after {iteration + 1} iterations!")
            solution.num_collocation_nodes_per_interval = current_polynomial_degrees.copy()
            solution.global_normalized_mesh_nodes = current_mesh_points.copy()
            solution.message = (
                f"Adaptive mesh converged to tolerance {adaptive_params.error_tolerance:.1e} "
                f"in {iteration + 1} iterations"
            )
            return solution

        # Refine mesh for next iteration
        try:
            current_polynomial_degrees, current_mesh_points = _refine_mesh(
                current_polynomial_degrees,
                current_mesh_points,
                errors,
                adaptive_params,
                solution,
                problem,
                state_evaluators,
                control_evaluators,
                safe_gamma,
            )
        except Exception as e:
            error_msg = f"Mesh refinement failed: {e}"
            logger.error(error_msg)
            if most_recent_solution is not None:
                most_recent_solution.message = error_msg
                most_recent_solution.success = False
                return most_recent_solution

    # Maximum iterations reached
    max_iter_msg = (
        f"Reached maximum iterations ({max_iterations}) without full convergence "
        f"to tolerance {adaptive_params.error_tolerance:.1e}"
    )
    logger.warning(max_iter_msg)

    if most_recent_solution is not None:
        most_recent_solution.message = max_iter_msg
        most_recent_solution.num_collocation_nodes_per_interval = current_polynomial_degrees.copy()
        most_recent_solution.global_normalized_mesh_nodes = current_mesh_points.copy()
        return most_recent_solution
    else:
        # Create failure solution
        failed_solution = OptimalControlSolution()
        failed_solution.success = False
        failed_solution.message = max_iter_msg + " No successful solution obtained."
        failed_solution.num_collocation_nodes_per_interval = current_polynomial_degrees
        failed_solution.global_normalized_mesh_nodes = current_mesh_points
        return failed_solution
