"""
PHS adaptive algorithm implementation as a pure function.
"""

from typing import cast

import numpy as np

from trajectolab.adaptive.phs.data_structures import (
    AdaptiveParameters,
    _extract_and_prepare_array,
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
from trajectolab.direct_solver import InitialGuess, OptimalControlSolution
from trajectolab.radau import (
    RadauBasisComponents,
    compute_barycentric_weights,
    compute_radau_collocation_components,
)
from trajectolab.tl_types import (
    ControlEvaluator,
    FloatArray,
    ProblemProtocol,
    StateEvaluator,
)


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

    Args:
        problem: The optimal control problem
        initial_polynomial_degrees: Initial polynomial degrees per interval
        initial_mesh_points: Initial mesh points in normalized time domain [-1, 1]
        error_tolerance: Error tolerance threshold for mesh refinement
        max_iterations: Maximum number of refinement iterations
        min_polynomial_degree: Minimum polynomial degree allowed
        max_polynomial_degree: Maximum polynomial degree allowed
        ode_solver_tolerance: ODE solver tolerance for error estimation
        num_error_sim_points: Number of simulation points for error estimation
        initial_guess: Optional initial guess for first iteration

    Returns:
        Final optimized solution

    Raises:
        ValueError: If problem configuration is invalid
    """
    # Validate REQUIRED parameters
    if not initial_polynomial_degrees:
        raise ValueError("initial_polynomial_degrees must be provided and non-empty")

    if initial_mesh_points is None:
        raise ValueError("initial_mesh_points must be provided")

    # Validate mesh configuration compatibility
    if len(initial_polynomial_degrees) != len(initial_mesh_points) - 1:
        raise ValueError(
            f"Number of polynomial degrees ({len(initial_polynomial_degrees)}) "
            f"must be one less than number of mesh points ({len(initial_mesh_points)})"
        )

    # Store and validate adaptive parameters
    adaptive_params = AdaptiveParameters(
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        min_polynomial_degree=min_polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        ode_solver_tolerance=ode_solver_tolerance,
        num_error_sim_points=num_error_sim_points,
    )

    # Store initial mesh configuration
    current_polynomial_degrees = list(initial_polynomial_degrees)
    current_mesh_points = initial_mesh_points.copy()

    # Validate mesh points
    if not np.isclose(current_mesh_points[0], -1.0):
        raise ValueError(f"First mesh point must be -1.0, got {current_mesh_points[0]}")

    if not np.isclose(current_mesh_points[-1], 1.0):
        raise ValueError(f"Last mesh point must be 1.0, got {current_mesh_points[-1]}")

    if not np.all(np.diff(current_mesh_points) > 1e-9):
        raise ValueError("Mesh points must be strictly increasing with minimum spacing of 1e-9")

    # Validate and enforce polynomial degree bounds
    N_min = adaptive_params.min_polynomial_degree
    N_max = adaptive_params.max_polynomial_degree

    for i in range(len(current_polynomial_degrees)):
        if current_polynomial_degrees[i] < N_min:
            raise ValueError(
                f"Initial polynomial degree {current_polynomial_degrees[i]} for interval {i} "
                f"is below minimum {N_min}"
            )
        if current_polynomial_degrees[i] > N_max:
            raise ValueError(
                f"Initial polynomial degree {current_polynomial_degrees[i]} for interval {i} "
                f"is above maximum {N_max}"
            )

    # Track most recent successful solution
    most_recent_solution: OptimalControlSolution | None = None

    # Import solver function
    from trajectolab.direct_solver import solve_single_phase_radau_collocation

    # Main adaptive refinement loop
    for iteration in range(max_iterations):
        print(f"\n--- Adaptive Iteration {iteration} ---")
        num_intervals = len(current_polynomial_degrees)

        # Configure problem mesh
        problem.set_mesh(current_polynomial_degrees, current_mesh_points)

        # Handle initial guess based on iteration
        if iteration == 0:
            # First iteration: use provided initial guess or let CasADi handle it
            if problem.initial_guess is not None:
                print("  Using initial guess from problem.set_initial_guess()")
                # Validate that user's guess matches the configured mesh
                try:
                    problem.validate_initial_guess()
                except ValueError as e:
                    raise ValueError(
                        f"Initial guess from problem.set_initial_guess() invalid for mesh: {e}"
                    ) from e
            elif initial_guess is not None:
                print("  Using initial guess from solve_adaptive() call")
                problem.initial_guess = initial_guess
                # Validate that provided guess matches the configured mesh
                try:
                    problem.validate_initial_guess()
                except ValueError as e:
                    raise ValueError(
                        f"Initial guess from solve_adaptive() call invalid for mesh: {e}"
                    ) from e
            else:
                print("  No initial guess provided. CasADi will use defaults (typically zeros)")
                problem.initial_guess = None

        else:
            # Subsequent iterations: propagate from previous solution
            if most_recent_solution is None:
                raise ValueError("No previous solution available for propagation")

            print("  Propagating initial guess from previous solution")
            propagated_guess = propagate_solution_to_new_mesh(
                most_recent_solution,
                problem,
                current_polynomial_degrees,
                current_mesh_points,
            )
            problem.initial_guess = propagated_guess

        # Log current mesh configuration
        print(f"  Mesh: {num_intervals} intervals, degrees = {current_polynomial_degrees}")
        print(f"  Mesh points: {np.array2string(current_mesh_points, precision=3)}")

        # Solve optimal control problem
        solution = solve_single_phase_radau_collocation(problem)

        if not solution.success:
            error_msg = f"Solver failed in adaptive iteration {iteration}: {solution.message}"
            print(f"  Error: {error_msg}")

            # Return best solution available
            if most_recent_solution is not None:
                most_recent_solution.message = (
                    f"Adaptive stopped due to solver failure: {error_msg}"
                )
                most_recent_solution.success = False
                return most_recent_solution
            else:
                solution.message = error_msg
                return solution

        # Extract solved trajectories for error estimation
        try:
            if solution.raw_solution is None or solution.opti_object is None:
                raise ValueError("Missing raw solution or opti object")

            opti = solution.opti_object
            raw_sol = solution.raw_solution

            # Extract state trajectories
            if hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables"):
                solution.solved_state_trajectories_per_interval = [
                    _extract_and_prepare_array(
                        raw_sol.value(var),
                        len(problem._states),
                        current_polynomial_degrees[i] + 1,
                    )
                    for i, var in enumerate(
                        opti.state_at_local_approximation_nodes_all_intervals_variables
                    )
                ]

            # Extract control trajectories
            if len(problem._controls) > 0 and hasattr(
                opti, "control_at_local_collocation_nodes_all_intervals_variables"
            ):
                solution.solved_control_trajectories_per_interval = [
                    _extract_and_prepare_array(
                        raw_sol.value(var),
                        len(problem._controls),
                        current_polynomial_degrees[i],
                    )
                    for i, var in enumerate(
                        opti.control_at_local_collocation_nodes_all_intervals_variables
                    )
                ]
            else:
                solution.solved_control_trajectories_per_interval = [
                    np.empty((0, current_polynomial_degrees[i]), dtype=np.float64)
                    for i in range(num_intervals)
                ]

        except Exception as extract_error:
            error_msg = f"Failed to extract trajectories: {extract_error}"
            print(f"  Error: {error_msg}")
            solution.message = error_msg
            solution.success = False
            return solution

        # Update most recent successful solution
        most_recent_solution = solution
        most_recent_solution.num_collocation_nodes_list_at_solve_time = (
            current_polynomial_degrees.copy()
        )
        most_recent_solution.global_mesh_nodes_at_solve_time = current_mesh_points.copy()

        # Calculate gamma normalization factors
        gamma_factors = calculate_gamma_normalizers(solution, problem)
        if gamma_factors is None and len(problem._states) > 0:
            error_msg = f"Failed to calculate gamma normalizers at iteration {iteration}"
            print(f"  Error: {error_msg}")
            solution.message = error_msg
            solution.success = False
            return solution

        # Create interpolants for error estimation
        basis_cache: dict[int, RadauBasisComponents] = {}
        control_weights_cache: dict[int, FloatArray] = {}
        state_evaluators: list[StateEvaluator | None] = [None] * num_intervals
        control_evaluators: list[ControlEvaluator | None] = [None] * num_intervals

        states_list = solution.solved_state_trajectories_per_interval
        controls_list = solution.solved_control_trajectories_per_interval

        for k in range(num_intervals):
            try:
                N_k = current_polynomial_degrees[k]

                # Use cached basis components
                if N_k not in basis_cache:
                    basis_cache[N_k] = compute_radau_collocation_components(N_k)

                basis = basis_cache[N_k]

                # Create state interpolant
                if states_list is None:
                    raise ValueError(f"Missing state trajectories for interval {k}")

                state_data = states_list[k]
                state_evaluators[k] = get_polynomial_interpolant(
                    basis.state_approximation_nodes,
                    state_data,
                    basis.barycentric_weights_for_state_nodes,
                )

                # Create control interpolant
                if len(problem._controls) > 0:
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

            except Exception as interp_error:
                print(f"  Warning: Error creating interpolant for interval {k}: {interp_error}")
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

        # Calculate error estimates for each interval
        errors: list[float] = [np.inf] * num_intervals

        for k in range(num_intervals):
            print(f"  Estimating error for interval {k}...")

            state_eval = state_evaluators[k]
            control_eval = control_evaluators[k]

            if state_eval is None or control_eval is None:
                print(f"    Warning: Missing interpolants for interval {k}")
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
            safe_gamma = (
                gamma_factors
                if gamma_factors is not None
                else np.ones((len(problem._states), 1), dtype=np.float64)
            )
            error = calculate_relative_error_estimate(k, sim_bundle, safe_gamma)

            errors[k] = error
            print(f"    Interval {k}: N={current_polynomial_degrees[k]}, Error={error:.4e}")

        print(f"  Error estimates: {[f'{e:.2e}' for e in errors]}")

        # Check convergence
        all_errors_within_tolerance = True
        if num_intervals == 0:
            all_errors_within_tolerance = True
        elif not errors:
            all_errors_within_tolerance = False
        else:
            for error in errors:
                if np.isnan(error) or np.isinf(error) or error > adaptive_params.error_tolerance:
                    all_errors_within_tolerance = False
                    break

        # If converged, return solution
        if all_errors_within_tolerance:
            print(f"Converged after {iteration + 1} iterations!")
            solution.num_collocation_nodes_per_interval = current_polynomial_degrees.copy()
            solution.global_normalized_mesh_nodes = current_mesh_points.copy()
            solution.message = (
                f"Adaptive mesh converged to tolerance {adaptive_params.error_tolerance:.1e} "
                f"in {iteration + 1} iterations"
            )
            return solution

        # Refine mesh for next iteration
        next_polynomial_degrees: list[int] = []
        next_mesh_points = [current_mesh_points[0]]

        k = 0
        while k < num_intervals:
            error_k = errors[k]
            N_k = current_polynomial_degrees[k]

            print(f"    Processing interval {k}: N={N_k}, Error={error_k:.2e}")

            if np.isnan(error_k) or np.isinf(error_k) or error_k > adaptive_params.error_tolerance:
                # Error above tolerance - try p-refinement
                print("      Error > tolerance, attempting p-refinement")
                p_result = p_refine_interval(error_k, N_k, adaptive_params.error_tolerance, N_max)

                if p_result.was_p_successful:
                    # p-refinement successful
                    print(f"        p-refinement: N {N_k} -> {p_result.actual_Nk_to_use}")
                    next_polynomial_degrees.append(p_result.actual_Nk_to_use)
                    next_mesh_points.append(current_mesh_points[k + 1])
                    k += 1
                else:
                    # p-refinement failed - apply h-refinement
                    print("        p-refinement failed, applying h-refinement")
                    h_result = h_refine_params(p_result.unconstrained_target_Nk, N_min)

                    print(
                        f"          h-refinement: Split into {h_result.num_new_subintervals} "
                        f"subintervals, each N={h_result.collocation_nodes_for_new_subintervals[0]}"
                    )
                    next_polynomial_degrees.extend(h_result.collocation_nodes_for_new_subintervals)

                    # Create new mesh nodes for subintervals
                    tau_start = current_mesh_points[k]
                    tau_end = current_mesh_points[k + 1]
                    new_nodes = np.linspace(
                        tau_start, tau_end, h_result.num_new_subintervals + 1, dtype=np.float64
                    )
                    next_mesh_points.extend(new_nodes[1:].tolist())
                    k += 1
            else:
                # Error within tolerance - check for h-reduction or p-reduction
                print(f"    Interval {k}: Error within tolerance")

                # Try h-reduction with next interval
                can_merge = False
                if k < num_intervals - 1:
                    next_error = errors[k + 1]
                    if (
                        not (np.isnan(next_error) or np.isinf(next_error))
                        and next_error <= adaptive_params.error_tolerance
                    ):
                        print("      Next interval also has low error, checking h-reduction")

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

                            safe_gamma = (
                                gamma_factors
                                if gamma_factors is not None
                                else np.ones((len(problem._states), 1), dtype=np.float64)
                            )

                            can_merge = h_reduce_intervals(
                                k,
                                solution,
                                problem,
                                adaptive_params,
                                safe_gamma,
                                state_eval_first,
                                control_eval_first,
                                state_eval_second,
                                control_eval_second,
                            )

                if can_merge:
                    # h-reduction successful
                    print(f"      h-reduction: Merged intervals {k} and {k + 1}")
                    merged_N = max(current_polynomial_degrees[k], current_polynomial_degrees[k + 1])
                    merged_N = max(N_min, min(N_max, merged_N))
                    next_polynomial_degrees.append(merged_N)
                    next_mesh_points.append(current_mesh_points[k + 2])
                    k += 2
                else:
                    # Try p-reduction
                    print("      h-reduction not applicable, attempting p-reduction")
                    p_reduce = p_reduce_interval(
                        N_k, error_k, adaptive_params.error_tolerance, N_min, N_max
                    )

                    if p_reduce.was_reduction_applied:
                        print(
                            f"        p-reduction: N {N_k} -> {p_reduce.new_num_collocation_nodes}"
                        )
                    else:
                        print("        p-reduction not applied")

                    next_polynomial_degrees.append(p_reduce.new_num_collocation_nodes)
                    next_mesh_points.append(current_mesh_points[k + 1])
                    k += 1

        # Update for next iteration
        current_polynomial_degrees = next_polynomial_degrees
        current_mesh_points = np.array(next_mesh_points, dtype=np.float64)

        # Validate new mesh
        if not current_polynomial_degrees:
            error_msg = "Mesh refinement resulted in empty polynomial degrees"
            print(f"  Error: {error_msg}")
            if most_recent_solution is not None:
                most_recent_solution.message = error_msg
                most_recent_solution.success = False
                return most_recent_solution

        if len(current_polynomial_degrees) != len(current_mesh_points) - 1:
            error_msg = (
                f"Mesh inconsistency: {len(current_polynomial_degrees)} degrees "
                f"vs {len(current_mesh_points)} mesh points"
            )
            print(f"  Error: {error_msg}")
            if most_recent_solution is not None:
                most_recent_solution.message = error_msg
                most_recent_solution.success = False
                return most_recent_solution

        # Check for duplicate mesh points
        if len(current_mesh_points) > 1:
            diffs = np.diff(current_mesh_points)
            if np.any(diffs <= 1e-9):
                problem_indices = np.where(diffs <= 1e-9)[0]
                error_msg = (
                    f"Mesh points too close or non-increasing at indices {problem_indices}. "
                    f"Points: {current_mesh_points}"
                )
                print(f"  Error: {error_msg}")
                if most_recent_solution is not None:
                    most_recent_solution.message = error_msg
                    most_recent_solution.success = False
                    return most_recent_solution

    # Maximum iterations reached
    max_iter_msg = (
        f"Reached maximum iterations ({max_iterations}) without full convergence "
        f"to tolerance {adaptive_params.error_tolerance:.1e}"
    )
    print(max_iter_msg)

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
