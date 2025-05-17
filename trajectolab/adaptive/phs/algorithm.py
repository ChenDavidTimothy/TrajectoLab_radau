"""
Main PHS adaptive algorithm implementation.
"""

from typing import Sequence, cast

import numpy as np

from trajectolab.adaptive.base import AdaptiveBase
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
    generate_robust_default_initial_guess,
    propagate_guess_from_previous,
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
    ProblemProtocol,
    _ControlEvaluator,
    _FloatArray,
    _StateEvaluator,
)


class PHSAdaptive(AdaptiveBase):
    """Implements the PHS-Adaptive mesh refinement algorithm."""

    adaptive_params: AdaptiveParameters
    _initial_polynomial_degrees: list[int] | None
    _initial_mesh_points: Sequence[float] | _FloatArray | None

    def __init__(
        self,
        error_tolerance: float = 1e-3,
        max_iterations: int = 30,
        min_polynomial_degree: int = 4,
        max_polynomial_degree: int = 16,
        ode_solver_tolerance: float = 1e-7,
        num_error_sim_points: int = 40,
        initial_polynomial_degrees: list[int] | None = None,
        initial_mesh_points: Sequence[float] | _FloatArray | None = None,
        initial_guess: InitialGuess | None = None,
    ) -> None:
        """
        Initialize the PHS-Adaptive mesh refinement algorithm.

        Args:
            error_tolerance: Error tolerance threshold for mesh refinement
            max_iterations: Maximum number of refinement iterations
            min_polynomial_degree: Minimum polynomial degree allowed
            max_polynomial_degree: Maximum polynomial degree allowed
            ode_solver_tolerance: ODE solver tolerance for error estimation
            num_error_sim_points: Number of simulation points for error estimation
            initial_polynomial_degrees: Initial list of polynomial degrees for each interval
            initial_mesh_points: Initial mesh points in normalized time domain [-1, 1]
            initial_guess: Optional initial guess for the solver
        """
        super().__init__(initial_guess)
        self.adaptive_params = AdaptiveParameters(
            error_tolerance=error_tolerance,
            max_iterations=max_iterations,
            min_polynomial_degree=min_polynomial_degree,
            max_polynomial_degree=max_polynomial_degree,
            ode_solver_tolerance=ode_solver_tolerance,
            num_error_sim_points=num_error_sim_points,
        )

        # Validate initial mesh configuration if provided
        if initial_polynomial_degrees is not None and initial_mesh_points is not None:
            if len(initial_polynomial_degrees) != len(initial_mesh_points) - 1:
                raise ValueError(
                    "Number of polynomial degrees must be one less than number of mesh points"
                )

        self._initial_polynomial_degrees = initial_polynomial_degrees
        self._initial_mesh_points = initial_mesh_points

    def run(
        self,
        problem: ProblemProtocol,
        initial_solution: OptimalControlSolution | None = None,
    ) -> OptimalControlSolution:
        """Run the PHS-Adaptive mesh refinement algorithm."""
        # Extract adaptive parameters
        error_tol = self.adaptive_params.error_tolerance
        max_iterations = self.adaptive_params.max_iterations
        N_min = self.adaptive_params.min_polynomial_degree
        N_max = self.adaptive_params.max_polynomial_degree
        ode_rtol = self.adaptive_params.ode_solver_tolerance
        num_sim_points = self.adaptive_params.num_error_sim_points

        # Initialize mesh configuration, checking modern API first
        if self._initial_polynomial_degrees is not None:
            current_nodes_list = list(self._initial_polynomial_degrees)

            if self._initial_mesh_points is not None:
                current_mesh = np.array(self._initial_mesh_points, dtype=np.float64)
            else:
                current_mesh = np.linspace(-1, 1, len(current_nodes_list) + 1, dtype=np.float64)
        # Fall back to problem defaults
        else:
            current_nodes_list = list(problem.collocation_points_per_interval)

            # Ensure we have at least one interval with minimum polynomial degree
            if not current_nodes_list:
                current_nodes_list = [N_min]

            if problem.global_normalized_mesh_nodes is not None:
                current_mesh = np.array(problem.global_normalized_mesh_nodes, dtype=np.float64)
            else:
                current_mesh = np.linspace(-1, 1, len(current_nodes_list) + 1, dtype=np.float64)

        # Enforce node count limits
        for i in range(len(current_nodes_list)):
            current_nodes_list[i] = max(N_min, min(N_max, current_nodes_list[i]))

        # Keep track of most recent solution
        most_recent_solution = initial_solution
        from trajectolab.direct_solver import solve_single_phase_radau_collocation

        # Main adaptive refinement loop
        for iteration in range(max_iterations):
            print(f"\n--- Adaptive Iteration M = {iteration} ---")
            num_intervals = len(current_nodes_list)

            # Update problem definition with current mesh
            problem.collocation_points_per_interval = list(current_nodes_list)
            # Convert list to NumPy array for global_normalized_mesh_nodes
            problem.global_normalized_mesh_nodes = np.array(current_mesh, dtype=np.float64)

            # Generate initial guess
            if iteration == 0:
                if self.initial_guess is not None:  # Use provided initial guess for first iteration
                    print("  Using user-provided initial guess for first iteration.")
                    initial_guess = self.initial_guess
                else:
                    print("  No user-provided initial guess. Using robust default.")
                    initial_guess = generate_robust_default_initial_guess(
                        problem, current_nodes_list
                    )
            elif most_recent_solution is None or not most_recent_solution.success:
                print("  Previous NLP failed. Using robust default initial guess.")
                initial_guess = generate_robust_default_initial_guess(problem, current_nodes_list)
            else:
                # Propagate from previous solution for subsequent iterations
                initial_guess = propagate_guess_from_previous(
                    most_recent_solution, problem, current_nodes_list, current_mesh
                )
            problem.initial_guess = initial_guess

            # Log current mesh configuration
            print(f"  Mesh K={num_intervals}, num_nodes_per_interval = {current_nodes_list}")
            print(f"  Mesh nodes_tau_global = {np.array2string(current_mesh, precision=3)}")

            # Solve optimal control problem
            solution = solve_single_phase_radau_collocation(problem)

            if not solution.success:
                error_msg = f"NLP solver failed in adaptive iteration {iteration}. " + (
                    solution.message or "Solver error."
                )
                print(f"  Error: {error_msg} Stopping.")

                if most_recent_solution is not None:
                    most_recent_solution.message = error_msg
                    most_recent_solution.success = False
                    return most_recent_solution
                else:
                    solution.message = error_msg
                    return solution

            # Store solved trajectories for propagation and error estimation
            try:
                # Check for None before accessing attributes
                if solution.raw_solution is None or solution.opti_object is None:
                    error_msg = f"Failed to extract trajectories from NLP solution at iter {iteration}: Raw solution or opti object is None. Stopping."
                    print(f"  Error: {error_msg}")
                    solution.message = error_msg
                    solution.success = False
                    return solution

                opti = solution.opti_object
                raw_sol = solution.raw_solution

                # Extract state trajectories if variables exist
                if hasattr(opti, "state_at_local_approximation_nodes_all_intervals_variables"):
                    solution.solved_state_trajectories_per_interval = [
                        _extract_and_prepare_array(
                            raw_sol.value(var),
                            len(problem._states),
                            current_nodes_list[i] + 1,
                        )
                        for i, var in enumerate(
                            opti.state_at_local_approximation_nodes_all_intervals_variables
                        )
                    ]

                # Extract control trajectories if variables exist
                if len(problem._controls) > 0 and hasattr(
                    opti, "control_at_local_collocation_nodes_all_intervals_variables"
                ):
                    solution.solved_control_trajectories_per_interval = [
                        _extract_and_prepare_array(
                            raw_sol.value(var),
                            len(problem._controls),
                            current_nodes_list[i],
                        )
                        for i, var in enumerate(
                            opti.control_at_local_collocation_nodes_all_intervals_variables
                        )
                    ]
                else:
                    solution.solved_control_trajectories_per_interval = [
                        np.empty((0, current_nodes_list[i]), dtype=np.float64)
                        for i in range(num_intervals)
                    ]

            except Exception as extract_error:
                error_msg = f"Failed to extract trajectories from NLP solution at iter {iteration}: {extract_error}. Stopping."
                print(f"  Error: {error_msg}")
                solution.message = error_msg
                solution.success = False
                return solution

            # Update most recent successful solution
            most_recent_solution = solution
            most_recent_solution.num_collocation_nodes_list_at_solve_time = list(current_nodes_list)
            most_recent_solution.global_mesh_nodes_at_solve_time = np.copy(current_mesh)

            # Calculate gamma normalization factors
            gamma_factors = calculate_gamma_normalizers(solution, problem)
            if gamma_factors is None and len(problem._states) > 0:
                error_msg = f"Failed to calculate gamma normalizers at iter {iteration}. Stopping."
                print(f"  Error: {error_msg}")
                solution.message = error_msg
                solution.success = False
                return solution

            # Create cache for basis components and polynomial interpolants
            basis_cache: dict[int, RadauBasisComponents] = {}
            control_weights_cache: dict[int, _FloatArray] = {}
            state_evaluators: list[_StateEvaluator | None] = [None] * num_intervals
            control_evaluators: list[_ControlEvaluator | None] = [None] * num_intervals

            # Get solved trajectories
            states_list = solution.solved_state_trajectories_per_interval
            controls_list = solution.solved_control_trajectories_per_interval

            # Create polynomial interpolants for each interval
            for k in range(num_intervals):
                try:
                    Nk = current_nodes_list[k]

                    # Use cache for basis components
                    if Nk not in basis_cache:
                        basis_cache[Nk] = compute_radau_collocation_components(Nk)

                    basis = basis_cache[Nk]

                    # Create state interpolant
                    if states_list is None:
                        # Handle the None case - perhaps use a default state or raise an error
                        print(
                            f"Warning: states_list is None for interval {k}. Using default state."
                        )
                        state_data = np.zeros((len(problem._states), Nk + 1), dtype=np.float64)
                    else:
                        # Extract the data and ensure it's recognized as a 2D array
                        orig_data = states_list[k]
                        # Make the shape explicit to satisfy the type checker
                        rows, cols = orig_data.shape[0], orig_data.shape[1]
                        state_data = np.array(orig_data, dtype=np.float64).reshape(rows, cols)
                    state_evaluators[k] = get_polynomial_interpolant(
                        basis.state_approximation_nodes,
                        state_data,
                        basis.barycentric_weights_for_state_nodes,
                    )

                    # Create control interpolant
                    if len(problem._controls) > 0:
                        control_data = controls_list[k]

                        # Use cache for control weights
                        if Nk not in control_weights_cache:
                            control_weights_cache[Nk] = compute_barycentric_weights(
                                basis.collocation_nodes
                            )

                        control_weights = control_weights_cache[Nk]

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
                                (
                                    (len(problem._controls) if len(problem._controls) > 0 else 0),
                                    2,
                                ),
                                np.nan,
                                dtype=np.float64,
                            ),
                            None,
                        )

            # Calculate error estimates for each interval
            errors: list[float] = [np.inf] * num_intervals

            for k in range(num_intervals):
                print(f"  Starting error simulation for interval {k}...")

                # Use pre-computed interpolants
                state_eval = state_evaluators[k]
                control_eval = control_evaluators[k]

                if state_eval is None or control_eval is None:
                    print(
                        f"    Warning: Missing interpolants for interval {k}. Assigning high error."
                    )
                    errors[k] = np.inf
                    continue

                # Simulate dynamics for error estimation
                sim_bundle = simulate_dynamics_for_error_estimation(
                    k,
                    solution,
                    problem,
                    state_eval,
                    control_eval,
                    ode_rtol=ode_rtol,
                    n_eval_points=num_sim_points,
                )

                # Calculate relative error - make sure gamma_factors is not None
                safe_gamma = (
                    gamma_factors
                    if gamma_factors is not None
                    else np.ones((len(problem._states), 1), dtype=np.float64)
                )
                error = calculate_relative_error_estimate(k, sim_bundle, safe_gamma)

                errors[k] = error
                print(f"    Interval {k}: Nk={current_nodes_list[k]}, Error={error:.4e}")

            print(f"  Overall errors: {[f'{e:.2e}' for e in errors]}")

            # Check if all errors are within tolerance
            all_errors_ok = True
            if num_intervals == 0:
                all_errors_ok = True
            elif not errors:
                all_errors_ok = False
            else:
                for e in errors:
                    if np.isnan(e) or np.isinf(e) or e > error_tol:
                        all_errors_ok = False
                        break

            # If all errors within tolerance, return solution
            if all_errors_ok:
                print(f"Mesh converged after {iteration+1} iterations.")
                solution.num_collocation_nodes_per_interval = current_nodes_list.copy()
                solution.global_normalized_mesh_nodes = np.copy(current_mesh)
                solution.message = f"Adaptive mesh converged to tolerance {error_tol:.1e} in {iteration+1} iterations."
                return solution

            # Refine mesh for next iteration
            next_nodes_list: list[int] = []
            next_mesh = [current_mesh[0]]

            k = 0
            while k < num_intervals:
                error_k = errors[k]
                Nk = current_nodes_list[k]
                print(f"    Processing interval {k}: Nk={Nk}, Error={error_k:.2e}")

                if np.isnan(error_k) or np.isinf(error_k) or error_k > error_tol:
                    # Apply p-refinement if error > tolerance
                    print(f"      Interval {k} error > tol. Attempting p-refinement.")
                    p_result = p_refine_interval(Nk, error_k, error_tol, N_max)

                    if p_result.was_p_successful:
                        # p-refinement successful
                        print(
                            f"        p-refinement applied: Nk {Nk} -> {p_result.actual_Nk_to_use}"
                        )
                        next_nodes_list.append(p_result.actual_Nk_to_use)
                        next_mesh.append(current_mesh[k + 1])
                        k += 1
                    else:
                        # p-refinement failed, apply h-refinement
                        print("        p-refinement failed. Attempting h-refinement.")
                        h_result = h_refine_params(p_result.unconstrained_target_Nk, N_min)

                        print(
                            f"          h-refinement: Splitting interval {k} into {h_result.num_new_subintervals} subintervals, each Nk={h_result.collocation_nodes_for_new_subintervals[0]}."
                        )
                        next_nodes_list.extend(h_result.collocation_nodes_for_new_subintervals)

                        # Create new mesh nodes for subintervals
                        tau_start = current_mesh[k]
                        tau_end = current_mesh[k + 1]
                        new_nodes = np.linspace(
                            tau_start, tau_end, h_result.num_new_subintervals + 1, dtype=np.float64
                        )
                        next_mesh.extend(list(new_nodes[1:]))
                        k += 1
                else:
                    # Error <= tolerance, check for h-reduction
                    print(f"    Interval {k} error <= tol.")
                    can_merge = False

                    # Check if next interval is eligible for merging
                    if k < num_intervals - 1:
                        next_error = errors[k + 1]
                        if (
                            not (np.isnan(next_error) or np.isinf(next_error))
                            and next_error <= error_tol
                        ):
                            print(
                                f"      Interval {k+1} also has low error ({next_error:.2e}). Eligible for h-reduction."
                            )

                            # Check if interpolants are non-None before passing to h_reduce_intervals
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

                                # Ensure we don't pass None to functions that can't handle it
                                state_eval_first = cast(_StateEvaluator, state_evaluators[k])
                                state_eval_second = cast(_StateEvaluator, state_evaluators[k + 1])
                                control_eval_first = control_evaluators[k]
                                control_eval_second = control_evaluators[k + 1]

                                # Make sure gamma_factors is not None
                                safe_gamma = (
                                    gamma_factors
                                    if gamma_factors is not None
                                    else np.ones((len(problem._states), 1), dtype=np.float64)
                                )

                                # Attempt h-reduction (interval merging)
                                can_merge = h_reduce_intervals(
                                    k,
                                    solution,
                                    problem,
                                    self.adaptive_params,
                                    safe_gamma,
                                    state_eval_first,
                                    control_eval_first,
                                    state_eval_second,
                                    control_eval_second,
                                )
                            else:
                                print(
                                    "      Skipping h-reduction attempt due to missing interpolants."
                                )

                    if can_merge:
                        # h-reduction successful, merge intervals
                        print(f"      h-reduction applied to merge interval {k} and {k+1}.")

                        # Use maximum Nk from the two intervals being merged
                        merged_Nk = max(current_nodes_list[k], current_nodes_list[k + 1])
                        merged_Nk = max(N_min, min(N_max, merged_Nk))

                        next_nodes_list.append(merged_Nk)
                        next_mesh.append(current_mesh[k + 2])
                        k += 2
                    else:
                        # h-reduction failed or not applicable, try p-reduction
                        print(
                            f"      h-reduction failed or not applicable. Attempting p-reduction for interval {k}."
                        )
                        p_reduce = p_reduce_interval(Nk, error_k, error_tol, N_min, N_max)

                        if p_reduce.was_reduction_applied:
                            print(
                                f"        p-reduction applied: Nk {Nk} -> {p_reduce.new_num_collocation_nodes}"
                            )
                        else:
                            print(f"        p-reduction not applied for Nk {Nk}.")

                        next_nodes_list.append(p_reduce.new_num_collocation_nodes)
                        next_mesh.append(current_mesh[k + 1])
                        k += 1

            # Update for next iteration
            current_nodes_list = next_nodes_list
            current_mesh = np.array(next_mesh, dtype=np.float64)

            # Perform mesh sanity checks
            early_return_solution = most_recent_solution
            if early_return_solution is not None:
                early_return_solution.num_collocation_nodes_per_interval = current_nodes_list
                early_return_solution.global_normalized_mesh_nodes = current_mesh

                # Check for mesh inconsistencies
                if not current_nodes_list and len(current_mesh) > 1:
                    error_msg = "Stopped due to mesh inconsistency (empty num_collocation_nodes_per_interval but mesh_nodes exist)."
                    print(f"  Error: {error_msg} Stopping.")
                    early_return_solution.message = error_msg
                    early_return_solution.success = False
                    return early_return_solution

                if current_nodes_list and len(current_nodes_list) != (len(current_mesh) - 1):
                    error_msg = f"Mesh structure inconsistent. num_nodes_list len: {len(current_nodes_list)}, mesh_nodes len-1: {len(current_mesh)-1}."
                    print(f"  Error: {error_msg} Stopping.")
                    early_return_solution.message = error_msg
                    early_return_solution.success = False
                    return early_return_solution

                if len(current_nodes_list) > 0:
                    # Check for duplicate mesh nodes
                    unique_nodes, counts = np.unique(
                        np.round(current_mesh, decimals=12), return_counts=True
                    )
                    if np.any(counts > 1):
                        duplicates = unique_nodes[counts > 1]
                        error_msg = f"Duplicate mesh nodes found: {duplicates}. Original nodes: {current_mesh}."
                        print(f"  Error: {error_msg} Stopping.")
                        early_return_solution.message = error_msg
                        early_return_solution.success = False
                        return early_return_solution

                    # Check for non-increasing mesh nodes
                    if len(unique_nodes) > 1 and not np.all(np.diff(unique_nodes) > 1e-9):
                        diffs = np.diff(unique_nodes)
                        problem_indices = np.where(diffs <= 1e-9)[0]
                        problem_pairs = (
                            ", ".join(
                                [
                                    f"({unique_nodes[i]:.3f}, {unique_nodes[i+1]:.3f})"
                                    for i in problem_indices
                                ]
                            )
                            if problem_indices.size > 0
                            else "N/A"
                        )

                        error_msg = f"Mesh nodes not strictly increasing or interval too small. Problem pairs: {problem_pairs}. All nodes: {current_mesh}."
                        print(f"  Error: {error_msg} Stopping.")
                        early_return_solution.message = error_msg
                        early_return_solution.success = False
                        return early_return_solution

        # Max iterations reached without convergence
        max_iter_msg = f"Adaptive mesh refinement reached max iterations ({max_iterations}) without full convergence to tolerance {error_tol:.1e}."
        print(max_iter_msg)

        if most_recent_solution is not None:
            most_recent_solution.message = max_iter_msg
            most_recent_solution.num_collocation_nodes_per_interval = current_nodes_list.copy()
            most_recent_solution.global_normalized_mesh_nodes = np.copy(current_mesh)
            return most_recent_solution
        else:
            failed = OptimalControlSolution()
            failed.success = False
            failed.message = (
                max_iter_msg + " No successful NLP solution obtained throughout iterations."
            )
            failed.num_collocation_nodes_per_interval = current_nodes_list
            # Ensure we use a numpy array for global_normalized_mesh_nodes
            failed.global_normalized_mesh_nodes = np.array(current_mesh, dtype=np.float64)
            return failed
