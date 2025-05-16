"""Core implementation of p-h-s adaptive mesh refinement for optimal control."""

import logging
from typing import List, Optional, Union

import numpy as np

from trajectolab.adaptive.base import AdaptiveBase
from trajectolab.adaptive.phs.adaptive_error import (
    calculate_gamma_normalizers,
    calculate_relative_error_estimate,
    simulate_dynamics_for_error_estimation,
)
from trajectolab.adaptive.phs.adaptive_initialization import (
    generate_robust_default_initial_guess,
    propagate_guess_from_previous,
)
from trajectolab.adaptive.phs.adaptive_interpolation import (
    PolynomialInterpolant,
    get_polynomial_interpolant,
)
from trajectolab.adaptive.phs.adaptive_refinement import (
    h_reduce_intervals,
    h_refine_params,
    p_reduce_interval,
    p_refine_interval,
)
from trajectolab.radau import compute_barycentric_weights, compute_radau_collocation_components
from trajectolab.trajectolab_types import (
    AdaptiveParameters,
    InitialGuess,
    OptimalControlProblem,
    OptimalControlSolution,
    RadauBasisComponents,
    _FloatArray,
    _MeshPoints,
    _NormalizedTimePoint,
    _Vector,
)

logger = logging.getLogger(__name__)


class PHSAdaptive(AdaptiveBase):
    """Implements P-H-S (p-refinement, h-refinement, smoothness) adaptive mesh refinement.

    This class provides a sophisticated mesh refinement strategy that:
    1. Solves the optimal control problem on an initial mesh
    2. Estimates errors by comparing the solution with simulated trajectories
    3. Adaptively refines the mesh by increasing polynomial degree (p-refinement)
       or splitting intervals (h-refinement) in high-error regions
    4. Reduces computational cost by merging intervals (h-reduction) or decreasing
       polynomial degree (p-reduction) in low-error regions
    5. Iterates until convergence or maximum iterations reached
    """

    adaptive_params: AdaptiveParameters
    _initial_polynomial_degrees: Optional[List[int]]
    _initial_mesh_points_global_tau: Optional[_MeshPoints]

    def __init__(
        self,
        error_tolerance: float = 1e-3,
        max_iterations: int = 30,
        min_polynomial_degree: int = 4,
        max_polynomial_degree: int = 16,
        ode_solver_tolerance: float = 1e-7,
        num_error_sim_points: int = 40,
        initial_polynomial_degrees: Optional[List[int]] = None,
        initial_mesh_points: Optional[Union[_MeshPoints, List[float]]] = None,
        initial_guess: Optional[InitialGuess] = None,
    ):
        """Initialize the adaptive mesh refinement strategy.

        Args:
            error_tolerance: Error tolerance for convergence
            max_iterations: Maximum number of iterations
            min_polynomial_degree: Minimum polynomial degree allowed
            max_polynomial_degree: Maximum polynomial degree allowed
            ode_solver_tolerance: Tolerance for ODE solver used in error estimation
            num_error_sim_points: Number of points to use for error estimation
            initial_polynomial_degrees: Initial polynomial degrees for each interval
            initial_mesh_points: Initial mesh points in normalized domain [-1, 1]
            initial_guess: Initial guess for the NLP solver
        """
        super().__init__(initial_guess)
        self.adaptive_params = AdaptiveParameters(
            error_tolerance,
            max_iterations,
            min_polynomial_degree,
            max_polynomial_degree,
            ode_solver_tolerance,
            num_error_sim_points,
        )

        if initial_polynomial_degrees is not None and initial_mesh_points is not None:
            if len(initial_polynomial_degrees) != len(initial_mesh_points) - 1:
                raise ValueError(
                    "Number of initial polynomial degrees must be one less than the number of initial mesh points."
                )

        self._initial_polynomial_degrees = (
            list(initial_polynomial_degrees) if initial_polynomial_degrees else None
        )
        if initial_mesh_points is not None:
            self._initial_mesh_points_global_tau = np.array(initial_mesh_points, dtype=np.float64)
        else:
            self._initial_mesh_points_global_tau = None

    def run(
        self,
        problem: Optional[OptimalControlProblem],  # Unused parameter, kept for API compatibility
        legacy_problem_instance: OptimalControlProblem,
        initial_solution: Optional[OptimalControlSolution] = None,
    ) -> OptimalControlSolution:
        """Run the adaptive mesh refinement algorithm.

        Args:
            problem: Unused parameter, kept for API compatibility
            legacy_problem_instance: Problem definition
            initial_solution: Optional initial solution to start from

        Returns:
            Final solution after adaptation
        """
        # Import solver dynamically to avoid circular dependencies
        try:
            from trajectolab.direct_solver import solve_single_phase_radau_collocation
        except ImportError as e:
            logger.error(f"Failed to import solve_single_phase_radau_collocation: {e}")
            raise

        # Extract parameters
        error_tol = self.adaptive_params.error_tolerance
        max_iter = self.adaptive_params.max_iterations
        N_min = self.adaptive_params.min_polynomial_degree
        N_max = self.adaptive_params.max_polynomial_degree
        ode_rtol_val = self.adaptive_params.ode_solver_tolerance
        num_sim_pts = self.adaptive_params.num_error_sim_points

        # Initialize current mesh structure
        current_collocation_nodes_list: List[int]
        current_global_mesh_tau: _MeshPoints

        if (
            self._initial_polynomial_degrees is not None
            and self._initial_mesh_points_global_tau is not None
        ):
            current_collocation_nodes_list = list(self._initial_polynomial_degrees)
            current_global_mesh_tau = np.copy(self._initial_mesh_points_global_tau)
        else:
            # Fallback to problem's initial mesh or a single default interval
            current_collocation_nodes_list = list(
                legacy_problem_instance.collocation_points_per_interval or [N_min]
            )
            if legacy_problem_instance.global_normalized_mesh_nodes is not None:
                current_global_mesh_tau = np.array(
                    legacy_problem_instance.global_normalized_mesh_nodes, dtype=np.float64
                )
            else:
                current_global_mesh_tau = np.linspace(
                    -1, 1, len(current_collocation_nodes_list) + 1, dtype=np.float64
                )

        # Ensure initial node counts are within bounds
        current_collocation_nodes_list = [
            max(N_min, min(N_max, nk)) for nk in current_collocation_nodes_list
        ]

        # The OCP definition that will be modified in each iteration
        current_ocp_definition: OptimalControlProblem = legacy_problem_instance
        most_recent_ocp_solution: Optional[OptimalControlSolution] = initial_solution

        for iteration_M in range(max_iter):
            logger.info(f"\n--- Adaptive Iteration M = {iteration_M} ---")
            num_intervals_K = len(current_collocation_nodes_list)

            # Update OCP definition with current mesh
            current_ocp_definition.collocation_points_per_interval = list(
                current_collocation_nodes_list
            )
            current_ocp_definition.global_normalized_mesh_nodes = list(current_global_mesh_tau)

            # Prepare initial guess for the NLP solver
            initial_guess_for_nlp: InitialGuess
            user_provided_initial_guess = self.initial_guess

            if iteration_M == 0 and user_provided_initial_guess is not None:
                initial_guess_for_nlp = user_provided_initial_guess
                logger.info("  Using user-provided initial guess for first iteration.")
            elif iteration_M == 0:
                initial_guess_for_nlp = generate_robust_default_initial_guess(
                    current_ocp_definition, current_collocation_nodes_list
                )
                logger.info("  Using robust default initial guess for first iteration.")
            elif not most_recent_ocp_solution or not most_recent_ocp_solution.success:
                logger.warning(
                    "  Previous NLP failed or no previous solution. Using robust default."
                )
                initial_guess_for_nlp = generate_robust_default_initial_guess(
                    current_ocp_definition, current_collocation_nodes_list
                )
            else:
                initial_guess_for_nlp = propagate_guess_from_previous(
                    most_recent_ocp_solution,
                    current_ocp_definition,
                    current_collocation_nodes_list,
                    current_global_mesh_tau,
                )
            current_ocp_definition.initial_guess = initial_guess_for_nlp

            logger.info(f"  Mesh K={num_intervals_K}, N_k = {current_collocation_nodes_list}")
            logger.info(
                f"  Mesh tau_global = {np.array2string(current_global_mesh_tau, precision=3)}"
            )

            # Solve the OCP
            solved_ocp_solution_this_iter: OptimalControlSolution = (
                solve_single_phase_radau_collocation(current_ocp_definition)
            )

            if not solved_ocp_solution_this_iter.success:
                error_msg = (
                    f"NLP solver failed in adaptive iteration {iteration_M}. "
                    f"{solved_ocp_solution_this_iter.message or 'Solver error.'} Stopping."
                )
                logger.error(f"  Error: {error_msg}")
                if most_recent_ocp_solution:
                    most_recent_ocp_solution.message = error_msg
                    most_recent_ocp_solution.success = False
                    return most_recent_ocp_solution
                return solved_ocp_solution_this_iter

            # Extract solution states and controls
            try:
                from trajectolab.adaptive.phs.adaptive_interpolation import (
                    extract_and_prepare_array,
                )

                opti_obj = solved_ocp_solution_this_iter.opti_object
                raw_casadi_sol = solved_ocp_solution_this_iter.raw_solution

                if (
                    opti_obj
                    and raw_casadi_sol
                    and hasattr(
                        opti_obj, "state_at_local_approximation_nodes_all_intervals_variables"
                    )
                    and opti_obj.state_at_local_approximation_nodes_all_intervals_variables
                ):
                    temp_states_list: List[_FloatArray] = []
                    for i, var_sx_or_mx in enumerate(
                        opti_obj.state_at_local_approximation_nodes_all_intervals_variables
                    ):
                        val = raw_casadi_sol.value(var_sx_or_mx)
                        temp_states_list.append(
                            extract_and_prepare_array(
                                val,
                                current_ocp_definition.num_states,
                                current_collocation_nodes_list[i] + 1,
                            )
                        )
                    solved_ocp_solution_this_iter.states = temp_states_list
                elif current_ocp_definition.num_states > 0:
                    logger.warning("  Could not extract states from NLP solution object.")
                    solved_ocp_solution_this_iter.states = []
                else:
                    solved_ocp_solution_this_iter.states = []

                if current_ocp_definition.num_controls > 0:
                    if (
                        opti_obj
                        and raw_casadi_sol
                        and hasattr(
                            opti_obj, "control_at_local_collocation_nodes_all_intervals_variables"
                        )
                        and opti_obj.control_at_local_collocation_nodes_all_intervals_variables
                    ):
                        temp_controls_list: List[_FloatArray] = []
                        for i, var_sx_or_mx in enumerate(
                            opti_obj.control_at_local_collocation_nodes_all_intervals_variables
                        ):
                            val = raw_casadi_sol.value(var_sx_or_mx)
                            temp_controls_list.append(
                                extract_and_prepare_array(
                                    val,
                                    current_ocp_definition.num_controls,
                                    current_collocation_nodes_list[i],
                                )
                            )
                        solved_ocp_solution_this_iter.controls = temp_controls_list
                    else:
                        logger.warning("  Could not extract controls from NLP solution object.")
                        solved_ocp_solution_this_iter.controls = []
                else:
                    solved_ocp_solution_this_iter.controls = [
                        np.empty((0, nk), dtype=np.float64) for nk in current_collocation_nodes_list
                    ]

            except Exception as e:
                error_msg = f"Failed to extract trajectories: {e}. Stopping."
                logger.error(f"  Error: {error_msg}")
                solved_ocp_solution_this_iter.message = error_msg
                solved_ocp_solution_this_iter.success = False
                return solved_ocp_solution_this_iter

            most_recent_ocp_solution = solved_ocp_solution_this_iter
            # Store mesh used for this successful solve
            most_recent_ocp_solution.num_collocation_nodes_list_at_solve_time = list(
                current_collocation_nodes_list
            )
            most_recent_ocp_solution.global_mesh_nodes_at_solve_time = list(
                np.copy(current_global_mesh_tau)
            )

            gamma_factors_col_vec: Optional[_Vector] = calculate_gamma_normalizers(
                most_recent_ocp_solution, current_ocp_definition
            )
            if gamma_factors_col_vec is None and current_ocp_definition.num_states > 0:
                error_msg = (
                    f"Failed to calculate gamma normalizers at iter {iteration_M}. Stopping."
                )
                logger.error(f"  Error: {error_msg}")
                most_recent_ocp_solution.message = error_msg
                most_recent_ocp_solution.success = False
                return most_recent_ocp_solution

            # Cache for Radau basis components
            radau_basis_cache: dict[int, RadauBasisComponents] = {}

            # Prepare interpolants for state and control trajectories
            state_evaluators_list: List[PolynomialInterpolant] = [None] * num_intervals_K  # type: ignore
            control_evaluators_list: List[PolynomialInterpolant] = [None] * num_intervals_K  # type: ignore

            # Create dummy interpolant for controls if no controls exist
            _dummy_control_interpolant_if_no_controls = None
            if current_ocp_definition.num_controls == 0:
                _dummy_nodes = np.array([-1.0, 1.0], dtype=np.float64)
                _dummy_control_values = np.empty((0, 2), dtype=np.float64)
                _dummy_control_interpolant_if_no_controls = PolynomialInterpolant(
                    _dummy_nodes, _dummy_control_values
                )

            for k_interval_idx in range(num_intervals_K):
                try:
                    Nk_val_current_interval = current_collocation_nodes_list[k_interval_idx]
                    if Nk_val_current_interval not in radau_basis_cache:
                        radau_basis_cache[Nk_val_current_interval] = (
                            compute_radau_collocation_components(Nk_val_current_interval)
                        )
                    basis_comps: RadauBasisComponents = radau_basis_cache[Nk_val_current_interval]

                    # State interpolant
                    if (
                        most_recent_ocp_solution.states
                        and k_interval_idx < len(most_recent_ocp_solution.states)
                        and most_recent_ocp_solution.states[k_interval_idx].size > 0
                    ):
                        state_values_k_interval: _FloatArray = most_recent_ocp_solution.states[
                            k_interval_idx
                        ]
                        state_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                            basis_comps.state_approximation_nodes,
                            state_values_k_interval,
                            basis_comps.barycentric_weights_for_state_nodes,
                        )
                    elif current_ocp_definition.num_states > 0:
                        logger.warning(f"  State trajectory missing for interval {k_interval_idx}.")
                        empty_state_vals = np.full(
                            (
                                current_ocp_definition.num_states,
                                len(basis_comps.state_approximation_nodes),
                            ),
                            np.nan,
                            dtype=np.float64,
                        )
                        state_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                            basis_comps.state_approximation_nodes,
                            empty_state_vals,
                            basis_comps.barycentric_weights_for_state_nodes,
                        )
                    else:  # num_states == 0
                        empty_state_vals = np.empty(
                            (0, len(basis_comps.state_approximation_nodes)), dtype=np.float64
                        )
                        state_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                            basis_comps.state_approximation_nodes,
                            empty_state_vals,
                            basis_comps.barycentric_weights_for_state_nodes,
                        )

                    # Control interpolant
                    if current_ocp_definition.num_controls > 0:
                        if (
                            most_recent_ocp_solution.controls
                            and k_interval_idx < len(most_recent_ocp_solution.controls)
                            and most_recent_ocp_solution.controls[k_interval_idx].size > 0
                        ):
                            control_values_k_interval: _FloatArray = (
                                most_recent_ocp_solution.controls[k_interval_idx]
                            )
                            control_bary_weights = compute_barycentric_weights(
                                basis_comps.collocation_nodes
                            )
                            control_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                                basis_comps.collocation_nodes,
                                control_values_k_interval,
                                control_bary_weights,
                            )
                        else:  # Controls expected but missing
                            logger.warning(
                                f"  Control trajectory missing for interval {k_interval_idx}."
                            )
                            empty_ctrl_vals = np.full(
                                (
                                    current_ocp_definition.num_controls,
                                    len(basis_comps.collocation_nodes),
                                ),
                                np.nan,
                                dtype=np.float64,
                            )
                            control_bary_weights = compute_barycentric_weights(
                                basis_comps.collocation_nodes
                            )
                            control_evaluators_list[k_interval_idx] = get_polynomial_interpolant(
                                basis_comps.collocation_nodes, empty_ctrl_vals, control_bary_weights
                            )
                    else:
                        control_evaluators_list[k_interval_idx] = _dummy_control_interpolant_if_no_controls  # type: ignore

                except Exception as e:
                    logger.error(f"  Error creating interpolant for interval {k_interval_idx}: {e}")
                    # Fallback to dummy interpolants
                    _dummy_nodes = np.array([-1.0, 1.0], dtype=np.float64)
                    _s_vals = np.empty((current_ocp_definition.num_states, 2), dtype=np.float64)
                    state_evaluators_list[k_interval_idx] = PolynomialInterpolant(
                        _dummy_nodes, _s_vals
                    )
                    if _dummy_control_interpolant_if_no_controls:
                        control_evaluators_list[k_interval_idx] = (
                            _dummy_control_interpolant_if_no_controls
                        )
                    else:
                        _c_vals = np.empty(
                            (current_ocp_definition.num_controls, 2), dtype=np.float64
                        )
                        control_evaluators_list[k_interval_idx] = PolynomialInterpolant(
                            _dummy_nodes, _c_vals
                        )

            # Estimate error for each interval
            interval_error_estimates: List[float] = [np.inf] * num_intervals_K

            for k_interval_idx in range(num_intervals_K):
                logger.info(f"  Starting error simulation for interval {k_interval_idx}...")
                sim_bundle = simulate_dynamics_for_error_estimation(
                    k_interval_idx,
                    most_recent_ocp_solution,
                    current_ocp_definition,
                    state_evaluators_list[k_interval_idx],
                    control_evaluators_list[k_interval_idx],
                    ode_rtol=ode_rtol_val,
                    n_eval_points=num_sim_pts,
                )

                # Ensure gamma_factors_col_vec is valid for error calculation
                gamma_for_calc: _Vector
                if current_ocp_definition.num_states > 0:
                    if (
                        gamma_factors_col_vec is not None
                        and gamma_factors_col_vec.size == current_ocp_definition.num_states
                    ):
                        gamma_for_calc = gamma_factors_col_vec
                    else:
                        logger.warning(f"   Gamma factors invalid for interval {k_interval_idx}.")
                        gamma_for_calc = np.full(
                            (current_ocp_definition.num_states, 1), np.nan, dtype=np.float64
                        )
                else:
                    gamma_for_calc = np.empty((0, 1), dtype=np.float64)

                error_val_this_interval = calculate_relative_error_estimate(
                    k_interval_idx, sim_bundle, gamma_for_calc
                )
                interval_error_estimates[k_interval_idx] = error_val_this_interval
                logger.info(
                    f"    Interval {k_interval_idx}: Nk={current_collocation_nodes_list[k_interval_idx]}, "
                    f"Error={error_val_this_interval:.4e}"
                )

            logger.info(f"  Overall errors: {[f'{e:.2e}' for e in interval_error_estimates]}")

            # Check for convergence: all errors below tolerance
            converged_this_iteration = False
            if not interval_error_estimates:
                converged_this_iteration = True
            else:
                converged_this_iteration = all(
                    err <= error_tol
                    for err in interval_error_estimates
                    if not (np.isnan(err) or np.isinf(err))
                )
                if any(np.isnan(err) or np.isinf(err) for err in interval_error_estimates):
                    converged_this_iteration = False

            if converged_this_iteration:
                success_msg = f"Adaptive mesh converged to tolerance {error_tol:.1e} in {iteration_M+1} iterations."
                logger.info(success_msg)
                most_recent_ocp_solution.num_collocation_nodes_per_interval = list(
                    current_collocation_nodes_list
                )
                most_recent_ocp_solution.global_normalized_mesh_nodes = list(
                    np.copy(current_global_mesh_tau)
                )
                most_recent_ocp_solution.message = success_msg
                return most_recent_ocp_solution

            # --- Mesh Adaptation Logic (p, h refinement/reduction) ---
            next_collocation_nodes_proposal: List[int] = []
            next_global_mesh_tau_proposal: List[_NormalizedTimePoint] = [current_global_mesh_tau[0]]

            current_interval_pointer = 0
            while current_interval_pointer < num_intervals_K:
                error_k = interval_error_estimates[current_interval_pointer]
                Nk_k = current_collocation_nodes_list[current_interval_pointer]
                logger.info(
                    f"    Adapting interval {current_interval_pointer}: Nk={Nk_k}, Error={error_k:.2e}"
                )

                if (
                    np.isnan(error_k) or np.isinf(error_k) or error_k > error_tol
                ):  # High error or invalid
                    logger.info(
                        f"      Interval {current_interval_pointer} error > tol. Attempting p-refinement."
                    )
                    p_refine_res = p_refine_interval(Nk_k, error_k, error_tol, N_max)
                    if p_refine_res.was_p_successful:
                        logger.info(
                            f"        p-refinement applied: Nk {Nk_k} -> {p_refine_res.actual_Nk_to_use}"
                        )
                        next_collocation_nodes_proposal.append(p_refine_res.actual_Nk_to_use)
                        next_global_mesh_tau_proposal.append(
                            current_global_mesh_tau[current_interval_pointer + 1]
                        )
                        current_interval_pointer += 1
                    else:  # p-refinement failed or hit N_max, try h-refinement
                        logger.info(
                            "        p-refinement failed or insufficient. Attempting h-refinement."
                        )
                        h_refine_res = h_refine_params(p_refine_res.unconstrained_target_Nk, N_min)
                        logger.info(
                            f"          h-refinement: Splitting int {current_interval_pointer} into {h_refine_res.num_new_subintervals} "
                            f"subintervals, each Nk={h_refine_res.collocation_nodes_for_new_subintervals[0]}."
                        )
                        next_collocation_nodes_proposal.extend(
                            h_refine_res.collocation_nodes_for_new_subintervals
                        )
                        # Create new mesh points for the split interval
                        new_split_mesh_segment = np.linspace(
                            current_global_mesh_tau[
                                current_interval_pointer
                            ],  # Start of current interval
                            current_global_mesh_tau[
                                current_interval_pointer + 1
                            ],  # End of current interval
                            h_refine_res.num_new_subintervals + 1,
                            dtype=np.float64,
                        )
                        next_global_mesh_tau_proposal.extend(
                            list(new_split_mesh_segment[1:])
                        )  # Add new internal points and end point
                        current_interval_pointer += 1
                else:  # Low error: error_k <= error_tol
                    logger.info(f"      Interval {current_interval_pointer} error <= tol.")
                    merged_intervals_in_this_step = False
                    # Check for h-reduction with the next interval if possible
                    if current_interval_pointer < num_intervals_K - 1:
                        error_kp1 = interval_error_estimates[current_interval_pointer + 1]
                        if (
                            not (np.isnan(error_kp1) or np.isinf(error_kp1))
                            and error_kp1 <= error_tol
                        ):
                            logger.info(
                                f"      Interval {current_interval_pointer+1} also low error ({error_kp1:.2e}). Checking h-reduction."
                            )
                            gamma_for_hr_calc = gamma_factors_col_vec
                            if current_ocp_definition.num_states > 0 and (
                                gamma_for_hr_calc is None
                                or gamma_for_hr_calc.size != current_ocp_definition.num_states
                            ):
                                logger.warning(
                                    "       Cannot perform h-reduction: gamma factors invalid for this check."
                                )
                                can_merge_flag = False
                            else:
                                can_merge_flag = h_reduce_intervals(
                                    current_interval_pointer,
                                    most_recent_ocp_solution,
                                    current_ocp_definition,
                                    self.adaptive_params,
                                    (
                                        gamma_for_hr_calc
                                        if gamma_for_hr_calc is not None
                                        else np.empty((0, 1))
                                    ),
                                    state_evaluators_list[current_interval_pointer],
                                    control_evaluators_list[current_interval_pointer],
                                    state_evaluators_list[current_interval_pointer + 1],
                                    control_evaluators_list[current_interval_pointer + 1],
                                )
                            if can_merge_flag:
                                logger.info(
                                    f"      h-reduction approved. Merging intervals {current_interval_pointer} and {current_interval_pointer+1}."
                                )
                                # New Nk for merged interval: max of the two, bounded by N_min, N_max
                                merged_Nk = max(
                                    current_collocation_nodes_list[current_interval_pointer],
                                    current_collocation_nodes_list[current_interval_pointer + 1],
                                )
                                merged_Nk = max(N_min, min(N_max, merged_Nk))
                                next_collocation_nodes_proposal.append(merged_Nk)
                                # The new interval spans from start of k to end of k+1.
                                # The end point is current_global_mesh_tau[current_interval_pointer + 2]
                                next_global_mesh_tau_proposal.append(
                                    current_global_mesh_tau[current_interval_pointer + 2]
                                )
                                current_interval_pointer += 2  # Advanced past two merged intervals
                                merged_intervals_in_this_step = True

                    if not merged_intervals_in_this_step:  # No merge, or was last interval
                        logger.info(
                            f"      h-reduction not applied or not applicable. Attempting p-reduction for interval {current_interval_pointer}."
                        )
                        p_reduce_res = p_reduce_interval(Nk_k, error_k, error_tol, N_min, N_max)
                        if p_reduce_res.was_reduction_applied:
                            logger.info(
                                f"        p-reduction applied: Nk {Nk_k} -> {p_reduce_res.new_num_collocation_nodes}"
                            )
                        else:
                            logger.info(f"        p-reduction not applied for Nk {Nk_k}.")
                        next_collocation_nodes_proposal.append(
                            p_reduce_res.new_num_collocation_nodes
                        )
                        next_global_mesh_tau_proposal.append(
                            current_global_mesh_tau[current_interval_pointer + 1]
                        )
                        current_interval_pointer += 1
            # --- End Mesh Adaptation Logic ---

            current_collocation_nodes_list = next_collocation_nodes_proposal
            current_global_mesh_tau = np.array(next_global_mesh_tau_proposal, dtype=np.float64)

            # Update solution object with the proposed mesh for the next iteration
            if most_recent_ocp_solution:
                most_recent_ocp_solution.num_collocation_nodes_per_interval = list(
                    current_collocation_nodes_list
                )
                most_recent_ocp_solution.global_normalized_mesh_nodes = list(
                    current_global_mesh_tau
                )

            # --- Mesh Sanity Checks ---
            if (
                not current_collocation_nodes_list and len(current_global_mesh_tau) > 1
            ):  # No intervals defined, but mesh points exist
                error_msg = "Adaptive process stopped: Mesh inconsistency (no collocation_nodes_list but mesh_nodes exist)."
                logger.error(f"  Error: {error_msg}")
                if most_recent_ocp_solution:
                    most_recent_ocp_solution.message = error_msg
                    most_recent_ocp_solution.success = False
                else:
                    most_recent_ocp_solution = OptimalControlSolution(
                        success=False, message=error_msg
                    )
                return most_recent_ocp_solution

            if current_collocation_nodes_list and (
                len(current_collocation_nodes_list) != (len(current_global_mesh_tau) - 1)
            ):
                error_msg = (
                    f"Adaptive process stopped: Mesh structure inconsistent. "
                    f"Num intervals from nodes_list: {len(current_collocation_nodes_list)}, "
                    f"Num intervals from mesh_tau: {len(current_global_mesh_tau)-1}."
                )
                logger.error(f"  Error: {error_msg}")
                if most_recent_ocp_solution:
                    most_recent_ocp_solution.message = error_msg
                    most_recent_ocp_solution.success = False
                else:
                    most_recent_ocp_solution = OptimalControlSolution(
                        success=False,
                        message=error_msg,
                        num_collocation_nodes_per_interval=list(current_collocation_nodes_list),
                        global_normalized_mesh_nodes=list(current_global_mesh_tau),
                    )
                return most_recent_ocp_solution

            if len(current_global_mesh_tau) > 1:
                # Check for duplicate or non-increasing mesh points (after rounding for stability)
                unique_mesh_nodes_rounded = np.round(current_global_mesh_tau, decimals=10)
                if len(np.unique(unique_mesh_nodes_rounded)) != len(current_global_mesh_tau):
                    error_msg = (
                        "Adaptive process stopped: Duplicate mesh nodes found after rounding."
                    )
                    logger.error(f"  Error: {error_msg}")
                    if most_recent_ocp_solution:
                        most_recent_ocp_solution.message = error_msg
                        most_recent_ocp_solution.success = False
                    else:
                        most_recent_ocp_solution = OptimalControlSolution(
                            success=False, message=error_msg
                        )
                    return most_recent_ocp_solution

                diffs = np.diff(current_global_mesh_tau)
                if not np.all(diffs > 1e-9):  # Check for strictly increasing and non-tiny intervals
                    problem_indices = np.where(diffs <= 1e-9)[0]
                    problem_pairs_str = ", ".join(
                        [
                            f"({current_global_mesh_tau[i]:.3e}, {current_global_mesh_tau[i+1]:.3e})"
                            for i in problem_indices
                        ]
                    )
                    error_msg = (
                        f"Adaptive process stopped: Mesh nodes not strictly increasing or interval too small. "
                        f"Problem pairs at indices {problem_indices}: {problem_pairs_str}."
                    )
                    logger.error(f"  Error: {error_msg}")
                    if most_recent_ocp_solution:
                        most_recent_ocp_solution.message = error_msg
                        most_recent_ocp_solution.success = False
                    else:
                        most_recent_ocp_solution = OptimalControlSolution(
                            success=False, message=error_msg
                        )
                    return most_recent_ocp_solution
            # --- End Mesh Sanity Checks ---

        # Loop finished due to max_iterations
        max_iter_msg = (
            f"Adaptive mesh refinement reached max iterations ({max_iter}) "
            f"without full convergence to tolerance {error_tol:.1e}."
        )
        logger.warning(max_iter_msg)
        if most_recent_ocp_solution:
            most_recent_ocp_solution.message = max_iter_msg
            return most_recent_ocp_solution
        else:  # No solution obtained at all
            return OptimalControlSolution(
                success=False,
                message=max_iter_msg
                + " No successful NLP solution obtained throughout iterations.",
                num_collocation_nodes_per_interval=list(current_collocation_nodes_list),
                global_normalized_mesh_nodes=list(current_global_mesh_tau),
            )
