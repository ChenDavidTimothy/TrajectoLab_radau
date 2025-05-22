import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


# --- Constants ---
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
A0_CL = -0.20704
A1_CL = 0.029244
B0_CD = 0.07854
B1_CD = -0.61592e-2
B2_CD = 0.621408e-3
C0_QA = 1.0672181
C1_QA = -0.19213774e-1
C2_QA = 0.21286289e-3
C3_QA = -0.10117249e-5
MU_EARTH = 0.14076539e17
R_EARTH = 20902900.0
S_REF = 2690.0
RHO0 = 0.002378
H_R = 23800.0
G0 = 32.174
WEIGHT = 203000.0
MASS = WEIGHT / G0


# --- get_scaling_params_generalized ---
def get_scaling_params_generalized(
    var_name,
    explicit_lower_bound,
    explicit_upper_bound,
    initial_guess_min,
    initial_guess_max,
    output_scaling_factors_dict,
):
    vk = 1.0
    rk = 0.0
    rule_applied = "2.4 (Default)"  # Corresponds to Rule 2 default in scale.txt if no info
    if (
        explicit_lower_bound is not None
        and explicit_upper_bound is not None
        and not np.isclose(explicit_upper_bound, explicit_lower_bound)
    ):
        # Implements Eqs. (4.250) and (4.251) from scale.txt
        # This is Rule 2.a from scale.txt
        vk = 1.0 / (explicit_upper_bound - explicit_lower_bound)
        rk = 0.5 - explicit_upper_bound / (explicit_upper_bound - explicit_lower_bound)
        rule_applied = "2.1.a (Explicit Bounds)"
    elif (
        initial_guess_min is not None
        and initial_guess_max is not None
        and not np.isclose(initial_guess_max, initial_guess_min)
    ):
        # Implements Eqs. (4.250) and (4.251) using guess range
        # This is Rule 2.b from scale.txt
        vk = 1.0 / (initial_guess_max - initial_guess_min)
        rk = 0.5 - initial_guess_max / (initial_guess_max - initial_guess_min)
        rule_applied = "2.1.b (Initial Guess Range)"
    output_scaling_factors_dict[var_name] = {"v": vk, "r": rk, "rule": rule_applied}
    return vk, rk


# --- generate_physical_guess_and_ranges ---
def generate_physical_guess_and_ranges(bank_angle_min_deg, num_guess_points=21):
    h0_actual, v0_actual = 260000.0, 25600.0
    phi0_actual, theta0_actual = 0.0 * DEG2RAD, 0.0 * DEG2RAD
    gamma0_actual, psi0_actual = -1.0 * DEG2RAD, 90.0 * DEG2RAD
    hf_actual, vf_actual = 80000.0, 2500.0
    gammaF_actual = -5.0 * DEG2RAD
    alpha0_actual = 15.0 * DEG2RAD
    beta0_actual = -30.0 * DEG2RAD
    if beta0_actual < bank_angle_min_deg * DEG2RAD:
        beta0_actual = bank_angle_min_deg * DEG2RAD
    if beta0_actual > 1.0 * DEG2RAD:
        beta0_actual = 1.0 * DEG2RAD

    t_param = np.linspace(0, 1, num_guess_points)
    states_physical_traj_flat = {
        "h": h0_actual + (hf_actual - h0_actual) * t_param,
        "phi": np.full(num_guess_points, phi0_actual),
        "theta": np.full(num_guess_points, theta0_actual),
        "v": v0_actual + (vf_actual - v0_actual) * t_param,
        "gamma": gamma0_actual + (gammaF_actual - gamma0_actual) * t_param,
        "psi": np.full(num_guess_points, psi0_actual),
    }
    num_control_points = num_guess_points - 1 if num_guess_points > 1 else 1
    controls_physical_traj_flat = {
        "alpha": np.full(num_control_points, alpha0_actual),
        "beta": np.full(num_control_points, beta0_actual),
    }
    initial_guess_ranges = {}
    for var_name, traj in states_physical_traj_flat.items():
        initial_guess_ranges[var_name] = {"min": np.min(traj), "max": np.max(traj)}
    for var_name, traj in controls_physical_traj_flat.items():
        if traj.size > 0:
            initial_guess_ranges[var_name] = {"min": np.min(traj), "max": np.max(traj)}
        else:
            initial_guess_ranges[var_name] = {"min": None, "max": None}
    return initial_guess_ranges, states_physical_traj_flat, controls_physical_traj_flat


class AutoscaledProblem:
    def __init__(self, name, initial_guess_ranges_physical=None):
        self.internal_problem = tl.Problem(name, auto_scaling=False)
        self.initial_guess_ranges_physical = (
            initial_guess_ranges_physical if initial_guess_ranges_physical else {}
        )
        self.scaling_factors = {}
        self._physical_sx = {}
        self._tilde_sx = {}
        self._state_names_ordered = []
        self._control_names_ordered = []
        self.time_symbol = None
        self._raw_states_physical_guess_flat = {}
        self._raw_controls_physical_guess_flat = {}
        self._raw_terminal_time_guess = None
        self.objective_scale_weight_ = 1.0

    def time(self, initial, free_final):
        self.time_symbol = self.internal_problem.time(initial=initial, free_final=free_final)

    def _determine_and_store_scaling(self, var_name, scale_guide_L, scale_guide_U, op_L, op_U):
        guess_min = self.initial_guess_ranges_physical.get(var_name, {}).get("min")
        guess_max = self.initial_guess_ranges_physical.get(var_name, {}).get("max")
        vk, rk = get_scaling_params_generalized(
            var_name, scale_guide_L, scale_guide_U, guess_min, guess_max, self.scaling_factors
        )
        return vk, rk

    def _get_scaled_props(self, var_name, vk, rk, initial_P, final_P, lower_P, upper_P):
        scaled_def = {}
        if initial_P is not None:
            scaled_def["initial"] = vk * initial_P + rk
        if final_P is not None:
            scaled_def["final"] = vk * final_P + rk

        if self.scaling_factors[var_name]["rule"] != "2.4 (Default)":
            # If scaled based on explicit or guess range, map to [-0.5, 0.5]
            scaled_def["lower"] = -0.5
            scaled_def["upper"] = 0.5
        else:
            scaled_def["lower"] = (vk * lower_P + rk) if lower_P is not None else None
            scaled_def["upper"] = (vk * upper_P + rk) if upper_P is not None else None
        return scaled_def

    def state(
        self,
        name_physical,
        initial=None,
        final=None,
        lower=None,
        upper=None,
        scale_guide_lower=None,
        scale_guide_upper=None,
    ):
        vk, rk = self._determine_and_store_scaling(
            name_physical, scale_guide_lower, scale_guide_upper, lower, upper
        )
        scaled_props = self._get_scaled_props(name_physical, vk, rk, initial, final, lower, upper)

        tilde_name = f"{name_physical}_tilde"
        tilde_symbol = self.internal_problem.state(tilde_name, **scaled_props)
        self._tilde_sx[name_physical] = tilde_symbol
        if np.isclose(vk, 0):
            raise ValueError(f"Scaling factor 'v' for {name_physical} is zero.")
        self._physical_sx[name_physical] = (tilde_symbol - rk) / vk
        if name_physical not in self._state_names_ordered:
            self._state_names_ordered.append(name_physical)
        return self._physical_sx[name_physical]

    def control(
        self, name_physical, lower=None, upper=None, scale_guide_lower=None, scale_guide_upper=None
    ):
        vk, rk = self._determine_and_store_scaling(
            name_physical, scale_guide_lower, scale_guide_upper, lower, upper
        )
        scaled_props = self._get_scaled_props(name_physical, vk, rk, None, None, lower, upper)

        tilde_name = f"{name_physical}_tilde"
        tilde_symbol = self.internal_problem.control(tilde_name, **scaled_props)
        self._tilde_sx[name_physical] = tilde_symbol
        if np.isclose(vk, 0):
            raise ValueError(f"Scaling factor 'v' for {name_physical} is zero.")
        self._physical_sx[name_physical] = (tilde_symbol - rk) / vk
        if name_physical not in self._control_names_ordered:
            self._control_names_ordered.append(name_physical)
        return self._physical_sx[name_physical]

    def symbol(self, name_physical):
        if name_physical in self._physical_sx:
            return self._physical_sx[name_physical]
        raise KeyError(
            f"Physical symbol for '{name_physical}' not defined. Define it as state or control first."
        )

    def dynamics(self, dynamics_dict_physical):
        scaled_dynamics_dict = {}
        for name_physical, physical_rhs_expr in dynamics_dict_physical.items():
            if name_physical not in self.scaling_factors or name_physical not in self._tilde_sx:
                raise KeyError(
                    f"Scaling factors or tilde symbol for '{name_physical}' not found. Ensure it's a defined state."
                )
            vk = self.scaling_factors[name_physical]["v"]
            # physical_rhs_expr is f_phys(y_phys, u_phys, t)
            # This implements Rule 3: ODE Defect Scaling (W_f = V_y)
            # d(y_tilde)/dt = vk * f_phys(y_phys(y_tilde), u_phys(u_tilde), t)
            scaled_dynamics_dict[self._tilde_sx[name_physical]] = vk * physical_rhs_expr
        self.internal_problem.dynamics(scaled_dynamics_dict)

    def subject_to(self, constraint_expr_physical, constraint_scale_weight=1.0):
        # constraint_expr_physical is g(y_phys, u_phys, t) <= C_phys or similar.
        # y_phys and u_phys are CasADi symbols already defined in terms of y_tilde, u_tilde.
        # This implements Rule 4: Path Constraint Scaling by W_g
        # The scaled constraint is constraint_scale_weight * constraint_expr_physical.
        # Default weight is 1.0 (Rule 4.b if no other info).
        self.internal_problem.subject_to(constraint_scale_weight * constraint_expr_physical)

    def minimize(self, objective_expr_physical, objective_scale_weight=1.0):
        # objective_expr_physical is J(y_phys, u_phys, t).
        # y_phys and u_phys are CasADi symbols already defined in terms of y_tilde, u_tilde.
        # This implements Rule 5: Objective Scaling by w_0
        # The scaled objective is objective_scale_weight * objective_expr_physical.
        # Default weight is 1.0 (user-specified option in Rule 5.b).
        self.objective_scale_weight_ = objective_scale_weight
        self.internal_problem.minimize(objective_scale_weight * objective_expr_physical)

    def set_mesh(self, degrees, points):
        self._current_mesh_degrees = degrees
        self.internal_problem.set_mesh(degrees, points)

    def set_initial_guess(
        self, states_physical_flat_dict, controls_physical_flat_dict, terminal_time
    ):
        self._raw_states_physical_guess_flat = states_physical_flat_dict
        self._raw_controls_physical_guess_flat = controls_physical_flat_dict
        self._raw_terminal_time_guess = terminal_time

    def _prepare_and_set_scaled_guess_on_internal_problem(self, polynomial_degrees_for_solve=None):
        if polynomial_degrees_for_solve is None:
            if not hasattr(self, "_current_mesh_degrees"):
                raise ValueError(
                    "Polynomial degrees for guess preparation not provided and no mesh set."
                )
            polynomial_degrees_for_solve = self._current_mesh_degrees

        scaled_states_guess_intervals = []
        scaled_controls_guess_intervals = []
        num_intervals = len(polynomial_degrees_for_solve)

        for interval_idx in range(num_intervals):
            N_poly_degree_state = polynomial_degrees_for_solve[interval_idx]
            num_state_pts_interval = N_poly_degree_state + 1
            current_scaled_states_list = []
            for var_name in self._state_names_ordered:
                phys_traj = self._raw_states_physical_guess_flat[var_name]
                if len(phys_traj) >= num_state_pts_interval:
                    actual_vals_interval = phys_traj[:num_state_pts_interval]
                else:
                    actual_vals_interval = np.interp(
                        np.linspace(0, 1, num_state_pts_interval),
                        np.linspace(0, 1, len(phys_traj)),
                        phys_traj,
                    )
                vk = self.scaling_factors[var_name]["v"]
                rk = self.scaling_factors[var_name]["r"]
                tilde_vals = vk * actual_vals_interval + rk
                current_scaled_states_list.append(tilde_vals)
            scaled_states_guess_intervals.append(np.vstack(current_scaled_states_list))

            num_control_pts_interval = N_poly_degree_state
            current_scaled_controls_list = []
            if num_control_pts_interval > 0:
                for var_name in self._control_names_ordered:
                    phys_traj_ctrl = self._raw_controls_physical_guess_flat[var_name]
                    if len(phys_traj_ctrl) >= num_control_pts_interval:
                        actual_vals_ctrl_interval = phys_traj_ctrl[:num_control_pts_interval]
                    else:
                        actual_vals_ctrl_interval = (
                            np.interp(
                                np.linspace(0, 1, num_control_pts_interval),
                                np.linspace(0, 1, len(phys_traj_ctrl)),
                                phys_traj_ctrl,
                            )
                            if len(phys_traj_ctrl) > 0
                            else np.zeros(num_control_pts_interval)
                        )
                    vk = self.scaling_factors[var_name]["v"]
                    rk = self.scaling_factors[var_name]["r"]
                    tilde_vals_ctrl = vk * actual_vals_ctrl_interval + rk
                    current_scaled_controls_list.append(tilde_vals_ctrl)
                scaled_controls_guess_intervals.append(np.vstack(current_scaled_controls_list))
            elif num_intervals == 1 and num_control_pts_interval == 0:
                scaled_controls_guess_intervals.append(
                    np.empty((len(self._control_names_ordered), 0))
                )

        self.internal_problem.set_initial_guess(
            states=scaled_states_guess_intervals,
            controls=scaled_controls_guess_intervals,
            terminal_time=self._raw_terminal_time_guess,
        )

    def get_symbolic_vars_for_postprocessing(self):
        output = {
            "scaling_factors": self.scaling_factors,
            "objective_scale_weight": getattr(self, "objective_scale_weight_", 1.0),
        }
        if self.time_symbol is not None:
            output["t"] = self.time_symbol
        for name_physical, tilde_sym in self._tilde_sx.items():
            output[f"{name_physical}_tilde"] = tilde_sym
        return output


# --- analyze_solution ---
def analyze_solution(
    solution,
    symbolic_vars,
    example_name,
    example_num,
    method,
    bank_min_deg,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    objective_scale_weight = symbolic_vars.get("objective_scale_weight", 1.0)

    if solution.success:
        final_time = solution.final_time

        if np.isclose(objective_scale_weight, 0):
            final_theta_actual_rad = -solution.objective
            print(
                "\nWARNING: Objective scale weight is close to zero. Physical objective calculation might be inaccurate."
            )
        else:
            # solution.objective is w_0 * J_physical. To get J_physical, divide by w_0.
            # Here, J_physical is -theta_physical_rad
            final_theta_actual_rad = -solution.objective / objective_scale_weight

        final_theta_actual_deg = final_theta_actual_rad * RAD2DEG

        J_formatted = f"{final_theta_actual_rad:.7e}".replace("e-0", "e-").replace("e+0", "e+")
        tf_formatted = f"{final_time:.7e}".replace("e+0", "e+")
        heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
        bank_str = f"β ∈ [{bank_min_deg}°, 1°]"
        print(f"\nExample {example_num}: {example_name} ({method})")
        print(f"Parameters: {bank_str}, {heat_str}")
        print("Optimal Results:")
        print(f"  J* = {J_formatted}  (final latitude in radians, -objective_physical)")
        print(f"  t_F* = {tf_formatted}  (final time in seconds)")
        print(f"  Final latitude: {final_theta_actual_deg:.4f}°")
        print(
            f"  (Solver objective was {solution.objective:.7e}, objective_scale_weight was {objective_scale_weight:.3e})"
        )

        if literature_J is not None and literature_tf is not None:
            J_abs_lit = abs(literature_J)
            tf_abs_lit = abs(literature_tf)
            J_diff = (
                abs(final_theta_actual_rad - literature_J) / J_abs_lit * 100
                if J_abs_lit > 1e-9
                else 0
            )
            tf_diff = abs(final_time - literature_tf) / tf_abs_lit * 100 if tf_abs_lit > 1e-9 else 0
            print("\nComparison with literature values:")
            print(f"  Literature J* = {literature_J:.7e}")
            print(f"  Literature t_F* = {literature_tf:.7e}")
            print(f"  J* difference: {J_diff:.4f}%")
            print(f"  t_F* difference: {tf_diff:.4f}%")

        if (
            method == "Adaptive Mesh"
            and hasattr(solution, "polynomial_degrees")
            and solution.polynomial_degrees is not None
        ):
            print("\nFinal mesh details (Adaptive):")
            print(f"  Polynomial degrees: {solution.polynomial_degrees}")
            if hasattr(solution, "mesh_points") and solution.mesh_points is not None:
                print(f"  Number of mesh intervals: {len(solution.mesh_points) - 1}")
                print(f"  Mesh points: {np.array2string(solution.mesh_points, precision=3)}")
        return True
    else:
        print(f"\nExample {example_num} Solution ({method}) Failed:")
        print(f"  Reason: {solution.message}")
        return False


# --- plot_solution ---
def plot_solution(solution, symbolic_vars, plot_title_suffix=""):
    scaling_factors = symbolic_vars["scaling_factors"]

    def unscale_var(var_tilde_vals, var_name):
        sf = scaling_factors[var_name]
        if np.isclose(sf["v"], 0):
            print(f"Warning: Scaling factor 'v' for {var_name} is zero during unscaling.")
            return var_tilde_vals
        return (var_tilde_vals - sf["r"]) / sf["v"]

    time_h, h_tilde_vals = solution.get_trajectory(symbolic_vars["h_tilde"])
    time_phi, phi_tilde_vals = solution.get_trajectory(symbolic_vars["phi_tilde"])
    time_theta, theta_tilde_vals = solution.get_trajectory(symbolic_vars["theta_tilde"])
    time_v, v_tilde_vals = solution.get_trajectory(symbolic_vars["v_tilde"])
    time_gamma, gamma_tilde_vals = solution.get_trajectory(symbolic_vars["gamma_tilde"])
    time_psi, psi_tilde_vals = solution.get_trajectory(symbolic_vars["psi_tilde"])
    time_alpha, alpha_tilde_vals = solution.get_trajectory(symbolic_vars["alpha_tilde"])
    time_beta, beta_tilde_vals = solution.get_trajectory(symbolic_vars["beta_tilde"])

    h_vals_actual = unscale_var(h_tilde_vals, "h")
    phi_vals_actual = unscale_var(phi_tilde_vals, "phi")
    theta_vals_actual = unscale_var(theta_tilde_vals, "theta")
    v_vals_actual = unscale_var(v_tilde_vals, "v")
    gamma_vals_actual = unscale_var(gamma_tilde_vals, "gamma")
    psi_vals_actual = unscale_var(psi_tilde_vals, "psi")
    alpha_vals_actual = unscale_var(alpha_tilde_vals, "alpha")
    beta_vals_actual = unscale_var(beta_tilde_vals, "beta")

    phi_deg = phi_vals_actual * RAD2DEG
    theta_deg = theta_vals_actual * RAD2DEG
    gamma_deg = gamma_vals_actual * RAD2DEG
    psi_deg = psi_vals_actual * RAD2DEG
    alpha_deg = alpha_vals_actual * RAD2DEG
    beta_deg = beta_vals_actual * RAD2DEG

    main_plot_title = f"Space Shuttle Reentry {plot_title_suffix} (Autoscaled - Physical Units)"
    fig_states, axs_states = plt.subplots(3, 2, figsize=(12, 12))
    fig_states.suptitle(f"State Variables (Physical Units) {plot_title_suffix}", fontsize=16)
    axs_states[0, 0].plot(time_h, h_vals_actual / 1e5)
    axs_states[0, 0].set_title("Altitude (10⁵ ft)")
    axs_states[0, 1].plot(time_v, v_vals_actual / 1e3)
    axs_states[0, 1].set_title("Velocity (10³ ft/s)")
    axs_states[1, 0].plot(time_phi, phi_deg)
    axs_states[1, 0].set_title("Longitude (deg)")
    axs_states[1, 1].plot(time_gamma, gamma_deg)
    axs_states[1, 1].set_title("Flight Path Angle (deg)")
    axs_states[2, 0].plot(time_theta, theta_deg)
    axs_states[2, 0].set_title("Latitude (deg)")
    axs_states[2, 1].plot(time_psi, psi_deg)
    axs_states[2, 1].set_title("Azimuth (deg)")
    for ax_row in axs_states:
        for ax in ax_row:
            ax.grid(True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig_ctrl, axs_ctrl = plt.subplots(2, 1, figsize=(10, 7))
    fig_ctrl.suptitle(f"Control Variables (Physical Units) {plot_title_suffix}", fontsize=16)
    axs_ctrl[0].plot(time_alpha, alpha_deg)
    axs_ctrl[0].set_title("Angle of Attack (deg)")
    axs_ctrl[1].plot(time_beta, beta_deg)
    axs_ctrl[1].set_title("Bank Angle (deg)")
    for ax in axs_ctrl:
        ax.grid(True)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(phi_deg, theta_deg, h_vals_actual / 1e5)
    ax_3d.set_xlabel("Longitude (deg)")
    ax_3d.set_ylabel("Latitude (deg)")
    ax_3d.set_zlabel("Altitude (10⁵ ft)")
    ax_3d.set_title(main_plot_title)
    plt.show()

    print("Displaying TrajectoLab standard solution plot (scaled variables of internal problem)...")
    solution.plot()


# --- Modified Solver Functions ---
def solve_with_fixed_mesh(
    autoscaled_problem,
    example_name,
    example_num,
    bank_min_deg,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    if not hasattr(autoscaled_problem, "_current_mesh_degrees"):
        raise ValueError(
            "Mesh must be set on AutoscaledProblem before calling solve_with_fixed_mesh."
        )

    autoscaled_problem._prepare_and_set_scaled_guess_on_internal_problem()

    try:
        current_mesh_degrees = (
            autoscaled_problem.internal_problem._collocation_options.polynomial_degrees
        )
        current_num_intervals = len(current_mesh_degrees)
        mesh_info_str = f"Mesh: {current_num_intervals} intervals, Degrees: {current_mesh_degrees}"
    except AttributeError:
        mesh_info_str = "Mesh: (Info not directly accessible from internal problem)"

    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min_deg}°, 1°]"
    print(f"\nSolving Example {example_num}: {example_name} (Fixed Mesh, Autoscaled)")
    print(f"Parameters: {bank_str}, {heat_str}")
    print(mesh_info_str)

    solution = tl.solve_fixed_mesh(
        autoscaled_problem.internal_problem,
        nlp_options={
            "ipopt.max_iter": 2000,
            "ipopt.mumps_pivtol": 5e-7,
            "ipopt.mumps_mem_percent": 50000,
            "ipopt.linear_solver": "mumps",
            "ipopt.constr_viol_tol": 1e-7,
            "ipopt.print_level": 5,
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.tol": 1e-8,
        },
    )
    symbolic_vars_for_analysis = autoscaled_problem.get_symbolic_vars_for_postprocessing()
    analyze_solution(
        solution,
        symbolic_vars_for_analysis,
        example_name,
        example_num,
        "Fixed Mesh",
        bank_min_deg,
        heating_limit,
        literature_J,
        literature_tf,
    )
    return solution


def solve_with_adaptive_mesh(
    autoscaled_problem,
    example_name,
    example_num,
    bank_min_deg,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
    error_tol=1e-5,
    max_adapt_iter=10,
):
    if not hasattr(autoscaled_problem, "_current_mesh_degrees"):
        raise ValueError(
            "Initial mesh must be set on AutoscaledProblem before calling solve_with_adaptive_mesh."
        )

    autoscaled_problem._prepare_and_set_scaled_guess_on_internal_problem()

    try:
        current_mesh_degrees = (
            autoscaled_problem.internal_problem._collocation_options.polynomial_degrees
        )
        current_num_intervals = len(current_mesh_degrees)
        mesh_info_str = (
            f"Initial Mesh: {current_num_intervals} intervals, Degrees: {current_mesh_degrees}"
        )
    except AttributeError:
        mesh_info_str = "Initial Mesh: (Info not directly accessible)"

    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min_deg}°, 1°]"
    print(f"\nSolving Example {example_num}: {example_name} (Adaptive Mesh, Autoscaled)")
    print(f"Parameters: {bank_str}, {heat_str}, Error Tol: {error_tol}, Max Iter: {max_adapt_iter}")
    print(mesh_info_str)

    solution = tl.solve_adaptive(
        autoscaled_problem.internal_problem,
        error_tolerance=error_tol,
        max_iterations=max_adapt_iter,
        min_polynomial_degree=4,
        max_polynomial_degree=10,
        nlp_options={
            "ipopt.max_iter": 2000,
            "ipopt.print_level": 0,
            "ipopt.tol": error_tol * 10,
            "ipopt.constr_viol_tol": error_tol * 10,
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
        },
    )
    symbolic_vars_for_analysis = autoscaled_problem.get_symbolic_vars_for_postprocessing()
    analyze_solution(
        solution,
        symbolic_vars_for_analysis,
        example_name,
        example_num,
        "Adaptive Mesh",
        bank_min_deg,
        heating_limit,
        literature_J,
        literature_tf,
    )
    return solution


def main():
    lit_J_ex137 = 5.9587608e-1
    lit_tf_ex137 = 2.0085881e3
    example_details = {
        "name": "SHUTTLE MAX CROSSRANGE",
        "num": "10.137 (Book Example - Corrected Scaling Logic)",
        "bank_min_deg": -90.0,
        # "heating_limit": 70.0,
        "heating_limit": None,  # No heating constraint for this run
        "lit_J": lit_J_ex137,
        "lit_tf": lit_tf_ex137,
    }

    user_scaling_guides = {
        "h": {"lower": 0.0, "upper": 260000.0},
        "theta": {"lower": -89.0 * DEG2RAD, "upper": 89.0 * DEG2RAD},
        "v": {"lower": 1.0, "upper": 25600.0},
        "gamma": {"lower": -89.0 * DEG2RAD, "upper": 89.0 * DEG2RAD},
        "alpha": {"lower": -90.0 * DEG2RAD, "upper": 90.0 * DEG2RAD},
        "beta": {"lower": example_details["bank_min_deg"] * DEG2RAD, "upper": 1.0 * DEG2RAD},
    }

    initial_guess_ranges, states_physical_traj_flat, controls_physical_traj_flat = (
        generate_physical_guess_and_ranges(
            bank_angle_min_deg=example_details["bank_min_deg"],
            num_guess_points=21,
        )
    )

    ap = AutoscaledProblem(
        "Space Shuttle Reentry (Autoscaled - Corrected)",
        initial_guess_ranges_physical=initial_guess_ranges,
    )

    ap.time(initial=0.0, free_final=True)

    h = ap.state(
        "h",
        initial=260000.0,
        final=80000.0,
        lower=0.0,
        upper=260000.0,
        scale_guide_lower=user_scaling_guides.get("h", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("h", {}).get("upper"),
    )
    ap.state(
        "phi",
        initial=0.0 * DEG2RAD,
        scale_guide_lower=user_scaling_guides.get("phi", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("phi", {}).get("upper"),
    )
    theta = ap.state(
        "theta",
        initial=0.0 * DEG2RAD,
        lower=-89.0 * DEG2RAD,
        upper=89.0 * DEG2RAD,
        scale_guide_lower=user_scaling_guides.get("theta", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("theta", {}).get("upper"),
    )
    v = ap.state(
        "v",
        initial=25600.0,
        final=2500.0,
        lower=1.0,
        upper=25600.0,
        scale_guide_lower=user_scaling_guides.get("v", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("v", {}).get("upper"),
    )
    gamma = ap.state(
        "gamma",
        initial=-1.0 * DEG2RAD,
        final=-5.0 * DEG2RAD,
        lower=-89.0 * DEG2RAD,
        upper=89.0 * DEG2RAD,
        scale_guide_lower=user_scaling_guides.get("gamma", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("gamma", {}).get("upper"),
    )
    psi = ap.state(
        "psi",
        initial=90.0 * DEG2RAD,
        scale_guide_lower=user_scaling_guides.get("psi", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("psi", {}).get("upper"),
    )

    alpha = ap.control(
        "alpha",
        lower=-90.0 * DEG2RAD,
        upper=90.0 * DEG2RAD,
        scale_guide_lower=user_scaling_guides.get("alpha", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("alpha", {}).get("upper"),
    )
    beta = ap.control(
        "beta",
        lower=example_details["bank_min_deg"] * DEG2RAD,
        upper=1.0 * DEG2RAD,
        scale_guide_lower=user_scaling_guides.get("beta", {}).get("lower"),
        scale_guide_upper=user_scaling_guides.get("beta", {}).get("upper"),
    )

    eps_div = 1e-10
    r_planet_dist = R_EARTH + h
    rho_atm = RHO0 * ca.exp(-h / H_R)
    g_local = MU_EARTH / (r_planet_dist**2)
    alpha_deg_calc = alpha * RAD2DEG
    CL = A0_CL + A1_CL * alpha_deg_calc
    CD = B0_CD + B1_CD * alpha_deg_calc + B2_CD * alpha_deg_calc**2
    q_dynamic = 0.5 * rho_atm * v**2
    L_force = q_dynamic * CL * S_REF
    D_force = q_dynamic * CD * S_REF

    dh_dt_physical = v * ca.sin(gamma)
    dphi_dt_physical = (v / r_planet_dist) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps_div)
    dtheta_dt_physical = (v / r_planet_dist) * ca.cos(gamma) * ca.cos(psi)
    dv_dt_physical = -(D_force / MASS) - g_local * ca.sin(gamma)
    dgamma_dt_physical = (L_force / (MASS * v + eps_div)) * ca.cos(beta) + ca.cos(gamma) * (
        (v / r_planet_dist) - (g_local / (v + eps_div))
    )
    dpsi_dt_physical = (L_force * ca.sin(beta) / (MASS * v * ca.cos(gamma) + eps_div)) + (
        v / (r_planet_dist * (ca.cos(theta) + eps_div))
    ) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta)

    ap.dynamics(
        {
            "h": dh_dt_physical,
            "phi": dphi_dt_physical,
            "theta": dtheta_dt_physical,
            "v": dv_dt_physical,
            "gamma": dgamma_dt_physical,
            "psi": dpsi_dt_physical,
        }
    )

    if example_details["heating_limit"] is not None:
        q_r_heat = 17700 * ca.sqrt(rho_atm) * (0.0001 * v) ** 3.07
        q_a_poly_heat = (
            C0_QA + C1_QA * alpha_deg_calc + C2_QA * alpha_deg_calc**2 + C3_QA * alpha_deg_calc**3
        )
        q_heat_actual = q_a_poly_heat * q_r_heat  # This is a physical value expression

        # Physical constraint: q_heat_actual <= example_details["heating_limit"]
        # The expression passed to subject_to for an inequality g(x) <= C is g(x) - C.
        # We want to scale this g(x)-C or g(x) itself.
        # If heating_limit is a good measure of the scale of q_heat_actual,
        # then 1.0 / heating_limit is a reasonable W_g.
        constraint_expression = q_heat_actual <= example_details["heating_limit"]
        path_constraint_weight = 1.0
        if example_details["heating_limit"] > 1e-6:
            path_constraint_weight = 1.0 / example_details["heating_limit"]

        print(f"Path constraint (heating) weight W_g: {path_constraint_weight:.3e}")
        ap.subject_to(constraint_expression, constraint_scale_weight=path_constraint_weight)

    # Objective is to maximize theta_final, so minimize -theta_final.
    # -theta is the physical objective expression.
    # Rule 5 from scale.txt involves scaling this physical objective expression.
    # User-specified w_0 is an option. Defaulting to 1.0 if no other info.
    objective_w0 = 1.0
    print(f"Objective weight w_0: {objective_w0:.3e}")
    ap.minimize(-theta, objective_scale_weight=objective_w0)

    print("\n--- Scaling Information (from AutoscaledProblem) ---")
    for var_name, sf_info in sorted(ap.scaling_factors.items()):
        print(
            f"Var: {var_name:<8s} | Rule: {sf_info['rule']:<28s} | v: {sf_info['v']:.3e} | r: {sf_info['r']:.3f}"
        )
    print("-------------------------\n")

    fixed_mesh_poly_degrees = [20] * 15
    num_intervals_fixed = len(fixed_mesh_poly_degrees)
    mesh_points_fixed = np.linspace(-1.0, 1.0, num_intervals_fixed + 1)
    ap.set_mesh(fixed_mesh_poly_degrees, mesh_points_fixed)
    ap.set_initial_guess(
        states_physical_flat_dict=states_physical_traj_flat,
        controls_physical_flat_dict=controls_physical_traj_flat,
        terminal_time=example_details["lit_tf"] or 2000.0,
    )
    solution_fixed = solve_with_fixed_mesh(
        ap,
        example_details["name"],
        example_details["num"],
        example_details["bank_min_deg"],
        example_details["heating_limit"],
        example_details["lit_J"],
        example_details["lit_tf"],
    )
    if solution_fixed.success:
        plot_solution(
            solution_fixed,
            ap.get_symbolic_vars_for_postprocessing(),
            plot_title_suffix="(Fixed Mesh - Corrected Scaling)",
        )

    adaptive_initial_poly_degrees = [6] * 9
    num_intervals_adaptive_initial = len(adaptive_initial_poly_degrees)
    mesh_points_adaptive_initial = np.linspace(-1.0, 1.0, num_intervals_adaptive_initial + 1)
    ap.set_mesh(adaptive_initial_poly_degrees, mesh_points_adaptive_initial)
    ap.set_initial_guess(
        states_physical_flat_dict=states_physical_traj_flat,
        controls_physical_flat_dict=controls_physical_traj_flat,
        terminal_time=example_details["lit_tf"] or 2000.0,
    )
    solution_adaptive = solve_with_adaptive_mesh(
        ap,
        example_details["name"],
        example_details["num"],
        example_details["bank_min_deg"],
        example_details["heating_limit"],
        example_details["lit_J"],
        example_details["lit_tf"],
        error_tol=1e-6,
        max_adapt_iter=20,
    )
    if solution_adaptive.success:
        plot_solution(
            solution_adaptive,
            ap.get_symbolic_vars_for_postprocessing(),
            plot_title_suffix="(Adaptive Mesh - Corrected Scaling)",
        )


if __name__ == "__main__":
    main()
