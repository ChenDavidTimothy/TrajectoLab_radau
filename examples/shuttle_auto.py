import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


# --- Constants ---
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Physical Constants for the Shuttle Problem
A0_CL = -0.20704
A1_CL = 0.029244
B0_CD = 0.07854
B1_CD = -0.61592e-2
B2_CD = 0.621408e-3
C0_QA = 1.0672181
C1_QA = -0.19213774e-1
C2_QA = 0.21286289e-3
C3_QA = -0.10117249e-5
MU_EARTH = 0.14076539e17  # Gravitational parameter (ft^3/s^2)
R_EARTH = 20902900.0  # Earth radius (ft)
S_REF = 2690.0  # Reference area (ft^2)
RHO0 = 0.002378  # Sea level atmospheric density (slug/ft^3)
H_R = 23800.0  # Density scale height (ft)
G0 = 32.174  # Gravitational acceleration (ft/s^2)
WEIGHT = 203000.0  # Vehicle weight (lb)
MASS = WEIGHT / G0  # Vehicle mass (slugs)


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
    rule_applied = "2.4 (Default)"
    if (
        explicit_lower_bound is not None
        and explicit_upper_bound is not None
        and not np.isclose(explicit_upper_bound, explicit_lower_bound)
    ):
        vk = 1.0 / (explicit_upper_bound - explicit_lower_bound)
        rk = 0.5 - explicit_upper_bound / (explicit_upper_bound - explicit_lower_bound)
        rule_applied = "2.1.a (Explicit Bounds)"
    elif (
        initial_guess_min is not None
        and initial_guess_max is not None
        and not np.isclose(initial_guess_max, initial_guess_min)
    ):
        vk = 1.0 / (initial_guess_max - initial_guess_min)
        rk = 0.5 - initial_guess_max / (initial_guess_max - initial_guess_min)
        rule_applied = "2.1.b (Initial Guess Range)"
    output_scaling_factors_dict[var_name] = {"v": vk, "r": rk, "rule": rule_applied}
    return vk, rk


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


def create_shuttle_reentry_problem_generalized(
    explicit_scaling_bounds, initial_guess_ranges, heating_constraint=None, bank_angle_min_deg=-90.0
):
    problem = tl.Problem("Space Shuttle Reentry (Generalized Scaling)")
    final_scaling_factors = {}
    var_physical_props = {
        "h": {"initial": 260000.0, "final": 80000.0, "op_lower": 0.0, "op_upper": 260000.0},
        "phi": {"initial": 0.0 * DEG2RAD, "final": None, "op_lower": None, "op_upper": None},
        "theta": {
            "initial": 0.0 * DEG2RAD,
            "final": None,
            "op_lower": -89.0 * DEG2RAD,
            "op_upper": 89.0 * DEG2RAD,
        },
        "v": {"initial": 25600.0, "final": 2500.0, "op_lower": 1.0, "op_upper": 25600.0},
        "gamma": {
            "initial": -1.0 * DEG2RAD,
            "final": -5.0 * DEG2RAD,
            "op_lower": -89.0 * DEG2RAD,
            "op_upper": 89.0 * DEG2RAD,
        },
        "psi": {"initial": 90.0 * DEG2RAD, "final": None, "op_lower": None, "op_upper": None},
        "alpha": {
            "initial": None,
            "final": None,
            "op_lower": -90.0 * DEG2RAD,
            "op_upper": 90.0 * DEG2RAD,
        },
        "beta": {
            "initial": None,
            "final": None,
            "op_lower": bank_angle_min_deg * DEG2RAD,
            "op_upper": 1.0 * DEG2RAD,
        },
    }
    scaled_var_definitions = {}
    for var_name in ["h", "phi", "theta", "v", "gamma", "psi", "alpha", "beta"]:
        expl_L = explicit_scaling_bounds.get(var_name, {}).get("lower")
        expl_U = explicit_scaling_bounds.get(var_name, {}).get("upper")
        guess_min = initial_guess_ranges.get(var_name, {}).get("min")
        guess_max = initial_guess_ranges.get(var_name, {}).get("max")
        vk, rk = get_scaling_params_generalized(
            var_name, expl_L, expl_U, guess_min, guess_max, final_scaling_factors
        )
        props = var_physical_props[var_name]
        current_s_params = {}
        if props.get("initial") is not None:
            current_s_params["initial"] = vk * props["initial"] + rk
        if props.get("final") is not None:
            current_s_params["final"] = vk * props["final"] + rk
        if final_scaling_factors[var_name]["rule"] != "2.4 (Default)":
            current_s_params["lower"] = -0.5
            current_s_params["upper"] = 0.5
        else:
            op_L, op_U = props.get("op_lower"), props.get("op_upper")
            current_s_params["lower"] = (vk * op_L + rk) if op_L is not None else None
            current_s_params["upper"] = (vk * op_U + rk) if op_U is not None else None
        scaled_var_definitions[var_name] = current_s_params

    t = problem.time(initial=0.0, free_final=True)
    h_tilde = problem.state("h_tilde", **scaled_var_definitions["h"])
    phi_tilde = problem.state("phi_tilde", **scaled_var_definitions["phi"])
    theta_tilde = problem.state("theta_tilde", **scaled_var_definitions["theta"])
    v_tilde = problem.state("v_tilde", **scaled_var_definitions["v"])
    gamma_tilde = problem.state("gamma_tilde", **scaled_var_definitions["gamma"])
    psi_tilde = problem.state("psi_tilde", **scaled_var_definitions["psi"])
    alpha_tilde = problem.control("alpha_tilde", **scaled_var_definitions["alpha"])
    beta_tilde = problem.control("beta_tilde", **scaled_var_definitions["beta"])
    symbolic_vars = {
        "t": t,
        "h_tilde": h_tilde,
        "phi_tilde": phi_tilde,
        "theta_tilde": theta_tilde,
        "v_tilde": v_tilde,
        "gamma_tilde": gamma_tilde,
        "psi_tilde": psi_tilde,
        "alpha_tilde": alpha_tilde,
        "beta_tilde": beta_tilde,
        "scaling_factors": final_scaling_factors,
    }

    def unscale(var_tilde, var_name_local):
        sf = final_scaling_factors[var_name_local]
        if np.isclose(sf["v"], 0):
            raise ValueError(f"Scale factor v for {var_name_local} is zero.")
        return (var_tilde - sf["r"]) / sf["v"]

    h_actual = unscale(h_tilde, "h")
    phi_actual = unscale(phi_tilde, "phi")
    theta_actual = unscale(theta_tilde, "theta")
    v_actual = unscale(v_tilde, "v")
    gamma_actual = unscale(gamma_tilde, "gamma")
    psi_actual = unscale(psi_tilde, "psi")
    alpha_actual = unscale(alpha_tilde, "alpha")
    beta_actual = unscale(beta_tilde, "beta")
    eps_div = 1e-10
    r_planet_dist = R_EARTH + h_actual
    rho_atm = RHO0 * ca.exp(-h_actual / H_R)
    g_local = MU_EARTH / (r_planet_dist * r_planet_dist)
    alpha_deg_calc = alpha_actual * RAD2DEG
    CL = A0_CL + A1_CL * alpha_deg_calc
    CD = B0_CD + B1_CD * alpha_deg_calc + B2_CD * alpha_deg_calc * alpha_deg_calc
    q_dynamic = 0.5 * rho_atm * v_actual * v_actual
    L_force = q_dynamic * CL * S_REF
    D_force = q_dynamic * CD * S_REF
    dh_dt_actual = v_actual * ca.sin(gamma_actual)
    dphi_dt_actual = (
        (v_actual / r_planet_dist)
        * ca.cos(gamma_actual)
        * ca.sin(psi_actual)
        / (ca.cos(theta_actual) + eps_div)
    )
    dtheta_dt_actual = (v_actual / r_planet_dist) * ca.cos(gamma_actual) * ca.cos(psi_actual)
    dv_dt_actual = -(D_force / MASS) - g_local * ca.sin(gamma_actual)
    dgamma_dt_actual = (L_force / (MASS * v_actual + eps_div)) * ca.cos(beta_actual) + ca.cos(
        gamma_actual
    ) * ((v_actual / r_planet_dist) - (g_local / (v_actual + eps_div)))
    dpsi_dt_actual = (
        L_force * ca.sin(beta_actual) / (MASS * v_actual * ca.cos(gamma_actual) + eps_div)
    ) + (v_actual / (r_planet_dist * (ca.cos(theta_actual) + eps_div))) * ca.cos(
        gamma_actual
    ) * ca.sin(psi_actual) * ca.sin(theta_actual)
    problem.dynamics(
        {
            h_tilde: final_scaling_factors["h"]["v"] * dh_dt_actual,
            phi_tilde: final_scaling_factors["phi"]["v"] * dphi_dt_actual,
            theta_tilde: final_scaling_factors["theta"]["v"] * dtheta_dt_actual,
            v_tilde: final_scaling_factors["v"]["v"] * dv_dt_actual,
            gamma_tilde: final_scaling_factors["gamma"]["v"] * dgamma_dt_actual,
            psi_tilde: final_scaling_factors["psi"]["v"] * dpsi_dt_actual,
        }
    )
    if heating_constraint is not None:
        q_r_heat = 17700 * ca.sqrt(rho_atm) * (0.0001 * v_actual) ** 3.07
        q_a_poly_heat = (
            C0_QA + C1_QA * alpha_deg_calc + C2_QA * alpha_deg_calc**2 + C3_QA * alpha_deg_calc**3
        )
        q_heat_actual = q_a_poly_heat * q_r_heat
        problem.subject_to(q_heat_actual <= heating_constraint)
    problem.minimize(-theta_actual)
    return problem, symbolic_vars


def prepare_scaled_initial_guess(
    problem,
    symbolic_vars,
    physical_states_trajectories_flat,
    physical_controls_trajectories_flat,
    polynomial_degrees_for_solve,
    initial_terminal_time,
):
    scaling_factors = symbolic_vars["scaling_factors"]
    scaled_states_guess_intervals = []
    scaled_controls_guess_intervals = []
    state_names_ordered = ["h", "phi", "theta", "v", "gamma", "psi"]
    control_names_ordered = ["alpha", "beta"]
    num_intervals = len(polynomial_degrees_for_solve)

    for interval_idx in range(num_intervals):
        N_poly_degree_state = polynomial_degrees_for_solve[interval_idx]
        num_state_pts_interval = N_poly_degree_state + 1
        current_scaled_states_list = []
        for var_name in state_names_ordered:
            phys_traj = physical_states_trajectories_flat[var_name]
            if len(phys_traj) >= num_state_pts_interval:
                actual_vals_interval = phys_traj[:num_state_pts_interval]
            else:
                actual_vals_interval = np.interp(
                    np.linspace(0, 1, num_state_pts_interval),
                    np.linspace(0, 1, len(phys_traj)),
                    phys_traj,
                )
            vk = scaling_factors[var_name]["v"]
            rk = scaling_factors[var_name]["r"]
            tilde_vals = vk * actual_vals_interval + rk
            current_scaled_states_list.append(tilde_vals)
        scaled_states_guess_intervals.append(np.vstack(current_scaled_states_list))

        num_control_pts_interval = N_poly_degree_state
        current_scaled_controls_list = []
        if num_control_pts_interval > 0:
            for var_name in control_names_ordered:
                phys_traj_ctrl = physical_controls_trajectories_flat[var_name]
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
                vk = scaling_factors[var_name]["v"]
                rk = scaling_factors[var_name]["r"]
                tilde_vals_ctrl = vk * actual_vals_ctrl_interval + rk
                current_scaled_controls_list.append(tilde_vals_ctrl)
            scaled_controls_guess_intervals.append(np.vstack(current_scaled_controls_list))
        elif num_intervals == 1 and num_control_pts_interval == 0:
            scaled_controls_guess_intervals.append(np.empty((len(control_names_ordered), 0)))
    problem.set_initial_guess(
        states=scaled_states_guess_intervals,
        controls=scaled_controls_guess_intervals,
        terminal_time=initial_terminal_time,
    )


def solve_with_fixed_mesh(
    problem,
    symbolic_vars,
    example_name,
    example_num,
    bank_min_deg,
    # Removed polynomial_degrees_solve from args, mesh is pre-set
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min_deg}°, 1°]"
    # Access current mesh info from problem object for printing
    try:  # Robustly access mesh info
        current_mesh_degrees = problem._collocation_options.polynomial_degrees
        current_num_intervals = len(current_mesh_degrees)
        mesh_info_str = f"Mesh: {current_num_intervals} intervals, Degrees: {current_mesh_degrees}"
    except AttributeError:
        mesh_info_str = "Mesh: (Info not directly accessible)"

    print(f"\nSolving Example {example_num}: {example_name} (Fixed Mesh, Gen. Scaled)")
    print(f"Parameters: {bank_str}, {heat_str}")
    print(mesh_info_str)

    solution = tl.solve_fixed_mesh(
        problem,
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
    analyze_solution(
        solution,
        symbolic_vars,
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
    problem,
    symbolic_vars,
    example_name,
    example_num,
    bank_min_deg,
    # Removed initial_polynomial_degrees_adaptive from args, mesh is pre-set for initial solve
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
    error_tol=1e-5,
    max_adapt_iter=10,
):
    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min_deg}°, 1°]"
    try:  # Robustly access mesh info
        current_mesh_degrees = problem._collocation_options.polynomial_degrees
        current_num_intervals = len(current_mesh_degrees)
        mesh_info_str = (
            f"Initial Mesh: {current_num_intervals} intervals, Degrees: {current_mesh_degrees}"
        )
    except AttributeError:
        mesh_info_str = "Initial Mesh: (Info not directly accessible)"

    print(f"\nSolving Example {example_num}: {example_name} (Adaptive Mesh, Gen. Scaled)")
    print(f"Parameters: {bank_str}, {heat_str}, Error Tol: {error_tol}, Max Iter: {max_adapt_iter}")
    print(mesh_info_str)

    solution = tl.solve_adaptive(
        problem,
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
    analyze_solution(
        solution,
        symbolic_vars,
        example_name,
        example_num,
        "Adaptive Mesh",
        bank_min_deg,
        heating_limit,
        literature_J,
        literature_tf,
    )
    return solution


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
    if solution.success:
        final_time = solution.final_time
        final_theta_actual_rad = -solution.objective
        final_theta_actual_deg = final_theta_actual_rad * RAD2DEG
        J_formatted = f"{final_theta_actual_rad:.7e}".replace("e-0", "e-").replace("e+0", "e+")
        tf_formatted = f"{final_time:.7e}".replace("e+0", "e+")
        heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
        bank_str = f"β ∈ [{bank_min_deg}°, 1°]"
        print(f"\nExample {example_num}: {example_name} ({method})")
        print(f"Parameters: {bank_str}, {heat_str}")
        print("Optimal Results:")
        print(f"  J* = {J_formatted}  (final latitude in radians, -objective)")
        print(f"  t_F* = {tf_formatted}  (final time in seconds)")
        print(f"  Final latitude: {final_theta_actual_deg:.4f}°")
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


def plot_solution(solution, symbolic_vars, plot_title_suffix=""):
    scaling_factors = symbolic_vars["scaling_factors"]

    def unscale_var(var_tilde_vals, var_name):
        sf = scaling_factors[var_name]
        if np.isclose(sf["v"], 0):
            return var_tilde_vals  # Should not happen for valid scaling
        return (var_tilde_vals - sf["r"]) / sf["v"]

    # Get scaled trajectories from TrajectoLab solution
    time_h, h_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["h_tilde"])
    time_phi, phi_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["phi_tilde"])
    time_theta, theta_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["theta_tilde"])
    time_v, v_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["v_tilde"])
    time_gamma, gamma_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["gamma_tilde"])
    time_psi, psi_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["psi_tilde"])
    time_alpha, alpha_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["alpha_tilde"])
    time_beta, beta_tilde_vals = solution.get_symbolic_trajectory(symbolic_vars["beta_tilde"])

    # Unscale for plotting physical values
    h_vals_actual = unscale_var(h_tilde_vals, "h")
    phi_vals_actual = unscale_var(phi_tilde_vals, "phi")
    theta_vals_actual = unscale_var(theta_tilde_vals, "theta")
    v_vals_actual = unscale_var(v_tilde_vals, "v")
    gamma_vals_actual = unscale_var(gamma_tilde_vals, "gamma")
    psi_vals_actual = unscale_var(psi_tilde_vals, "psi")
    alpha_vals_actual = unscale_var(alpha_tilde_vals, "alpha")
    beta_vals_actual = unscale_var(beta_tilde_vals, "beta")

    # Convert angles to degrees for plotting if desired
    phi_deg = phi_vals_actual * RAD2DEG
    theta_deg = theta_vals_actual * RAD2DEG
    gamma_deg = gamma_vals_actual * RAD2DEG
    psi_deg = psi_vals_actual * RAD2DEG
    alpha_deg = alpha_vals_actual * RAD2DEG
    beta_deg = beta_vals_actual * RAD2DEG

    main_plot_title = f"Space Shuttle Reentry {plot_title_suffix} (Gen. Scaled - Physical Units)"

    # State Variables Plot
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
    plt.show()

    # Control Variables Plot
    fig_ctrl, axs_ctrl = plt.subplots(2, 1, figsize=(10, 7))
    fig_ctrl.suptitle(f"Control Variables (Physical Units) {plot_title_suffix}", fontsize=16)
    axs_ctrl[0].plot(time_alpha, alpha_deg)
    axs_ctrl[0].set_title("Angle of Attack (deg)")
    axs_ctrl[1].plot(time_beta, beta_deg)
    axs_ctrl[1].set_title("Bank Angle (deg)")
    for ax in axs_ctrl:
        ax.grid(True)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    # 3D Trajectory Plot
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(phi_deg, theta_deg, h_vals_actual / 1e5)  # Using physical altitude
    ax_3d.set_xlabel("Longitude (deg)")
    ax_3d.set_ylabel("Latitude (deg)")
    ax_3d.set_zlabel("Altitude (10⁵ ft)")
    ax_3d.set_title(main_plot_title)
    plt.show()

    # Removed the TrajectoLab standard solution plot to ensure only physical plots are shown:
    print("Displaying TrajectoLab standard solution plot (scaled variables)...")
    solution.plot()


def main():
    lit_J_ex137 = 5.9587608e-1
    lit_tf_ex137 = 2.0085881e3
    example_details = {
        "name": "SHUTTLE MAX CROSSRANGE",
        "num": "10.137 (Book Example)",
        "bank_min_deg": -90.0,
        "heating_limit": None,
        "lit_J": lit_J_ex137,
        "lit_tf": lit_tf_ex137,
    }
    explicit_scaling_bounds = {
        "h": {"lower": 0.0, "upper": 260000.0},
        "theta": {"lower": -89.0 * DEG2RAD, "upper": 89.0 * DEG2RAD},
        "v": {"lower": 1.0, "upper": 25600.0},
        "gamma": {"lower": -89.0 * DEG2RAD, "upper": 89.0 * DEG2RAD},
        "alpha": {"lower": -90.0 * DEG2RAD, "upper": 90.0 * DEG2RAD},
        "beta": {"lower": example_details["bank_min_deg"] * DEG2RAD, "upper": 1.0 * DEG2RAD},
    }
    initial_guess_ranges, states_physical_traj_flat, controls_physical_traj_flat = (
        generate_physical_guess_and_ranges(
            bank_angle_min_deg=example_details["bank_min_deg"], num_guess_points=21
        )
    )
    problem, symbolic_vars = create_shuttle_reentry_problem_generalized(
        explicit_scaling_bounds=explicit_scaling_bounds,
        initial_guess_ranges=initial_guess_ranges,
        heating_constraint=example_details["heating_limit"],
        bank_angle_min_deg=example_details["bank_min_deg"],
    )
    print("\n--- Scaling Information ---")
    for var_name, sf_info in sorted(symbolic_vars["scaling_factors"].items()):
        print(
            f"Var: {var_name:<8s} | Rule: {sf_info['rule']:<28s} | v: {sf_info['v']:.3e} | r: {sf_info['r']:.3f}"
        )
    print("-------------------------\n")

    # --- Solve with Fixed Mesh ---
    fixed_mesh_poly_degrees = [20] * 15  # Example: 15 intervals, degree 20
    num_intervals_fixed = len(fixed_mesh_poly_degrees)
    mesh_points_fixed = np.linspace(-1.0, 1.0, num_intervals_fixed + 1)
    problem.set_mesh(fixed_mesh_poly_degrees, mesh_points_fixed)  # SET MESH FIRST
    prepare_scaled_initial_guess(
        problem,
        symbolic_vars,
        states_physical_traj_flat,
        controls_physical_traj_flat,
        polynomial_degrees_for_solve=fixed_mesh_poly_degrees,
        initial_terminal_time=example_details["lit_tf"] or 2000.0,
    )
    solution_fixed = solve_with_fixed_mesh(
        problem,
        symbolic_vars,
        example_details["name"],
        example_details["num"],
        example_details["bank_min_deg"],
        example_details["heating_limit"],
        example_details["lit_J"],
        example_details["lit_tf"],
    )
    if solution_fixed.success:
        plot_solution(solution_fixed, symbolic_vars, plot_title_suffix="(Fixed Mesh)")

    # --- Solve with Adaptive Mesh ---
    adaptive_initial_poly_degrees = [6] * 9  # Example: 9 intervals, degree 6
    num_intervals_adaptive_initial = len(adaptive_initial_poly_degrees)
    mesh_points_adaptive_initial = np.linspace(-1.0, 1.0, num_intervals_adaptive_initial + 1)
    problem.set_mesh(adaptive_initial_poly_degrees, mesh_points_adaptive_initial)  # SET MESH FIRST
    prepare_scaled_initial_guess(
        problem,
        symbolic_vars,
        states_physical_traj_flat,
        controls_physical_traj_flat,
        polynomial_degrees_for_solve=adaptive_initial_poly_degrees,
        initial_terminal_time=example_details["lit_tf"] or 2000.0,
    )
    solution_adaptive = solve_with_adaptive_mesh(
        problem,
        symbolic_vars,
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
        plot_solution(solution_adaptive, symbolic_vars, plot_title_suffix="(Adaptive Mesh)")


if __name__ == "__main__":
    main()
