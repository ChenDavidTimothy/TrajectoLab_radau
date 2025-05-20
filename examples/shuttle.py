import casadi as ca
import numpy as np

import trajectolab as tl


def create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0):
    """
    Create the Space Shuttle reentry optimal control problem.

    Args:
        heating_constraint: Upper bound on aerodynamic heating (BTU/ft²/sec) or None for no constraint
        bank_angle_min: Minimum bank angle in degrees (default: -90°)

    Returns:
        A tuple containing the TrajectoLab Problem instance and a dictionary of symbolic variables.
    """
    problem = tl.Problem("Space Shuttle Reentry Trajectory")

    # Define constants
    a0 = -0.20704
    a1 = 0.029244
    b0 = 0.07854
    b1 = -0.61592e-2
    b2 = 0.621408e-3
    c0 = 1.0672181
    c1 = -0.19213774e-1
    c2 = 0.21286289e-3
    c3 = -0.10117249e-5
    mu = 0.14076539e17
    Re = 20902900
    S = 2690
    rho0 = 0.002378
    hr = 23800
    g0 = 32.174
    weight = 203000
    mass = weight / g0
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi
    h_scale = 1e5
    v_scale = 1e4

    t = problem.time(initial=0.0, free_final=True)

    h_scaled = problem.state(
        "h_scaled",
        initial=260000.0 / h_scale,
        lower=0.0,
        final=80000.0 / h_scale,
    )
    phi = problem.state("phi", initial=0.0 * deg2rad)
    theta = problem.state(
        "theta",
        initial=0.0 * deg2rad,
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
    )
    v_scaled = problem.state(
        "v_scaled",
        initial=25600.0 / v_scale,
        lower=1.0 / v_scale,
        final=2500.0 / v_scale,
    )
    gamma = problem.state(
        "gamma",
        initial=-1.0 * deg2rad,
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
        final=-5.0 * deg2rad,
    )
    psi = problem.state("psi", initial=90.0 * deg2rad)

    alpha = problem.control("alpha", lower=-90.0 * deg2rad, upper=90.0 * deg2rad)
    beta = problem.control("beta", lower=bank_angle_min * deg2rad, upper=1.0 * deg2rad)

    # Store symbolic variables
    symbolic_vars = {
        "t": t,
        "h_scaled": h_scaled,
        "phi": phi,
        "theta": theta,
        "v_scaled": v_scaled,
        "gamma": gamma,
        "psi": psi,
        "alpha": alpha,
        "beta": beta,
    }

    h_actual = h_scaled * h_scale
    v_actual = v_scaled * v_scale
    eps = 1e-10
    r = Re + h_actual
    rho = rho0 * ca.exp(-h_actual / hr)
    g = mu / (r * r)
    alpha_deg_calc = alpha * rad2deg
    CL = a0 + a1 * alpha_deg_calc
    CD = b0 + b1 * alpha_deg_calc + b2 * alpha_deg_calc * alpha_deg_calc
    q_dyn = 0.5 * rho * v_actual * v_actual
    L = q_dyn * CL * S
    D = q_dyn * CD * S
    qr = 17700 * ca.sqrt(rho) * (0.0001 * v_actual) ** 3.07
    qa = c0 + c1 * alpha_deg_calc + c2 * alpha_deg_calc**2 + c3 * alpha_deg_calc**3
    q_heat = qa * qr

    problem.dynamics(
        {
            h_scaled: (v_actual * ca.sin(gamma)) / h_scale,
            phi: (v_actual / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
            theta: (v_actual / r) * ca.cos(gamma) * ca.cos(psi),
            v_scaled: (-(D / mass) - g * ca.sin(gamma)) / v_scale,
            gamma: (L / (mass * v_actual + eps)) * ca.cos(beta)
            + ca.cos(gamma) * ((v_actual / r) - (g / (v_actual + eps))),
            psi: (1 / (mass * v_actual * ca.cos(gamma) + eps)) * L * ca.sin(beta)
            + (v_actual / (r * (ca.cos(theta) + eps)))
            * ca.cos(gamma)
            * ca.sin(psi)
            * ca.sin(theta),
        }
    )

    if heating_constraint is not None:
        problem.subject_to(q_heat <= heating_constraint)

    problem.minimize(-theta)  # Maximize final latitude

    return problem, symbolic_vars


def prepare_initial_guess(problem, polynomial_degrees, deg2rad):
    states_guess = []
    controls_guess = []
    h_scale = 1e5
    v_scale = 1e4
    h0, v0 = 260000.0 / h_scale, 25600.0 / v_scale
    phi0, theta0 = 0.0, 0.0
    gamma0, psi0 = -1.0 * deg2rad, 90.0 * deg2rad
    hf, vf = 80000.0 / h_scale, 2500.0 / v_scale
    gammaF = -5.0 * deg2rad

    for N in polynomial_degrees:
        t_param = np.linspace(0, 1, N + 1)
        h_vals = h0 + (hf - h0) * t_param
        phi_vals = phi0 * np.ones(N + 1)
        theta_vals = theta0 * np.ones(N + 1)
        v_vals = v0 + (vf - v0) * t_param
        gamma_vals = gamma0 + (gammaF - gamma0) * t_param
        psi_vals = psi0 * np.ones(N + 1)
        state_array = np.vstack([h_vals, phi_vals, theta_vals, v_vals, gamma_vals, psi_vals])
        states_guess.append(state_array)
        alpha_vals = np.zeros(N)
        beta_vals = -45.0 * deg2rad * np.ones(N)
        control_array = np.vstack([alpha_vals, beta_vals])
        controls_guess.append(control_array)

    problem.set_initial_guess(states=states_guess, controls=controls_guess, terminal_time=2000.0)


def solve_with_fixed_mesh(
    problem,
    symbolic_vars,  # Added symbolic_vars
    example_name,
    example_num,
    bank_min,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    num_intervals = 15
    polynomial_degrees = [20] * num_intervals
    mesh_points = np.linspace(-1.0, 1.0, num_intervals + 1)
    problem.set_mesh(polynomial_degrees, mesh_points)
    deg2rad = np.pi / 180.0
    prepare_initial_guess(problem, polynomial_degrees, deg2rad)

    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min}°, 1°]"
    print(f"\nSolving Example {example_num}: {example_name} (Fixed Mesh)")
    print(f"Parameters: {bank_str}, {heat_str}")

    solution = tl.solve_fixed_mesh(
        problem,
        polynomial_degrees=polynomial_degrees,
        mesh_points=mesh_points,
        nlp_options={
            "ipopt.max_iter": 2000,
            "ipopt.mumps_pivtol": 5e-7,
            "ipopt.mumps_mem_percent": 50000,
            "ipopt.linear_solver": "mumps",
            "ipopt.constr_viol_tol": 1e-7,
            "ipopt.print_level": 5,  # Reduced print level for brevity
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.tol": 1e-8,
        },
    )

    analyze_solution(
        solution,
        symbolic_vars,  # Pass symbolic_vars
        example_name,
        example_num,
        "Fixed Mesh",
        bank_min,
        heating_limit,
        literature_J,
        literature_tf,
    )
    return solution


def analyze_solution(
    solution,
    symbolic_vars,  # Added symbolic_vars (though not strictly needed for final_theta with current objective setup)
    example_name,
    example_num,
    method,
    bank_min,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    if solution.success:
        final_time = solution.final_time
        # The problem minimizes -theta, so solution.objective is -final_theta.
        # Thus, final_theta = -solution.objective.
        final_theta_rad = -solution.objective
        final_theta_deg = final_theta_rad * 180.0 / np.pi

        J_formatted = f"{final_theta_rad:.7e}".replace("e-0", "e-").replace("e+0", "e+")
        tf_formatted = f"{final_time:.7e}".replace("e+0", "e+")
        heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
        bank_str = f"β ∈ [{bank_min}°, 1°]"

        print(f"\nExample {example_num}: {example_name} ({method})")
        print(f"Parameters: {bank_str}, {heat_str}")
        print("Optimal Results:")
        print(f"  J* = {J_formatted}  (final latitude in radians, which is -objective)")
        print(f"  t_F* = {tf_formatted}  (final time in seconds)")
        print(f"  Final latitude: {final_theta_deg:.4f}°")

        if literature_J is not None and literature_tf is not None:
            J_diff = (
                abs(final_theta_rad - literature_J) / abs(literature_J) * 100
                if abs(literature_J) > 1e-9
                else 0
            )
            tf_diff = (
                abs(final_time - literature_tf) / abs(literature_tf) * 100
                if abs(literature_tf) > 1e-9
                else 0
            )
            print("\nComparison with literature values:")
            print(f"  Literature J* = {literature_J:.7e}")
            print(f"  Literature t_F* = {literature_tf:.7e}")
            print(f"  J* difference: {J_diff:.4f}%")
            print(f"  t_F* difference: {tf_diff:.4f}%")

        if method == "Adaptive Mesh" and solution.polynomial_degrees is not None:
            print("\nFinal mesh details:")
            print(f"  Polynomial degrees: {solution.polynomial_degrees}")
            if solution.mesh_points is not None:
                print(f"  Number of mesh intervals: {len(solution.mesh_points) - 1}")
        return True
    else:
        print(f"\nExample {example_num} Solution ({method}) Failed:")
        print(f"  Reason: {solution.message}")
        return False


def plot_solution(solution, symbolic_vars):  # Added symbolic_vars
    import matplotlib.pyplot as plt

    h_scale = 1e5
    v_scale = 1e4
    rad2deg = 180.0 / np.pi

    # Use get_symbolic_trajectory with the symbolic variable objects
    time_h, h_scaled_vals = solution.get_symbolic_trajectory(symbolic_vars["h_scaled"])
    time_phi, phi_vals = solution.get_symbolic_trajectory(symbolic_vars["phi"])
    time_theta, theta_vals = solution.get_symbolic_trajectory(symbolic_vars["theta"])
    time_v, v_scaled_vals = solution.get_symbolic_trajectory(symbolic_vars["v_scaled"])
    time_gamma, gamma_vals = solution.get_symbolic_trajectory(symbolic_vars["gamma"])
    time_psi, psi_vals = solution.get_symbolic_trajectory(symbolic_vars["psi"])

    time_alpha, alpha_vals = solution.get_symbolic_trajectory(symbolic_vars["alpha"])
    time_beta, beta_vals = solution.get_symbolic_trajectory(symbolic_vars["beta"])

    h_vals = h_scaled_vals * h_scale
    v_vals = v_scaled_vals * v_scale
    phi_deg = phi_vals * rad2deg
    theta_deg = theta_vals * rad2deg
    gamma_deg = gamma_vals * rad2deg
    psi_deg = psi_vals * rad2deg
    alpha_deg = alpha_vals * rad2deg
    beta_deg = beta_vals * rad2deg

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs[0, 0].plot(time_h, h_vals / 1e5)
    axs[0, 0].set_title("Altitude (100,000 ft)")
    axs[0, 0].grid(True)
    axs[0, 1].plot(time_v, v_vals / 1e3)
    axs[0, 1].set_title("Velocity (1,000 ft/s)")
    axs[0, 1].grid(True)
    axs[1, 0].plot(time_phi, phi_deg)
    axs[1, 0].set_title("Longitude (deg)")
    axs[1, 0].grid(True)
    axs[1, 1].plot(time_gamma, gamma_deg)
    axs[1, 1].set_title("Flight Path Angle (deg)")
    axs[1, 1].grid(True)
    axs[2, 0].plot(time_theta, theta_deg)
    axs[2, 0].set_title("Latitude (deg)")
    axs[2, 0].grid(True)
    axs[2, 1].plot(time_psi, psi_deg)
    axs[2, 1].set_title("Azimuth (deg)")
    axs[2, 1].grid(True)
    plt.tight_layout()
    plt.suptitle("State Variables", y=1.02)
    plt.show()

    fig_ctrl, axs_ctrl = plt.subplots(2, 1, figsize=(10, 8))
    axs_ctrl[0].plot(time_alpha, alpha_deg)
    axs_ctrl[0].set_title("Angle of Attack (deg)")
    axs_ctrl[0].grid(True)
    axs_ctrl[1].plot(time_beta, beta_deg)
    axs_ctrl[1].set_title("Bank Angle (deg)")
    axs_ctrl[1].grid(True)
    plt.tight_layout()
    plt.suptitle("Control Variables", y=1.02)
    plt.show()

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(phi_deg, theta_deg, h_vals / 1e5)
    ax_3d.set_xlabel("Longitude (deg)")
    ax_3d.set_ylabel("Latitude (deg)")
    ax_3d.set_zlabel("Altitude (100,000 ft)")
    ax_3d.set_title("Space Shuttle Reentry Trajectory")
    plt.show()


def main():
    lit_J_ex137 = 5.9587608e-1  # Corresponds to ~34.14 degrees
    lit_tf_ex137 = 2.0085881e3

    # Create problem and get symbolic variables
    problem, symbolic_vars = create_shuttle_reentry_problem(
        heating_constraint=None, bank_angle_min=-90.0
    )

    # Solve with fixed mesh, passing symbolic_vars
    solution = solve_with_fixed_mesh(
        problem,
        symbolic_vars,  # Pass symbolic_vars
        "SHUTTLE MAXIMUM CROSSRANGE",
        "10.137 (Book Example)",
        -90,
        None,
        lit_J_ex137,
        lit_tf_ex137,
    )

    if solution.success:
        # Pass symbolic_vars to plot_solution
        plot_solution(solution, symbolic_vars)
        # Optionally, call the library's default plot
        solution.plot()


if __name__ == "__main__":
    main()
