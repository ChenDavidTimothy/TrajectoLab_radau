"""
Space Shuttle Reentry Problem with Automatic Scaling.

This implementation uses TrajectoLab's automatic scaling features
to avoid manual scaling calculations while maintaining numerical stability.
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


def create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0):
    """
    Create the Space Shuttle reentry optimal control problem with automatic scaling.

    Args:
        heating_constraint: Upper bound on aerodynamic heating (BTU/ft²/sec) or None for no constraint
        bank_angle_min: Minimum bank angle in degrees (default: -90°)

    Returns:
        A tuple containing the TrajectoLab Problem instance and a dictionary of symbolic variables.
    """
    # Create problem with automatic scaling enabled
    problem = tl.Problem("Space Shuttle Reentry Trajectory", auto_scaling=True)

    # Define constants (unchanged from shuttle.py)
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

    # Define time variable (provide bounds for better scaling)
    t = problem.time(initial=0.0, final=2500.0, free_final=True)

    # Define state variables with physical values and bounds
    # No manual scaling needed - automatic scaling will be applied
    h = problem.state(
        "h",
        initial=260000.0,
        lower=0.0,
        final=80000.0,
        upper=300000.0,
    )
    phi = problem.state("phi", initial=0.0)
    theta = problem.state(
        "theta",
        initial=0.0 * deg2rad,
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
    )
    v = problem.state(
        "v",
        initial=25600.0,
        lower=1.0,
        final=2500.0,
        upper=30000.0,
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

    symbolic_vars = {
        "t": t,
        "h": h,
        "phi": phi,
        "theta": theta,
        "v": v,
        "gamma": gamma,
        "psi": psi,
        "alpha": alpha,
        "beta": beta,
    }

    # Intermediate calculations using physical values
    # No manual scaling/unscaling needed
    eps = 1e-10
    r = Re + h
    rho = rho0 * ca.exp(-h / hr)
    g = mu / (r * r)
    alpha_deg = alpha * rad2deg
    CL = a0 + a1 * alpha_deg
    CD = b0 + b1 * alpha_deg + b2 * alpha_deg * alpha_deg
    q_dyn = 0.5 * rho * v * v
    L = q_dyn * CL * S
    D = q_dyn * CD * S
    qr = 17700 * ca.sqrt(rho) * (0.0001 * v) ** 3.07
    qa = c0 + c1 * alpha_deg + c2 * alpha_deg**2 + c3 * alpha_deg**3
    q_heat = qa * qr

    # Define dynamics with physical values
    # No manual scaling/unscaling needed in dynamics equations
    problem.dynamics(
        {
            h: v * ca.sin(gamma),
            phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
            theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
            v: -(D / mass) - g * ca.sin(gamma),
            gamma: (L / (mass * v + eps)) * ca.cos(beta)
            + ca.cos(gamma) * ((v / r) - (g / (v + eps))),
            psi: (1 / (mass * v * ca.cos(gamma) + eps)) * L * ca.sin(beta)
            + (v / (r * (ca.cos(theta) + eps))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
        }
    )

    # Add heating constraint if specified
    if heating_constraint is not None:
        problem.subject_to(q_heat <= heating_constraint)

    # Define objective (maximize final latitude)
    problem.minimize(-theta)

    return problem, symbolic_vars


def prepare_initial_guess(problem, polynomial_degrees, deg2rad, initial_terminal_time=2000.0):
    """
    Prepare initial guess in physical units.

    Args:
        problem: Problem instance
        polynomial_degrees: List of polynomial degrees for mesh intervals
        deg2rad: Degrees to radians conversion factor
        initial_terminal_time: Initial guess for terminal time
    """
    states_guess = []
    controls_guess = []

    # Initial and final values in physical units
    h0, hf = 260000.0, 80000.0
    v0, vf = 25600.0, 2500.0
    phi0, theta0 = 0.0, 0.0
    gamma0, gammaF = -1.0 * deg2rad, -5.0 * deg2rad
    psi0 = 90.0 * deg2rad

    for N in polynomial_degrees:
        # Create state guess for this interval
        t_param = np.linspace(0, 1, N + 1)
        h_vals = h0 + (hf - h0) * t_param
        phi_vals = phi0 * np.ones(N + 1)
        theta_vals = theta0 * np.ones(N + 1)
        v_vals = v0 + (vf - v0) * t_param
        gamma_vals = gamma0 + (gammaF - gamma0) * t_param
        psi_vals = psi0 * np.ones(N + 1)

        state_array = np.vstack([h_vals, phi_vals, theta_vals, v_vals, gamma_vals, psi_vals])
        states_guess.append(state_array)

        # Create control guess for this interval
        alpha_vals = np.zeros(N)
        beta_vals = -45.0 * deg2rad * np.ones(N)
        control_array = np.vstack([alpha_vals, beta_vals])
        controls_guess.append(control_array)

    # Set initial guess (in physical units - scaling happens automatically)
    problem.set_initial_guess(
        states=states_guess, controls=controls_guess, terminal_time=initial_terminal_time
    )


def solve_shuttle_reentry(
    bank_min=-90,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    """
    Solve the shuttle reentry problem with automatic scaling.

    Args:
        bank_min: Minimum bank angle in degrees
        heating_limit: Heating constraint limit or None
        literature_J: Literature optimal objective value for comparison
        literature_tf: Literature optimal time value for comparison

    Returns:
        The solution object
    """
    # Create problem with automatic scaling
    problem, symbolic_vars = create_shuttle_reentry_problem(
        heating_constraint=heating_limit,
        bank_angle_min=bank_min,
    )

    # Configure fixed mesh
    num_intervals = 15
    polynomial_degrees = [20] * num_intervals
    mesh_points = np.linspace(-1.0, 1.0, num_intervals + 1)
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Conversion factor
    deg2rad = np.pi / 180.0

    # Prepare initial guess in physical units
    prepare_initial_guess(
        problem, polynomial_degrees, deg2rad, initial_terminal_time=literature_tf or 2000.0
    )

    # Problem configuration details
    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min}°, 1°]"
    print("\nSolving Shuttle Reentry Problem with Automatic Scaling")
    print(f"Parameters: {bank_str}, {heat_str}")

    # Solver options (unchanged from shuttle.py)
    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.max_iter": 2000,
            "ipopt.mumps_pivtol": 5e-7,
            "ipopt.mumps_mem_percent": 50000,
            "ipopt.linear_solver": "mumps",
            "ipopt.constr_viol_tol": 1e-7,
            "ipopt.print_level": 0,
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.tol": 1e-8,
        },
    )

    # Analyze solution
    if solution.success:
        final_time = solution.final_time
        final_theta_rad = -solution.objective  # Objective is -theta
        final_theta_deg = final_theta_rad * 180.0 / np.pi

        J_formatted = f"{final_theta_rad:.7e}".replace("e-0", "e-").replace("e+0", "e+")
        tf_formatted = f"{final_time:.7e}".replace("e+0", "e+")

        print("\nOptimal Results:")
        print(f"  J* = {J_formatted}  (final latitude in radians, -objective)")
        print(f"  t_F* = {tf_formatted}  (final time in seconds)")
        print(f"  Final latitude: {final_theta_deg:.4f}°")

        if literature_J is not None and literature_tf is not None:
            J_abs_lit = abs(literature_J)
            tf_abs_lit = abs(literature_tf)
            J_diff = (
                abs(final_theta_rad - literature_J) / J_abs_lit * 100 if J_abs_lit > 1e-9 else 0
            )
            tf_diff = abs(final_time - literature_tf) / tf_abs_lit * 100 if tf_abs_lit > 1e-9 else 0
            print("\nComparison with literature values:")
            print(f"  Literature J* = {literature_J:.7e}")
            print(f"  Literature t_F* = {literature_tf:.7e}")
            print(f"  J* difference: {J_diff:.4f}%")
            print(f"  t_F* difference: {tf_diff:.4f}%")
    else:
        print("\nSolution Failed:")
        print(f"  Reason: {solution.message}")

    return solution, symbolic_vars


def plot_solution(solution, symbolic_vars):
    """Plot the solution trajectories."""
    rad2deg = 180.0 / np.pi

    # Get trajectories in physical units (auto-unscaled by solution object)
    time_h, h_vals = solution.get_symbolic_trajectory(symbolic_vars["h"])
    time_phi, phi_vals = solution.get_symbolic_trajectory(symbolic_vars["phi"])
    time_theta, theta_vals = solution.get_symbolic_trajectory(symbolic_vars["theta"])
    time_v, v_vals = solution.get_symbolic_trajectory(symbolic_vars["v"])
    time_gamma, gamma_vals = solution.get_symbolic_trajectory(symbolic_vars["gamma"])
    time_psi, psi_vals = solution.get_symbolic_trajectory(symbolic_vars["psi"])
    time_alpha, alpha_vals = solution.get_symbolic_trajectory(symbolic_vars["alpha"])
    time_beta, beta_vals = solution.get_symbolic_trajectory(symbolic_vars["beta"])

    # Convert angles to degrees for plotting
    phi_deg = phi_vals * rad2deg
    theta_deg = theta_vals * rad2deg
    gamma_deg = gamma_vals * rad2deg
    psi_deg = psi_vals * rad2deg
    alpha_deg = alpha_vals * rad2deg
    beta_deg = beta_vals * rad2deg

    plot_title = "Space Shuttle Reentry (Automatic Scaling)"

    # Plot states
    fig_states, axs_states = plt.subplots(3, 2, figsize=(12, 12))
    fig_states.suptitle(f"State Variables - {plot_title}", fontsize=16)

    axs_states[0, 0].plot(time_h, h_vals / 1e5)
    axs_states[0, 0].set_title("Altitude (10⁵ ft)")
    axs_states[0, 0].grid(True)

    axs_states[0, 1].plot(time_v, v_vals / 1e3)
    axs_states[0, 1].set_title("Velocity (10³ ft/s)")
    axs_states[0, 1].grid(True)

    axs_states[1, 0].plot(time_phi, phi_deg)
    axs_states[1, 0].set_title("Longitude (deg)")
    axs_states[1, 0].grid(True)

    axs_states[1, 1].plot(time_gamma, gamma_deg)
    axs_states[1, 1].set_title("Flight Path Angle (deg)")
    axs_states[1, 1].grid(True)

    axs_states[2, 0].plot(time_theta, theta_deg)
    axs_states[2, 0].set_title("Latitude (deg)")
    axs_states[2, 0].grid(True)

    axs_states[2, 1].plot(time_psi, psi_deg)
    axs_states[2, 1].set_title("Azimuth (deg)")
    axs_states[2, 1].grid(True)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    # Plot controls
    fig_ctrl, axs_ctrl = plt.subplots(2, 1, figsize=(10, 7))
    fig_ctrl.suptitle(f"Control Variables - {plot_title}", fontsize=16)

    axs_ctrl[0].plot(time_alpha, alpha_deg)
    axs_ctrl[0].set_title("Angle of Attack (deg)")
    axs_ctrl[0].grid(True)

    axs_ctrl[1].plot(time_beta, beta_deg)
    axs_ctrl[1].set_title("Bank Angle (deg)")
    axs_ctrl[1].grid(True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    # 3D trajectory plot
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(phi_deg, theta_deg, h_vals / 1e5)
    ax_3d.set_xlabel("Longitude (deg)")
    ax_3d.set_ylabel("Latitude (deg)")
    ax_3d.set_zlabel("Altitude (10⁵ ft)")
    ax_3d.set_title(plot_title)
    plt.show()


def main():
    # Literature values for comparison
    lit_J = 5.9587608e-1  # Final latitude in radians
    lit_tf = 2.0085881e3  # Final time in seconds

    # Solve the problem with automatic scaling
    solution, symbolic_vars = solve_shuttle_reentry(
        bank_min=-90,
        heating_limit=None,  # Remove heating constraint as in example 10.137
        literature_J=lit_J,
        literature_tf=lit_tf,
    )

    # Plot solution if successful
    if solution.success:
        plot_solution(solution, symbolic_vars)

        # Also display TrajectoLab's standard plot
        print("Displaying TrajectoLab standard solution plot...")
        solution.plot()


if __name__ == "__main__":
    main()
