"""
Space Shuttle Reentry Example using TrajectoLab's auto-scaling feature.

This example demonstrates the use of TrajectoLab for solving a shuttle reentry
trajectory optimization problem with automatic variable scaling.
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


# --- Physical Constants ---
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Orbital and atmospheric parameters
MU_EARTH = 0.14076539e17
R_EARTH = 20902900.0
S_REF = 2690.0
RHO0 = 0.002378
H_R = 23800.0
G0 = 32.174
WEIGHT = 203000.0
MASS = WEIGHT / G0

# Aerodynamic coefficients
A0_CL = -0.20704
A1_CL = 0.029244
B0_CD = 0.07854
B1_CD = -0.61592e-2
B2_CD = 0.621408e-3
C0_QA = 1.0672181
C1_QA = -0.19213774e-1
C2_QA = 0.21286289e-3
C3_QA = -0.10117249e-5


def generate_physical_initial_guess(bank_angle_min_deg, num_points=21):
    """Generate initial guess in physical units."""
    h0, v0 = 260000.0, 25600.0
    phi0, theta0 = 0.0 * DEG2RAD, 0.0 * DEG2RAD
    gamma0, psi0 = -1.0 * DEG2RAD, 90.0 * DEG2RAD
    hf, vf = 80000.0, 2500.0
    gammaf = -5.0 * DEG2RAD
    alpha0 = 15.0 * DEG2RAD
    beta0 = -30.0 * DEG2RAD

    # Ensure bank angle is within bounds
    if beta0 < bank_angle_min_deg * DEG2RAD:
        beta0 = bank_angle_min_deg * DEG2RAD
    if beta0 > 1.0 * DEG2RAD:  # Max bank angle for guess is 1 degree
        beta0 = 1.0 * DEG2RAD

    # Create linear trajectories for initial guess
    t_param = np.linspace(0, 1, num_points)

    # State trajectories
    states_guess = {
        "h": h0 + (hf - h0) * t_param,
        "phi": np.full(num_points, phi0),
        "theta": np.full(num_points, theta0),
        "v": v0 + (vf - v0) * t_param,
        "gamma": gamma0 + (gammaf - gamma0) * t_param,
        "psi": np.full(num_points, psi0),
    }

    # Control trajectories
    num_control_points = num_points - 1 if num_points > 1 else 1
    controls_guess = {
        "alpha": np.full(num_control_points, alpha0),
        "beta": np.full(num_control_points, beta0),
    }

    return states_guess, controls_guess


def prepare_initial_guess_for_mesh(states_dict, controls_dict, polynomial_degrees):
    """Convert flat guess dictionaries to trajectories per interval for the mesh."""
    num_intervals = len(polynomial_degrees)

    # Prepare state arrays per interval
    states_per_interval = []
    for k in range(num_intervals):
        # Number of points in this interval
        n_points = polynomial_degrees[k] + 1

        # Prepare state array
        state_vars = ["h", "phi", "theta", "v", "gamma", "psi"]
        state_array = np.zeros((len(state_vars), n_points), dtype=np.float64)

        # Sample from our flat guess trajectories
        t_indices = np.linspace(0, len(states_dict["h"]) - 1, n_points, dtype=int)
        for i, var in enumerate(state_vars):
            for j, idx in enumerate(t_indices):
                state_array[i, j] = states_dict[var][idx]

        states_per_interval.append(state_array)

    # Prepare control arrays per interval
    controls_per_interval = []
    for k in range(num_intervals):
        # Number of points in this interval
        n_points = polynomial_degrees[k]

        # Prepare control array
        control_vars = ["alpha", "beta"]
        control_array = np.zeros((len(control_vars), n_points), dtype=np.float64)

        # Sample from our flat guess trajectories
        t_indices = np.linspace(0, len(controls_dict["alpha"]) - 1, n_points, dtype=int)
        for i, var in enumerate(control_vars):
            for j, idx in enumerate(t_indices):
                control_array[i, j] = controls_dict[var][idx]

        controls_per_interval.append(control_array)

    return states_per_interval, controls_per_interval


def analyze_solution(
    solution, example_name, bank_min_deg, heating_limit=None, literature_J=None, literature_tf=None
):
    """Analyze the optimization solution and compare with literature values."""
    if solution.success:
        final_time = solution.final_time
        final_theta_actual_rad = -solution.objective  # Objective is -theta
        final_theta_actual_deg = final_theta_actual_rad * RAD2DEG

        # Format output
        J_formatted = f"{final_theta_actual_rad:.7e}".replace("e-0", "e-").replace("e+0", "e+")
        tf_formatted = f"{final_time:.7e}".replace("e+0", "e+")
        heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
        bank_str = f"β ∈ [{bank_min_deg}°, 1°]"

        print(f"\n{example_name} Results:")
        print(f"Parameters: {bank_str}, {heat_str}")
        print(f"  J* = {J_formatted}  (final latitude in radians, -objective)")
        print(f"  t_F* = {tf_formatted}  (final time in seconds)")
        print(f"  Final latitude: {final_theta_actual_deg:.4f}°")

        # Compare with literature values if provided
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

        # Print mesh details for adaptive solution
        if hasattr(solution, "polynomial_degrees") and solution.polynomial_degrees is not None:
            print("\nFinal mesh details:")
            print(f"  Polynomial degrees: {solution.polynomial_degrees}")
            if hasattr(solution, "mesh_points") and solution.mesh_points is not None:
                print(f"  Number of mesh intervals: {len(solution.mesh_points) - 1}")
                print(f"  Mesh points: {np.array2string(solution.mesh_points, precision=3)}")

        return True
    else:
        print("\nSolution Failed:")
        print(f"  Reason: {solution.message}")
        return False


def plot_solution(solution, title_suffix=""):
    """Plot the solution trajectories in physical units."""
    # Get trajectories (automatically unscaled)
    time_h, h_vals = solution.get_state_trajectory("h")
    time_phi, phi_vals = solution.get_state_trajectory("phi")
    time_theta, theta_vals = solution.get_state_trajectory("theta")
    time_v, v_vals = solution.get_state_trajectory("v")
    time_gamma, gamma_vals = solution.get_state_trajectory("gamma")
    time_psi, psi_vals = solution.get_state_trajectory("psi")
    time_alpha, alpha_vals = solution.get_control_trajectory("alpha")
    time_beta, beta_vals = solution.get_control_trajectory("beta")

    # Convert angles to degrees for plotting
    phi_deg = phi_vals * RAD2DEG
    theta_deg = theta_vals * RAD2DEG
    gamma_deg = gamma_vals * RAD2DEG
    psi_deg = psi_vals * RAD2DEG
    alpha_deg = alpha_vals * RAD2DEG
    beta_deg = beta_vals * RAD2DEG

    # Plot state variables
    main_plot_title = f"Space Shuttle Reentry {title_suffix} (Physical Units)"
    fig_states, axs_states = plt.subplots(3, 2, figsize=(12, 12))
    fig_states.suptitle(f"State Variables {title_suffix}", fontsize=16)

    axs_states[0, 0].plot(time_h, h_vals / 1e5)
    axs_states[0, 0].set_title("Altitude (10⁵ ft)")
    axs_states[0, 1].plot(time_v, v_vals / 1e3)
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

    # Plot control variables
    fig_ctrl, axs_ctrl = plt.subplots(2, 1, figsize=(10, 7))
    fig_ctrl.suptitle(f"Control Variables {title_suffix}", fontsize=16)

    axs_ctrl[0].plot(time_alpha, alpha_deg)
    axs_ctrl[0].set_title("Angle of Attack (deg)")
    axs_ctrl[1].plot(time_beta, beta_deg)
    axs_ctrl[1].set_title("Bank Angle (deg)")

    for ax in axs_ctrl:
        ax.grid(True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    # 3D trajectory plot
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(phi_deg, theta_deg, h_vals / 1e5)
    ax_3d.set_xlabel("Longitude (deg)")
    ax_3d.set_ylabel("Latitude (deg)")
    ax_3d.set_zlabel("Altitude (10⁵ ft)")
    ax_3d.set_title(main_plot_title)
    plt.show()


def solve_shuttle_reentry(use_adaptive_mesh=False):
    """Solve the shuttle reentry problem."""
    # Problem reference values for comparison
    lit_J_ex137 = 5.9587608e-1  # Literature optimal latitude
    lit_tf_ex137 = 2.0085881e3  # Literature optimal time

    # Problem parameters
    bank_min_deg = -90.0  # Minimum bank angle (degrees)
    heating_limit = None  # Set to a value like 70.0 for heating constraint

    print("\n" + "=" * 80)
    print(
        f"Space Shuttle Reentry Example {'(Adaptive Mesh)' if use_adaptive_mesh else '(Fixed Mesh)'}"
    )
    print("=" * 80)
    print("This example demonstrates TrajectoLab's automatic scaling capability.")
    print("All variables are defined in their natural physical units.")

    # Create problem with auto-scaling enabled
    problem = tl.Problem("Space Shuttle Reentry", auto_scaling=True)

    # --- Define variables with physical units ---
    # Time variable
    problem.time(initial=0.0, free_final=True)

    # State variables
    h = problem.state("h", initial=260000.0, final=80000.0, lower=0.0, upper=260000.0)
    phi = problem.state("phi", initial=0.0 * DEG2RAD)
    theta = problem.state(
        "theta", initial=0.0 * DEG2RAD, lower=-89.0 * DEG2RAD, upper=89.0 * DEG2RAD
    )
    v = problem.state("v", initial=25600.0, final=2500.0, lower=1.0, upper=25600.0)
    gamma = problem.state(
        "gamma",
        initial=-1.0 * DEG2RAD,
        final=-5.0 * DEG2RAD,
        lower=-89.0 * DEG2RAD,
        upper=89.0 * DEG2RAD,
    )
    psi = problem.state("psi", initial=90.0 * DEG2RAD)

    # Control variables
    alpha = problem.control("alpha", lower=-90.0 * DEG2RAD, upper=90.0 * DEG2RAD)
    beta = problem.control("beta", lower=bank_min_deg * DEG2RAD, upper=1.0 * DEG2RAD)

    # --- Define dynamics with physical expressions ---
    # Small constant to avoid division by zero
    eps_div = 1e-10

    # Derived physical quantities
    r_planet_dist = R_EARTH + h  # Distance from Earth center
    rho_atm = RHO0 * ca.exp(-h / H_R)  # Atmospheric density
    g_local = MU_EARTH / (r_planet_dist**2)  # Local gravitational acceleration

    # Aerodynamic forces
    alpha_deg_calc = alpha * RAD2DEG
    CL = A0_CL + A1_CL * alpha_deg_calc  # Lift coefficient
    CD = B0_CD + B1_CD * alpha_deg_calc + B2_CD * alpha_deg_calc**2  # Drag coefficient
    q_dynamic = 0.5 * rho_atm * v**2  # Dynamic pressure
    L_force = q_dynamic * CL * S_REF  # Lift force
    D_force = q_dynamic * CD * S_REF  # Drag force

    # State derivatives (completely in physical units)
    dh_dt = v * ca.sin(gamma)
    dphi_dt = (v / r_planet_dist) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps_div)
    dtheta_dt = (v / r_planet_dist) * ca.cos(gamma) * ca.cos(psi)
    dv_dt = -(D_force / MASS) - g_local * ca.sin(gamma)
    dgamma_dt = (L_force / (MASS * v + eps_div)) * ca.cos(beta) + ca.cos(gamma) * (
        (v / r_planet_dist) - (g_local / (v + eps_div))
    )
    dpsi_dt = (L_force * ca.sin(beta) / (MASS * v * ca.cos(gamma) + eps_div)) + (
        v / (r_planet_dist * (ca.cos(theta) + eps_div))
    ) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta)

    # Set dynamics
    problem.dynamics(
        {
            h: dh_dt,
            phi: dphi_dt,
            theta: dtheta_dt,
            v: dv_dt,
            gamma: dgamma_dt,
            psi: dpsi_dt,
        }
    )

    # Add heating constraint if specified
    if heating_limit is not None:
        q_r_heat = 17700 * ca.sqrt(rho_atm) * (0.0001 * v) ** 3.07
        q_a_poly_heat = (
            C0_QA + C1_QA * alpha_deg_calc + C2_QA * alpha_deg_calc**2 + C3_QA * alpha_deg_calc**3
        )
        q_heat_actual = q_a_poly_heat * q_r_heat
        problem.subject_to(q_heat_actual <= heating_limit)

    # Objective: maximize final latitude (theta)
    problem.minimize(-theta)

    # --- Configure mesh ---
    if use_adaptive_mesh:
        # Simple initial mesh for adaptive method
        initial_polynomial_degrees = [6] * 9
        mesh_points = np.linspace(-1.0, 1.0, len(initial_polynomial_degrees) + 1)
        problem.set_mesh(initial_polynomial_degrees, mesh_points)
    else:
        # Fixed mesh with higher resolution
        polynomial_degrees = [20] * 15
        mesh_points = np.linspace(-1.0, 1.0, len(polynomial_degrees) + 1)
        problem.set_mesh(polynomial_degrees, mesh_points)

    # --- Generate and set initial guess ---
    # Create initial guess in physical units
    states_guess_flat, controls_guess_flat = generate_physical_initial_guess(bank_min_deg)

    # Convert to per-interval format
    states_guess, controls_guess = prepare_initial_guess_for_mesh(
        states_guess_flat, controls_guess_flat, problem.collocation_points_per_interval
    )

    # Set initial guess (TrajectoLab will automatically scale it internally)
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        terminal_time=lit_tf_ex137,  # Use literature value as starting guess
    )

    # --- Solve the problem ---
    if use_adaptive_mesh:
        print("\nSolving with adaptive mesh refinement...")
        problem.print_scaling_summary()
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=20,
            min_polynomial_degree=4,  # Min degree for refinement
            max_polynomial_degree=10,  # Max degree for refinement
            nlp_options={
                "ipopt.max_iter": 2000,
                "ipopt.print_level": 0,
                "ipopt.tol": 1e-5,
                "ipopt.constr_viol_tol": 1e-5,
                "ipopt.nlp_scaling_method": "gradient-based",
                "ipopt.mu_strategy": "adaptive",
            },
        )
    else:
        print("\nSolving with fixed mesh...")
        problem.print_scaling_summary()
        solution = tl.solve_fixed_mesh(
            problem,
            nlp_options={
                "ipopt.max_iter": 2000,
                "ipopt.print_level": 5,
                "ipopt.tol": 1e-8,
                "ipopt.mu_strategy": "adaptive",
            },
        )

    # --- Analyze and plot results ---
    method_str = "Adaptive Mesh" if use_adaptive_mesh else "Fixed Mesh"
    if analyze_solution(
        solution,
        f"SHUTTLE MAX CROSSRANGE ({method_str})",
        bank_min_deg,
        heating_limit,
        lit_J_ex137,
        lit_tf_ex137,
    ):
        plot_solution(solution, f"({method_str})")

    return solution


if __name__ == "__main__":
    # Solve with fixed mesh
    solution_fixed = solve_shuttle_reentry(use_adaptive_mesh=False)

    # Solve with adaptive mesh
    solution_adaptive = solve_shuttle_reentry(use_adaptive_mesh=True)
