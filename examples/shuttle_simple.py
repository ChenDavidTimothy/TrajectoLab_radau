"""
Space Shuttle Reentry Example - Simple User Version

This example shows how a normal user would solve the shuttle reentry problem
using TrajectoLab's auto-scaling feature. Much simpler than research-level code!
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


# Physical constants
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


def solve_shuttle_reentry():
    """Solve the shuttle reentry problem - the simple way!"""

    print("=" * 60)
    print("Space Shuttle Reentry - Maximum Crossrange")
    print("=" * 60)
    print("Using TrajectoLab's auto-scaling for easy setup!")

    # Create problem with auto-scaling enabled
    problem = tl.Problem("Space Shuttle Reentry", auto_scaling=True)

    # Define time variable (free final time - we want to find optimal duration)
    problem.time(initial=0.0, free_final=True)

    # Define state variables with physical units and constraints
    h = problem.state(
        "h", initial=260000.0, final=80000.0, lower=0.0, upper=260000.0
    )  # Altitude (ft)
    phi = problem.state("phi", initial=0.0)  # Longitude (rad)
    theta = problem.state(
        "theta", initial=0.0, lower=-89.0 * DEG2RAD, upper=89.0 * DEG2RAD
    )  # Latitude (rad)
    v = problem.state(
        "v", initial=25600.0, final=2500.0, lower=1.0, upper=25600.0
    )  # Velocity (ft/s)
    gamma = problem.state(
        "gamma",
        initial=-1.0 * DEG2RAD,
        final=-5.0 * DEG2RAD,
        lower=-89.0 * DEG2RAD,
        upper=89.0 * DEG2RAD,
    )  # Flight path angle (rad)
    psi = problem.state("psi", initial=90.0 * DEG2RAD)  # Azimuth angle (rad)

    # Define control variables
    alpha = problem.control(
        "alpha", lower=-90.0 * DEG2RAD, upper=90.0 * DEG2RAD
    )  # Angle of attack (rad)
    beta = problem.control("beta", lower=-90.0 * DEG2RAD, upper=1.0 * DEG2RAD)  # Bank angle (rad)

    # Physical calculations for dynamics
    eps_div = 1e-10  # Small constant to avoid division by zero

    # Derived quantities
    r = R_EARTH + h  # Distance from Earth center
    rho = RHO0 * ca.exp(-h / H_R)  # Atmospheric density
    g = MU_EARTH / (r**2)  # Local gravity

    # Aerodynamic forces
    alpha_deg = alpha * RAD2DEG
    CL = A0_CL + A1_CL * alpha_deg  # Lift coefficient
    CD = B0_CD + B1_CD * alpha_deg + B2_CD * alpha_deg**2  # Drag coefficient
    q = 0.5 * rho * v**2  # Dynamic pressure
    L = q * CL * S_REF  # Lift force
    D = q * CD * S_REF  # Drag force

    # System dynamics (6 differential equations)
    problem.dynamics(
        {
            h: v * ca.sin(gamma),
            phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps_div),
            theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
            v: -(D / MASS) - g * ca.sin(gamma),
            gamma: (L / (MASS * v + eps_div)) * ca.cos(beta)
            + ca.cos(gamma) * (v / r - g / (v + eps_div)),
            psi: (L * ca.sin(beta) / (MASS * v * ca.cos(gamma) + eps_div))
            + (v / (r * (ca.cos(theta) + eps_div))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
        }
    )

    # Objective: maximize final latitude (crossrange)
    problem.minimize(-theta)  # Minimize negative latitude = maximize latitude

    # Set up mesh for adaptive solver
    problem.set_mesh([6] * 8, np.linspace(-1.0, 1.0, 9))

    # Simple initial guess - just give a reasonable final time
    # No need for complex trajectory guesses!
    problem.set_initial_guess(terminal_time=2000.0)  # Roughly 33 minutes

    print("\nScaling information:")
    problem.print_scaling_summary()

    return problem


def solve_and_analyze(problem, method="adaptive"):
    """Solve the problem and analyze results."""

    print(f"\n--- Solving with {method} mesh ---")

    if method == "adaptive":
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=15,
            min_polynomial_degree=4,
            max_polynomial_degree=10,
            nlp_options={
                "ipopt.print_level": 0,  # Quiet solve
                "ipopt.max_iter": 2000,
                "ipopt.tol": 1e-8,
            },
        )
    else:  # fixed mesh
        # For fixed mesh, use a higher resolution mesh
        problem.set_mesh([12] * 12, np.linspace(-1.0, 1.0, 13))
        # Reset initial guess for new mesh
        problem.set_initial_guess(terminal_time=2000.0)

        solution = tl.solve_fixed_mesh(
            problem,
            nlp_options={
                "ipopt.print_level": 5,
                "ipopt.max_iter": 2000,
                "ipopt.tol": 1e-8,
            },
        )

    # Analyze results
    if solution.success:
        final_time = solution.final_time
        final_latitude_rad = -solution.objective  # Remember we minimized -theta
        final_latitude_deg = final_latitude_rad * RAD2DEG

        print("✅ Solution successful!")
        print(f"   Final time: {final_time:.1f} seconds ({final_time / 60:.1f} minutes)")
        print(f"   Final latitude: {final_latitude_deg:.3f} degrees")
        print(f"   Crossrange: {final_latitude_rad:.6f} radians")

        if method == "adaptive" and hasattr(solution, "polynomial_degrees"):
            print(f"   Final mesh: {len(solution.polynomial_degrees)} intervals")
            print(f"   Polynomial degrees: {solution.polynomial_degrees}")
    else:
        print(f"❌ Solution failed: {solution.message}")

    return solution


def plot_solution(solution, title_suffix=""):
    """Plot the shuttle trajectory."""
    if not solution.success:
        print("Cannot plot - solution failed")
        return

    # Get trajectories (automatically in physical units thanks to auto-scaling!)
    time_h, h_vals = solution.get_state_trajectory("h")
    time_phi, phi_vals = solution.get_state_trajectory("phi")
    time_theta, theta_vals = solution.get_state_trajectory("theta")
    time_v, v_vals = solution.get_state_trajectory("v")
    time_gamma, gamma_vals = solution.get_state_trajectory("gamma")
    time_psi, psi_vals = solution.get_state_trajectory("psi")
    time_alpha, alpha_vals = solution.get_control_trajectory("alpha")
    time_beta, beta_vals = solution.get_control_trajectory("beta")

    # Convert to convenient units
    h_km = h_vals / 3280.84  # ft to km
    v_mach = v_vals / 1125.0  # ft/s to roughly Mach number
    phi_deg = phi_vals * RAD2DEG
    theta_deg = theta_vals * RAD2DEG
    gamma_deg = gamma_vals * RAD2DEG
    psi_deg = psi_vals * RAD2DEG
    alpha_deg = alpha_vals * RAD2DEG
    beta_deg = beta_vals * RAD2DEG

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Space Shuttle Reentry {title_suffix}", fontsize=16, fontweight="bold")

    # Altitude
    axes[0, 0].plot(time_h, h_km, "b-", linewidth=2)
    axes[0, 0].set_title("Altitude")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity
    axes[0, 1].plot(time_v, v_mach, "r-", linewidth=2)
    axes[0, 1].set_title("Velocity")
    axes[0, 1].set_ylabel("Velocity (~Mach)")
    axes[0, 1].grid(True, alpha=0.3)

    # Flight path angle
    axes[0, 2].plot(time_gamma, gamma_deg, "g-", linewidth=2)
    axes[0, 2].set_title("Flight Path Angle")
    axes[0, 2].set_ylabel("Gamma (deg)")
    axes[0, 2].grid(True, alpha=0.3)

    # Latitude (the important one!)
    axes[1, 0].plot(time_theta, theta_deg, "m-", linewidth=3)
    axes[1, 0].set_title("Latitude (Crossrange)", fontweight="bold")
    axes[1, 0].set_ylabel("Latitude (deg)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    # Angle of attack
    axes[1, 1].plot(time_alpha, alpha_deg, "c-", linewidth=2)
    axes[1, 1].set_title("Angle of Attack")
    axes[1, 1].set_ylabel("Alpha (deg)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    # Bank angle
    axes[1, 2].plot(time_beta, beta_deg, "orange", linewidth=2)
    axes[1, 2].set_title("Bank Angle")
    axes[1, 2].set_ylabel("Beta (deg)")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 3D trajectory plot
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(phi_deg, theta_deg, h_km, "b-", linewidth=2)
    ax_3d.scatter(phi_deg[0], theta_deg[0], h_km[0], color="green", s=100, label="Start")
    ax_3d.scatter(phi_deg[-1], theta_deg[-1], h_km[-1], color="red", s=100, label="End")
    ax_3d.set_xlabel("Longitude (deg)")
    ax_3d.set_ylabel("Latitude (deg)")
    ax_3d.set_zlabel("Altitude (km)")
    ax_3d.set_title(f"3D Shuttle Trajectory {title_suffix}")
    ax_3d.legend()
    plt.show()


def main():
    """Main function - solve the problem both ways."""

    # Set up the problem
    problem = solve_shuttle_reentry()

    # Solve with adaptive mesh
    print("\n" + "=" * 60)
    solution_adaptive = solve_and_analyze(problem, method="adaptive")
    if solution_adaptive.success:
        plot_solution(solution_adaptive, "(Adaptive Mesh)")

    # Solve with fixed mesh
    print("\n" + "=" * 60)
    solution_fixed = solve_and_analyze(problem, method="fixed")
    if solution_fixed.success:
        plot_solution(solution_fixed, "(Fixed Mesh)")

    # Compare results if both succeeded
    if solution_adaptive.success and solution_fixed.success:
        print("\n" + "=" * 60)
        print("COMPARISON OF METHODS")
        print("=" * 60)
        print("Adaptive mesh:")
        print(f"  Final latitude: {-solution_adaptive.objective * RAD2DEG:.4f}°")
        print(f"  Final time: {solution_adaptive.final_time:.1f} s")
        print("Fixed mesh:")
        print(f"  Final latitude: {-solution_fixed.objective * RAD2DEG:.4f}°")
        print(f"  Final time: {solution_fixed.final_time:.1f} s")

        lat_diff = abs(-solution_adaptive.objective - (-solution_fixed.objective)) * RAD2DEG
        time_diff = abs(solution_adaptive.final_time - solution_fixed.final_time)
        print("Differences:")
        print(f"  Latitude: {lat_diff:.4f}°")
        print(f"  Time: {time_diff:.1f} s")


if __name__ == "__main__":
    main()
