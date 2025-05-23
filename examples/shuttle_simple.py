"""
Space Shuttle Reentry - Simple Scaling Demonstration

"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Physical constants
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Shuttle parameters
MU_EARTH = 0.14076539e17
R_EARTH = 20902900.0
S_REF = 2690.0
RHO0 = 0.002378
H_R = 23800.0
MASS = 203000.0 / 32.174

# Aerodynamic coefficients
A0, A1 = -0.20704, 0.029244
B0, B1, B2 = 0.07854, -0.61592e-2, 0.621408e-3

# Scaling factors
H_SCALE = 1e5  # Altitude scaling
V_SCALE = 1e4  # Velocity scaling

print(f"Scaling: altitude ÷ {H_SCALE:.0e}, velocity ÷ {V_SCALE:.0e}")


def solve_scaled_shuttle():
    """Solve shuttle reentry with scaled variables."""

    problem = tl.Problem("Shuttle Reentry - Scaled")

    # Free final time
    problem.time(initial=0.0)

    # Scaled state variables
    h_s = problem.state("h_s", initial=2.6, final=0.8, boundary=(0, None))  # ÷1e5
    phi = problem.state("phi", initial=0.0)
    theta = problem.state("theta", initial=0.0, boundary=(-89 * DEG2RAD, 89 * DEG2RAD))
    v_s = problem.state("v_s", initial=2.56, final=0.25, boundary=(1e-4, None))  # ÷1e4
    gamma = problem.state(
        "gamma", initial=-1 * DEG2RAD, final=-5 * DEG2RAD, boundary=(-89 * DEG2RAD, 89 * DEG2RAD)
    )
    psi = problem.state("psi", initial=90 * DEG2RAD)

    # Controls
    alpha = problem.control("alpha", boundary=(-90 * DEG2RAD, 90 * DEG2RAD))  # Angle of attack
    beta = problem.control("beta", boundary=(-90 * DEG2RAD, 1 * DEG2RAD))  # Bank angle

    # Convert scaled variables to physical units
    h = h_s * H_SCALE  # Back to feet
    v = v_s * V_SCALE  # Back to ft/sec

    # Physics (using physical units)
    r = R_EARTH + h
    rho = RHO0 * ca.exp(-h / H_R)
    g = MU_EARTH / r**2

    # Aerodynamics
    alpha_deg = alpha * RAD2DEG
    CL = A0 + A1 * alpha_deg
    CD = B0 + B1 * alpha_deg + B2 * alpha_deg**2
    q = 0.5 * rho * v**2
    L = q * CL * S_REF
    D = q * CD * S_REF

    eps = 1e-10

    # Scaled dynamics: d(scaled_var)/dt = physical_rate / SCALE
    problem.dynamics(
        {
            h_s: (v * ca.sin(gamma)) / H_SCALE,  # Scaled altitude rate
            phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
            theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
            v_s: (-(D / MASS) - g * ca.sin(gamma)) / V_SCALE,  # Scaled velocity rate
            gamma: (L / (MASS * v + eps)) * ca.cos(beta) + ca.cos(gamma) * (v / r - g / (v + eps)),
            psi: (L * ca.sin(beta) / (MASS * v * ca.cos(gamma) + eps))
            + (v / (r * (ca.cos(theta) + eps))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
        }
    )

    # Objective: maximize crossrange
    problem.minimize(-theta)

    # Simple mesh
    problem.set_mesh([20] * 15, np.linspace(-1.0, 1.0, 16))

    # Simple initial guess: linear interpolation
    states_guess = []
    controls_guess = []

    for N in [20] * 15:  # For each interval
        # Linear interpolation between initial and final
        t = np.linspace(0, 1, N + 1)
        h_traj = 2.6 + (0.8 - 2.6) * t  # h_s: 2.6 → 0.8
        phi_traj = np.zeros(N + 1)  # phi: 0 → 0
        theta_traj = np.zeros(N + 1)  # theta: 0 → free
        v_traj = 2.56 + (0.25 - 2.56) * t  # v_s: 2.56 → 0.25
        gamma_traj = -1 * DEG2RAD + (-5 * DEG2RAD - (-1 * DEG2RAD)) * t  # γ: -1° → -5°
        psi_traj = 90 * DEG2RAD * np.ones(N + 1)  # ψ: 90° constant

        states_guess.append(np.vstack([h_traj, phi_traj, theta_traj, v_traj, gamma_traj, psi_traj]))
        controls_guess.append(np.vstack([np.zeros(N), -45 * DEG2RAD * np.ones(N)]))  # α=0°, β=-45°

    problem.set_initial_guess(states=states_guess, controls=controls_guess, terminal_time=2000.0)

    return problem


def main():
    """Demonstrate scaled shuttle reentry solution."""

    print("=" * 60)
    print("SCALED SPACE SHUTTLE REENTRY DEMONSTRATION")
    print("=" * 60)

    # Solve
    problem = solve_scaled_shuttle()

    print(f"\nSolving with {len(problem.collocation_points_per_interval)} intervals...")

    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 5,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-7,
            "ipopt.linear_solver": "mumps",
            "ipopt.nlp_scaling_method": "gradient-based",
        },
    )

    # Results
    if solution.success:
        crossrange_deg = -solution.objective * RAD2DEG
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Scaled formulation solved the problem.")
        print("=" * 60)
        print(f"Final time: {solution.final_time:.1f} seconds")
        print(f"Crossrange: {crossrange_deg:.2f} degrees")
        print(f"Max latitude: {crossrange_deg:.4f}°")

        # Verify scaling worked by checking final values
        time_h, h_s_vals = solution.get_trajectory("h_s")
        time_v, v_s_vals = solution.get_trajectory("v_s")
        print("\nScaled final values:")
        print(f"  h_s = {h_s_vals[-1]:.3f} (physical: {h_s_vals[-1] * H_SCALE:.0f} ft)")
        print(f"  v_s = {v_s_vals[-1]:.3f} (physical: {v_s_vals[-1] * V_SCALE:.0f} ft/s)")

        # Simple plot
        solution.plot()

    else:
        print(f"\n❌ Solution failed: {solution.message}")
        print("Try adjusting mesh resolution or solver tolerances.")


if __name__ == "__main__":
    main()
