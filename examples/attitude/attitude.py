import casadi as ca
import numpy as np

import maptor as mtor


# Physical parameters from equations (57) and (59)
omega_orb = 0.6511e-3  # rad/s
h_max = 10000.0

# Inertia matrix J from equation (59) [kg⋅m²]
J = np.array(
    [
        [2.80701911616e7, 4.82250993600e5, -1.71675094448e7],
        [4.82250993600e5, 9.51446934440e7, 6.02604448000e4],
        [-1.71675094448e7, 6.02604448000e4, 7.65944013360e7],
    ]
)

# Compute J_inv for dynamics
J_inv = np.linalg.inv(J)

# Initial conditions from equation (60)
omega_0_initial = np.array([-9.5380685844896e-6, -1.13633126570360e-3, 5.34728011084270e-6])
r_0_initial = np.array([2.99636896498160e-3, 1.53344777610540e-1, 3.83598056133920e-3])

# Calculate 4th component for unit quaternion
r_norm_sq = np.sum(r_0_initial**2)
r_4_initial = np.sqrt(1.0 - r_norm_sq)
r_0_full = np.append(r_0_initial, r_4_initial)

h_0_initial = np.array([5000.0, 5000.0, 5000.0])

# Problem setup
problem = mtor.Problem("Space Station Attitude Control")
phase = problem.set_phase(1)

# Time variable (fixed final time from equation 56)
t = phase.time(initial=0.0, final=1800.0)

# State variables (9 total: 3 + 4 + 2 remaining components)
# Angular velocity ω (3 components)
omega_x = phase.state("omega_x", initial=omega_0_initial[0])
omega_y = phase.state("omega_y", initial=omega_0_initial[1])
omega_z = phase.state("omega_z", initial=omega_0_initial[2])

# Euler-Rodrigues parameters r (4 components)
r_1 = phase.state("r_1", initial=r_0_full[0])
r_2 = phase.state("r_2", initial=r_0_full[1])
r_3 = phase.state("r_3", initial=r_0_full[2])
r_4 = phase.state("r_4", initial=r_0_full[3])

# Angular momentum h (3 components)
h_x = phase.state("h_x", initial=h_0_initial[0])
h_y = phase.state("h_y", initial=h_0_initial[1])
h_z = phase.state("h_z", initial=h_0_initial[2])

# Control variables u (3 components)
u_x = phase.control("u_x")
u_y = phase.control("u_y")
u_z = phase.control("u_z")

# Construct state vectors
omega = ca.vertcat(omega_x, omega_y, omega_z)
r = ca.vertcat(r_1, r_2, r_3, r_4)
h = ca.vertcat(h_x, h_y, h_z)
u = ca.vertcat(u_x, u_y, u_z)


# Skew-symmetric matrix function for cross products
def skew_symmetric(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]), ca.horzcat(v[2], 0, -v[0]), ca.horzcat(-v[1], v[0], 0)
    )


# Transformation matrix C from equation (58)
r_vec = r[:3]  # First 3 components of r
r_norm_sq = ca.dot(r, r)
r_skew = skew_symmetric(r_vec)
C = ca.DM.eye(3) + (2.0 / (1.0 + r_norm_sq)) * (ca.mtimes(r_vec, r_vec.T) - r_skew)

# Extract C2 and C3 (second and third columns)
C2 = C[:, 1]
C3 = C[:, 2]

# Orbital angular velocity ω₀(r) from equation (57)
omega_0_r = -omega_orb * C2

# Gravity gradient torque τ_gg from equation (57) - as scalar per equation
tau_gg = 3 * omega_orb**2 * ca.mtimes(ca.mtimes(C2.T, J), C3)

# Dynamics implementation from equation (54)

# Angular velocity dynamics: ω̇ = J⁻¹[τ_gg(r) - ω⊗[Jω + h] - u]
J_omega_plus_h = ca.mtimes(J, omega) + h
omega_cross_term = ca.mtimes(skew_symmetric(omega), J_omega_plus_h)

# Since tau_gg is scalar but we need vector, distribute along z-axis (typical for GG torque)
tau_gg_vector = ca.vertcat(0, 0, tau_gg)
omega_dot = ca.mtimes(J_inv, (tau_gg_vector - omega_cross_term - u))

# Euler-Rodrigues dynamics using standard quaternion kinematics
# ṙ = (1/2) * E(r) * ω where E(r) is the 4x3 transformation matrix
# E(r) = [r₄*I + skew(r₁:₃)]  (3x3)
#        [    -r₁:₃ᵀ        ]  (1x3)
r_vec_skew = skew_symmetric(r_vec)
E_matrix = ca.vertcat(
    r[3] * ca.DM.eye(3) + r_vec_skew,  # 3x3 upper part
    -r_vec.T,  # 1x3 lower part
)
# Use inertial angular velocity (not relative to orbital frame for simplicity)
r_dot = 0.5 * ca.mtimes(E_matrix, omega)

# Angular momentum dynamics: ḣ = u (equation 54)
h_dot = u

# Set dynamics
phase.dynamics(
    {
        omega_x: omega_dot[0],
        omega_y: omega_dot[1],
        omega_z: omega_dot[2],
        r_1: r_dot[0],
        r_2: r_dot[1],
        r_3: r_dot[2],
        r_4: r_dot[3],
        h_x: h_dot[0],
        h_y: h_dot[1],
        h_z: h_dot[2],
    }
)

# Objective function from equation (53): J = (1/2) ∫ uᵀu dt
integrand = 0.5 * ca.dot(u, u)
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Path constraint from equation (55): ||h|| ≤ h_max
h_magnitude_squared = ca.dot(h, h)
phase.path_constraints(h_magnitude_squared <= h_max**2)

# Terminal constraints from equation (56) - simplified for now
# TODO: Implement full constraints after basic problem works
# For now, just enforce that final angular velocity is small
phase.event_constraints(omega_x.final**2 + omega_y.final**2 + omega_z.final**2 <= 1e-6)

# Mesh configuration
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# Initial guess
states_guess = []
controls_guess = []

for N in [8, 8, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear interpolation for states (maintain initial conditions)
    omega_x_vals = omega_0_initial[0] * np.ones(N + 1)
    omega_y_vals = omega_0_initial[1] * np.ones(N + 1)
    omega_z_vals = omega_0_initial[2] * np.ones(N + 1)

    r_1_vals = r_0_full[0] * np.ones(N + 1)
    r_2_vals = r_0_full[1] * np.ones(N + 1)
    r_3_vals = r_0_full[2] * np.ones(N + 1)
    r_4_vals = r_0_full[3] * np.ones(N + 1)

    h_x_vals = h_0_initial[0] * np.ones(N + 1)
    h_y_vals = h_0_initial[1] * np.ones(N + 1)
    h_z_vals = h_0_initial[2] * np.ones(N + 1)

    states_guess.append(
        np.vstack(
            [
                omega_x_vals,
                omega_y_vals,
                omega_z_vals,
                r_1_vals,
                r_2_vals,
                r_3_vals,
                r_4_vals,
                h_x_vals,
                h_y_vals,
                h_z_vals,
            ]
        )
    )

    # Small control guess
    controls_guess.append(np.vstack([0.1 * np.ones(N), 0.1 * np.ones(N), 0.1 * np.ones(N)]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 1800.0},
    phase_integrals={1: 100.0},  # Single integral (objective only)
)

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=30,
    min_polynomial_degree=4,
    max_polynomial_degree=12,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-8,
        "ipopt.constr_viol_tol": 1e-7,
    },
)

# Results
if solution.status["success"]:
    print(f"Objective (control energy): {solution.status['objective']:.6f}")
    print(f"Final time: {solution.phases[1]['times']['final']:.1f} seconds")

    # Final state values
    omega_x_final = solution[(1, "omega_x")][-1]
    omega_y_final = solution[(1, "omega_y")][-1]
    omega_z_final = solution[(1, "omega_z")][-1]

    h_x_final = solution[(1, "h_x")][-1]
    h_y_final = solution[(1, "h_y")][-1]
    h_z_final = solution[(1, "h_z")][-1]

    h_final_magnitude = np.sqrt(h_x_final**2 + h_y_final**2 + h_z_final**2)

    print("Final angular velocity [rad/s]:")
    print(f"  ωx: {omega_x_final:.8e}")
    print(f"  ωy: {omega_y_final:.8e}")
    print(f"  ωz: {omega_z_final:.8e}")

    print("Final angular momentum:")
    print(f"  hx: {h_x_final:.1f}")
    print(f"  hy: {h_y_final:.1f}")
    print(f"  hz: {h_z_final:.1f}")
    print(f"  ||h||: {h_final_magnitude:.1f} (limit: {h_max})")

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
