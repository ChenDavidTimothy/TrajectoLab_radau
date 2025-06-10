import casadi as ca
import numpy as np

import maptor as mtor


def skew_symmetric(v):
    """Create skew-symmetric matrix from 3D vector for cross product operations."""
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]), ca.horzcat(v[2], 0, -v[0]), ca.horzcat(-v[1], v[0], 0)
    )


# Problem constants
omega_orb = 0.6511 * np.pi / 180.0  # rad/s
h_max = 10000.0

# Inertia matrix J (given in problem)
J = np.array(
    [
        [2.80701911616e7, 4.822509936e5, -1.71675094448e7],
        [4.822509936e5, 9.514463944e7, 6.02604448e4],
        [-1.71675094448e7, 6.02604448e4, 7.659440136e7],
    ]
)

# Initial conditions (given in problem)
omega_0 = np.array([-9.538068584896e-6, -1.136331265036e-3, 5.347280110427e-6])
r_0 = np.array([2.996368964816e-3, 1.533447776154e-1, 3.835980561392e-3])
h_0 = np.array([5000.0, 5000.0, 5000.0])

# Problem setup
problem = mtor.Problem("Space Station Attitude Control")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0, final=1800.0)
omega1 = phase.state("omega1", initial=omega_0[0])
omega2 = phase.state("omega2", initial=omega_0[1])
omega3 = phase.state("omega3", initial=omega_0[2])
r1 = phase.state("r1", initial=r_0[0])
r2 = phase.state("r2", initial=r_0[1])
r3 = phase.state("r3", initial=r_0[2])
h1 = phase.state("h1", initial=h_0[0])
h2 = phase.state("h2", initial=h_0[1])
h3 = phase.state("h3", initial=h_0[2])

u1 = phase.control("u1")
u2 = phase.control("u2")
u3 = phase.control("u3")

# State vectors
omega = ca.vertcat(omega1, omega2, omega3)
r = ca.vertcat(r1, r2, r3)
h = ca.vertcat(h1, h2, h3)
u = ca.vertcat(u1, u2, u3)

# Compute auxiliary matrices
r_cross = skew_symmetric(r)
r_norm_sq = ca.dot(r, r)
C = ca.DM.eye(3) + (2.0 / (1.0 + r_norm_sq)) * (r_cross @ r_cross - r_cross)

# Extract columns C2 and C3
C2 = C[:, 1]  # Second column (0-indexed)
C3 = C[:, 2]  # Third column (0-indexed)

# Auxiliary relations
omega_0_func = -omega_orb * C2
C3_cross = skew_symmetric(C3)
tau_gg = 3.0 * omega_orb**2 * (C3_cross @ (ca.DM(J) @ C3))

# Dynamics equations
omega_cross = skew_symmetric(omega)
J_omega_plus_h = ca.DM(J) @ omega + h
omega_cross_term = omega_cross @ J_omega_plus_h

# ω̇ = J⁻¹{τ_gg(r) - ω×[Jω + h] - u}
J_inv = ca.DM(np.linalg.inv(J))
omega_dot = J_inv @ (tau_gg - omega_cross_term - u)

# ṙ = (1/2)[rr^T + I + r×][ω - ω₀(r)]
rr_transpose = r @ r.T
bracket_matrix = rr_transpose + ca.DM.eye(3) + r_cross
r_dot = 0.5 * (bracket_matrix @ (omega - omega_0_func))

# ḣ = u
h_dot = u

# Set dynamics
phase.dynamics(
    {
        omega1: omega_dot[0],
        omega2: omega_dot[1],
        omega3: omega_dot[2],
        r1: r_dot[0],
        r2: r_dot[1],
        r3: r_dot[2],
        h1: h_dot[0],
        h2: h_dot[1],
        h3: h_dot[2],
    }
)

# Path constraints: ||h|| ≤ h_max
h_norm = ca.sqrt(h1**2 + h2**2 + h3**2)
phase.path_constraints(h_norm <= h_max)

# Final boundary conditions: equilibrium at final time
# At tf: ω̇ = 0 and ṙ = 0 (equations 56)
omega_final = ca.vertcat(omega1.final, omega2.final, omega3.final)
r_final = ca.vertcat(r1.final, r2.final, r3.final)
h_final = ca.vertcat(h1.final, h2.final, h3.final)

# Compute final equilibrium conditions
r_final_cross = skew_symmetric(r_final)
r_final_norm_sq = ca.dot(r_final, r_final)
C_final = ca.DM.eye(3) + (2.0 / (1.0 + r_final_norm_sq)) * (
    r_final_cross @ r_final_cross - r_final_cross
)
C2_final = C_final[:, 1]
C3_final = C_final[:, 2]

omega_0_final = -omega_orb * C2_final
C3_final_cross = skew_symmetric(C3_final)
tau_gg_final = 3.0 * omega_orb**2 * (C3_final_cross @ (ca.DM(J) @ C3_final))

# Final equilibrium: τ_gg - ω×[Jω + h] = u_final (so ω̇_final = 0)
omega_final_cross = skew_symmetric(omega_final)
J_omega_final_plus_h = ca.DM(J) @ omega_final + h_final
equilibrium_torque = tau_gg_final - omega_final_cross @ J_omega_final_plus_h

# Final ṙ = 0 condition
rr_transpose_final = r_final @ r_final.T
bracket_matrix_final = rr_transpose_final + ca.DM.eye(3) + r_final_cross
r_dot_final = 0.5 * (bracket_matrix_final @ (omega_final - omega_0_final))

# Event constraints: final equilibrium conditions
phase.event_constraints(r_dot_final[0] == 0, r_dot_final[1] == 0, r_dot_final[2] == 0)

# Objective: minimize (1/2) ∫ u^T u dt
integrand = 0.5 * (u1**2 + u2**2 + u3**2)
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and guess
phase.mesh([10, 10, 10], [-1.0, -1 / 3, 1 / 3, 1.0])

# Initial guess: constant values as specified in literature
# States: constant (ω̄₀, r̄₀, h̄₀), Controls: zero
states_guess = []
controls_guess = []
for N in [10, 10, 10]:
    # Constant state values throughout each interval
    omega1_vals = np.full(N + 1, omega_0[0])
    omega2_vals = np.full(N + 1, omega_0[1])
    omega3_vals = np.full(N + 1, omega_0[2])
    r1_vals = np.full(N + 1, r_0[0])
    r2_vals = np.full(N + 1, r_0[1])
    r3_vals = np.full(N + 1, r_0[2])
    h1_vals = np.full(N + 1, h_0[0])
    h2_vals = np.full(N + 1, h_0[1])
    h3_vals = np.full(N + 1, h_0[2])

    state_array = np.vstack(
        [
            omega1_vals,
            omega2_vals,
            omega3_vals,
            r1_vals,
            r2_vals,
            r3_vals,
            h1_vals,
            h2_vals,
            h3_vals,
        ]
    )
    states_guess.append(state_array)

    # Zero control inputs throughout each interval
    u1_vals = np.zeros(N)
    u2_vals = np.zeros(N)
    u3_vals = np.zeros(N)

    control_array = np.vstack([u1_vals, u2_vals, u3_vals])
    controls_guess.append(control_array)

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_integrals={1: 0.0},  # Consistent with zero control guess
)

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=25,
    min_polynomial_degree=4,
    max_polynomial_degree=10,
    nlp_options={
        "ipopt.max_iter": 20000,
        "ipopt.mumps_pivtol": 5e-7,
        "ipopt.mumps_mem_percent": 50000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-7,
        "ipopt.print_level": 5,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.check_derivatives_for_naninf": "yes",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-8,
    },
)

# Results
if solution.status["success"]:
    print(f"Objective: {solution.status['objective']:.12e}")
    print(
        f"Reference: 3.586751e-06 (Error: {abs(solution.status['objective'] - 3.586751e-6) / 3.586751e-6 * 100:.3f}%)"
    )

    # Final state verification
    omega_final_vals = [solution[(1, f"omega{i}")][-1] for i in range(1, 4)]
    r_final_vals = [solution[(1, f"r{i}")][-1] for i in range(1, 4)]
    h_final_vals = [solution[(1, f"h{i}")][-1] for i in range(1, 4)]

    print("Final states:")
    print(
        f"  ω_final: [{omega_final_vals[0]:.6e}, {omega_final_vals[1]:.6e}, {omega_final_vals[2]:.6e}]"
    )
    print(f"  r_final: [{r_final_vals[0]:.6e}, {r_final_vals[1]:.6e}, {r_final_vals[2]:.6e}]")
    print(f"  h_final: [{h_final_vals[0]:.1f}, {h_final_vals[1]:.1f}, {h_final_vals[2]:.1f}]")

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
