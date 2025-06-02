"""
TrajectoLab Example: Asymmetric Rigid Body Reorientation
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Constants
Ix, Iy, Iz = 10.0, 20.0, 30.0
phi_rad = 150.0 * np.pi / 180.0

# Problem setup
problem = tl.Problem("Asymmetric Rigid Body Reorientation")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0, final=(0.01, 50.0))
q1 = phase.state("q1", initial=0.0, final=ca.sin(phi_rad / 2), boundary=(-1.1, 1.1))
q2 = phase.state("q2", initial=0.0, final=0.0, boundary=(-1.1, 1.1))
q3 = phase.state("q3", initial=0.0, final=0.0, boundary=(-1.1, 1.1))
omega1 = phase.state("omega1", initial=0.0, final=0.0)
omega2 = phase.state("omega2", initial=0.0, final=0.0)
omega3 = phase.state("omega3", initial=0.0, final=0.0)
u1 = phase.control("torque_x", boundary=(-50.0, 50.0))
u2 = phase.control("torque_y", boundary=(-50.0, 50.0))
u3 = phase.control("torque_z", boundary=(-50.0, 50.0))

# Quaternion algebra
q4 = ca.sqrt(1 - q1**2 - q2**2 - q3**2)
phase.subject_to(q1**2 + q2**2 + q3**2 <= 1.0)

# Dynamics
q1_dot = 0.5 * (omega1 * q4 - omega2 * q3 + omega3 * q2)
q2_dot = 0.5 * (omega1 * q3 + omega2 * q4 - omega3 * q1)
q3_dot = 0.5 * (-omega1 * q2 + omega2 * q1 + omega3 * q4)
omega1_dot = u1 / Ix - ((Iz - Iy) / Ix) * omega2 * omega3
omega2_dot = u2 / Iy - ((Ix - Iz) / Iy) * omega1 * omega3
omega3_dot = u3 / Iz - ((Iy - Ix) / Iz) * omega1 * omega2

phase.dynamics(
    {
        q1: q1_dot,
        q2: q2_dot,
        q3: q3_dot,
        omega1: omega1_dot,
        omega2: omega2_dot,
        omega3: omega3_dot,
    }
)

# Objective
problem.minimize(t.final)

# Mesh and guess
phase.mesh([8, 8, 8], [-1.0, -0.3, 0.3, 1.0])

q1_final = np.sin(phi_rad / 2)
estimated_final_time = 30.0

states_guess = []
controls_guess = []
for N in [8, 8, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    q1_vals = q1_final * t_norm
    q2_vals = np.zeros(N + 1)
    q3_vals = np.zeros(N + 1)
    omega_profile = np.sin(np.pi * t_norm)
    omega1_vals = 2.0 * omega_profile
    omega2_vals = 1.0 * omega_profile
    omega3_vals = 0.5 * omega_profile
    omega1_vals[-1] = omega2_vals[-1] = omega3_vals[-1] = 0.0

    states_guess.append(
        np.array([q1_vals, q2_vals, q3_vals, omega1_vals, omega2_vals, omega3_vals])
    )

    control_profile = np.cos(np.pi * np.linspace(0, 1, N))
    u1_vals = 10.0 * control_profile
    u2_vals = 5.0 * control_profile
    u3_vals = 2.0 * control_profile

    controls_guess.append(np.array([u1_vals, u2_vals, u3_vals]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: estimated_final_time},
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=20,
    min_polynomial_degree=3,
    max_polynomial_degree=12,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 3000, "ipopt.tol": 1e-8},
)

# Results
if solution.success:
    print(f"Final time: {solution.objective:.7f}")
    print(
        f"Reference: 28.6304077 (Error: {abs(solution.objective - 28.6304077) / 28.6304077 * 100:.3f}%)"
    )
    solution.plot()
else:
    print(f"Failed: {solution.message}")
