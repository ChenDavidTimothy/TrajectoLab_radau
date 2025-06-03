"""
TrajectoLab Example: Underwater Vehicle Control
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Constants
cx = 0.5
rx = 0.1
ux = 2.0
cz = 0.1
uz = 0.1

# Problem setup
problem = tl.Problem("Underwater Vehicle Control")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0, final=1.0)
y1 = phase.state("y1", initial=0.0, final=1.0)
y2 = phase.state("y2", initial=0.0, final=0.5)
y3 = phase.state("y3", initial=0.2, final=0.0)
y4 = phase.state(
    "y4", initial=np.pi / 2, final=np.pi / 2, boundary=(np.pi / 2 - 0.02, np.pi / 2 + 0.02)
)
y5 = phase.state("y5", initial=0.1, final=0.0)
y6 = phase.state("y6", initial=-np.pi / 4, final=0.0)
y7 = phase.state("y7", initial=1.0, final=0.0)
y8 = phase.state("y8", initial=0.0, final=0.0)
y9 = phase.state("y9", initial=0.5, final=0.0)
y10 = phase.state("y10", initial=0.1, final=0.0)

u1 = phase.control("u1", boundary=(-15.0, 15.0))
u2 = phase.control("u2", boundary=(-15.0, 15.0))
u3 = phase.control("u3", boundary=(-15.0, 15.0))
u4 = phase.control("u4", boundary=(-15.0, 15.0))

# Physics
E = ca.exp(-(((y1 - cx) / rx) ** 2))
Rx = -ux * E * (y1 - cx) * ((y3 - cz) / cz) ** 2
Rz = -uz * E * ((y3 - cz) / cz) ** 2

# Dynamics
phase.dynamics(
    {
        y1: y7 * ca.cos(y6) * ca.cos(y5) + Rx,
        y2: y7 * ca.sin(y6) * ca.cos(y5),
        y3: -y7 * ca.sin(y5) + Rz,
        y4: y8 + y9 * ca.sin(y4) * ca.tan(y5) + y10 * ca.cos(y4) * ca.tan(y5),
        y5: y9 * ca.cos(y4) - y10 * ca.sin(y4),
        y6: (y9 * ca.sin(y4) + y10 * ca.cos(y4)) / ca.cos(y5),
        y7: u1,
        y8: u2,
        y9: u3,
        y10: u4,
    }
)

# Objective
integrand = u1**2 + u2**2 + u3**2 + u4**2
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and guess
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

states_guess = []
controls_guess = []
for N in [8, 8, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear interpolation between initial and final conditions
    y1_vals = 0.0 + (1.0 - 0.0) * t_norm
    y2_vals = 0.0 + (0.5 - 0.0) * t_norm
    y3_vals = 0.2 + (0.0 - 0.2) * t_norm
    y4_vals = np.pi / 2 + (np.pi / 2 - np.pi / 2) * t_norm
    y5_vals = 0.1 + (0.0 - 0.1) * t_norm
    y6_vals = -np.pi / 4 + (0.0 - (-np.pi / 4)) * t_norm
    y7_vals = 1.0 + (0.0 - 1.0) * t_norm
    y8_vals = 0.0 + (0.0 - 0.0) * t_norm
    y9_vals = 0.5 + (0.0 - 0.5) * t_norm
    y10_vals = 0.1 + (0.0 - 0.1) * t_norm

    states_guess.append(
        np.vstack(
            [
                y1_vals,
                y2_vals,
                y3_vals,
                y4_vals,
                y5_vals,
                y6_vals,
                y7_vals,
                y8_vals,
                y9_vals,
                y10_vals,
            ]
        )
    )

    controls_guess.append(np.vstack([np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]))

problem.guess(
    phase_states={1: states_guess}, phase_controls={1: controls_guess}, phase_integrals={1: 100.0}
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=20,
    min_polynomial_degree=3,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 3000, "ipopt.tol": 1e-8},
)

# Results
if solution.success:
    print(f"Objective: {solution.objective:.6f}")
    print(
        f"Reference: 236.527851 (Error: {abs(solution.objective - 236.527851) / 236.527851 * 100:.2f}%)"
    )
    solution.plot()
else:
    print(f"Failed: {solution.message}")
