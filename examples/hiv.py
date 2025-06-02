"""
TrajectoLab Example: HIV Immunology Model
"""

import numpy as np

import trajectolab as tl


# Problem constants
s1, s2, mu = 2.0, 1.5, 0.002
k, c, g = 2.5e-4, 0.007, 30.0
b1, b2 = 14.0, 1.0
A1, A2 = 2.5e5, 75.0
t_final = 50.0

# Create HIV immunology problem
problem = tl.Problem("HIV Immunology")

# Single phase using new API
phase = problem.set_phase(1)

# Time (fixed final time)
t = phase.time(initial=0.0, final=t_final)

# States: T-cells and viral load
T = phase.state("T_cells", initial=400.0, boundary=(0, 1200))
V = phase.state("viral_load", initial=3.0, boundary=(0.05, 5))

# Controls: treatment levels
u1 = phase.control("treatment_1", boundary=(0, 0.02))
u2 = phase.control("treatment_2", boundary=(0, 0.9))

# System dynamics (HIV immunology model)
T_dot = s1 - (s2 * V) / (b1 + V) - mu * T - k * V * T + u1 * T
V_dot = (g * (1 - u2) * V) / (b2 + V) - c * V * T

phase.dynamics({T: T_dot, V: V_dot})

# Objective: maximize T-cells minus treatment costs
integrand = T - (A1 * u1**2 + A2 * u2**2)
integral_var = phase.add_integral(integrand)
problem.minimize(-integral_var)  # Maximize

# Mesh and initial guess
phase.mesh([20, 20, 20], [-1.0, -1 / 3, 1 / 3, 1.0])

# Simple initial guess: linear interpolation
states_guess = []
controls_guess = []
for N in [20, 20, 20]:
    # States: T goes 400→600, V goes 3→0.05
    tau = np.linspace(-1, 1, N + 1)
    T_vals = 400.0 + (600.0 - 400.0) * (tau + 1) / 2
    V_vals = 3.0 + (0.05 - 3.0) * (tau + 1) / 2
    states_guess.append(np.vstack([T_vals, V_vals]))

    # Controls: use midpoint values
    controls_guess.append(
        np.vstack(
            [
                np.full(N, 0.01),  # u1 midpoint
                np.full(N, 0.45),  # u2 midpoint
            ]
        )
    )

problem.set_initial_guess(
    phase_states={1: states_guess}, phase_controls={1: controls_guess}, phase_integrals={1: 25000.0}
)

# Solve
solution = tl.solve_fixed_mesh(
    problem,
    nlp_options={
        "ipopt.print_level": 3,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-4,
    },
)

# Results
if solution.success:
    objective_value = -solution.objective  # Convert back to maximize
    print("HIV model solved successfully!")
    print(f"Objective: {objective_value:.1f}")
    print(f"Reference: 29514.4 (Error: {abs(objective_value - 29514.4) / 29514.4 * 100:.2f}%)")
    solution.plot()
else:
    print(f"Failed: {solution.message}")
