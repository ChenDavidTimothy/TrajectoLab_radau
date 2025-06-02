# examples/hypersensitive_multiphase.py
"""
TrajectoLab Example: Hypersensitive Problem using Multiphase Framework
This should produce identical results to the original single-phase version.
"""

import numpy as np

import trajectolab as tl


# Create new problem instance for fixed mesh
problem_fixed = tl.Problem("Hypersensitive Fixed Mesh")

phase1_fixed = problem_fixed.set_phase(1)
# Same problem definition
t_fixed = phase1_fixed.time(initial=0, final=10000)
x_fixed = phase1_fixed.state("x", initial=1.5, final=1.0)
u_fixed = phase1_fixed.control("u")
phase1_fixed.dynamics({x_fixed: -(x_fixed**3) + u_fixed})
integrand_fixed = 0.5 * (x_fixed**2 + u_fixed**2)
integral_var_fixed = phase1_fixed.add_integral(integrand_fixed)
# Refined fixed mesh
phase1_fixed.set_mesh([20, 12, 20], [-1.0, -1 / 3, 1 / 3, 1.0])

problem_fixed.minimize(integral_var_fixed)
# Set detailed initial guess for fixed mesh using multiphase format

states_guess = []
controls_guess = []

for N in [20, 12, 20]:
    tau = np.linspace(-1, 1, N + 1)
    x_vals = 1.5 + (1.0 - 1.5) * (tau + 1) / 2  # Linear from 1.5 to 1.0
    states_guess.append(x_vals.reshape(1, -1))
    controls_guess.append(np.zeros((1, N)))

problem_fixed.set_initial_guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_initial_times={1: 0.0},
    phase_terminal_times={1: 10000.0},
    phase_integrals={1: 0.1},
)

fixed_solution = tl.solve_adaptive(
    problem_fixed,
    error_tolerance=7.47e-7,
    max_iterations=30,
    min_polynomial_degree=3,
    max_polynomial_degree=10,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 500,
    },
)
if fixed_solution.success:
    print(f"Fixed mesh objective: {fixed_solution.objective:.6f}")

    fixed_solution.plot(phase_id=1)
else:
    print(f"Fixed mesh failed: {fixed_solution.message}")
