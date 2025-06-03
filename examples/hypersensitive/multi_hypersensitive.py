"""
TrajectoLab Example: Multi-Phase Hypersensitive Problem
"""

import numpy as np

import trajectolab as tl


# Problem setup
problem = tl.Problem("Multi-Phase Hypersensitive")

# Phase 1: [0, 5000]
phase1 = problem.set_phase(1)
t1 = phase1.time(initial=0.0, final=5000.0)
x1 = phase1.state("x", initial=1.5)
u1 = phase1.control("u")
phase1.dynamics({x1: -(x1**3) + u1})
integral1 = phase1.add_integral(0.5 * (x1**2 + u1**2))
phase1.mesh([4, 4, 4], [-1.0, -1 / 3, 1 / 3, 1.0])

# Phase 2: [5000, 10000]
phase2 = problem.set_phase(2)
t2 = phase2.time(initial=t1.final, final=10000.0)
x2 = phase2.state("x", initial=x1.final, final=1.0)
u2 = phase2.control("u")
phase2.dynamics({x2: -(x2**3) + u2})
integral2 = phase2.add_integral(0.5 * (x2**2 + u2**2))
phase2.mesh([4, 4, 4], [-1.0, -1 / 3, 1 / 3, 1.0])

# Objective
problem.minimize(integral1 + integral2)

# Guess
states_guess_p1 = []
controls_guess_p1 = []
states_guess_p2 = []
controls_guess_p2 = []

for N in [4, 4, 4]:
    tau = np.linspace(-1, 1, N + 1)
    x_vals = 1.5 + (0.5 - 1.5) * (tau + 1) / 2
    states_guess_p1.append(x_vals.reshape(1, -1))
    controls_guess_p1.append(np.zeros((1, N)))

    x_vals = 0.5 + (1.0 - 0.5) * (tau + 1) / 2
    states_guess_p2.append(x_vals.reshape(1, -1))
    controls_guess_p2.append(np.zeros((1, N)))

problem.guess(
    phase_states={1: states_guess_p1, 2: states_guess_p2},
    phase_controls={1: controls_guess_p1, 2: controls_guess_p2},
    phase_initial_times={1: 0.0, 2: 5000.0},
    phase_terminal_times={1: 5000.0, 2: 10000.0},
    phase_integrals={1: 0.1, 2: 0.1},
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=7.47e-7,
    max_iterations=30,
    min_polynomial_degree=3,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500, "ipopt.tol": 1e-8},
)

# Results
if solution.success:
    print(f"Objective: {solution.objective:.8f}")
    solution.plot(show_phase_boundaries=True)
else:
    print(f"Failed: {solution.message}")
