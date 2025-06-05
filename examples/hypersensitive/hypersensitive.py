"""
TrajectoLab Example: Hypersensitive Problem
"""

import numpy as np

import trajectolab as tl


# Problem setup
problem = tl.Problem("Hypersensitive Problem")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0, final=40)
x = phase.state("x", initial=1.5, final=1.0)
u = phase.control("u")

# Dynamics
phase.dynamics({x: -(x**3) + u})

# Objective
integrand = 0.5 * (x**2 + u**2)
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and guess
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# Solve with adaptive mesh
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=5,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 200},
)

# Results
if solution.status["success"]:
    print(f"Adaptive objective: {solution.status['objective']:.6f}")
    solution.plot()

    # Compare with fixed mesh
    phase.mesh([20, 12, 20], [-1.0, -1 / 3, 1 / 3, 1.0])

    states_guess = []
    controls_guess = []
    for N in [20, 12, 20]:
        tau = np.linspace(-1, 1, N + 1)
        x_vals = 1.5 + (1.0 - 1.5) * (tau + 1) / 2
        states_guess.append(x_vals.reshape(1, -1))
        controls_guess.append(np.zeros((1, N)))

    problem.guess(
        phase_states={1: states_guess},
        phase_controls={1: controls_guess},
        phase_initial_times={1: 0.0},
        phase_terminal_times={1: 40.0},
        phase_integrals={1: 0.1},
    )

    fixed_solution = tl.solve_fixed_mesh(
        problem, nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 200}
    )

    if fixed_solution.status["success"]:
        print(f"Fixed mesh objective: {fixed_solution.status['objective']:.6f}")
        print(
            f"Difference: {abs(solution.status['objective'] - fixed_solution.status['objective']):.2e}"
        )
    else:
        print(f"Fixed mesh failed: {fixed_solution.status['message']}")

else:
    print(f"Failed: {solution.status['message']}")
