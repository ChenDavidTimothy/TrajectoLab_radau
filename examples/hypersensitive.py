"""
TrajectoLab Example: Hypersensitive Problem
"""

import numpy as np

import trajectolab as tl


# Create hypersensitive problem
problem = tl.Problem("Hypersensitive Problem")

# Fixed time horizon
t = problem.time(initial=0, final=40)

# State with boundary conditions
x = problem.state("x", initial=1.5, final=1.0)

# Control (unbounded)
u = problem.control("u")

# System dynamics
problem.dynamics({x: -(x**3) + u})

# Objective: quadratic cost
integrand = 0.5 * (x**2 + u**2)
integral_var = problem.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and initial guess for adaptive solving
problem.set_mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])
# problem.set_initial_guess(integrals=0.1)

# Solve with adaptive mesh
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=5,
    max_polynomial_degree=15,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 200,
    },
)

# Results
if solution.success:
    print("Hypersensitive problem solved successfully!")
    print(f"Adaptive objective: {solution.objective:.6f}")
    solution.plot()

    # Also try fixed mesh for comparison
    print("\nSolving with fixed mesh...")

    # Refined fixed mesh
    problem.set_mesh([20, 12, 20], [-1.0, -1 / 3, 1 / 3, 1.0])

    # More detailed initial guess for fixed mesh
    states_guess = []
    controls_guess = []
    for N in [20, 12, 20]:
        tau = np.linspace(-1, 1, N + 1)
        x_vals = 1.5 + (1.0 - 1.5) * (tau + 1) / 2  # Linear from 1.5 to 1.0
        states_guess.append(x_vals.reshape(1, -1))
        controls_guess.append(np.zeros((1, N)))

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,
    )

    fixed_solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
        },
    )

    if fixed_solution.success:
        print(f"Fixed mesh objective: {fixed_solution.objective:.6f}")
        print(f"Difference: {abs(solution.objective - fixed_solution.objective):.2e}")
        fixed_solution.plot()
    else:
        print(f"Fixed mesh failed: {fixed_solution.message}")

else:
    print(f"Failed: {solution.message}")
