import maptor as mtor


# Problem setup
problem = mtor.Problem("Hypersensitive Problem")
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
solution = mtor.solve_adaptive(
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

    fixed_solution = mtor.solve_fixed_mesh(
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
