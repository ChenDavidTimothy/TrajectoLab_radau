from trajectolab import FixedMesh, PHSAdaptive, Problem, RadauDirectSolver, solve


# Define the hypersensitive problem using the symbolic API
problem = Problem("Hypersensitive Problem")

# Define time variable
t = problem.time(initial=0.0, final=40.0)

# Add state with boundary conditions
x = problem.state("x", initial=1.5, final=1.0)

# Add control
u = problem.control("u")

# Define dynamics using symbolic expressions
problem.dynamics({x: -(x**3) + u})

# Define objective using symbolic expressions
integral_expr = 0.5 * (x**2 + u**2)
integral_var = problem.add_integral(integral_expr)

# Set the objective to minimize
problem.minimize(integral_var)

# Configure the solver
adaptive_solver = RadauDirectSolver(
    mesh_method=PHSAdaptive(
        initial_polynomial_degrees=[8, 8, 8],
        initial_mesh_points=[-1.0, -1 / 3, 1 / 3, 1.0],
        error_tolerance=1e-3,
        max_iterations=30,
        min_polynomial_degree=4,
        max_polynomial_degree=8,
    ),
    nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 200},
)

# Solve the problem
solution = solve(problem, adaptive_solver)

# Analyze solution
if solution.success:
    print(f"Successfully solved! Objective: {solution.objective:.6f}")
    print(f"Final mesh intervals: {solution.polynomial_degrees}")
    print(f"Mesh points: {solution.mesh_points}")

    # Get state trajectory using the symbolic variable
    t_vals, x_vals = solution.get_symbolic_trajectory(x)

    # Plot the solution
    solution.plot()
else:
    print(f"Solution failed: {solution.message}")

# Use the fixed mesh solver with the initial guess
print("Solving with fixed mesh...")
fixed_mesh_solver = RadauDirectSolver(
    mesh_method=FixedMesh(
        polynomial_degrees=[20, 8, 20],
        mesh_points=[-1.0, -1 / 3, 1 / 3, 1.0],
    ),
    nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 200},
)

fixed_solution = solve(problem, fixed_mesh_solver)

if fixed_solution.success:
    print(f"Fixed mesh solution successful! Objective: {fixed_solution.objective:.6f}")
    fixed_solution.plot()
else:
    print(f"Fixed mesh solution failed: {fixed_solution.message}")
