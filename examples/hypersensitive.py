import numpy as np

import trajectolab as tl


# Define the hypersensitive problem using the symbolic API
problem = tl.Problem("Hypersensitive Problem")

# Define time variable
t = problem.time(initial=0, final=40)

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

# EXPLICITLY set the initial mesh for adaptive algorithm
initial_polynomial_degrees = [8, 8, 8]
initial_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]
problem.set_mesh(initial_polynomial_degrees, initial_mesh_points)

reqs = problem.get_initial_guess_requirements()
print(reqs)  # Shows exact shapes required

# OPTIONALLY provide initial guess (not required)
# You can provide none, some, or all components
# Example: partial initial guess with just integral
problem.set_initial_guess(
    # states=None,        # Let CasADi handle it
    # controls=None,      # Let CasADi handle it
    # initial_time=None,  # Let CasADi handle it
    # terminal_time=None, # Let CasADi handle it
    integrals=0.1,  # Provide hint for integral
)

# Solve with adaptive mesh - clean and simple!
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=4,
    max_polynomial_degree=8,
    nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 200},
)

# Analyze solution
if solution.success:
    print(f"Adaptive mesh solution successful! Objective: {solution.objective:.6f}")
    # Plot the solution
    solution.plot()
else:
    print(f"Solution failed: {solution.message}")
#
# Use the fixed mesh solver
print("\nSolving with fixed mesh...")

# Configure the fixed mesh problem
fixed_polynomial_degrees = [20, 12, 20]
fixed_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]

# Create complete initial guess for fixed mesh
states_guess = []
controls_guess = []

for N in fixed_polynomial_degrees:
    # Create simple linear state guess
    tau_points = np.linspace(-1, 1, N + 1)
    x_vals = 1.5 + (1.0 - 1.5) * (tau_points + 1) / 2
    states_guess.append(x_vals.reshape(1, -1))

    # Create zero control guess
    controls_guess.append(np.zeros((1, N), dtype=np.float64))

# Then set the new initial guess
problem.set_initial_guess(
    states=states_guess,
    controls=controls_guess,
    initial_time=0.0,
    terminal_time=40.0,
    integrals=0.1,
)

# Set mesh first
problem.set_mesh(fixed_polynomial_degrees, fixed_mesh_points)


# Solve with fixed mesh - clean function call without mesh parameters
fixed_solution = tl.solve_fixed_mesh(
    problem,
    nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 200},
)

if fixed_solution.success:
    print(f"Fixed mesh solution successful! Objective: {fixed_solution.objective:.6f}")
    # Access solver statistics
    iter_count = fixed_solution.raw_solution.stats()["iter_count"]
    print(f"Solver iterations: {iter_count}")

    # Access Lagrange multipliers
    lam_g = fixed_solution.raw_solution.value(fixed_solution.opti.lam_g)
    print(f"Constraint multipliers: {lam_g}")

    fixed_solution.plot()
else:
    print(f"Fixed mesh solution failed: {fixed_solution.message}")
