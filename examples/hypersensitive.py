import numpy as np

from trajectolab import (
    Constraint,
    FixedMesh,
    InitialGuess,
    PHSAdaptive,
    Problem,
    RadauDirectSolver,
    solve,
)

# Define the hypersensitive problem
problem = Problem("Hypersensitive Problem")

# Set time bounds - fixed endpoints for hypersensitive problem
problem.set_time_bounds(t0=0.0, tf=5000.0)

# Add state with boundary conditions
problem.add_state(
    name="x", initial_constraint=Constraint(equals=1.5), final_constraint=Constraint(equals=1.0)
)

# Add control
problem.add_control(name="u")


# Define dynamics - no need to include time and params if not used
def dynamics(states, controls):
    x = states["x"]
    u = controls["u"]
    return {"x": -(x**3) + u}


problem.set_dynamics(dynamics)


# Define objective integrand - no need to include time and params if not used
def objective_integrand(states, controls):
    x = states["x"]
    u = controls["u"]
    return 0.5 * (x**2 + u**2)


problem.add_integral(objective_integrand)


# Define objective function - only need to include the integrals parameter
def objective(integrals):
    return integrals[0]  # directly return the integral value


problem.set_objective("integral", objective)

# Create an initial guess for the hypersensitive problem
initial_guess = InitialGuess(
    initial_time_variable=0.0,
    terminal_time_variable=40.0,
    states=[
        np.array([[1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7]]),
        np.array([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]]),
        np.array([[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, 1.0]]),
    ],
    controls=[
        np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, -0.1]]),
        np.array([[-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]]),
    ],
    integrals=[1.0],
)

# Configure the adaptive solver with appropriate settings
adaptive_solver = RadauDirectSolver(
    mesh_method=PHSAdaptive(
        initial_polynomial_degrees=[8, 8, 8],
        initial_mesh_points=[-1.0, -1 / 3, 1 / 3, 1.0],
        error_tolerance=4e-8,
        max_iterations=30,
        min_polynomial_degree=4,
        max_polynomial_degree=8,
        initial_guess=initial_guess,
    ),
    nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 200},
)

print("Solving with adaptive mesh refinement...")
solution = solve(problem, adaptive_solver)

if solution.success:
    print(f"Successfully solved! Objective: {solution.objective:.6f}")
    print(f"Final mesh intervals: {solution.polynomial_degrees}")
    print(f"Mesh points: {solution.mesh_points}")
    solution.plot()
else:
    print(f"Solution failed: {solution.message}")

# Use the fixed mesh solver with the initial guess
print("Solving with fixed mesh...")
fixed_mesh_solver = RadauDirectSolver(
    mesh_method=FixedMesh(
        polynomial_degrees=[20, 8, 20],
        mesh_points=[-1.0, -1 / 3, 1 / 3, 1.0],
        initial_guess=initial_guess,  # Pass the initial guess
    ),
    nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 200},
)

fixed_solution = solve(problem, fixed_mesh_solver)

if fixed_solution.success:
    print(f"Fixed mesh solution successful! Objective: {fixed_solution.objective:.6f}")
    fixed_solution.plot()
else:
    print(f"Fixed mesh solution failed: {fixed_solution.message}")
