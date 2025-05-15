import numpy as np
import matplotlib.pyplot as plt
from trajectolab import Problem, Constraint, solve, PHSAdaptive, RadauDirectSolver, FixedMesh

# Define the hypersensitive problem
problem = Problem("Hypersensitive Problem")

# Set time bounds - fixed endpoints for hypersensitive problem
problem.set_time_bounds(t0=0.0, tf=40.0)

# Add state with boundary conditions
problem.add_state(
    name="x",
    initial_constraint=Constraint(equals=1.5),
    final_constraint=Constraint(equals=1.0)
)

# Add control
problem.add_control(name="u")

# Define dynamics - simple with symbolic expressions
def dynamics(states, controls, time, params):
    x = states["x"]
    u = controls["u"]
    return {"x": -x**3 + u}

problem.set_dynamics(dynamics)

# Define objective integrand
def objective_integrand(states, controls, time, params):
    x = states["x"]
    u = controls["u"]
    return 0.5 * (x**2 + u**2)  # Don't use float() here!

problem.add_integral(objective_integrand)

# Define objective function - simple return of the integral
def objective(initial_time, final_time, initial_states, final_states, integrals, params):
    return integrals[0]  # directly return the integral value

problem.set_objective("integral", objective)

# Configure the adaptive solver with appropriate settings
adaptive_solver = RadauDirectSolver(
    mesh_method=PHSAdaptive(
        error_tolerance=1e-3,
        max_iterations=30,
        min_polynomial_degree=4,
        max_polynomial_degree=8
    ),
    nlp_options={'ipopt.print_level': 0, 'print_time': 0}  # Less verbose output
)

# Initialize with multiple intervals
adaptive_solver.mesh_method.initial_mesh = {
    'polynomial_degrees': [8, 8, 8],
    'mesh_points': [-1.0, -1/3, 1/3, 1.0]
}

print("Solving with adaptive mesh refinement...")
solution = solve(problem, adaptive_solver)

if solution.success:
    print(f"Successfully solved! Objective: {solution.objective:.6f}")
    print(f"Final mesh intervals: {solution.polynomial_degrees}")
    print(f"Mesh points: {solution.mesh_points}")
    solution.plot_by_interval()
else:
    print(f"Solution failed: {solution.message}")

# Fixed mesh solver with same settings
print("Solving with fixed mesh...")
fixed_mesh_solver = RadauDirectSolver(
    mesh_method=FixedMesh(
        polynomial_degrees=[8, 8, 8],
        mesh_points=[-1.0, -1/3, 1/3, 1.0]
    ),
    nlp_options={'ipopt.print_level': 0, 'print_time': 0}
)

fixed_solution = solve(problem, fixed_mesh_solver)

if fixed_solution.success:
    print(f"Fixed mesh solution successful! Objective: {fixed_solution.objective:.6f}")
    fixed_solution.plot()
else:
    print(f"Fixed mesh solution failed: {fixed_solution.message}")