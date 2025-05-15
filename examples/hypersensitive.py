import numpy as np
import matplotlib.pyplot as plt
from trajectolab import Problem, Constraint, solve, PHSAdaptive, RadauDirectSolver

# Define the hypersensitive problem
problem = Problem("Hypersensitive Problem")

# Set time bounds
problem.set_time_bounds(t0=0.0, tf=40.0)

# Add state with boundary conditions
problem.add_state(
    name="x",
    initial_constraint=Constraint(equals=1.5),
    final_constraint=Constraint(equals=1.0)
)

# Add control
problem.add_control(name="u")

# Define dynamics
def dynamics(states, controls, time, params):
    x = states["x"]
    u = controls["u"]
    return {"x": -x**3 + u}

problem.set_dynamics(dynamics)

# Define objective integrand
def objective_integrand(states, controls, time, params):
    x = states["x"]
    u = controls["u"]
    return 0.5 * (x**2 + u**2)

problem.add_integral(objective_integrand)

# Define objective function
def objective(initial_time, final_time, initial_states, final_states, integrals, params):
    return integrals[0]

problem.set_objective("mayer", objective)

# Solve with default settings (adaptive mesh refinement)
print("Solving with adaptive mesh refinement...")
solution = solve(problem)

# Check solution and plot
if solution.success:
    print(f"Successfully solved! Objective: {solution.objective:.6f}")
    print(f"Final mesh intervals: {solution.polynomial_degrees}")
    print(f"Mesh points: {solution.mesh_points}")
    
    # Plot solution with interval coloring
    solution.plot_by_interval()
else:
    print(f"Solution failed: {solution.message}")

# Alternatively, solve with a fixed mesh
print("Solving with fixed mesh...")
fixed_mesh_solver = RadauDirectSolver(
    mesh_method=FixedMesh(
        polynomial_degrees=[8, 8, 8],
        mesh_points=[-1.0, -1/3, 1/3, 1.0]
    )
)

fixed_solution = solve(problem, fixed_mesh_solver)

if fixed_solution.success:
    print(f"Fixed mesh solution successful! Objective: {fixed_solution.objective:.6f}")
    fixed_solution.plot()
else:
    print(f"Fixed mesh solution failed: {fixed_solution.message}") 