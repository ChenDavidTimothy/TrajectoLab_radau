import casadi as ca
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Simple Bicycle Model")
phase = problem.set_phase(1)

# Variables - exact same as dynamic_obstacle_avoidance.py
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=10.0)
y = phase.state("y_position", initial=0.0, final=10.0)
theta = phase.state("heading", initial=np.pi / 4.0)
v = phase.state("velocity", initial=1.0, boundary=(0.5, 20.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))
a = phase.control("acceleration", boundary=(-3.0, 3.0))

# Dynamics - bicycle model (exact same as their example)
L = 2.5  # Wheelbase (m)
phase.dynamics(
    {
        x: v * ca.cos(theta),
        y: v * ca.sin(theta),
        theta: v * ca.tan(delta) / L,
        v: a,
    }
)

# Objective: minimize time
problem.minimize(t.final)

# Mesh and solve - using their proven configuration
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=5,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000},
)

# Results
if solution.status["success"]:
    print(f"Minimum time: {solution.status['objective']:.3f} seconds")
    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
