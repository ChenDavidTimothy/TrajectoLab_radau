"""
TrajectoLab Example: Car Race
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Create the car race problem - minimize time to complete the track
problem = tl.Problem("Car Race")

# Time is free (we want to minimize it)
t = problem.time(initial=0.0)

# States: position (0 to 1) and speed (start from rest)
pos = problem.state("position", initial=0.0, final=1.0)
speed = problem.state("speed", initial=0.0)

# Control: throttle (0 to 1)
throttle = problem.control("throttle", boundary=(0.0, 1.0))

# Dynamics: position changes with speed, speed changes with throttle minus drag
problem.dynamics({pos: speed, speed: throttle - speed})

# Speed limit varies with position: limit = 1 - sin(2*pi*pos)/2
speed_limit = 1 - ca.sin(2 * ca.pi * pos) / 2
problem.subject_to(speed <= speed_limit)

# Minimize lap time
problem.minimize(t.final)

# Set up mesh and initial guess
problem.set_mesh([8, 8, 8], np.array([-1.0, -0.3, 0.3, 1.0]))
problem.set_initial_guess(terminal_time=2.0)

# Solve with adaptive mesh
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-5,
    max_iterations=20,
    min_polynomial_degree=4,
    max_polynomial_degree=8,
    nlp_options={"ipopt.print_level": 0},
)

# Results
if solution.success:
    print(f"Optimal lap time: {solution.final_time:.3f} seconds")
    print(f"Objective: {solution.objective:.6f}")

    # Check constraint satisfaction
    pos_vals = solution["position"]
    speed_vals = solution["speed"]
    speed_limit_vals = 1 - np.sin(2.0 * np.pi * pos_vals) / 2
    max_violation = np.max(speed_vals - speed_limit_vals)
    print(f"Max speed limit violation: {max_violation:.6f}")

    solution.plot()
else:
    print(f"Failed: {solution.message}")
