"""
TrajectoLab Example: Car Race
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Create the car race problem - minimize time to complete the track
problem = tl.Problem("Car Race")

# Single phase using new API
phase = problem.add_phase(1)

# Time is free (we want to minimize it)
t = phase.time(initial=0.0)

# States: position (0 to 1) and speed (start from rest)
pos = phase.state("position", initial=0.0, final=1.0)
speed = phase.state("speed", initial=0.0)

# Control: throttle (0 to 1)
throttle = phase.control("throttle", boundary=(0.0, 1.0))

# Dynamics: position changes with speed, speed changes with throttle minus drag
phase.dynamics({pos: speed, speed: throttle - speed})

# Speed limit varies with position: limit = 1 - sin(2*pi*pos)/2
speed_limit = 1 - ca.sin(2 * ca.pi * pos) / 2
phase.subject_to(speed <= speed_limit)

# Minimize lap time
problem.minimize(t.final)

# Set up mesh and initial guess
phase.set_mesh([8, 8, 8], np.array([-1.0, -0.3, 0.3, 1.0]))
problem.set_initial_guess(phase_terminal_times={1: 2.0})

# Solve with adaptive mesh
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=20,
    min_polynomial_degree=4,
    max_polynomial_degree=8,
    nlp_options={"ipopt.print_level": 0},
)

# Results
if solution.success:
    solution.plot()
else:
    print(f"Failed: {solution.message}")
