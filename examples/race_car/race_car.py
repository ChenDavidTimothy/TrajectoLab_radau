import casadi as ca
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Car Race")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
pos = phase.state("position", initial=0.0, final=1.0)
speed = phase.state("speed", initial=0.0)
throttle = phase.control("throttle", boundary=(0.0, 1.0))

# Dynamics
phase.dynamics({pos: speed, speed: throttle - speed})

# Constraints
speed_limit = 1 - ca.sin(2 * ca.pi * pos) / 2
phase.path_constraints(speed <= speed_limit)

# Objective
problem.minimize(t.final)

# Mesh and guess
phase.mesh([8, 8, 8], np.array([-1.0, -0.3, 0.3, 1.0]))
problem.guess(phase_terminal_times={1: 2.0})

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=20,
    min_polynomial_degree=4,
    max_polynomial_degree=8,
    nlp_options={"ipopt.print_level": 0},
)

# Results
if solution.status["success"]:
    print(f"Lap time: {solution.status['objective']:.6f}")
    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
