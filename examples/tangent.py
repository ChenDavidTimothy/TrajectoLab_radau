"""
TrajectoLab Example: Linear Tangent Steering Problem
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Constants
a = 100.0

# Problem setup
problem = tl.Problem("Linear Tangent Steering")

# Static parameters
p1 = problem.parameter("p1", boundary=(0.0, None))
p2 = problem.parameter("p2", boundary=(0.0, None))

# Phase 1
phase1 = problem.set_phase(1)
t1 = phase1.time(initial=0.0)
x1_1 = phase1.state("x1", initial=0.0)
x2_1 = phase1.state("x2", initial=0.0)
x3_1 = phase1.state("x3", initial=0.0)
x4_1 = phase1.state("x4", initial=0.0)

tan_u1 = p1 - p2 * t1
D1 = ca.power(1 + tan_u1**2, -0.5)
sin_u1 = D1 * tan_u1
cos_u1 = D1

phase1.dynamics(
    {
        x1_1: x3_1,
        x2_1: x4_1,
        x3_1: a * cos_u1,
        x4_1: a * sin_u1,
    }
)
phase1.mesh([4, 4], [-1.0, 0.0, 1.0])

# Phase 2
phase2 = problem.set_phase(2)
t2 = phase2.time(initial=t1.final)
x1_2 = phase2.state("x1", initial=x1_1.final)
x2_2 = phase2.state("x2", initial=x2_1.final)
x3_2 = phase2.state("x3", initial=x3_1.final)
x4_2 = phase2.state("x4", initial=x4_1.final)

tan_u2 = p1 - p2 * t2
D2 = ca.power(1 + tan_u2**2, -0.5)
sin_u2 = D2 * tan_u2
cos_u2 = D2

phase2.dynamics(
    {
        x1_2: x3_2,
        x2_2: x4_2,
        x3_2: a * cos_u2,
        x4_2: a * sin_u2,
    }
)
phase2.mesh([4, 4], [-1.0, 0.0, 1.0])

# Phase 3
phase3 = problem.set_phase(3)
t3 = phase3.time(initial=t2.final)
x1_3 = phase3.state("x1", initial=x1_2.final)
x2_3 = phase3.state("x2", initial=x2_2.final, final=5.0)
x3_3 = phase3.state("x3", initial=x3_2.final, final=45.0)
x4_3 = phase3.state("x4", initial=x4_2.final, final=0.0)

tan_u3 = p1 - p2 * t3
D3 = ca.power(1 + tan_u3**2, -0.5)
sin_u3 = D3 * tan_u3
cos_u3 = D3

phase3.dynamics(
    {
        x1_3: x3_3,
        x2_3: x4_3,
        x3_3: a * cos_u3,
        x4_3: a * sin_u3,
    }
)
phase3.mesh([4, 4], [-1.0, 0.0, 1.0])

# Objective
problem.minimize(t3.final)

# Guess
total_time = 0.56
phase_duration = total_time / 3.0

states_p1 = []
controls_p1 = []
states_p2 = []
controls_p2 = []
states_p3 = []
controls_p3 = []

for _i in range(2):
    tau = np.linspace(-1, 1, 5)
    t_phys = (tau + 1) / 2 * phase_duration

    # Phase 1
    x1_vals = t_phys * 2.0
    x2_vals = t_phys * 1.0
    x3_vals = t_phys * 10.0
    x4_vals = t_phys * 5.0
    states_p1.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
    controls_p1.append(np.zeros((0, 4)))

    # Phase 2
    x1_vals = t_phys * 3.0 + 0.5
    x2_vals = t_phys * 1.5 + 0.2
    x3_vals = t_phys * 15.0 + 2.0
    x4_vals = t_phys * 7.0 + 1.0
    states_p2.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
    controls_p2.append(np.zeros((0, 4)))

    # Phase 3
    x1_vals = t_phys * 1.0 + 2.0
    x2_vals = 4.0 + (tau + 1) / 2 * 1.0
    x3_vals = 35.0 + (tau + 1) / 2 * 10.0
    x4_vals = 1.0 - (tau + 1) / 2 * 1.0
    states_p3.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
    controls_p3.append(np.zeros((0, 4)))

problem.guess(
    phase_states={1: states_p1, 2: states_p2, 3: states_p3},
    phase_controls={1: controls_p1, 2: controls_p2, 3: controls_p3},
    phase_initial_times={1: 0.0, 2: phase_duration, 3: 2 * phase_duration},
    phase_terminal_times={1: phase_duration, 2: 2 * phase_duration, 3: total_time},
    static_parameters=np.array([1.4, 5.0]),
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=2,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 3000, "ipopt.tol": 1e-8},
)

# Results
if solution.success:
    print(f"Final time: {solution.objective:.8f}")
    print("Reference: 0.55457088")
    if solution.static_parameters is not None:
        print(f"p1: {solution.static_parameters[0]:.7f} (ref: 1.4085084)")
        print(f"p2: {solution.static_parameters[1]:.7f} (ref: 5.0796333)")
    solution.plot(show_phase_boundaries=True)
else:
    print(f"Failed: {solution.message}")
