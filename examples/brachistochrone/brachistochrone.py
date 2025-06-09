import casadi as ca
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Brachistochrone Problem")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
x = phase.state("x", initial=0.0, final=1.0, boundary=(0, 10))
y = phase.state("y", initial=0.0, boundary=(0, 10))
v = phase.state("v", initial=0.0, boundary=(0, 10))
u = phase.control("u", boundary=(0, np.pi / 2))

# Dynamics
g0 = 32.174  # ft/sec^2
phase.dynamics({x: v * ca.cos(u), y: v * ca.sin(u), v: g0 * ca.sin(u)})

# Objective
problem.minimize(t.final)

# Mesh and guess
phase.mesh([5, 5], [-1.0, 0.0, 1.0])

states_guess = []
controls_guess = []

for N in [5, 5]:
    # Simple linear state trajectories
    x_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    # y_vals = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.2])
    v_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5])
    states_guess.append(np.array([x_vals, v_vals]))

    # Simple constant control
    u_vals = np.ones(N) * 0.5
    controls_guess.append(np.array([u_vals]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_initial_times={1: 0.0},
    phase_terminal_times={1: 0.4},
)

# Solve with adaptive mesh
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=30,
    min_polynomial_degree=3,
    max_polynomial_degree=12,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500},
)

# Results
if solution.status["success"]:
    print(f"Adaptive objective: {solution.status['objective']:.9f}")
    print("Literature reference: 0.312480130")
    print(f"Difference: {abs(solution.status['objective'] - 0.312480130):.2e}")
    solution.plot()

    # Compare with fixed mesh
    phase.mesh([6, 6], [-1.0, 0.0, 1.0])

    states_guess = []
    controls_guess = []
    for N in [6, 6]:
        x_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0])
        y_vals = np.array([0.0, 0.05, 0.2, 0.45, 0.8, 1.2, 1.4])
        v_vals = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.2])
        states_guess.append(np.array([x_vals, y_vals, v_vals]))

        u_vals = np.ones(N) * 0.6
        controls_guess.append(np.array([u_vals]))

    problem.guess(
        phase_states={1: states_guess},
        phase_controls={1: controls_guess},
        phase_initial_times={1: 0.0},
        phase_terminal_times={1: 0.4},
    )

    fixed_solution = mtor.solve_fixed_mesh(
        problem, nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500}
    )

    if fixed_solution.status["success"]:
        print(f"Fixed mesh objective: {fixed_solution.status['objective']:.9f}")
        print(
            f"Difference: {abs(solution.status['objective'] - fixed_solution.status['objective']):.2e}"
        )
    else:
        print(f"Fixed mesh failed: {fixed_solution.status['message']}")

else:
    print(f"Failed: {solution.status['message']}")
