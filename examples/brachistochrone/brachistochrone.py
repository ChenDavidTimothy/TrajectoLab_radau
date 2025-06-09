import casadi as ca
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Brachistochrone Problem")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
x = phase.state("x", initial=0.0, final=10.0)
y = phase.state("y", initial=10.0, final=5.0)
v = phase.state("v", initial=0.0)
u = phase.control("u")

# Dynamics
g0 = 9.81
phase.dynamics({x: v * ca.sin(u), y: -v * ca.cos(u), v: g0 * ca.cos(u)})

# Objective
problem.minimize(t.final)

# Mesh and guess
phase.mesh([6, 6], [-1.0, 0.0, 1.0])

states_guess = [
    # Interval 1: 7 state points
    np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0],  # x
            [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0],  # y
            [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0],  # v
        ]
    ),
    # Interval 2: 7 state points
    np.array(
        [
            [5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0],  # x
            [7.0, 6.5, 6.0, 5.8, 5.5, 5.2, 5.0],  # y
            [4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5],  # v
        ]
    ),
]

controls_guess = [
    # Interval 1: 6 control points
    np.array([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]),  # u (angles in radians)
    # Interval 2: 6 control points
    np.array([[0.3, 0.2, 0.1, 0.0, -0.1, -0.2]]),  # u
]

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 1.0},
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
