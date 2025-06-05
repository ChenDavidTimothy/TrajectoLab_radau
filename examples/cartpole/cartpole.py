import casadi as ca
import numpy as np

import trajectolab as tl


# Problem setup
problem = tl.Problem("Cart-Pole Swing-Up")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
x = phase.state("x", initial=0.0, final=0.0)  # Cart position
theta = phase.state("theta", initial=0.0, final=np.pi)  # Pendulum angle (0=down, π=up)
x_dot = phase.state("x_dot", initial=0.0, final=0.0)  # Cart velocity
theta_dot = phase.state("theta_dot", initial=0.0, final=0.0)  # Pendulum angular velocity
f_x = phase.control("f_x", boundary=(-20.0, 20.0))  # Horizontal force on cart

# Dynamics (from simplified equations 18-19)
# 2ẍ + θ̈ cos θ - θ̇² sin θ = f_x
# ẍ cos θ + θ̈ + sin θ = 0
# Solving for accelerations:
denominator = 2 - ca.cos(theta)**2
x_ddot = (f_x + ca.sin(theta) * ca.cos(theta) + theta_dot**2 * ca.sin(theta)) / denominator
theta_ddot = (-ca.cos(theta) * f_x - theta_dot**2 * ca.sin(theta) * ca.cos(theta) - 2 * ca.sin(theta)) / denominator

phase.dynamics({
    x: x_dot,
    theta: theta_dot,
    x_dot: x_ddot,
    theta_dot: theta_ddot,
})

# Objective: minimize time + control effort
integrand = 0.01 * f_x**2
integral_var = phase.add_integral(integrand)
problem.minimize(t.final + integral_var)

# Mesh and guess
phase.mesh([8, 8, 8], [-1.0, -1/3, 1/3, 1.0])

# INITIAL GUESS SHOWCASE
# Simple initial guess: linear swing from down (0) to up (π)
states_guess = [
    # Interval 1: 9 state points, 8 control points
    np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x: cart stays at origin
        [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2],  # theta: swing up gradually
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x_dot: cart velocity
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # theta_dot: angular velocity
    ]),
    # Interval 2: continue swing
    np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]),
    # Interval 3: final approach to upright
    np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.6, 2.0, 2.4, 2.8, 3.0, 3.1, 3.13, 3.14, 3.14159],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]),
]

problem.guess(
    phase_states={1: states_guess},
    phase_terminal_times={1: 3.0},
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=20,
    min_polynomial_degree=4,
    max_polynomial_degree=10,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-8,
    },
)

# Results
if solution.status["success"]:
    swing_time = solution.phases[1]["times"]["final"]
    objective = solution.status["objective"]
    print(f"Swing-up time: {swing_time:.6f} seconds")
    print(f"Total objective (time + control effort): {objective:.6f}")

    # Final state verification
    x_final = solution[(1, "x")][-1]
    theta_final = solution[(1, "theta")][-1]
    x_dot_final = solution[(1, "x_dot")][-1]
    theta_dot_final = solution[(1, "theta_dot")][-1]

    print("Final states:")
    print(f"  Cart position: {x_final:.6f} (target: 0.0)")
    print(f"  Pendulum angle: {theta_final:.6f} (target: {np.pi:.6f})")
    print(f"  Cart velocity: {x_dot_final:.6f} (target: 0.0)")
    print(f"  Angular velocity: {theta_dot_final:.6f} (target: 0.0)")

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
