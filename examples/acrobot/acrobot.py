import casadi as ca
import numpy as np

import maptor as mtor


# ============================================================================
# Physical Parameters
# ============================================================================

# Link masses (kg)
m1 = 1.0  # Upper arm mass
m2 = 1.0  # Forearm mass

# Link lengths (m)
l1 = 1.0  # Upper arm length
l2 = 1.0  # Forearm length

# Center of mass distances (m)
lc1 = 0.5  # Upper arm COM distance from shoulder
lc2 = 0.5  # Forearm COM distance from elbow

# Moments of inertia about pivots (kg⋅m²)
I1 = m1 * l1**2 / 3  # Upper arm about shoulder
I2 = m2 * l2**2 / 3  # Forearm about elbow

# Gravity
g = 9.81  # m/s²


# ============================================================================
# Problem Setup
# ============================================================================

problem = mtor.Problem("Acrobot Swing-Up")
phase = problem.set_phase(1)


# ============================================================================
# Variables
# ============================================================================

# Time variable
t = phase.time(initial=0.0)

# State variables
theta1 = phase.state("theta1", initial=0.0, final=np.pi)  # Shoulder: down to up
theta2 = phase.state("theta2", initial=0.0, final=0.0)  # Elbow: straight to straight
theta1_dot = phase.state("theta1_dot", initial=0.0, final=0.0)  # Start and end at rest
theta2_dot = phase.state("theta2_dot", initial=0.0, final=0.0)  # Start and end at rest

# Control variable (only elbow joint is actuated)
tau = phase.control("tau", boundary=(-20.0, 20.0))  # Elbow torque (N⋅m)


# ============================================================================
# Dynamics (Generated from acrobot_dynamics.py)
# ============================================================================

phase.dynamics(
    {
        theta1: theta1_dot,
        theta2: theta2_dot,
        theta1_dot: (
            -I2
            * (
                g * l1 * m2 * ca.sin(theta1)
                + g * lc1 * m1 * ca.sin(theta1)
                + g * lc2 * m2 * ca.sin(theta1 + theta2)
                - l1 * lc2 * m2 * (2 * theta1_dot + theta2_dot) * ca.sin(theta2) * theta2_dot
            )
            + (I2 + l1 * lc2 * m2 * ca.cos(theta2))
            * (
                g * lc2 * m2 * ca.sin(theta1 + theta2)
                + l1 * lc2 * m2 * ca.sin(theta2) * theta1_dot**2
                - tau
            )
        )
        / (I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * ca.cos(theta2) ** 2),
        theta2_dot: (
            (I2 + l1 * lc2 * m2 * ca.cos(theta2))
            * (
                g * l1 * m2 * ca.sin(theta1)
                + g * lc1 * m1 * ca.sin(theta1)
                + g * lc2 * m2 * ca.sin(theta1 + theta2)
                - l1 * lc2 * m2 * (2 * theta1_dot + theta2_dot) * ca.sin(theta2) * theta2_dot
            )
            - (
                g * lc2 * m2 * ca.sin(theta1 + theta2)
                + l1 * lc2 * m2 * ca.sin(theta2) * theta1_dot**2
                - tau
            )
            * (I1 + I2 + l1**2 * m2 + 2 * l1 * lc2 * m2 * ca.cos(theta2))
        )
        / (I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * ca.cos(theta2) ** 2),
    }
)


# ============================================================================
# Constraints
# ============================================================================

# Joint angle limits (realistic for acrobot)
phase.path_constraints(
    theta1 >= -np.pi / 6,  # Shoulder limit: -30° to 210°
    theta1 <= 7 * np.pi / 6,
    theta2 >= -2 * np.pi / 3,  # Elbow limit: -120° to 120°
    theta2 <= 2 * np.pi / 3,
)


# ============================================================================
# Objective
# ============================================================================

# Minimize time to swing up
problem.minimize(t.final)


# ============================================================================
# Mesh Configuration and Initial Guess
# ============================================================================

phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# Initial guess - linear interpolation for joint angles
states_guess = []
controls_guess = []

for N in [8, 8, 8]:
    tau_vals = np.linspace(-1, 1, N + 1)
    t_norm = (tau_vals + 1) / 2

    # Linear transition from downward (0) to upward (π)
    theta1_vals = np.pi * t_norm
    theta2_vals = np.zeros(N + 1)  # Straight configuration
    theta1_dot_vals = np.zeros(N + 1)
    theta2_dot_vals = np.zeros(N + 1)

    states_guess.append(np.vstack([theta1_vals, theta2_vals, theta1_dot_vals, theta2_dot_vals]))

    # Small control guess
    tau_vals = np.ones(N) * 0.1
    controls_guess.append(np.array([tau_vals]))

phase.guess(
    terminal_time=5.0,
)


# ============================================================================
# Solve
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-4,
    max_iterations=30,
    min_polynomial_degree=3,
    max_polynomial_degree=8,
    ode_solver_tolerance=1e-4,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-4,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
    },
)


# ============================================================================
# Results
# ============================================================================

if solution.status["success"]:
    print(f"Objective: {solution.status['objective']:.6f}")
    print(f"Mission time: {solution.status['total_mission_time']:.3f} seconds")

    # Final joint angles
    theta1_final = solution["theta1"][-1]
    theta2_final = solution["theta2"][-1]
    print(f"Final shoulder angle: {theta1_final:.6f} rad ({np.degrees(theta1_final):.2f}°)")
    print(f"Final elbow angle: {theta2_final:.6f} rad ({np.degrees(theta2_final):.2f}°)")

    # End-effector position analysis
    # Initial position (both links hanging down)
    x_ee_initial = l1 * np.sin(0) + l2 * np.sin(0 + 0)
    y_ee_initial = -l1 * np.cos(0) - l2 * np.cos(0 + 0)

    # Final position (both links pointing up)
    x_ee_final = l1 * np.sin(theta1_final) + l2 * np.sin(theta1_final + theta2_final)
    y_ee_final = -l1 * np.cos(theta1_final) - l2 * np.cos(theta1_final + theta2_final)

    print(
        f"End-effector moved from ({x_ee_initial:.3f}, {y_ee_initial:.3f}) to ({x_ee_final:.3f}, {y_ee_final:.3f})"
    )

    # Control statistics
    tau_max = max(np.abs(solution["tau"]))
    print(f"Maximum elbow torque: {tau_max:.3f} N⋅m")

    # Swing-up success verification
    swing_up_success = abs(theta1_final - np.pi) < 0.1 and abs(theta2_final) < 0.1
    print(f"Swing-up successful: {'✓' if swing_up_success else '✗'}")

    solution.plot()

else:
    print(f"Failed: {solution.status['message']}")
