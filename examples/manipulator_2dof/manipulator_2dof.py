import casadi as ca
import numpy as np

import maptor as mtor


# ============================================================================
# Physical Parameters (Realistic 2DOF Manipulator)
# ============================================================================

# Link masses (kg)
m1 = 2.0  # Link 1 mass
m2 = 1.5  # Link 2 mass

# Link lengths (m)
l1 = 0.5  # Link 1 length
l2 = 0.4  # Link 2 length

# Center of mass distances (m)
lc1 = 0.25  # Link 1 COM distance from joint 1
lc2 = 0.20  # Link 2 COM distance from joint 2

# Moments of inertia about COM (kg⋅m²)
I1 = m1 * l1**2 / 12  # Uniform rod approximation
I2 = m2 * l2**2 / 12  # Uniform rod approximation

# Gravity
g = 9.81  # m/s²


# ============================================================================
# Problem Setup
# ============================================================================

problem = mtor.Problem("2DOF Manipulator Point-to-Point")
phase = problem.set_phase(1)


# ============================================================================
# Variables
# ============================================================================

# Time variable
t = phase.time(initial=0.0)

# State variables (joint angles and velocities)
q1 = phase.state("q1", initial=np.pi / 2, final=0.0)  # Joint 1: vertical to horizontal
q2 = phase.state("q2", initial=0.0, final=0.0)  # Joint 2: straight to straight
q1_dot = phase.state("q1_dot", initial=0.0, final=0.0)  # Start and end at rest
q2_dot = phase.state("q2_dot", initial=0.0, final=0.0)  # Start and end at rest

# Control variables (joint torques)
tau1 = phase.control("tau1", boundary=(-20.0, 20.0))  # Joint 1 torque (N⋅m)
tau2 = phase.control("tau2", boundary=(-10.0, 10.0))  # Joint 2 torque (N⋅m)


# ============================================================================
# Dynamics (Generated from SymPy + Control Input)
# ============================================================================

phase.dynamics(
    {
        q1: q1_dot,
        q2: q2_dot,
        q1_dot: tau1
        * (-I2 - lc2**2 * m2)
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        )
        + tau2
        * (I2 + l1 * lc2 * m2 * ca.cos(q2) + lc2**2 * m2)
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        )
        + (-I2 - lc2**2 * m2)
        * (
            -g * l1 * m2 * ca.cos(q1)
            - g * lc1 * m1 * ca.cos(q1)
            - g * lc2 * m2 * (-ca.sin(q1) * ca.sin(q2) + ca.cos(q1) * ca.cos(q2))
            - m2
            * (
                -2 * l1 * lc2 * (q1_dot + q2_dot) * ca.sin(q2) * q2_dot
                - 2 * l1 * lc2 * ca.sin(q2) * q1_dot * q2_dot
            )
            / 2
        )
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        )
        + (I2 + l1 * lc2 * m2 * ca.cos(q2) + lc2**2 * m2)
        * (
            -g * lc2 * m2 * (-ca.sin(q1) * ca.sin(q2) + ca.cos(q1) * ca.cos(q2))
            - l1 * lc2 * m2 * (q1_dot + q2_dot) * ca.sin(q2) * q1_dot
            + l1 * lc2 * m2 * ca.sin(q2) * q1_dot * q2_dot
        )
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        ),
        q2_dot: tau1
        * (I2 + l1 * lc2 * m2 * ca.cos(q2) + lc2**2 * m2)
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        )
        + tau2
        * (-I1 - I2 - l1**2 * m2 - 2 * l1 * lc2 * m2 * ca.cos(q2) - lc1**2 * m1 - lc2**2 * m2)
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        )
        + (I2 + l1 * lc2 * m2 * ca.cos(q2) + lc2**2 * m2)
        * (
            -g * l1 * m2 * ca.cos(q1)
            - g * lc1 * m1 * ca.cos(q1)
            - g * lc2 * m2 * (-ca.sin(q1) * ca.sin(q2) + ca.cos(q1) * ca.cos(q2))
            - m2
            * (
                -2 * l1 * lc2 * (q1_dot + q2_dot) * ca.sin(q2) * q2_dot
                - 2 * l1 * lc2 * ca.sin(q2) * q1_dot * q2_dot
            )
            / 2
        )
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        )
        + (
            -g * lc2 * m2 * (-ca.sin(q1) * ca.sin(q2) + ca.cos(q1) * ca.cos(q2))
            - l1 * lc2 * m2 * (q1_dot + q2_dot) * ca.sin(q2) * q1_dot
            + l1 * lc2 * m2 * ca.sin(q2) * q1_dot * q2_dot
        )
        * (-I1 - I2 - l1**2 * m2 - 2 * l1 * lc2 * m2 * ca.cos(q2) - lc1**2 * m1 - lc2**2 * m2)
        / (
            -I1 * I2
            - I1 * lc2**2 * m2
            - I2 * l1**2 * m2
            - I2 * lc1**2 * m1
            + l1**2 * lc2**2 * m2**2 * ca.cos(q2) ** 2
            - l1**2 * lc2**2 * m2**2
            - lc1**2 * lc2**2 * m1 * m2
        ),
    }
)


# ============================================================================
# Constraints
# ============================================================================

# Joint angle limits (realistic for manipulator)
phase.path_constraints(
    q1 >= -np.pi,
    q1 <= np.pi,
    q2 >= -np.pi,
    q2 <= np.pi,
)


# ============================================================================
# Objective
# ============================================================================

# Minimize energy consumption (torque-squared integral)
energy = phase.add_integral(tau1**2 + tau2**2)
problem.minimize(t.final + energy)


# ============================================================================
# Mesh Configuration and Initial Guess
# ============================================================================

phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# Initial guess - linear interpolation for joint angles
states_guess = []
controls_guess = []

for N in [8, 8, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear transition from initial to final joint angles
    q1_vals = np.pi / 2 * (1 - t_norm)  # π/2 to 0
    q2_vals = np.zeros(N + 1)  # 0 to 0
    q1_dot_vals = np.zeros(N + 1)  # Start and end at rest
    q2_dot_vals = np.zeros(N + 1)  # Start and end at rest

    states_guess.append(np.vstack([q1_vals, q2_vals, q1_dot_vals, q2_dot_vals]))

    # Small control torque guess
    tau1_vals = np.ones(N) * 0.1
    tau2_vals = np.ones(N) * 0.1
    controls_guess.append(np.vstack([tau1_vals, tau2_vals]))

phase.guess(
    states=states_guess,
    controls=controls_guess,
    terminal_time=5.0,
)


# ============================================================================
# Solve
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=5e-3,
    max_iterations=15,
    min_polynomial_degree=3,
    max_polynomial_degree=8,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-4,
    },
)


# ============================================================================
# Results
# ============================================================================

if solution.status["success"]:
    print(f"Objective (energy): {solution.status['objective']:.6f}")
    print(f"Mission time: {solution.status['total_mission_time']:.3f} seconds")

    # Final joint angles
    q1_final = solution["q1"][-1]
    q2_final = solution["q2"][-1]
    print(f"Final joint 1 angle: {q1_final:.6f} rad ({np.degrees(q1_final):.2f}°)")
    print(f"Final joint 2 angle: {q2_final:.6f} rad ({np.degrees(q2_final):.2f}°)")

    # End-effector position
    x_ee_initial = l1 * np.cos(np.pi / 2) + l2 * np.cos(np.pi / 2 + 0)
    y_ee_initial = l1 * np.sin(np.pi / 2) + l2 * np.sin(np.pi / 2 + 0)
    x_ee_final = l1 * np.cos(q1_final) + l2 * np.cos(q1_final + q2_final)
    y_ee_final = l1 * np.sin(q1_final) + l2 * np.sin(q1_final + q2_final)

    print(
        f"End-effector moved from ({x_ee_initial:.3f}, {y_ee_initial:.3f}) to ({x_ee_final:.3f}, {y_ee_final:.3f})"
    )

    # Control statistics
    tau1_max = max(np.abs(solution["tau1"]))
    tau2_max = max(np.abs(solution["tau2"]))
    print(f"Maximum joint 1 torque: {tau1_max:.3f} N⋅m")
    print(f"Maximum joint 2 torque: {tau2_max:.3f} N⋅m")

    solution.plot()

else:
    print(f"Failed: {solution.status['message']}")
