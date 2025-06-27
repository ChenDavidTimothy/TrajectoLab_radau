import casadi as ca
import numpy as np

import maptor as mtor


# ============================================================================
# Physical Parameters (Realistic 3DOF Manipulator)
# ============================================================================

# Link masses (kg)
m1 = 3.0  # Base link mass
m2 = 2.0  # Upper arm mass
m3 = 1.0  # Forearm mass

# Link lengths (m)
l1 = 0.3  # Base link length (vertical)
l2 = 0.4  # Upper arm length
l3 = 0.3  # Forearm length

# Center of mass distances (m)
lc1 = 0.15  # Base link COM distance from joint 1
lc2 = 0.20  # Upper arm COM distance from joint 2
lc3 = 0.15  # Forearm COM distance from joint 3

# Moments of inertia about COM (kg⋅m²)
I1 = m1 * l1**2 / 12  # Uniform rod approximation
I2 = m2 * l2**2 / 12  # Uniform rod approximation
I3 = m3 * l3**2 / 12  # Uniform rod approximation

# Gravity
g = 9.81  # m/s²


# ============================================================================
# Problem Setup
# ============================================================================

problem = mtor.Problem("3DOF Manipulator Point-to-Point 3D")
phase = problem.set_phase(1)


# ============================================================================
# Variables
# ============================================================================

# Time variable
t = phase.time(initial=0.0)

# State variables (joint angles and velocities)
q1 = phase.state("q1", initial=0.0, final=np.pi / 2)  # Base rotation: 0° to 90°
q2 = phase.state("q2", initial=np.pi / 2, final=np.pi / 4)  # Shoulder: vertical to 45°
q3 = phase.state("q3", initial=0.0, final=-np.pi / 6)  # Elbow: straight to -30°
q1_dot = phase.state("q1_dot", initial=0.0, final=0.0)  # Start and end at rest
q2_dot = phase.state("q2_dot", initial=0.0, final=0.0)  # Start and end at rest
q3_dot = phase.state("q3_dot", initial=0.0, final=0.0)  # Start and end at rest

# Control variables (joint torques)
tau1 = phase.control("tau1", boundary=(-30.0, 30.0))  # Base torque (N⋅m)
tau2 = phase.control("tau2", boundary=(-20.0, 20.0))  # Shoulder torque (N⋅m)
tau3 = phase.control("tau3", boundary=(-10.0, 10.0))  # Elbow torque (N⋅m)


# ============================================================================
# Dynamics (Generated from SymPy)
# ============================================================================

phase.dynamics(
    {
        q1: q1_dot,
        q2: q2_dot,
        q3: q3_dot,
        q1_dot: (
            2 * I2 * ca.sin(2 * q2) * q1_dot * q2_dot
            + 2 * I3 * ca.sin(2 * q2 + 2 * q3) * q1_dot * q2_dot
            + 2 * I3 * ca.sin(2 * q2 + 2 * q3) * q1_dot * q3_dot
            - 2 * l2**2 * m3 * ca.sin(2 * q2) * q1_dot * q2_dot
            - 4 * l2 * lc3 * m3 * ca.sin(2 * q2 + q3) * q1_dot * q2_dot
            - 2 * l2 * lc3 * m3 * ca.sin(2 * q2 + q3) * q1_dot * q3_dot
            - 2 * l2 * lc3 * m3 * ca.sin(q3) * q1_dot * q3_dot
            - 2 * lc2**2 * m2 * ca.sin(2 * q2) * q1_dot * q2_dot
            - 2 * lc3**2 * m3 * ca.sin(2 * q2 + 2 * q3) * q1_dot * q2_dot
            - 2 * lc3**2 * m3 * ca.sin(2 * q2 + 2 * q3) * q1_dot * q3_dot
            - 2 * tau1
        )
        / (
            -2 * I1
            + I2 * ca.cos(2 * q2)
            - I2
            + I3 * ca.cos(2 * q2 + 2 * q3)
            - I3
            - l2**2 * m3 * ca.cos(2 * q2)
            - l2**2 * m3
            - 2 * l2 * lc3 * m3 * ca.cos(2 * q2 + q3)
            - 2 * l2 * lc3 * m3 * ca.cos(q3)
            - lc2**2 * m2 * ca.cos(2 * q2)
            - lc2**2 * m2
            - lc3**2 * m3 * ca.cos(2 * q2 + 2 * q3)
            - lc3**2 * m3
        ),
        q2_dot: (
            lc3
            * (
                I2 * ca.sin(2 * q2) * q1_dot**2 / 2
                + I3 * ca.sin(2 * q2 + 2 * q3) * q1_dot**2 / 2
                + g * l2 * m3 * ca.cos(q2)
                + g * lc2 * m2 * ca.cos(q2)
                + g * lc3 * m3 * ca.cos(q2 + q3)
                + l2 * lc3 * m3 * (2 * q2_dot + q3_dot) * ca.sin(q3) * q3_dot
                - lc2**2 * m2 * ca.sin(2 * q2) * q1_dot**2 / 2
                - m3
                * (
                    l2**2 * ca.sin(2 * q2) / 2
                    + l2 * lc3 * ca.sin(2 * q2 + q3)
                    + lc3**2 * ca.sin(2 * q2 + 2 * q3) / 2
                )
                * q1_dot**2
                + tau2
            )
            + (l2 * ca.cos(q3) + lc3)
            * (
                -I3 * ca.sin(2 * q2 + 2 * q3) * q1_dot**2 / 2
                - g * lc3 * m3 * ca.cos(q2 + q3)
                + l2 * lc3 * m3 * ca.sin(2 * q2 + q3) * q1_dot**2 / 2
                + l2 * lc3 * m3 * ca.sin(q3) * q1_dot**2 / 2
                + l2 * lc3 * m3 * ca.sin(q3) * q2_dot**2
                + lc3**2 * m3 * ca.sin(2 * q2 + 2 * q3) * q1_dot**2 / 2
                - tau3
            )
        )
        / (lc3 * (l2**2 * m3 * ca.sin(q3) ** 2 + lc2**2 * m2)),
        q3_dot: (
            -lc3
            * m3
            * (l2 * ca.cos(q3) + lc3)
            * (
                I2 * ca.sin(2 * q2) * q1_dot**2 / 2
                + I3 * ca.sin(2 * q2 + 2 * q3) * q1_dot**2 / 2
                + g * l2 * m3 * ca.cos(q2)
                + g * lc2 * m2 * ca.cos(q2)
                + g * lc3 * m3 * ca.cos(q2 + q3)
                + l2 * lc3 * m3 * (2 * q2_dot + q3_dot) * ca.sin(q3) * q3_dot
                - lc2**2 * m2 * ca.sin(2 * q2) * q1_dot**2 / 2
                - m3
                * (
                    l2**2 * ca.sin(2 * q2) / 2
                    + l2 * lc3 * ca.sin(2 * q2 + q3)
                    + lc3**2 * ca.sin(2 * q2 + 2 * q3) / 2
                )
                * q1_dot**2
                + tau2
            )
            - (l2**2 * m3 + 2 * l2 * lc3 * m3 * ca.cos(q3) + lc2**2 * m2 + lc3**2 * m3)
            * (
                -I3 * ca.sin(2 * q2 + 2 * q3) * q1_dot**2 / 2
                - g * lc3 * m3 * ca.cos(q2 + q3)
                + l2 * lc3 * m3 * ca.sin(2 * q2 + q3) * q1_dot**2 / 2
                + l2 * lc3 * m3 * ca.sin(q3) * q1_dot**2 / 2
                + l2 * lc3 * m3 * ca.sin(q3) * q2_dot**2
                + lc3**2 * m3 * ca.sin(2 * q2 + 2 * q3) * q1_dot**2 / 2
                - tau3
            )
        )
        / (lc3**2 * m3 * (l2**2 * m3 * ca.sin(q3) ** 2 + lc2**2 * m2)),
    }
)


# ============================================================================
# Constraints
# ============================================================================

# Joint angle limits (realistic for 3D manipulator)
phase.path_constraints(
    q1 >= -np.pi,  # Base rotation limits
    q1 <= np.pi,
    q2 >= 0,  # Shoulder pitch limits
    q2 <= np.pi,
    q3 >= -np.pi / 2,  # Elbow pitch limits
    q3 <= np.pi / 2,
)

# End-effector workspace constraints (avoid ground collision)
# End-effector Z-position (height above base)
z_ee = l1 + l2 * ca.cos(q2) + l3 * ca.cos(q2 + q3)
phase.path_constraints(z_ee >= 0.05)  # Minimum 5cm above ground


# ============================================================================
# Objective
# ============================================================================

# Minimize energy consumption (torque-squared integral) plus time
energy = phase.add_integral(tau1**2 + tau2**2 + tau3**2)
problem.minimize(t.final + 0.1 * energy)


# ============================================================================
# Mesh Configuration and Initial Guess
# ============================================================================

phase.mesh([6, 6, 6], [-1.0, -1 / 3, 1 / 3, 1.0])

# Initial guess - linear interpolation for joint angles
states_guess = []
controls_guess = []

for N in [6, 6, 6]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear transition from initial to final joint angles
    q1_vals = 0.0 + (np.pi / 2) * t_norm  # 0 to π/2
    q2_vals = np.pi / 2 + (np.pi / 4 - np.pi / 2) * t_norm  # π/2 to π/4
    q3_vals = 0.0 + (-np.pi / 6) * t_norm  # 0 to -π/6
    q1_dot_vals = np.zeros(N + 1)  # Start and end at rest
    q2_dot_vals = np.zeros(N + 1)  # Start and end at rest
    q3_dot_vals = np.zeros(N + 1)  # Start and end at rest

    states_guess.append(
        np.vstack([q1_vals, q2_vals, q3_vals, q1_dot_vals, q2_dot_vals, q3_dot_vals])
    )

    # Small control torque guess
    tau1_vals = np.ones(N) * 0.1
    tau2_vals = np.ones(N) * 0.1
    tau3_vals = np.ones(N) * 0.1
    controls_guess.append(np.vstack([tau1_vals, tau2_vals, tau3_vals]))

phase.guess(
    states=states_guess,
    controls=controls_guess,
    terminal_time=3.0,
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
    print(f"Objective (time + energy): {solution.status['objective']:.6f}")
    print(f"Mission time: {solution.status['total_mission_time']:.3f} seconds")

    # Final joint angles
    q1_final = solution["q1"][-1]
    q2_final = solution["q2"][-1]
    q3_final = solution["q3"][-1]
    print(f"Final joint 1 angle: {q1_final:.6f} rad ({np.degrees(q1_final):.2f}°)")
    print(f"Final joint 2 angle: {q2_final:.6f} rad ({np.degrees(q2_final):.2f}°)")
    print(f"Final joint 3 angle: {q3_final:.6f} rad ({np.degrees(q3_final):.2f}°)")

    # End-effector position analysis
    # Initial position
    x_ee_initial = l2 * np.cos(np.pi / 2) * np.cos(0) + l3 * np.cos(np.pi / 2 + 0) * np.cos(0)
    y_ee_initial = l2 * np.cos(np.pi / 2) * np.sin(0) + l3 * np.cos(np.pi / 2 + 0) * np.sin(0)
    z_ee_initial = l1 + l2 * np.cos(np.pi / 2) + l3 * np.cos(np.pi / 2 + 0)

    # Final position
    x_ee_final = l2 * np.cos(q2_final) * np.cos(q1_final) + l3 * np.cos(
        q2_final + q3_final
    ) * np.cos(q1_final)
    y_ee_final = l2 * np.cos(q2_final) * np.sin(q1_final) + l3 * np.cos(
        q2_final + q3_final
    ) * np.sin(q1_final)
    z_ee_final = l1 + l2 * np.cos(q2_final) + l3 * np.cos(q2_final + q3_final)

    print(
        f"End-effector moved from ({x_ee_initial:.3f}, {y_ee_initial:.3f}, {z_ee_initial:.3f}) "
        f"to ({x_ee_final:.3f}, {y_ee_final:.3f}, {z_ee_final:.3f})"
    )

    # Control statistics
    tau1_max = max(np.abs(solution["tau1"]))
    tau2_max = max(np.abs(solution["tau2"]))
    tau3_max = max(np.abs(solution["tau3"]))
    print(f"Maximum joint 1 torque: {tau1_max:.3f} N⋅m")
    print(f"Maximum joint 2 torque: {tau2_max:.3f} N⋅m")
    print(f"Maximum joint 3 torque: {tau3_max:.3f} N⋅m")

    solution.plot()

else:
    print(f"Failed: {solution.status['message']}")
