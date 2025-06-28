import casadi as ca
import numpy as np

import maptor as mtor


# ============================================================================
# Physical Parameters
# ============================================================================

m1 = 3.0
m2 = 2.5
m3 = 1.5
m_box = 5.0

l1 = 0.3
l2 = 0.4
l3 = 0.4

lc1 = 0.15
lc2 = 0.20
lc3 = 0.20

I1 = m1 * l1**2 / 12
I2 = m2 * l2**2 / 12
I3 = m3 * l3**2 / 12

g = 9.81


# ============================================================================
# End-Effector Position Specification
# ============================================================================

x_ee_initial = 0.0
y_ee_initial = 0.5
z_ee_initial = 0.1

x_ee_final = 0.0
y_ee_final = -0.5
z_ee_final = 0.1


# ============================================================================
# Inverse Kinematics
# ============================================================================


def calculate_inverse_kinematics(x_target, y_target, z_target, l1, l2, l3):
    q1 = np.arctan2(y_target, x_target)

    r_horizontal = np.sqrt(x_target**2 + y_target**2)
    z_relative = z_target - l1
    r_total = np.sqrt(r_horizontal**2 + z_relative**2)

    max_reach = l2 + l3
    min_reach = abs(l2 - l3)
    if r_total > max_reach:
        raise ValueError(f"Target unreachable: distance {r_total:.3f} > max reach {max_reach:.3f}")
    if r_total < min_reach:
        raise ValueError(f"Target too close: distance {r_total:.3f} < min reach {min_reach:.3f}")

    cos_q3 = (r_total**2 - l2**2 - l3**2) / (2 * l2 * l3)
    q3 = -np.arccos(np.clip(cos_q3, -1, 1))

    alpha = np.arctan2(z_relative, r_horizontal)
    beta = np.arctan2(l3 * np.sin(-q3), l2 + l3 * np.cos(-q3))
    q2 = alpha + beta

    return q1, q2, q3


def verify_forward_kinematics(q1, q2, q3, l1, l2, l3):
    x_ee = (l2 * np.cos(q2) + l3 * np.cos(q2 + q3)) * np.cos(q1)
    y_ee = (l2 * np.cos(q2) + l3 * np.cos(q2 + q3)) * np.sin(q1)
    z_ee = l1 + l2 * np.sin(q2) + l3 * np.sin(q2 + q3)
    return x_ee, y_ee, z_ee


q1_initial, q2_initial, q3_initial = calculate_inverse_kinematics(
    x_ee_initial, y_ee_initial, z_ee_initial, l1, l2, l3
)
q1_final, q2_final, q3_final = calculate_inverse_kinematics(
    x_ee_final, y_ee_final, z_ee_final, l1, l2, l3
)

x_check, y_check, z_check = verify_forward_kinematics(q1_final, q2_final, q3_final, l1, l2, l3)
position_error = np.sqrt(
    (x_check - x_ee_final) ** 2 + (y_check - y_ee_final) ** 2 + (z_check - z_ee_final) ** 2
)

print("=== INVERSE KINEMATICS VERIFICATION ===")
print(f"Target position: ({x_ee_final:.3f}, {y_ee_final:.3f}, {z_ee_final:.3f}) m")
print(
    f"Calculated joint angles: q1={np.degrees(q1_final):.1f}°, q2={np.degrees(q2_final):.1f}°, q3={np.degrees(q3_final):.1f}°"
)
print(f"Forward kinematics check: ({x_check:.3f}, {y_check:.3f}, {z_check:.3f}) m")
print(f"Position error: {position_error:.6f} m")
print()


# ============================================================================
# Problem Setup
# ============================================================================

problem = mtor.Problem("KUKA-Scale 3DOF Manipulator")
phase = problem.set_phase(1)


# ============================================================================
# Variables
# ============================================================================

t = phase.time(initial=0.0)

q1 = phase.state("q1", initial=q1_initial, final=q1_final, boundary=(-np.pi, np.pi))
q2 = phase.state("q2", initial=q2_initial, final=q2_final, boundary=(-np.pi / 6, 5 * np.pi / 6))
q3 = phase.state("q3", initial=q3_initial, final=q3_final, boundary=(-2.5, 2.5))

q1_dot = phase.state("q1_dot", initial=0.0, final=0.0, boundary=(-1.5, 1.5))
q2_dot = phase.state("q2_dot", initial=0.0, final=0.0, boundary=(-1.2, 1.2))
q3_dot = phase.state("q3_dot", initial=0.0, final=0.0, boundary=(-2.0, 2.0))

tau1 = phase.control("tau1", boundary=(-80.0, 80.0))
tau2 = phase.control("tau2", boundary=(-120.0, 120.0))
tau3 = phase.control("tau3", boundary=(-60.0, 60.0))


# ============================================================================
# Dynamics
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
                + g * l2 * m_box * ca.cos(q2)
                + g * l3 * m_box * ca.cos(q2 + q3)
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
                - g * l3 * m_box * ca.cos(q2 + q3)
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
                + g * l2 * m_box * ca.cos(q2)
                + g * l3 * m_box * ca.cos(q2 + q3)
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
                - g * l3 * m_box * ca.cos(q2 + q3)
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

z_ee = l1 + l2 * ca.sin(q2) + l3 * ca.sin(q2 + q3)
phase.path_constraints(z_ee >= 0.05)


# ============================================================================
# Objective
# ============================================================================

energy = phase.add_integral(tau1**2 + tau2**2 + tau3**2)
problem.minimize(t.final + 0.01 * energy)


# ============================================================================
# Mesh Configuration and Initial Guess
# ============================================================================

num_interval = 12
degree = [4]
final_mesh = degree * num_interval
phase.mesh(final_mesh, np.linspace(-1.0, 1.0, num_interval + 1))

states_guess = []
controls_guess = []

for N in final_mesh:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    q1_vals = q1_initial + (q1_final - q1_initial) * t_norm
    q2_vals = q2_initial + (q2_final - q2_initial) * t_norm
    q3_vals = q3_initial + (q3_final - q3_initial) * t_norm

    q1_dot_vals = (q1_final - q1_initial) * np.sin(np.pi * t_norm) * 0.5
    q2_dot_vals = (q2_final - q2_initial) * np.sin(np.pi * t_norm) * 0.5
    q3_dot_vals = (q3_final - q3_initial) * np.sin(np.pi * t_norm) * 0.5

    states_guess.append(
        np.vstack([q1_vals, q2_vals, q3_vals, q1_dot_vals, q2_dot_vals, q3_dot_vals])
    )

    tau1_vals = np.ones(N) * 5.0
    tau2_vals = np.ones(N) * 10.0
    tau3_vals = np.ones(N) * 5.0
    controls_guess.append(np.vstack([tau1_vals, tau2_vals, tau3_vals]))

phase.guess(
    states=states_guess,
    controls=controls_guess,
    terminal_time=6.0,
)


# ============================================================================
# Solve
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-3,
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

    q1_solved = solution["q1"][-1]
    q2_solved = solution["q2"][-1]
    q3_solved = solution["q3"][-1]
    print(f"Final joint 1 angle: {q1_solved:.6f} rad ({np.degrees(q1_solved):.2f}°)")
    print(f"Final joint 2 angle: {q2_solved:.6f} rad ({np.degrees(q2_solved):.2f}°)")
    print(f"Final joint 3 angle: {q3_solved:.6f} rad ({np.degrees(q3_solved):.2f}°)")

    x_ee_achieved = (l2 * np.cos(q2_solved) + l3 * np.cos(q2_solved + q3_solved)) * np.cos(
        q1_solved
    )
    y_ee_achieved = (l2 * np.cos(q2_solved) + l3 * np.cos(q2_solved + q3_solved)) * np.sin(
        q1_solved
    )
    z_ee_achieved = l1 + l2 * np.sin(q2_solved) + l3 * np.sin(q2_solved + q3_solved)

    position_error_final = np.sqrt(
        (x_ee_achieved - x_ee_final) ** 2
        + (y_ee_achieved - y_ee_final) ** 2
        + (z_ee_achieved - z_ee_final) ** 2
    )

    print(f"Target end-effector position: ({x_ee_final:.3f}, {y_ee_final:.3f}, {z_ee_final:.3f}) m")
    print(
        f"Achieved end-effector position: ({x_ee_achieved:.3f}, {y_ee_achieved:.3f}, {z_ee_achieved:.3f}) m"
    )
    print(f"Position error: {position_error_final:.6f} m")

    tau1_max = max(np.abs(solution["tau1"]))
    tau2_max = max(np.abs(solution["tau2"]))
    tau3_max = max(np.abs(solution["tau3"]))
    print(f"Maximum joint 1 torque: {tau1_max:.3f} N⋅m")
    print(f"Maximum joint 2 torque: {tau2_max:.3f} N⋅m")
    print(f"Maximum joint 3 torque: {tau3_max:.3f} N⋅m")

    q1_dot_max = max(np.abs(solution["q1_dot"]))
    q2_dot_max = max(np.abs(solution["q2_dot"]))
    q3_dot_max = max(np.abs(solution["q3_dot"]))
    print(f"Maximum joint 1 velocity: {q1_dot_max:.3f} rad/s ({np.degrees(q1_dot_max):.1f}°/s)")
    print(f"Maximum joint 2 velocity: {q2_dot_max:.3f} rad/s ({np.degrees(q2_dot_max):.1f}°/s)")
    print(f"Maximum joint 3 velocity: {q3_dot_max:.3f} rad/s ({np.degrees(q3_dot_max):.1f}°/s)")

    solution.plot()

else:
    print(f"Failed: {solution.status['message']}")
