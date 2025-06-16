import casadi as ca
import numpy as np

import maptor as mtor


# ============================================================================
# Physical Parameters and Constants - DJI Mavic 3 Specifications
# ============================================================================

# Vehicle parameters (DJI Mavic 3 specifications)
m = 0.92  # Mass (kg) - average of Mavic 3E (915g) and 3T (920g)
g = 9.81  # Gravitational acceleration (m/s²)
l_arm = 0.19  # Arm length (m) - calculated from 380.1mm diagonal

# Inertia parameters (kg⋅m²) - scaled from original based on DJI Mavic 3 mass/size
Jx = 0.0123  # Scaled: 0.0347563 × 0.353
Jy = 0.0123  # Scaled: 0.0347563 × 0.353
Jz = 0.0345  # Scaled: 0.0977 × 0.353
Jr = 0.000030  # Rotor inertia - scaled: 0.000084 × 0.353

# Aerodynamic coefficients (adjusted for DJI Mavic 3 size/performance)
K_T = 2.5e-6  # Thrust coefficient - reduced for smaller/more efficient motors
K_d = 6.3e-8  # Drag coefficient - scaled proportionally

# Drag coefficients (linear damping) - reduced for more streamlined design
K_dx = 0.15
K_dy = 0.15
K_dz = 0.20

# Motor constraints (based on DJI Mavic 3 performance: 200°/s max angular velocity)
omega_max = 8000.0  # Maximum motor speed (rad/s) - reduced for smaller drone
omega_min = 0.0  # Minimum motor speed (rad/s)


# ============================================================================
# Problem Setup
# ============================================================================

problem = mtor.Problem("Quadrotor Minimum Time Trajectory")
phase = problem.set_phase(1)


# ============================================================================
# Variables Definition
# ============================================================================

# Time variable (free final time for minimum time problem)
t = phase.time(initial=0.0)

# Position states (global frame)
X = phase.state("X", initial=5.0, final=20.0)
Y = phase.state("Y", initial=5.0, final=20.0)
Z = phase.state("Z", initial=5.0, final=20.0, boundary=(0.0, None))

# Velocity states (global frame)
X_dot = phase.state("X_dot", initial=0.0, final=0)
Y_dot = phase.state("Y_dot", initial=0.0, final=0)
Z_dot = phase.state("Z_dot", initial=0.0, final=0)

# Attitude states (Euler angles)
phi = phase.state("phi", initial=0.0, final=0.0, boundary=(-np.pi / 3, np.pi / 3))
theta = phase.state("theta", initial=0.0, final=0.0, boundary=(-np.pi / 3, np.pi / 3))
psi = phase.state("psi", initial=0.0, boundary=(-np.pi, np.pi))

# Angular rate states (body frame)
p = phase.state("p", initial=0.0, final=0.0, boundary=(-10.0, 10.0))
q = phase.state("q", initial=0.0, final=0.0, boundary=(-10.0, 10.0))
r = phase.state("r", initial=0.0, final=0.0, boundary=(-5.0, 5.0))

# Control inputs (motor speeds)
omega1 = phase.control("omega1", boundary=(omega_min, omega_max))
omega2 = phase.control("omega2", boundary=(omega_min, omega_max))
omega3 = phase.control("omega3", boundary=(omega_min, omega_max))
omega4 = phase.control("omega4", boundary=(omega_min, omega_max))


# ============================================================================
# Dynamics Implementation
# ============================================================================

# Note: The first equation in the provided image shows the rotation matrix
# relationship between body frame and global frame velocities. In this
# implementation, we use global frame velocities (X_dot, Y_dot, Z_dot)
# directly as states, which is the standard approach for quadrotor control.

# Total thrust and gyroscopic effect from motor speeds
F_T = K_T * (omega1**2 + omega2**2 + omega3**2 + omega4**2)
omega_prop = omega1 - omega2 + omega3 - omega4

# Trigonometric functions for rotation matrices
cos_phi, sin_phi = ca.cos(phi), ca.sin(phi)
cos_theta, sin_theta = ca.cos(theta), ca.sin(theta)
cos_psi, sin_psi = ca.cos(psi), ca.sin(psi)
tan_theta = ca.tan(theta)

# Position dynamics (simple integration of global velocities)
X_dynamics = X_dot
Y_dynamics = Y_dot
Z_dynamics = Z_dot

# Translational dynamics (global frame accelerations)
# Exact implementation from image equations:
# [Ẍ^G]   [1/m (- [cos(φ)cos(ψ)sin(θ) + sin(φ)sin(ψ)] F_T^b - K_dx Ẋ^G)]
# [Ÿ^G] = [1/m (- [cos(φ)sin(ψ)sin(θ) - cos(ψ)sin(φ)] F_T^b - K_dy Ÿ^G)]
# [Z̈^G]   [1/m (- [cos(φ)cos(θ)] F_T^b - K_dz Ż^G) + g                ]

X_dot_dynamics = (1 / m) * (
    -(cos_phi * cos_psi * sin_theta + sin_phi * sin_psi) * F_T - K_dx * X_dot
)
Y_dot_dynamics = (1 / m) * (
    -(cos_phi * sin_psi * sin_theta - cos_psi * sin_phi) * F_T - K_dy * Y_dot
)
Z_dot_dynamics = (1 / m) * (-cos_phi * cos_theta * F_T - K_dz * Z_dot) + g

# Attitude dynamics (exact implementation from image)
# [φ̇]   [1  sin(φ)tan(θ)  cos(φ)tan(θ)] [p]
# [θ̇] = [0  cos(φ)        -sin(φ)     ] [q]
# [ψ̇]   [0  sin(φ)/cos(θ) cos(φ)/cos(θ)] [r]

cos_theta_safe = ca.fmax(ca.fabs(cos_theta), 1e-6)
sec_theta = 1.0 / cos_theta_safe

phi_dynamics = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
theta_dynamics = cos_phi * q - sin_phi * r
psi_dynamics = (sin_phi * q + cos_phi * r) * sec_theta

# Angular acceleration dynamics (exact implementation from image)
# [ṗ]   [1/Jx [(Jy - Jz)qr - Jr q(ω1 - ω2 + ω3 - ω4) + ℓKT(ω1² - ω3²)]]
# [q̇] = [1/Jy [(Jz - Jx)pr + Jr p(ω1 - ω2 + ω3 - ω4) + ℓKT(ω2² - ω4²)]]
# [ṙ]   [1/Jz [(Jx - Jy)pq + Kd(ω1² - ω2² + ω3² - ω4²)              ]]

p_dynamics = (1 / Jx) * (
    (Jy - Jz) * q * r - Jr * q * omega_prop + l_arm * K_T * (omega1**2 - omega3**2)
)
q_dynamics = (1 / Jy) * (
    (Jz - Jx) * p * r + Jr * p * omega_prop + l_arm * K_T * (omega2**2 - omega4**2)
)
r_dynamics = (1 / Jz) * ((Jx - Jy) * p * q + K_d * (omega1**2 - omega2**2 + omega3**2 - omega4**2))

# Define complete dynamics
phase.dynamics(
    {
        X: X_dynamics,
        Y: Y_dynamics,
        Z: Z_dynamics,
        X_dot: X_dot_dynamics,
        Y_dot: Y_dot_dynamics,
        Z_dot: Z_dot_dynamics,
        phi: phi_dynamics,
        theta: theta_dynamics,
        psi: psi_dynamics,
        p: p_dynamics,
        q: q_dynamics,
        r: r_dynamics,
    }
)

# ============================================================================
# Objective Function
# ============================================================================

omega_reg = omega1**2 + omega2**2 + omega3**2 + omega4**2
omega_integral = phase.add_integral(omega_reg)
# Minimize flight time
problem.minimize(t.final + 0.2 * omega_integral)


# ============================================================================
# Mesh Configuration and Initial Guess
# ============================================================================

# Configure mesh for complex dynamics
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# Generate initial guess
states_guess = []
controls_guess = []

for N in [8, 8, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear trajectory guess
    X_vals = 0.0 + 20.0 * t_norm
    Y_vals = 0.0 + 20.0 * t_norm
    Z_vals = 0.0 + 20.0 * t_norm

    # Smooth velocity profile
    X_dot_vals = 10.0 * np.sin(np.pi * t_norm)
    Y_dot_vals = 10.0 * np.sin(np.pi * t_norm)
    Z_dot_vals = 5.0 * np.sin(np.pi * t_norm)

    # Small attitude variations
    phi_vals = 0.1 * np.sin(2 * np.pi * t_norm)
    theta_vals = 0.1 * np.cos(2 * np.pi * t_norm)
    psi_vals = 0.2 * t_norm

    # Angular rates near zero
    p_vals = np.zeros(N + 1)
    q_vals = np.zeros(N + 1)
    r_vals = np.zeros(N + 1)

    states_guess.append(
        np.vstack(
            [
                X_vals,
                Y_vals,
                Z_vals,
                X_dot_vals,
                Y_dot_vals,
                Z_dot_vals,
                phi_vals,
                theta_vals,
                psi_vals,
                p_vals,
                q_vals,
                r_vals,
            ]
        )
    )

    # Hover motor speed baseline
    hover_speed = np.sqrt(m * g / (4 * K_T))
    omega_vals = np.full(N, hover_speed)

    controls_guess.append(np.vstack([omega_vals, omega_vals, omega_vals, omega_vals]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 8.0},
)


# ============================================================================
# Solve the Problem
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-2,
    max_iterations=15,
    min_polynomial_degree=3,
    max_polynomial_degree=8,
    ode_solver_tolerance=1e-2,
    nlp_options={
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 3,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.acceptable_iter": 5,
    },
)


# ============================================================================
# Results Analysis
# ============================================================================

if solution.status["success"]:
    flight_time = solution.status["objective"]
    print(f"Minimum flight time: {flight_time:.3f} seconds")

    # Final state verification
    X_final = solution["X"][-1]
    Y_final = solution["Y"][-1]
    Z_final = solution["Z"][-1]
    position_error = np.sqrt((X_final - 20.0) ** 2 + (Y_final - 20.0) ** 2 + (Z_final - 20.0) ** 2)

    print(f"Final position: ({X_final:.3f}, {Y_final:.3f}, {Z_final:.3f}) m")
    print(f"Position error: {position_error:.3f} m")

    # Flight envelope analysis
    max_speed = max(
        np.sqrt(solution["X_dot"] ** 2 + solution["Y_dot"] ** 2 + solution["Z_dot"] ** 2)
    )
    max_phi = max(np.abs(solution["phi"])) * 180 / np.pi
    max_theta = max(np.abs(solution["theta"])) * 180 / np.pi

    print(f"Maximum speed: {max_speed:.2f} m/s")
    print(f"Maximum roll angle: {max_phi:.1f}°")
    print(f"Maximum pitch angle: {max_theta:.1f}°")

    # Motor usage analysis
    avg_motor_speed = (
        solution["omega1"] + solution["omega2"] + solution["omega3"] + solution["omega4"]
    ) / 4
    max_motor_speed = max(
        np.maximum.reduce(
            [solution["omega1"], solution["omega2"], solution["omega3"], solution["omega4"]]
        )
    )

    print(f"Average motor speed: {np.mean(avg_motor_speed):.1f} rad/s")
    print(f"Maximum motor speed: {max_motor_speed:.1f} rad/s")
    print(f"Motor utilization: {max_motor_speed / omega_max * 100:.1f}%")

    solution.plot()

else:
    print(f"Optimization failed: {solution.status['message']}")
