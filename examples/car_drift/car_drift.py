import casadi as ca
import numpy as np

import maptor as mtor


# Standalone high-fidelity vehicle dynamics model - ORGANIZED VERSION
problem = mtor.Problem("Vehicle Dynamics - Simple Point-to-Point")
phase = problem.set_phase(1)

# ============================================================================
# VEHICLE PARAMETERS
# ============================================================================

# Basic vehicle properties
m_v = 1200.0  # vehicle mass (kg)
I_z = 1800.0  # yaw moment of inertia (kg*m^2)
I_w = 1.2  # wheel moment of inertia (kg*m^2)
l_f = 1.25  # distance from CG to front axle (m)
l_r = 1.45  # distance from CG to rear axle (m)
l = l_f + l_r  # wheelbase (m)
w_t = 1.55  # track width (m)
h_c = 0.48  # height of CG (m)
r_w = 0.325  # effective rolling radius (m)
g = 9.81  # gravity (m/s^2)

# Aerodynamic properties
rho = 1.225  # air density (kg/m^3)
C_d = 0.32  # drag coefficient
C_l = 0.15  # lift coefficient
A = 1.6  # frontal area (m^2)

# Tire model parameters (simplified Magic Formula)
mu_0 = 1.0  # reference friction coefficient
B_x, C_x = 12.0, 1.35  # longitudinal stiffness and shape factors
d_1x, d_2x = 0.85, 220.0  # longitudinal peak factor coefficients
B_y, C_y = 9.5, 1.28  # lateral stiffness and shape factors
d_1y, d_2y = 0.88, 200.0  # lateral peak factor coefficients

# Drivetrain setup (RWD)
k_t = 0.0  # traction distribution (RWD = rear only)
k_b = 0.6  # brake distribution (front bias)

# Physical limits
mu_x_max = 1.0  # max longitudinal friction
mu_y_max = 1.0  # max lateral friction
mu = 1.0  # road friction coefficient
P_rl_max = 180000.0  # max power rear left motor (W)
P_rr_max = 180000.0  # max power rear right motor (W)

# ============================================================================
# STATE VARIABLES (Properly Ordered: Positions → Velocities → Wheels)
# ============================================================================

t = phase.time(initial=0.0, final=8.0)

# Global position states (positions first)
X = phase.state("global_x", initial=0.0, final=10.0, boundary=(-50.0, 50.0))
Y = phase.state("global_y", initial=0.0, final=10.0, boundary=(-50.0, 50.0))
psi = phase.state("heading", initial=0.0, boundary=(-4 * np.pi, 4 * np.pi))

# Vehicle motion states (velocities and rates)
V = phase.state("velocity", initial=5.0, boundary=(1.0, 40.0))
beta = phase.state("sideslip_angle", initial=0.0, boundary=(-0.35, 0.35))
gamma = phase.state("yaw_rate", initial=0.0, boundary=(-3.0, 3.0))

# Wheel states (angular velocities)
omega_fl = phase.state("wheel_speed_fl", initial=15.0, boundary=(0.0, 200.0))
omega_fr = phase.state("wheel_speed_fr", initial=15.0, boundary=(0.0, 200.0))
omega_rl = phase.state("wheel_speed_rl", initial=15.0, boundary=(0.0, 200.0))
omega_rr = phase.state("wheel_speed_rr", initial=15.0, boundary=(0.0, 200.0))

# ============================================================================
# CONTROL INPUTS
# ============================================================================

T_t = phase.control("traction_torque", boundary=(0.0, 4000.0))
T_b = phase.control("brake_torque", boundary=(0.0, 6000.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))

# Algebraic variables as controls (for load transfer coupling)
a_x_bar = phase.control("aux_long_accel", boundary=(-12.0, 12.0))
a_y_bar = phase.control("aux_lat_accel", boundary=(-12.0, 12.0))

# ============================================================================
# VEHICLE DYNAMICS CALCULATIONS
# ============================================================================

# Vehicle body velocity components
V_x = V * ca.cos(beta)
V_y = V * ca.sin(beta)

# Aerodynamic forces
F_drag = 0.5 * rho * C_d * A * V_x**2
F_lift = 0.5 * rho * C_l * A * V_x**2

# Load transfer model (equations 15-18)
F_zfl = (
    0.5 * m_v * (l_r / l * g - h_c / l * a_x_bar)
    - m_v * (l_r / l * g - h_c / l * a_x_bar) * h_c / w_t * a_y_bar / g
    - F_lift / 4
)
F_zfr = (
    0.5 * m_v * (l_r / l * g - h_c / l * a_x_bar)
    + m_v * (l_r / l * g - h_c / l * a_x_bar) * h_c / w_t * a_y_bar / g
    - F_lift / 4
)
F_zrl = (
    0.5 * m_v * (l_f / l * g + h_c / l * a_x_bar)
    - m_v * (l_f / l * g + h_c / l * a_x_bar) * h_c / w_t * a_y_bar / g
    - F_lift / 4
)
F_zrr = (
    0.5 * m_v * (l_f / l * g + h_c / l * a_x_bar)
    + m_v * (l_f / l * g + h_c / l * a_x_bar) * h_c / w_t * a_y_bar / g
    - F_lift / 4
)

# Torque distribution (equations 23-24)
T_f = k_t * T_t + k_b * T_b
T_r = (1 - k_t) * T_t + (1 - k_b) * T_b

# Individual wheel torques (equations 25-28)
F_zf = F_zfl + F_zfr
F_zr = F_zrl + F_zrr
Delta_F_zf = 0.5 * (F_zfr - F_zfl)
Delta_F_zr = 0.5 * (F_zrr - F_zrl)

T_fl = T_f / 2 * (1 - Delta_F_zf / ca.fmax(F_zf, 100.0))
T_fr = T_f / 2 * (1 + Delta_F_zf / ca.fmax(F_zf, 100.0))
T_rl = T_r / 2 * (1 - Delta_F_zr / ca.fmax(F_zr, 100.0))
T_rr = T_r / 2 * (1 + Delta_F_zr / ca.fmax(F_zr, 100.0))

# Tire contact patch velocities (equations 30-37)
cos_delta = ca.cos(delta)
sin_delta = ca.sin(delta)

V_xfl = (V_x - w_t / 2 * gamma) * cos_delta + (V_y + l_f * gamma) * sin_delta
V_xfr = (V_x + w_t / 2 * gamma) * cos_delta + (V_y + l_f * gamma) * sin_delta
V_xrl = V_x - w_t / 2 * gamma
V_xrr = V_x + w_t / 2 * gamma

V_yfl = (V_y + l_f * gamma) * cos_delta - (V_x - w_t / 2 * gamma) * sin_delta
V_yfr = (V_y + l_f * gamma) * cos_delta - (V_x + w_t / 2 * gamma) * sin_delta
V_yrl = V_y - l_r * gamma
V_yrr = V_y - l_r * gamma

# Tire slip calculations (equations 38-39)
eps_v = 0.1
lambda_fl = (r_w * omega_fl - V_xfl) / ca.fmax(ca.fabs(V_xfl), eps_v)
lambda_fr = (r_w * omega_fr - V_xfr) / ca.fmax(ca.fabs(V_xfr), eps_v)
lambda_rl = (r_w * omega_rl - V_xrl) / ca.fmax(ca.fabs(V_xrl), eps_v)
lambda_rr = (r_w * omega_rr - V_xrr) / ca.fmax(ca.fabs(V_xrr), eps_v)

alpha_fl = ca.atan2(V_yfl, ca.fmax(ca.fabs(V_xfl), eps_v))
alpha_fr = ca.atan2(V_yfr, ca.fmax(ca.fabs(V_xfr), eps_v))
alpha_rl = ca.atan2(V_yrl, ca.fmax(ca.fabs(V_xrl), eps_v))
alpha_rr = ca.atan2(V_yrr, ca.fmax(ca.fabs(V_xrr), eps_v))

# Combined slip magnitudes (equation 44)
sigma_fl = ca.sqrt(lambda_fl**2 + ca.tan(alpha_fl) ** 2)
sigma_fr = ca.sqrt(lambda_fr**2 + ca.tan(alpha_fr) ** 2)
sigma_rl = ca.sqrt(lambda_rl**2 + ca.tan(alpha_rl) ** 2)
sigma_rr = ca.sqrt(lambda_rr**2 + ca.tan(alpha_rr) ** 2)


# Magic Formula tire force functions
def tire_force_longitudinal(sigma, F_z):
    D_x = d_1x * F_z + d_2x
    return (mu / mu_0) * D_x * ca.sin(C_x * ca.atan(B_x * sigma))


def tire_force_lateral(sigma, F_z):
    D_y = d_1y * F_z + d_2y
    return -(mu / mu_0) * D_y * ca.sin(C_y * ca.atan(B_y * sigma))


# Pure slip forces (equations 40-43)
F_x0_fl = tire_force_longitudinal(sigma_fl, ca.fmax(F_zfl, 50.0))
F_x0_fr = tire_force_longitudinal(sigma_fr, ca.fmax(F_zfr, 50.0))
F_x0_rl = tire_force_longitudinal(sigma_rl, ca.fmax(F_zrl, 50.0))
F_x0_rr = tire_force_longitudinal(sigma_rr, ca.fmax(F_zrr, 50.0))

F_y0_fl = tire_force_lateral(sigma_fl, ca.fmax(F_zfl, 50.0))
F_y0_fr = tire_force_lateral(sigma_fr, ca.fmax(F_zfr, 50.0))
F_y0_rl = tire_force_lateral(sigma_rl, ca.fmax(F_zrl, 50.0))
F_y0_rr = tire_force_lateral(sigma_rr, ca.fmax(F_zrr, 50.0))

# Combined slip tire forces (equation 45)
eps_sigma = 1e-6
F_xfl = lambda_fl / ca.fmax(sigma_fl, eps_sigma) * F_x0_fl
F_xfr = lambda_fr / ca.fmax(sigma_fr, eps_sigma) * F_x0_fr
F_xrl = lambda_rl / ca.fmax(sigma_rl, eps_sigma) * F_x0_rl
F_xrr = lambda_rr / ca.fmax(sigma_rr, eps_sigma) * F_x0_rr

F_yfl = ca.tan(alpha_fl) / ca.fmax(sigma_fl, eps_sigma) * F_y0_fl
F_yfr = ca.tan(alpha_fr) / ca.fmax(sigma_fr, eps_sigma) * F_y0_fr
F_yrl = ca.tan(alpha_rl) / ca.fmax(sigma_rl, eps_sigma) * F_y0_rl
F_yrr = ca.tan(alpha_rr) / ca.fmax(sigma_rr, eps_sigma) * F_y0_rr

# Total axle forces
F_xf = F_xfl + F_xfr
F_yf = F_yfl + F_yfr
F_xr = F_xrl + F_xrr
F_yr = F_yrl + F_yrr

# Vehicle body accelerations (equations 7-8)
a_x = (1 / m_v) * (F_xf * cos_delta - F_yf * sin_delta + F_xr - F_drag)
a_y = (1 / m_v) * (F_yf * cos_delta + F_xf * sin_delta + F_yr)

# Course-aligned accelerations (equations 9-10)
cos_beta = ca.cos(beta)
sin_beta = ca.sin(beta)
a_t = (1 / m_v) * (
    F_xf * ca.cos(delta - beta)
    - F_yf * ca.sin(delta - beta)
    + F_xr * cos_beta
    + F_yr * sin_beta
    - F_drag * cos_beta
)
a_n = (1 / m_v) * (
    F_xf * ca.sin(delta - beta)
    + F_yf * ca.cos(delta - beta)
    - F_xr * sin_beta
    + F_yr * cos_beta
    + F_drag * sin_beta
)

# Yaw moment about CG (equation 11)
M_z = (
    l_f * F_yf * cos_delta
    + l_f * F_xf * sin_delta
    - l_r * F_yr
    + w_t / 2 * ((F_yfl - F_yfr) * sin_delta + (F_xfr - F_xfl) * cos_delta + (F_xrr - F_xrl))
)

# ============================================================================
# SYSTEM DYNAMICS (Time Domain)
# ============================================================================

# Global position kinematics
X_dot = V * ca.cos(psi + beta)
Y_dot = V * ca.sin(psi + beta)
psi_dot = gamma

# Vehicle motion dynamics (equations 12-14)
V_dot = a_t
beta_dot = a_n / ca.fmax(V, 1.0) - gamma
gamma_dot = M_z / I_z

# Wheel rotational dynamics (equation 29)
omega_fl_dot = (1 / I_w) * (T_fl - r_w * F_xfl)
omega_fr_dot = (1 / I_w) * (T_fr - r_w * F_xfr)
omega_rl_dot = (1 / I_w) * (T_rl - r_w * F_xrl)
omega_rr_dot = (1 / I_w) * (T_rr - r_w * F_xrr)

# System dynamics definition
phase.dynamics(
    {
        X: X_dot,
        Y: Y_dot,
        psi: psi_dot,
        V: V_dot,
        beta: beta_dot,
        gamma: gamma_dot,
        omega_fl: omega_fl_dot,
        omega_fr: omega_fr_dot,
        omega_rl: omega_rl_dot,
        omega_rr: omega_rr_dot,
    }
)

# ============================================================================
# PHYSICAL CONSTRAINTS
# ============================================================================

# Algebraic constraints for load transfer coupling (equations 19-20)
phase.path_constraints(a_x_bar - a_x == 0, a_y_bar - a_y == 0)

# Tire friction ellipse constraints (equation 47)
phase.path_constraints(
    (F_xfl / (mu_x_max * ca.fmax(F_zfl, 50.0))) ** 2
    + (F_yfl / (mu_y_max * ca.fmax(F_zfl, 50.0))) ** 2
    <= 0.99,
    (F_xfr / (mu_x_max * ca.fmax(F_zfr, 50.0))) ** 2
    + (F_yfr / (mu_y_max * ca.fmax(F_zfr, 50.0))) ** 2
    <= 0.99,
    (F_xrl / (mu_x_max * ca.fmax(F_zrl, 50.0))) ** 2
    + (F_yrl / (mu_y_max * ca.fmax(F_zrl, 50.0))) ** 2
    <= 0.99,
    (F_xrr / (mu_x_max * ca.fmax(F_zrr, 50.0))) ** 2
    + (F_yrr / (mu_y_max * ca.fmax(F_zrr, 50.0))) ** 2
    <= 0.99,
)

# Drivetrain constraints
phase.path_constraints(
    T_t * T_b <= 0.01,  # No simultaneous traction and braking (equation 48)
    T_rl * omega_rl <= P_rl_max + 500.0,  # Motor power limits (equation 49)
    T_rr * omega_rr <= P_rr_max + 500.0,
)

# Safety constraints
phase.path_constraints(
    F_zfl >= 50.0,
    F_zfr >= 50.0,
    F_zrl >= 50.0,
    F_zrr >= 50.0,  # Minimum tire loads
    V >= 1.0,  # Minimum speed for numerical stability
)

# ============================================================================
# OBJECTIVE: Simple Point-to-Point with Minimum Control Effort
# ============================================================================

control_effort = phase.add_integral(T_t**2 + T_b**2 + 200.0 * delta**2)
problem.minimize(control_effort)

# ============================================================================
# SOLUTION SETUP
# ============================================================================

phase.mesh([10, 10], [-1.0, 0.0, 1.0])

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-4,
    max_iterations=25,
    nlp_options={
        "ipopt.print_level": 5,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-5,
        "ipopt.constr_viol_tol": 1e-5,
        "ipopt.linear_solver": "mumps",
    },
)

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

if solution.status["success"]:
    print("✓ Vehicle dynamics working correctly!")
    print(f"Mission time: {solution.phases[1]['times']['final']:.2f} seconds")
    print(f"Final position: ({solution['global_x'][-1]:.2f}, {solution['global_y'][-1]:.2f})")
    print("Target position: (10.0, 10.0)")
    print(
        f"Position error: {np.sqrt((solution['global_x'][-1] - 10.0) ** 2 + (solution['global_y'][-1] - 10.0) ** 2):.3f} m"
    )
    print(f"Final velocity: {solution['velocity'][-1]:.2f} m/s")
    print(f"Max longitudinal acceleration: {max(abs(solution['aux_long_accel'])):.2f} m/s²")
    print(f"Max lateral acceleration: {max(abs(solution['aux_lat_accel'])):.2f} m/s²")

    solution.plot()
else:
    print(f"Optimization failed: {solution.status['message']}")
