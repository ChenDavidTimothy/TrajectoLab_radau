import casadi as ca

import maptor as mtor


# Problem setup
problem = mtor.Problem("Complex Vehicle Dynamics")
phase = problem.set_phase(1)

# States with more conservative bounds
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=10.0)
y = phase.state("y_position", initial=0.0, final=10.0)
theta = phase.state("heading", initial=0)
v = phase.state("velocity", initial=5.0, boundary=(1.0, 20.0))  # Avoid zero velocity
beta = phase.state("sideslip_angle", initial=0.0, boundary=(-0.15, 0.15))
gamma = phase.state("yaw_rate", initial=0.0, boundary=(-2.0, 2.0))
psi = phase.state("roll_angle", initial=0.0, boundary=(-0.2, 0.2))
p = phase.state("roll_rate", initial=0.0, boundary=(-3.0, 3.0))

# Controls with conservative bounds
delta_f = phase.control("front_steering", boundary=(-0.25, 0.25))
omega_f = phase.control("front_wheel_speed", boundary=(5.0, 80.0))  # Avoid zero
omega_r = phase.control("rear_wheel_speed", boundary=(5.0, 80.0))

# Vehicle parameters
m = 1126.7
m_b = 1111.0
I_z = 2038.0
I_x = 550.0
I_xz = 50.0
l_f = 1.265
l_r = 1.335
h_b = 0.136
h_g = 0.518
K_phi = 60000.0
C_phi = 6000.0
r_ef = 0.31
r_er = 0.31
mu = 0.65
K_yf = 110000.0
K_yr = 105000.0
K_xf = 200000.0
K_xr = 200000.0
E = 1.0
lambda_d = 1.0
A_f = 1.6
C_d = 0.32
rho_a = 1.225
g = 9.8

# Enhanced numerical safety parameters
eps_small = 1e-4  # Increased from 1e-6
eps_large = 1e-2  # Increased from 1e-3
max_exp_arg = 5.0  # Reduced from 10.0
slip_limit = 0.7  # Reduced from 0.9

# Step 1: Wheel center velocities with bounded trigonometric functions
cos_beta = ca.cos(ca.fmax(ca.fmin(beta, 0.15), -0.15))
sin_beta = ca.sin(ca.fmax(ca.fmin(beta, 0.15), -0.15))
cos_delta_f = ca.cos(ca.fmax(ca.fmin(delta_f, 0.25), -0.25))
sin_delta_f = ca.sin(ca.fmax(ca.fmin(delta_f, 0.25), -0.25))

v_safe = ca.fmax(v, 1.0)  # Ensure positive velocity

v_xf = v_safe * cos_beta * cos_delta_f + (v_safe * sin_beta + l_f * gamma) * sin_delta_f
v_yf = -v_safe * cos_beta * sin_delta_f + (v_safe * sin_beta + l_f * gamma) * cos_delta_f
v_xr = v_safe * cos_beta
v_yr = v_safe * sin_beta - l_r * gamma

# Step 2: Tire slip ratios with enhanced safety
omega_f_safe = ca.fmax(omega_f, 5.0)
omega_r_safe = ca.fmax(omega_r, 5.0)

wheel_speed_f = omega_f_safe * r_ef
wheel_speed_r = omega_r_safe * r_er

# More conservative slip ratio calculation
S_xf = ca.fmax(
    ca.fmin((wheel_speed_f - v_xf) / (wheel_speed_f + eps_large), slip_limit), -slip_limit
)
S_yf = ca.fmax(ca.fmin(-v_yf / (wheel_speed_f + eps_large), slip_limit), -slip_limit)
S_xr = ca.fmax(
    ca.fmin((wheel_speed_r - v_xr) / (wheel_speed_r + eps_large), slip_limit), -slip_limit
)
S_yr = ca.fmax(ca.fmin(-v_yr / (wheel_speed_r + eps_large), slip_limit), -slip_limit)

# Step 3: Vertical tire loads (simplified for stability)
F_zf = m * g * l_r / (l_f + l_r)
F_zr = m * g * l_f / (l_f + l_r)

# Step 4: Tire forces with enhanced numerical conditioning
mu_safe = ca.fmax(mu, 0.1)
F_zf_safe = ca.fmax(F_zf, 100.0)  # Minimum normal force
F_zr_safe = ca.fmax(F_zr, 100.0)

phi_xf = ca.fabs(K_xf * S_xf) / (mu_safe * F_zf_safe + eps_large)
phi_yf = ca.fabs(K_yf * S_yf) / (mu_safe * F_zf_safe + eps_large)
phi_f = ca.sqrt(phi_xf**2 + phi_yf**2 + eps_small)

phi_xr = ca.fabs(K_xr * S_xr) / (mu_safe * F_zr_safe + eps_large)
phi_yr = ca.fabs(K_yr * S_yr) / (mu_safe * F_zr_safe + eps_large)
phi_r = ca.sqrt(phi_xr**2 + phi_yr**2 + eps_small)

# Clamped exponential with more conservative limits
exp_arg_f = ca.fmax(
    ca.fmin(phi_f + E * phi_f**2 + (E**2 + 1 / 12) * phi_f**3, max_exp_arg), -max_exp_arg
)
exp_arg_r = ca.fmax(
    ca.fmin(phi_r + E * phi_r**2 + (E**2 + 1 / 12) * phi_r**3, max_exp_arg), -max_exp_arg
)

F_bar_f = ca.fmax(1 - ca.exp(-exp_arg_f), eps_small)
F_bar_r = ca.fmax(1 - ca.exp(-exp_arg_r), eps_small)


# Smooth sign function replacement to avoid discontinuities
def smooth_sign(x, smooth_param=0.01):
    return ca.tanh(x / smooth_param)


denom_f = ca.sqrt((lambda_d * phi_xf) ** 2 + phi_yf**2 + eps_small)
denom_r = ca.sqrt((lambda_d * phi_xr) ** 2 + phi_yr**2 + eps_small)

F_xf = mu_safe * F_zf_safe * F_bar_f * smooth_sign(S_xf) * (lambda_d * phi_xf) / denom_f
F_yf = mu_safe * F_zf_safe * F_bar_f * smooth_sign(S_yf) * phi_yf / denom_f
F_xr = mu_safe * F_zr_safe * F_bar_r * smooth_sign(S_xr) * (lambda_d * phi_xr) / denom_r
F_yr = mu_safe * F_zr_safe * F_bar_r * smooth_sign(S_yr) * phi_yr / denom_r

# Step 5: Auxiliary forces
F_d = 0.5 * rho_a * C_d * A_f * (v_safe * cos_beta) ** 2

# Step 6: Complex dynamics with enhanced numerical safety
# Common terms with better conditioning
I_x_I_z_m = I_x * I_z * m
I_xz_sq_m = I_xz**2 * m
I_z_hb_sq_mb_sq = I_z * h_b**2 * m_b**2

# Base denominator with safety
D_base = I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq

# Enhanced denominator safety with minimum threshold
min_denom = 1e3  # Minimum acceptable denominator magnitude
D_v = m * (-I_x_I_z_m + I_xz_sq_m + I_z_hb_sq_mb_sq)
D_beta = m * v_safe * (-I_x_I_z_m + I_xz_sq_m + I_z_hb_sq_mb_sq)
D_gamma_p = -I_x_I_z_m + I_xz_sq_m + I_z_hb_sq_mb_sq


# Robust denominator conditioning
def safe_denom(d, min_val=min_denom):
    d_abs = ca.fabs(d)
    d_safe = ca.fmax(d_abs, min_val)
    return d_safe * ca.sign(d + eps_small)  # Add small value to avoid exact zero


D_v_safe = safe_denom(D_v)
D_beta_safe = safe_denom(D_beta)
D_gamma_p_safe = safe_denom(D_gamma_p)

# Bounded trigonometric functions for roll dynamics
sin_psi = ca.sin(ca.fmax(ca.fmin(psi, 0.2), -0.2))
cos_psi = ca.cos(ca.fmax(ca.fmin(psi, 0.2), -0.2))

# Complex dynamics implementation with numerical conditioning
v_dot = (1 / D_v_safe) * (
    C_phi * I_z * h_b * m * m_b * p * sin_beta
    + F_d * cos_beta * (I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq)
    - F_xf * ca.cos(beta - delta_f) * (I_x_I_z_m - I_xz_sq_m)
    - F_xf * I_xz * h_b * l_f * m * m_b * sin_beta * sin_delta_f
    + F_xf * I_z * h_b**2 * m_b**2 * cos_beta * cos_delta_f
    - F_xr * cos_beta * (I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq)
    - F_yf * ca.sin(beta - delta_f) * (I_x_I_z_m - I_xz_sq_m)
    - F_yf * I_xz * h_b * l_f * m * m_b * sin_beta * cos_delta_f
    - F_yf * I_z * h_b**2 * m_b**2 * sin_delta_f * cos_beta
    - F_yr * sin_beta * (I_x_I_z_m - I_xz_sq_m - I_xz * h_b * l_r * m * m_b)
    + gamma * h_b * m * m_b * p * cos_beta * (I_x * I_z - I_xz**2 - I_z * h_b**2 * m_b**2)
    + I_z * h_b * m * m_b * sin_beta * (K_phi * psi - g * h_b * m_b * sin_psi)
)

beta_dot = (1 / D_beta_safe) * (
    C_phi * I_z * h_b * m * m_b * p * cos_beta
    - F_d * sin_beta * (I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq)
    + F_xf * ca.sin(beta - delta_f) * (I_x_I_z_m - I_xz_sq_m)
    - F_xf * I_xz * h_b * l_f * m * m_b * cos_beta * sin_delta_f
    - F_xf * I_z * h_b**2 * m_b**2 * sin_beta * cos_delta_f
    + F_xr * sin_beta * (I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq)
    - F_yf * ca.cos(beta - delta_f) * (I_x_I_z_m - I_xz_sq_m)
    - F_yf * I_xz * h_b * l_f * m * m_b * cos_beta * cos_delta_f
    + F_yf * I_z * h_b**2 * m_b**2 * sin_beta * sin_delta_f
    - F_yr * cos_beta * (I_x_I_z_m - I_xz_sq_m - I_xz * h_b * l_r * m * m_b)
    - gamma * h_b * m * m_b * p * sin_beta * (I_x * I_z - I_xz**2 - I_z * h_b**2 * m_b**2)
    + I_z * h_b * m * m_b * cos_beta * (K_phi * psi - g * h_b * m_b * sin_psi)
    - gamma * m * v_safe * (I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq)
)

gamma_dot = (1 / D_gamma_p_safe) * (
    C_phi * I_xz * m * p
    - F_xf * sin_delta_f * (I_x * l_f * m + I_xz * h_b * m_b - h_b**2 * l_f * m_b**2)
    - F_yf * cos_delta_f * (I_x * l_f * m + I_xz * h_b * m_b - h_b**2 * l_f * m_b**2)
    + F_yr * (I_x * l_r * m - I_xz * h_b * m_b - h_b**2 * l_r * m_b**2)
    + I_xz * m * (K_phi * psi - g * h_b * m_b * sin_psi)
)

p_dot = (1 / D_gamma_p_safe) * (
    C_phi * I_z * m * p
    - F_xf * sin_delta_f * (I_xz * l_f * m + I_z * h_b * m_b)
    - F_yf * cos_delta_f * (I_xz * l_f * m + I_z * h_b * m_b)
    + F_yr * (I_xz * l_r * m - I_z * h_b * m_b)
    + I_z * m * (K_phi * psi - g * h_b * m_b * sin_psi)
)

# Complete dynamics
phase.dynamics(
    {
        x: v_safe * ca.cos(theta + beta),
        y: v_safe * ca.sin(theta + beta),
        theta: gamma,
        v: v_dot,
        beta: beta_dot,
        gamma: gamma_dot,
        psi: p,
        p: p_dot,
    }
)

# Objective: minimize time
problem.minimize(t.final)

# Conservative mesh and solver settings
phase.mesh([8, 8], [-1.0, 0.0, 1.0])

problem.guess(phase_terminal_times={1: 8.0})

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-2,
    max_iterations=1,
    min_polynomial_degree=2,
    max_polynomial_degree=4,
    nlp_options={
        "ipopt.max_iter": 3000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-2,
        "ipopt.print_level": 5,
        "ipopt.tol": 1e-2,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.check_derivatives_for_naninf": "yes",
        "ipopt.bound_relax_factor": 1e-6,
    },
)

# Results
if solution.status["success"]:
    print(f"Minimum time: {solution.status['objective']:.3f} seconds")
    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
