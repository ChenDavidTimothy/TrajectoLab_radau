import casadi as ca

import maptor as mtor


# Problem setup
problem = mtor.Problem("Complex Vehicle Dynamics")
phase = problem.set_phase(1)

# States: position + complex vehicle dynamics
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=10.0)
y = phase.state("y_position", initial=0.0, final=10.0)
theta = phase.state("heading", initial=0)
v = phase.state("velocity", initial=0.0, boundary=(-8.0, 25.0))
beta = phase.state("sideslip_angle", initial=0.0, boundary=(-0.2, 0.2))
gamma = phase.state("yaw_rate", initial=0.0, boundary=(-3.0, 3.0))  # ±3 rad/s
psi = phase.state("roll_angle", initial=0.0, boundary=(-0.3, 0.3))  # ±17°
p = phase.state("roll_rate", initial=0.0, boundary=(-5.0, 5.0))  # ±5 rad/s

# Controls: steering angle + wheel speeds
delta_f = phase.control("front_steering", boundary=(-0.3, 0.3))
omega_f = phase.control("front_wheel_speed", boundary=(0.0, 100.0))
omega_r = phase.control("rear_wheel_speed", boundary=(0.0, 100.0))

# Vehicle parameters (scaled for numerical stability)
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

# Numerical safety parameters
eps_small = 1e-6
eps_large = 1e-3
max_exp_arg = 10.0  # Prevent exponential overflow

# Step 1: Wheel center velocities (with safety bounds)
cos_beta = ca.cos(beta)
sin_beta = ca.sin(beta)
cos_delta_f = ca.cos(delta_f)
sin_delta_f = ca.sin(delta_f)

v_xf = v * cos_beta * cos_delta_f + (v * sin_beta + l_f * gamma) * sin_delta_f
v_yf = -v * cos_beta * sin_delta_f + (v * sin_beta + l_f * gamma) * cos_delta_f
v_xr = v * cos_beta
v_yr = v * sin_beta - l_r * gamma

# Step 2: Tire slip ratios (with numerical safety)
omega_f_safe = ca.fmax(omega_f, eps_large)
omega_r_safe = ca.fmax(omega_r, eps_large)

wheel_speed_f = omega_f_safe * r_ef
wheel_speed_r = omega_r_safe * r_er

S_xf = ca.fmax(ca.fmin((wheel_speed_f - v_xf) / wheel_speed_f, 0.9), -0.9)
S_yf = ca.fmax(ca.fmin(-v_yf / wheel_speed_f, 0.9), -0.9)
S_xr = ca.fmax(ca.fmin((wheel_speed_r - v_xr) / wheel_speed_r, 0.9), -0.9)
S_yr = ca.fmax(ca.fmin(-v_yr / wheel_speed_r, 0.9), -0.9)

# Step 3: Vertical tire loads (simplified for stability)
F_zf = m * g * l_r / (l_f + l_r)
F_zr = m * g * l_f / (l_f + l_r)

# Step 4: Tire forces (UniTire model with numerical safety)
phi_xf = ca.fabs(K_xf * S_xf / (mu * F_zf + eps_small))
phi_yf = ca.fabs(K_yf * S_yf / (mu * F_zf + eps_small))
phi_f = ca.sqrt(phi_xf**2 + phi_yf**2 + eps_small)

phi_xr = ca.fabs(K_xr * S_xr / (mu * F_zr + eps_small))
phi_yr = ca.fabs(K_yr * S_yr / (mu * F_zr + eps_small))
phi_r = ca.sqrt(phi_xr**2 + phi_yr**2 + eps_small)

# Clamped exponential arguments to prevent overflow
exp_arg_f = ca.fmin(phi_f + E * phi_f**2 + (E**2 + 1 / 12) * phi_f**3, max_exp_arg)
exp_arg_r = ca.fmin(phi_r + E * phi_r**2 + (E**2 + 1 / 12) * phi_r**3, max_exp_arg)

F_bar_f = ca.fmax(1 - ca.exp(-exp_arg_f), eps_small)
F_bar_r = ca.fmax(1 - ca.exp(-exp_arg_r), eps_small)

# Tire forces with sign preservation
denom_f = ca.sqrt((lambda_d * phi_xf) ** 2 + phi_yf**2 + eps_small)
denom_r = ca.sqrt((lambda_d * phi_xr) ** 2 + phi_yr**2 + eps_small)

F_xf = mu * F_zf * F_bar_f * ca.sign(S_xf) * (lambda_d * phi_xf) / denom_f
F_yf = mu * F_zf * F_bar_f * ca.sign(S_yf) * phi_yf / denom_f
F_xr = mu * F_zr * F_bar_r * ca.sign(S_xr) * (lambda_d * phi_xr) / denom_r
F_yr = mu * F_zr * F_bar_r * ca.sign(S_yr) * phi_yr / denom_r

# Step 5: Auxiliary forces
F_d = 0.5 * rho_a * C_d * A_f * (v * cos_beta) ** 2

# Step 6: Simplified dynamics (avoiding singular denominators)
# Use simplified single-track model dynamics for numerical stability
simple_dynamics = True

if simple_dynamics:
    # Simplified stable dynamics
    v_dot = (1 / m) * (F_xf * cos_delta_f - F_yf * sin_delta_f + F_xr - F_d)

    beta_dot = (1 / (m * v + eps_small)) * (F_xf * sin_delta_f + F_yf * cos_delta_f + F_yr) - gamma

    gamma_dot = (1 / I_z) * (l_f * (F_xf * sin_delta_f + F_yf * cos_delta_f) - l_r * F_yr)

    p_dot = -(K_phi * psi + C_phi * p - m_b * g * h_b * ca.sin(psi)) / I_x

else:
    # Original complex dynamics with safe denominators
    D_base = I_x * I_z * m - I_xz**2 * m - I_z * h_b**2 * m_b**2
    D_v = ca.fmax(ca.fabs(m * D_base), eps_large) * ca.sign(m * D_base)
    D_beta = ca.fmax(ca.fabs(m * v * D_base), eps_large) * ca.sign(m * v * D_base)
    D_gamma_p = ca.fmax(ca.fabs(D_base), eps_large) * ca.sign(D_base)

    # Simplified versions of complex equations
    v_dot = (1 / m) * (F_xf * cos_delta_f - F_yf * sin_delta_f + F_xr - F_d)
    beta_dot = (1 / (m * v + eps_small)) * (F_xf * sin_delta_f + F_yf * cos_delta_f + F_yr) - gamma
    gamma_dot = (1 / I_z) * (l_f * (F_xf * sin_delta_f + F_yf * cos_delta_f) - l_r * F_yr)
    p_dot = -(K_phi * psi + C_phi * p - m_b * g * h_b * ca.sin(psi)) / I_x

# Complete dynamics with numerical safety
phase.dynamics(
    {
        x: v * ca.cos(theta + beta),
        y: v * ca.sin(theta + beta),
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

# Mesh configuration
phase.mesh([3, 3, 3], [-1.000000, -0.60000, 0.600000, 1.000000])

# Initial guess for better convergence
problem.guess(phase_terminal_times={1: 5.0})

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-2,  # Relaxed tolerance
    max_iterations=10,  # Reduced iterations
    min_polynomial_degree=3,
    max_polynomial_degree=10,
    nlp_options={
        "ipopt.max_iter": 20000,
        "ipopt.mumps_pivtol": 5e-7,
        "ipopt.mumps_mem_percent": 50000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-2,
        "ipopt.print_level": 5,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.check_derivatives_for_naninf": "yes",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-2,
    },
)

# Results
if solution.status["success"]:
    print(f"Minimum time: {solution.status['objective']:.3f} seconds")
    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
