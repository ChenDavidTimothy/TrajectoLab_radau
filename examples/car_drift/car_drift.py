import casadi as ca
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Direct Force Control Vehicle Dynamics")
phase = problem.set_phase(1)

# States - unchanged from document
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=10.0)
y = phase.state("y_position", initial=0.0, final=10.0)
theta = phase.state("heading", initial=np.pi / 7)
v = phase.state("velocity", initial=0.1, final=0, boundary=(-20.0, 20.0))
beta = phase.state("sideslip_angle", initial=0.0, boundary=(-0.15, 0.15))
gamma = phase.state("yaw_rate", initial=0.0, boundary=(-2.0, 2.0))
psi = phase.state("roll_angle", initial=0.0, boundary=(-0.2, 0.2))
p = phase.state("roll_rate", initial=0.0, boundary=(-3.0, 3.0))

# Controls - changed to direct force control per document Section 2.1
delta_f = phase.control("front_steering", boundary=(-0.25, 0.25))
F_xf = phase.control("front_long_force", boundary=(-5000.0, 5000.0))  # N
F_xr = phase.control("rear_long_force", boundary=(-5000.0, 5000.0))  # N

# Vehicle parameters from document Table 1 - exact values
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
mu = 0.65
K_yf = 110000.0  # Front cornering stiffness
K_yr = 105000.0  # Rear cornering stiffness
A_f = 1.6
C_d = 0.32
rho_a = 1.225
g = 9.8

# Numerical safety parameters
eps_small = 1e-4
singularity_threshold = 1e-3

# Step 1: Calculate wheel center velocities (unchanged per document)
cos_beta = ca.cos(ca.fmax(ca.fmin(beta, 0.15), -0.15))
sin_beta = ca.sin(ca.fmax(ca.fmin(beta, 0.15), -0.15))
cos_delta_f = ca.cos(ca.fmax(ca.fmin(delta_f, 0.25), -0.25))
sin_delta_f = ca.sin(ca.fmax(ca.fmin(delta_f, 0.25), -0.25))

v_xf = v * cos_beta * cos_delta_f + (v * sin_beta + l_f * gamma) * sin_delta_f
v_yf = -v * cos_beta * sin_delta_f + (v * sin_beta + l_f * gamma) * cos_delta_f
v_xr = v * cos_beta
v_yr = v * sin_beta - l_r * gamma

# Step 2: Calculate tire slip angles (NEW - document Section 4.2)
# Safe atan2 calculation with singularity handling
v_xf_safe = ca.fmax(ca.fabs(v_xf), singularity_threshold) * ca.sign(v_xf + eps_small)
v_xr_safe = ca.fmax(ca.fabs(v_xr), singularity_threshold) * ca.sign(v_xr + eps_small)

alpha_f = ca.atan2(v_yf, v_xf_safe)
alpha_r = ca.atan2(v_yr, v_xr_safe)

# Step 3: Calculate vertical tire loads (document Section 4.3)
F_d = 0.5 * rho_a * C_d * A_f * (v * cos_beta) * ca.fabs(v * cos_beta)

# Total longitudinal force calculation
sum_F_x = F_xf * cos_delta_f + F_xr - F_d  # Note: F_yf term handled iteratively

# Vertical load distribution (document equations)
wheelbase = l_f + l_r
wheelbase_safe = ca.fmax(wheelbase, 0.1)

# Force and moment balance
F_zf = ca.fmax((m * g * l_r - sum_F_x * h_g) / wheelbase_safe, 50.0)
F_zr = ca.fmax(m * g - F_zf, 50.0)

# Ensure physical constraints
total_weight = m * g
F_zf = ca.fmax(ca.fmin(F_zf, total_weight * 0.95), total_weight * 0.05)
F_zr = total_weight - F_zf

# Step 4: Calculate lateral tire forces using friction ellipse (document Section 4.4)
F_zf_safe = ca.fmax(F_zf, 50.0)
F_zr_safe = ca.fmax(F_zr, 50.0)

# Maximum available lateral force (friction ellipse)
mu_F_zf = mu * F_zf_safe
mu_F_zr = mu * F_zr_safe

F_yf_max = ca.sqrt(ca.fmax(mu_F_zf**2 - F_xf**2, eps_small))
F_yr_max = ca.sqrt(ca.fmax(mu_F_zr**2 - F_xr**2, eps_small))

# Demanded lateral force (linear tire model)
F_yf_demand = -K_yf * alpha_f
F_yr_demand = -K_yr * alpha_r

# Saturated lateral forces (document Section 4.4.3)
F_yf = ca.fmax(-F_yf_max, ca.fmin(F_yf_max, F_yf_demand))
F_yr = ca.fmax(-F_yr_max, ca.fmin(F_yr_max, F_yr_demand))

# Step 5: Final state derivatives (unchanged per document Section 5)
I_x_I_z_m = I_x * I_z * m
I_xz_sq_m = I_xz**2 * m
I_z_hb_sq_mb_sq = I_z * h_b**2 * m_b**2

# Denominators with safety checks
min_denom = 1e3
D_v = m * (-I_x_I_z_m + I_xz_sq_m + I_z_hb_sq_mb_sq)
D_beta = m * v * (-I_x_I_z_m + I_xz_sq_m + I_z_hb_sq_mb_sq)
D_gamma_p = -I_x_I_z_m + I_xz_sq_m + I_z_hb_sq_mb_sq


def safe_denom(d, min_val=min_denom):
    d_abs = ca.fabs(d)
    d_safe = ca.fmax(d_abs, min_val)
    return d_safe * ca.sign(d + eps_small)


D_v_safe = safe_denom(D_v)
D_beta_safe = safe_denom(D_beta)
D_gamma_p_safe = safe_denom(D_gamma_p)

# Roll angle trigonometry with bounds
sin_psi = ca.sin(ca.fmax(ca.fmin(psi, 0.2), -0.2))

# Document-exact dynamics equations (unchanged)
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
    - gamma * m * v * (I_x_I_z_m - I_xz_sq_m - I_z_hb_sq_mb_sq)
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

# Dynamics definition (position equations unchanged)
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

# Mesh and solver settings
phase.mesh([10, 10], [-1.0, 0.0, 1.0])

problem.guess(phase_terminal_times={1: 3.0})

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=5e-2,
    max_iterations=20,
    min_polynomial_degree=3,
    max_polynomial_degree=10,
    nlp_options={
        "ipopt.max_iter": 3000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-2,
        "ipopt.print_level": 5,
        "ipopt.tol": 1e-2,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
    },
)

# Results
if solution.status["success"]:
    print(f"Minimum time: {solution.status['objective']:.3f} seconds")
    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
