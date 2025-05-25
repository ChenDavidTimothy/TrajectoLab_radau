"""
TrajectoLab Example: Space Shuttle Reentry
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Physical constants
DEG2RAD = np.pi / 180.0
MU_EARTH = 0.14076539e17
R_EARTH = 20902900.0
S_REF = 2690.0
RHO0 = 0.002378
H_R = 23800.0
MASS = 203000.0 / 32.174

# Aerodynamic coefficients
A0, A1 = -0.20704, 0.029244
B0, B1, B2 = 0.07854, -0.61592e-2, 0.621408e-3

# Scaling factors for numerical conditioning
H_SCALE = 1e5  # altitude scaling
V_SCALE = 1e4  # velocity scaling

# Create shuttle reentry problem
problem = tl.Problem("Shuttle Reentry")

# Free final time
t = problem.time(initial=0.0)

# Scaled state variables
h_s = problem.state("altitude_scaled", initial=2.6, final=0.8, boundary=(0, None))
phi = problem.state("longitude", initial=0.0)
theta = problem.state("latitude", initial=0.0, boundary=(-89 * DEG2RAD, 89 * DEG2RAD))
v_s = problem.state("velocity_scaled", initial=2.56, final=0.25, boundary=(1e-4, None))
gamma = problem.state(
    "flight_path_angle",
    initial=-1 * DEG2RAD,
    final=-5 * DEG2RAD,
    boundary=(-89 * DEG2RAD, 89 * DEG2RAD),
)
psi = problem.state("heading_angle", initial=90 * DEG2RAD)

# Controls
alpha = problem.control("angle_of_attack", boundary=(-90 * DEG2RAD, 90 * DEG2RAD))
beta = problem.control("bank_angle", boundary=(-90 * DEG2RAD, 1 * DEG2RAD))

# Convert scaled variables to physical units
h = h_s * H_SCALE  # altitude in feet
v = v_s * V_SCALE  # velocity in ft/sec

# Physics calculations
r = R_EARTH + h
rho = RHO0 * ca.exp(-h / H_R)
g = MU_EARTH / r**2

# Aerodynamics
alpha_deg = alpha * 180.0 / np.pi
CL = A0 + A1 * alpha_deg
CD = B0 + B1 * alpha_deg + B2 * alpha_deg**2
q = 0.5 * rho * v**2
L = q * CL * S_REF
D = q * CD * S_REF

# Small epsilon to avoid division by zero
eps = 1e-10

# Scaled dynamics (divide physical rates by scaling factors)
problem.dynamics(
    {
        h_s: (v * ca.sin(gamma)) / H_SCALE,
        phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
        theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
        v_s: (-(D / MASS) - g * ca.sin(gamma)) / V_SCALE,
        gamma: (L / (MASS * v + eps)) * ca.cos(beta) + ca.cos(gamma) * (v / r - g / (v + eps)),
        psi: (L * ca.sin(beta) / (MASS * v * ca.cos(gamma) + eps))
        + (v / (r * (ca.cos(theta) + eps))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
    }
)

# Objective: maximize crossrange (latitude)
problem.minimize(-theta.final)

# Mesh and initial guess
problem.set_mesh([8] * 3, np.linspace(-1.0, 1.0, 4))

# Simple initial guess: linear interpolation
states_guess = []
controls_guess = []
for N in [8] * 3:
    t_norm = np.linspace(0, 1, N + 1)
    # Linear state trajectories
    h_traj = 2.6 + (0.8 - 2.6) * t_norm
    phi_traj = np.zeros(N + 1)
    theta_traj = np.zeros(N + 1)  # Will be optimized
    v_traj = 2.56 + (0.25 - 2.56) * t_norm
    gamma_traj = -1 * DEG2RAD + (-5 * DEG2RAD - (-1 * DEG2RAD)) * t_norm
    psi_traj = 90 * DEG2RAD * np.ones(N + 1)

    states_guess.append(np.vstack([h_traj, phi_traj, theta_traj, v_traj, gamma_traj, psi_traj]))

    # Control guess
    controls_guess.append(
        np.vstack(
            [
                np.zeros(N),  # alpha = 0
                -45 * DEG2RAD * np.ones(N),  # beta = -45 deg
            ]
        )
    )

problem.set_initial_guess(states=states_guess, controls=controls_guess, terminal_time=2000.0)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-7,
    nlp_options={
        "ipopt.print_level": 3,
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1e-7,
        "ipopt.linear_solver": "mumps",
    },
)

# Results
if solution.success:
    crossrange_deg = -solution.objective * 180.0 / np.pi
    print("Shuttle reentry solved successfully!")
    print(f"Final time: {solution.final_time:.1f} seconds")
    print(f"Crossrange: {crossrange_deg:.2f} degrees")
    solution.plot()
else:
    print(f"Failed: {solution.message}")
