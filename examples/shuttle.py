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

# Single phase using new API
phase = problem.add_phase(1)

# Free final time
t = phase.time(initial=0.0)

# Scaled state variables
h_s = phase.state("altitude_scaled", initial=2.6, final=0.8, boundary=(0, None))
phi = phase.state("longitude", initial=0.0)
theta = phase.state("latitude", initial=0.0, boundary=(-89 * DEG2RAD, 89 * DEG2RAD))
v_s = phase.state("velocity_scaled", initial=2.56, final=0.25, boundary=(1e-4, None))
gamma = phase.state(
    "flight_path_angle",
    initial=-1 * DEG2RAD,
    final=-5 * DEG2RAD,
    boundary=(-89 * DEG2RAD, 89 * DEG2RAD),
)
psi = phase.state("heading_angle", initial=90 * DEG2RAD)

# Controls
alpha = phase.control("angle_of_attack", boundary=(-90 * DEG2RAD, 90 * DEG2RAD))
beta = phase.control("bank_angle", boundary=(-90 * DEG2RAD, 1 * DEG2RAD))

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
phase.dynamics(
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
phase.set_mesh([8] * 3, np.linspace(-1.0, 1.0, 4))

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

problem.set_initial_guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 2000.0},
)

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

if solution.success:
    crossrange_deg = -solution.objective * 180.0 / np.pi
    print("Shuttle reentry solved successfully!")
    print(f"Final time: {solution.get_phase_final_time(1):.1f} seconds")
    print(f"Crossrange: {crossrange_deg:.2f} degrees")
    print()

    # Access trajectory data directly
    time_states = solution["time_states"]
    time_controls = solution["time_controls"]

    # Physical trajectory data (convert from scaled variables)
    altitude_scaled = solution["altitude_scaled"]
    velocity_scaled = solution["velocity_scaled"]
    latitude = solution["latitude"]
    longitude = solution["longitude"]
    flight_path_angle = solution["flight_path_angle"]
    heading_angle = solution["heading_angle"]

    # Control trajectories
    angle_of_attack = solution["angle_of_attack"]
    bank_angle = solution["bank_angle"]

    # Convert to physical units for engineering analysis
    altitude_ft = altitude_scaled * H_SCALE
    velocity_fps = velocity_scaled * V_SCALE

    # Engineering analysis using direct array access
    print("=== TRAJECTORY ANALYSIS ===")
    print(f"Initial altitude: {altitude_ft[0] / 1000:.1f} kft")
    print(f"Final altitude: {altitude_ft[-1] / 1000:.1f} kft")
    print(f"Altitude loss: {(altitude_ft[0] - altitude_ft[-1]) / 1000:.1f} kft")
    print()

    print(f"Initial velocity: {velocity_fps[0]:.0f} ft/s")
    print(f"Final velocity: {velocity_fps[-1]:.0f} ft/s")
    print(f"Velocity reduction: {velocity_fps[0] - velocity_fps[-1]:.0f} ft/s")
    print()

    print(f"Maximum crossrange: {np.max(np.abs(latitude)) * 180 / np.pi:.2f} degrees")
    print(f"Final longitude: {longitude[-1] * 180 / np.pi:.2f} degrees")
    print()

    # Control analysis
    alpha_deg = angle_of_attack * 180 / np.pi
    beta_deg = bank_angle * 180 / np.pi
    print(f"Angle of attack range: [{np.min(alpha_deg):.1f}, {np.max(alpha_deg):.1f}] deg")
    print(f"Bank angle range: [{np.min(beta_deg):.1f}, {np.max(beta_deg):.1f}] deg")
    print()

    # Plot the solution
    solution.plot()

else:
    print(f"Failed: {solution.message}")
