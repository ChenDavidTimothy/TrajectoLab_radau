import casadi as ca
import numpy as np

import maptor as mtor


# Physical constants - EXACT PSOPT VALUES
M = 100.0  # mass (kg)
G = 9.80665  # gravity (m/s^2)
U_M = 2.5  # updraft parameter
R = 100.0  # updraft parameter (m)
C0 = 0.034  # drag coefficient constant
K = 0.069662  # drag coefficient parameter
S = 14.0  # wing area (m^2)
RHO = 1.13  # air density (kg/m^3)

# Problem setup
problem = mtor.Problem("Hang Glider")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0, final=(0.1, 200.0))
x = phase.state("x", initial=0.0, final=(0.0, 1500.0))
y = phase.state("y", initial=1000.0, final=900.0)
vx = phase.state("vx", initial=13.2275675, final=13.2275675)
vy = phase.state("vy", initial=-1.28750052, final=-1.28750052)
CL = phase.control("CL", boundary=(0.0, 1.4))

# Aerodynamic model - EXACT PSOPT FORMULATION
CD = C0 + K * CL * CL
vr = ca.sqrt(vx * vx + vy * vy)
D = 0.5 * CD * RHO * S * vr * vr
L = 0.5 * CL * RHO * S * vr * vr
X = ca.power(x / R - 2.5, 2.0)
ua = U_M * (1.0 - X) * ca.exp(-X)
Vy = vy - ua
sin_eta = Vy / vr
cos_eta = vx / vr
W = M * G

# Dynamics - EXACT PSOPT DAE IMPLEMENTATION
phase.dynamics(
    {
        x: vx,
        y: vy,
        vx: (1.0 / M) * (-L * sin_eta - D * cos_eta),
        vy: (1.0 / M) * (L * cos_eta - D * sin_eta - W),
    }
)

# Objective: maximize final horizontal distance
problem.minimize(-x.final)

# Mesh and guess
phase.mesh([8, 10, 12, 10, 8], [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])

# Initial guess - MATCHING PSOPT INITIAL CONDITIONS
states_guess = []
controls_guess = []
for N in [8, 10, 12, 10, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear interpolation matching PSOPT guess
    x_vals = 0.0 + 1250.0 * t_norm
    y_vals = 1000.0 + (900.0 - 1000.0) * t_norm
    vx_vals = 13.23 * np.ones(N + 1)
    vy_vals = -1.288 * np.ones(N + 1)

    states_guess.append(np.vstack([x_vals, y_vals, vx_vals, vy_vals]))
    controls_guess.append(np.ones((1, N)))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 105.0},
)

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=2e-1,
    max_iterations=25,
    min_polynomial_degree=4,
    max_polynomial_degree=15,
    nlp_options={
        "ipopt.max_iter": 3000,
        "ipopt.mumps_pivtol": 5e-7,
        "ipopt.mumps_mem_percent": 50000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-7,
        "ipopt.print_level": 5,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.check_derivatives_for_naninf": "yes",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-8,
    },
)

# Results
if solution.status["success"]:
    final_x = solution[(1, "x")][-1]
    flight_time = solution.phases[1]["times"]["final"]

    print(f"Maximum range: {final_x:.2f} m")
    print(f"Flight time: {flight_time:.2f} s")

    # Final state verification
    x_final = solution[(1, "x")][-1]
    y_final = solution[(1, "y")][-1]
    vx_final = solution[(1, "vx")][-1]
    vy_final = solution[(1, "vy")][-1]

    print("Final states:")
    print(f"  x: {x_final:.2f} m")
    print(f"  y: {y_final:.2f} m (target: 900.0)")
    print(f"  vx: {vx_final:.6f} m/s (target: 13.2275675)")
    print(f"  vy: {vy_final:.6f} m/s (target: -1.28750052)")

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
