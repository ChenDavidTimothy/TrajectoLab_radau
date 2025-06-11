import casadi as ca
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Alp rider problem")
phase = problem.set_phase(1)

# Variables - exact values from C++ PSOPT
t = phase.time(initial=0.0, final=20.0)
x1 = phase.state("x1", initial=2.0, final=2.0, boundary=(-4.0, 4.0))
x2 = phase.state("x2", initial=1.0, final=3.0, boundary=(-4.0, 4.0))
x3 = phase.state("x3", initial=2.0, final=1.0, boundary=(-4.0, 4.0))
x4 = phase.state("x4", initial=1.0, final=-2.0, boundary=(-4.0, 4.0))
u1 = phase.control("u1", boundary=(-500.0, 500.0))
u2 = phase.control("u2", boundary=(-500.0, 500.0))

# Dynamics - exact from C++ PSOPT dae function
phase.dynamics(
    {
        x1: -10 * x1 + u1 + u2,
        x2: -2 * x2 + u1 + 2 * u2,
        x3: -3 * x3 + 5 * x4 + u1 - u2,
        x4: 5 * x3 - 3 * x4 + u1 + 3 * u2,
    }
)

# Auxiliary function pk(t, a, b) = exp(-b*(t-a)^2) from C++ PSOPT
def pk(t_val, a, b):
    return ca.exp(-b * (t_val - a) ** 2)

# Path constraint - exact from C++ PSOPT path[0] constraint
# x1² + x2² + x3² + x4² ≥ 3*pk(t,3,12) + 3*pk(t,6,10) + 3*pk(t,10,6) + 8*pk(t,15,4) + 0.01
terrain_function = (
    3 * pk(t, 3, 12)
    + 3 * pk(t, 6, 10)
    + 3 * pk(t, 10, 6)
    + 8 * pk(t, 15, 4)
    + 0.01
)

state_norm_squared = x1**2 + x2**2 + x3**2 + x4**2
phase.path_constraints(state_norm_squared >= terrain_function)

# Objective - exact from C++ PSOPT integrand_cost function
integrand = 100.0 * (x1**2 + x2**2 + x3**2 + x4**2) + 0.01 * (u1**2 + u2**2)
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and guess
phase.mesh([14, 14, 14], [-1.0, -1 / 3, 1 / 3, 1.0])

# Initial guess - exact replication of C++ PSOPT guess
states_guess = []
controls_guess = []
for N in [14, 14, 14]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear interpolation between initial and final conditions (linspace equivalent)
    x1_vals = 2.0 + (2.0 - 2.0) * t_norm  # linspace(2, 2, N+1)
    x2_vals = 1.0 + (3.0 - 1.0) * t_norm  # linspace(1, 3, N+1)
    x3_vals = 2.0 + (1.0 - 2.0) * t_norm  # linspace(2, 1, N+1)
    x4_vals = 1.0 + (-2.0 - 1.0) * t_norm  # linspace(1, -2, N+1)

    states_guess.append(np.vstack([x1_vals, x2_vals, x3_vals, x4_vals]))
    controls_guess.append(np.vstack([np.zeros(N), np.zeros(N)]))  # zeros(2, N)

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_integrals={1: 2000.0},
)

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=5e-4,
    max_iterations=20,
    min_polynomial_degree=6,
    max_polynomial_degree=12,
    ode_method="LSODA",
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
    print(f"Objective: {solution.status['objective']:.8f}")
    print(f"Reference: 2030.85609 (Error: {abs(solution.status['objective'] - 2030.85609) / 2030.85609 * 100:.3f}%)")

    # Final state verification
    x1_final = solution[(1, "x1")][-1]
    x2_final = solution[(1, "x2")][-1]
    x3_final = solution[(1, "x3")][-1]
    x4_final = solution[(1, "x4")][-1]

    print("Final states:")
    print(f"  x1: {x1_final:.6f} (target: 2.0)")
    print(f"  x2: {x2_final:.6f} (target: 3.0)")
    print(f"  x3: {x3_final:.6f} (target: 1.0)")
    print(f"  x4: {x4_final:.6f} (target: -2.0)")

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
