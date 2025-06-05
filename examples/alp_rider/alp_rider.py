import casadi as ca
import numpy as np

import trajectolab as tl


# Problem setup
problem = tl.Problem("Alp Rider")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0, final=20.0)
y1 = phase.state("y1", initial=2.0, final=2.0)
y2 = phase.state("y2", initial=1.0, final=3.0)
y3 = phase.state("y3", initial=2.0, final=1.0)
y4 = phase.state("y4", initial=1.0, final=-2.0)
u1 = phase.control("u1")
u2 = phase.control("u2")

# Dynamics
phase.dynamics(
    {
        y1: -10 * y1 + u1 + u2,
        y2: -2 * y2 + u1 + 2 * u2,
        y3: -3 * y3 + 5 * y4 + u1 - u2,
        y4: 5 * y3 - 3 * y4 + u1 + 3 * u2,
    }
)


# Path constraint: y1² + y2² + y3² + y4² ≥ terrain following function
def p_function(t_val, a, b):
    return ca.exp(-b * (t_val - a) ** 2)


terrain_function = (
    3 * p_function(t, 3, 12)
    + 3 * p_function(t, 6, 10)
    + 3 * p_function(t, 10, 6)
    + 8 * p_function(t, 15, 4)
    + 0.01
)

state_norm_squared = y1**2 + y2**2 + y3**2 + y4**2
phase.path_constraints(state_norm_squared >= terrain_function)

# Objective
integrand = 100 * (y1**2 + y2**2 + y3**2 + y4**2) + 0.01 * (u1**2 + u2**2)
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and guess
phase.mesh([14, 14, 14], [-1.0, -1 / 3, 1 / 3, 1.0])

states_guess = []
controls_guess = []
for N in [14, 14, 14]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear interpolation between initial and final conditions
    y1_vals = 2.0 + (2.0 - 2.0) * t_norm  # 2 to 2
    y2_vals = 1.0 + (3.0 - 1.0) * t_norm  # 1 to 3
    y3_vals = 2.0 + (1.0 - 2.0) * t_norm  # 2 to 1
    y4_vals = 1.0 + (-2.0 - 1.0) * t_norm  # 1 to -2

    states_guess.append(np.vstack([y1_vals, y2_vals, y3_vals, y4_vals]))
    controls_guess.append(np.vstack([np.zeros(N), np.zeros(N)]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_integrals={1: 2000.0},
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-4,
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
        "ipopt.print_level": 0,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.check_derivatives_for_naninf": "yes",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-8,
    },
)

# Results
if solution.status["success"]:
    print(f"Objective: {solution.status['objective']:.5f}")
    print(
        f"Reference: 2030.85609 (Error: {(abs(solution.status['objective'] - 2030.85609) / 2030.85609) * 100:.3f}%)"
    )

    # Final state values
    y1_final = solution[(1, "y1")][-1]
    y2_final = solution[(1, "y2")][-1]
    y3_final = solution[(1, "y3")][-1]
    y4_final = solution[(1, "y4")][-1]
    print(
        f"Final states: y1={y1_final:.6f}, y2={y2_final:.6f}, y3={y3_final:.6f}, y4={y4_final:.6f}"
    )

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
