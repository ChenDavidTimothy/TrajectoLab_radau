import casadi as ca

import maptor as mtor


# Problem setup
problem = mtor.Problem("Brachistochrone Problem")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
x = phase.state("x", initial=0.0, final=10.0)
y = phase.state("y", initial=10.0, final=5.0)
v = phase.state("v", initial=0.0)
u = phase.control("u")

# Dynamics
g0 = 9.81
phase.dynamics({x: v * ca.sin(u), y: -v * ca.cos(u), v: g0 * ca.cos(u)})

# Objective
problem.minimize(t.final)

# Mesh and guess
phase.mesh([6, 6], [-1.0, 0.0, 1.0])


# Solve with adaptive mesh
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=30,
    min_polynomial_degree=3,
    max_polynomial_degree=12,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500},
)

# Results
if solution.status["success"]:
    print(f"Adaptive objective: {solution.status['objective']:.9f}")
    print("Literature reference: 0.312480130")
    print(f"Difference: {abs(solution.status['objective'] - 0.312480130):.2e}")
    solution.plot()

    # Compare with fixed mesh
    phase.mesh([6, 6], [-1.0, 0.0, 1.0])

    fixed_solution = mtor.solve_fixed_mesh(
        problem, nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500}
    )

    if fixed_solution.status["success"]:
        print(f"Fixed mesh objective: {fixed_solution.status['objective']:.9f}")
        print(
            f"Difference: {abs(solution.status['objective'] - fixed_solution.status['objective']):.2e}"
        )
    else:
        print(f"Fixed mesh failed: {fixed_solution.status['message']}")

else:
    print(f"Failed: {solution.status['message']}")
