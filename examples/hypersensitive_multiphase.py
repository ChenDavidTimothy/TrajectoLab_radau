# examples/hypersensitive_multiphase.py
"""
TrajectoLab Example: Hypersensitive Problem using Multiphase Framework
This should produce identical results to the original single-phase version.
"""

import numpy as np

import trajectolab as tl


# Create hypersensitive problem using multiphase framework
problem = tl.Problem("Hypersensitive Problem (Multiphase)")

# Single phase definition - should behave identically to original
with problem.phase(1) as phase1:
    # Fixed time horizon
    t = phase1.time(initial=0, final=40)

    # State with boundary conditions
    x = phase1.state("x", initial=1.5, final=1.0)

    # Control (unbounded)
    u = phase1.control("u")

    # System dynamics
    phase1.dynamics({x: -(x**3) + u})

    # Objective: quadratic cost (integral within phase)
    integrand = 0.5 * (x**2 + u**2)
    integral_var = phase1.add_integral(integrand)

    # Mesh configuration for this phase
    phase1.set_mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# Global objective (minimize the integral from phase 1)
problem.minimize(integral_var)

# Solve with adaptive mesh
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=5,
    max_polynomial_degree=15,
    nlp_options={
        "ipopt.print_level": 0,
        "ipopt.max_iter": 200,
    },
)

# Results
if solution.success:
    print("Hypersensitive problem (multiphase) solved successfully!")
    print(f"Adaptive objective: {solution.objective:.6f}")
    print(f"Phase 1 duration: {solution.get_phase_duration(1):.6f}")
    print(f"Final time: {solution.get_phase_final_time(1):.6f}")

    # Access solution data using new multiphase interface
    print(f"Initial state x: {solution[(1, 'x')][0]:.6f}")
    print(f"Final state x: {solution[(1, 'x')][-1]:.6f}")
    print(f"Integral value: {solution.phase_integrals[1]:.6f}")

    # Plot using multiphase interface
    solution.plot(phase_id=1)  # Plot only phase 1

    # Also try fixed mesh for comparison
    print("\nSolving with fixed mesh...")

    # Create new problem instance for fixed mesh
    problem_fixed = tl.Problem("Hypersensitive Fixed Mesh")

    with problem_fixed.phase(1) as phase1_fixed:
        # Same problem definition
        t_fixed = phase1_fixed.time(initial=0, final=40)
        x_fixed = phase1_fixed.state("x", initial=1.5, final=1.0)
        u_fixed = phase1_fixed.control("u")
        phase1_fixed.dynamics({x_fixed: -(x_fixed**3) + u_fixed})

        integrand_fixed = 0.5 * (x_fixed**2 + u_fixed**2)
        integral_var_fixed = phase1_fixed.add_integral(integrand_fixed)

        # Refined fixed mesh
        phase1_fixed.set_mesh([20, 12, 20], [-1.0, -1 / 3, 1 / 3, 1.0])

    problem_fixed.minimize(integral_var_fixed)

    # Set detailed initial guess for fixed mesh using multiphase format
    states_guess = []
    controls_guess = []
    for N in [20, 12, 20]:
        tau = np.linspace(-1, 1, N + 1)
        x_vals = 1.5 + (1.0 - 1.5) * (tau + 1) / 2  # Linear from 1.5 to 1.0
        states_guess.append(x_vals.reshape(1, -1))
        controls_guess.append(np.zeros((1, N)))

    problem_fixed.set_initial_guess(
        phase_states={1: states_guess},
        phase_controls={1: controls_guess},
        phase_initial_times={1: 0.0},
        phase_terminal_times={1: 40.0},
        phase_integrals={1: 0.1},
    )

    fixed_solution = tl.solve_fixed_mesh(
        problem_fixed,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
        },
    )

    if fixed_solution.success:
        print(f"Fixed mesh objective: {fixed_solution.objective:.6f}")
        print(f"Difference: {abs(solution.objective - fixed_solution.objective):.2e}")

        # Compare key metrics
        print("\nComparison with original single-phase approach:")
        print(f"Adaptive - Objective: {solution.objective:.6f}")
        print(f"Fixed    - Objective: {fixed_solution.objective:.6f}")
        print("Both solutions should be identical to original single-phase results")

        fixed_solution.plot(phase_id=1)
    else:
        print(f"Fixed mesh failed: {fixed_solution.message}")

else:
    print(f"Failed: {solution.message}")

print("\n" + "=" * 60)
print("VERIFICATION: This multiphase solution should be identical")
print("to the original single-phase hypersensitive.py results.")
print("=" * 60)
