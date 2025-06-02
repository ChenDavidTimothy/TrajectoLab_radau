"""
TrajectoLab Example: Two-Phase Constrained Optimal Control Problem

Minimize: J = 5(x₀(tF)² + x₁(tF)²) + ∫₀^tF 0 dt
Subject to:
- Dynamics: ẋ₀ = x₁, ẋ₁ = u - 0.1(1 + 2x₀²)x₁
- Phase 1: t ∈ [0, 1] with elliptical path constraint and bounds
- Phase 2: t ∈ [1, 2.9] with state continuity
"""

import casadi as ca
import numpy as np

import trajectolab as tl


def solve_two_phase_constrained_problem():
    """
    Implement two-phase constrained optimal control problem.

    This problem demonstrates:
    - Terminal cost minimization of final state values
    - Elliptical path constraints (feasible region outside ellipse)
    - Two-phase structure with state continuity
    - Mixed bounded/unbounded control variables

    Returns:
        Problem: Configured TrajectoLab problem ready for solving
    """

    problem = tl.Problem("Two-Phase Constrained Control")

    # ========================================================================
    # Phase 1: [0, 1] with constraints
    # ========================================================================
    phase1 = problem.set_phase(1)

    # Time variable - fixed duration
    t1 = phase1.time(initial=0.0, final=1.0)

    # State variables with initial conditions
    x0_1 = phase1.state("x0", initial=1.0)
    x1_1 = phase1.state("x1", initial=1.0, boundary=(-0.8, None))  # x₁ ≥ -0.8

    # Control variable with bounds
    u1 = phase1.control("u", boundary=(-1.0, 1.0))  # -1 ≤ u ≤ 1

    # System dynamics
    phase1.dynamics({
        x0_1: x1_1,  # ẋ₀ = x₁
        x1_1: u1 - 0.1 * (1 + 2 * x0_1**2) * x1_1  # ẋ₁ = u - 0.1(1 + 2x₀²)x₁
    })

    # Elliptical path constraint: 1 - 9(x₀ - 1)² - ((x₁ - 0.4)/0.3)² ≤ 0
    # This creates a feasible region OUTSIDE the ellipse centered at (1, 0.4)
    elliptical_constraint = 1 - 9 * (x0_1 - 1)**2 - ((x1_1 - 0.4) / 0.3)**2
    phase1.subject_to(elliptical_constraint <= 0)

    # Mesh configuration for Phase 1
    phase1.mesh([6, 6], [-1.0, 0.0, 1.0])

    # ========================================================================
    # Phase 2: [1, 2.9] with state continuity
    # ========================================================================
    phase2 = problem.set_phase(2)

    # Time variable continuing from Phase 1
    t2 = phase2.time(initial=1.0, final=2.9)

    # State variables with continuity from Phase 1
    x0_2 = phase2.state("x0", initial=x0_1.final)
    x1_2 = phase2.state("x1", initial=x1_1.final)

    # Control variable (unconstrained in Phase 2: u ∈ ℝ)
    u2 = phase2.control("u")

    # Same system dynamics
    phase2.dynamics({
        x0_2: x1_2,  # ẋ₀ = x₁
        x1_2: u2 - 0.1 * (1 + 2 * x0_2**2) * x1_2  # ẋ₁ = u - 0.1(1 + 2x₀²)x₁
    })

    # Mesh configuration for Phase 2
    phase2.mesh([8, 8], [-1.0, 0.0, 1.0])

    # ========================================================================
    # Objective Function
    # ========================================================================

    # Minimize final state values: J = 5(x₀(tF)² + x₁(tF)²)
    # Note: The integral ∫₀^tF 0 dt = 0, so only terminal cost matters
    objective_expr = 5 * (x0_2.final**2 + x1_2.final**2)
    problem.minimize(objective_expr)

    return problem


def create_initial_guess():
    """
    Create reasonable initial guess for the two-phase constrained problem.

    Strategy:
    - Phase 1: Navigate around the elliptical constraint while satisfying bounds
    - Phase 2: Drive states toward origin to minimize terminal cost
    """

    # Phase 1: t ∈ [0, 1] - navigate around elliptical constraint
    states_p1 = []
    controls_p1 = []

    for N in [6, 6]:  # Match mesh configuration
        # States: N+1 points per interval
        tau_states = np.linspace(-1, 1, N + 1)
        t_norm_states = (tau_states + 1) / 2  # Map to [0, 1]

        # Initial guess: move away from ellipse center (1, 0.4) toward feasible region
        # Start at (1, 1) and move to avoid ellipse
        x0_vals = 1.0 + 0.2 * t_norm_states  # Move slightly in x0 direction
        x1_vals = 1.0 - 0.3 * t_norm_states  # Move away from ellipse center in x1

        states_p1.append(np.array([x0_vals, x1_vals]))

        # Controls: N points per interval (not N+1)
        t_norm_controls = np.linspace(0, 1, N)
        u_vals = 0.3 * np.sin(np.pi * t_norm_controls)
        controls_p1.append(np.array([u_vals]))

    # Phase 2: t ∈ [1, 2.9] - drive states toward origin
    states_p2 = []
    controls_p2 = []

    for N in [8, 8]:  # Match mesh configuration
        # States: N+1 points per interval
        tau_states = np.linspace(-1, 1, N + 1)
        t_norm_states = (tau_states + 1) / 2  # Map to [0, 1] over phase duration

        # Continue from Phase 1 end and drive states toward origin
        # Estimate Phase 1 end values
        x0_end_p1 = 1.2  # Estimated from Phase 1 trajectory
        x1_end_p1 = 0.7  # Estimated from Phase 1 trajectory

        # Linear trajectory toward origin
        x0_vals = x0_end_p1 * (1 - 0.8 * t_norm_states)  # Drive toward small x0
        x1_vals = x1_end_p1 * (1 - 0.9 * t_norm_states)  # Drive toward small x1

        states_p2.append(np.array([x0_vals, x1_vals]))

        # Controls: N points per interval (not N+1)
        t_norm_controls = np.linspace(0, 1, N)
        u_vals = -1.0 + 0.5 * t_norm_controls  # Start strong negative, reduce magnitude
        controls_p2.append(np.array([u_vals]))

    return {
        "phase_states": {1: states_p1, 2: states_p2},
        "phase_controls": {1: controls_p1, 2: controls_p2},
        "phase_initial_times": {1: 0.0, 2: 1.0},
        "phase_terminal_times": {1: 1.0, 2: 2.9}
    }


def main():
    """Main execution function for two-phase constrained problem."""

    print("=" * 80)
    print("TrajectoLab Implementation: Two-Phase Constrained Optimal Control")
    print("Terminal Cost Minimization with Elliptical Path Constraints")
    print("=" * 80)

    # Create and configure the problem
    problem = solve_two_phase_constrained_problem()

    # Set initial guess to aid convergence
    initial_guess = create_initial_guess()
    problem.guess(**initial_guess)

    print("Problem Classification: Two-phase terminal cost minimization")
    print("Objective: Minimize J = 5(x₀(tF)² + x₁(tF)²)")
    print("Dynamics: ẋ₀ = x₁, ẋ₁ = u - 0.1(1 + 2x₀²)x₁")
    print("Phase 1: t ∈ [0, 1] with elliptical path constraint")
    print("  - Constraint: 1 - 9(x₀-1)² - ((x₁-0.4)/0.3)² ≤ 0")
    print("  - State bound: x₁ ≥ -0.8")
    print("  - Control bound: -1 ≤ u ≤ 1")
    print("Phase 2: t ∈ [1, 2.9] with state continuity and unconstrained control")
    print("Initial conditions: x₀(0) = 1, x₁(0) = 1")

    print("\nSolving two-phase constrained problem...")

    # Solve with adaptive mesh refinement
    solution = tl.solve_adaptive(
        problem,
        error_tolerance=1e-6,
        max_iterations=25,
        min_polynomial_degree=3,
        max_polynomial_degree=12,
        nlp_options={
            "ipopt.print_level": 5,
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-8,
            "ipopt.constr_viol_tol": 1e-8,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.linear_solver": "mumps"
        }
    )

    # Display results
    if solution.success:
        print("\n" + "=" * 60)
        print("SOLUTION SUCCESSFUL!")
        print("=" * 60)

        objective_value = solution.objective
        print(f"Objective Value: {objective_value:.8f}")

        # Extract final state values
        try:
            x0_final = solution[(2, "x0")][-1]
            x1_final = solution[(2, "x1")][-1]

            print(f"\nFinal state values:")
            print(f"x₀(tF) = {x0_final:.6f}")
            print(f"x₁(tF) = {x1_final:.6f}")
            print(f"x₀²(tF) + x₁²(tF) = {x0_final**2 + x1_final**2:.6f}")
            print(f"5(x₀²(tF) + x₁²(tF)) = {5*(x0_final**2 + x1_final**2):.6f}")

            # Verify objective calculation
            expected_obj = 5 * (x0_final**2 + x1_final**2)
            print(f"Objective verification: {expected_obj:.8f} vs {objective_value:.8f}")

            # Check constraint satisfaction in Phase 1
            x0_p1 = solution[(1, "x0")]
            x1_p1 = solution[(1, "x1")]

            # Evaluate elliptical constraint: should be ≤ 0
            constraint_vals = 1 - 9*(x0_p1 - 1)**2 - ((x1_p1 - 0.4)/0.3)**2
            max_violation = np.max(constraint_vals)

            print(f"\nConstraint satisfaction:")
            print(f"Max elliptical constraint value: {max_violation:.6e} (should be ≤ 0)")
            print(f"Min x₁ value in Phase 1: {np.min(x1_p1):.6f} (should be ≥ -0.8)")

            if max_violation > 1e-6:
                print(f"⚠️  Elliptical constraint violated by {max_violation:.2e}")
            else:
                print("✅ Elliptical constraint satisfied")

            if np.min(x1_p1) < -0.8 - 1e-6:
                print(f"⚠️  State bound x₁ ≥ -0.8 violated")
            else:
                print("✅ State bound x₁ ≥ -0.8 satisfied")

            # Control analysis
            u1_data = solution[(1, "u")]
            u2_data = solution[(2, "u")]
            print(f"\nControl analysis:")
            print(f"Phase 1 control range: [{np.min(u1_data):.3f}, {np.max(u1_data):.3f}] (bounds: [-1, 1])")
            print(f"Phase 2 control range: [{np.min(u2_data):.3f}, {np.max(u2_data):.3f}] (unconstrained)")

            # Phase timing
            print(f"\nPhase timing:")
            print(f"Phase 1 duration: {solution.get_phase_duration(1):.3f} (fixed at 1.0)")
            print(f"Phase 2 duration: {solution.get_phase_duration(2):.3f} (fixed at 1.9)")
            print(f"Total mission time: {solution.get_total_mission_time():.3f}")

        except KeyError as e:
            print(f"Could not extract all solution data: {e}")

        # Plot the solution
        print("\nGenerating solution plots...")
        solution.plot(show_phase_boundaries=True)

        return solution

    else:
        print("\n" + "=" * 60)
        print("SOLUTION FAILED!")
        print("=" * 60)
        print(f"Message: {solution.message}")
        return None


if __name__ == "__main__":
    main()
