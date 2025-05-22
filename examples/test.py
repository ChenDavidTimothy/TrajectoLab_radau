"""
Test script to verify proper scaling implementation fixes the hypersensitive problem.
"""

import numpy as np

import trajectolab as tl


def test_hypersensitive_proper_scaling():
    """Test hypersensitive problem with proper scaling."""
    print("=" * 80)
    print("TESTING PROPER SCALING ON HYPERSENSITIVE PROBLEM")
    print("=" * 80)

    # Create problem with new proper auto-scaling
    problem = tl.Problem("Hypersensitive Problem", auto_scaling=True)

    # Define time variable
    t = problem.time(initial=0, final=40)

    # Add state with boundary conditions
    x = problem.state("x", initial=1.5, final=1.0)

    # Add control
    u = problem.control("u")

    # Define dynamics using original symbolic expressions
    problem.dynamics({x: -(x**3) + u})

    # Define objective using original symbolic expressions
    integral_expr = 0.5 * (x**2 + u**2)
    integral_var = problem.add_integral(integral_expr)

    # Set the objective to minimize (this should now use w_0 * integral)
    problem.minimize(integral_var)

    print("\nPROPER SCALING CONFIGURATION:")
    problem.print_scaling_summary()

    # Set mesh
    polynomial_degrees = [20, 12, 20]
    mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Create initial guess
    states_guess = []
    controls_guess = []

    for N in polynomial_degrees:
        # State guess: linear from 1.5 to 1.0
        tau_points = np.linspace(-1, 1, N + 1)
        x_vals = 1.5 + (1.0 - 1.5) * (tau_points + 1) / 2
        states_guess.append(x_vals.reshape(1, -1))

        # Control guess: zeros
        controls_guess.append(np.zeros((1, N), dtype=np.float64))

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,
    )

    # Solve with fixed mesh
    print("\nSOLVING WITH PROPER SCALING...")
    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 200,
        },
    )

    if solution.success:
        print("✅ PROPER SCALING SOLUTION SUCCESSFUL!")
        print(f"📊 Objective: {solution.objective:.6f}")
        print("🎯 Expected: ~1.33 (should match non-scaled solution)")

        # Compare with expected result
        expected_objective = 1.33
        error = abs(solution.objective - expected_objective)
        relative_error = error / expected_objective * 100

        print(f"📈 Absolute error: {error:.6f}")
        print(f"📈 Relative error: {relative_error:.2f}%")

        if relative_error < 5.0:  # 5% tolerance
            print("🎉 SUCCESS: Proper scaling gives correct result!")
        else:
            print("❌ ERROR: Result still incorrect with proper scaling")

        # Show scaling information
        scaling_info = solution.get_scaling_info()
        if scaling_info.get("auto_scaling_enabled"):
            print("\n📋 SCALING DETAILS:")
            obj_scaling = scaling_info.get("objective_scaling", {})
            print(f"  Objective scaling factor (w_0): {obj_scaling.get('w_0', 1.0)}")

            var_scaling = scaling_info.get("variable_scaling_factors", {})
            for var_name, factors in var_scaling.items():
                print(
                    f"  Variable '{var_name}': v={factors.get('v', 1.0):.3e}, r={factors.get('r', 0.0):.3f}"
                )

    else:
        print(f"❌ SOLUTION FAILED: {solution.message}")

    return solution


def compare_with_original_scaling():
    """Compare results with original (broken) scaling."""
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL SCALING")
    print("=" * 80)

    # This would use the old AutoScalingManager (if still available)
    # For now, just show the theoretical comparison

    print("THEORETICAL COMPARISON:")
    print("  Original (broken) auto-scaling: ~39.9")
    print("  No scaling: ~1.33")
    print("  Proper scaling (new): Should match no scaling (~1.33)")
    print("  If proper scaling works, objective should be ~1.33, not ~39.9")


if __name__ == "__main__":
    # Test the new proper scaling
    solution = test_hypersensitive_proper_scaling()

    # Show comparison
    compare_with_original_scaling()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("If objective ≈ 1.33: ✅ Proper scaling implementation SUCCESS")
    print("If objective ≈ 39.9: ❌ Still using broken scaling approach")
    print("=" * 80)
