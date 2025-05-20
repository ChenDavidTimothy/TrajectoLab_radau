"""
Simple Scaling Test - Verification of Variable Scaling Implementation

This creates a simple optimal control problem with known large scale differences
to verify that the scaling implementation works correctly.
"""

import numpy as np

import trajectolab as tl


def test_simple_scaling():
    """Test scaling with a simple problem that has large scale differences."""
    print("=" * 60)
    print("SIMPLE SCALING VERIFICATION TEST")
    print("=" * 60)

    # Create a simple problem with large scale differences
    problem = tl.Problem("Simple Scaling Test")

    # Time
    t = problem.time(initial=0.0, final=1.0)

    # State with large scale (order 10^6)
    # This could represent something like position in meters over long distances
    x1 = problem.state("x1", initial=1000000.0, final=2000000.0, lower=0.0, upper=5000000.0)

    # State with small scale (order 10^0)
    # This could represent something like normalized quantities
    x2 = problem.state("x2", initial=1.0, final=2.0, lower=0.1, upper=5.0)

    # Control with medium scale (order 10^3)
    u = problem.control("u", lower=-10000.0, upper=10000.0)

    # Simple dynamics
    problem.dynamics(
        {
            x1: u,  # x1_dot = u
            x2: u / 1000000,  # x2_dot = u/1000000 (creates coupling between scales)
        }
    )

    # Simple objective - minimize control effort
    integral_expr = 0.5 * u**2
    objective_integral = problem.add_integral(integral_expr)
    problem.minimize(objective_integral)

    # Test mesh
    degrees = [8, 8]
    mesh = [-1.0, 0.0, 1.0]

    # Test 1: Without scaling
    print("\n1. WITHOUT SCALING")
    problem1 = tl.Problem("No Scaling")

    # Recreate problem structure
    t1 = problem1.time(initial=0.0, final=1.0)
    x1_1 = problem1.state("x1", initial=1000000.0, final=2000000.0, lower=0.0, upper=5000000.0)
    x2_1 = problem1.state("x2", initial=1.0, final=2.0, lower=0.1, upper=5.0)
    u1 = problem1.control("u", lower=-10000.0, upper=10000.0)

    problem1.dynamics({x1_1: u1, x2_1: u1 / 1000000})

    integral_1 = problem1.add_integral(0.5 * u1**2)
    problem1.minimize(integral_1)

    problem1.set_mesh(degrees, mesh)

    # Simple initial guess
    states_guess = []
    controls_guess = []
    for N in degrees:
        # State guess - linear interpolation with correct number of nodes
        tau_points = np.linspace(-1, 1, N + 1)  # N+1 nodes for states
        x1_vals = 1000000.0 + (1000000.0) * (tau_points + 1) / 2
        x2_vals = 1.0 + 1.0 * (tau_points + 1) / 2
        state_array = np.vstack([x1_vals, x2_vals])
        states_guess.append(state_array)

        # Control guess - zero with correct number of nodes
        control_array = np.zeros((1, N))  # N nodes for controls
        controls_guess.append(control_array)

    problem1.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=1.0,
        integrals=1000000.0,  # Large scale for integral guess
    )

    try:
        import time

        start_time = time.time()
        solution1 = tl.solve_fixed_mesh(
            problem1,
            polynomial_degrees=degrees,
            mesh_points=mesh,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-6},
        )
        time1 = time.time() - start_time

        if solution1.success:
            print(f"   ✓ Success: Objective = {solution1.objective:.3e}, Time = {time1:.3f}s")
        else:
            print(f"   ✗ Failed: {solution1.message}")
            time1 = float("inf")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
        solution1 = None
        time1 = float("inf")

    # Test 2: With scaling
    print("\n2. WITH SCALING")
    problem2 = tl.Problem("With Scaling")

    # Recreate problem structure
    t2 = problem2.time(initial=0.0, final=1.0)
    x1_2 = problem2.state("x1", initial=1000000.0, final=2000000.0, lower=0.0, upper=5000000.0)
    x2_2 = problem2.state("x2", initial=1.0, final=2.0, lower=0.1, upper=5.0)
    u2 = problem2.control("u", lower=-10000.0, upper=10000.0)

    problem2.dynamics({x1_2: u2, x2_2: u2 / 1000000})

    integral_2 = problem2.add_integral(0.5 * u2**2)
    problem2.minimize(integral_2)

    problem2.set_mesh(degrees, mesh)

    # Enable scaling
    problem2.enable_variable_scaling(True)
    scaling_info = problem2.compute_scaling()

    print("   Scaling computed:")
    for name, scaling in scaling_info.state_scaling.items():
        if scaling.lower_bound is not None:
            print(
                f"     State {name}: range {scaling.upper_bound - scaling.lower_bound:.0e} → "
                + f"weight={scaling.scale_weight:.3e}, shift={scaling.shift:.3f}"
            )
    for name, scaling in scaling_info.control_scaling.items():
        if scaling.lower_bound is not None:
            print(
                f"     Control {name}: range {scaling.upper_bound - scaling.lower_bound:.0e} → "
                + f"weight={scaling.scale_weight:.3e}, shift={scaling.shift:.3f}"
            )

    problem2.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=1.0,
        integrals=1000000.0,
    )

    try:
        start_time = time.time()
        solution2 = tl.solve_fixed_mesh(
            problem2,
            polynomial_degrees=degrees,
            mesh_points=mesh,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-6},
        )
        time2 = time.time() - start_time

        if solution2.success:
            print(f"   ✓ Success: Objective = {solution2.objective:.3e}, Time = {time2:.3f}s")
        else:
            print(f"   ✗ Failed: {solution2.message}")
            time2 = float("inf")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
        solution2 = None
        time2 = float("inf")

    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    if solution1 and solution1.success and solution2 and solution2.success:
        print(f"Without scaling: {solution1.objective:.6e} (time: {time1:.3f}s)")
        print(f"With scaling:    {solution2.objective:.6e} (time: {time2:.3f}s)")
        print(f"Objective diff:  {abs(solution1.objective - solution2.objective):.2e}")

        if time1 > 0 and time2 > 0:
            speedup = time1 / time2
            improvement = (time1 - time2) / time1 * 100
            print(f"Time improvement: {improvement:.1f}% (speedup: {speedup:.2f}x)")

            if improvement > 10:
                print("✓ Scaling provides significant benefit!")
            else:
                print("~ Scaling provides minor benefit")

        # Plot solutions if successful
        print("\nPlotting solutions...")
        try:
            solution1.plot()
            solution2.plot()
        except Exception as e:
            print(f"Plotting failed: {e}")

    elif solution2 and solution2.success:
        print("✓ Only scaled version succeeded - scaling essential for this problem!")
    elif solution1 and solution1.success:
        print("~ Only unscaled version succeeded - problem may not need scaling")
    else:
        print("✗ Both versions failed - problem may be too difficult")


if __name__ == "__main__":
    test_simple_scaling()
