"""
Ultra-minimal test to isolate the exact issue.
"""

import numpy as np

import trajectolab as tl


def test_scaling_only():
    """Test just the scaling functionality without solving."""
    print("=" * 50)
    print("SCALING ONLY TEST")
    print("=" * 50)

    # Create problem
    problem = tl.Problem("Scaling Only")

    # Add variables with bounds
    x = problem.state("x", initial=1000000.0, final=2000000.0, lower=500000.0, upper=3000000.0)
    u = problem.control("u", lower=-5000.0, upper=5000.0)

    print("Variables with bounds:")
    print(f"  State x: bounds [{500000.0}, {3000000.0}]")
    print(f"  Control u: bounds [{-5000.0}, {5000.0}]")

    # Test scaling step by step
    print("\nStep 1: Enable scaling...")
    problem.enable_variable_scaling(True)

    print("Step 2: Compute scaling...")
    scaling_info = problem.compute_scaling()

    print("Step 3: Check results...")
    print(f"  Scaling enabled: {scaling_info.scaling_enabled}")
    print(f"  State scaling entries: {len(scaling_info.state_scaling)}")
    print(f"  Control scaling entries: {len(scaling_info.control_scaling)}")

    for name, scaling in scaling_info.state_scaling.items():
        print(f"  State {name}:")
        print(f"    Weight: {scaling.scale_weight:.6e}")
        print(f"    Shift: {scaling.shift:.6f}")
        print(f"    Method: {scaling.scaling_method}")
        print(f"    Original bounds: [{scaling.lower_bound}, {scaling.upper_bound}]")

        # Test scaled bounds calculation
        try:
            from trajectolab.utils.variable_scaling import get_scaled_variable_bounds

            scaled_lower, scaled_upper = get_scaled_variable_bounds(scaling)
            print(f"    Scaled bounds: [{scaled_lower:.6f}, {scaled_upper:.6f}]")
        except Exception as e:
            print(f"    Error calculating scaled bounds: {e}")

    for name, scaling in scaling_info.control_scaling.items():
        print(f"  Control {name}:")
        print(f"    Weight: {scaling.scale_weight:.6e}")
        print(f"    Shift: {scaling.shift:.6f}")
        print(f"    Method: {scaling.scaling_method}")
        print(f"    Original bounds: [{scaling.lower_bound}, {scaling.upper_bound}]")

        # Test scaled bounds calculation
        try:
            from trajectolab.utils.variable_scaling import get_scaled_variable_bounds

            scaled_lower, scaled_upper = get_scaled_variable_bounds(scaling)
            print(f"    Scaled bounds: [{scaled_lower:.6f}, {scaled_upper:.6f}]")
        except Exception as e:
            print(f"    Error calculating scaled bounds: {e}")

    return scaling_info


def test_problem_setup_only():
    """Test just setting up the problem without solving."""
    print("\n" + "=" * 50)
    print("PROBLEM SETUP ONLY TEST")
    print("=" * 50)

    try:
        # Create problem
        problem = tl.Problem("Setup Only")

        # Add variables
        x = problem.state("x", initial=1000000.0, final=2000000.0, lower=500000.0, upper=3000000.0)
        u = problem.control("u", lower=-5000.0, upper=5000.0)

        print("✓ Variables added")

        # Add dynamics
        problem.dynamics({x: u})
        print("✓ Dynamics added")

        # Add objective
        integral_expr = 0.5 * u**2
        objective_integral = problem.add_integral(integral_expr)
        problem.minimize(objective_integral)
        print("✓ Objective added")

        # Set mesh
        problem.set_mesh([4], [-1.0, 1.0])
        print("✓ Mesh configured")

        # Enable scaling
        problem.enable_variable_scaling(True)
        scaling_info = problem.compute_scaling()
        print("✓ Scaling computed")

        # Check requirements
        requirements = problem.get_initial_guess_requirements()
        print(f"✓ Requirements: {requirements.states_shapes}, {requirements.controls_shapes}")

        # Set initial guess
        states_guess = [np.linspace(1000000, 2000000, 5).reshape(1, -1)]
        controls_guess = [np.zeros((1, 4))]

        problem.set_initial_guess(
            states=states_guess,
            controls=controls_guess,
            initial_time=0.0,
            terminal_time=1.0,
            integrals=1000.0,
        )
        print("✓ Initial guess set")

        # Get solver functions (this is where issues often occur)
        print("Testing solver function creation...")

        try:
            dynamics_func = problem.get_dynamics_function()
            print("✓ Dynamics function created")
        except Exception as e:
            print(f"✗ Dynamics function error: {e}")
            import traceback

            traceback.print_exc()

        try:
            objective_func = problem.get_objective_function()
            print("✓ Objective function created")
        except Exception as e:
            print(f"✗ Objective function error: {e}")
            import traceback

            traceback.print_exc()

        try:
            path_constraints_func = problem.get_path_constraints_function()
            print(f"✓ Path constraints function: {path_constraints_func is not None}")
        except Exception as e:
            print(f"✗ Path constraints error: {e}")
            import traceback

            traceback.print_exc()

        try:
            event_constraints_func = problem.get_event_constraints_function()
            print(f"✓ Event constraints function: {event_constraints_func is not None}")
        except Exception as e:
            print(f"✗ Event constraints error: {e}")
            import traceback

            traceback.print_exc()

        print("✓ All solver functions created successfully")

    except Exception as e:
        print(f"✗ Error during setup: {e}")
        import traceback

        traceback.print_exc()


def test_solve_attempt():
    """Test actual solving with maximum simplification."""
    print("\n" + "=" * 50)
    print("SOLVE ATTEMPT TEST")
    print("=" * 50)

    try:
        # Create the simplest possible problem
        problem = tl.Problem("Simple Solve")

        # Single state, single control
        x = problem.state("x", initial=1000000.0, final=2000000.0)
        u = problem.control("u")

        # Simplest dynamics
        problem.dynamics({x: u})

        # Simplest objective
        problem.minimize(x)  # Just minimize final state

        # Simplest mesh
        problem.set_mesh([2], [-1.0, 1.0])  # Degree 2, single interval

        # Simple initial guess
        states_guess = [np.array([[1000000.0, 1500000.0, 2000000.0]])]
        controls_guess = [np.array([[0.0, 0.0]])]

        problem.set_initial_guess(states=states_guess, controls=controls_guess)

        print("Problem setup complete, attempting solve...")

        solution = tl.solve_fixed_mesh(
            problem,
            polynomial_degrees=[2],
            mesh_points=[-1.0, 1.0],
            nlp_options={"ipopt.print_level": 1, "ipopt.max_iter": 10},
        )

        if solution.success:
            print(f"✓ Solve successful: {solution.objective}")
        else:
            print(f"✗ Solve failed: {solution.message}")

    except Exception as e:
        print(f"✗ Solve error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run tests in order of complexity
    scaling_info = test_scaling_only()
    test_problem_setup_only()
    test_solve_attempt()
