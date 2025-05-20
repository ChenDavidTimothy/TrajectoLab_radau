import time

import numpy as np

import trajectolab as tl


def solve_hypersensitive_problem(use_scaling=True, method="adaptive"):
    """
    Solve the hypersensitive problem with or without scaling.

    Args:
        use_scaling: Whether to use automatic scaling
        method: 'fixed' or 'adaptive' mesh

    Returns:
        Solution object, solve time, and symbolic references
    """
    # Create the problem with scaling setting
    problem = tl.Problem("Hypersensitive Problem", use_scaling=use_scaling)

    # Define time variable
    t = problem.time(initial=0, final=40)

    # Add state with boundary conditions
    x = problem.state("x", initial=1.5, final=1.0)

    # Add control
    u = problem.control("u")

    # Define dynamics using symbolic expressions
    problem.dynamics({x: -(x**3) + u})

    # Define objective using symbolic expressions
    integral_expr = 0.5 * (x**2 + u**2)
    integral_var = problem.add_integral(integral_expr)

    # Set the objective to minimize
    problem.minimize(integral_var)

    # Set initial polynomial degrees and mesh points
    initial_polynomial_degrees = [8, 8, 8]
    initial_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]

    # IMPORTANT: Configure mesh BEFORE setting initial guess
    problem.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # Now provide the initial guess
    problem.set_initial_guess(integrals=0.1)

    # Common NLP options for both methods
    nlp_options = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 200,
    }

    # Start timing
    start_time = time.time()

    # Solve with appropriate method
    if method == "fixed":
        solution = tl.solve_fixed_mesh(
            problem,
            polynomial_degrees=initial_polynomial_degrees,
            mesh_points=initial_mesh_points,
            nlp_options=nlp_options,
        )
    else:  # adaptive
        solution = tl.solve_adaptive(
            problem,
            initial_polynomial_degrees=initial_polynomial_degrees,
            initial_mesh_points=initial_mesh_points,
            error_tolerance=1e-3,
            max_iterations=30,
            min_polynomial_degree=4,
            max_polynomial_degree=8,
            nlp_options=nlp_options,
        )

    # Calculate solve time
    solve_time = time.time() - start_time

    # Return solution, time, and symbolic references
    return solution, solve_time, {"x": x, "u": u, "t": t}


def run_and_report_comparison():
    """Run all four configurations and report results."""
    # Run the four configurations
    print("Running Hypersensitive Problem with Different Scaling Configurations")
    print("-" * 70)

    # Fixed mesh with scaling
    print("\n1. Fixed Mesh with Scaling")
    fixed_scaled_solution, fixed_scaled_time, fixed_scaled_symbols = solve_hypersensitive_problem(
        use_scaling=True, method="fixed"
    )

    # Fixed mesh without scaling
    print("\n2. Fixed Mesh without Scaling")
    fixed_unscaled_solution, fixed_unscaled_time, fixed_unscaled_symbols = (
        solve_hypersensitive_problem(use_scaling=False, method="fixed")
    )

    # Adaptive mesh with scaling
    print("\n3. Adaptive Mesh with Scaling")
    adaptive_scaled_solution, adaptive_scaled_time, adaptive_scaled_symbols = (
        solve_hypersensitive_problem(use_scaling=True, method="adaptive")
    )

    # Adaptive mesh without scaling
    print("\n4. Adaptive Mesh without Scaling")
    adaptive_unscaled_solution, adaptive_unscaled_time, adaptive_unscaled_symbols = (
        solve_hypersensitive_problem(use_scaling=False, method="adaptive")
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Success':<8} {'Objective':<15} {'Solve Time (s)':<15}")
    print("-" * 70)

    print(
        f"{'Fixed Mesh + Scaling':<25} {fixed_scaled_solution.success!s:<8} "
        f"{fixed_scaled_solution.objective if fixed_scaled_solution.success else 'N/A':<15.6f} "
        f"{fixed_scaled_time:<15.3f}"
    )

    print(
        f"{'Fixed Mesh No Scaling':<25} {fixed_unscaled_solution.success!s:<8} "
        f"{fixed_unscaled_solution.objective if fixed_unscaled_solution.success else 'N/A':<15.6f} "
        f"{fixed_unscaled_time:<15.3f}"
    )

    print(
        f"{'Adaptive Mesh + Scaling':<25} {adaptive_scaled_solution.success!s:<8} "
        f"{adaptive_scaled_solution.objective if adaptive_scaled_solution.success else 'N/A':<15.6f} "
        f"{adaptive_scaled_time:<15.3f}"
    )

    print(
        f"{'Adaptive Mesh No Scaling':<25} {adaptive_unscaled_solution.success!s:<8} "
        f"{adaptive_unscaled_solution.objective if adaptive_unscaled_solution.success else 'N/A':<15.6f} "
        f"{adaptive_unscaled_time:<15.3f}"
    )

    print("=" * 70)

    # Report adaptive mesh iterations and final mesh
    if adaptive_scaled_solution.success:
        print("\nAdaptive Mesh + Scaling Final Configuration:")
        print(f"Final polynomial degrees: {adaptive_scaled_solution.polynomial_degrees}")
        if adaptive_scaled_solution.mesh_points is not None:
            print(
                f"Final mesh points: {np.array2string(adaptive_scaled_solution.mesh_points, precision=3)}"
            )

    if adaptive_unscaled_solution.success:
        print("\nAdaptive Mesh No Scaling Final Configuration:")
        print(f"Final polynomial degrees: {adaptive_unscaled_solution.polynomial_degrees}")
        if adaptive_unscaled_solution.mesh_points is not None:
            print(
                f"Final mesh points: {np.array2string(adaptive_unscaled_solution.mesh_points, precision=3)}"
            )


if __name__ == "__main__":
    run_and_report_comparison()
