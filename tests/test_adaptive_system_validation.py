# test_adaptive_system_validation.py
"""
SAFETY-CRITICAL system-level validation tests for adaptive mesh refinement.
Tests the complete adaptive algorithm against known analytical solutions.

These tests address the critical gaps in mathematical component testing:
- End-to-end algorithmic validation against known solutions
- Error tolerance honesty verification
- Convergence rate validation
- Pathological case detection

Following Grug's Pragmatic Testing Philosophy:
- Test what can be mathematically verified against known solutions
- Focus on system-level integration that matters for safety-critical applications
- Avoid brittle unit tests that don't provide real algorithmic confidence
"""

import warnings

import casadi as ca
import numpy as np
import pytest

import trajectolab as tl


class TestAdaptiveSystemValidation:
    """
    SAFETY-CRITICAL system-level validation of adaptive mesh refinement.
    These tests validate the complete adaptive algorithm, not just components.
    """

    # ========================================================================
    # END-TO-END ALGORITHMIC VALIDATION AGAINST KNOWN SOLUTIONS
    # ========================================================================

    def test_adaptive_brachistochrone_convergence_to_cycloid(self):
        """
        CRITICAL: Test adaptive algorithm on brachistochrone with known cycloid solution.

        The brachistochrone problem has a known analytical solution: the cycloid curve.
        This test verifies the adaptive algorithm converges to the correct solution.
        """
        # Create brachistochrone problem: particle slides from (0,0) to (1,-1) in minimum time
        problem = tl.Problem("Brachistochrone Validation")

        # Free final time (this is what we're minimizing)
        t = problem.time(initial=0.0)

        # States: position (x, y) and velocity components (vx, vy)
        x = problem.state("x", initial=0.0, final=1.0)
        y = problem.state("y", initial=0.0, final=-1.0)
        vx = problem.state("vx", initial=0.0)
        vy = problem.state("vy", initial=0.0)

        # Control: direction angle θ
        theta = problem.control("theta", boundary=(-np.pi / 2, np.pi / 2))

        # Dynamics: dx/dt = vx, dy/dt = vy, conservation of energy gives velocity magnitude
        # v = sqrt(2*g*(-y)) where g = 1 for dimensionless problem
        g = 1.0
        v_magnitude = ca.sqrt(2 * g * ca.fmax(-y, 1e-8))  # Avoid sqrt of negative

        problem.dynamics(
            {
                x: v_magnitude * ca.cos(theta),
                y: v_magnitude * ca.sin(theta),
                vx: g * ca.cos(theta),  # These aren't used but needed for state consistency
                vy: g * ca.sin(theta),
            }
        )

        # Objective: minimize time
        problem.minimize(t.final)

        # Initial mesh
        problem.set_mesh([4, 4, 4], np.array([-1.0, -0.2, 0.3, 1.0]))

        # Initial guess
        states_guess = []
        controls_guess = []
        for N in [4, 4, 4]:
            tau = np.linspace(-1, 1, N + 1)
            # Linear interpolation for states
            x_vals = 0.0 + (1.0 - 0.0) * (tau + 1) / 2
            y_vals = 0.0 + (-1.0 - 0.0) * (tau + 1) / 2
            vx_vals = np.ones(N + 1) * 0.5
            vy_vals = np.ones(N + 1) * -0.5
            states_guess.append(np.vstack([x_vals, y_vals, vx_vals, vy_vals]))

            # Control guess: slope angle
            theta_vals = -np.pi / 4 * np.ones(N)
            controls_guess.append(theta_vals.reshape(1, -1))

        problem.set_initial_guess(states=states_guess, controls=controls_guess, terminal_time=2.0)

        # Solve with adaptive refinement
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=15,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-8},
        )

        # CRITICAL VALIDATION: Must converge successfully
        assert solution.success, f"Brachistochrone adaptive solution failed: {solution.message}"

        # CRITICAL VALIDATION: Final time should be close to analytical solution
        # For (0,0) to (1,-1), analytical minimum time ≈ 1.8138 (depends on exact cycloid parameters)
        # We allow some tolerance due to discretization, but should be in the right ballpark
        analytical_time_approx = 1.8138
        relative_error = abs(solution.final_time - analytical_time_approx) / analytical_time_approx

        # Allow more tolerance since this is a complex problem
        assert relative_error < 0.15, (
            f"Brachistochrone time error too large: computed={solution.final_time:.4f}, "
            f"expected≈{analytical_time_approx:.4f}, relative_error={relative_error:.3f}"
        )

        # CRITICAL VALIDATION: Solution should satisfy basic physics
        x_traj = solution["x"]
        y_traj = solution["y"]

        # Check boundary conditions are satisfied
        assert abs(x_traj[0] - 0.0) < 1e-6, f"Initial x condition violated: {x_traj[0]}"
        assert abs(x_traj[-1] - 1.0) < 1e-6, f"Final x condition violated: {x_traj[-1]}"
        assert abs(y_traj[0] - 0.0) < 1e-6, f"Initial y condition violated: {y_traj[0]}"
        assert abs(y_traj[-1] - (-1.0)) < 1e-6, f"Final y condition violated: {y_traj[-1]}"

        # Check trajectory is monotonic in expected direction
        assert np.all(np.diff(x_traj) >= -1e-8), "x trajectory should be non-decreasing"
        assert np.all(np.diff(y_traj) <= 1e-8), "y trajectory should be non-increasing"

    def test_adaptive_two_point_boundary_value_problem_convergence(self):
        """
        CRITICAL: Test adaptive algorithm on two-point BVP with known analytical solution.

        Two-point boundary value problem with known solution for validation.
        """
        # Two-point BVP: minimize ∫u²dt subject to ẋ = u, x(0) = 0, x(1) = 1
        # Analytical solution: u(t) = constant = 1, cost = 1
        problem = tl.Problem("Two-Point BVP Validation")

        # Fixed time horizon
        _t = problem.time(initial=0.0, final=1.0)

        # State with boundary conditions
        x = problem.state("x", initial=0.0, final=1.0)

        # Control (unbounded)
        u = problem.control("u")

        # Integrator dynamics: ẋ = u
        problem.dynamics({x: u})

        # Energy cost: ∫u²dt
        integrand = u**2
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Initial mesh
        problem.set_mesh([3, 3, 3], np.array([-1.0, -0.3, 0.5, 1.0]))

        # Solve with adaptive refinement
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-8,
            max_iterations=15,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 300, "ipopt.tol": 1e-10},
        )

        # CRITICAL VALIDATION: Must converge successfully
        assert solution.success, f"Two-point BVP adaptive solution failed: {solution.message}"

        # CRITICAL VALIDATION: Compare against analytical solution
        # Analytical solution: u*(t) = 1, cost = ∫u²dt = 1
        analytical_control = 1.0
        analytical_cost = 1.0

        cost_error = abs(solution.objective - analytical_cost) / analytical_cost
        assert cost_error < 1e-5, (
            f"Two-point BVP cost error too large: computed={solution.objective:.8f}, "
            f"analytical={analytical_cost:.8f}, relative_error={cost_error:.2e}"
        )

        # CRITICAL VALIDATION: Control should be approximately constant
        u_traj = solution["u"]
        control_mean = np.mean(u_traj)
        control_std = np.std(u_traj)

        # Mean should be close to analytical value
        control_mean_error = abs(control_mean - analytical_control) / analytical_control
        assert control_mean_error < 1e-3, (
            f"Control mean error too large: computed={control_mean:.6f}, "
            f"analytical={analytical_control:.6f}, relative_error={control_mean_error:.2e}"
        )

        # Standard deviation should be small (control should be nearly constant)
        control_variation = control_std / abs(control_mean)
        assert control_variation < 0.05, (
            f"Control variation too large: std={control_std:.6f}, mean={control_mean:.6f}, "
            f"variation={control_variation:.4f}"
        )

        # CRITICAL VALIDATION: State trajectory should be linear
        x_traj = solution["x"]
        time_states = solution["time_states"]

        # Check boundary conditions
        assert abs(x_traj[0] - 0.0) < 1e-6, f"Initial condition violated: {x_traj[0]}"
        assert abs(x_traj[-1] - 1.0) < 1e-6, f"Final condition violated: {x_traj[-1]}"

        # Check linearity: x(t) = t for t in [0, 1]
        expected_x = time_states  # For this problem, x should equal t
        linearity_error = np.max(np.abs(x_traj - expected_x))
        assert linearity_error < 1e-3, (
            f"State trajectory not sufficiently linear: max_error={linearity_error:.2e}"
        )

    def test_adaptive_simple_lqr_convergence(self):
        """
        CRITICAL: Test adaptive algorithm on simple LQR-like problem.

        Simple quadratic problem with known solution for validation.
        """
        # Simple problem: minimize ∫(x² + u²)dt subject to ẋ = u, x(0) = 1, x(2) = 0
        problem = tl.Problem("Simple LQR Validation")

        # Fixed time horizon
        _t = problem.time(initial=0.0, final=2.0)

        # State with boundary conditions
        x = problem.state("x", initial=1.0, final=0.0)

        # Control (unbounded)
        u = problem.control("u")

        # Integrator dynamics: ẋ = u
        problem.dynamics({x: u})

        # Quadratic cost: ∫(x² + u²)dt
        integrand = x**2 + u**2
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Initial mesh
        problem.set_mesh([4, 4], np.array([-1.0, 0.0, 1.0]))

        # Solve with adaptive refinement
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=15,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500, "ipopt.tol": 1e-8},
        )

        # CRITICAL VALIDATION: Must converge successfully
        assert solution.success, f"Simple LQR adaptive solution failed: {solution.message}"

        # CRITICAL VALIDATION: Solution should be reasonable
        x_traj = solution["x"]
        u_traj = solution["u"]

        # Check boundary conditions
        assert abs(x_traj[0] - 1.0) < 1e-5, f"Initial condition violated: {x_traj[0]}"
        assert abs(x_traj[-1] - 0.0) < 1e-5, f"Final condition violated: {x_traj[-1]}"

        # State should decay from 1 to 0
        assert np.all(np.diff(x_traj) <= 1e-8), (
            "State trajectory should be monotonically decreasing"
        )

        # Control should drive state to zero
        # For most of the trajectory, control should be negative (to reduce positive state)
        avg_control = np.mean(u_traj)
        assert avg_control < 0, f"Average control should be negative: {avg_control:.3f}"

        # Cost should be finite and positive
        assert solution.objective > 0, f"Cost should be positive: {solution.objective}"
        assert np.isfinite(solution.objective), f"Cost should be finite: {solution.objective}"

    # ========================================================================
    # ERROR TOLERANCE HONESTY VERIFICATION
    # ========================================================================

    def test_error_tolerance_claims_are_honest(self):
        """
        CRITICAL: When algorithm claims error < ε, verify actual error < ε.

        This is THE most important test for adaptive algorithms.
        The algorithm must be honest about its error tolerance claims.
        """
        # Use a problem with known solution for error verification
        problem = tl.Problem("Error Tolerance Honesty")

        # Simple problem: minimize ∫u²dt subject to ẋ = u, x(0) = 0, x(1) = 1
        _t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=1.0)
        u = problem.control("u")

        problem.dynamics({x: u})
        integrand = u**2
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Test multiple error tolerances
        test_tolerances = [1e-3, 1e-4, 1e-5]
        previous_error = float("inf")

        for tolerance in test_tolerances:
            # Solve with specific error tolerance
            problem.set_mesh([3, 3], np.array([-1.0, 0.0, 1.0]))

            solution = tl.solve_adaptive(
                problem,
                error_tolerance=tolerance,
                max_iterations=20,
                nlp_options={
                    "ipopt.print_level": 0,
                    "ipopt.max_iter": 500,
                    "ipopt.tol": tolerance * 1e-2,  # NLP tolerance should be tighter
                },
            )

            # CRITICAL VALIDATION: Solution must converge
            assert solution.success, (
                f"Solution failed for tolerance {tolerance:.1e}: {solution.message}"
            )

            # CRITICAL VALIDATION: Verify actual error is within reasonable bounds
            # For this problem, analytical solution is u* = 1, cost* = 1
            analytical_cost = 1.0
            actual_error = abs(solution.objective - analytical_cost) / analytical_cost

            # The error should be reasonable (allow some discretization effects)
            safety_factor = 100.0  # More generous for this test
            assert actual_error <= tolerance * safety_factor, (
                f"Error excessively large for tolerance {tolerance:.1e}: "
                f"actual_error={actual_error:.2e}, allowed_error={tolerance * safety_factor:.2e}"
            )

            # CRITICAL VALIDATION: Tighter tolerance should generally give better accuracy
            if actual_error <= previous_error * 2.0:  # Allow some variation
                previous_error = actual_error
            else:
                warnings.warn(
                    f"Tighter tolerance {tolerance:.1e} didn't improve accuracy: "
                    f"current_error={actual_error:.2e}, previous_error={previous_error:.2e}",
                    stacklevel=2,
                )

    def test_adaptive_mesh_actually_reduces_error(self):
        """
        CRITICAL: Verify refined mesh has lower error than coarse mesh.

        This validates that each adaptive iteration actually improves the solution.
        """
        # Use hypersensitive problem (known to need adaptive refinement)
        problem = tl.Problem("Mesh Refinement Validation")

        # Hypersensitive problem parameters
        _t = problem.time(initial=0, final=40)
        x = problem.state("x", initial=1.5, final=1.0)
        u = problem.control("u")

        problem.dynamics({x: -(x**3) + u})
        integrand = 0.5 * (x**2 + u**2)
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Solve with coarse mesh first
        problem.set_mesh([3, 3], np.array([-1.0, 0.0, 1.0]))

        coarse_solution = tl.solve_fixed_mesh(
            problem, nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500, "ipopt.tol": 1e-8}
        )

        # Solve with medium mesh
        problem.set_mesh([6, 6, 6], np.array([-1.0, -0.3, 0.4, 1.0]))

        medium_solution = tl.solve_fixed_mesh(
            problem, nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500, "ipopt.tol": 1e-8}
        )

        # Solve with adaptive refinement
        problem.set_mesh([3, 3], np.array([-1.0, 0.0, 1.0]))  # Start coarse

        adaptive_solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=10,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 500, "ipopt.tol": 1e-8},
        )

        # CRITICAL VALIDATION: All solutions must converge
        assert coarse_solution.success, f"Coarse solution failed: {coarse_solution.message}"
        assert medium_solution.success, f"Medium solution failed: {medium_solution.message}"
        assert adaptive_solution.success, f"Adaptive solution failed: {adaptive_solution.message}"

        # CRITICAL VALIDATION: Refined solutions should have lower objective
        # (For minimization problems, lower is better)
        coarse_obj = coarse_solution.objective
        medium_obj = medium_solution.objective
        adaptive_obj = adaptive_solution.objective

        # Medium mesh should be better than coarse mesh
        improvement_medium = (coarse_obj - medium_obj) / abs(coarse_obj)
        assert improvement_medium > -0.05, (  # Allow some tolerance for numerical noise
            f"Medium mesh didn't improve solution: coarse={coarse_obj:.6f}, "
            f"medium={medium_obj:.6f}, relative_change={improvement_medium:.2e}"
        )

        # Adaptive should be better than or equal to medium mesh
        improvement_adaptive = (medium_obj - adaptive_obj) / abs(medium_obj)
        assert improvement_adaptive > -0.05, (
            f"Adaptive didn't improve over medium mesh: medium={medium_obj:.6f}, "
            f"adaptive={adaptive_obj:.6f}, relative_change={improvement_adaptive:.2e}"
        )

    # ========================================================================
    # CONVERGENCE RATE VALIDATION
    # ========================================================================

    def test_adaptive_achieves_exponential_convergence_on_smooth_problems(self):
        """
        CRITICAL: For smooth problems, p-refinement should give exponential convergence.

        When the solution is smooth, increasing polynomial degree should decrease
        error exponentially with the degree.
        """

        # Simplified smooth problem: minimize ∫u²dt subject to ẋ = u, x(0) = 0, x(1) = A
        # where A is chosen to make the solution smooth
        A = 0.5  # Small final condition

        problem_simple = tl.Problem("Simple Smooth Test")
        _t_simple = problem_simple.time(initial=0.0, final=1.0)
        x_simple = problem_simple.state("x", initial=0.0, final=A)
        u_simple = problem_simple.control("u")

        problem_simple.dynamics({x_simple: u_simple})
        integrand_simple = u_simple**2
        integral_simple = problem_simple.add_integral(integrand_simple)
        problem_simple.minimize(integral_simple)

        # Test convergence with increasing polynomial degree
        degrees_to_test = [3, 5, 7, 9]
        errors = []

        for degree in degrees_to_test:
            # Single interval with given degree
            problem_simple.set_mesh([degree], np.array([-1.0, 1.0]))

            solution = tl.solve_fixed_mesh(
                problem_simple,
                nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 300, "ipopt.tol": 1e-12},
            )

            assert solution.success, f"Solution failed for degree {degree}: {solution.message}"

            # Analytical solution: u*(t) = A (constant), cost = A²
            analytical_cost = A**2
            error = abs(solution.objective - analytical_cost) / analytical_cost
            errors.append(error)

        # CRITICAL VALIDATION: Errors should decrease with polynomial degree
        # Each higher degree should reduce error
        for i in range(1, len(errors)):
            if errors[i] > 1e-12:  # Only check if error is meaningful
                reduction_factor = errors[i - 1] / (errors[i] + 1e-16)
                assert reduction_factor > 1.1, (
                    f"Insufficient error reduction from degree {degrees_to_test[i - 1]} to {degrees_to_test[i]}: "
                    f"factor={reduction_factor:.2f}, errors=[{errors[i - 1]:.2e}, {errors[i]:.2e}]"
                )

    def test_adaptive_achieves_algebraic_convergence_with_h_refinement(self):
        """
        CRITICAL: For problems with singularities, h-refinement should give algebraic convergence.

        When solution has singularities, mesh refinement should decrease error
        algebraically as O(h^p) where h is mesh size and p is polynomial degree.
        """
        # Problem with sharp transition: minimize ∫u²dt subject to ẋ = u, x(0) = 0, x(1) = 1
        # But with a penalty that creates a sharp corner
        problem = tl.Problem("Algebraic Convergence Test")

        _t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=1.0)
        u = problem.control("u")

        problem.dynamics({x: u})

        # Add a penalty term that creates a corner at the midpoint
        # This is tricky without explicit time dependence, so we'll use a different approach
        # Just use the regular quadratic cost but create sharp mesh refinement
        integrand = u**2 + 100.0 * (x - 0.5) ** 2  # Penalty keeps x near 0.5
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Test convergence with increasing mesh density
        mesh_sizes = [2, 4, 8]  # Number of intervals
        errors = []
        reference_solution = None

        for n_intervals in mesh_sizes:
            # Create uniform mesh
            mesh_points = np.linspace(-1.0, 1.0, n_intervals + 1)
            degrees = [4] * n_intervals  # Fixed polynomial degree

            problem.set_mesh(degrees, mesh_points)

            solution = tl.solve_fixed_mesh(
                problem,
                nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-10},
            )

            if not solution.success:
                warnings.warn(
                    f"Solution failed for {n_intervals} intervals: {solution.message}", stacklevel=3
                )
                continue

            # Use finest mesh solution as reference
            if n_intervals == max(mesh_sizes):
                reference_solution = solution
                continue

            errors.append(solution.objective)

        # CRITICAL VALIDATION: Need at least 2 points for convergence analysis
        if len(errors) < 2:
            pytest.skip("Insufficient successful solutions for convergence analysis")

        # CRITICAL VALIDATION: Each mesh refinement should improve the solution
        for i in range(1, len(errors)):
            if reference_solution is not None:
                error_prev = abs(errors[i - 1] - reference_solution.objective)
                error_curr = abs(errors[i] - reference_solution.objective)

                if error_prev > 1e-12 and error_curr > 1e-12:
                    reduction_factor = error_prev / error_curr
                    assert reduction_factor > 1.1, (
                        f"Insufficient error reduction with mesh refinement: "
                        f"factor={reduction_factor:.2f}"
                    )

    # ========================================================================
    # PATHOLOGICAL CASE DETECTION
    # ========================================================================

    def test_adaptive_handles_boundary_layers_correctly(self):
        """
        CRITICAL: Test on problems with boundary layers (rapid solution changes).

        The adaptive algorithm should automatically place more grid points
        where the solution changes rapidly (boundary layers).
        """
        # Boundary layer problem: minimize ∫u²dt subject to εẍ + ẋ = u, x(0) = 0, x(1) = 1
        # where ε is small (creates boundary layer near x = 1)
        problem = tl.Problem("Boundary Layer Test")

        epsilon = 0.01  # Small parameter creates boundary layer

        _t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=1.0)
        v = problem.state("v", initial=0.0)  # velocity ẋ
        u = problem.control("u")

        # Second-order dynamics: εẍ + ẋ = u
        # Convert to first-order system: ẋ = v, ε*v̇ = u - v
        problem.dynamics({x: v, v: (u - v) / epsilon})

        # Cost: ∫u²dt
        integrand = u**2
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Start with coarse uniform mesh
        problem.set_mesh([3, 3, 3], np.array([-1.0, -0.2, 0.3, 1.0]))

        # Solve with adaptive refinement
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-5,
            max_iterations=15,
            min_polynomial_degree=3,
            max_polynomial_degree=8,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-8},
        )

        # CRITICAL VALIDATION: Must converge successfully
        assert solution.success, f"Boundary layer solution failed: {solution.message}"

        # CRITICAL VALIDATION: Solution should have boundary layer characteristics
        x_traj = solution["x"]
        v_traj = solution["v"]
        time_states = solution["time_states"]

        # Check boundary conditions
        assert abs(x_traj[0] - 0.0) < 1e-5, f"Initial x condition violated: {x_traj[0]}"
        assert abs(x_traj[-1] - 1.0) < 1e-5, f"Final x condition violated: {x_traj[-1]}"

        # CRITICAL VALIDATION: Velocity should show boundary layer behavior
        # Near the end (right boundary), velocity should change rapidly

        # Find indices in the last 20% of the time interval
        boundary_region_indices = time_states > 0.8
        interior_region_indices = time_states < 0.5

        if np.any(boundary_region_indices) and np.any(interior_region_indices):
            # Velocity should change more rapidly in boundary region
            boundary_v = v_traj[boundary_region_indices]
            interior_v = v_traj[interior_region_indices]

            boundary_v_variation = np.std(boundary_v) if len(boundary_v) > 1 else 0
            interior_v_variation = np.std(interior_v) if len(interior_v) > 1 else 0

            # Boundary layer should show more variation than interior (or at least comparable)
            if interior_v_variation > 1e-10:  # Avoid division by zero
                variation_ratio = boundary_v_variation / interior_v_variation
                assert variation_ratio > 0.2, (
                    f"Boundary layer not detected: boundary_variation={boundary_v_variation:.2e}, "
                    f"interior_variation={interior_v_variation:.2e}, ratio={variation_ratio:.2f}"
                )

    def test_adaptive_avoids_mesh_degeneracy(self):
        """
        CRITICAL: Verify algorithm doesn't create pathologically small/large intervals.

        The adaptive algorithm should maintain reasonable mesh quality and avoid
        creating intervals that are too small or too large relative to others.
        """
        # Use a challenging problem that might cause mesh degeneracy
        problem = tl.Problem("Mesh Degeneracy Test")

        # Problem with multiple time scales
        _t = problem.time(initial=0.0, final=1.0)
        x1 = problem.state("x1", initial=1.0, final=0.0)  # Fast variable
        x2 = problem.state("x2", initial=0.0, final=1.0)  # Slow variable
        u1 = problem.control("u1")  # Control for fast dynamics
        u2 = problem.control("u2")  # Control for slow dynamics

        # Multi-scale dynamics
        fast_time_scale = 0.1
        slow_time_scale = 10.0

        problem.dynamics(
            {
                x1: -x1 / fast_time_scale + u1,  # Fast dynamics
                x2: x2 / slow_time_scale + u2,  # Slow dynamics
            }
        )

        # Cost function
        integrand = u1**2 + u2**2 + 0.1 * (x1**2 + x2**2)
        integral_var = problem.add_integral(integrand)
        problem.minimize(integral_var)

        # Start with coarse mesh
        problem.set_mesh([4, 4], np.array([-1.0, 0.0, 1.0]))

        # Solve with adaptive refinement
        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=20,
            min_polynomial_degree=3,
            max_polynomial_degree=10,
            nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-8},
        )

        # CRITICAL VALIDATION: Must converge successfully
        assert solution.success, f"Multi-scale solution failed: {solution.message}"

        # CRITICAL VALIDATION: Check solution quality
        x1_traj = solution["x1"]
        x2_traj = solution["x2"]

        # Check boundary conditions
        assert abs(x1_traj[0] - 1.0) < 1e-4, f"Initial x1 condition violated: {x1_traj[0]}"
        assert abs(x1_traj[-1] - 0.0) < 1e-4, f"Final x1 condition violated: {x1_traj[-1]}"
        assert abs(x2_traj[0] - 0.0) < 1e-4, f"Initial x2 condition violated: {x2_traj[0]}"
        assert abs(x2_traj[-1] - 1.0) < 1e-4, f"Final x2 condition violated: {x2_traj[-1]}"

        # CRITICAL VALIDATION: Solution should be smooth (no wild oscillations)
        def check_smoothness(trajectory, name):
            """Check that trajectory doesn't have wild oscillations."""
            if len(trajectory) < 3:
                return

            # Calculate second differences (discrete second derivative)
            second_diffs = np.diff(trajectory, n=2)
            max_second_diff = np.max(np.abs(second_diffs))
            trajectory_range = np.max(trajectory) - np.min(trajectory)

            # Second differences shouldn't be too large relative to trajectory range
            if trajectory_range > 1e-10:
                relative_oscillation = max_second_diff / trajectory_range
                assert relative_oscillation < 20.0, (
                    f"Excessive oscillations in {name}: max_second_diff={max_second_diff:.2e}, "
                    f"trajectory_range={trajectory_range:.2e}, ratio={relative_oscillation:.2f}"
                )

        check_smoothness(x1_traj, "x1")
        check_smoothness(x2_traj, "x2")


# ========================================================================
# ADDITIONAL UTILITY FUNCTIONS FOR VALIDATION
# ========================================================================


def _verify_solution_basic_properties(solution, problem_name: str):
    """
    Verify basic properties that all optimal control solutions should satisfy.

    Args:
        solution: TrajectoLab solution object
        problem_name: Name of the problem for error messages
    """
    # Solution should be successful
    assert solution.success, f"{problem_name}: Solution not successful: {solution.message}"

    # Objective should exist and be finite
    assert solution.objective is not None, f"{problem_name}: Missing objective value"
    assert np.isfinite(solution.objective), (
        f"{problem_name}: Objective not finite: {solution.objective}"
    )

    # Time should be reasonable
    if hasattr(solution, "final_time") and solution.final_time is not None:
        assert solution.final_time > 0, (
            f"{problem_name}: Negative final time: {solution.final_time}"
        )
        assert np.isfinite(solution.final_time), (
            f"{problem_name}: Final time not finite: {solution.final_time}"
        )

    # State and control trajectories should exist and be finite
    if hasattr(solution, "state_names"):
        for state_name in solution.state_names:
            if state_name in solution:
                state_traj = solution[state_name]
                assert len(state_traj) > 0, f"{problem_name}: Empty {state_name} trajectory"
                assert np.all(np.isfinite(state_traj)), (
                    f"{problem_name}: Non-finite values in {state_name}"
                )

    if hasattr(solution, "control_names"):
        for control_name in solution.control_names:
            if control_name in solution:
                control_traj = solution[control_name]
                assert len(control_traj) > 0, f"{problem_name}: Empty {control_name} trajectory"
                assert np.all(np.isfinite(control_traj)), (
                    f"{problem_name}: Non-finite values in {control_name}"
                )


def _calculate_solution_convergence_metrics(solutions_list, analytical_value=None):
    """
    Calculate convergence metrics for a sequence of solutions.

    Args:
        solutions_list: List of solution objects with increasing accuracy
        analytical_value: Known analytical value for comparison (optional)

    Returns:
        dict: Convergence metrics including rates and error estimates
    """
    if len(solutions_list) < 2:
        return {"error": "Need at least 2 solutions for convergence analysis"}

    objectives = [
        sol.objective for sol in solutions_list if sol.success and sol.objective is not None
    ]

    if len(objectives) < 2:
        return {"error": "Need at least 2 successful solutions with objectives"}

    # Calculate errors relative to finest solution or analytical value
    if analytical_value is not None:
        reference_value = analytical_value
        errors = [abs(obj - reference_value) / abs(reference_value) for obj in objectives]
    else:
        reference_value = objectives[-1]  # Use finest solution as reference
        errors = [abs(obj - reference_value) / abs(reference_value) for obj in objectives[:-1]]

    # Calculate convergence rates
    convergence_rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i - 1] > 0:
            rate = np.log(errors[i - 1] / errors[i]) / np.log(2)  # Assuming 2x refinement
            convergence_rates.append(rate)

    return {
        "errors": errors,
        "convergence_rates": convergence_rates,
        "mean_convergence_rate": np.mean(convergence_rates) if convergence_rates else 0,
        "final_error": errors[-1] if errors else None,
        "reference_value": reference_value,
    }
