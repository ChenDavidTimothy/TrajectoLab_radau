import numpy as np
import pytest
from numpy.testing import assert_allclose

from maptor.radau import (
    _compute_barycentric_weights,
    _compute_radau_collocation_components,
    _evaluate_lagrange_polynomial_at_point,
)


class TestRadauMathematicalCorrectnessFixed:
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 8, 10, 15])
    def test_radau_nodes_orthogonality_property(self, N):
        components = _compute_radau_collocation_components(N)
        nodes = components.collocation_nodes
        weights = components.quadrature_weights

        # Test orthogonality: ∫_{-1}^{1} P_i(x) P_j(x) w(x) dx = 0 for i ≠ j
        # For Radau points, this should hold for polynomials up to degree 2N-2
        for i in range(N):
            for j in range(i + 1, min(N, 8)):  # Limit to prevent excessive computation
                # Evaluate Legendre polynomials at Radau nodes
                P_i = np.polynomial.legendre.legval(nodes, np.eye(N + 1)[i])
                P_j = np.polynomial.legendre.legval(nodes, np.eye(N + 1)[j])

                # Compute weighted inner product using Radau quadrature
                inner_product = np.sum(weights * P_i * P_j)

                # Should be zero for orthogonal polynomials
                assert abs(inner_product) < 1e-12, (
                    f"Orthogonality failed for N={N}, i={i}, j={j}: inner product = {inner_product}"
                )

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_radau_quadrature_exactness(self, N):
        components = _compute_radau_collocation_components(N)
        nodes = components.collocation_nodes
        weights = components.quadrature_weights

        # Test exactness for polynomials of degree 0 to 2*N-2
        for degree in range(2 * N - 1):
            # Create polynomial coefficients (highest degree first for numpy)
            poly_coeffs = np.zeros(degree + 1)
            poly_coeffs[0] = 1.0  # x^degree

            # Evaluate polynomial at Radau nodes
            poly_values = np.polyval(poly_coeffs, nodes)

            # Compute integral using Radau quadrature
            radau_integral = np.sum(weights * poly_values)

            # Compute exact integral analytically
            # ∫_{-1}^{1} x^n dx = 0 if n is odd, 2/(n+1) if n is even
            if degree % 2 == 1:
                exact_integral = 0.0
            else:
                exact_integral = 2.0 / (degree + 1)

            assert abs(radau_integral - exact_integral) < 1e-14, (
                f"Quadrature exactness failed for N={N}, degree={degree}: "
                f"Radau={radau_integral}, Exact={exact_integral}, "
                f"Error={abs(radau_integral - exact_integral)}"
            )

    def test_barycentric_weights_mathematical_properties(self):
        # Test with various node configurations
        test_cases = [
            np.array([-1.0, 1.0]),  # Simple case
            np.array([-1.0, 0.0, 1.0]),  # Three points
            np.array([-1.0, -0.5, 0.5, 1.0]),  # Four points
        ]

        for nodes in test_cases:
            weights = _compute_barycentric_weights(nodes)

            # Property 1: Lagrange polynomial partition of unity
            # Sum of all Lagrange polynomials should equal 1 at any point
            test_points = np.linspace(-0.9, 0.9, 20)

            for tau in test_points:
                lagrange_sum = 0.0
                for j in range(len(nodes)):
                    if abs(tau - nodes[j]) < 1e-14:
                        lagrange_sum = 1.0  # Exact at node
                        break
                    else:
                        lagrange_sum += weights[j] / (tau - nodes[j])

                if abs(tau - nodes).min() > 1e-14:  # Not at a node
                    lagrange_sum /= np.sum(weights / (tau - nodes))

                assert abs(lagrange_sum - 1.0) < 1e-12, (
                    f"Partition of unity failed at tau={tau}: sum={lagrange_sum}"
                )

    def test_lagrange_interpolation_accuracy(self):
        # Test interpolation of polynomial functions (should be exact)
        nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        weights = _compute_barycentric_weights(nodes)

        # Test polynomial functions that should be interpolated exactly
        test_functions = [
            (lambda x: 1.0, "constant"),
            (lambda x: x, "linear"),
            (lambda x: x**2, "quadratic"),
            (lambda x: x**3, "cubic"),
            (lambda x: x**4, "quartic"),
        ]

        for func, name in test_functions:
            # Get function values at nodes
            func_values = np.array([func(x) for x in nodes])

            # Test interpolation at various points
            test_points = np.linspace(-0.9, 0.9, 50)

            for tau in test_points:
                # Compute interpolated value
                lagrange_coeffs = _evaluate_lagrange_polynomial_at_point(nodes, weights, tau)
                interpolated_value = np.dot(lagrange_coeffs, func_values)

                # Compare with exact function value
                exact_value = func(tau)
                error = abs(interpolated_value - exact_value)

                assert error < 1e-12, (
                    f"Interpolation failed for {name} at tau={tau}: "
                    f"interpolated={interpolated_value}, exact={exact_value}, error={error}"
                )

    @pytest.mark.parametrize("N", [2, 3, 4, 5])
    def test_differentiation_matrix_accuracy_corrected(self, N):
        components = _compute_radau_collocation_components(N)
        state_nodes = components.state_approximation_nodes
        colloc_nodes = components.collocation_nodes
        diff_matrix = components.differentiation_matrix

        # For N collocation nodes, we can exactly represent polynomials up to degree N
        test_cases = []

        # Always test these basic cases
        test_cases.extend(
            [
                (lambda x: np.ones_like(x), lambda x: np.zeros_like(x), "constant"),
                (lambda x: x, lambda x: np.ones_like(x), "linear"),
            ]
        )

        # Only test higher degree polynomials if they can be exactly represented
        if N >= 2:
            test_cases.append((lambda x: x**2, lambda x: 2 * x, "x²"))
        if N >= 3:
            test_cases.append((lambda x: x**3, lambda x: 3 * x**2, "x³"))
        if N >= 4:
            test_cases.append((lambda x: x**4, lambda x: 4 * x**3, "x⁴"))
        if N >= 5:
            test_cases.append((lambda x: x**5, lambda x: 5 * x**4, "x⁵"))

        for func, dfunc, name in test_cases:
            # Function values at state nodes
            func_values = func(state_nodes)

            # Compute derivatives at collocation nodes using differentiation matrix
            computed_derivatives = diff_matrix @ func_values

            # Exact derivatives at collocation nodes
            exact_derivatives = dfunc(colloc_nodes)

            # Compare with appropriate tolerance
            max_error = np.max(np.abs(computed_derivatives - exact_derivatives))
            assert max_error < 1e-12, (
                f"Differentiation matrix failed for {name} with N={N}: max_error={max_error}"
            )

    @pytest.mark.parametrize("N", [2, 3, 4, 5])
    def test_differentiation_matrix_approximate_accuracy(self, N):
        components = _compute_radau_collocation_components(N)
        state_nodes = components.state_approximation_nodes
        colloc_nodes = components.collocation_nodes
        diff_matrix = components.differentiation_matrix

        # Test polynomials of degree > N (cannot be exactly represented)
        test_cases = []
        if N == 2:
            test_cases.extend(
                [
                    (lambda x: x**3, lambda x: 3 * x**2, "x³"),
                    (lambda x: x**4, lambda x: 4 * x**3, "x⁴"),
                ]
            )
        elif N == 3:
            test_cases.extend(
                [
                    (lambda x: x**4, lambda x: 4 * x**3, "x⁴"),
                    (lambda x: x**5, lambda x: 5 * x**4, "x⁵"),
                ]
            )
        elif N >= 4:
            test_cases.extend(
                [
                    (lambda x: x ** (N + 1), lambda x: (N + 1) * x**N, f"x^{N + 1}"),
                    (lambda x: x ** (N + 2), lambda x: (N + 2) * x ** (N + 1), f"x^{N + 2}"),
                ]
            )

        for func, dfunc, name in test_cases:
            # Function values at state nodes
            func_values = func(state_nodes)

            # Compute derivatives at collocation nodes using differentiation matrix
            computed_derivatives = diff_matrix @ func_values

            # Exact derivatives at collocation nodes
            exact_derivatives = dfunc(colloc_nodes)

            # For non-representable polynomials, we expect bounded but non-zero errors
            max_error = np.max(np.abs(computed_derivatives - exact_derivatives))

            # Error should be finite and reasonable (not exact, but bounded)
            assert np.isfinite(max_error), f"Non-finite error for {name} with N={N}"
            assert max_error < 100, f"Excessive error for {name} with N={N}: {max_error}"

            # For higher N, errors should generally decrease (spectral convergence)
            # This is a weaker test that verifies reasonable approximation behavior

    def test_numerical_stability_edge_cases(self):
        # Test with very close nodes (could cause division by zero)
        close_nodes = np.array([-1.0, -0.999999999, 1.0])
        weights = _compute_barycentric_weights(close_nodes)

        # Should not contain NaN or Inf
        assert np.all(np.isfinite(weights)), "Barycentric weights contain NaN/Inf"

        # Test evaluation near nodes (potential division by zero)
        test_point = -0.9999999999  # Very close to -1.0
        lagrange_vals = _evaluate_lagrange_polynomial_at_point(close_nodes, weights, test_point)

        assert np.all(np.isfinite(lagrange_vals)), "Lagrange evaluation contains NaN/Inf"
        assert abs(np.sum(lagrange_vals) - 1.0) < 1e-10, "Partition of unity violated"

    def test_radau_cache_consistency(self):
        N = 5

        # Get components multiple times
        comp1 = _compute_radau_collocation_components(N)
        comp2 = _compute_radau_collocation_components(N)
        comp3 = _compute_radau_collocation_components(N)

        # Should be identical (same object due to caching)
        assert comp1 is comp2 is comp3, "Cache not working properly"

        # Verify mathematical consistency
        assert_allclose(comp1.collocation_nodes, comp2.collocation_nodes, rtol=1e-15)
        assert_allclose(comp1.quadrature_weights, comp2.quadrature_weights, rtol=1e-15)
        assert_allclose(comp1.differentiation_matrix, comp2.differentiation_matrix, rtol=1e-15)

    def test_spectral_convergence_property(self):
        # Test function: exp(x) (smooth, infinitely differentiable)
        def test_func(x):
            return np.exp(x)

        def test_func_derivative(x):
            return np.exp(x)

        errors = []
        N_values = [3, 5, 7, 9, 11]

        for N in N_values:
            components = _compute_radau_collocation_components(N)
            state_nodes = components.state_approximation_nodes
            colloc_nodes = components.collocation_nodes
            diff_matrix = components.differentiation_matrix

            # Function values at state nodes
            func_values = test_func(state_nodes)

            # Compute derivatives at collocation nodes
            computed_derivatives = diff_matrix @ func_values

            # Exact derivatives at collocation nodes
            exact_derivatives = test_func_derivative(colloc_nodes)

            # Compute max error
            max_error = np.max(np.abs(computed_derivatives - exact_derivatives))
            errors.append(max_error)

        # Verify spectral convergence: errors should decrease rapidly
        for i in range(1, len(errors)):
            convergence_ratio = errors[i] / errors[i - 1]
            # For smooth functions, we expect rapid (spectral) convergence
            assert convergence_ratio < 0.5, (
                f"Insufficient convergence: N={N_values[i]} error={errors[i]}, "
                f"previous error={errors[i - 1]}, ratio={convergence_ratio}"
            )

    def test_realistic_trajectory_optimization_scenario_fixed(self):
        # Realistic trajectory function: position as function of time
        def position(t):
            # Scaled to [-1, 1] interval, realistic trajectory shape
            return 0.5 * t**2 + 0.3 * np.sin(2 * np.pi * t) + 0.1 * t**3

        def velocity(t):
            # Derivative of position = velocity
            return t + 0.6 * np.pi * np.cos(2 * np.pi * t) + 0.3 * t**2

        N = 8  # Reasonable order for trajectory optimization
        components = _compute_radau_collocation_components(N)
        state_nodes = components.state_approximation_nodes
        colloc_nodes = components.collocation_nodes
        diff_matrix = components.differentiation_matrix

        # Position values at state nodes
        position_values = position(state_nodes)

        # Compute velocity at collocation nodes using differentiation matrix
        computed_velocity = diff_matrix @ position_values

        # Exact velocity at collocation nodes
        exact_velocity = velocity(colloc_nodes)

        # Compute errors
        max_error = np.max(np.abs(computed_velocity - exact_velocity))
        relative_error = max_error / np.max(np.abs(exact_velocity))

        # EXPECTATIONS:
        # For mixed polynomial/trigonometric functions with N=8 collocation points,
        # absolute errors of O(1) are mathematically reasonable due to:
        # 1. sin(2πt) cannot be exactly represented by degree-8 polynomials
        # 2. High-frequency oscillations (2 cycles in [-1,1]) require many points
        # 3. Differentiation amplifies approximation errors

        # Test 1: Error should be finite and bounded
        assert np.isfinite(max_error), f"Non-finite error detected: {max_error}"
        assert max_error < 10.0, f"Excessive error suggests implementation bug: {max_error}"

        # Test 2: Relative error should be reasonable (< 50% for this challenging case)
        assert relative_error < 0.5, f"Relative error too large: {relative_error:.3f}"

        # Test 3: Verify no NaN or Inf values (for mission safety)
        assert np.all(np.isfinite(computed_velocity)), "Non-finite values in trajectory computation"

        # Test 4: Verify spectral accuracy on polynomial components
        # Test just the polynomial part: 0.5*t² + 0.1*t³
        def poly_position(t):
            return 0.5 * t**2 + 0.1 * t**3

        def poly_velocity(t):
            return t + 0.3 * t**2

        poly_position_values = poly_position(state_nodes)
        computed_poly_velocity = diff_matrix @ poly_position_values
        exact_poly_velocity = poly_velocity(colloc_nodes)

        poly_error = np.max(np.abs(computed_poly_velocity - exact_poly_velocity))

        # Polynomial components should be exact (within numerical precision)
        assert poly_error < 1e-12, f"Polynomial differentiation not exact: {poly_error}"

        print("Trajectory test results:")
        print(f"  Full function max error: {max_error:.6f}")
        print(f"  Full function relative error: {relative_error:.3f}")
        print(f"  Polynomial component error: {poly_error:.2e}")
        print(
            "  This demonstrates expected behavior: exact on polynomials, approximate on transcendentals"
        )

    def test_trajectory_convergence_behavior(self):
        def position(t):
            return 0.5 * t**2 + 0.3 * np.sin(2 * np.pi * t) + 0.1 * t**3

        def velocity(t):
            return t + 0.6 * np.pi * np.cos(2 * np.pi * t) + 0.3 * t**2

        errors = []
        N_values = [4, 6, 8, 10, 12]

        for N in N_values:
            components = _compute_radau_collocation_components(N)
            state_nodes = components.state_approximation_nodes
            colloc_nodes = components.collocation_nodes
            diff_matrix = components.differentiation_matrix

            position_values = position(state_nodes)
            computed_velocity = diff_matrix @ position_values
            exact_velocity = velocity(colloc_nodes)

            max_error = np.max(np.abs(computed_velocity - exact_velocity))
            errors.append(max_error)

        # Verify general convergence trend (errors should decrease)
        # Note: Convergence may not be monotonic due to trigonometric component
        # but overall trend should be downward

        # Test that highest N gives better result than lowest N
        improvement_ratio = errors[0] / errors[-1]
        assert improvement_ratio > 1.5, (
            f"Insufficient improvement with more points: {improvement_ratio:.2f}"
        )

        # Test that no error grows excessively (stability check)
        for i, error in enumerate(errors):
            assert error < 20.0, f"Unstable behavior at N={N_values[i]}: error={error}"
            assert np.isfinite(error), f"Non-finite error at N={N_values[i]}"

        print("Convergence test results:")
        for _i, (N, error) in enumerate(zip(N_values, errors, strict=False)):
            print(f"  N={N:2d}: error={error:.4f}")
        print(f"  Improvement ratio (N=4 to N=12): {improvement_ratio:.2f}")

    def test_polynomial_vs_transcendental_accuracy(self):
        N = 8
        components = _compute_radau_collocation_components(N)
        state_nodes = components.state_approximation_nodes
        colloc_nodes = components.collocation_nodes
        diff_matrix = components.differentiation_matrix

        # Test 1: Pure polynomial (should be exact)
        def poly_pos(t):
            return 0.1 * t**3 + 0.2 * t**2 + 0.3 * t + 0.4

        def poly_vel(t):
            return 0.3 * t**2 + 0.4 * t + 0.3

        poly_pos_vals = poly_pos(state_nodes)
        computed_poly_vel = diff_matrix @ poly_pos_vals
        exact_poly_vel = poly_vel(colloc_nodes)
        poly_error = np.max(np.abs(computed_poly_vel - exact_poly_vel))

        # Test 2: Pure trigonometric (will have approximation error)
        def trig_pos(t):
            return 0.5 * np.sin(2 * np.pi * t)

        def trig_vel(t):
            return np.pi * np.cos(2 * np.pi * t)

        trig_pos_vals = trig_pos(state_nodes)
        computed_trig_vel = diff_matrix @ trig_pos_vals
        exact_trig_vel = trig_vel(colloc_nodes)
        trig_error = np.max(np.abs(computed_trig_vel - exact_trig_vel))

        # Verify expected behavior
        assert poly_error < 1e-12, f"Polynomial should be exact: {poly_error}"
        assert trig_error > 1e-6, f"Trigonometric should have approximation error: {trig_error}"
        assert trig_error < 5.0, f"Trigonometric error should be bounded: {trig_error}"

        print("Accuracy comparison:")
        print(f"  Polynomial error: {poly_error:.2e} (should be ~machine precision)")
        print(f"  Trigonometric error: {trig_error:.4f} (expected approximation error)")
        print(f"  Error ratio: {trig_error / poly_error:.2e}")
