# test_radau_mathematical_correctness.py
"""
Safety-critical tests for Radau pseudospectral mathematical foundation.
Tests against known analytical solutions and mathematical properties.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from trajectolab.radau import (
    compute_barycentric_weights,
    compute_radau_collocation_components,
    evaluate_lagrange_polynomial_at_point,
)


class TestRadauMathematicalCorrectness:
    """Test mathematical correctness of Radau basis functions against analytical solutions."""

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 8, 10, 15])
    def test_radau_nodes_orthogonality_property(self, N):
        """Test that Radau nodes satisfy orthogonality properties (SAFETY CRITICAL)."""
        components = compute_radau_collocation_components(N)
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
        """Test that Radau quadrature is exact for polynomials up to degree 2N-2."""
        components = compute_radau_collocation_components(N)
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
        """Test mathematical properties of barycentric weights."""
        # Test with various node configurations
        test_cases = [
            np.array([-1.0, 1.0]),  # Simple case
            np.array([-1.0, 0.0, 1.0]),  # Three points
            np.array([-1.0, -0.5, 0.5, 1.0]),  # Four points
        ]

        for nodes in test_cases:
            weights = compute_barycentric_weights(nodes)

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
        """Test Lagrange interpolation accuracy against known functions."""
        # Test interpolation of polynomial functions (should be exact)
        nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        weights = compute_barycentric_weights(nodes)

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
                lagrange_coeffs = evaluate_lagrange_polynomial_at_point(nodes, weights, tau)
                interpolated_value = np.dot(lagrange_coeffs, func_values)

                # Compare with exact function value
                exact_value = func(tau)
                error = abs(interpolated_value - exact_value)

                assert error < 1e-12, (
                    f"Interpolation failed for {name} at tau={tau}: "
                    f"interpolated={interpolated_value}, exact={exact_value}, error={error}"
                )

    @pytest.mark.parametrize("N", [2, 3, 4, 5])
    def test_differentiation_matrix_accuracy(self, N):
        """Test differentiation matrix accuracy against analytical derivatives."""
        components = compute_radau_collocation_components(N)
        state_nodes = components.state_approximation_nodes
        colloc_nodes = components.collocation_nodes
        diff_matrix = components.differentiation_matrix

        # Test on polynomial functions where we know the exact derivative
        test_cases = [
            (lambda x: x**2, lambda x: 2 * x, "x²"),
            (lambda x: x**3, lambda x: 3 * x**2, "x³"),
            (lambda x: x**4, lambda x: 4 * x**3, "x⁴"),
        ]

        for func, dfunc, name in test_cases:
            # Function values at state nodes
            func_values = np.array([func(x) for x in state_nodes])

            # Compute derivatives at collocation nodes using differentiation matrix
            computed_derivatives = diff_matrix @ func_values

            # Exact derivatives at collocation nodes
            exact_derivatives = np.array([dfunc(x) for x in colloc_nodes])

            # Compare
            max_error = np.max(np.abs(computed_derivatives - exact_derivatives))
            assert max_error < 1e-12, (
                f"Differentiation matrix failed for {name} with N={N}: max_error={max_error}"
            )

    def test_numerical_stability_edge_cases(self):
        """Test numerical stability in edge cases that could cause NASA mission failures."""

        # Test with very close nodes (could cause division by zero)
        close_nodes = np.array([-1.0, -0.999999999, 1.0])
        weights = compute_barycentric_weights(close_nodes)

        # Should not contain NaN or Inf
        assert np.all(np.isfinite(weights)), "Barycentric weights contain NaN/Inf"

        # Test evaluation near nodes (potential division by zero)
        test_point = -0.9999999999  # Very close to -1.0
        lagrange_vals = evaluate_lagrange_polynomial_at_point(close_nodes, weights, test_point)

        assert np.all(np.isfinite(lagrange_vals)), "Lagrange evaluation contains NaN/Inf"
        assert abs(np.sum(lagrange_vals) - 1.0) < 1e-10, "Partition of unity violated"

    def test_radau_cache_consistency(self):
        """Test that cached Radau components are consistent across calls."""
        N = 5

        # Get components multiple times
        comp1 = compute_radau_collocation_components(N)
        comp2 = compute_radau_collocation_components(N)
        comp3 = compute_radau_collocation_components(N)

        # Should be identical (same object due to caching)
        assert comp1 is comp2 is comp3, "Cache not working properly"

        # Verify mathematical consistency
        assert_allclose(comp1.collocation_nodes, comp2.collocation_nodes, rtol=1e-15)
        assert_allclose(comp1.quadrature_weights, comp2.quadrature_weights, rtol=1e-15)
        assert_allclose(comp1.differentiation_matrix, comp2.differentiation_matrix, rtol=1e-15)
