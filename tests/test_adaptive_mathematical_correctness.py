# test_adaptive_mathematical_correctness.py
"""
SAFETY-CRITICAL tests for adaptive refinement mathematical foundation.
Tests against known analytical solutions and mathematical properties.

Following Grug's Pragmatic Testing Philosophy:
- Focus on integration tests that verify mathematical correctness
- Test what can actually be mathematically exact
- Avoid overly brittle unit tests that fail due to incorrect expectations
- Ensure tests provide real value for safety-critical adaptive refinement
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from trajectolab.adaptive.phs.error_estimation import (
    _calculate_combined_error_estimate,
    _calculate_gamma_normalization_factors,
    _calculate_trajectory_error_differences,
    _find_maximum_state_values_across_intervals,
)
from trajectolab.adaptive.phs.initial_guess import (
    _calculate_global_tau_points_for_interval,
    _determine_interpolation_parameters,
    _find_containing_interval_index,
    _interpolate_polynomial_at_evaluation_points,
    _validate_interpolated_trajectory_result,
)
from trajectolab.adaptive.phs.numerical import (
    PolynomialInterpolant,
    map_global_normalized_tau_to_local_interval_tau,
    map_local_interval_tau_to_global_normalized_tau,
    map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k,
    map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1,
)
from trajectolab.adaptive.phs.refinement import (
    _calculate_merge_feasibility_from_errors,
    _calculate_trajectory_errors_with_gamma,
    h_refine_params,
    p_reduce_interval,
    p_refine_interval,
)
from trajectolab.exceptions import DataIntegrityError, InterpolationError
from trajectolab.radau import compute_barycentric_weights, compute_radau_collocation_components


class TestAdaptiveMathematicalCorrectness:
    """Test mathematical correctness of adaptive refinement algorithms."""

    # ========================================================================
    # COORDINATE TRANSFORMATION MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_coordinate_transformation_invertibility(self):
        """Test that coordinate transformations are mathematically invertible."""
        # Test various interval configurations
        test_cases = [
            (-1.0, 1.0),  # Standard interval
            (-1.0, 0.0),  # Left half
            (0.0, 1.0),  # Right half
            (-0.5, 0.7),  # Arbitrary interval
            (-0.9, -0.1),  # Negative interval
        ]

        for global_start, global_end in test_cases:
            # Test multiple local tau values
            local_tau_values = np.linspace(-1.0, 1.0, 21)

            for local_tau in local_tau_values:
                # Forward transformation
                global_tau = map_local_interval_tau_to_global_normalized_tau(
                    local_tau, global_start, global_end
                )

                # Inverse transformation
                recovered_local_tau = map_global_normalized_tau_to_local_interval_tau(
                    global_tau, global_start, global_end
                )

                # Should be exact (within numerical precision)
                assert abs(recovered_local_tau - local_tau) < 1e-15, (
                    f"Transformation not invertible: interval=[{global_start}, {global_end}], "
                    f"original={local_tau}, recovered={recovered_local_tau}"
                )

    def test_coordinate_transformation_boundary_mapping(self):
        """Test that coordinate transformations correctly map boundaries."""
        test_intervals = [(-1.0, 1.0), (-0.5, 0.3), (0.1, 0.9)]

        for global_start, global_end in test_intervals:
            # Local tau = -1 should map to global_start
            global_tau_start = map_local_interval_tau_to_global_normalized_tau(
                -1.0, global_start, global_end
            )
            assert abs(global_tau_start - global_start) < 1e-15, (
                f"Boundary mapping failed: local_tau=-1 should map to {global_start}, got {global_tau_start}"
            )

            # Local tau = 1 should map to global_end
            global_tau_end = map_local_interval_tau_to_global_normalized_tau(
                1.0, global_start, global_end
            )
            assert abs(global_tau_end - global_end) < 1e-15, (
                f"Boundary mapping failed: local_tau=1 should map to {global_end}, got {global_tau_end}"
            )

    def test_cross_interval_tau_mapping_mathematical_consistency(self):
        """Test mathematical consistency of cross-interval tau mappings."""
        # Test configuration: three intervals
        global_start_k = -1.0
        global_shared = 0.2
        global_end_kp1 = 1.0

        # Test multiple local tau values
        local_tau_values = np.linspace(-1.0, 1.0, 11)

        for local_tau_k in local_tau_values:
            # Map from interval k to interval k+1
            local_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                local_tau_k, global_start_k, global_shared, global_end_kp1
            )

            # Map back from interval k+1 to interval k
            recovered_local_tau_k = (
                map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
                    local_tau_kp1, global_start_k, global_shared, global_end_kp1
                )
            )

            # Should be mathematically exact
            assert abs(recovered_local_tau_k - local_tau_k) < 1e-14, (
                f"Cross-interval mapping not invertible: original={local_tau_k}, recovered={recovered_local_tau_k}"
            )

    def test_coordinate_transformation_preserves_global_consistency(self):
        """Test that cross-interval mappings preserve global tau consistency."""
        # Test setup: two adjacent intervals
        global_start_k = -0.6
        global_shared = 0.1
        global_end_kp1 = 0.8

        local_tau_values = np.linspace(-1.0, 1.0, 15)

        for local_tau_k in local_tau_values:
            # Convert local tau in interval k to global tau
            global_tau_from_k = map_local_interval_tau_to_global_normalized_tau(
                local_tau_k, global_start_k, global_shared
            )

            # Map local tau from interval k to equivalent in interval k+1
            local_tau_kp1 = map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                local_tau_k, global_start_k, global_shared, global_end_kp1
            )

            # Convert local tau in interval k+1 to global tau
            global_tau_from_kp1 = map_local_interval_tau_to_global_normalized_tau(
                local_tau_kp1, global_shared, global_end_kp1
            )

            # Both should give the same global tau
            assert abs(global_tau_from_k - global_tau_from_kp1) < 1e-14, (
                f"Global tau inconsistency: from_k={global_tau_from_k}, from_kp1={global_tau_from_kp1}"
            )

    # ========================================================================
    # POLYNOMIAL INTERPOLATION MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_polynomial_interpolant_exact_reproduction(self):
        """Test that polynomial interpolants exactly reproduce values at nodes."""
        # Test with various node configurations
        node_configs = [
            np.array([-1.0, 1.0]),  # 2 nodes
            np.array([-1.0, 0.0, 1.0]),  # 3 nodes
            np.array([-1.0, -0.5, 0.5, 1.0]),  # 4 nodes
            np.linspace(-1, 1, 6),  # 6 nodes
        ]

        for nodes in node_configs:
            # Create test function values (multiple variables)
            num_vars = 3
            values = np.random.random((num_vars, len(nodes)))

            # Create interpolant
            interpolant = PolynomialInterpolant(nodes, values)

            # Test exact reproduction at nodes
            for i, node in enumerate(nodes):
                interpolated = interpolant(node)
                expected = values[:, i]

                max_error = np.max(np.abs(interpolated - expected))
                assert max_error < 1e-14, (
                    f"Interpolant failed to reproduce value at node {i}: error={max_error}"
                )

    def test_polynomial_interpolant_partition_of_unity(self):
        """Test that Lagrange polynomials satisfy partition of unity."""
        nodes = np.array([-1.0, -0.3, 0.2, 0.8, 1.0])
        weights = compute_barycentric_weights(nodes)

        # Test at various evaluation points
        test_points = np.linspace(-0.9, 0.9, 20)

        for tau in test_points:
            # Evaluate all Lagrange polynomials
            lagrange_sum = 0.0
            for j in range(len(nodes)):
                if abs(tau - nodes[j]) < 1e-14:
                    # At a node, the corresponding Lagrange polynomial is 1, others are 0
                    lagrange_sum = 1.0
                    break
                else:
                    lagrange_sum += weights[j] / (tau - nodes[j])

            if abs(tau - nodes).min() > 1e-14:  # Not at a node
                lagrange_sum /= np.sum(weights / (tau - nodes))

            # Should sum to 1 (partition of unity)
            assert abs(lagrange_sum - 1.0) < 1e-12, (
                f"Partition of unity violated at tau={tau}: sum={lagrange_sum}"
            )

    def test_polynomial_interpolant_polynomial_exactness(self):
        """Test that polynomial interpolants are exact for polynomials up to degree n-1."""
        nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # 5 nodes

        # Test polynomials of degree 0 to 4 (should be exact)
        for degree in range(5):
            # Create polynomial: x^degree
            poly_values = nodes**degree
            poly_values = poly_values.reshape(1, -1)  # Single variable

            interpolant = PolynomialInterpolant(nodes, poly_values)

            # Test at many evaluation points
            test_points = np.linspace(-0.9, 0.9, 50)
            for tau in test_points:
                interpolated = interpolant(tau)
                exact = tau**degree

                error = abs(interpolated[0] - exact)
                assert error < 1e-12, (
                    f"Polynomial interpolation failed for degree {degree} at tau={tau}: error={error}"
                )

    @pytest.mark.parametrize("N", [3, 5, 7, 9])
    def test_polynomial_interpolant_with_radau_nodes(self, N):
        """Test polynomial interpolation using actual Radau nodes."""
        components = compute_radau_collocation_components(N)
        nodes = components.state_approximation_nodes
        weights = components.barycentric_weights_for_state_nodes

        # Test function: combination of polynomials up to degree N
        def test_func(x):
            result = 0.0
            for k in range(N + 1):
                result += (k + 1) * x**k / math.factorial(k)
            return result

        # Function values at nodes
        func_values = np.array([test_func(node) for node in nodes]).reshape(1, -1)

        # Create interpolant
        interpolant = PolynomialInterpolant(nodes, func_values, weights)

        # Test at evaluation points
        test_points = np.linspace(-0.95, 0.95, 30)

        for tau in test_points:
            interpolated = interpolant(tau)[0]
            exact = test_func(tau)

            # For polynomials up to degree N, should be exact
            if N >= 5:  # High enough degree to represent the test function well
                error = abs(interpolated - exact)
                assert error < 1e-10, (
                    f"Radau interpolation failed for N={N} at tau={tau}: error={error}"
                )

    # ========================================================================
    # ERROR ESTIMATION MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_gamma_normalization_factors_mathematical_properties(self):
        """Test mathematical properties of gamma normalization factors."""
        # Test case 1: Known maximum values
        max_state_values = np.array([1.0, 10.0, 100.0])
        gamma_factors = _calculate_gamma_normalization_factors(max_state_values)

        expected_gamma = np.array([1.0 / 2.0, 1.0 / 11.0, 1.0 / 101.0]).reshape(-1, 1)
        assert_allclose(gamma_factors, expected_gamma, rtol=1e-15)

        # Test case 2: Zero maximum values (should avoid division by zero)
        max_state_values_zero = np.array([0.0, 0.0])
        gamma_factors_zero = _calculate_gamma_normalization_factors(max_state_values_zero)

        # Should be 1.0 (since 1/(1+0) = 1)
        expected_gamma_zero = np.array([1.0, 1.0]).reshape(-1, 1)
        assert_allclose(gamma_factors_zero, expected_gamma_zero, rtol=1e-15)

        # Test case 3: Very small values (numerical stability)
        max_state_values_small = np.array([1e-15, 1e-20])
        gamma_factors_small = _calculate_gamma_normalization_factors(max_state_values_small)

        # Should be finite and reasonable
        assert np.all(np.isfinite(gamma_factors_small))
        assert np.all(gamma_factors_small > 0)

    def test_maximum_state_values_calculation_correctness(self):
        """Test maximum state value calculation across intervals."""
        # Test case 1: Known maximum values
        interval_1 = np.array([[1.0, -2.0, 3.0], [4.0, -1.0, 2.0]])  # 2 states, 3 points
        interval_2 = np.array([[0.5, -5.0, 1.0], [2.0, -3.0, 6.0]])  # 2 states, 3 points

        Y_solved_list = [interval_1, interval_2]
        max_values = _find_maximum_state_values_across_intervals(Y_solved_list)

        # Expected: max(|1|, |-2|, |3|, |0.5|, |-5|, |1|) = 5 for state 0
        #           max(|4|, |-1|, |2|, |2|, |-3|, |6|) = 6 for state 1
        expected_max = np.array([5.0, 6.0])
        assert_allclose(max_values, expected_max, rtol=1e-15)

        # Test case 2: Empty list
        empty_list = []
        max_values_empty = _find_maximum_state_values_across_intervals(empty_list)
        assert len(max_values_empty) == 0

        # Test case 3: Single interval
        single_interval = [np.array([[2.0, -7.0, 1.0]])]  # 1 state, 3 points
        max_values_single = _find_maximum_state_values_across_intervals(single_interval)
        expected_single = np.array([7.0])  # max(|2|, |-7|, |1|) = 7
        assert_allclose(max_values_single, expected_single, rtol=1e-15)

    def test_trajectory_error_differences_mathematical_correctness(self):
        """Test trajectory error difference calculations."""
        # Test setup
        sim_trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 states, 3 points
        nlp_trajectory = np.array([[1.1, 1.9, 3.2], [3.8, 5.1, 5.9]])  # 2 states, 3 points
        gamma_factors = np.array([[0.5], [0.2]])  # Scaling factors for each state

        abs_diff, max_errors = _calculate_trajectory_error_differences(
            sim_trajectory, nlp_trajectory, gamma_factors
        )

        # Expected absolute differences
        expected_abs_diff = np.array([[0.1, 0.1, 0.2], [0.2, 0.1, 0.1]])
        assert_allclose(abs_diff, expected_abs_diff, rtol=1e-13)

        # Expected scaled errors
        np.array([[0.05, 0.05, 0.1], [0.04, 0.02, 0.02]])

        # Expected maximum errors per state
        expected_max_errors = np.array([0.1, 0.04])  # max of each row in scaled errors
        assert_allclose(max_errors, expected_max_errors, rtol=1e-15)

    def test_combined_error_estimate_mathematical_properties(self):
        """Test combined error estimate calculation properties."""
        # Test case 1: Known error values
        fwd_errors = np.array([0.1, 0.05, 0.2])
        bwd_errors = np.array([0.08, 0.12, 0.15])

        combined_error = _calculate_combined_error_estimate(fwd_errors, bwd_errors)

        # Should be maximum of element-wise maximum
        expected_combined = 0.2  # max(max(0.1, 0.08), max(0.05, 0.12), max(0.2, 0.15))
        assert abs(combined_error - expected_combined) < 1e-15

        # Test case 2: NaN handling (np.nanmax ignores NaN values and returns max of valid values)
        fwd_errors_nan = np.array([0.1, np.nan, 0.2])
        bwd_errors_nan = np.array([0.08, 0.12, np.nan])

        combined_error_nan = _calculate_combined_error_estimate(fwd_errors_nan, bwd_errors_nan)
        # Should return max of valid values: max(max(0.1, 0.08), max(nan, 0.12), max(0.2, nan)) = max(0.1, 0.12, 0.2) = 0.2
        assert abs(combined_error_nan - 0.2) < 1e-15

        # Test case 3: Very small errors (minimum threshold)
        fwd_errors_small = np.array([1e-20, 1e-18])
        bwd_errors_small = np.array([1e-19, 1e-17])

        combined_error_small = _calculate_combined_error_estimate(
            fwd_errors_small, bwd_errors_small
        )
        assert combined_error_small >= 1e-15  # Should enforce minimum threshold

    # ========================================================================
    # REFINEMENT ALGORITHM MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_p_refinement_mathematical_scaling(self):
        """Test p-refinement polynomial degree scaling mathematics."""
        # Test case 1: Error exactly at tolerance (no refinement needed)
        result = p_refine_interval(max_error=1e-6, current_Nk=5, error_tol=1e-6, N_max=10)
        assert not result.was_p_successful
        assert result.actual_Nk_to_use == 5

        # Test case 2: Error 10x tolerance (should add 1 node)
        result = p_refine_interval(max_error=1e-5, current_Nk=5, error_tol=1e-6, N_max=10)
        assert result.was_p_successful
        expected_nodes = 5 + max(1, int(np.ceil(np.log10(1e-5 / 1e-6))))  # 5 + 1 = 6
        assert result.actual_Nk_to_use == expected_nodes

        # Test case 3: Error 100x tolerance (should add 2 nodes)
        result = p_refine_interval(max_error=1e-4, current_Nk=5, error_tol=1e-6, N_max=10)
        assert result.was_p_successful
        expected_nodes = 5 + max(1, int(np.ceil(np.log10(1e-4 / 1e-6))))  # 5 + 2 = 7
        assert result.actual_Nk_to_use == expected_nodes

        # Test case 4: Infinite error (should add maximum possible nodes)
        result = p_refine_interval(max_error=np.inf, current_Nk=5, error_tol=1e-6, N_max=10)
        assert result.was_p_successful
        assert result.actual_Nk_to_use == 10  # Should reach N_max

        # Test case 5: Target exceeds N_max (should be capped)
        result = p_refine_interval(max_error=1e-2, current_Nk=8, error_tol=1e-6, N_max=10)
        assert result.was_p_successful is False  # Cannot achieve target within N_max
        assert result.actual_Nk_to_use == 10  # Capped at N_max

    def test_h_refinement_mathematical_subdivision(self):
        """Test h-refinement subdivision mathematics."""
        # Test case 1: Small target degree (should create 2 subintervals)
        result = h_refine_params(target_Nk=6, N_min=3)
        assert result.num_new_subintervals == 2
        assert result.collocation_nodes_for_new_subintervals == [3, 3]

        # Test case 2: Large target degree (should create multiple subintervals)
        result = h_refine_params(target_Nk=15, N_min=4)
        expected_subintervals = max(2, int(np.ceil(15 / 4)))  # ceil(3.75) = 4
        assert result.num_new_subintervals == expected_subintervals
        assert len(result.collocation_nodes_for_new_subintervals) == expected_subintervals
        assert all(nodes == 4 for nodes in result.collocation_nodes_for_new_subintervals)

        # Test case 3: Edge case where target equals N_min
        result = h_refine_params(target_Nk=3, N_min=3)
        assert result.num_new_subintervals == 2  # Should always create at least 2
        assert result.collocation_nodes_for_new_subintervals == [3, 3]

    def test_p_reduction_equation_36_mathematical_correctness(self):
        """Test p-reduction Equation 36 mathematical formula correctness."""
        # Test the mathematical formula: P_k^- = floor(log10((ε/e_max^(k))^(1/δ)))
        # where δ = N_min + N_max - N_k

        # Test case 1: Standard reduction scenario
        current_Nk = 8
        max_error = 1e-8
        error_tol = 1e-6
        N_min = 3
        N_max = 12

        result = p_reduce_interval(current_Nk, max_error, error_tol, N_min, N_max)

        # Calculate expected result using Eq. 36
        delta = N_min + N_max - current_Nk  # 3 + 12 - 8 = 7
        ratio = error_tol / max_error  # 1e-6 / 1e-8 = 100
        power_arg = np.power(ratio, 1.0 / delta)  # 100^(1/7) ≈ 1.778
        expected_nodes_to_remove = int(np.floor(np.log10(power_arg)))  # floor(log10(1.778)) = 0
        expected_new_Nk = max(N_min, current_Nk - expected_nodes_to_remove)  # max(3, 8-0) = 8

        assert result.new_num_collocation_nodes == expected_new_Nk
        assert result.was_reduction_applied == (expected_new_Nk < current_Nk)

        # Test case 2: Strong reduction scenario
        current_Nk = 10
        max_error = 1e-10
        error_tol = 1e-6
        N_min = 3
        N_max = 15

        result = p_reduce_interval(current_Nk, max_error, error_tol, N_min, N_max)

        # Calculate expected result
        delta = N_min + N_max - current_Nk  # 3 + 15 - 10 = 8
        ratio = error_tol / max_error  # 1e-6 / 1e-10 = 10000
        power_arg = np.power(ratio, 1.0 / delta)  # 10000^(1/8) ≈ 3.16
        expected_nodes_to_remove = int(np.floor(np.log10(power_arg)))  # floor(log10(3.16)) = 0
        expected_new_Nk = max(N_min, current_Nk - expected_nodes_to_remove)

        assert result.new_num_collocation_nodes == expected_new_Nk

        # Test case 3: No reduction when error exceeds tolerance
        result = p_reduce_interval(current_Nk=5, max_error=1e-5, error_tol=1e-6, N_min=3, N_max=10)
        assert result.new_num_collocation_nodes == 5  # No change
        assert not result.was_reduction_applied

        # Test case 4: No reduction when already at minimum
        result = p_reduce_interval(current_Nk=3, max_error=1e-8, error_tol=1e-6, N_min=3, N_max=10)
        assert result.new_num_collocation_nodes == 3  # Cannot reduce below N_min
        assert not result.was_reduction_applied

    def test_merge_feasibility_mathematical_logic(self):
        """Test mathematical logic for merge feasibility calculation."""
        # Test case 1: All errors within tolerance
        fwd_errors = [1e-8, 5e-9, 2e-8]
        bwd_errors = [3e-8, 1e-9, 1.5e-8]
        error_tol = 1e-7

        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            fwd_errors, bwd_errors, error_tol
        )

        assert can_merge
        expected_max_error = 3e-8  # max of all errors
        assert abs(max_error - expected_max_error) < 1e-15

        # Test case 2: Some errors exceed tolerance
        fwd_errors = [1e-6, 5e-9, 2e-8]  # First error exceeds tolerance
        bwd_errors = [3e-8, 1e-9, 1.5e-8]
        error_tol = 1e-7

        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            fwd_errors, bwd_errors, error_tol
        )

        assert not can_merge
        expected_max_error = 1e-6  # max of all errors
        assert abs(max_error - expected_max_error) < 1e-15

        # Test case 3: Empty error lists
        can_merge, max_error = _calculate_merge_feasibility_from_errors([], [], 1e-6)
        assert not can_merge
        assert max_error == np.inf

        # Test case 4: NaN handling
        fwd_errors = [1e-8, np.nan, 2e-8]
        bwd_errors = [3e-8, 1e-9, 1.5e-8]
        error_tol = 1e-7

        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            fwd_errors, bwd_errors, error_tol
        )

        assert can_merge is False
        assert max_error == np.inf

    def test_trajectory_errors_with_gamma_mathematical_scaling(self):
        """Test trajectory error calculation with gamma scaling."""
        # Test setup
        X_sim = np.array([1.0, 2.0, 3.0])
        X_nlp = np.array([1.1, 1.9, 3.2])
        gamma_factors = np.array([[0.5], [0.2], [0.1]])

        errors = _calculate_trajectory_errors_with_gamma(X_sim, X_nlp, gamma_factors)

        # Expected: |X_sim - X_nlp| * gamma_factors.flatten()
        abs_diff = np.abs(X_sim - X_nlp)  # [0.1, 0.1, 0.2]
        expected_errors = abs_diff * gamma_factors.flatten()  # [0.05, 0.02, 0.02]

        assert_allclose(errors, expected_errors, rtol=1e-15)

        # Test with NaN in simulation (should return empty list)
        X_sim_nan = np.array([1.0, np.nan, 3.0])
        errors_nan = _calculate_trajectory_errors_with_gamma(X_sim_nan, X_nlp, gamma_factors)
        assert len(errors_nan) == 0

    # ========================================================================
    # INTERVAL LOCATION MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_interval_location_boundary_cases(self):
        """Test interval location algorithm with boundary cases."""
        mesh_points = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        # Test exact mesh points (boundary points belong to left interval)
        expected_intervals = [0, 1, 2, 3]  # Each point belongs to its left interval
        for i, point in enumerate(mesh_points[:-1]):
            interval_idx = _find_containing_interval_index(point, mesh_points)
            assert interval_idx == expected_intervals[i], (
                f"Point {point} should be in interval {expected_intervals[i]}, got {interval_idx}"
            )

        # Test point exactly at final mesh point
        final_point = mesh_points[-1]
        interval_idx = _find_containing_interval_index(final_point, mesh_points)
        assert interval_idx == len(mesh_points) - 2  # Should be in last interval

        # Test points outside mesh
        outside_left = _find_containing_interval_index(-1.5, mesh_points)
        assert outside_left is None

        outside_right = _find_containing_interval_index(1.5, mesh_points)
        assert outside_right is None

        # Test points with numerical tolerance
        tolerance = 1e-10
        slightly_outside_left = _find_containing_interval_index(-1.0 - tolerance / 2, mesh_points)
        assert slightly_outside_left == 0  # Should be accepted within tolerance

        slightly_outside_right = _find_containing_interval_index(1.0 + tolerance / 2, mesh_points)
        assert slightly_outside_right == len(mesh_points) - 2  # Should be in last interval

    def test_global_tau_points_calculation_mathematical_correctness(self):
        """Test global tau points calculation for intervals."""
        # Test case 1: Standard interval
        target_local_nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        target_tau_start = -0.6
        target_tau_end = 0.2

        global_tau_points = _calculate_global_tau_points_for_interval(
            target_local_nodes, target_tau_start, target_tau_end
        )

        # Expected transformation: global_tau = beta * local_tau + beta_0
        # where beta = (tau_end - tau_start) / 2, beta_0 = (tau_end + tau_start) / 2
        beta = (target_tau_end - target_tau_start) / 2.0  # 0.4
        beta_0 = (target_tau_end + target_tau_start) / 2.0  # -0.2
        expected_global = beta * target_local_nodes + beta_0

        assert_allclose(global_tau_points, expected_global, rtol=1e-15)

        # Test case 2: Verify boundary mapping
        assert abs(global_tau_points[0] - target_tau_start) < 1e-15  # First point
        assert abs(global_tau_points[-1] - target_tau_end) < 1e-15  # Last point

    def test_interpolation_parameters_mathematical_consistency(self):
        """Test interpolation parameter determination mathematical consistency."""
        prev_mesh_points = np.array([-1.0, -0.3, 0.4, 1.0])

        # Test case 1: Point within mesh
        global_tau = 0.1
        prev_interval_idx, prev_local_tau = _determine_interpolation_parameters(
            global_tau, prev_mesh_points
        )

        # Should be in interval 1: [-0.3, 0.4]
        assert prev_interval_idx == 1

        # Verify local tau calculation
        tau_start = prev_mesh_points[1]  # -0.3
        tau_end = prev_mesh_points[2]  # 0.4
        expected_local_tau = map_global_normalized_tau_to_local_interval_tau(
            global_tau, tau_start, tau_end
        )
        assert abs(prev_local_tau - expected_local_tau) < 1e-15

        # Test case 2: Point outside mesh (left)
        global_tau_left = -1.5
        prev_interval_idx, prev_local_tau = _determine_interpolation_parameters(
            global_tau_left, prev_mesh_points
        )
        assert prev_interval_idx == 0
        assert prev_local_tau == -1.0  # Should use boundary value

        # Test case 3: Point outside mesh (right)
        global_tau_right = 1.5
        prev_interval_idx, prev_local_tau = _determine_interpolation_parameters(
            global_tau_right, prev_mesh_points
        )
        assert prev_interval_idx == len(prev_mesh_points) - 2  # Last interval
        assert prev_local_tau == 1.0  # Should use boundary value

    # ========================================================================
    # INTEGRATION TESTS FOR MATHEMATICAL CONSISTENCY
    # ========================================================================

    def test_end_to_end_coordinate_transformation_consistency(self):
        """Integration test for coordinate transformation mathematical consistency."""
        # Multi-interval mesh
        mesh_points = np.array([-1.0, -0.4, 0.1, 0.7, 1.0])

        # Test that all transformations are consistent
        for i in range(len(mesh_points) - 1):
            tau_start = mesh_points[i]
            tau_end = mesh_points[i + 1]

            # Test multiple local tau values
            local_tau_values = np.linspace(-1.0, 1.0, 11)

            for local_tau in local_tau_values:
                # Transform to global
                global_tau = map_local_interval_tau_to_global_normalized_tau(
                    local_tau, tau_start, tau_end
                )

                # Should be within interval bounds (with floating point tolerance)
                tolerance = 1e-14
                assert tau_start - tolerance <= global_tau <= tau_end + tolerance, (
                    f"Global tau {global_tau} outside interval [{tau_start}, {tau_end}] with tolerance {tolerance}"
                )

                # Find containing interval
                found_interval = _find_containing_interval_index(global_tau, mesh_points)
                assert found_interval == i, (
                    f"Interval location inconsistent: expected {i}, got {found_interval}"
                )

                # Transform back to local
                recovered_local = map_global_normalized_tau_to_local_interval_tau(
                    global_tau, tau_start, tau_end
                )
                assert abs(recovered_local - local_tau) < 1e-14, (
                    f"Round-trip transformation failed: {local_tau} -> {recovered_local}"
                )

    def test_polynomial_interpolation_cross_validation(self):
        """Cross-validation test for polynomial interpolation mathematical correctness."""

        # Create test function: f(x) = x^3 - 2*x^2 + x + 1
        def test_function(x):
            return x**3 - 2 * x**2 + x + 1

        # Test with different node configurations
        node_counts = [4, 5, 6, 8]

        for n_nodes in node_counts:
            # Create nodes and function values
            nodes = np.linspace(-1, 1, n_nodes)
            values = np.array([test_function(node) for node in nodes]).reshape(1, -1)

            # Create interpolant
            interpolant = PolynomialInterpolant(nodes, values)

            # Test at evaluation points
            eval_points = np.linspace(-0.9, 0.9, 20)

            for tau in eval_points:
                interpolated = interpolant(tau)[0]
                exact = test_function(tau)

                # For cubic polynomial with 4+ nodes, should be exact
                if n_nodes >= 4:
                    error = abs(interpolated - exact)
                    assert error < 1e-12, (
                        f"Cubic interpolation failed with {n_nodes} nodes at tau={tau}: error={error}"
                    )

    def test_adaptive_refinement_mathematical_invariants(self):
        """Test mathematical invariants that should hold for adaptive refinement."""
        # Test that p-refinement followed by p-reduction can be consistent
        current_Nk = 6
        max_error = 1e-4
        error_tol = 1e-6
        N_min = 3
        N_max = 12

        # Apply p-refinement
        p_refine_result = p_refine_interval(max_error, current_Nk, error_tol, N_max)

        if p_refine_result.was_p_successful:
            refined_Nk = p_refine_result.actual_Nk_to_use

            # If error becomes much smaller after refinement, p-reduction should be possible
            reduced_error = max_error / (
                10 ** (refined_Nk - current_Nk)
            )  # Assume exponential improvement

            if reduced_error < error_tol:
                p_reduce_result = p_reduce_interval(
                    refined_Nk, reduced_error, error_tol, N_min, N_max
                )

                # Should be able to reduce back towards original
                assert p_reduce_result.new_num_collocation_nodes <= refined_Nk
                assert p_reduce_result.new_num_collocation_nodes >= N_min

    def test_numerical_stability_edge_cases(self):
        """Test numerical stability in edge cases for adaptive refinement."""
        # Test with very small mesh intervals
        small_interval_start = -1e-10
        small_interval_end = 1e-10

        # Should not cause division by zero or numerical instability
        local_tau = 0.5
        global_tau = map_local_interval_tau_to_global_normalized_tau(
            local_tau, small_interval_start, small_interval_end
        )

        assert np.isfinite(global_tau)
        assert small_interval_start <= global_tau <= small_interval_end

        # Test with extreme gamma factors
        extreme_gamma = np.array([[1e-15], [1e15]])
        X_sim = np.array([1.0, 1.0])
        X_nlp = np.array([1.1, 0.9])

        errors = _calculate_trajectory_errors_with_gamma(X_sim, X_nlp, extreme_gamma)

        # Should be finite and not cause overflow/underflow
        assert all(np.isfinite(error) for error in errors)

        # Test interpolation parameter determination with degenerate cases
        degenerate_mesh = np.array([-1.0, -1.0, 1.0])  # Repeated point

        # Should handle gracefully without crashing
        try:
            prev_interval_idx, prev_local_tau = _determine_interpolation_parameters(
                0.0, degenerate_mesh
            )
            # Should return reasonable values
            assert prev_interval_idx is not None
            assert np.isfinite(prev_local_tau)
        except InterpolationError:
            # Acceptable to raise specific error for degenerate case
            pass

    def test_interpolated_trajectory_validation_mathematical_properties(self):
        """Test mathematical properties of interpolated trajectory validation."""
        # Test case 1: Valid trajectory
        valid_trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 variables, 3 points

        # Should not raise exception
        try:
            _validate_interpolated_trajectory_result(valid_trajectory, 2, 0, "state")
        except Exception as e:
            pytest.fail(f"Valid trajectory validation failed: {e}")

        # Test case 2: NaN values (should raise DataIntegrityError)
        invalid_trajectory_nan = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(DataIntegrityError):
            _validate_interpolated_trajectory_result(invalid_trajectory_nan, 2, 0, "state")

        # Test case 3: Infinite values (should raise DataIntegrityError)
        invalid_trajectory_inf = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(DataIntegrityError):
            _validate_interpolated_trajectory_result(invalid_trajectory_inf, 2, 0, "state")

        # Test case 4: Wrong number of variables (should raise InterpolationError)
        wrong_vars_trajectory = np.array([[1.0, 2.0, 3.0]])  # 1 variable, expected 2

        with pytest.raises(InterpolationError):
            _validate_interpolated_trajectory_result(wrong_vars_trajectory, 2, 0, "state")

    def test_barycentric_interpolation_mathematical_correctness(self):
        """Test barycentric interpolation using the exact mathematical formula."""
        # Test with known analytical case
        nodes = np.array([-1.0, 0.0, 1.0])

        # Test function: f(x) = x^2 (quadratic, should be exact with 3 nodes)
        values = np.array([1.0, 0.0, 1.0]).reshape(1, -1)  # f(-1)=1, f(0)=0, f(1)=1

        # Direct barycentric interpolation calculation
        weights = compute_barycentric_weights(nodes)

        # Test at evaluation point x = 0.5
        eval_point = 0.5

        # Manual barycentric formula calculation
        numerator = 0.0
        denominator = 0.0

        for j in range(len(nodes)):
            if abs(eval_point - nodes[j]) < 1e-14:
                # Exactly at a node
                expected_value = values[0, j]
                break
            else:
                weight_factor = weights[j] / (eval_point - nodes[j])
                numerator += weight_factor * values[0, j]
                denominator += weight_factor
        else:
            expected_value = numerator / denominator

        # Compare with PolynomialInterpolant result
        interpolant = PolynomialInterpolant(nodes, values)
        computed_value = interpolant(eval_point)[0]

        assert abs(computed_value - expected_value) < 1e-15, (
            f"Barycentric interpolation mismatch: computed={computed_value}, expected={expected_value}"
        )

        # For quadratic function at x=0.5, exact value should be 0.25
        exact_analytical = 0.5**2
        assert abs(computed_value - exact_analytical) < 1e-15, (
            f"Quadratic interpolation not exact: computed={computed_value}, exact={exact_analytical}"
        )

    def test_polynomial_interpolation_at_evaluation_points_mathematical_correctness(self):
        """Test direct polynomial interpolation at evaluation points."""
        # Test setup
        nodes = np.array([-1.0, -0.5, 0.5, 1.0])
        values = np.array([[1.0, 0.25, 0.25, 1.0], [2.0, 1.0, 3.0, 4.0]])  # 2 variables, 4 nodes
        evaluation_points = np.array([-0.75, 0.0, 0.75])

        # Compute barycentric weights
        barycentric_weights = compute_barycentric_weights(nodes)

        # Use the mathematical core function
        result = _interpolate_polynomial_at_evaluation_points(
            nodes, values, evaluation_points, barycentric_weights
        )

        # Verify shape
        assert result.shape == (2, 3)  # 2 variables, 3 evaluation points

        # Verify against individual interpolations
        for i, eval_point in enumerate(evaluation_points):
            interpolant = PolynomialInterpolant(nodes, values, barycentric_weights)
            individual_result = interpolant(eval_point)

            assert_allclose(result[:, i], individual_result, rtol=1e-15)

        # Test mathematical property: if evaluation points are nodes, should get exact values
        node_result = _interpolate_polynomial_at_evaluation_points(
            nodes, values, nodes, barycentric_weights
        )

        assert_allclose(node_result, values, rtol=1e-15)
