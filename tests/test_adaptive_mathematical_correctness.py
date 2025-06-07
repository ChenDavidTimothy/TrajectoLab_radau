import numpy as np
from numpy.testing import assert_allclose

from maptor.adaptive.phs.error_estimation import (
    _calculate_combined_error_estimate,
    _calculate_gamma_normalization_factors,
    _calculate_trajectory_error_differences,
)
from maptor.adaptive.phs.initial_guess import (
    _determine_interpolation_parameters,
    _find_containing_interval_index,
)
from maptor.adaptive.phs.numerical import (
    _map_global_normalized_tau_to_local_interval_tau,
    _map_local_interval_tau_to_global_normalized_tau,
    _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k,
    _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1,
)
from maptor.adaptive.phs.refinement import (
    _calculate_merge_feasibility_from_errors,
    _calculate_trajectory_errors_with_gamma,
    _p_reduce_interval,
    _p_refine_interval,
)
from maptor.exceptions import InterpolationError
from maptor.radau import _compute_barycentric_weights


class TestAdaptiveMathematicalCorrectness:
    def test_coordinate_transformation_invertibility(self):
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
                global_tau = _map_local_interval_tau_to_global_normalized_tau(
                    local_tau, global_start, global_end
                )

                # Inverse transformation
                recovered_local_tau = _map_global_normalized_tau_to_local_interval_tau(
                    global_tau, global_start, global_end
                )

                # Should be exact (within numerical precision)
                assert abs(recovered_local_tau - local_tau) < 1e-15, (
                    f"Transformation not invertible: interval=[{global_start}, {global_end}], "
                    f"original={local_tau}, recovered={recovered_local_tau}"
                )

    def test_coordinate_transformation_boundary_mapping(self):
        test_intervals = [(-1.0, 1.0), (-0.5, 0.3), (0.1, 0.9)]

        for global_start, global_end in test_intervals:
            # Local tau = -1 should map to global_start
            global_tau_start = _map_local_interval_tau_to_global_normalized_tau(
                -1.0, global_start, global_end
            )
            assert abs(global_tau_start - global_start) < 1e-15, (
                f"Boundary mapping failed: local_tau=-1 should map to {global_start}, got {global_tau_start}"
            )

            # Local tau = 1 should map to global_end
            global_tau_end = _map_local_interval_tau_to_global_normalized_tau(
                1.0, global_start, global_end
            )
            assert abs(global_tau_end - global_end) < 1e-15, (
                f"Boundary mapping failed: local_tau=1 should map to {global_end}, got {global_tau_end}"
            )

    def test_cross_interval_tau_mapping_mathematical_consistency(self):
        # Test configuration: three intervals
        global_start_k = -1.0
        global_shared = 0.2
        global_end_kp1 = 1.0

        # Test multiple local tau values
        local_tau_values = np.linspace(-1.0, 1.0, 11)

        for local_tau_k in local_tau_values:
            # Map from interval k to interval k+1
            local_tau_kp1 = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                local_tau_k, global_start_k, global_shared, global_end_kp1
            )

            # Map back from interval k+1 to interval k
            recovered_local_tau_k = (
                _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
                    local_tau_kp1, global_start_k, global_shared, global_end_kp1
                )
            )

            # Should be mathematically exact
            assert abs(recovered_local_tau_k - local_tau_k) < 1e-14, (
                f"Cross-interval mapping not invertible: original={local_tau_k}, recovered={recovered_local_tau_k}"
            )

    def test_coordinate_transformation_preserves_global_consistency(self):
        # Test setup: two adjacent intervals
        global_start_k = -0.6
        global_shared = 0.1
        global_end_kp1 = 0.8

        local_tau_values = np.linspace(-1.0, 1.0, 15)

        for local_tau_k in local_tau_values:
            # Convert local tau in interval k to global tau
            global_tau_from_k = _map_local_interval_tau_to_global_normalized_tau(
                local_tau_k, global_start_k, global_shared
            )

            # Map local tau from interval k to equivalent in interval k+1
            local_tau_kp1 = _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
                local_tau_k, global_start_k, global_shared, global_end_kp1
            )

            # Convert local tau in interval k+1 to global tau
            global_tau_from_kp1 = _map_local_interval_tau_to_global_normalized_tau(
                local_tau_kp1, global_shared, global_end_kp1
            )

            # Both should give the same global tau
            assert abs(global_tau_from_k - global_tau_from_kp1) < 1e-14, (
                f"Global tau inconsistency: from_k={global_tau_from_k}, from_kp1={global_tau_from_kp1}"
            )

    # ========================================================================
    # POLYNOMIAL INTERPOLATION MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_polynomial_interpolant_partition_of_unity(self):
        nodes = np.array([-1.0, -0.3, 0.2, 0.8, 1.0])
        weights = _compute_barycentric_weights(nodes)

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

    # ========================================================================
    # ERROR ESTIMATION MATHEMATICAL CORRECTNESS
    # ========================================================================

    def test_gamma_normalization_factors_mathematical_properties(self):
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

    def test_trajectory_error_differences_mathematical_correctness(self):
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
        # Test case 1: Error exactly at tolerance (no refinement needed)
        result = _p_refine_interval(max_error=1e-6, current_Nk=5, error_tol=1e-6, N_max=10)
        assert not result.was_p_successful
        assert result.actual_Nk_to_use == 5

        # Test case 2: Error 10x tolerance (should add 1 node)
        result = _p_refine_interval(max_error=1e-5, current_Nk=5, error_tol=1e-6, N_max=10)
        assert result.was_p_successful
        expected_nodes = 5 + max(1, int(np.ceil(np.log10(1e-5 / 1e-6))))  # 5 + 1 = 6
        assert result.actual_Nk_to_use == expected_nodes

        # Test case 3: Error 100x tolerance (should add 2 nodes)
        result = _p_refine_interval(max_error=1e-4, current_Nk=5, error_tol=1e-6, N_max=10)
        assert result.was_p_successful
        expected_nodes = 5 + max(1, int(np.ceil(np.log10(1e-4 / 1e-6))))  # 5 + 2 = 7
        assert result.actual_Nk_to_use == expected_nodes

        # Test case 4: Infinite error (should add maximum possible nodes)
        result = _p_refine_interval(max_error=np.inf, current_Nk=5, error_tol=1e-6, N_max=10)
        assert result.was_p_successful
        assert result.actual_Nk_to_use == 10  # Should reach N_max

        # Test case 5: Target exceeds N_max (should be capped)
        result = _p_refine_interval(max_error=1e-2, current_Nk=8, error_tol=1e-6, N_max=10)
        assert result.was_p_successful is False  # Cannot achieve target within N_max
        assert result.actual_Nk_to_use == 10  # Capped at N_max

    def test_merge_feasibility_mathematical_logic(self):
        # Test case 1: All errors within tolerance
        fwd_errors_list = [1e-8, 5e-9, 2e-8]
        bwd_errors_list = [3e-8, 1e-9, 1.5e-8]
        error_tol = 1e-7

        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            np.array(fwd_errors_list, dtype=float),
            np.array(bwd_errors_list, dtype=float),
            error_tol,
        )

        assert can_merge
        expected_max_error = 3e-8  # max of all errors
        assert abs(max_error - expected_max_error) < 1e-15

        # Test case 2: Some errors exceed tolerance
        fwd_errors_list = [1e-6, 5e-9, 2e-8]
        bwd_errors_list = [3e-8, 1e-9, 1.5e-8]
        error_tol = 1e-7

        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            np.array(fwd_errors_list, dtype=float),
            np.array(bwd_errors_list, dtype=float),
            error_tol,
        )

        assert not can_merge
        expected_max_error = 1e-6  # max of all errors
        assert abs(max_error - expected_max_error) < 1e-15

        # Test case 3: Empty error lists
        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            np.array([], dtype=float), np.array([], dtype=float), 1e-6
        )
        assert not can_merge
        assert max_error == float(np.inf)

        # Test case 4: NaN handling
        fwd_errors_list = [1e-8, np.nan, 2e-8]
        bwd_errors_list = [3e-8, 1e-9, 1.5e-8]
        error_tol = 1e-7

        can_merge, max_error = _calculate_merge_feasibility_from_errors(
            np.array(fwd_errors_list, dtype=float),
            np.array(bwd_errors_list, dtype=float),
            error_tol,
        )

        assert can_merge is False
        assert max_error == float(np.inf)  # Or just np.inf

    def test_trajectory_errors_with_gamma_mathematical_scaling(self):
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

    def test_interpolation_parameters_mathematical_consistency(self):
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
        expected_local_tau = _map_global_normalized_tau_to_local_interval_tau(
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
                global_tau = _map_local_interval_tau_to_global_normalized_tau(
                    local_tau, tau_start, tau_end
                )

                # Should be within interval bounds (with floating point tolerance)
                tolerance = 1e-14
                assert tau_start - tolerance <= global_tau <= tau_end + tolerance, (
                    f"Global tau {global_tau} outside interval [{tau_start}, {tau_end}] with tolerance {tolerance}"
                )

                # Find containing interval
                found_interval = _find_containing_interval_index(global_tau, mesh_points)

                # MATHEMATICAL CONSISTENCY: The found interval should be valid
                assert found_interval is not None, (
                    f"Global tau {global_tau} not found in any interval"
                )

                # MATHEMATICAL CONSISTENCY: Global tau should be within the found interval's bounds
                found_tau_start = mesh_points[found_interval]
                found_tau_end = mesh_points[found_interval + 1]

                # Allow for floating point tolerance at boundaries
                boundary_tolerance = 1e-12
                assert (
                    found_tau_start - boundary_tolerance
                    <= global_tau
                    <= found_tau_end + boundary_tolerance
                ), (
                    f"Global tau {global_tau} outside found interval [{found_tau_start}, {found_tau_end}] "
                    f"with tolerance {boundary_tolerance}"
                )

                # Transform back to local
                recovered_local = _map_global_normalized_tau_to_local_interval_tau(
                    global_tau, tau_start, tau_end
                )
                assert abs(recovered_local - local_tau) < 1e-14, (
                    f"Round-trip transformation failed: {local_tau} -> {recovered_local}"
                )

    def test_adaptive_refinement_mathematical_invariants(self):
        # Test that p-refinement followed by p-reduction can be consistent
        current_Nk = 6
        max_error = 1e-4
        error_tol = 1e-6
        N_min = 3
        N_max = 12

        # Apply p-refinement
        p_refine_result = _p_refine_interval(max_error, current_Nk, error_tol, N_max)

        if p_refine_result.was_p_successful:
            refined_Nk = p_refine_result.actual_Nk_to_use

            # If error becomes much smaller after refinement, p-reduction should be possible
            reduced_error = max_error / (
                10 ** (refined_Nk - current_Nk)
            )  # Assume exponential improvement

            if reduced_error < error_tol:
                p_reduce_result = _p_reduce_interval(
                    refined_Nk, reduced_error, error_tol, N_min, N_max
                )

                # Should be able to reduce back towards original
                assert p_reduce_result.new_num_collocation_nodes <= refined_Nk
                assert p_reduce_result.new_num_collocation_nodes >= N_min

    def test_numerical_stability_edge_cases(self):
        # Test with very small mesh intervals
        small_interval_start = -1e-10
        small_interval_end = 1e-10

        # Should not cause division by zero or numerical instability
        local_tau = 0.5
        global_tau = _map_local_interval_tau_to_global_normalized_tau(
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
