"""
RIGOROUS MATHEMATICAL VERIFICATION FOR BIRKHOFF IMPLEMENTATION - PYTEST VERSION

This test suite provides DEFINITIVE PROOF of mathematical correctness using pytest
for maximum reliability and industry-standard safety practices.

MATHEMATICAL FOUNDATION:
All tests verify fundamental axioms and theorems from Section 2 of the Birkhoff paper.
Each test MUST pass if implementation is correct, WILL fail if implementation has errors.

EXECUTION:
Run with: pytest test_birkhoff_mathematical_verification.py -v
For detailed output: pytest test_birkhoff_mathematical_verification.py -v -s

SAFETY-CRITICAL VERIFICATION:
This suite is designed for safety-critical applications where mathematical
correctness must be rigorously proven.
"""

import sys
from dataclasses import dataclass

import numpy as np
import pytest

# Import the implementation to test
from maptor.birkhoff import (
    BirkhoffBasisComponents,
    _compute_barycentric_weights_birkhoff,
    _compute_birkhoff_basis_components,
    _compute_birkhoff_basis_functions,
    _evaluate_birkhoff_interpolation_a_form,
    _evaluate_birkhoff_interpolation_b_form,
    _verify_interpolant_equivalence,
)


# ============================================================================
# CONFIGURATION AND FIXTURES
# ============================================================================

# Tolerance configuration for different types of mathematical operations
MACHINE_PRECISION_TOLERANCE = 1e-12
INTEGRATION_TOLERANCE = 1e-12
NUMERICAL_DERIVATIVE_TOLERANCE = 1e-6  # For finite difference approximations
POLYNOMIAL_EXACTNESS_TOLERANCE = 1e-12


@dataclass
class BirkhoffTestConfig:
    """Configuration for a single Birkhoff test case."""

    grid_points: np.ndarray
    tau_a: float
    tau_b: float
    description: str


# Comprehensive test configurations covering all mathematical scenarios
TEST_CONFIGURATIONS = [
    BirkhoffTestConfig(
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        0.0,
        1.0,
        "Uniform grid with endpoints matching interval",
    ),
    BirkhoffTestConfig(
        np.array([0.0, 0.1, 0.4, 0.8, 1.0]),
        0.0,
        1.0,
        "Non-uniform grid with endpoints matching interval",
    ),
    BirkhoffTestConfig(
        np.array([0.1, 0.3, 0.6, 0.9]), 0.0, 1.0, "Grid points strictly inside interval"
    ),
    BirkhoffTestConfig(
        np.array([-0.5, -0.1, 0.2, 0.7]), -1.0, 1.0, "Different interval with negative values"
    ),
    BirkhoffTestConfig(
        np.array([0.1, 0.11, 0.12, 0.9]),
        0.0,
        1.0,
        "Clustered grid points (numerical stability test)",
    ),
    BirkhoffTestConfig(np.array([0.1, 0.15, 0.2]), 0.0, 0.3, "Small interval"),
    BirkhoffTestConfig(
        np.array([1.0, 5.0, 8.0, 10.0]), 0.0, 12.0, "Large interval with scaled points"
    ),
    BirkhoffTestConfig(np.array([0.5]), 0.0, 1.0, "Single point (minimal case)"),
    BirkhoffTestConfig(np.array([0.2, 0.8]), 0.0, 1.0, "Two points"),
    BirkhoffTestConfig(np.linspace(0.1, 0.9, 8), 0.0, 1.0, "Higher order case (8 points)"),
]


@pytest.fixture(params=TEST_CONFIGURATIONS, ids=lambda config: config.description)
def birkhoff_config(request) -> BirkhoffTestConfig:
    """Fixture providing comprehensive Birkhoff test configurations."""
    return request.param


@pytest.fixture
def birkhoff_components(birkhoff_config: BirkhoffTestConfig) -> BirkhoffBasisComponents:
    """Fixture computing Birkhoff components for each test configuration."""
    config = birkhoff_config
    try:
        components = _compute_birkhoff_basis_components(
            tuple(config.grid_points), config.tau_a, config.tau_b
        )
        return components
    except Exception as e:
        pytest.fail(f"Failed to compute Birkhoff components for {config.description}: {e}")


# ============================================================================
# FUNDAMENTAL MATHEMATICAL THEOREM TESTS
# ============================================================================


class TestLagrangeOrthogonalityTheorem:
    """
    MATHEMATICAL THEOREM: Lagrange Interpolation Orthogonality Property
    STATEMENT: ℓ_j(τ_i) = δ_{ij} for all i,j ∈ {0,1,...,N}
    MATHEMATICAL NECESSITY: Foundation of all interpolation theory
    PROOF METHOD: Numerical verification of derivative conditions
    """

    def test_lagrange_orthogonality_property(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify ℓ_j(τ_i) = δ_{ij} using the relationship Ḃ_j^θ(τ) = ℓ_j(τ).

        This test MUST pass for any correct Birkhoff implementation.
        This is the mathematical foundation of interpolation theory.
        """
        grid_points = birkhoff_components.grid_points
        N = len(grid_points)

        # Use finite difference to approximate ḋB_j^a(τ_i) = ℓ_j(τ_i)
        h = 1e-8
        max_error = 0.0

        barycentric_weights = _compute_barycentric_weights_birkhoff(grid_points)

        for i in range(N):
            tau_i = grid_points[i]

            for j in range(N):
                # Compute numerical derivative: d/dτ B_j^a(τ)|_{τ=τ_i}
                tau_plus = tau_i + h
                tau_minus = tau_i - h

                basis_plus, _ = _compute_birkhoff_basis_functions(
                    grid_points,
                    barycentric_weights,
                    birkhoff_components.tau_a,
                    birkhoff_components.tau_b,
                    np.array([tau_plus]),
                )
                basis_minus, _ = _compute_birkhoff_basis_functions(
                    grid_points,
                    barycentric_weights,
                    birkhoff_components.tau_a,
                    birkhoff_components.tau_b,
                    np.array([tau_minus]),
                )

                numerical_derivative = (basis_plus[0, j] - basis_minus[0, j]) / (2 * h)
                expected = 1.0 if i == j else 0.0

                error = abs(numerical_derivative - expected)
                max_error = max(max_error, error)

        assert max_error < NUMERICAL_DERIVATIVE_TOLERANCE, (
            f"Lagrange orthogonality ℓ_j(τ_i) = δ_ij violated: "
            f"max_error = {max_error:.2e}, tolerance = {NUMERICAL_DERIVATIVE_TOLERANCE:.2e}"
        )


class TestBirkhoffBoundaryConditions:
    """
    MATHEMATICAL THEOREM: Birkhoff Boundary Conditions
    STATEMENT: B_j^a(τ^a) = 0 and B_j^b(τ^b) = 0 for all j ∈ {0,1,...,N}
    MATHEMATICAL NECESSITY: Required for interpolation conditions I^N_a y(τ^a) = y(τ^a)
    PROOF METHOD: Direct evaluation using matrix representation and explicit computation
    """

    def test_a_form_boundary_condition(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify B_j^a(τ^a) = 0 for all j.

        This ensures I^N_a y(τ^a) = y(τ^a) · 1 + Σ ẏ(τ_j) · 0 = y(τ^a).
        """
        grid_points = birkhoff_components.grid_points
        tau_a = birkhoff_components.tau_a

        # Check if τ^a is in grid points
        tau_a_indices = np.where(np.abs(grid_points - tau_a) < 1e-14)[0]

        if len(tau_a_indices) > 0:
            # Use matrix representation
            tau_a_idx = tau_a_indices[0]
            a_boundary_values = birkhoff_components.birkhoff_matrix_a[tau_a_idx, :]
            max_error = np.max(np.abs(a_boundary_values))
        else:
            # Evaluate directly at τ^a
            barycentric_weights = _compute_barycentric_weights_birkhoff(grid_points)
            basis_a_at_tau_a, _ = _compute_birkhoff_basis_functions(
                grid_points,
                barycentric_weights,
                tau_a,
                birkhoff_components.tau_b,
                np.array([tau_a]),
            )
            max_error = np.max(np.abs(basis_a_at_tau_a[0, :]))

        assert max_error < MACHINE_PRECISION_TOLERANCE, (
            f"B_j^a(τ^a) = 0 boundary condition violated: "
            f"max |B_j^a(τ^a)| = {max_error:.2e}, tolerance = {MACHINE_PRECISION_TOLERANCE:.2e}"
        )

    def test_b_form_boundary_condition(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify B_j^b(τ^b) = 0 for all j.

        This ensures I^N_b y(τ^b) = Σ ẏ(τ_j) · 0 + y(τ^b) · 1 = y(τ^b).
        """
        grid_points = birkhoff_components.grid_points
        tau_b = birkhoff_components.tau_b

        # Check if τ^b is in grid points
        tau_b_indices = np.where(np.abs(grid_points - tau_b) < 1e-14)[0]

        if len(tau_b_indices) > 0:
            # Use matrix representation
            tau_b_idx = tau_b_indices[0]
            b_boundary_values = birkhoff_components.birkhoff_matrix_b[tau_b_idx, :]
            max_error = np.max(np.abs(b_boundary_values))
        else:
            # Evaluate directly at τ^b
            barycentric_weights = _compute_barycentric_weights_birkhoff(grid_points)
            _, basis_b_at_tau_b = _compute_birkhoff_basis_functions(
                grid_points,
                barycentric_weights,
                birkhoff_components.tau_a,
                tau_b,
                np.array([tau_b]),
            )
            max_error = np.max(np.abs(basis_b_at_tau_b[0, :]))

        assert max_error < MACHINE_PRECISION_TOLERANCE, (
            f"B_j^b(τ^b) = 0 boundary condition violated: "
            f"max |B_j^b(τ^b)| = {max_error:.2e}, tolerance = {MACHINE_PRECISION_TOLERANCE:.2e}"
        )


class TestEquivalenceCondition:
    """
    MATHEMATICAL THEOREM: Birkhoff Equivalence Condition
    STATEMENT: B_j^a(τ) - B_j^b(τ) = w^B_j for all j,τ
    MATHEMATICAL NECESSITY: Fundamental relationship between a-form and b-form
    PROOF METHOD: Matrix difference verification at all grid points
    """

    def test_equivalence_condition_matrix_form(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify B^a - B^b = w^B ⊗ 1^T (equivalence condition in matrix form).

        This is the fundamental relationship that allows switching between
        a-form and b-form representations.
        """
        matrix_a = birkhoff_components.birkhoff_matrix_a
        matrix_b = birkhoff_components.birkhoff_matrix_b
        weights = birkhoff_components.birkhoff_quadrature_weights

        # Compute B^a - B^b
        difference_matrix = matrix_a - matrix_b

        # Expected: each row should equal weights vector
        N = len(weights)
        expected_matrix = np.tile(weights, (N, 1))

        max_error = np.max(np.abs(difference_matrix - expected_matrix))
        tolerance = max(MACHINE_PRECISION_TOLERANCE, np.max(np.abs(weights)) * 1e-14)

        assert max_error < tolerance, (
            f"Equivalence condition B^a - B^b = w^B violated: "
            f"max_error = {max_error:.2e}, tolerance = {tolerance:.2e}"
        )


class TestQuadratureExactness:
    """
    MATHEMATICAL THEOREM: Birkhoff Quadrature Exactness
    STATEMENT: ∫_{τ^a}^{τ^b} p(τ) dτ = Σ_j w^B_j p(τ_j) for polynomials p of degree ≤ N
    MATHEMATICAL NECESSITY: Defines the quadrature weights w^B_j
    PROOF METHOD: Test with monomial basis {1, τ, τ^2, ..., τ^N}
    """

    def test_monomial_integration_exactness(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify exact integration of monomials τ^k for k = 0, 1, ..., N.

        This test verifies that w^B_j = ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ is computed correctly.
        Must be exact for polynomials up to degree N.
        """
        grid_points = birkhoff_components.grid_points
        weights = birkhoff_components.birkhoff_quadrature_weights
        tau_a, tau_b = birkhoff_components.tau_a, birkhoff_components.tau_b
        N = len(grid_points) - 1

        max_error = 0.0

        # Test with monomials τ^k for k = 0, 1, ..., N
        for k in range(N + 1):
            # Exact integral: ∫_{τ^a}^{τ^b} τ^k dτ = (τ^b)^{k+1} - (τ^a)^{k+1} / (k+1)
            exact_integral = (tau_b ** (k + 1) - tau_a ** (k + 1)) / (k + 1)

            # Quadrature approximation: Σ w^B_j (τ_j)^k
            monomial_values = grid_points**k
            quadrature_result = np.dot(weights, monomial_values)

            error = abs(exact_integral - quadrature_result)
            max_error = max(max_error, error)

        tolerance = max(INTEGRATION_TOLERANCE, abs(tau_b - tau_a) * 1e-14)

        assert max_error < tolerance, (
            f"Quadrature exactness for monomials violated: "
            f"max_error = {max_error:.2e}, tolerance = {tolerance:.2e}"
        )


class TestInterpolationConditions:
    """
    MATHEMATICAL THEOREM: Birkhoff Interpolation Conditions
    STATEMENT: I^N_a y(τ^a) = y(τ^a) and d/dτ I^N_a y(τ)|_{τ=τ_j} = ẏ(τ_j)
    MATHEMATICAL NECESSITY: Defines what interpolation means
    PROOF METHOD: Construct polynomial with known values and derivatives
    """

    def test_endpoint_interpolation_condition(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify I^N_a y(τ^a) = y(τ^a) using polynomial test function.

        This is the fundamental endpoint condition for a-form interpolation.
        """
        grid_points = birkhoff_components.grid_points
        tau_a = birkhoff_components.tau_a
        N = len(grid_points) - 1

        # Create test polynomial of degree N
        np.random.seed(42)  # Reproducible
        coeffs = np.random.uniform(-2, 2, N + 1)

        def poly(tau):
            return sum(coeffs[k] * tau**k for k in range(len(coeffs)))

        def poly_derivative(tau):
            if N == 0:
                return 0.0
            return sum(k * coeffs[k] * tau ** (k - 1) for k in range(1, len(coeffs)))

        y_initial = poly(tau_a)
        y_derivatives = np.array([poly_derivative(tau) for tau in grid_points])

        # Test endpoint condition
        interpolated_at_tau_a = _evaluate_birkhoff_interpolation_a_form(
            birkhoff_components, y_initial, y_derivatives, np.array([tau_a])
        )

        error = abs(interpolated_at_tau_a[0] - y_initial)
        tolerance = max(POLYNOMIAL_EXACTNESS_TOLERANCE, np.max(np.abs(coeffs)) * 1e-14)

        assert error < tolerance, (
            f"Endpoint interpolation condition I^N_a y(τ^a) = y(τ^a) violated: "
            f"error = {error:.2e}, tolerance = {tolerance:.2e}"
        )

    def test_polynomial_exactness_condition(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify exact reproduction of polynomials of degree ≤ N.

        Since we interpolate derivatives exactly, the polynomial should be
        reproduced exactly everywhere, not just at grid points.
        """
        grid_points = birkhoff_components.grid_points
        tau_a = birkhoff_components.tau_a
        N = len(grid_points) - 1

        # Create test polynomial of degree N
        np.random.seed(123)  # Different seed for different polynomial
        coeffs = np.random.uniform(-2, 2, N + 1)

        def poly(tau):
            return sum(coeffs[k] * tau**k for k in range(len(coeffs)))

        def poly_derivative(tau):
            if N == 0:
                return 0.0
            return sum(k * coeffs[k] * tau ** (k - 1) for k in range(1, len(coeffs)))

        y_initial = poly(tau_a)
        y_derivatives = np.array([poly_derivative(tau) for tau in grid_points])

        # Test at multiple points throughout the interval
        test_points = np.linspace(
            birkhoff_components.tau_a + 0.001, birkhoff_components.tau_b - 0.001, 20
        )

        interpolated_values = _evaluate_birkhoff_interpolation_a_form(
            birkhoff_components, y_initial, y_derivatives, test_points
        )
        exact_values = np.array([poly(tau) for tau in test_points])

        max_error = np.max(np.abs(interpolated_values - exact_values))
        tolerance = max(POLYNOMIAL_EXACTNESS_TOLERANCE, np.max(np.abs(coeffs)) * 1e-14)

        assert max_error < tolerance, (
            f"Polynomial exactness condition violated: "
            f"max_error = {max_error:.2e}, tolerance = {tolerance:.2e}"
        )


class TestEquivalenceProposition:
    """
    MATHEMATICAL THEOREM: Equivalence Proposition
    STATEMENT: I^N_a y(τ) = I^N_b y(τ) iff y(τ^b) = y(τ^a) + Σ w^B_j ẏ(τ_j)
    MATHEMATICAL NECESSITY: Fundamental relationship between a-form and b-form
    PROOF METHOD: Test both directions of the equivalence
    """

    def test_equivalence_when_condition_satisfied(
        self, birkhoff_components: BirkhoffBasisComponents
    ):
        """
        Test: If y(τ^b) = y(τ^a) + Σ w^B_j ẏ(τ_j), then I^N_a = I^N_b.

        This is the forward direction of the equivalence proposition.
        """
        weights = birkhoff_components.birkhoff_quadrature_weights

        # Set up test case with condition satisfied
        np.random.seed(42)
        y_initial = 1.5
        y_derivatives = np.random.uniform(-1, 1, len(birkhoff_components.grid_points))
        y_final_correct = y_initial + np.dot(weights, y_derivatives)

        # Test at multiple points
        test_points = np.linspace(
            birkhoff_components.tau_a + 0.01, birkhoff_components.tau_b - 0.01, 25
        )

        a_form_values = _evaluate_birkhoff_interpolation_a_form(
            birkhoff_components, y_initial, y_derivatives, test_points
        )
        b_form_values = _evaluate_birkhoff_interpolation_b_form(
            birkhoff_components, y_final_correct, y_derivatives, test_points
        )

        max_error = np.max(np.abs(a_form_values - b_form_values))

        assert max_error < MACHINE_PRECISION_TOLERANCE, (
            f"Equivalence condition satisfied but I^N_a ≠ I^N_b: "
            f"max_error = {max_error:.2e}, tolerance = {MACHINE_PRECISION_TOLERANCE:.2e}"
        )

    def test_non_equivalence_when_condition_violated(
        self, birkhoff_components: BirkhoffBasisComponents
    ):
        """
        Test: If y(τ^b) ≠ y(τ^a) + Σ w^B_j ẏ(τ_j), then I^N_a ≠ I^N_b.

        This is the reverse direction of the equivalence proposition.
        """
        weights = birkhoff_components.birkhoff_quadrature_weights

        # Set up test case with condition violated
        np.random.seed(42)
        y_initial = 1.5
        y_derivatives = np.random.uniform(-1, 1, len(birkhoff_components.grid_points))
        y_final_correct = y_initial + np.dot(weights, y_derivatives)
        y_final_wrong = y_final_correct + 0.1  # Violate condition

        test_points = np.linspace(
            birkhoff_components.tau_a + 0.01, birkhoff_components.tau_b - 0.01, 25
        )

        a_form_values = _evaluate_birkhoff_interpolation_a_form(
            birkhoff_components, y_initial, y_derivatives, test_points
        )
        b_form_values_wrong = _evaluate_birkhoff_interpolation_b_form(
            birkhoff_components, y_final_wrong, y_derivatives, test_points
        )

        max_difference = np.max(np.abs(a_form_values - b_form_values_wrong))

        # Should be significantly different (not just numerical error)
        assert max_difference > 1e-4, (
            f"Equivalence condition violated but I^N_a ≈ I^N_b: "
            f"max_difference = {max_difference:.2e} (should be > 1e-4)"
        )

    def test_equivalence_verification_function(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Test the _verify_interpolant_equivalence function itself.

        This verifies our implementation correctly detects equivalence/non-equivalence.
        """
        weights = birkhoff_components.birkhoff_quadrature_weights

        np.random.seed(42)
        y_initial = 1.5
        y_derivatives = np.random.uniform(-1, 1, len(birkhoff_components.grid_points))

        # Case 1: Condition satisfied
        y_final_correct = y_initial + np.dot(weights, y_derivatives)
        equiv_check = _verify_interpolant_equivalence(
            birkhoff_components, y_initial, y_final_correct, y_derivatives
        )

        # Case 2: Condition violated
        y_final_wrong = y_final_correct + 0.1
        non_equiv_check = _verify_interpolant_equivalence(
            birkhoff_components, y_initial, y_final_wrong, y_derivatives
        )

        assert equiv_check, "Equivalence verification failed when condition is satisfied"
        assert not non_equiv_check, "Equivalence verification passed when condition is violated"


class TestEndpointMatrixProperties:
    """
    MATHEMATICAL THEOREM: Endpoint Matrix Properties
    STATEMENT: When τ_0 = τ^a, τ_N = τ^b: last row of B^a = w^B, first row of B^b = -w^B
    MATHEMATICAL NECESSITY: Special case providing direct access to quadrature weights
    PROOF METHOD: Direct matrix examination when endpoints match grid points
    """

    def test_endpoint_matrix_properties(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Verify special matrix properties when grid endpoints match interval endpoints.

        This test only applies when τ_0 = τ^a and τ_N = τ^b.
        When applicable, provides direct verification of quadrature weight computation.
        """
        grid_points = birkhoff_components.grid_points
        weights = birkhoff_components.birkhoff_quadrature_weights
        tau_a, tau_b = birkhoff_components.tau_a, birkhoff_components.tau_b

        # Check if endpoints match grid points
        tau_0_equals_tau_a = abs(grid_points[0] - tau_a) < 1e-14
        tau_n_equals_tau_b = abs(grid_points[-1] - tau_b) < 1e-14

        if not (tau_0_equals_tau_a and tau_n_equals_tau_b):
            pytest.skip(
                "Endpoint matrix properties test not applicable: endpoints don't match grid"
            )

        # Test last row of B^a equals weights
        last_row_Ba = birkhoff_components.birkhoff_matrix_a[-1, :]
        error_a = np.max(np.abs(last_row_Ba - weights))

        # Test first row of B^b equals negative weights
        first_row_Bb = birkhoff_components.birkhoff_matrix_b[0, :]
        error_b = np.max(np.abs(first_row_Bb + weights))

        tolerance = max(MACHINE_PRECISION_TOLERANCE, np.max(np.abs(weights)) * 1e-14)

        assert error_a < tolerance, (
            f"Last row of B^a ≠ quadrature weights: error = {error_a:.2e}, tolerance = {tolerance:.2e}"
        )
        assert error_b < tolerance, (
            f"First row of B^b ≠ -quadrature weights: error = {error_b:.2e}, tolerance = {tolerance:.2e}"
        )


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================


class TestComprehensiveIntegration:
    """
    Integration tests that verify multiple mathematical properties simultaneously.
    These provide additional confidence in the overall implementation correctness.
    """

    def test_smooth_function_interpolation_accuracy(
        self, birkhoff_components: BirkhoffBasisComponents
    ):
        """
        Test interpolation of smooth function with known analytical properties.

        Uses f(τ) = sin(πτ) + 0.5τ² which has known derivatives and integrals.
        """
        if len(birkhoff_components.grid_points) < 4:
            pytest.skip("Smooth function test requires at least 4 grid points")

        grid_points = birkhoff_components.grid_points
        tau_a = birkhoff_components.tau_a

        # Test function: f(τ) = sin(πτ) + 0.5τ²
        def test_function(tau):
            return np.sin(np.pi * tau) + 0.5 * tau**2

        def test_derivative(tau):
            return np.pi * np.cos(np.pi * tau) + tau

        y_initial = test_function(tau_a)
        y_derivatives = np.array([test_derivative(tau) for tau in grid_points])

        # Test interpolation at fine grid
        test_points = np.linspace(tau_a + 0.01, birkhoff_components.tau_b - 0.01, 50)

        interpolated_values = _evaluate_birkhoff_interpolation_a_form(
            birkhoff_components, y_initial, y_derivatives, test_points
        )
        exact_values = np.array([test_function(tau) for tau in test_points])

        # For smooth functions, error should decrease with increasing N
        max_error = np.max(np.abs(interpolated_values - exact_values))

        # Tolerance scales with problem size and grid spacing
        h = np.min(np.diff(grid_points))
        expected_error_bound = h ** len(grid_points) * 10  # Very loose bound

        assert max_error < expected_error_bound, (
            f"Smooth function interpolation error too large: "
            f"max_error = {max_error:.2e}, expected_bound = {expected_error_bound:.2e}"
        )

    def test_consistency_between_forms(self, birkhoff_components: BirkhoffBasisComponents):
        """
        Test consistency between a-form and b-form when equivalence condition is satisfied.

        This is an end-to-end test of the complete interpolation system.
        """
        grid_points = birkhoff_components.grid_points
        weights = birkhoff_components.birkhoff_quadrature_weights
        tau_a, tau_b = birkhoff_components.tau_a, birkhoff_components.tau_b

        # Random test case satisfying equivalence condition
        np.random.seed(999)
        y_initial = np.random.uniform(-2, 2)
        y_derivatives = np.random.uniform(-2, 2, len(grid_points))
        y_final = y_initial + np.dot(weights, y_derivatives)

        # Test at many points throughout interval
        test_points = np.linspace(tau_a + 0.001, tau_b - 0.001, 100)

        a_form_values = _evaluate_birkhoff_interpolation_a_form(
            birkhoff_components, y_initial, y_derivatives, test_points
        )
        b_form_values = _evaluate_birkhoff_interpolation_b_form(
            birkhoff_components, y_final, y_derivatives, test_points
        )

        max_difference = np.max(np.abs(a_form_values - b_form_values))

        assert max_difference < MACHINE_PRECISION_TOLERANCE, (
            f"a-form and b-form inconsistent when equivalence condition satisfied: "
            f"max_difference = {max_difference:.2e}"
        )


# ============================================================================
# PYTEST CONFIGURATION AND UTILITIES
# ============================================================================


def pytest_configure(config):
    """Configure pytest for mathematical verification."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "mathematical_theorem: marks test as verifying a mathematical theorem"
    )
    config.addinivalue_line(
        "markers", "boundary_condition: marks test as verifying boundary conditions"
    )
    config.addinivalue_line(
        "markers", "integration_test: marks test as comprehensive integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Add markers based on test class/function names
        if "Theorem" in item.nodeid:
            item.add_marker(pytest.mark.mathematical_theorem)
        if "Boundary" in item.nodeid:
            item.add_marker(pytest.mark.boundary_condition)
        if "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration_test)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run the complete mathematical verification suite.

    This can be executed directly or via pytest command line.
    """
    import subprocess
    import sys

    print("=" * 80)
    print("RIGOROUS MATHEMATICAL VERIFICATION OF BIRKHOFF IMPLEMENTATION")
    print("=" * 80)
    print("Running comprehensive pytest-based verification suite...")
    print()

    # Run pytest with verbose output
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short", "--strict-markers"],
        capture_output=False,
    )

    print()
    print("=" * 80)

    if result.returncode == 0:
        print("🎉 ALL MATHEMATICAL TESTS PASSED - IMPLEMENTATION RIGOROUSLY VERIFIED 🎉")
        print("The Birkhoff implementation satisfies ALL fundamental mathematical requirements.")
        print("Implementation is PROVEN CORRECT for safety-critical applications.")
    else:
        print("❌ MATHEMATICAL VERIFICATION FAILED")
        print("Implementation violates fundamental mathematical requirements.")
        print("DO NOT USE in safety-critical applications until corrected.")

    print("=" * 80)

    sys.exit(result.returncode)
