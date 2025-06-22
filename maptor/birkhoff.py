from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.integrate import quad

from .input_validation import _validate_positive_integer
from .mtor_types import FloatArray
from .utils.constants import NUMERICAL_ZERO


@dataclass
class BirkhoffBasisComponents:
    """Components for Birkhoff interpolation method as defined in Section 2 of the paper.

    Contains both a-form and b-form Birkhoff basis functions, quadrature weights,
    and matrices for arbitrary grid pseudospectral methods following the exact
    mathematical theory from the Birkhoff paper.
    """

    grid_points: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    tau_a: float = 0.0
    tau_b: float = 1.0

    # Lagrange basis functions and antiderivatives (L_j(τ) from Lemma)
    lagrange_antiderivatives_at_tau_a: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    lagrange_antiderivatives_at_tau_b: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # Birkhoff quadrature weights w^B_j := ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ (Definition, Section 2)
    birkhoff_quadrature_weights: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # Birkhoff basis functions B_j^a(τ) and B_j^b(τ) (Lemma, Section 2)
    birkhoff_basis_a: FloatArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    birkhoff_basis_b: FloatArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))

    # Birkhoff matrices B^a and B^b (Definition, Section 2)
    birkhoff_matrix_a: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    birkhoff_matrix_b: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )

    # Mathematical verification flags
    interpolation_conditions_verified: bool = False
    equivalence_condition_verified: bool = False
    endpoint_conditions_verified: bool = False


def _compute_barycentric_weights_birkhoff(nodes: FloatArray) -> FloatArray:
    """Compute barycentric weights for Lagrange interpolation on arbitrary grid.

    Mathematical foundation for Birkhoff basis construction. Required for
    computing ℓ_j(τ) which are the derivatives of Birkhoff basis functions.

    Mathematical requirement: ℓ_j(τ_i) = δ_{ij}
    """
    num_nodes = len(nodes)
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    nodes_col = nodes[:, np.newaxis]
    nodes_row = nodes[np.newaxis, :]
    differences_matrix = nodes_col - nodes_row

    diagonal_mask = np.eye(num_nodes, dtype=bool)

    near_zero_mask = np.abs(differences_matrix) < NUMERICAL_ZERO
    perturbation = np.sign(differences_matrix) * NUMERICAL_ZERO
    perturbation[perturbation == 0] = NUMERICAL_ZERO

    off_diagonal_near_zero = near_zero_mask & ~diagonal_mask
    differences_matrix = np.where(off_diagonal_near_zero, perturbation, differences_matrix)

    differences_matrix[diagonal_mask] = 1.0

    products = np.prod(differences_matrix, axis=1, dtype=np.float64)

    small_product_mask = np.abs(products) < NUMERICAL_ZERO**2
    safe_products = np.where(
        small_product_mask,
        np.where(products == 0, 1.0 / (NUMERICAL_ZERO**2), np.sign(products) / (NUMERICAL_ZERO**2)),
        1.0 / products,
    )

    return safe_products.astype(np.float64)


def _evaluate_lagrange_basis_at_point(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point: float,
) -> FloatArray:
    """Evaluate Lagrange basis functions ℓ_j(τ) at given point.

    Required for computing Birkhoff quadrature weights and verifying
    interpolation conditions. Must satisfy ℓ_j(τ_i) = δ_{ij}.
    """
    num_nodes = len(nodes)

    differences = np.abs(evaluation_point - nodes)
    coincident_mask = differences < NUMERICAL_ZERO

    if np.any(coincident_mask):
        lagrange_values = np.zeros(num_nodes, dtype=np.float64)
        lagrange_values[coincident_mask] = 1.0
        return lagrange_values

    diffs = evaluation_point - nodes

    near_zero_mask = np.abs(diffs) < NUMERICAL_ZERO
    safe_diffs = np.where(
        near_zero_mask, np.where(diffs == 0, NUMERICAL_ZERO, np.sign(diffs) * NUMERICAL_ZERO), diffs
    )

    terms = barycentric_weights / safe_diffs
    sum_terms = np.sum(terms)

    if abs(sum_terms) < NUMERICAL_ZERO:
        return np.zeros(num_nodes, dtype=np.float64)

    normalized_terms = terms / sum_terms
    return cast(FloatArray, normalized_terms)


def _compute_lagrange_antiderivative_at_point(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point: float,
    reference_point: float,
) -> FloatArray:
    """Compute antiderivatives L_j(τ) = ∫_{reference_point}^{evaluation_point} ℓ_j(s) ds.

    Critical for Birkhoff basis function construction per Lemma in Section 2:
    B_j^a(τ) = L_j(τ) - L_j(τ^a)
    B_j^b(τ) = L_j(τ) - L_j(τ^b)

    The reference_point cancels out in the difference expressions.
    """
    num_nodes = len(nodes)
    antiderivatives = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):

        def lagrange_j(tau):
            """Lagrange basis function ℓ_j(τ)"""
            lagrange_vals = _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)
            return lagrange_vals[j]

        # L_j(τ) = ∫_{reference_point}^{evaluation_point} ℓ_j(s) ds
        if abs(evaluation_point - reference_point) > NUMERICAL_ZERO:
            integral_result, _ = quad(
                lagrange_j, reference_point, evaluation_point, epsabs=1e-12, epsrel=1e-10, limit=200
            )
            antiderivatives[j] = integral_result

    return antiderivatives


def _compute_birkhoff_quadrature_weights(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
) -> FloatArray:
    """Compute Birkhoff quadrature weights w^B_j := ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ.

    From Definition in Section 2 of the paper.
    These weights satisfy the equivalence condition:
    B^a_j(τ) - B^b_j(τ) = w^B_j (Lemma, Section 2)
    """
    num_nodes = len(nodes)
    weights = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):

        def lagrange_j(tau):
            """Lagrange basis function ℓ_j(τ)"""
            lagrange_vals = _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)
            return lagrange_vals[j]

        # w^B_j := ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ
        integral_result, _ = quad(lagrange_j, tau_a, tau_b, epsabs=1e-12, epsrel=1e-10, limit=200)
        weights[j] = integral_result

    return weights


def _compute_birkhoff_basis_functions(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
    evaluation_points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute Birkhoff basis functions B_j^a(τ) and B_j^b(τ).

    From Lemma in Section 2:
    B_j^a(τ) = L_j(τ) - L_j(τ^a), j = 0, ..., N
    B_j^b(τ) = L_j(τ) - L_j(τ^b), j = 0, ..., N

    where L_j(τ) is the antiderivative of ℓ_j(τ).

    Mathematical requirement: These must satisfy the interpolation conditions.
    """
    num_nodes = len(nodes)
    num_eval_points = len(evaluation_points)

    # Initialize basis function matrices
    basis_a = np.zeros((num_eval_points, num_nodes), dtype=np.float64)
    basis_b = np.zeros((num_eval_points, num_nodes), dtype=np.float64)

    # Choose consistent reference point for integration
    reference_point = tau_a

    # Compute L_j(τ^a) and L_j(τ^b) for all j
    antiderivatives_at_tau_a = _compute_lagrange_antiderivative_at_point(
        nodes, barycentric_weights, tau_a, reference_point
    )
    antiderivatives_at_tau_b = _compute_lagrange_antiderivative_at_point(
        nodes, barycentric_weights, tau_b, reference_point
    )

    # Compute B_j^a(τ) and B_j^b(τ) for each evaluation point
    for i, tau in enumerate(evaluation_points):
        # Compute L_j(τ) for all j at this evaluation point
        antiderivatives_at_tau = _compute_lagrange_antiderivative_at_point(
            nodes, barycentric_weights, tau, reference_point
        )

        # B_j^a(τ) = L_j(τ) - L_j(τ^a)
        basis_a[i, :] = antiderivatives_at_tau - antiderivatives_at_tau_a

        # B_j^b(τ) = L_j(τ) - L_j(τ^b)
        basis_b[i, :] = antiderivatives_at_tau - antiderivatives_at_tau_b

    return basis_a, basis_b


def _construct_birkhoff_matrices(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
) -> tuple[FloatArray, FloatArray]:
    """Construct Birkhoff matrices B^a and B^b.

    From Definition in Section 2: B^θ_{ij} = B_j^θ(τ_i) where θ ∈ {a, b}.
    These matrices encode the basis function values at all grid points.
    """
    # Evaluate basis functions at grid points
    basis_a, basis_b = _compute_birkhoff_basis_functions(
        nodes, barycentric_weights, tau_a, tau_b, nodes
    )

    return basis_a, basis_b


def _verify_lagrange_interpolation_property(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
) -> bool:
    """Verify fundamental Lagrange interpolation property: ℓ_j(τ_i) = δ_{ij}.

    This is the foundation for all Birkhoff interpolation conditions.
    Mathematical requirement from paper: Ḃ_j^θ(τ_i) = ℓ_j(τ_i) = δ_{ij}
    """
    num_nodes = len(nodes)
    tolerance = 1e-10

    # Compute Lagrange basis functions at all grid points
    lagrange_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, tau in enumerate(nodes):
        lagrange_values = _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)
        lagrange_matrix[i, :] = lagrange_values

    # lagrange_matrix[i, j] = ℓ_j(τ_i) should equal δ_{ij}
    identity_matrix = np.eye(num_nodes)
    lagrange_error = np.max(np.abs(lagrange_matrix - identity_matrix))

    return lagrange_error < tolerance


def _verify_birkhoff_boundary_conditions(
    nodes: FloatArray,
    tau_a: float,
    tau_b: float,
    birkhoff_matrix_a: FloatArray,
    birkhoff_matrix_b: FloatArray,
) -> bool:
    """Verify Birkhoff boundary conditions from interpolation requirements.

    Mathematical requirements from Section 2:
    - B_j^a(τ^a) = 0 for j = 0, ..., N
    - B_j^b(τ^b) = 0 for j = 0, ..., N

    These ensure the interpolants satisfy I^N_a y(τ^a) = y(τ^a) and I^N_b y(τ^b) = y(τ^b).
    """
    tolerance = 1e-10

    # Check B_j^a(τ^a) = 0 for all j
    # Find index of τ^a in grid points
    tau_a_indices = np.where(np.abs(nodes - tau_a) < NUMERICAL_ZERO)[0]
    if len(tau_a_indices) > 0:
        tau_a_idx = tau_a_indices[0]
        a_boundary_error = np.max(np.abs(birkhoff_matrix_a[tau_a_idx, :]))
        if a_boundary_error > tolerance:
            return False

    # Check B_j^b(τ^b) = 0 for all j
    # Find index of τ^b in grid points
    tau_b_indices = np.where(np.abs(nodes - tau_b) < NUMERICAL_ZERO)[0]
    if len(tau_b_indices) > 0:
        tau_b_idx = tau_b_indices[0]
        b_boundary_error = np.max(np.abs(birkhoff_matrix_b[tau_b_idx, :]))
        if b_boundary_error > tolerance:
            return False

    return True


def _verify_birkhoff_derivative_conditions(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    birkhoff_matrix_a: FloatArray,
    birkhoff_matrix_b: FloatArray,
) -> bool:
    """Verify Birkhoff derivative conditions from interpolation requirements.

    Mathematical requirements from Section 2:
    Ḃ_j^θ(τ_i) = ℓ_j(τ_i) = δ_{ij} for θ ∈ {a, b}

    Since B_j^θ(τ) = L_j(τ) - L_j(τ^θ), we have Ḃ_j^θ(τ) = ℓ_j(τ).
    This is already verified by _verify_lagrange_interpolation_property.
    """
    # This condition is automatically satisfied if Lagrange interpolation is correct
    return _verify_lagrange_interpolation_property(nodes, barycentric_weights)


def _verify_equivalence_condition(
    birkhoff_quadrature_weights: FloatArray,
    birkhoff_matrix_a: FloatArray,
    birkhoff_matrix_b: FloatArray,
) -> bool:
    """Verify Birkhoff equivalence condition from Lemma in Section 2.

    Mathematical requirement: B^a_j(τ) - B^b_j(τ) = w^B_j for all j and τ.
    This is the fundamental relationship between a-form and b-form basis functions.
    """
    # Check if B^a - B^b equals the quadrature weights matrix
    difference_matrix = birkhoff_matrix_a - birkhoff_matrix_b
    weights_matrix = np.outer(
        np.ones(len(birkhoff_quadrature_weights)), birkhoff_quadrature_weights
    )

    # Use appropriate tolerance for numerical integration results
    max_error = np.max(np.abs(difference_matrix - weights_matrix))
    tolerance = max(1e-10, np.max(np.abs(birkhoff_quadrature_weights)) * 1e-12)

    return max_error < tolerance


def _verify_endpoint_conditions(
    nodes: FloatArray,
    tau_a: float,
    tau_b: float,
    birkhoff_matrix_a: FloatArray,
    birkhoff_matrix_b: FloatArray,
    birkhoff_quadrature_weights: FloatArray,
) -> bool:
    """Verify special endpoint conditions from Lemma in Section 2.

    When τ_0 = τ^a and τ_N = τ^b:
    - Last row of B^a equals quadrature weights
    - First row of B^b equals negative quadrature weights

    These provide direct access to quadrature weights from matrix structure.
    """
    tolerance = max(1e-10, np.max(np.abs(birkhoff_quadrature_weights)) * 1e-12)

    # Check if endpoints match grid points
    tau_0_equals_tau_a = abs(nodes[0] - tau_a) < NUMERICAL_ZERO
    tau_n_equals_tau_b = abs(nodes[-1] - tau_b) < NUMERICAL_ZERO

    if tau_0_equals_tau_a and tau_n_equals_tau_b:
        # Last row of B^a should equal quadrature weights
        last_row_error = np.max(np.abs(birkhoff_matrix_a[-1, :] - birkhoff_quadrature_weights))

        # First row of B^b should equal negative quadrature weights
        first_row_error = np.max(np.abs(birkhoff_matrix_b[0, :] + birkhoff_quadrature_weights))

        return (last_row_error < tolerance) and (first_row_error < tolerance)

    return True  # Conditions don't apply if endpoints don't match


@functools.lru_cache(maxsize=32)
def _compute_birkhoff_basis_components(
    grid_points_tuple: tuple[float, ...],
    tau_a: float,
    tau_b: float,
) -> BirkhoffBasisComponents:
    """Compute complete Birkhoff basis components for arbitrary grid.

    Implements the complete mathematical theory from Section 2 of the Birkhoff paper
    with rigorous verification of all mathematical conditions and requirements.

    Mathematical foundations:
    1. Lagrange basis functions ℓ_j(τ) with ℓ_j(τ_i) = δ_{ij}
    2. Antiderivatives L_j(τ) = ∫ ℓ_j(s) ds
    3. Birkhoff basis functions: B_j^a(τ) = L_j(τ) - L_j(τ^a), B_j^b(τ) = L_j(τ) - L_j(τ^b)
    4. Quadrature weights: w^B_j = ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ
    5. Equivalence condition: B^a_j(τ) - B^b_j(τ) = w^B_j

    Args:
        grid_points_tuple: Arbitrary grid points π^N = {τ_0, τ_1, ..., τ_N}
        tau_a: Left endpoint of interval [τ^a, τ^b]
        tau_b: Right endpoint of interval [τ^a, τ^b]

    Returns:
        BirkhoffBasisComponents with complete verified mathematical objects

    Raises:
        ValueError: If grid points violate mathematical requirements or
                   if any mathematical condition fails verification
    """
    _validate_positive_integer(len(grid_points_tuple), "number of grid points")

    # Convert to numpy array and validate mathematical requirements
    grid_points = np.array(grid_points_tuple, dtype=np.float64)

    # Validate interval ordering: τ^a < τ^b (fundamental requirement)
    if tau_b <= tau_a:
        raise ValueError(f"Interval endpoints must satisfy τ^a < τ^b, got τ^a={tau_a}, τ^b={tau_b}")

    # Validate grid ordering: τ^a ≤ τ_0 < τ_1 < ... < τ_N ≤ τ^b (Equation 2 in paper)
    if not np.all(np.diff(grid_points) > 0):
        raise ValueError("Grid points must be strictly increasing: τ_0 < τ_1 < ... < τ_N")
    if grid_points[0] < tau_a or grid_points[-1] > tau_b:
        raise ValueError("Grid points must satisfy τ^a ≤ τ_0 < ... < τ_N ≤ τ^b")

    # Compute barycentric weights for Lagrange interpolation
    barycentric_weights = _compute_barycentric_weights_birkhoff(grid_points)

    # Verify fundamental Lagrange interpolation property: ℓ_j(τ_i) = δ_{ij}
    if not _verify_lagrange_interpolation_property(grid_points, barycentric_weights):
        raise ValueError("Lagrange interpolation property ℓ_j(τ_i) = δ_{ij} not satisfied")

    # Compute Birkhoff quadrature weights: w^B_j := ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ
    birkhoff_quadrature_weights = _compute_birkhoff_quadrature_weights(
        grid_points, barycentric_weights, tau_a, tau_b
    )

    # Use consistent reference point for all antiderivative computations
    reference_point = tau_a

    # Compute antiderivatives at endpoints for basis function construction
    antiderivatives_at_tau_a = _compute_lagrange_antiderivative_at_point(
        grid_points, barycentric_weights, tau_a, reference_point
    )
    antiderivatives_at_tau_b = _compute_lagrange_antiderivative_at_point(
        grid_points, barycentric_weights, tau_b, reference_point
    )

    # Construct Birkhoff matrices: B^θ_{ij} = B_j^θ(τ_i)
    birkhoff_matrix_a, birkhoff_matrix_b = _construct_birkhoff_matrices(
        grid_points, barycentric_weights, tau_a, tau_b
    )

    # Compute basis functions over grid points for general evaluation
    basis_a, basis_b = _compute_birkhoff_basis_functions(
        grid_points, barycentric_weights, tau_a, tau_b, grid_points
    )

    # COMPREHENSIVE MATHEMATICAL VERIFICATION per paper requirements

    # 1. Verify Birkhoff boundary conditions: B_j^a(τ^a) = 0, B_j^b(τ^b) = 0
    if not _verify_birkhoff_boundary_conditions(
        grid_points, tau_a, tau_b, birkhoff_matrix_a, birkhoff_matrix_b
    ):
        raise ValueError(
            "Birkhoff boundary conditions B_j^a(τ^a) = 0, B_j^b(τ^b) = 0 not satisfied"
        )

    # 2. Verify derivative conditions: Ḃ_j^θ(τ_i) = δ_{ij}
    if not _verify_birkhoff_derivative_conditions(
        grid_points, barycentric_weights, birkhoff_matrix_a, birkhoff_matrix_b
    ):
        raise ValueError("Birkhoff derivative conditions Ḃ_j^θ(τ_i) = δ_{ij} not satisfied")

    # 3. Verify equivalence condition: B^a_j(τ) - B^b_j(τ) = w^B_j (Lemma, Section 2)
    if not _verify_equivalence_condition(
        birkhoff_quadrature_weights, birkhoff_matrix_a, birkhoff_matrix_b
    ):
        raise ValueError("Birkhoff equivalence condition B^a - B^b = w^B not satisfied")

    # 4. Verify special endpoint conditions when applicable (Lemma, Section 2)
    if not _verify_endpoint_conditions(
        grid_points, tau_a, tau_b, birkhoff_matrix_a, birkhoff_matrix_b, birkhoff_quadrature_weights
    ):
        raise ValueError("Birkhoff endpoint conditions not satisfied")

    return BirkhoffBasisComponents(
        grid_points=grid_points,
        tau_a=tau_a,
        tau_b=tau_b,
        lagrange_antiderivatives_at_tau_a=antiderivatives_at_tau_a,
        lagrange_antiderivatives_at_tau_b=antiderivatives_at_tau_b,
        birkhoff_quadrature_weights=birkhoff_quadrature_weights,
        birkhoff_basis_a=basis_a,
        birkhoff_basis_b=basis_b,
        birkhoff_matrix_a=birkhoff_matrix_a,
        birkhoff_matrix_b=birkhoff_matrix_b,
        interpolation_conditions_verified=True,
        equivalence_condition_verified=True,
        endpoint_conditions_verified=True,
    )


def _evaluate_birkhoff_interpolation_a_form(
    basis_components: BirkhoffBasisComponents,
    y_initial: float,
    y_derivatives: FloatArray,
    evaluation_points: FloatArray,
) -> FloatArray:
    """Evaluate a-form Birkhoff interpolant: I^N_a y(τ).

    From equation (4a) in Section 2:
    I^N_a y(τ) := y(τ^a) B_0^0(τ) + Σ_{j=0}^N ẏ(τ_j) B_j^a(τ)

    From Lemma in Section 2: B^0_0(τ) = 1 (constant function)
    Therefore: I^N_a y(τ) = y(τ^a) · 1 + Σ_{j=0}^N ẏ(τ_j) B_j^a(τ)
                          = y(τ^a) + Σ_{j=0}^N ẏ(τ_j) B_j^a(τ)

    This satisfies the interpolation conditions:
    - I^N_a y(τ^a) = y(τ^a) [since B_j^a(τ^a) = 0]
    - d/dτ(I^N_a y(τ))|_{τ=τ_i} = ẏ(τ_i) [since Ḃ_j^a(τ_i) = δ_{ij}]
    """
    if len(y_derivatives) != len(basis_components.grid_points):
        raise ValueError(
            f"y_derivatives length {len(y_derivatives)} must match grid points {len(basis_components.grid_points)}"
        )

    num_eval_points = len(evaluation_points)
    interpolated_values = np.zeros(num_eval_points, dtype=np.float64)

    # Compute basis functions at evaluation points
    nodes = basis_components.grid_points
    barycentric_weights = _compute_barycentric_weights_birkhoff(nodes)
    basis_a, _ = _compute_birkhoff_basis_functions(
        nodes,
        barycentric_weights,
        basis_components.tau_a,
        basis_components.tau_b,
        evaluation_points,
    )

    # I^N_a y(τ) = y(τ^a) + Σ_{j=0}^N ẏ(τ_j) B_j^a(τ)
    # Note: B^0_0(τ) = 1 from Lemma, so y(τ^a) B_0^0(τ) = y(τ^a)
    for i in range(num_eval_points):
        interpolated_values[i] = y_initial + np.dot(basis_a[i, :], y_derivatives)

    return interpolated_values


def _evaluate_birkhoff_interpolation_b_form(
    basis_components: BirkhoffBasisComponents,
    y_final: float,
    y_derivatives: FloatArray,
    evaluation_points: FloatArray,
) -> FloatArray:
    """Evaluate b-form Birkhoff interpolant: I^N_b y(τ).

    From equation (4b) in Section 2:
    I^N_b y(τ) := Σ_{j=0}^N ẏ(τ_j) B_j^b(τ) + y(τ^b) B_N^N(τ)

    From Lemma in Section 2: B^N_N(τ) = 1 (constant function)
    Therefore: I^N_b y(τ) = Σ_{j=0}^N ẏ(τ_j) B_j^b(τ) + y(τ^b) · 1
                          = Σ_{j=0}^N ẏ(τ_j) B_j^b(τ) + y(τ^b)

    This satisfies the interpolation conditions:
    - I^N_b y(τ^b) = y(τ^b) [since B_j^b(τ^b) = 0]
    - d/dτ(I^N_b y(τ))|_{τ=τ_i} = ẏ(τ_i) [since Ḃ_j^b(τ_i) = δ_{ij}]
    """
    if len(y_derivatives) != len(basis_components.grid_points):
        raise ValueError(
            f"y_derivatives length {len(y_derivatives)} must match grid points {len(basis_components.grid_points)}"
        )

    num_eval_points = len(evaluation_points)
    interpolated_values = np.zeros(num_eval_points, dtype=np.float64)

    # Compute basis functions at evaluation points
    nodes = basis_components.grid_points
    barycentric_weights = _compute_barycentric_weights_birkhoff(nodes)
    _, basis_b = _compute_birkhoff_basis_functions(
        nodes,
        barycentric_weights,
        basis_components.tau_a,
        basis_components.tau_b,
        evaluation_points,
    )

    # I^N_b y(τ) = Σ_{j=0}^N ẏ(τ_j) B_j^b(τ) + y(τ^b)
    # Note: B^N_N(τ) = 1 from Lemma, so y(τ^b) B_N^N(τ) = y(τ^b)
    for i in range(num_eval_points):
        interpolated_values[i] = np.dot(basis_b[i, :], y_derivatives) + y_final

    return interpolated_values
