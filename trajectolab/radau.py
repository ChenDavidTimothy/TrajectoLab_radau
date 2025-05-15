from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray  # For specific NumPy array typing
from scipy.special import roots_jacobi

# Tolerance for floating point comparisons
ZERO_TOLERANCE = 1e-12


@dataclass
class RadauBasisComponents:
    """
    Data class to store all components required for Radau collocation.
    """

    state_approximation_nodes: Optional[NDArray[np.float64]] = None
    collocation_nodes: Optional[NDArray[np.float64]] = None
    quadrature_weights: Optional[NDArray[np.float64]] = None
    differentiation_matrix: Optional[NDArray[np.float64]] = None
    barycentric_weights_for_state_nodes: Optional[NDArray[np.float64]] = None
    lagrange_at_tau_plus_one: Optional[NDArray[np.float64]] = None


@dataclass
class RadauNodesAndWeights:
    """
    Data class to store the initial set of nodes and weights for LGR scheme.
    """

    state_approximation_nodes: Optional[NDArray[np.float64]] = None
    collocation_nodes: Optional[NDArray[np.float64]] = None
    quadrature_weights: Optional[NDArray[np.float64]] = None


def compute_legendre_gauss_radau_nodes_and_weights(
    num_collocation_nodes: int,
) -> RadauNodesAndWeights:
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ValueError("Number of collocation points must be an integer >= 1.")

    # Initialize with the fixed left endpoint
    collocation_nodes: NDArray[np.float64] = np.array([-1.0], dtype=np.float64)
    quadrature_weights: NDArray[np.float64]

    if num_collocation_nodes == 1:
        # Only one collocation point case: node is -1, weight is 2
        quadrature_weights = np.array([2.0], dtype=np.float64)
    else:
        # Multi-point case: compute interior roots and weights
        num_interior_roots = num_collocation_nodes - 1
        interior_roots_raw, jacobi_weights_raw, _ = roots_jacobi(num_interior_roots, 0, 1, mu=True)

        # Ensure correct types from roots_jacobi if they are not already ndarray of float
        interior_roots: NDArray[np.float64] = np.asarray(interior_roots_raw, dtype=np.float64)
        jacobi_weights: NDArray[np.float64] = np.asarray(jacobi_weights_raw, dtype=np.float64)

        interior_weights: NDArray[np.float64] = jacobi_weights * 2.0 / (1.0 + interior_roots)
        left_endpoint_weight: float = 2.0 / (num_collocation_nodes**2)

        collocation_nodes = np.concatenate([collocation_nodes, interior_roots])
        quadrature_weights = np.concatenate(
            [np.array([left_endpoint_weight], dtype=np.float64), interior_weights]
        )

    state_approximation_nodes: NDArray[np.float64] = np.concatenate(
        [collocation_nodes, np.array([1.0], dtype=np.float64)]
    )

    return RadauNodesAndWeights(
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
    )


def compute_barycentric_weights(nodes: NDArray[np.float64]) -> NDArray[np.float64]:
    num_nodes = len(nodes)
    barycentric_weights: NDArray[np.float64] = np.ones(num_nodes, dtype=np.float64)

    for j in range(num_nodes):
        node_differences: NDArray[np.float64] = nodes[j] - np.delete(nodes, j)

        abs_diff = np.abs(node_differences)
        too_small_mask = abs_diff < (ZERO_TOLERANCE * 1e-1)

        if np.any(too_small_mask):
            signs = np.sign(node_differences[too_small_mask])
            signs[signs == 0] = 1
            node_differences[too_small_mask] = signs * ZERO_TOLERANCE * 1e-1

        product_of_differences: np.float64 = np.prod(node_differences)
        if abs(product_of_differences) < ZERO_TOLERANCE * 1e-10:  # Avoid division by zero
            barycentric_weights[j] = (
                np.inf
                if product_of_differences == 0
                else (1.0 / (np.sign(product_of_differences) * ZERO_TOLERANCE * 1e-10))
            )
        else:
            barycentric_weights[j] = 1.0 / product_of_differences

    return barycentric_weights


def evaluate_lagrange_polynomial_at_point(
    polynomial_definition_nodes: NDArray[np.float64],
    barycentric_weights: NDArray[np.float64],
    evaluation_point_tau: float,
) -> NDArray[np.float64]:
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_polynomial_values_at_evaluation_point: NDArray[np.float64] = np.zeros(
        num_polynomial_definition_nodes, dtype=np.float64
    )

    for j in range(num_polynomial_definition_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_polynomial_values_at_evaluation_point[j] = 1.0
            return lagrange_polynomial_values_at_evaluation_point

    weighted_inverse_evaluation_point_differences: NDArray[np.float64] = np.zeros(
        num_polynomial_definition_nodes, dtype=np.float64
    )

    for j in range(num_polynomial_definition_nodes):
        term_denominator: float = evaluation_point_tau - polynomial_definition_nodes[j]
        if abs(term_denominator) < ZERO_TOLERANCE:
            term_denominator = (
                np.sign(term_denominator) * ZERO_TOLERANCE
                if term_denominator != 0
                else ZERO_TOLERANCE
            )
            if abs(term_denominator) < ZERO_TOLERANCE:
                term_denominator = ZERO_TOLERANCE

        weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / term_denominator

    barycentric_sum_denominator: np.float64 = np.sum(weighted_inverse_evaluation_point_differences)

    if abs(barycentric_sum_denominator) < ZERO_TOLERANCE:
        return lagrange_polynomial_values_at_evaluation_point

    lagrange_polynomial_values_at_evaluation_point = (
        weighted_inverse_evaluation_point_differences / barycentric_sum_denominator
    )
    return lagrange_polynomial_values_at_evaluation_point


def compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: NDArray[np.float64],
    barycentric_weights: NDArray[np.float64],
    evaluation_point_tau: float,  # This was the source of arg-type error, ensure it's float
) -> NDArray[np.float64]:
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_derivative_coefficients: NDArray[np.float64] = np.zeros(
        num_polynomial_definition_nodes, dtype=np.float64
    )
    matched_node_index: int = -1

    for current_node_idx in range(num_polynomial_definition_nodes):
        if (
            abs(evaluation_point_tau - polynomial_definition_nodes[current_node_idx])
            < ZERO_TOLERANCE
        ):
            matched_node_index = current_node_idx
            break

    if matched_node_index != -1:
        m = matched_node_index
        sum_for_diagonal_derivative_coefficient: np.float64 = 0.0  # Explicit type
        for j in range(num_polynomial_definition_nodes):
            if j == m:
                continue

            node_difference_denominator: np.float64 = (  # Explicit type
                polynomial_definition_nodes[m] - polynomial_definition_nodes[j]
            )

            if (
                abs(node_difference_denominator) < ZERO_TOLERANCE
                or abs(barycentric_weights[m]) < ZERO_TOLERANCE
            ):
                lagrange_derivative_coefficients[j] = 0.0
            else:
                if abs(barycentric_weights[j]) < ZERO_TOLERANCE:
                    lagrange_derivative_coefficients[j] = 0.0
                else:
                    lagrange_derivative_coefficients[j] = (
                        barycentric_weights[j] / barycentric_weights[m]
                    ) / node_difference_denominator
            sum_for_diagonal_derivative_coefficient += lagrange_derivative_coefficients[j]

        lagrange_derivative_coefficients[m] = -sum_for_diagonal_derivative_coefficient
    else:
        pass

    return lagrange_derivative_coefficients


def compute_radau_collocation_components(num_collocation_nodes: int) -> RadauBasisComponents:
    lgr_data = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)

    if (
        lgr_data.state_approximation_nodes is None
        or lgr_data.collocation_nodes is None
        or lgr_data.quadrature_weights is None
    ):
        raise ValueError("Failed to compute LGR nodes and weights properly: attributes are None.")

    state_nodes: NDArray[np.float64] = lgr_data.state_approximation_nodes
    collocation_nodes: NDArray[np.float64] = lgr_data.collocation_nodes
    quadrature_weights: NDArray[np.float64] = lgr_data.quadrature_weights

    num_state_nodes = len(state_nodes)
    num_actual_collocation_nodes = len(collocation_nodes)

    if num_state_nodes != num_collocation_nodes + 1:
        raise ValueError(
            f"Mismatch in expected number of state approximation (basis) points. "
            f"Expected {num_collocation_nodes + 1}, Got {num_state_nodes}"
        )
    if num_actual_collocation_nodes != num_collocation_nodes:
        raise ValueError(
            f"Mismatch in expected number of collocation points. "
            f"Expected {num_collocation_nodes}, Got {num_actual_collocation_nodes}"
        )

    bary_weights_state_nodes: NDArray[np.float64] = compute_barycentric_weights(state_nodes)

    diff_matrix: NDArray[np.float64] = np.zeros(
        (num_actual_collocation_nodes, num_state_nodes), dtype=np.float64
    )

    for i in range(num_actual_collocation_nodes):
        evaluation_point: float = float(collocation_nodes[i])  # Ensure it's treated as float
        diff_matrix[i, :] = compute_lagrange_derivative_coefficients_at_point(
            state_nodes, bary_weights_state_nodes, evaluation_point
        )

    lagrange_at_tau_plus_one: NDArray[np.float64] = np.zeros(num_state_nodes, dtype=np.float64)

    if abs(state_nodes[-1] - 1.0) < ZERO_TOLERANCE:
        lagrange_at_tau_plus_one[-1] = 1.0
    else:
        tau_plus_one_indices = np.where(np.isclose(state_nodes, 1.0, atol=ZERO_TOLERANCE))[0]
        if len(tau_plus_one_indices) == 1:
            lagrange_at_tau_plus_one[tau_plus_one_indices[0]] = 1.0
        elif len(tau_plus_one_indices) > 1:
            print("Warning: Multiple basis points found close to +1.0. Using the last one found.")
            lagrange_at_tau_plus_one[tau_plus_one_indices[-1]] = 1.0
        else:
            print("Warning: +1.0 not found precisely in state_nodes. Interpolating as fallback.")
            lagrange_at_tau_plus_one = evaluate_lagrange_polynomial_at_point(
                state_nodes, bary_weights_state_nodes, 1.0
            )

    return RadauBasisComponents(
        differentiation_matrix=diff_matrix,
        state_approximation_nodes=state_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
        barycentric_weights_for_state_nodes=bary_weights_state_nodes,
        lagrange_at_tau_plus_one=lagrange_at_tau_plus_one,
    )
