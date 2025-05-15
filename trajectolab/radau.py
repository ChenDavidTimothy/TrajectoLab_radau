from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_jacobi

# Type aliases
_FloatArray: TypeAlias = NDArray[np.float64]
_IntArray: TypeAlias = NDArray[np.int64]
_ArrayLike: TypeAlias = Sequence[float] | _FloatArray

# Tolerance for floating point comparisons
ZERO_TOLERANCE: float = 1e-12


@dataclass
class RadauBasisComponents:
    # Use field() with default_factory to create proper default values
    differentiation_matrix: _FloatArray | None = field(default=None)
    state_approximation_nodes: _FloatArray | None = field(default=None)
    collocation_nodes: _FloatArray | None = field(default=None)
    quadrature_weights: _FloatArray | None = field(default=None)
    barycentric_weights_for_state_nodes: _FloatArray | None = field(default=None)
    lagrange_at_tau_plus_one: _FloatArray | None = field(default=None)


@dataclass
class RadauNodesAndWeights:
    state_approximation_nodes: _FloatArray | None = field(default=None)
    collocation_nodes: _FloatArray | None = field(default=None)
    quadrature_weights: _FloatArray | None = field(default=None)


def compute_legendre_gauss_radau_nodes_and_weights(
    num_collocation_nodes: int,
) -> RadauNodesAndWeights:
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ValueError("Number of collocation points must be an integer >= 1.")

    # Initialize with the fixed left endpoint
    collocation_nodes = np.array([-1.0], dtype=np.float64)

    if num_collocation_nodes == 1:
        # Only one collocation point case
        quadrature_weights = np.array([2.0], dtype=np.float64)
    else:
        # Multi-point case: compute interior roots and weights
        num_interior_roots = num_collocation_nodes - 1
        interior_roots, jacobi_weights, _ = roots_jacobi(num_interior_roots, 0, 1, mu=True)

        # Adjust Jacobi weights for standard Legendre measure
        interior_weights = jacobi_weights / (1.0 + interior_roots)

        # Weight for the -1 point
        left_endpoint_weight = 2.0 / (num_collocation_nodes**2)

        # Combine nodes and weights
        collocation_nodes = np.concatenate([collocation_nodes, interior_roots.astype(np.float64)])
        quadrature_weights = np.concatenate(
            [
                np.array([left_endpoint_weight], dtype=np.float64),
                interior_weights.astype(np.float64),
            ]
        )

    # Create state approximation nodes (collocation nodes + right endpoint)
    state_approximation_nodes = np.concatenate(
        [collocation_nodes, np.array([1.0], dtype=np.float64)]
    )

    return RadauNodesAndWeights(
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
    )


def compute_barycentric_weights(nodes: _ArrayLike) -> _FloatArray:
    num_nodes = len(nodes)
    barycentric_weights = np.ones(num_nodes, dtype=np.float64)
    nodes_array = np.asarray(nodes, dtype=np.float64)

    for j in range(num_nodes):
        node_differences = nodes_array[j] - np.delete(nodes_array, j)
        # Ensure no zero node_differences if nodes were extremely close
        mask = np.abs(node_differences) < ZERO_TOLERANCE * 1e-1
        if np.any(mask):
            node_differences[mask] = (
                np.sign(node_differences[mask]) * ZERO_TOLERANCE * 1e-1
                if np.any(node_differences[mask] != 0)
                else ZERO_TOLERANCE * 1e-1
            )
        barycentric_weights[j] = 1.0 / np.prod(node_differences)

    return barycentric_weights


def evaluate_lagrange_polynomial_at_point(
    polynomial_definition_nodes: _ArrayLike,
    barycentric_weights: _ArrayLike,
    evaluation_point_tau: float,
) -> _FloatArray:
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_polynomial_values_at_evaluation_point = np.zeros(
        num_polynomial_definition_nodes, dtype=np.float64
    )

    # Check if evaluation_point_tau is one of the nodes (within tolerance)
    for j in range(num_polynomial_definition_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_polynomial_values_at_evaluation_point[j] = 1.0
            return lagrange_polynomial_values_at_evaluation_point

    barycentric_sum_denominator = 0.0
    weighted_inverse_evaluation_point_differences = np.zeros(
        num_polynomial_definition_nodes, dtype=np.float64
    )

    for j in range(num_polynomial_definition_nodes):
        evaluation_point_difference_from_node = (
            evaluation_point_tau - polynomial_definition_nodes[j]
        )
        if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE:
            evaluation_point_difference_from_node = (
                np.sign(evaluation_point_difference_from_node) * ZERO_TOLERANCE
                if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE
                else evaluation_point_difference_from_node
            )
            if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE:
                weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / (
                    np.sign(polynomial_definition_nodes[j]) * 1e-100
                    if polynomial_definition_nodes[j] != 0
                    else 1e-100
                )
            else:
                weighted_inverse_evaluation_point_differences[j] = (
                    barycentric_weights[j] / evaluation_point_difference_from_node
                )
        else:
            weighted_inverse_evaluation_point_differences[j] = (
                barycentric_weights[j] / evaluation_point_difference_from_node
            )
        barycentric_sum_denominator += weighted_inverse_evaluation_point_differences[j]

    if abs(barycentric_sum_denominator) < ZERO_TOLERANCE:
        return lagrange_polynomial_values_at_evaluation_point

    # Return a new array here instead of modifying in-place
    result = weighted_inverse_evaluation_point_differences / barycentric_sum_denominator
    return result


def compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: _ArrayLike,
    barycentric_weights: _ArrayLike,
    evaluation_point_tau: float,
) -> _FloatArray:
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_derivative_coefficients = np.zeros(num_polynomial_definition_nodes, dtype=np.float64)
    matched_node_index = -1

    for current_node_index_for_match_check in range(num_polynomial_definition_nodes):
        if (
            abs(
                evaluation_point_tau
                - polynomial_definition_nodes[current_node_index_for_match_check]
            )
            < ZERO_TOLERANCE
        ):
            matched_node_index = current_node_index_for_match_check
            break

    if matched_node_index != -1:
        sum_for_diagonal_derivative_coefficient = 0.0
        for polynomial_index in range(num_polynomial_definition_nodes):
            if polynomial_index == matched_node_index:
                continue

            node_difference_denominator = (
                polynomial_definition_nodes[matched_node_index]
                - polynomial_definition_nodes[polynomial_index]
            )
            if (
                abs(node_difference_denominator) < ZERO_TOLERANCE
                or abs(barycentric_weights[matched_node_index]) < ZERO_TOLERANCE
            ):
                lagrange_derivative_coefficients[polynomial_index] = 0.0
            else:
                lagrange_derivative_coefficients[polynomial_index] = (
                    barycentric_weights[polynomial_index] / barycentric_weights[matched_node_index]
                ) / node_difference_denominator
            sum_for_diagonal_derivative_coefficient += lagrange_derivative_coefficients[
                polynomial_index
            ]

        lagrange_derivative_coefficients[matched_node_index] = (
            -sum_for_diagonal_derivative_coefficient
        )
    return lagrange_derivative_coefficients


def compute_radau_collocation_components(num_collocation_nodes: int) -> RadauBasisComponents:
    lgr_components = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)
    state_nodes = lgr_components.state_approximation_nodes
    collocation_nodes = lgr_components.collocation_nodes
    quadrature_weights = lgr_components.quadrature_weights

    if state_nodes is None or collocation_nodes is None or quadrature_weights is None:
        raise ValueError("Unable to compute Radau nodes and weights")

    num_state_nodes = len(state_nodes)
    num_actual_collocation_nodes = len(collocation_nodes)

    # Validate dimensions
    if num_state_nodes != num_collocation_nodes + 1:
        raise ValueError(
            f"Mismatch in expected number of basis points. Expected {num_collocation_nodes + 1}, Got {num_state_nodes}"
        )
    if num_actual_collocation_nodes != num_collocation_nodes:
        raise ValueError(
            f"Mismatch in expected number of collocation points. Expected {num_collocation_nodes}, Got {num_actual_collocation_nodes}"
        )

    # Calculate barycentric weights for the basis points
    bary_weights = compute_barycentric_weights(state_nodes)

    # Calculate Differentiation Matrix D
    diff_matrix = np.zeros((num_actual_collocation_nodes, num_state_nodes), dtype=np.float64)

    for collocation_node_index in range(num_actual_collocation_nodes):
        evaluation_point_at_collocation_node_tau = collocation_nodes[collocation_node_index]
        diff_matrix[collocation_node_index, :] = compute_lagrange_derivative_coefficients_at_point(
            state_nodes, bary_weights, evaluation_point_at_collocation_node_tau
        )

    # Calculate Lagrange Polynomial values at tau = +1
    lagrange_at_tau_plus_one: _FloatArray | None = None

    # Find index of +1.0 in state_nodes using np.isclose for robust floating point comparison
    tau_plus_one_indices = np.where(np.isclose(state_nodes, 1.0, atol=ZERO_TOLERANCE))[0]

    if len(tau_plus_one_indices) == 1:
        # Create a new array filled with zeros
        values = np.zeros(num_state_nodes, dtype=np.float64)
        # Set the appropriate index to 1.0
        values[tau_plus_one_indices[0]] = 1.0
        lagrange_at_tau_plus_one = values
    elif len(tau_plus_one_indices) > 1:
        # Should not happen if state_nodes are distinct
        print("Warning: Multiple basis points found close to +1.0. Using the last one.")
        # Create a new array filled with zeros
        values = np.zeros(num_state_nodes, dtype=np.float64)
        # Set the last matching index to 1.0
        values[tau_plus_one_indices[-1]] = 1.0
        lagrange_at_tau_plus_one = values
    else:
        # Fallback
        print("Warning: +1.0 not found precisely in state_nodes. Interpolating as fallback.")
        # Use the evaluate function directly
        lagrange_at_tau_plus_one = evaluate_lagrange_polynomial_at_point(
            state_nodes, bary_weights, 1.0
        )

    return RadauBasisComponents(
        differentiation_matrix=diff_matrix,
        state_approximation_nodes=state_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
        barycentric_weights_for_state_nodes=bary_weights,
        lagrange_at_tau_plus_one=lagrange_at_tau_plus_one,
    )
