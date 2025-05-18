import logging
from dataclasses import dataclass, field
from typing import Literal, cast, overload

import numpy as np
from scipy.special import roots_jacobi as _scipy_roots_jacobi

# Import centralized type definitions and constants
from .tl_types import ZERO_TOLERANCE, FloatArray, FloatMatrix


# --- Dataclasses for Structured Radau Components ---


@dataclass
class RadauBasisComponents:
    state_approximation_nodes: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    collocation_nodes: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    quadrature_weights: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    differentiation_matrix: FloatMatrix = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    barycentric_weights_for_state_nodes: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    lagrange_at_tau_plus_one: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )


@dataclass
class RadauNodesAndWeights:
    state_approximation_nodes: FloatArray
    collocation_nodes: FloatArray
    quadrature_weights: FloatArray


# --- Gauss-Jacobi Quadrature Wrapper ---


@overload
def roots_jacobi(
    n: int, alpha: float, beta: float, mu: Literal[False]
) -> tuple[FloatArray, FloatArray]: ...


@overload
def roots_jacobi(
    n: int, alpha: float, beta: float, mu: Literal[True]
) -> tuple[FloatArray, FloatArray, float]: ...


def roots_jacobi(
    n: int, alpha: float, beta: float, mu: bool = False
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, float]:
    if mu:
        result = _scipy_roots_jacobi(n, alpha, beta, mu=True)
        x_val = result[0]
        w_val = result[1]
        mu_val: float = result[2]
        return (
            cast(FloatArray, x_val.astype(np.float64)),
            cast(FloatArray, w_val.astype(np.float64)),
            float(mu_val),
        )
    else:
        result = _scipy_roots_jacobi(n, alpha, beta, mu=False)
        x_val = result[0]
        w_val = result[1]
        return (
            cast(FloatArray, x_val.astype(np.float64)),
            cast(FloatArray, w_val.astype(np.float64)),
        )


# --- Core Radau Collocation Computations ---


def compute_legendre_gauss_radau_nodes_and_weights(
    num_collocation_nodes: int,
) -> RadauNodesAndWeights:
    if num_collocation_nodes < 1:
        raise ValueError("Number of collocation points must be an integer >= 1.")

    collocation_nodes_list: list[float] = [-1.0]

    if num_collocation_nodes == 1:
        quadrature_weights_list: list[float] = [2.0]
    else:
        num_interior_roots = num_collocation_nodes - 1
        interior_roots, jacobi_weights, _ = roots_jacobi(num_interior_roots, 0.0, 1.0, mu=True)
        interior_weights = jacobi_weights / (1.0 + interior_roots)
        left_endpoint_weight = 2.0 / (num_collocation_nodes**2)
        collocation_nodes_list.extend(list(interior_roots))
        quadrature_weights_list = [left_endpoint_weight, *list(interior_weights)]

    final_collocation_nodes = np.array(collocation_nodes_list, dtype=np.float64)
    final_quadrature_weights = np.array(quadrature_weights_list, dtype=np.float64)

    state_approximation_nodes_temp = np.concatenate(
        [final_collocation_nodes, np.array([1.0], dtype=np.float64)]
    )
    state_approximation_nodes = np.unique(state_approximation_nodes_temp)

    return RadauNodesAndWeights(
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=final_collocation_nodes,
        quadrature_weights=final_quadrature_weights,
    )


def compute_barycentric_weights(nodes: FloatArray) -> FloatArray:
    num_nodes = len(nodes)
    if num_nodes < 1:
        raise ValueError("Barycentric weights require at least 1 node.")
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    barycentric_weights: FloatArray = np.ones(num_nodes, dtype=np.float64)

    for j in range(num_nodes):
        other_nodes = np.delete(nodes, j)
        node_differences = nodes[j] - other_nodes

        mask_near_zero = np.abs(node_differences) < ZERO_TOLERANCE
        if np.any(mask_near_zero):
            logging.debug(
                f"Node {nodes[j]:.4f}: Perturbing {np.sum(mask_near_zero)} near-zero differences."
            )
            perturbation = np.sign(node_differences[mask_near_zero]) * ZERO_TOLERANCE
            perturbation[perturbation == 0] = ZERO_TOLERANCE
            node_differences[mask_near_zero] = perturbation

        product_val: float = float(np.prod(node_differences, dtype=np.float64))

        if abs(product_val) < ZERO_TOLERANCE**2:
            logging.warning(
                f"Product of node differences for node {nodes[j]:.4f} is extremely small "
                f"({product_val:.2e}). May indicate duplicate/close nodes. Fallback used."
            )
            barycentric_weights[j] = (
                np.sign(product_val) * (1.0 / (ZERO_TOLERANCE**2))
                if product_val != 0
                else 1.0 / (ZERO_TOLERANCE**2)
            )
        else:
            barycentric_weights[j] = 1.0 / product_val

    return barycentric_weights


def evaluate_lagrange_polynomial_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    num_nodes = len(polynomial_definition_nodes)
    lagrange_values: FloatArray = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_values[j] = 1.0
            return lagrange_values

    terms: FloatArray = np.zeros(num_nodes, dtype=np.float64)
    for j in range(num_nodes):
        diff = evaluation_point_tau - polynomial_definition_nodes[j]
        if abs(diff) < ZERO_TOLERANCE:
            logging.warning(
                f"Difference tau-x_j ({diff:.2e}) became near-zero unexpectedly in Lagrange eval. "
                f"Using perturbed value."
            )
            diff = np.sign(diff) * ZERO_TOLERANCE if diff != 0 else ZERO_TOLERANCE

        terms[j] = barycentric_weights[j] / diff

    sum_of_terms = np.sum(terms)

    if abs(sum_of_terms) < ZERO_TOLERANCE:
        logging.warning(
            f"Barycentric sum denominator is close to zero ({sum_of_terms:.2e}) "
            f"at tau={evaluation_point_tau:.4f}. Returning zero vector for Lagrange values."
        )
        return lagrange_values

    lagrange_values = terms / sum_of_terms

    return lagrange_values


def compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    num_nodes = len(polynomial_definition_nodes)
    derivatives: FloatArray = np.zeros(num_nodes, dtype=np.float64)

    matched_node_idx_k = -1
    for i in range(num_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[i]) < ZERO_TOLERANCE:
            matched_node_idx_k = i
            break

    if matched_node_idx_k == -1:
        logging.warning(
            f"Derivative requested at tau={evaluation_point_tau:.4f} which is not a node. "
            "Standard formula for D matrix elements not applicable. Returning zeros."
        )
        return derivatives

    for j in range(num_nodes):
        if j == matched_node_idx_k:
            sum_val = 0.0
            for i in range(num_nodes):
                if i == matched_node_idx_k:
                    continue
                diff = (
                    polynomial_definition_nodes[matched_node_idx_k] - polynomial_definition_nodes[i]
                )
                if abs(diff) < ZERO_TOLERANCE:
                    logging.warning(
                        f"Coincident nodes x_k=x_i for k={matched_node_idx_k}, i={i}. Derivative component will be large."
                    )
                    sum_val += 1.0 / (
                        np.sign(diff) * ZERO_TOLERANCE if diff != 0 else ZERO_TOLERANCE
                    )
                else:
                    sum_val += 1.0 / diff
            derivatives[j] = sum_val
        else:
            diff = polynomial_definition_nodes[matched_node_idx_k] - polynomial_definition_nodes[j]
            if abs(barycentric_weights[matched_node_idx_k]) < ZERO_TOLERANCE:
                logging.warning(
                    f"Barycentric weight for node x_k (k={matched_node_idx_k}) is near zero. Derivative L'_{j}(x_k) unstable, setting to 0."
                )
                derivatives[j] = 0.0
            elif abs(diff) < ZERO_TOLERANCE:
                logging.warning(
                    f"Coincident nodes x_k=x_j for k={matched_node_idx_k}, j={j} in off-diagonal. Derivative L'_{j}(x_k) will be large."
                )
                derivatives[j] = (
                    barycentric_weights[j] / barycentric_weights[matched_node_idx_k]
                ) / (np.sign(diff) * ZERO_TOLERANCE if diff != 0 else ZERO_TOLERANCE)
            else:
                derivatives[j] = (
                    barycentric_weights[j] / barycentric_weights[matched_node_idx_k]
                ) / diff
    return derivatives


def compute_radau_collocation_components(
    num_collocation_nodes: int,
) -> RadauBasisComponents:
    lgr_data = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)

    state_nodes: FloatArray = lgr_data.state_approximation_nodes
    collocation_nodes: FloatArray = lgr_data.collocation_nodes
    quadrature_weights: FloatArray = lgr_data.quadrature_weights

    num_state_nodes = len(state_nodes)
    num_actual_collocation_nodes = len(collocation_nodes)

    if num_state_nodes != num_collocation_nodes + 1:
        raise ValueError(
            f"Mismatch in expected number of state approximation nodes. "
            f"Expected {num_collocation_nodes + 1}, Got {num_state_nodes}."
        )
    if num_actual_collocation_nodes != num_collocation_nodes:
        raise ValueError(
            f"Mismatch in expected number of collocation nodes. "
            f"Expected {num_collocation_nodes}, Got {num_actual_collocation_nodes}."
        )

    bary_weights_state_nodes = compute_barycentric_weights(state_nodes)

    diff_matrix: FloatMatrix = np.zeros(
        (num_actual_collocation_nodes, num_state_nodes), dtype=np.float64
    )
    for i in range(num_actual_collocation_nodes):
        tau_c_i = collocation_nodes[i]
        diff_matrix[i, :] = compute_lagrange_derivative_coefficients_at_point(
            state_nodes, bary_weights_state_nodes, tau_c_i
        )

    lagrange_at_tau_plus_one = evaluate_lagrange_polynomial_at_point(
        state_nodes, bary_weights_state_nodes, 1.0
    )

    return RadauBasisComponents(
        state_approximation_nodes=state_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
        differentiation_matrix=diff_matrix,
        barycentric_weights_for_state_nodes=bary_weights_state_nodes,
        lagrange_at_tau_plus_one=lagrange_at_tau_plus_one,
    )
