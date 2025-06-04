import threading
from dataclasses import dataclass, field
from typing import ClassVar, Literal, cast, overload

import numpy as np
from scipy.special import roots_jacobi as _scipy_roots_jacobi

from .input_validation import validate_polynomial_degree
from .tl_types import FloatArray
from .utils.constants import ZERO_TOLERANCE


@dataclass
class RadauBasisComponents:
    """Components for Radau pseudospectral method basis functions."""

    state_approximation_nodes: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    collocation_nodes: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    quadrature_weights: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    differentiation_matrix: FloatArray = field(
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


class RadauBasisCache:
    """Thread-safe global cache for Radau basis components."""

    _instance: ClassVar["RadauBasisCache | None"] = None
    _cache: ClassVar[dict[int, RadauBasisComponents]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "RadauBasisCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_components(self, num_collocation_nodes: int) -> RadauBasisComponents:
        # Cache prevents expensive recomputation of basis functions and differentiation matrices
        with self._lock:
            if num_collocation_nodes not in self._cache:
                self._cache[num_collocation_nodes] = self._compute_components(num_collocation_nodes)
            return self._cache[num_collocation_nodes]

    def _compute_components(self, num_collocation_nodes: int) -> RadauBasisComponents:
        # Expensive computation - parameter validation assumed done by caller
        lgr_data = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)

        state_nodes = lgr_data.state_approximation_nodes
        collocation_nodes = lgr_data.collocation_nodes
        quadrature_weights = lgr_data.quadrature_weights

        num_state_nodes = len(state_nodes)
        num_actual_collocation_nodes = len(collocation_nodes)

        bary_weights_state_nodes = _compute_barycentric_weights(state_nodes)

        # Differentiation matrix enables pseudospectral derivative approximation
        diff_matrix = np.zeros((num_actual_collocation_nodes, num_state_nodes), dtype=np.float64)
        for i in range(num_actual_collocation_nodes):
            tau_c_i = collocation_nodes[i]
            diff_matrix[i, :] = compute_lagrange_derivative_coefficients_at_point(
                state_nodes, bary_weights_state_nodes, tau_c_i
            )

        # Lagrange evaluation at terminal boundary for boundary condition application
        lagrange_at_tau_plus_one = _evaluate_lagrange_polynomial_at_point(
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


# Global cache instance
_radau_cache = RadauBasisCache()


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
    # Wrapper for scipy roots_jacobi with proper typing - parameter validation assumed
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


def compute_legendre_gauss_radau_nodes_and_weights(
    num_collocation_nodes: int,
) -> RadauNodesAndWeights:
    # Left-sided Radau quadrature includes left endpoint for boundary condition enforcement
    collocation_nodes_list: list[float] = [-1.0]

    if num_collocation_nodes == 1:
        quadrature_weights_list: list[float] = [2.0]
    else:
        # Interior roots from Jacobi polynomial for optimal quadrature accuracy
        num_interior_roots = num_collocation_nodes - 1
        interior_roots, jacobi_weights, _ = roots_jacobi(num_interior_roots, 0.0, 1.0, mu=True)
        interior_weights = jacobi_weights / (np.add(1.0, interior_roots))
        left_endpoint_weight = 2.0 / (num_collocation_nodes**2)
        collocation_nodes_list.extend(list(interior_roots))
        quadrature_weights_list = [left_endpoint_weight, *list(interior_weights)]

    final_collocation_nodes = np.array(collocation_nodes_list, dtype=np.float64)
    final_quadrature_weights = np.array(quadrature_weights_list, dtype=np.float64)

    # State nodes include terminal boundary for complete state approximation
    state_approximation_nodes_temp = np.concatenate(
        [final_collocation_nodes, np.array([1.0], dtype=np.float64)]
    )
    state_approximation_nodes = np.unique(state_approximation_nodes_temp)

    return RadauNodesAndWeights(
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=final_collocation_nodes,
        quadrature_weights=final_quadrature_weights,
    )


def _compute_barycentric_weights(nodes: FloatArray) -> FloatArray:
    num_nodes = len(nodes)
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    # Vectorized difference matrix construction for efficiency
    nodes_col = nodes[:, np.newaxis]
    nodes_row = nodes[np.newaxis, :]
    differences_matrix = nodes_col - nodes_row

    diagonal_mask = np.eye(num_nodes, dtype=bool)

    # Perturbation prevents division by zero for near-coincident nodes
    near_zero_mask = np.abs(differences_matrix) < ZERO_TOLERANCE
    perturbation = np.sign(differences_matrix) * ZERO_TOLERANCE
    perturbation[perturbation == 0] = ZERO_TOLERANCE

    off_diagonal_near_zero = near_zero_mask & ~diagonal_mask
    differences_matrix = np.where(off_diagonal_near_zero, perturbation, differences_matrix)

    # Diagonal set to 1 for product computation
    differences_matrix[diagonal_mask] = 1.0

    products = np.prod(differences_matrix, axis=1, dtype=np.float64)

    # Regularization for small products prevents numerical instability
    small_product_mask = np.abs(products) < ZERO_TOLERANCE**2
    safe_products = np.where(
        small_product_mask,
        np.where(products == 0, 1.0 / (ZERO_TOLERANCE**2), np.sign(products) / (ZERO_TOLERANCE**2)),
        1.0 / products,
    )

    return safe_products.astype(np.float64)


def _evaluate_lagrange_polynomial_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    # Barycentric formula provides numerically stable polynomial evaluation
    num_nodes = len(polynomial_definition_nodes)

    # Direct evaluation when point coincides with node
    differences = np.abs(evaluation_point_tau - polynomial_definition_nodes)
    coincident_mask = differences < ZERO_TOLERANCE

    if np.any(coincident_mask):
        lagrange_values = np.zeros(num_nodes, dtype=np.float64)
        lagrange_values[coincident_mask] = 1.0
        return lagrange_values

    diffs = evaluation_point_tau - polynomial_definition_nodes

    # Regularization prevents division by zero for near-coincident cases
    near_zero_mask = np.abs(diffs) < ZERO_TOLERANCE
    safe_diffs = np.where(
        near_zero_mask, np.where(diffs == 0, ZERO_TOLERANCE, np.sign(diffs) * ZERO_TOLERANCE), diffs
    )

    terms = barycentric_weights / safe_diffs
    sum_terms = np.sum(terms)

    if abs(sum_terms) < ZERO_TOLERANCE:
        return np.zeros(num_nodes, dtype=np.float64)

    normalized_terms = terms / sum_terms
    return cast(FloatArray, normalized_terms)


def compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    # Derivative coefficients enable differentiation matrix construction for pseudospectral methods
    num_nodes = len(polynomial_definition_nodes)

    differences = np.abs(evaluation_point_tau - polynomial_definition_nodes)
    matched_indices = np.where(differences < ZERO_TOLERANCE)[0]

    if len(matched_indices) == 0:
        return np.zeros(num_nodes, dtype=np.float64)

    matched_node_idx_k = matched_indices[0]
    derivatives = np.zeros(num_nodes, dtype=np.float64)

    node_diffs = polynomial_definition_nodes[matched_node_idx_k] - polynomial_definition_nodes

    # Regularization for near-zero differences
    near_zero_mask = np.abs(node_diffs) < ZERO_TOLERANCE
    safe_diffs = np.where(
        near_zero_mask,
        np.where(node_diffs == 0, ZERO_TOLERANCE, np.sign(node_diffs) * ZERO_TOLERANCE),
        node_diffs,
    )

    non_diagonal_mask = np.arange(num_nodes) != matched_node_idx_k

    if abs(barycentric_weights[matched_node_idx_k]) < ZERO_TOLERANCE:
        derivatives[non_diagonal_mask] = 0.0
    else:
        weight_ratios = barycentric_weights / barycentric_weights[matched_node_idx_k]
        derivatives[non_diagonal_mask] = (
            weight_ratios[non_diagonal_mask] / safe_diffs[non_diagonal_mask]
        )

    # Diagonal derivative from sum of reciprocals (differentiation matrix property)
    derivatives[matched_node_idx_k] = np.sum(1.0 / safe_diffs[non_diagonal_mask])

    return derivatives


def compute_radau_collocation_components(
    num_collocation_nodes: int,
) -> RadauBasisComponents:
    """Get Radau components from global cache for massive speedup.

    Centralized validation and caching for Radau pseudospectral basis components.
    """
    validate_polynomial_degree(num_collocation_nodes, "collocation nodes")

    return _radau_cache.get_components(num_collocation_nodes)
