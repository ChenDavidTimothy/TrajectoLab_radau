import functools
from dataclasses import dataclass, field
from typing import Literal, cast, overload

import numpy as np
from scipy.special import roots_jacobi as _scipy_roots_jacobi

from .exceptions import DataIntegrityError
from .input_validation import _validate_positive_integer
from .tl_types import FloatArray
from .utils.constants import (
    DEFAULT_LRU_CACHE_SIZE,
    MACHINE_EPS,
)
from .utils.precision import _is_mathematically_zero, _validate_mathematical_condition


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
    barycentric_weights_for_collocation_nodes: FloatArray = field(
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


@overload
def _roots_jacobi(
    n: int, alpha: float, beta: float, mu: Literal[False]
) -> tuple[FloatArray, FloatArray]: ...


@overload
def _roots_jacobi(
    n: int, alpha: float, beta: float, mu: Literal[True]
) -> tuple[FloatArray, FloatArray, float]: ...


def _roots_jacobi(
    n: int, alpha: float, beta: float, mu: bool = False
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, float]:
    if mu:
        result = _scipy_roots_jacobi(n, alpha, beta, mu=True)
        x_val = result[0]
        w_val = result[1]
        mu_val: float = result[2]
        return (
            x_val.astype(np.float64),
            w_val.astype(np.float64),
            float(mu_val),
        )
    else:
        result = _scipy_roots_jacobi(n, alpha, beta, mu=False)
        x_val = result[0]
        w_val = result[1]
        return (
            x_val.astype(np.float64),
            w_val.astype(np.float64),
        )


def _compute_legendre_gauss_radau_nodes_and_weights(
    num_collocation_nodes: int,
) -> RadauNodesAndWeights:
    collocation_nodes_list: list[float] = [-1.0]

    if num_collocation_nodes == 1:
        quadrature_weights_list: list[float] = [2.0]
    else:
        num_interior_roots = num_collocation_nodes - 1
        interior_roots, jacobi_weights, _ = _roots_jacobi(num_interior_roots, 0.0, 1.0, mu=True)
        interior_weights = jacobi_weights / (np.add(1.0, interior_roots))
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


def _compute_barycentric_weights(nodes: FloatArray) -> FloatArray:
    num_nodes = len(nodes)
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    nodes_col = nodes[:, np.newaxis]
    nodes_row = nodes[np.newaxis, :]
    differences_matrix = nodes_col - nodes_row

    # Scale-relative condition number validation
    node_scale = np.max(np.abs(nodes))
    min_spacing = np.min(np.diff(np.sort(nodes)))
    condition_estimate = node_scale / (min_spacing + MACHINE_EPS)

    _validate_mathematical_condition(condition_estimate, "barycentric weights computation")

    # Check for mathematically zero differences
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and _is_mathematically_zero(differences_matrix[i, j], node_scale):
                raise DataIntegrityError(
                    f"Nodes {i} and {j} are mathematically identical relative to scale {node_scale}",
                    "Node spacing too small for reliable computation",
                )

    differences_matrix[np.eye(num_nodes, dtype=bool)] = 1.0
    products = np.prod(differences_matrix, axis=1, dtype=np.float64)

    return (1.0 / products).astype(np.float64)


def _evaluate_lagrange_polynomial_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    num_nodes = len(polynomial_definition_nodes)

    node_scale = max(np.max(np.abs(polynomial_definition_nodes)), 1.0)
    differences = evaluation_point_tau - polynomial_definition_nodes
    coincident_mask = np.array(
        [_is_mathematically_zero(diff, node_scale) for diff in differences], dtype=bool
    )

    if np.any(coincident_mask):
        lagrange_values = np.zeros(num_nodes, dtype=np.float64)
        lagrange_values[coincident_mask] = 1.0
        return lagrange_values

    diffs = evaluation_point_tau - polynomial_definition_nodes

    # Use scale-relative safe division
    from .utils.precision import _safe_division

    try:
        terms = np.array(
            [_safe_division(barycentric_weights[i], diffs[i], node_scale) for i in range(num_nodes)]
        )
    except ZeroDivisionError:
        return np.zeros(num_nodes, dtype=np.float64)

    sum_terms = np.sum(terms)
    if _is_mathematically_zero(sum_terms, np.max(np.abs(terms))):
        return np.zeros(num_nodes, dtype=np.float64)

    return cast(FloatArray, terms / sum_terms)


def _compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    num_nodes = len(polynomial_definition_nodes)

    node_scale = max(np.max(np.abs(polynomial_definition_nodes)), 1.0)
    differences = evaluation_point_tau - polynomial_definition_nodes

    # Find nodes that are mathematically coincident with evaluation point
    matched_indices = [
        i for i, diff in enumerate(differences) if _is_mathematically_zero(diff, node_scale)
    ]

    if len(matched_indices) == 0:
        return np.zeros(num_nodes, dtype=np.float64)

    matched_node_idx_k = matched_indices[0]
    derivatives = np.zeros(num_nodes, dtype=np.float64)

    node_diffs = polynomial_definition_nodes[matched_node_idx_k] - polynomial_definition_nodes
    non_diagonal_mask = np.arange(num_nodes) != matched_node_idx_k

    if _is_mathematically_zero(
        barycentric_weights[matched_node_idx_k], np.max(np.abs(barycentric_weights))
    ):
        derivatives[non_diagonal_mask] = 0.0
    else:
        from .utils.precision import _safe_division

        weight_ratios = barycentric_weights / barycentric_weights[matched_node_idx_k]

        for i in range(num_nodes):
            if i != matched_node_idx_k:
                try:
                    derivatives[i] = _safe_division(weight_ratios[i], node_diffs[i], node_scale)
                except ZeroDivisionError:
                    derivatives[i] = 0.0

    # Diagonal element
    try:
        from .utils.precision import _safe_division

        diagonal_sum = sum(
            _safe_division(1.0, node_diffs[i], node_scale)
            for i in range(num_nodes)
            if i != matched_node_idx_k and not _is_mathematically_zero(node_diffs[i], node_scale)
        )
        derivatives[matched_node_idx_k] = diagonal_sum
    except ZeroDivisionError:
        derivatives[matched_node_idx_k] = 0.0

    return derivatives


@functools.lru_cache(maxsize=DEFAULT_LRU_CACHE_SIZE)
def _compute_radau_collocation_components(
    num_collocation_nodes: int,
) -> RadauBasisComponents:
    _validate_positive_integer(num_collocation_nodes, "collocation nodes")

    lgr_data = _compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)

    state_nodes = lgr_data.state_approximation_nodes
    collocation_nodes = lgr_data.collocation_nodes
    quadrature_weights = lgr_data.quadrature_weights

    num_state_nodes = len(state_nodes)
    num_actual_collocation_nodes = len(collocation_nodes)

    bary_weights_state_nodes = _compute_barycentric_weights(state_nodes)
    bary_weights_collocation_nodes = _compute_barycentric_weights(collocation_nodes)

    diff_matrix = np.zeros((num_actual_collocation_nodes, num_state_nodes), dtype=np.float64)
    for i in range(num_actual_collocation_nodes):
        tau_c_i = collocation_nodes[i]
        diff_matrix[i, :] = _compute_lagrange_derivative_coefficients_at_point(
            state_nodes, bary_weights_state_nodes, tau_c_i
        )

    lagrange_at_tau_plus_one = _evaluate_lagrange_polynomial_at_point(
        state_nodes, bary_weights_state_nodes, 1.0
    )

    return RadauBasisComponents(
        state_approximation_nodes=state_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
        differentiation_matrix=diff_matrix,
        barycentric_weights_for_state_nodes=bary_weights_state_nodes,
        barycentric_weights_for_collocation_nodes=bary_weights_collocation_nodes,
        lagrange_at_tau_plus_one=lagrange_at_tau_plus_one,
    )


def _evaluate_lagrange_interpolation_at_points(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    values: FloatArray,
    eval_points: float | FloatArray,
) -> FloatArray:
    is_scalar = np.isscalar(eval_points)
    eval_array = np.atleast_1d(eval_points)
    values_2d = np.atleast_2d(values)
    num_vars = values_2d.shape[0]

    result = np.zeros((num_vars, len(eval_array)), dtype=np.float64)

    for i, zeta in enumerate(eval_array):
        L_j = _evaluate_lagrange_polynomial_at_point(nodes, barycentric_weights, zeta)
        result[:, i] = np.dot(values_2d, L_j)

    return result[:, 0] if is_scalar else result
