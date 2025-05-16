import logging
from dataclasses import dataclass, field
from typing import Literal, cast, overload

import numpy as np

# Import centralized type definitions and constants
from .tl_types import ZERO_TOLERANCE, FloatArray, FloatMatrix

# SciPy is a common dependency for such calculations, ensure it's available.


# --- Dataclasses for Structured Radau Components ---


@dataclass
class RadauBasisComponents:
    """
    Stores all components required for Radau collocation basis.

    Attributes:
        state_approximation_nodes: Nodes used for approximating the state (τ_i).
                                   Includes collocation points and the final time point (+1).
        collocation_nodes: Legendre-Gauss-Radau (LGR) nodes (τ_c).
                           Does not include the final time point.
        quadrature_weights: Quadrature weights corresponding to collocation_nodes.
        differentiation_matrix: Matrix D such that dP/dτ = D @ P_coeffs,
                                evaluated at collocation_nodes.
        barycentric_weights_for_state_nodes: Barycentric weights for interpolation
                                             using state_approximation_nodes.
        lagrange_at_tau_plus_one: Values of Lagrange basis polynomials (defined on
                                  state_approximation_nodes) evaluated at τ = +1.
    """

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
    """
    Stores essential Radau nodes and weights.
    This is an intermediate structure primarily used by
    `compute_legendre_gauss_radau_nodes_and_weights`.
    """

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
    """
    Computes Gauss-Jacobi quadrature nodes and weights.

    This is a wrapper around `scipy.special.roots_jacobi` to provide
    consistent typing with FloatArray and handle the `mu` parameter's
    return type variation.

    Args:
        n: Order of the Jacobi polynomial.
        alpha: Jacobi polynomial parameter.
        beta: Jacobi polynomial parameter.
        mu: If True, returns the third parameter `mu0` (integral of the weight function).
            Defaults to False.

    Returns:
        If mu is False: A tuple (nodes, weights).
        If mu is True: A tuple (nodes, weights, mu0).
    """
    try:
        from scipy.special import roots_jacobi as scipy_roots_jacobi_impl
    except ImportError:
        logging.error("SciPy is required for roots_jacobi. Please install it: pip install scipy")
        raise

    if mu:
        x_val, w_val, mu_val = scipy_roots_jacobi_impl(n, alpha, beta, mu=True)
        return cast(FloatArray, x_val), cast(FloatArray, w_val), float(mu_val)
    else:
        x_val, w_val = scipy_roots_jacobi_impl(n, alpha, beta, mu=False)
        return cast(FloatArray, x_val), cast(FloatArray, w_val)


# --- Core Radau Collocation Computations ---


def compute_legendre_gauss_radau_nodes_and_weights(
    num_collocation_nodes: int,
) -> RadauNodesAndWeights:
    """
    Computes Legendre-Gauss-Radau (LGR) nodes and weights.
    LGR quadrature includes the left endpoint (-1) and `num_collocation_nodes - 1`
    interior points.

    Args:
        num_collocation_nodes: The total number of collocation points (K).
                               Must be an integer >= 1.

    Returns:
        A RadauNodesAndWeights object containing:
            - state_approximation_nodes: Collocation nodes + right endpoint (+1). Size K+1.
            - collocation_nodes: LGR nodes (τ_c). Size K.
            - quadrature_weights: Quadrature weights for LGR nodes. Size K.

    Raises:
        ValueError: If num_collocation_nodes is less than 1.
    """
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ValueError("Number of collocation points must be an integer >= 1.")

    collocation_nodes_list = [-1.0]

    if num_collocation_nodes == 1:
        quadrature_weights_list = [2.0]
    else:
        num_interior_roots = num_collocation_nodes - 1
        interior_roots, jacobi_weights, _ = roots_jacobi(num_interior_roots, 0.0, 1.0, mu=True)
        interior_weights = jacobi_weights / (1.0 + interior_roots)
        left_endpoint_weight = 2.0 / (num_collocation_nodes**2)
        collocation_nodes_list.extend(list(interior_roots))
        quadrature_weights_list = [left_endpoint_weight] + list(interior_weights)

    final_collocation_nodes = np.array(collocation_nodes_list, dtype=np.float64)
    final_quadrature_weights = np.array(quadrature_weights_list, dtype=np.float64)

    state_approximation_nodes_temp = np.concatenate(
        [final_collocation_nodes, np.array([1.0], dtype=np.float64)]
    )
    state_approximation_nodes = np.unique(state_approximation_nodes_temp)

    return RadauNodesAndWeights(
        state_approximation_nodes=cast(FloatArray, state_approximation_nodes),
        collocation_nodes=cast(FloatArray, final_collocation_nodes),
        quadrature_weights=cast(FloatArray, final_quadrature_weights),
    )


def compute_barycentric_weights(nodes: FloatArray) -> FloatArray:
    """
    Computes barycentric weights for a given set of distinct nodes.
    w_j = 1 / product_{k!=j} (x_j - x_k)

    Args:
        nodes: A 1D array of distinct node locations.

    Returns:
        A 1D array of barycentric weights corresponding to the input nodes.

    Raises:
        ValueError: If fewer than 1 node is provided.
    """
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

        product_val = np.prod(node_differences)

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
    """
    Evaluates all Lagrange basis polynomials (defined by `polynomial_definition_nodes`)
    at a single `evaluation_point_tau` using the second (true) barycentric formula.
    L_j(τ) = [ w_j / (τ - x_j) ] / [ sum_{k=0}^{N} w_k / (τ - x_k) ]
    If τ is one of x_j, then L_j(τ) = 1 and L_k(τ) = 0 for k!=j.

    Args:
        polynomial_definition_nodes: Nodes x_j on which Lagrange polynomials are defined.
        barycentric_weights: Precomputed barycentric weights w_j for these nodes.
        evaluation_point_tau: The point τ at which to evaluate the polynomials.

    Returns:
        A 1D array where the j-th element is L_j(τ).
    """
    num_nodes = len(polynomial_definition_nodes)
    # Explicitly type lagrange_values
    lagrange_values: FloatArray = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_values[j] = 1.0
            return lagrange_values

    # Explicitly type terms
    terms: FloatArray = np.zeros(num_nodes, dtype=np.float64)
    for j in range(num_nodes):
        diff = evaluation_point_tau - polynomial_definition_nodes[j]
        if abs(diff) < ZERO_TOLERANCE:
            logging.warning(
                f"Difference tau-x_j ({diff:.2e}) became near-zero unexpectedly in Lagrange eval. "
                f"Using perturbed value."
            )
            diff = np.sign(diff) * ZERO_TOLERANCE if diff != 0 else ZERO_TOLERANCE

        terms[j] = barycentric_weights[j] / diff  # diff cannot be zero here due to perturbation

    sum_of_terms = np.sum(terms)

    if abs(sum_of_terms) < ZERO_TOLERANCE:
        logging.warning(
            f"Barycentric sum denominator is close to zero ({sum_of_terms:.2e}) "
            f"at tau={evaluation_point_tau:.4f}. Returning zero vector for Lagrange values."
        )
        return lagrange_values

    # Cast the RHS of the assignment to FloatArray to resolve Pylance inference issue
    lagrange_values = cast(FloatArray, terms / sum_of_terms)

    return lagrange_values  # No need for another cast here if lagrange_values is correctly typed


def compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    """
    Computes coefficients of the derivative of Lagrange polynomials at a specific point.
    This is typically used to form rows of the differentiation matrix D_kj = L'_j(x_k).

    Args:
        polynomial_definition_nodes: Nodes x_j on which Lagrange polynomials are defined.
        barycentric_weights: Precomputed barycentric weights w_j for these nodes.
        evaluation_point_tau: The point (typically one of x_j) at which to evaluate derivatives.

    Returns:
        A 1D array representing the j-th derivative L'_j(evaluation_point_tau).
    """
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
            # Ensure w_k (barycentric_weights[matched_node_idx_k]) is not zero before division
            if abs(barycentric_weights[matched_node_idx_k]) < ZERO_TOLERANCE:
                logging.warning(
                    f"Barycentric weight for node x_k (k={matched_node_idx_k}) is near zero. Derivative L'_{j}(x_k) unstable, setting to 0."
                )
                derivatives[j] = 0.0  # Avoid division by zero if w_k is zero
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
    """
    Computes all necessary components for Radau collocation.

    Args:
        num_collocation_nodes: The number of LGR collocation points (K).

    Returns:
        A RadauBasisComponents object populated with all matrices and vectors.

    Raises:
        ValueError: If inconsistencies are found in node counts.
    """
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
        differentiation_matrix=diff_matrix,  # Already FloatMatrix by initialization
        barycentric_weights_for_state_nodes=bary_weights_state_nodes,
        lagrange_at_tau_plus_one=lagrange_at_tau_plus_one,
    )
