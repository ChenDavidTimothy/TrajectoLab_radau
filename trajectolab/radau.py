"""
Radau pseudospectral method implementation - SIMPLIFIED with ENHANCED ERROR HANDLING.
Consolidated cache system, removed redundant code patterns.
Added targeted input validation for computational functions.
"""

import threading
from dataclasses import dataclass, field
from typing import ClassVar, Literal, cast, overload

import numpy as np
from scipy.special import roots_jacobi as _scipy_roots_jacobi

from .exceptions import ConfigurationError, DataIntegrityError
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
    """SIMPLIFIED thread-safe global cache for Radau basis components."""

    _instance: ClassVar["RadauBasisCache | None"] = None
    _cache: ClassVar[dict[int, RadauBasisComponents]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "RadauBasisCache":
        """Singleton pattern for global cache."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_components(self, num_collocation_nodes: int) -> RadauBasisComponents:
        """Get cached Radau components or compute if not cached."""
        with self._lock:
            if num_collocation_nodes not in self._cache:
                self._cache[num_collocation_nodes] = self._compute_components(num_collocation_nodes)
            return self._cache[num_collocation_nodes]

    def _compute_components(self, num_collocation_nodes: int) -> RadauBasisComponents:
        """Compute Radau components - expensive operation."""
        # Guard clause: Validate input
        if not isinstance(num_collocation_nodes, int):
            raise ConfigurationError(
                f"Number of collocation nodes must be integer, got {type(num_collocation_nodes)}",
                "TrajectoLab polynomial degree specification error",
            )

        if num_collocation_nodes < 1:
            raise ConfigurationError(
                f"Number of collocation points must be >= 1, got {num_collocation_nodes}",
                "Invalid polynomial degree for Radau collocation",
            )

        lgr_data = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)

        state_nodes = lgr_data.state_approximation_nodes
        collocation_nodes = lgr_data.collocation_nodes
        quadrature_weights = lgr_data.quadrature_weights

        num_state_nodes = len(state_nodes)
        num_actual_collocation_nodes = len(collocation_nodes)

        # Guard clause: Validate computed dimensions
        if num_state_nodes != num_collocation_nodes + 1:
            raise DataIntegrityError(
                f"State approximation nodes dimension mismatch: expected {num_collocation_nodes + 1}, got {num_state_nodes}",
                "TrajectoLab Radau basis computation error",
            )
        if num_actual_collocation_nodes != num_collocation_nodes:
            raise DataIntegrityError(
                f"Collocation nodes dimension mismatch: expected {num_collocation_nodes}, got {num_actual_collocation_nodes}",
                "TrajectoLab Radau basis computation error",
            )

        bary_weights_state_nodes = compute_barycentric_weights(state_nodes)

        diff_matrix = np.zeros((num_actual_collocation_nodes, num_state_nodes), dtype=np.float64)
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
    """Wrapper for scipy roots_jacobi with proper typing and validation."""
    # Guard clause: Validate inputs
    if not isinstance(n, int) or n < 0:
        raise ConfigurationError(
            f"n must be non-negative integer, got {n}", "Invalid Jacobi polynomial degree"
        )

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
    """Compute Legendre-Gauss-Radau nodes and weights with enhanced validation."""
    # Guard clause: Validate input
    if not isinstance(num_collocation_nodes, int):
        raise ConfigurationError(
            f"Number of collocation nodes must be integer, got {type(num_collocation_nodes)}",
            "TrajectoLab polynomial degree specification error",
        )

    if num_collocation_nodes < 1:
        raise ConfigurationError(
            f"Number of collocation points must be >= 1, got {num_collocation_nodes}",
            "Invalid polynomial degree for Radau collocation",
        )

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
    """Compute barycentric weights for Lagrange interpolation with enhanced validation."""
    # Guard clause: Validate input
    if not isinstance(nodes, np.ndarray):
        raise ConfigurationError(
            f"Nodes must be numpy array, got {type(nodes)}", "TrajectoLab node specification error"
        )

    num_nodes = len(nodes)
    if num_nodes < 1:
        raise ConfigurationError(
            f"Barycentric weights require at least 1 node, got {num_nodes}",
            "Invalid node count for interpolation",
        )
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    # Fix: Use cast to ensure correct type annotation
    barycentric_weights = cast(FloatArray, np.ones(num_nodes, dtype=np.float64))

    for j in range(num_nodes):
        other_nodes = np.delete(nodes, j)
        node_differences = nodes[j] - other_nodes

        # Handle near-zero differences
        mask_near_zero = np.abs(node_differences) < ZERO_TOLERANCE
        if np.any(mask_near_zero):
            perturbation = np.sign(node_differences[mask_near_zero]) * ZERO_TOLERANCE
            perturbation[perturbation == 0] = ZERO_TOLERANCE
            node_differences[mask_near_zero] = perturbation

        product_val = float(np.prod(node_differences, dtype=np.float64))

        if abs(product_val) < ZERO_TOLERANCE**2:
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
    """Evaluate Lagrange polynomial at a specific point using barycentric formula with enhanced validation."""
    # Guard clause: Validate inputs
    if not isinstance(polynomial_definition_nodes, np.ndarray):
        raise ConfigurationError(
            f"Polynomial definition nodes must be numpy array, got {type(polynomial_definition_nodes)}",
            "TrajectoLab node specification error",
        )

    if not isinstance(barycentric_weights, np.ndarray):
        raise ConfigurationError(
            f"Barycentric weights must be numpy array, got {type(barycentric_weights)}",
            "TrajectoLab weight specification error",
        )

    if len(polynomial_definition_nodes) != len(barycentric_weights):
        raise ConfigurationError(
            f"Node count ({len(polynomial_definition_nodes)}) must match weight count ({len(barycentric_weights)})",
            "TrajectoLab interpolation setup error",
        )

    if not isinstance(evaluation_point_tau, int | float):
        raise ConfigurationError(
            f"Evaluation point must be numeric, got {type(evaluation_point_tau)}",
            "TrajectoLab evaluation point specification error",
        )

    num_nodes = len(polynomial_definition_nodes)
    lagrange_values = np.zeros(num_nodes, dtype=np.float64)

    # Check if evaluation point coincides with a node
    for j in range(num_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_values[j] = 1.0
            return lagrange_values

    # Standard barycentric interpolation
    terms = np.zeros(num_nodes, dtype=np.float64)
    for j in range(num_nodes):
        diff = evaluation_point_tau - polynomial_definition_nodes[j]
        if abs(diff) < ZERO_TOLERANCE:
            diff = np.sign(diff) * ZERO_TOLERANCE if diff != 0 else ZERO_TOLERANCE
        terms[j] = barycentric_weights[j] / diff

    sum_of_terms = np.sum(terms)
    if abs(sum_of_terms) < ZERO_TOLERANCE:
        return lagrange_values

    # FIX: Avoid problematic reassignment by using new variable and returning directly
    normalized_terms = terms / sum_of_terms
    return np.asarray(normalized_terms, dtype=np.float64)


def compute_lagrange_derivative_coefficients_at_point(
    polynomial_definition_nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point_tau: float,
) -> FloatArray:
    """Compute Lagrange polynomial derivative coefficients at a specific point with enhanced validation."""
    # Guard clause: Validate inputs
    if not isinstance(polynomial_definition_nodes, np.ndarray):
        raise ConfigurationError(
            f"Polynomial definition nodes must be numpy array, got {type(polynomial_definition_nodes)}",
            "TrajectoLab node specification error",
        )

    if not isinstance(barycentric_weights, np.ndarray):
        raise ConfigurationError(
            f"Barycentric weights must be numpy array, got {type(barycentric_weights)}",
            "TrajectoLab weight specification error",
        )

    if len(polynomial_definition_nodes) != len(barycentric_weights):
        raise ConfigurationError(
            f"Node count ({len(polynomial_definition_nodes)}) must match weight count ({len(barycentric_weights)})",
            "TrajectoLab interpolation setup error",
        )

    num_nodes = len(polynomial_definition_nodes)
    derivatives = np.zeros(num_nodes, dtype=np.float64)

    # Find if evaluation point matches a node
    matched_node_idx_k = -1
    for i in range(num_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[i]) < ZERO_TOLERANCE:
            matched_node_idx_k = i
            break

    if matched_node_idx_k == -1:
        return derivatives

    # Compute derivative coefficients using standard formulation
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
                    sum_val += 1.0 / (
                        np.sign(diff) * ZERO_TOLERANCE if diff != 0 else ZERO_TOLERANCE
                    )
                else:
                    sum_val += 1.0 / diff
            derivatives[j] = sum_val
        else:
            diff = polynomial_definition_nodes[matched_node_idx_k] - polynomial_definition_nodes[j]
            if abs(barycentric_weights[matched_node_idx_k]) < ZERO_TOLERANCE:
                derivatives[j] = 0.0
            elif abs(diff) < ZERO_TOLERANCE:
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
    """Get Radau components from global cache for massive speedup with enhanced validation."""
    # Guard clause: Final validation before cache access
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ConfigurationError(
            f"Number of collocation nodes must be positive integer, got {num_collocation_nodes}",
            "Invalid polynomial degree for Radau collocation",
        )

    return _radau_cache.get_components(num_collocation_nodes)
