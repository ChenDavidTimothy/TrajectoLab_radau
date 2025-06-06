import numpy as np

from maptor.radau import (
    _compute_barycentric_weights,
    _evaluate_lagrange_polynomial_at_point,
)
from maptor.tl_types import FloatArray


__all__ = [
    "PolynomialInterpolant",
    "_map_global_normalized_tau_to_local_interval_tau",
    "_map_local_interval_tau_to_global_normalized_tau",
    "_map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k",
    "_map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1",
]


def _validate_interpolant_dimensions(values: FloatArray, nodes: FloatArray) -> tuple[int, int, int]:
    values_2d = np.atleast_2d(values)
    nodes_array = np.asarray(nodes, dtype=np.float64)

    num_vars, num_nodes_val = values_2d.shape
    num_nodes_pts = len(nodes_array)

    if num_nodes_pts != num_nodes_val:
        raise ValueError(
            f"Mismatch in number of nodes ({num_nodes_pts}) and values columns ({num_nodes_val})"
        )

    return num_vars, num_nodes_val, num_nodes_pts


def _prepare_barycentric_weights(
    nodes_array: FloatArray, barycentric_weights: FloatArray | None, num_nodes_pts: int
) -> FloatArray:
    if barycentric_weights is None:
        return _compute_barycentric_weights(nodes_array)

    weights = np.asarray(barycentric_weights, dtype=np.float64)
    if len(weights) != num_nodes_pts:
        raise ValueError("Barycentric weights length does not match nodes length")

    return weights


def _evaluate_single_point(
    zeta: float, nodes_array: FloatArray, bary_weights: FloatArray, values_at_nodes: FloatArray
) -> FloatArray:
    L_j = _evaluate_lagrange_polynomial_at_point(nodes_array, bary_weights, zeta)
    return np.asarray(np.dot(values_at_nodes, L_j), dtype=np.float64)


def _compute_interval_parameters(global_start: float, global_end: float) -> tuple[float, float]:
    beta = (global_end - global_start) / 2.0
    beta0 = (global_end + global_start) / 2.0
    return beta, beta0


class PolynomialInterpolant:
    """Callable class that implements Lagrange polynomial interpolation using the barycentric formula"""

    def __init__(
        self,
        nodes: FloatArray,
        values: FloatArray,
        barycentric_weights: FloatArray | None = None,
    ) -> None:
        self.num_vars, self.num_nodes_val, self.num_nodes_pts = _validate_interpolant_dimensions(
            values, nodes
        )

        self.values_at_nodes = np.atleast_2d(values)
        self.nodes_array = np.asarray(nodes, dtype=np.float64)
        self.bary_weights = _prepare_barycentric_weights(
            self.nodes_array, barycentric_weights, self.num_nodes_pts
        )

    def __call__(self, points: float | FloatArray) -> FloatArray:
        is_scalar = np.isscalar(points)
        zeta_arr = np.atleast_1d(points)
        result = np.zeros((self.num_vars, len(zeta_arr)), dtype=np.float64)

        for i, zeta in enumerate(zeta_arr):
            result[:, i] = _evaluate_single_point(
                zeta, self.nodes_array, self.bary_weights, self.values_at_nodes
            )

        return result[:, 0] if is_scalar else result


def _map_global_normalized_tau_to_local_interval_tau(
    global_tau: float, global_start: float, global_end: float
) -> float:
    # Maps global tau to local zeta in [-1, 1]
    beta, beta0 = _compute_interval_parameters(global_start, global_end)

    if abs(beta) < 1e-12:
        return 0.0

    return (global_tau - beta0) / beta


def _map_local_interval_tau_to_global_normalized_tau(
    local_tau: float, global_start: float, global_end: float
) -> float:
    # Maps local zeta in [-1, 1] to global tau.
    beta, beta0 = _compute_interval_parameters(global_start, global_end)
    return beta * local_tau + beta0


def _map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
    local_tau_k: float, global_start_k: float, global_shared: float, global_end_kp1: float
) -> float:
    # Transforms zeta in interval k to zeta in interval k+1
    global_tau = _map_local_interval_tau_to_global_normalized_tau(
        local_tau_k, global_start_k, global_shared
    )
    return _map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_shared, global_end_kp1
    )


def _map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
    local_tau_kp1: float, global_start_k: float, global_shared: float, global_end_kp1: float
) -> float:
    # Transforms zeta in interval k+1 to zeta in interval k.
    global_tau = _map_local_interval_tau_to_global_normalized_tau(
        local_tau_kp1, global_shared, global_end_kp1
    )
    return _map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_start_k, global_shared
    )
