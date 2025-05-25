"""
Numerical utilities for polynomial interpolation and coordinate transformations.
"""

import numpy as np

from trajectolab.radau import (
    compute_barycentric_weights,
    evaluate_lagrange_polynomial_at_point,
)
from trajectolab.tl_types import FloatArray


__all__ = [
    "PolynomialInterpolant",
    "map_global_normalized_tau_to_local_interval_tau",
    "map_local_interval_tau_to_global_normalized_tau",
    "map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k",
    "map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1",
]


class PolynomialInterpolant:
    """
    Callable class that implements Lagrange polynomial interpolation
    using the barycentric formula - SIMPLIFIED to use unified types.
    """

    values_at_nodes: FloatArray
    nodes_array: FloatArray
    num_vars: int
    num_nodes_val: int
    num_nodes_pts: int
    bary_weights: FloatArray

    def __init__(
        self,
        nodes: FloatArray,
        values: FloatArray,
        barycentric_weights: FloatArray | None = None,
    ) -> None:
        """Creates a Lagrange polynomial interpolant using barycentric formula."""
        # Convert to arrays if needed and ensure 2D values
        self.values_at_nodes = np.atleast_2d(values)
        self.nodes_array = np.asarray(nodes, dtype=np.float64)
        self.num_vars, self.num_nodes_val = self.values_at_nodes.shape
        self.num_nodes_pts = len(self.nodes_array)

        if self.num_nodes_pts != self.num_nodes_val:
            raise ValueError(
                f"Mismatch in number of nodes ({self.num_nodes_pts}) and values columns ({self.num_nodes_val})"
            )

        # Compute or use provided barycentric weights
        self.bary_weights = (
            compute_barycentric_weights(self.nodes_array)
            if barycentric_weights is None
            else np.asarray(barycentric_weights, dtype=np.float64)
        )

        if len(self.bary_weights) != self.num_nodes_pts:
            raise ValueError("Barycentric weights length does not match nodes length")

    def __call__(self, points: float | FloatArray) -> FloatArray:
        """Evaluates the interpolant at the given point(s)."""
        is_scalar = np.isscalar(points)
        zeta_arr = np.atleast_1d(points)
        result = np.zeros((self.num_vars, len(zeta_arr)), dtype=np.float64)

        for i, zeta in enumerate(zeta_arr):
            L_j = evaluate_lagrange_polynomial_at_point(self.nodes_array, self.bary_weights, zeta)
            result[:, i] = np.dot(self.values_at_nodes, L_j)

        # Return appropriate shape based on input
        return result[:, 0] if is_scalar else result


def map_global_normalized_tau_to_local_interval_tau(
    global_tau: float, global_start: float, global_end: float
) -> float:
    """Maps global tau to local zeta in [-1, 1]."""
    beta = (global_end - global_start) / 2.0
    beta0 = (global_end + global_start) / 2.0

    if abs(beta) < 1e-12:
        return 0.0

    return (global_tau - beta0) / beta


def map_local_interval_tau_to_global_normalized_tau(
    local_tau: float, global_start: float, global_end: float
) -> float:
    """Maps local zeta in [-1, 1] to global tau."""
    beta = (global_end - global_start) / 2.0
    beta0 = (global_end + global_start) / 2.0

    return beta * local_tau + beta0


def map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
    local_tau_k: float, global_start_k: float, global_shared: float, global_end_kp1: float
) -> float:
    """Transforms zeta in interval k to zeta in interval k+1."""
    global_tau = map_local_interval_tau_to_global_normalized_tau(
        local_tau_k, global_start_k, global_shared
    )
    return map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_shared, global_end_kp1
    )


def map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
    local_tau_kp1: float, global_start_k: float, global_shared: float, global_end_kp1: float
) -> float:
    """Transforms zeta in interval k+1 to zeta in interval k."""
    global_tau = map_local_interval_tau_to_global_normalized_tau(
        local_tau_kp1, global_shared, global_end_kp1
    )
    return map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_start_k, global_shared
    )
