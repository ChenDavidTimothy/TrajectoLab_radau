"""Coordinate transformation utilities for adaptive mesh refinement."""

from trajectolab.trajectolab_types import _NormalizedTimePoint


def map_global_normalized_tau_to_local_interval_tau(
    global_tau: _NormalizedTimePoint,
    global_interval_start_tau: _NormalizedTimePoint,
    global_interval_end_tau: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Map global normalized tau to local interval tau.

    Args:
        global_tau: Global normalized time point
        global_interval_start_tau: Global tau at interval start
        global_interval_end_tau: Global tau at interval end

    Returns:
        Corresponding local tau value in [-1, 1]
    """
    beta = (global_interval_end_tau - global_interval_start_tau) / 2.0
    beta0 = (global_interval_end_tau + global_interval_start_tau) / 2.0
    if abs(beta) < 1e-12:
        return 0.0  # Midpoint for zero-length interval
    return (global_tau - beta0) / beta


def map_local_interval_tau_to_global_normalized_tau(
    local_tau: _NormalizedTimePoint,
    global_interval_start_tau: _NormalizedTimePoint,
    global_interval_end_tau: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Map local interval tau to global normalized tau.

    Args:
        local_tau: Local normalized time in [-1, 1]
        global_interval_start_tau: Global tau at interval start
        global_interval_end_tau: Global tau at interval end

    Returns:
        Corresponding global tau value
    """
    beta = (global_interval_end_tau - global_interval_start_tau) / 2.0
    beta0 = (global_interval_end_tau + global_interval_start_tau) / 2.0
    return beta * local_tau + beta0


def map_local_tau_from_interval_k_to_equivalent_in_interval_k_plus_1(
    local_tau_k: _NormalizedTimePoint,
    global_start_tau_k: _NormalizedTimePoint,
    global_shared_tau: _NormalizedTimePoint,
    global_end_tau_kp1: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Map local tau from interval k to equivalent in interval k+1.

    Args:
        local_tau_k: Local normalized time in interval k
        global_start_tau_k: Global tau at start of interval k
        global_shared_tau: Global tau at boundary between intervals k and k+1
        global_end_tau_kp1: Global tau at end of interval k+1

    Returns:
        Equivalent local tau in interval k+1
    """
    global_tau = map_local_interval_tau_to_global_normalized_tau(
        local_tau_k, global_start_tau_k, global_shared_tau
    )
    return map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_shared_tau, global_end_tau_kp1
    )


def map_local_tau_from_interval_k_plus_1_to_equivalent_in_interval_k(
    local_tau_kp1: _NormalizedTimePoint,
    global_start_tau_k: _NormalizedTimePoint,
    global_shared_tau: _NormalizedTimePoint,
    global_end_tau_kp1: _NormalizedTimePoint,
) -> _NormalizedTimePoint:
    """Map local tau from interval k+1 to equivalent in interval k.

    Args:
        local_tau_kp1: Local normalized time in interval k+1
        global_start_tau_k: Global tau at start of interval k
        global_shared_tau: Global tau at boundary between intervals k and k+1
        global_end_tau_kp1: Global tau at end of interval k+1

    Returns:
        Equivalent local tau in interval k
    """
    global_tau = map_local_interval_tau_to_global_normalized_tau(
        local_tau_kp1, global_shared_tau, global_end_tau_kp1
    )
    return map_global_normalized_tau_to_local_interval_tau(
        global_tau, global_start_tau_k, global_shared_tau
    )
