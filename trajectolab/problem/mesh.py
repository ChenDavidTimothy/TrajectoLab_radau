"""
Mesh configuration and management functions for optimal control problems.
"""

from __future__ import annotations

import numpy as np

from ..tl_types import NumericArrayLike
from .state import MeshState


def configure_mesh(
    state: MeshState,
    polynomial_degrees: list[int],
    mesh_points: NumericArrayLike,
) -> None:
    """Configure the mesh structure."""
    # Convert to numpy array if needed
    if isinstance(mesh_points, list):
        mesh_array = np.array(mesh_points, dtype=np.float64)
    else:
        mesh_array = np.asarray(mesh_points, dtype=np.float64)

    # Validate mesh structure
    if len(polynomial_degrees) != len(mesh_array) - 1:
        raise ValueError(
            f"Number of polynomial degrees ({len(polynomial_degrees)}) must be exactly "
            f"one less than number of mesh points ({len(mesh_array)})"
        )

    # Validate polynomial degrees
    for k, degree in enumerate(polynomial_degrees):
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError(
                f"Polynomial degree for interval {k} must be positive integer, got {degree}"
            )

    # Validate mesh points
    if not np.isclose(mesh_array[0], -1.0):
        raise ValueError(f"First mesh point must be -1.0, got {mesh_array[0]}")

    if not np.isclose(mesh_array[-1], 1.0):
        raise ValueError(f"Last mesh point must be 1.0, got {mesh_array[-1]}")

    if not np.all(np.diff(mesh_array) > 1e-9):
        raise ValueError("Mesh points must be strictly increasing with minimum spacing of 1e-9")

    # Set mesh configuration
    state.collocation_points_per_interval = polynomial_degrees
    state.global_normalized_mesh_nodes = mesh_array
    state.configured = True


def clear_mesh(state: MeshState) -> None:
    """Clear mesh configuration."""
    state.collocation_points_per_interval = []
    state.global_normalized_mesh_nodes = None
    state.configured = False
