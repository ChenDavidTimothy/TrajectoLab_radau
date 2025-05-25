"""
Mesh configuration and validation for pseudospectral discretization.
"""

import logging

import numpy as np

from ..tl_types import NumericArrayLike
from .state import MeshState


# Library logger
logger = logging.getLogger(__name__)


def configure_mesh(
    state: MeshState,
    polynomial_degrees: list[int],
    mesh_points: NumericArrayLike,
) -> None:
    """
    Configure the mesh structure.
    """

    # Convert to numpy array if needed
    if isinstance(mesh_points, list):
        mesh_array = np.array(mesh_points, dtype=np.float64)
    else:
        mesh_array = np.asarray(mesh_points, dtype=np.float64)

    # Log mesh configuration (DEBUG)
    logger.debug(
        "Configuring mesh: %d degrees, %d points", len(polynomial_degrees), len(mesh_array)
    )

    # Set mesh configuration - validation done at entry point
    state.collocation_points_per_interval = polynomial_degrees
    state.global_normalized_mesh_nodes = mesh_array
    state.configured = True

    # Log successful configuration (DEBUG - details)
    logger.debug(
        "Mesh configuration complete: intervals=%d, total_nodes=%d",
        len(polynomial_degrees),
        len(mesh_array),
    )


def clear_mesh(state: MeshState) -> None:
    """Clear mesh configuration."""
    logger.debug("Clearing mesh configuration")

    state.collocation_points_per_interval = []
    state.global_normalized_mesh_nodes = None
    state.configured = False
