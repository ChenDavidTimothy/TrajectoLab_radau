# trajectolab/problem/mesh.py
"""
Mesh configuration for multiphase pseudospectral discretization - PURGED.
All redundancy eliminated, using centralized validation.
"""

import logging

import numpy as np

from ..input_validation import validate_mesh_configuration
from ..tl_types import NumericArrayLike
from .state import PhaseDefinition


logger = logging.getLogger(__name__)


def configure_phase_mesh(
    phase_def: PhaseDefinition, polynomial_degrees: list[int], mesh_points: NumericArrayLike
) -> None:
    """Configure the mesh structure for a specific phase."""
    # Convert to numpy array
    mesh_array = np.asarray(mesh_points, dtype=np.float64)

    # Log mesh configuration
    logger.debug(
        "Configuring mesh for phase %d: %d degrees, %d points",
        phase_def.phase_id,
        len(polynomial_degrees),
        len(mesh_array),
    )

    # SINGLE comprehensive validation call
    validate_mesh_configuration(polynomial_degrees, mesh_array, len(polynomial_degrees))

    # Set mesh configuration
    phase_def.collocation_points_per_interval = polynomial_degrees
    phase_def.global_normalized_mesh_nodes = mesh_array
    phase_def.mesh_configured = True

    logger.debug(
        "Mesh configuration complete for phase %d: intervals=%d, total_nodes=%d",
        phase_def.phase_id,
        len(polynomial_degrees),
        len(mesh_array),
    )


def clear_phase_mesh(phase_def: PhaseDefinition) -> None:
    """Clear mesh configuration for a specific phase."""
    logger.debug("Clearing mesh configuration for phase %d", phase_def.phase_id)

    phase_def.collocation_points_per_interval = []
    phase_def.global_normalized_mesh_nodes = None
    phase_def.mesh_configured = False
