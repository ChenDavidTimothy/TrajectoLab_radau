"""
Type definitions and data structure containers for the direct solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import casadi as ca

from ..tl_types import (
    FloatArray,
)


# Internal type aliases
_VariableBundle: TypeAlias = tuple[
    ca.MX,  # initial_time
    ca.MX,  # terminal_time
    list[ca.MX],  # state_at_mesh_nodes
    list[ca.MX],  # control_variables
    ca.MX | None,  # integral_variables
]

_IntervalBundle: TypeAlias = tuple[ca.MX, ca.MX | None]  # state_matrix, interior_nodes


@dataclass
class VariableReferences:
    """Container for optimization variable references."""

    initial_time: ca.MX
    terminal_time: ca.MX
    state_at_mesh_nodes: list[ca.MX]
    control_variables: list[ca.MX]
    integral_variables: ca.MX | None
    state_matrices: list[ca.MX] = field(default_factory=list)
    interior_variables: list[ca.MX | None] = field(default_factory=list)


@dataclass
class MetadataBundle:
    """Container for solver metadata."""

    local_state_tau: list[FloatArray] = field(default_factory=list)
    local_control_tau: list[FloatArray] = field(default_factory=list)
    global_mesh_nodes: FloatArray = field(default_factory=lambda: FloatArray([]))
    objective_expression: ca.MX | None = None
