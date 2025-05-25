"""
Type definitions and data structure containers for the direct solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from ..tl_types import (
    CasadiMX,
    FloatArray,
    ListOfCasadiMX,
)


# Internal type aliases
_VariableBundle: TypeAlias = tuple[
    CasadiMX,  # initial_time
    CasadiMX,  # terminal_time
    ListOfCasadiMX,  # state_at_mesh_nodes
    ListOfCasadiMX,  # control_variables
    CasadiMX | None,  # integral_variables
]

_IntervalBundle: TypeAlias = tuple[CasadiMX, CasadiMX | None]  # state_matrix, interior_nodes


@dataclass
class VariableReferences:
    """Container for optimization variable references."""

    initial_time: CasadiMX
    terminal_time: CasadiMX
    state_at_mesh_nodes: ListOfCasadiMX
    control_variables: ListOfCasadiMX
    integral_variables: CasadiMX | None
    state_matrices: list[CasadiMX] = field(default_factory=list)
    interior_variables: list[CasadiMX | None] = field(default_factory=list)


@dataclass
class MetadataBundle:
    """Container for solver metadata."""

    local_state_tau: list[FloatArray] = field(default_factory=list)
    local_control_tau: list[FloatArray] = field(default_factory=list)
    global_mesh_nodes: FloatArray = field(default_factory=lambda: FloatArray([]))
    objective_expression: CasadiMX | None = None
