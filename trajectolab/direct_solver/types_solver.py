# trajectolab/direct_solver/types_solver.py
"""
Type definitions and data structure containers for the multiphase direct solver.
OPTIMIZED: Eliminated redundant metadata storage - use problem data directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import casadi as ca

from ..tl_types import PhaseID


# Internal type aliases for multiphase solver
_PhaseVariableBundle: TypeAlias = tuple[
    ca.MX,  # initial_time
    ca.MX,  # terminal_time
    list[ca.MX],  # state_at_mesh_nodes
    list[ca.MX],  # control_variables
    ca.MX | None,  # integral_variables
]

_PhaseIntervalBundle: TypeAlias = tuple[ca.MX, ca.MX | None]  # state_matrix, interior_nodes


@dataclass
class PhaseVariableReferences:
    """Container for optimization variable references for a single phase."""

    phase_id: PhaseID
    initial_time: ca.MX
    terminal_time: ca.MX
    state_at_mesh_nodes: list[ca.MX]
    control_variables: list[ca.MX]
    integral_variables: ca.MX | None
    state_matrices: list[ca.MX] = field(default_factory=list)
    interior_variables: list[ca.MX | None] = field(default_factory=list)


@dataclass
class MultiPhaseVariableReferences:
    """Container for optimization variable references for multiphase problems."""

    phase_variables: dict[PhaseID, PhaseVariableReferences] = field(default_factory=dict)
    static_parameters: ca.MX | None = None
