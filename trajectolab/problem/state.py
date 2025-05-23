"""
State data classes for problem definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType


@dataclass
class VariableState:
    """State for all variables and expressions."""

    # Symbolic variables
    sym_states: dict[str, SymType] = field(default_factory=dict)
    sym_controls: dict[str, SymType] = field(default_factory=dict)
    sym_parameters: dict[str, SymType] = field(default_factory=dict)
    sym_time: SymType | None = None
    sym_time_initial: SymType | None = None
    sym_time_final: SymType | None = None

    # Variable metadata
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    # Expressions
    dynamics_expressions: dict[SymType, SymExpr] = field(default_factory=dict)
    objective_expression: SymExpr | None = None

    # Integral tracking
    integral_expressions: list[SymExpr] = field(default_factory=list)
    integral_symbols: list[SymType] = field(default_factory=list)
    num_integrals: int = 0

    # Time bounds
    t0_bounds: tuple[float, float] = (0.0, 0.0)
    tf_bounds: tuple[float, float] = (1.0, 1.0)

    # ========================================================================
    # EFFICIENT ORDERING METHODS - Added to fix performance issues
    # ========================================================================

    def get_ordered_state_items(self) -> list[tuple[str, SymType]]:
        """Get (name, symbol) pairs ordered by index."""
        sorted_items = sorted(self.states.items(), key=lambda item: item[1]["index"])
        return [(name, self.sym_states[name]) for name, _ in sorted_items]

    def get_ordered_control_items(self) -> list[tuple[str, SymType]]:
        """Get (name, symbol) pairs ordered by index."""
        sorted_items = sorted(self.controls.items(), key=lambda item: item[1]["index"])
        return [(name, self.sym_controls[name]) for name, _ in sorted_items]

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols ordered by index."""
        sorted_items = sorted(self.states.items(), key=lambda item: item[1]["index"])
        return [self.sym_states[name] for name, _ in sorted_items]

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols ordered by index."""
        sorted_items = sorted(self.controls.items(), key=lambda item: item[1]["index"])
        return [self.sym_controls[name] for name, _ in sorted_items]

    def get_ordered_state_names(self) -> list[str]:
        """Get state names ordered by index."""
        sorted_items = sorted(self.states.items(), key=lambda item: item[1]["index"])
        return [name for name, _ in sorted_items]

    def get_ordered_control_names(self) -> list[str]:
        """Get control names ordered by index."""
        sorted_items = sorted(self.controls.items(), key=lambda item: item[1]["index"])
        return [name for name, _ in sorted_items]


@dataclass
class ConstraintState:
    """State for constraints."""

    constraints: list[SymExpr] = field(default_factory=list)


@dataclass
class MeshState:
    """State for mesh configuration."""

    collocation_points_per_interval: list[int] = field(default_factory=list)
    global_normalized_mesh_nodes: FloatArray | None = None
    configured: bool = False
