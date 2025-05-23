"""
State data classes for problem definition.
OPTIMIZED: Pre-sorted variable access for O(1) performance.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType


@dataclass
class VariableState:
    """State for all variables and expressions with optimized ordering."""

    # Symbolic variables (maintained for compatibility)
    sym_states: dict[str, SymType] = field(default_factory=dict)
    sym_controls: dict[str, SymType] = field(default_factory=dict)
    sym_parameters: dict[str, SymType] = field(default_factory=dict)
    sym_time: SymType | None = None
    sym_time_initial: SymType | None = None
    sym_time_final: SymType | None = None

    # Variable metadata (maintained for compatibility)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    # OPTIMIZED: Pre-sorted access structures for O(1) performance
    _ordered_state_names: list[str] = field(default_factory=list)
    _ordered_control_names: list[str] = field(default_factory=list)
    _ordered_state_symbols: list[SymType] = field(default_factory=list)
    _ordered_control_symbols: list[SymType] = field(default_factory=list)
    _state_name_to_index: dict[str, int] = field(default_factory=dict)
    _control_name_to_index: dict[str, int] = field(default_factory=dict)
    _ordering_lock: threading.Lock = field(default_factory=threading.Lock)

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
    # OPTIMIZED ORDERING METHODS - O(1) performance instead of O(n log n)
    # ========================================================================

    def add_state_optimized(self, name: str, symbol: SymType, **metadata) -> None:
        """Add state while maintaining optimized ordering."""
        with self._ordering_lock:
            if name in self._state_name_to_index:
                raise ValueError(f"State {name} already exists")

            index = len(self._ordered_state_names)
            self._state_name_to_index[name] = index
            self._ordered_state_names.append(name)
            self._ordered_state_symbols.append(symbol)

            # Update legacy structures for compatibility
            self.states[name] = {"index": index, **metadata}
            self.sym_states[name] = symbol

    def add_control_optimized(self, name: str, symbol: SymType, **metadata) -> None:
        """Add control while maintaining optimized ordering."""
        with self._ordering_lock:
            if name in self._control_name_to_index:
                raise ValueError(f"Control {name} already exists")

            index = len(self._ordered_control_names)
            self._control_name_to_index[name] = index
            self._ordered_control_names.append(name)
            self._ordered_control_symbols.append(symbol)

            # Update legacy structures for compatibility
            self.controls[name] = {"index": index, **metadata}
            self.sym_controls[name] = symbol

    def get_ordered_state_items(self) -> list[tuple[str, SymType]]:
        """Get (name, symbol) pairs in O(1) time."""
        return list(zip(self._ordered_state_names, self._ordered_state_symbols, strict=False))

    def get_ordered_control_items(self) -> list[tuple[str, SymType]]:
        """Get (name, symbol) pairs in O(1) time."""
        return list(zip(self._ordered_control_names, self._ordered_control_symbols, strict=False))

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols in O(1) time."""
        return self._ordered_state_symbols.copy()

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols in O(1) time."""
        return self._ordered_control_symbols.copy()

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in O(1) time."""
        return self._ordered_state_names.copy()

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in O(1) time."""
        return self._ordered_control_names.copy()


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
