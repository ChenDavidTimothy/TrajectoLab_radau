"""
State data classes for problem definition - SIMPLIFIED.
Removed ALL legacy dual storage systems. Uses only optimized single storage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType


# Internal constraint class for boundaries
class _BoundaryConstraint:
    """Internal class for representing boundary constraints."""

    def __init__(
        self,
        equals: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        self.equals = equals
        self.lower = lower
        self.upper = upper

        if equals is not None:
            self.lower = equals
            self.upper = equals


@dataclass
class _VariableInfo:
    """Internal storage for variable metadata."""

    symbol: SymType
    lower: float | None = None
    upper: float | None = None
    initial_constraint: _BoundaryConstraint | None = None
    final_constraint: _BoundaryConstraint | None = None


@dataclass
class VariableState:
    """State for all variables and expressions - SIMPLIFIED SINGLE STORAGE."""

    # SINGLE STORAGE SYSTEM - optimized ordering
    _state_info: list[_VariableInfo] = field(default_factory=list)
    _control_info: list[_VariableInfo] = field(default_factory=list)
    _state_name_to_index: dict[str, int] = field(default_factory=dict)
    _control_name_to_index: dict[str, int] = field(default_factory=dict)
    _state_names: list[str] = field(default_factory=list)
    _control_names: list[str] = field(default_factory=list)
    _ordering_lock: threading.Lock = field(default_factory=threading.Lock)

    # Parameters (simple dict is sufficient)
    parameters: dict[str, Any] = field(default_factory=dict)

    # Symbolic time variables
    sym_time: SymType | None = None
    sym_time_initial: SymType | None = None
    sym_time_final: SymType | None = None

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
    # UNIFIED VARIABLE MANAGEMENT - Single source of truth
    # ========================================================================

    def add_state(
        self,
        name: str,
        symbol: SymType,
        initial_constraint: _BoundaryConstraint | None = None,
        final_constraint: _BoundaryConstraint | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """Add state variable to unified storage."""
        with self._ordering_lock:
            if name in self._state_name_to_index:
                raise ValueError(f"State {name} already exists")

            index = len(self._state_names)
            self._state_name_to_index[name] = index
            self._state_names.append(name)

            var_info = _VariableInfo(
                symbol=symbol,
                lower=lower,
                upper=upper,
                initial_constraint=initial_constraint,
                final_constraint=final_constraint,
            )
            self._state_info.append(var_info)

    def add_control(
        self,
        name: str,
        symbol: SymType,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """Add control variable to unified storage."""
        with self._ordering_lock:
            if name in self._control_name_to_index:
                raise ValueError(f"Control {name} already exists")

            index = len(self._control_names)
            self._control_name_to_index[name] = index
            self._control_names.append(name)

            var_info = _VariableInfo(
                symbol=symbol,
                lower=lower,
                upper=upper,
            )
            self._control_info.append(var_info)

    # ========================================================================
    # EFFICIENT ACCESS METHODS - Direct O(1) access
    # ========================================================================

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols in order - O(1) access."""
        return [info.symbol for info in self._state_info]

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols in order - O(1) access."""
        return [info.symbol for info in self._control_info]

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order - O(1) access."""
        return self._state_names.copy()

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order - O(1) access."""
        return self._control_names.copy()

    def get_state_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get state bounds in order."""
        return [(info.lower, info.upper) for info in self._state_info]

    def get_control_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get control bounds in order."""
        return [(info.lower, info.upper) for info in self._control_info]

    def get_state_initial_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get initial state constraints in order."""
        return [info.initial_constraint for info in self._state_info]

    def get_state_final_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get final state constraints in order."""
        return [info.final_constraint for info in self._state_info]

    def get_variable_counts(self) -> tuple[int, int]:
        """Get (num_states, num_controls)."""
        return len(self._state_info), len(self._control_info)


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
