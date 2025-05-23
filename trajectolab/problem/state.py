"""
State data classes for problem definition - UNIFIED CONSTRAINT API.
Updated to support unified constraint specification with initial/final/boundary parameters.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType


# Constraint input type definition
ConstraintInput = float | int | tuple[float | int | None, float | int | None] | None


class _BoundaryConstraint:
    """Internal class for representing boundary constraints with unified API."""

    def __init__(self, constraint_input: ConstraintInput = None) -> None:
        """
        Create boundary constraint from unified constraint input.

        Args:
            constraint_input:
                - float/int: Equality constraint (variable = value)
                - tuple(lower, upper): Range constraint with None for unbounded
                - None: No constraint
        """
        self.equals: float | None = None
        self.lower: float | None = None
        self.upper: float | None = None

        if constraint_input is None:
            # No constraint
            pass
        elif isinstance(constraint_input, int | float):
            # Equality constraint
            self.equals = float(constraint_input)
            self.lower = float(constraint_input)
            self.upper = float(constraint_input)
        elif isinstance(constraint_input, tuple):
            # Range constraint
            if len(constraint_input) != 2:
                raise ValueError(
                    f"Constraint tuple must have exactly 2 elements, got {len(constraint_input)}"
                )

            lower_val, upper_val = constraint_input

            # Convert to float, handling None
            self.lower = None if lower_val is None else float(lower_val)
            self.upper = None if upper_val is None else float(upper_val)

            # Validate bounds relationship
            if self.lower is not None and self.upper is not None and self.lower > self.upper:
                raise ValueError(
                    f"Lower bound ({self.lower}) cannot be greater than upper bound ({self.upper})"
                )
        else:
            raise TypeError(
                f"Invalid constraint input type: {type(constraint_input)}. "
                f"Expected float, int, tuple, or None"
            )

    def has_constraint(self) -> bool:
        """Check if this boundary constraint is actually constraining anything."""
        return self.equals is not None or self.lower is not None or self.upper is not None

    def __repr__(self) -> str:
        if self.equals is not None:
            return f"_BoundaryConstraint(equals={self.equals})"

        if self.lower is not None and self.upper is not None:
            return f"_BoundaryConstraint(lower={self.lower}, upper={self.upper})"
        elif self.lower is not None:
            return f"_BoundaryConstraint(lower={self.lower})"
        elif self.upper is not None:
            return f"_BoundaryConstraint(upper={self.upper})"
        else:
            return "_BoundaryConstraint(no constraint)"


@dataclass
class _VariableInfo:
    """Internal storage for variable metadata with unified constraint API."""

    symbol: SymType
    initial_constraint: _BoundaryConstraint | None = None
    final_constraint: _BoundaryConstraint | None = None
    boundary_constraint: _BoundaryConstraint | None = None


@dataclass
class VariableState:
    """State for all variables and expressions - UNIFIED CONSTRAINT API."""

    # UNIFIED STORAGE SYSTEM - optimized ordering
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

    # Time bounds (using unified constraint format)
    t0_constraint: _BoundaryConstraint = field(default_factory=lambda: _BoundaryConstraint(0.0))
    tf_constraint: _BoundaryConstraint = field(default_factory=lambda: _BoundaryConstraint())

    # ========================================================================
    # UNIFIED VARIABLE MANAGEMENT - Single source of truth
    # ========================================================================

    def add_state(
        self,
        name: str,
        symbol: SymType,
        initial_constraint: _BoundaryConstraint | None = None,
        final_constraint: _BoundaryConstraint | None = None,
        boundary_constraint: _BoundaryConstraint | None = None,
    ) -> None:
        """Add state variable to unified storage."""
        with self._ordering_lock:
            if name in self._state_name_to_index:
                raise ValueError(f"State '{name}' already exists")

            index = len(self._state_names)
            self._state_name_to_index[name] = index
            self._state_names.append(name)

            var_info = _VariableInfo(
                symbol=symbol,
                initial_constraint=initial_constraint,
                final_constraint=final_constraint,
                boundary_constraint=boundary_constraint,
            )
            self._state_info.append(var_info)

    def add_control(
        self,
        name: str,
        symbol: SymType,
        initial_constraint: _BoundaryConstraint | None = None,
        final_constraint: _BoundaryConstraint | None = None,
        boundary_constraint: _BoundaryConstraint | None = None,
    ) -> None:
        """Add control variable to unified storage."""
        with self._ordering_lock:
            if name in self._control_name_to_index:
                raise ValueError(f"Control '{name}' already exists")

            index = len(self._control_names)
            self._control_name_to_index[name] = index
            self._control_names.append(name)

            var_info = _VariableInfo(
                symbol=symbol,
                initial_constraint=initial_constraint,
                final_constraint=final_constraint,
                boundary_constraint=boundary_constraint,
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

    def get_variable_counts(self) -> tuple[int, int]:
        """Get (num_states, num_controls)."""
        return len(self._state_info), len(self._control_info)

    # ========================================================================
    # UNIFIED CONSTRAINT ACCESS METHODS
    # ========================================================================

    def get_state_initial_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get initial state constraints in order."""
        return [info.initial_constraint for info in self._state_info]

    def get_state_final_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get final state constraints in order."""
        return [info.final_constraint for info in self._state_info]

    def get_state_boundary_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get boundary state constraints in order."""
        return [info.boundary_constraint for info in self._state_info]

    def get_control_initial_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get initial control constraints in order."""
        return [info.initial_constraint for info in self._control_info]

    def get_control_final_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get final control constraints in order."""
        return [info.final_constraint for info in self._control_info]

    def get_control_boundary_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get boundary control constraints in order."""
        return [info.boundary_constraint for info in self._control_info]

    # ========================================================================
    # TIME BOUNDS ACCESS (converted from constraint objects)
    # ========================================================================

    @property
    def t0_bounds(self) -> tuple[float, float]:
        """Get time initial bounds as tuple for compatibility."""
        if self.t0_constraint.equals is not None:
            return (self.t0_constraint.equals, self.t0_constraint.equals)

        lower = self.t0_constraint.lower if self.t0_constraint.lower is not None else -1e6
        upper = self.t0_constraint.upper if self.t0_constraint.upper is not None else 1e6
        return (lower, upper)

    @property
    def tf_bounds(self) -> tuple[float, float]:
        """Get time final bounds as tuple for compatibility."""
        if self.tf_constraint.equals is not None:
            return (self.tf_constraint.equals, self.tf_constraint.equals)

        lower = self.tf_constraint.lower if self.tf_constraint.lower is not None else -1e6
        upper = self.tf_constraint.upper if self.tf_constraint.upper is not None else 1e6
        return (lower, upper)


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
