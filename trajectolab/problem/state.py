"""
State management classes for variables, constraints, and mesh configuration.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import DataIntegrityError
from ..input_validation import validate_constraint_input_format, validate_variable_name
from ..tl_types import FloatArray, SymExpr, SymType


# Constraint input type definition
ConstraintInput = float | int | tuple[float | int | None, float | int | None] | None


class _BoundaryConstraint:
    """Internal class for representing boundary constraints - USES CENTRALIZED VALIDATION."""

    def __init__(self, constraint_input: ConstraintInput = None) -> None:
        """
        Create boundary constraint from unified constraint input.

        Uses centralized validation from input_validation.py
        """
        # CENTRALIZED VALIDATION - single call replaces all scattered validation logic
        validate_constraint_input_format(constraint_input, "boundary constraint")

        self.equals: float | None = None
        self.lower: float | None = None
        self.upper: float | None = None

        if constraint_input is None:
            # No constraint
            pass
        elif isinstance(constraint_input, int | float):
            # Equality constraint (validation already done)
            self.equals = float(constraint_input)
            self.lower = float(constraint_input)
            self.upper = float(constraint_input)
        elif isinstance(constraint_input, tuple):
            # Range constraint (validation already done)
            lower_val, upper_val = constraint_input
            self.lower = None if lower_val is None else float(lower_val)
            self.upper = None if upper_val is None else float(upper_val)

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
    """Internal storage for variable metadata with initial/final symbol support."""

    symbol: SymType
    initial_symbol: SymType | None = None  # For states only
    final_symbol: SymType | None = None  # For states only
    initial_constraint: _BoundaryConstraint | None = None
    final_constraint: _BoundaryConstraint | None = None
    boundary_constraint: _BoundaryConstraint | None = None

    def __post_init__(self) -> None:
        """Validate variable info after initialization."""
        # Guard clause: Validate symbol (this is internal data integrity)
        if self.symbol is None:
            raise DataIntegrityError(
                "Variable symbol cannot be None", "TrajectoLab variable definition error"
            )


@dataclass
class VariableState:
    """State for all variables and expressions - USES CENTRALIZED VALIDATION."""

    # UNIFIED STORAGE SYSTEM
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
    # UNIFIED VARIABLE MANAGEMENT - USES CENTRALIZED VALIDATION
    # ========================================================================

    def add_state(
        self,
        name: str,
        symbol: SymType,
        initial_symbol: SymType | None = None,
        final_symbol: SymType | None = None,
        initial_constraint: _BoundaryConstraint | None = None,
        final_constraint: _BoundaryConstraint | None = None,
        boundary_constraint: _BoundaryConstraint | None = None,
    ) -> None:
        """Add state variable to unified storage - USES CENTRALIZED VALIDATION."""
        # CENTRALIZED VALIDATION - single call replaces scattered validation
        validate_variable_name(name, "state")

        # Data integrity check (not user configuration)
        if symbol is None:
            raise DataIntegrityError(
                f"State symbol for '{name}' cannot be None", "TrajectoLab variable definition error"
            )

        with self._ordering_lock:
            # Data integrity check (internal consistency)
            if name in self._state_name_to_index:
                raise DataIntegrityError(
                    f"State '{name}' already exists", "TrajectoLab variable naming conflict"
                )

            index = len(self._state_names)
            self._state_name_to_index[name] = index
            self._state_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
                    initial_symbol=initial_symbol,
                    final_symbol=final_symbol,
                    initial_constraint=initial_constraint,
                    final_constraint=final_constraint,
                    boundary_constraint=boundary_constraint,
                )
                self._state_info.append(var_info)
            except Exception as e:
                # Clean up on failure
                self._state_name_to_index.pop(name, None)
                self._state_names.pop()
                raise DataIntegrityError(
                    f"Failed to create state variable info for '{name}': {e}",
                    "TrajectoLab variable creation error",
                ) from e

    def add_control(
        self,
        name: str,
        symbol: SymType,
        boundary_constraint: _BoundaryConstraint | None = None,
    ) -> None:
        """Add control variable to unified storage - USES CENTRALIZED VALIDATION."""
        # CENTRALIZED VALIDATION - single call replaces scattered validation
        validate_variable_name(name, "control")

        # Data integrity check (not user configuration)
        if symbol is None:
            raise DataIntegrityError(
                f"Control symbol for '{name}' cannot be None",
                "TrajectoLab variable definition error",
            )

        with self._ordering_lock:
            # Data integrity check (internal consistency)
            if name in self._control_name_to_index:
                raise DataIntegrityError(
                    f"Control '{name}' already exists", "TrajectoLab variable naming conflict"
                )

            index = len(self._control_names)
            self._control_name_to_index[name] = index
            self._control_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
                    initial_symbol=None,  # Controls don't have initial/final symbols
                    final_symbol=None,  # Controls don't have initial/final symbols
                    initial_constraint=None,  # Controls don't have initial constraints
                    final_constraint=None,  # Controls don't have final constraints
                    boundary_constraint=boundary_constraint,
                )
                self._control_info.append(var_info)
            except Exception as e:
                # Clean up on failure
                self._control_name_to_index.pop(name, None)
                self._control_names.pop()
                raise DataIntegrityError(
                    f"Failed to create control variable info for '{name}': {e}",
                    "TrajectoLab variable creation error",
                ) from e

    # ========================================================================
    # EFFICIENT ACCESS METHODS - Data integrity validation only
    # ========================================================================

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols in order - with data integrity validation."""
        # Data integrity check (internal consistency)
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                f"State info count ({len(self._state_info)}) doesn't match names count ({len(self._state_names)})",
                "TrajectoLab variable storage inconsistency",
            )

        return [info.symbol for info in self._state_info]

    def get_ordered_state_initial_symbols(self) -> list[SymType]:
        """Get state initial symbols in order."""
        # Data integrity check (internal consistency)
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                f"State info count ({len(self._state_info)}) doesn't match names count ({len(self._state_names)})",
                "TrajectoLab variable storage inconsistency",
            )

        symbols = []
        for info in self._state_info:
            if info.initial_symbol is None:
                raise DataIntegrityError(
                    "State initial symbol is None", "TrajectoLab state symbol corruption"
                )
            symbols.append(info.initial_symbol)
        return symbols

    def get_ordered_state_final_symbols(self) -> list[SymType]:
        """Get state final symbols in order."""
        # Data integrity check (internal consistency)
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                f"State info count ({len(self._state_info)}) doesn't match names count ({len(self._state_names)})",
                "TrajectoLab variable storage inconsistency",
            )

        symbols = []
        for info in self._state_info:
            if info.final_symbol is None:
                raise DataIntegrityError(
                    "State final symbol is None", "TrajectoLab state symbol corruption"
                )
            symbols.append(info.final_symbol)
        return symbols

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols in order - with data integrity validation."""
        # Data integrity check (internal consistency)
        if len(self._control_info) != len(self._control_names):
            raise DataIntegrityError(
                f"Control info count ({len(self._control_info)}) doesn't match names count ({len(self._control_names)})",
                "TrajectoLab variable storage inconsistency",
            )

        return [info.symbol for info in self._control_info]

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order."""
        return self._state_names.copy()

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order."""
        return self._control_names.copy()

    def get_variable_counts(self) -> tuple[int, int]:
        """Get (num_states, num_controls) with consistency validation."""
        num_states = len(self._state_info)
        num_controls = len(self._control_info)

        # Data integrity checks (internal consistency)
        if num_states != len(self._state_names):
            raise DataIntegrityError(
                f"State count inconsistency: info={num_states}, names={len(self._state_names)}",
                "TrajectoLab variable storage corruption",
            )

        if num_controls != len(self._control_names):
            raise DataIntegrityError(
                f"Control count inconsistency: info={num_controls}, names={len(self._control_names)}",
                "TrajectoLab variable storage corruption",
            )

        return num_states, num_controls

    # ========================================================================
    # UNIFIED CONSTRAINT ACCESS METHODS - Data integrity validation only
    # ========================================================================

    def get_state_initial_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get initial state constraints in order with validation."""
        # Data integrity check (internal consistency)
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                "State constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.initial_constraint for info in self._state_info]

    def get_state_final_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get final state constraints in order with validation."""
        # Data integrity check (internal consistency)
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                "State constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.final_constraint for info in self._state_info]

    def get_state_boundary_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get boundary state constraints in order with validation."""
        # Data integrity check (internal consistency)
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                "State constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.boundary_constraint for info in self._state_info]

    def get_control_boundary_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get boundary control constraints in order with validation."""
        # Data integrity check (internal consistency)
        if len(self._control_info) != len(self._control_names):
            raise DataIntegrityError(
                "Control constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.boundary_constraint for info in self._control_info]

    # ========================================================================
    # TIME BOUNDS ACCESS - Data integrity validation only
    # ========================================================================

    @property
    def t0_bounds(self) -> tuple[float, float]:
        """Get time initial bounds as tuple for compatibility."""
        # Data integrity check (internal consistency)
        if self.t0_constraint is None:
            raise DataIntegrityError(
                "Initial time constraint is None", "TrajectoLab time bounds corruption"
            )

        if self.t0_constraint.equals is not None:
            return (self.t0_constraint.equals, self.t0_constraint.equals)

        lower = self.t0_constraint.lower if self.t0_constraint.lower is not None else -1e6
        upper = self.t0_constraint.upper if self.t0_constraint.upper is not None else 1e6
        return (lower, upper)

    @property
    def tf_bounds(self) -> tuple[float, float]:
        """Get time final bounds as tuple for compatibility."""
        # Data integrity check (internal consistency)
        if self.tf_constraint is None:
            raise DataIntegrityError(
                "Final time constraint is None", "TrajectoLab time bounds corruption"
            )

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

    def __post_init__(self) -> None:
        """Validate mesh state after initialization - data integrity only."""
        # Data integrity checks (internal consistency)
        if not isinstance(self.collocation_points_per_interval, list):
            raise DataIntegrityError(
                f"Collocation points must be a list, got {type(self.collocation_points_per_interval)}",
                "TrajectoLab mesh storage error",
            )

        if self.global_normalized_mesh_nodes is not None:
            import numpy as np

            if not isinstance(self.global_normalized_mesh_nodes, np.ndarray):
                raise DataIntegrityError(
                    f"Mesh nodes must be numpy array, got {type(self.global_normalized_mesh_nodes)}",
                    "TrajectoLab mesh storage error",
                )

            # Check for NaN/Inf in mesh nodes (data integrity)
            if np.any(np.isnan(self.global_normalized_mesh_nodes)) or np.any(
                np.isinf(self.global_normalized_mesh_nodes)
            ):
                raise DataIntegrityError(
                    "Mesh nodes contain NaN or infinite values", "TrajectoLab mesh data corruption"
                )
