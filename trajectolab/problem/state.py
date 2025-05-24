"""
State data classes for problem definition - UNIFIED CONSTRAINT API with ENHANCED ERROR HANDLING.
Updated to support unified constraint specification with initial/final/boundary parameters.
Added targeted constraint validation and storage operation guard clauses.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import ConfigurationError, DataIntegrityError
from ..tl_types import FloatArray, SymExpr, SymType


# Constraint input type definition
ConstraintInput = float | int | tuple[float | int | None, float | int | None] | None


class _BoundaryConstraint:
    """Internal class for representing boundary constraints with unified API and enhanced validation."""

    def __init__(self, constraint_input: ConstraintInput = None) -> None:
        """
        Create boundary constraint from unified constraint input with enhanced validation.

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
            # Guard clause: Validate numeric values
            if not isinstance(constraint_input, int | float):
                raise ConfigurationError(
                    f"Numeric constraint must be int or float, got {type(constraint_input)}",
                    "TrajectoLab constraint specification error",
                )

            # Guard clause: Check for invalid numeric values
            import math

            if math.isnan(constraint_input) or math.isinf(constraint_input):
                raise ConfigurationError(
                    f"Constraint value cannot be NaN or infinite, got {constraint_input}",
                    "TrajectoLab constraint value error",
                )

            # Equality constraint
            self.equals = float(constraint_input)
            self.lower = float(constraint_input)
            self.upper = float(constraint_input)
        elif isinstance(constraint_input, tuple):
            # Guard clause: Validate tuple structure
            if len(constraint_input) != 2:
                raise ConfigurationError(
                    f"Constraint tuple must have exactly 2 elements, got {len(constraint_input)}",
                    "TrajectoLab constraint specification error",
                )

            lower_val, upper_val = constraint_input

            # Guard clause: Validate tuple elements
            for i, val in enumerate([lower_val, upper_val]):
                if val is not None:
                    if not isinstance(val, int | float):
                        raise ConfigurationError(
                            f"Constraint bound {i} must be numeric or None, got {type(val)}",
                            "TrajectoLab constraint specification error",
                        )

                    import math

                    if math.isnan(val) or math.isinf(val):
                        raise ConfigurationError(
                            f"Constraint bound {i} cannot be NaN or infinite, got {val}",
                            "TrajectoLab constraint value error",
                        )

            # Convert to float, handling None
            self.lower = None if lower_val is None else float(lower_val)
            self.upper = None if upper_val is None else float(upper_val)

            # Guard clause: Validate bounds relationship
            if self.lower is not None and self.upper is not None and self.lower > self.upper:
                raise ConfigurationError(
                    f"Lower bound ({self.lower}) cannot be greater than upper bound ({self.upper})",
                    "TrajectoLab constraint bounds ordering error",
                )
        else:
            raise ConfigurationError(
                f"Invalid constraint input type: {type(constraint_input)}. Expected float, int, tuple, or None",
                "TrajectoLab constraint specification error",
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
    """Internal storage for variable metadata with unified constraint API and validation."""

    symbol: SymType
    initial_constraint: _BoundaryConstraint | None = None
    final_constraint: _BoundaryConstraint | None = None
    boundary_constraint: _BoundaryConstraint | None = None

    def __post_init__(self) -> None:
        """Validate variable info after initialization."""
        # Guard clause: Validate symbol
        if self.symbol is None:
            raise ConfigurationError(
                "Variable symbol cannot be None", "TrajectoLab variable definition error"
            )


@dataclass
class VariableState:
    """State for all variables and expressions - UNIFIED CONSTRAINT API with ENHANCED ERROR HANDLING."""

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
    # UNIFIED VARIABLE MANAGEMENT - Single source of truth with enhanced validation
    # ========================================================================

    def add_state(
        self,
        name: str,
        symbol: SymType,
        initial_constraint: _BoundaryConstraint | None = None,
        final_constraint: _BoundaryConstraint | None = None,
        boundary_constraint: _BoundaryConstraint | None = None,
    ) -> None:
        """Add state variable to unified storage with enhanced validation."""
        # Guard clause: Validate inputs
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(
                f"State name must be non-empty string, got {name!r}",
                "TrajectoLab variable naming error",
            )

        if symbol is None:
            raise ConfigurationError(
                f"State symbol for '{name}' cannot be None", "TrajectoLab variable definition error"
            )

        with self._ordering_lock:
            # Guard clause: Check for duplicate names
            if name in self._state_name_to_index:
                raise ConfigurationError(
                    f"State '{name}' already exists", "TrajectoLab variable naming conflict"
                )

            index = len(self._state_names)
            self._state_name_to_index[name] = index
            self._state_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
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
        """Add control variable to unified storage with enhanced validation."""
        # Guard clause: Validate inputs
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(
                f"Control name must be non-empty string, got {name!r}",
                "TrajectoLab variable naming error",
            )

        if symbol is None:
            raise ConfigurationError(
                f"Control symbol for '{name}' cannot be None",
                "TrajectoLab variable definition error",
            )

        with self._ordering_lock:
            # Guard clause: Check for duplicate names
            if name in self._control_name_to_index:
                raise ConfigurationError(
                    f"Control '{name}' already exists", "TrajectoLab variable naming conflict"
                )

            index = len(self._control_names)
            self._control_name_to_index[name] = index
            self._control_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
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
    # EFFICIENT ACCESS METHODS - Direct O(1) access with validation
    # ========================================================================

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols in order - O(1) access with validation."""
        # Guard clause: Check for consistency
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                f"State info count ({len(self._state_info)}) doesn't match names count ({len(self._state_names)})",
                "TrajectoLab variable storage inconsistency",
            )

        return [info.symbol for info in self._state_info]

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols in order - O(1) access with validation."""
        # Guard clause: Check for consistency
        if len(self._control_info) != len(self._control_names):
            raise DataIntegrityError(
                f"Control info count ({len(self._control_info)}) doesn't match names count ({len(self._control_names)})",
                "TrajectoLab variable storage inconsistency",
            )

        return [info.symbol for info in self._control_info]

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order - O(1) access."""
        return self._state_names.copy()

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order - O(1) access."""
        return self._control_names.copy()

    def get_variable_counts(self) -> tuple[int, int]:
        """Get (num_states, num_controls) with consistency validation."""
        num_states = len(self._state_info)
        num_controls = len(self._control_info)

        # Guard clause: Validate consistency
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
    # UNIFIED CONSTRAINT ACCESS METHODS with validation
    # ========================================================================

    def get_state_initial_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get initial state constraints in order with validation."""
        # Guard clause: Check consistency
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                "State constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.initial_constraint for info in self._state_info]

    def get_state_final_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get final state constraints in order with validation."""
        # Guard clause: Check consistency
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                "State constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.final_constraint for info in self._state_info]

    def get_state_boundary_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get boundary state constraints in order with validation."""
        # Guard clause: Check consistency
        if len(self._state_info) != len(self._state_names):
            raise DataIntegrityError(
                "State constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.boundary_constraint for info in self._state_info]

    def get_control_boundary_constraints(self) -> list[_BoundaryConstraint | None]:
        """Get boundary control constraints in order with validation."""
        # Guard clause: Check consistency
        if len(self._control_info) != len(self._control_names):
            raise DataIntegrityError(
                "Control constraint access failed due to storage inconsistency",
                "TrajectoLab variable storage corruption",
            )

        return [info.boundary_constraint for info in self._control_info]

    # ========================================================================
    # TIME BOUNDS ACCESS (converted from constraint objects) with validation
    # ========================================================================

    @property
    def t0_bounds(self) -> tuple[float, float]:
        """Get time initial bounds as tuple for compatibility with validation."""
        # Guard clause: Validate time constraint
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
        """Get time final bounds as tuple for compatibility with validation."""
        # Guard clause: Validate time constraint
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
    """State for constraints with enhanced validation."""

    constraints: list[SymExpr] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate constraint state after initialization."""
        # Guard clause: Validate constraints list
        if not isinstance(self.constraints, list):
            raise ConfigurationError(
                f"Constraints must be a list, got {type(self.constraints)}",
                "TrajectoLab constraint storage error",
            )


@dataclass
class MeshState:
    """State for mesh configuration with enhanced validation."""

    collocation_points_per_interval: list[int] = field(default_factory=list)
    global_normalized_mesh_nodes: FloatArray | None = None
    configured: bool = False

    def __post_init__(self) -> None:
        """Validate mesh state after initialization."""
        # Guard clause: Validate collocation points
        if not isinstance(self.collocation_points_per_interval, list):
            raise ConfigurationError(
                f"Collocation points must be a list, got {type(self.collocation_points_per_interval)}",
                "TrajectoLab mesh storage error",
            )

        # Guard clause: Validate mesh nodes if present
        if self.global_normalized_mesh_nodes is not None:
            import numpy as np

            if not isinstance(self.global_normalized_mesh_nodes, np.ndarray):
                raise ConfigurationError(
                    f"Mesh nodes must be numpy array, got {type(self.global_normalized_mesh_nodes)}",
                    "TrajectoLab mesh storage error",
                )

            # Guard clause: Check for NaN/Inf in mesh nodes
            if np.any(np.isnan(self.global_normalized_mesh_nodes)) or np.any(
                np.isinf(self.global_normalized_mesh_nodes)
            ):
                raise DataIntegrityError(
                    "Mesh nodes contain NaN or infinite values", "TrajectoLab mesh data corruption"
                )
