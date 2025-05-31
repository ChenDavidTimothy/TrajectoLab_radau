# trajectolab/problem/state.py
"""
State management classes for multiphase variables, constraints, and mesh configuration - PURGED.
All redundancy eliminated, using centralized validation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import casadi as ca

from ..exceptions import DataIntegrityError
from ..input_validation import validate_constraint_input_format, validate_string_not_empty
from ..tl_types import FloatArray, PhaseID


# Enhanced Constraint input type definition to include CasADi symbolic
ConstraintInput = float | int | tuple[float | int | None, float | int | None] | None | ca.MX


class _BoundaryConstraint:
    """Internal class for representing boundary constraints with symbolic expression support."""

    def __init__(self, constraint_input: ConstraintInput = None) -> None:
        """Create boundary constraint from unified constraint input including symbolic expressions."""
        # SINGLE validation call
        validate_constraint_input_format(constraint_input, "boundary constraint")

        self.equals: float | None = None
        self.lower: float | None = None
        self.upper: float | None = None
        self.symbolic_expression: ca.MX | None = None

        if constraint_input is None:
            pass  # No constraint
        elif isinstance(constraint_input, ca.MX):
            self.symbolic_expression = constraint_input
        elif isinstance(constraint_input, (int, float)):
            self.equals = float(constraint_input)
            self.lower = float(constraint_input)
            self.upper = float(constraint_input)
        elif isinstance(constraint_input, tuple):
            lower_val, upper_val = constraint_input
            self.lower = None if lower_val is None else float(lower_val)
            self.upper = None if upper_val is None else float(upper_val)

    def has_constraint(self) -> bool:
        """Check if this boundary constraint is actually constraining anything."""
        return (
            self.equals is not None
            or self.lower is not None
            or self.upper is not None
            or self.symbolic_expression is not None
        )

    def is_symbolic(self) -> bool:
        """Check if this is a symbolic constraint."""
        return self.symbolic_expression is not None

    def __repr__(self) -> str:
        if self.symbolic_expression is not None:
            return f"_BoundaryConstraint(symbolic={self.symbolic_expression})"
        elif self.equals is not None:
            return f"_BoundaryConstraint(equals={self.equals})"
        elif self.lower is not None and self.upper is not None:
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

    symbol: ca.MX
    initial_symbol: ca.MX | None = None
    final_symbol: ca.MX | None = None
    initial_constraint: _BoundaryConstraint | None = None
    final_constraint: _BoundaryConstraint | None = None
    boundary_constraint: _BoundaryConstraint | None = None

    def __post_init__(self) -> None:
        """Validate variable info after initialization."""
        if self.symbol is None:
            raise DataIntegrityError("Variable symbol cannot be None", "Variable definition error")


@dataclass
class PhaseDefinition:
    """Complete definition of a single phase in multiphase optimal control problem."""

    phase_id: PhaseID

    # Variable management
    state_info: list[_VariableInfo] = field(default_factory=list)
    control_info: list[_VariableInfo] = field(default_factory=list)
    state_name_to_index: dict[str, int] = field(default_factory=dict)
    control_name_to_index: dict[str, int] = field(default_factory=dict)
    state_names: list[str] = field(default_factory=list)
    control_names: list[str] = field(default_factory=list)

    # Symbolic variables
    sym_time: ca.MX | None = None
    sym_time_initial: ca.MX | None = None
    sym_time_final: ca.MX | None = None

    # Phase expressions
    dynamics_expressions: dict[ca.MX, ca.MX] = field(default_factory=dict)
    path_constraints: list[ca.MX] = field(default_factory=list)

    # Integral tracking
    integral_expressions: list[ca.MX] = field(default_factory=list)
    integral_symbols: list[ca.MX] = field(default_factory=list)
    num_integrals: int = 0

    # Time bounds
    t0_constraint: _BoundaryConstraint = field(default_factory=lambda: _BoundaryConstraint(0.0))
    tf_constraint: _BoundaryConstraint = field(default_factory=lambda: _BoundaryConstraint())

    # Mesh configuration
    collocation_points_per_interval: list[int] = field(default_factory=list)
    global_normalized_mesh_nodes: FloatArray | None = None
    mesh_configured: bool = False

    # Thread safety
    _ordering_lock: threading.Lock = field(default_factory=threading.Lock)

    # Symbolic boundary constraints collection
    symbolic_boundary_constraints: list[tuple[str, str, ca.MX]] = field(default_factory=list)

    def add_state(
        self,
        name: str,
        symbol: ca.MX,
        initial_symbol: ca.MX | None = None,
        final_symbol: ca.MX | None = None,
        initial_constraint: _BoundaryConstraint | None = None,
        final_constraint: _BoundaryConstraint | None = None,
        boundary_constraint: _BoundaryConstraint | None = None,
    ) -> None:
        """Add state variable to phase with automatic symbolic constraint collection."""
        # SINGLE validation call
        validate_string_not_empty(name, "State variable name")

        if symbol is None:
            raise DataIntegrityError(
                f"State symbol for '{name}' cannot be None", "Variable definition error"
            )

        with self._ordering_lock:
            if name in self.state_name_to_index:
                raise DataIntegrityError(
                    f"State '{name}' already exists in phase {self.phase_id}",
                    "Variable naming conflict",
                )

            index = len(self.state_names)
            self.state_name_to_index[name] = index
            self.state_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
                    initial_symbol=initial_symbol,
                    final_symbol=final_symbol,
                    initial_constraint=initial_constraint,
                    final_constraint=final_constraint,
                    boundary_constraint=boundary_constraint,
                )
                self.state_info.append(var_info)

                # Collect symbolic boundary constraints for automatic processing
                if initial_constraint is not None and initial_constraint.is_symbolic():
                    self.symbolic_boundary_constraints.append(
                        (name, "initial", initial_constraint.symbolic_expression)
                    )
                if final_constraint is not None and final_constraint.is_symbolic():
                    self.symbolic_boundary_constraints.append(
                        (name, "final", final_constraint.symbolic_expression)
                    )
                if boundary_constraint is not None and boundary_constraint.is_symbolic():
                    self.symbolic_boundary_constraints.append(
                        (name, "boundary", boundary_constraint.symbolic_expression)
                    )

            except Exception as e:
                self.state_name_to_index.pop(name, None)
                self.state_names.pop()
                raise DataIntegrityError(
                    f"Failed to create state variable info for '{name}': {e}",
                    "Variable creation error",
                ) from e

    def add_control(
        self, name: str, symbol: ca.MX, boundary_constraint: _BoundaryConstraint | None = None
    ) -> None:
        """Add control variable to phase."""
        # SINGLE validation call
        validate_string_not_empty(name, "Control variable name")

        if symbol is None:
            raise DataIntegrityError(
                f"Control symbol for '{name}' cannot be None", "Variable definition error"
            )

        with self._ordering_lock:
            if name in self.control_name_to_index:
                raise DataIntegrityError(
                    f"Control '{name}' already exists in phase {self.phase_id}",
                    "Variable naming conflict",
                )

            index = len(self.control_names)
            self.control_name_to_index[name] = index
            self.control_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
                    initial_symbol=None,
                    final_symbol=None,
                    initial_constraint=None,
                    final_constraint=None,
                    boundary_constraint=boundary_constraint,
                )
                self.control_info.append(var_info)
            except Exception as e:
                self.control_name_to_index.pop(name, None)
                self.control_names.pop()
                raise DataIntegrityError(
                    f"Failed to create control variable info for '{name}': {e}",
                    "Variable creation error",
                ) from e

    def get_variable_counts(self) -> tuple[int, int]:
        """Get (num_states, num_controls) with consistency validation."""
        num_states = len(self.state_info)
        num_controls = len(self.control_info)

        if num_states != len(self.state_names):
            raise DataIntegrityError(
                f"Phase {self.phase_id} state count inconsistency: info={num_states}, names={len(self.state_names)}",
                "Variable storage corruption",
            )

        if num_controls != len(self.control_names):
            raise DataIntegrityError(
                f"Phase {self.phase_id} control count inconsistency: info={num_controls}, names={len(self.control_names)}",
                "Variable storage corruption",
            )

        return num_states, num_controls

    def get_ordered_state_symbols(self) -> list[ca.MX]:
        """Get state symbols in order with data integrity validation."""
        if len(self.state_info) != len(self.state_names):
            raise DataIntegrityError(
                f"Phase {self.phase_id} state info count ({len(self.state_info)}) != names count ({len(self.state_names)})",
                "Variable storage inconsistency",
            )

        return [info.symbol for info in self.state_info]

    def get_ordered_control_symbols(self) -> list[ca.MX]:
        """Get control symbols in order with data integrity validation."""
        if len(self.control_info) != len(self.control_names):
            raise DataIntegrityError(
                f"Phase {self.phase_id} control info count ({len(self.control_info)}) != names count ({len(self.control_names)})",
                "Variable storage inconsistency",
            )

        return [info.symbol for info in self.control_info]

    def get_ordered_state_initial_symbols(self) -> list[ca.MX]:
        """Get state initial symbols in order."""
        symbols = []
        for info in self.state_info:
            if info.initial_symbol is None:
                raise DataIntegrityError(
                    f"Phase {self.phase_id} state initial symbol is None", "State symbol corruption"
                )
            symbols.append(info.initial_symbol)
        return symbols

    def get_ordered_state_final_symbols(self) -> list[ca.MX]:
        """Get state final symbols in order."""
        symbols = []
        for info in self.state_info:
            if info.final_symbol is None:
                raise DataIntegrityError(
                    f"Phase {self.phase_id} state final symbol is None", "State symbol corruption"
                )
            symbols.append(info.final_symbol)
        return symbols

    @property
    def t0_bounds(self) -> tuple[float, float]:
        """Get time initial bounds as tuple for compatibility."""
        if self.t0_constraint is None:
            raise DataIntegrityError(
                f"Phase {self.phase_id} initial time constraint is None", "Time bounds corruption"
            )

        # Handle symbolic constraints by providing reasonable bounds
        if self.t0_constraint.is_symbolic():
            return (-1e6, 1e6)

        if self.t0_constraint.equals is not None:
            return (self.t0_constraint.equals, self.t0_constraint.equals)

        lower = self.t0_constraint.lower if self.t0_constraint.lower is not None else -1e6
        upper = self.t0_constraint.upper if self.t0_constraint.upper is not None else 1e6
        return (lower, upper)

    @property
    def tf_bounds(self) -> tuple[float, float]:
        """Get time final bounds as tuple for compatibility."""
        if self.tf_constraint is None:
            raise DataIntegrityError(
                f"Phase {self.phase_id} final time constraint is None", "Time bounds corruption"
            )

        # Handle symbolic constraints by providing reasonable bounds
        if self.tf_constraint.is_symbolic():
            return (-1e6, 1e6)

        if self.tf_constraint.equals is not None:
            return (self.tf_constraint.equals, self.tf_constraint.equals)

        lower = self.tf_constraint.lower if self.tf_constraint.lower is not None else -1e6
        upper = self.tf_constraint.upper if self.tf_constraint.upper is not None else 1e6
        return (lower, upper)


@dataclass
class StaticParameterState:
    """State for static parameters that span across all phases."""

    parameter_info: list[_VariableInfo] = field(default_factory=list)
    parameter_name_to_index: dict[str, int] = field(default_factory=dict)
    parameter_names: list[str] = field(default_factory=list)
    _ordering_lock: threading.Lock = field(default_factory=threading.Lock)

    def add_parameter(
        self, name: str, symbol: ca.MX, boundary_constraint: _BoundaryConstraint | None = None
    ) -> None:
        """Add static parameter."""
        # SINGLE validation call
        validate_string_not_empty(name, "Parameter name")

        if symbol is None:
            raise DataIntegrityError(
                f"Parameter symbol for '{name}' cannot be None", "Variable definition error"
            )

        with self._ordering_lock:
            if name in self.parameter_name_to_index:
                raise DataIntegrityError(
                    f"Parameter '{name}' already exists", "Variable naming conflict"
                )

            index = len(self.parameter_names)
            self.parameter_name_to_index[name] = index
            self.parameter_names.append(name)

            try:
                var_info = _VariableInfo(
                    symbol=symbol,
                    initial_symbol=None,
                    final_symbol=None,
                    initial_constraint=None,
                    final_constraint=None,
                    boundary_constraint=boundary_constraint,
                )
                self.parameter_info.append(var_info)
            except Exception as e:
                self.parameter_name_to_index.pop(name, None)
                self.parameter_names.pop()
                raise DataIntegrityError(
                    f"Failed to create parameter variable info for '{name}': {e}",
                    "Variable creation error",
                ) from e

    def get_parameter_count(self) -> int:
        """Get number of static parameters."""
        return len(self.parameter_info)

    def get_ordered_parameter_symbols(self) -> list[ca.MX]:
        """Get parameter symbols in order."""
        return [info.symbol for info in self.parameter_info]


@dataclass
class MultiPhaseVariableState:
    """Complete variable state for multiphase optimal control problems."""

    phases: dict[PhaseID, PhaseDefinition] = field(default_factory=dict)
    static_parameters: StaticParameterState = field(default_factory=StaticParameterState)
    cross_phase_constraints: list[ca.MX] = field(default_factory=list)
    objective_expression: ca.MX | None = None

    def add_phase(self, phase_id: PhaseID) -> PhaseDefinition:
        """Add new phase to multiphase problem."""
        if phase_id in self.phases:
            raise DataIntegrityError(
                f"Phase {phase_id} already exists", "Phase definition conflict"
            )

        phase_def = PhaseDefinition(phase_id=phase_id)
        self.phases[phase_id] = phase_def
        return phase_def

    def get_phase_ids(self) -> list[PhaseID]:
        """Get ordered list of phase IDs."""
        return sorted(self.phases.keys())

    def get_total_variable_counts(self) -> tuple[int, int, int]:
        """Get (total_states, total_controls, num_static_params) across all phases."""
        total_states = sum(len(phase.state_info) for phase in self.phases.values())
        total_controls = sum(len(phase.control_info) for phase in self.phases.values())
        num_static_params = self.static_parameters.get_parameter_count()
        return total_states, total_controls, num_static_params
