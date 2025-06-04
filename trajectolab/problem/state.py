from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import casadi as ca

from ..exceptions import DataIntegrityError
from ..input_validation import validate_constraint_input_format, validate_string_not_empty
from ..tl_types import FloatArray, PhaseID


# Enhanced Constraint input type definition to include CasADi symbolic
ConstraintInput: TypeAlias = (
    float | int | tuple[float | int | None, float | int | None] | None | ca.MX
)


def _register_variable_name(
    name: str, name_to_index: dict[str, int], names_list: list[str], error_context: str
) -> int:
    # Thread-safe registration with collision detection for variable naming consistency
    if name in name_to_index:
        raise DataIntegrityError(
            f"{error_context} '{name}' already exists", "Variable naming conflict"
        )

    index = len(names_list)
    name_to_index[name] = index
    names_list.append(name)
    return index


def _rollback_variable_registration(
    name: str, name_to_index: dict[str, int], names_list: list[str]
) -> None:
    # Rollback mechanism for exception safety during variable creation
    name_to_index.pop(name, None)
    names_list.pop()


class _BoundaryConstraint:
    """Internal class for representing boundary constraints with symbolic expression support."""

    def __init__(self, constraint_input: ConstraintInput = None) -> None:
        # Unified constraint creation supporting both numerical and symbolic specifications
        validate_constraint_input_format(constraint_input, "boundary constraint")

        self.equals: float | None = None
        self.lower: float | None = None
        self.upper: float | None = None
        self.symbolic_expression: ca.MX | None = None

        if constraint_input is None:
            pass
        elif isinstance(constraint_input, ca.MX):
            self.symbolic_expression = constraint_input
        elif isinstance(constraint_input, int | float):
            self.equals = float(constraint_input)
            self.lower = float(constraint_input)
            self.upper = float(constraint_input)
        elif isinstance(constraint_input, tuple):
            lower_val, upper_val = constraint_input
            self.lower = None if lower_val is None else float(lower_val)
            self.upper = None if upper_val is None else float(upper_val)

    def has_constraint(self) -> bool:
        return (
            self.equals is not None
            or self.lower is not None
            or self.upper is not None
            or self.symbolic_expression is not None
        )

    def is_symbolic(self) -> bool:
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


@dataclass
class PhaseDefinition:
    """Complete definition of a single phase in multiphase optimal control problem."""

    phase_id: PhaseID

    # Variable management with thread-safe ordering
    state_info: list[_VariableInfo] = field(default_factory=list)
    control_info: list[_VariableInfo] = field(default_factory=list)
    state_name_to_index: dict[str, int] = field(default_factory=dict)
    control_name_to_index: dict[str, int] = field(default_factory=dict)
    state_names: list[str] = field(default_factory=list)
    control_names: list[str] = field(default_factory=list)

    # Symbolic variables for expression building
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

    # Time bounds with defaults for trajectory optimization
    t0_constraint: _BoundaryConstraint = field(default_factory=lambda: _BoundaryConstraint(0.0))
    tf_constraint: _BoundaryConstraint = field(default_factory=lambda: _BoundaryConstraint())

    # Mesh configuration for numerical discretization
    collocation_points_per_interval: list[int] = field(default_factory=list)
    global_normalized_mesh_nodes: FloatArray | None = None
    mesh_configured: bool = False

    # Thread safety for concurrent access
    _ordering_lock: threading.Lock = field(default_factory=threading.Lock)

    # Symbolic boundary constraints collection for automatic processing
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
        # State variable addition with automatic symbolic constraint collection
        validate_string_not_empty(name, "State variable name")

        with self._ordering_lock:
            _register_variable_name(
                name, self.state_name_to_index, self.state_names, f"State in phase {self.phase_id}"
            )

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

                # Automatic symbolic constraint collection for cross-phase processing
                if initial_constraint is not None and initial_constraint.is_symbolic():
                    self.symbolic_boundary_constraints.append(
                        (name, "initial", cast(ca.MX, initial_constraint.symbolic_expression))
                    )
                if final_constraint is not None and final_constraint.is_symbolic():
                    self.symbolic_boundary_constraints.append(
                        (name, "final", cast(ca.MX, final_constraint.symbolic_expression))
                    )
                if boundary_constraint is not None and boundary_constraint.is_symbolic():
                    self.symbolic_boundary_constraints.append(
                        (name, "boundary", cast(ca.MX, boundary_constraint.symbolic_expression))
                    )

            except Exception as e:
                _rollback_variable_registration(name, self.state_name_to_index, self.state_names)
                raise DataIntegrityError(
                    f"Failed to create state variable info for '{name}': {e}",
                    "Variable creation error",
                ) from e

    def add_control(
        self, name: str, symbol: ca.MX, boundary_constraint: _BoundaryConstraint | None = None
    ) -> None:
        validate_string_not_empty(name, "Control variable name")

        with self._ordering_lock:
            _register_variable_name(
                name,
                self.control_name_to_index,
                self.control_names,
                f"Control in phase {self.phase_id}",
            )

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
                _rollback_variable_registration(
                    name, self.control_name_to_index, self.control_names
                )
                raise DataIntegrityError(
                    f"Failed to create control variable info for '{name}': {e}",
                    "Variable creation error",
                ) from e

    def get_variable_counts(self) -> tuple[int, int]:
        return len(self.state_info), len(self.control_info)

    def get_ordered_state_symbols(self) -> list[ca.MX]:
        return [info.symbol for info in self.state_info]

    def get_ordered_control_symbols(self) -> list[ca.MX]:
        return [info.symbol for info in self.control_info]

    def get_ordered_state_initial_symbols(self) -> list[ca.MX | None]:
        return [info.initial_symbol for info in self.state_info]

    def get_ordered_state_final_symbols(self) -> list[ca.MX | None]:
        return [info.final_symbol for info in self.state_info]

    def _get_time_bounds(
        self, constraint: _BoundaryConstraint, constraint_type: str
    ) -> tuple[float, float]:
        # Time bounds extraction with consistent logic for symbolic constraints
        if constraint.is_symbolic():
            return (-1e6, 1e6)
        if constraint.equals is not None:
            return (constraint.equals, constraint.equals)
        lower = constraint.lower if constraint.lower is not None else -1e6
        upper = constraint.upper if constraint.upper is not None else 1e6
        return (lower, upper)

    @property
    def t0_bounds(self) -> tuple[float, float]:
        return self._get_time_bounds(self.t0_constraint, "initial")

    @property
    def tf_bounds(self) -> tuple[float, float]:
        return self._get_time_bounds(self.tf_constraint, "final")


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
        validate_string_not_empty(name, "Parameter name")

        with self._ordering_lock:
            _register_variable_name(
                name, self.parameter_name_to_index, self.parameter_names, "Parameter"
            )

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
                _rollback_variable_registration(
                    name, self.parameter_name_to_index, self.parameter_names
                )
                raise DataIntegrityError(
                    f"Failed to create parameter variable info for '{name}': {e}",
                    "Variable creation error",
                ) from e

    def get_parameter_count(self) -> int:
        return len(self.parameter_info)

    def get_ordered_parameter_symbols(self) -> list[ca.MX]:
        return [info.symbol for info in self.parameter_info]


@dataclass
class MultiPhaseVariableState:
    """Complete variable state for multiphase optimal control problems."""

    phases: dict[PhaseID, PhaseDefinition] = field(default_factory=dict)
    static_parameters: StaticParameterState = field(default_factory=StaticParameterState)
    cross_phase_constraints: list[ca.MX] = field(default_factory=list)
    objective_expression: ca.MX | None = None

    def set_phase(self, phase_id: PhaseID) -> PhaseDefinition:
        # Phase creation with uniqueness enforcement
        if phase_id in self.phases:
            raise DataIntegrityError(
                f"Phase {phase_id} already exists", "Phase definition conflict"
            )

        phase_def = PhaseDefinition(phase_id=phase_id)
        self.phases[phase_id] = phase_def
        return phase_def

    def _get_phase_ids(self) -> list[PhaseID]:
        return sorted(self.phases.keys())

    def get_total_variable_counts(self) -> tuple[int, int, int]:
        # Aggregate counts across all phases for solver sizing
        total_states = sum(len(phase.state_info) for phase in self.phases.values())
        total_controls = sum(len(phase.control_info) for phase in self.phases.values())
        num_static_params = self.static_parameters.get_parameter_count()
        return total_states, total_controls, num_static_params
