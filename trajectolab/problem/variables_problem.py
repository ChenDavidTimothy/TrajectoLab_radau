from __future__ import annotations

from typing import Any, cast

import casadi as ca

from ..input_validation import validate_constraint_input_format, validate_string_not_empty
from .state import (
    ConstraintInput,
    MultiPhaseVariableState,
    PhaseDefinition,
    StaticParameterState,
    _BoundaryConstraint,
)


def _extract_casadi_symbol(expr: Any) -> ca.MX:
    """Extract CasADi symbol from wrapper objects, otherwise return as-is."""
    if hasattr(expr, "_symbolic_var"):
        return expr._symbolic_var
    return expr


class _SymbolicVariableBase:
    """Base class for symbolic variable wrappers with CasADi integration."""

    def __init__(self, symbolic_var: ca.MX) -> None:
        self._symbolic_var = symbolic_var

    def __call__(self, other: Any = None) -> ca.MX:
        if other is None:
            return self._symbolic_var
        raise NotImplementedError("Variable indexing not yet implemented")

    def __casadi_MX__(self) -> ca.MX:  # noqa: N802
        return self._symbolic_var

    def __array_function__(self, func, types, args, kwargs):
        converted_args = [self._symbolic_var if arg is self else arg for arg in args]
        return func(*converted_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._symbolic_var, name)

    def __hash__(self) -> int:
        return hash(self._symbolic_var)

    def __eq__(self, other: Any) -> bool:
        if hasattr(other, "_symbolic_var"):
            return self._symbolic_var is other._symbolic_var
        return self._symbolic_var is other

    # Arithmetic operators
    def __add__(self, other: Any) -> ca.MX:
        return self._symbolic_var + other

    def __radd__(self, other: Any) -> ca.MX:
        return other + self._symbolic_var

    def __sub__(self, other: Any) -> ca.MX:
        return self._symbolic_var - other

    def __rsub__(self, other: Any) -> ca.MX:
        return other - self._symbolic_var

    def __mul__(self, other: Any) -> ca.MX:
        return self._symbolic_var * other

    def __rmul__(self, other: Any) -> ca.MX:
        return other * self._symbolic_var

    def __truediv__(self, other: Any) -> ca.MX:
        return self._symbolic_var / other

    def __rtruediv__(self, other: Any) -> ca.MX:
        return other / self._symbolic_var

    def __pow__(self, other: Any) -> ca.MX:
        return self._symbolic_var**other

    def __neg__(self) -> ca.MX:
        return cast(ca.MX, -self._symbolic_var)

    # Comparison operators
    def __lt__(self, other: Any) -> ca.MX:
        return self._symbolic_var < other

    def __le__(self, other: Any) -> ca.MX:
        return self._symbolic_var <= other

    def __gt__(self, other: Any) -> ca.MX:
        return self._symbolic_var > other

    def __ge__(self, other: Any) -> ca.MX:
        return self._symbolic_var >= other

    def __ne__(self, other: Any) -> ca.MX:
        return self._symbolic_var != other


class TimeVariableImpl(_SymbolicVariableBase):
    """Implementation of time variable with initial/final properties."""

    def __init__(self, sym_var: ca.MX, sym_initial: ca.MX, sym_final: ca.MX) -> None:
        super().__init__(sym_var)
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    @property
    def initial(self) -> ca.MX:
        return self._sym_initial

    @property
    def final(self) -> ca.MX:
        return self._sym_final


class StateVariableImpl(_SymbolicVariableBase):
    """Implementation of state variable with initial/final properties."""

    def __init__(self, sym_var: ca.MX, sym_initial: ca.MX, sym_final: ca.MX) -> None:
        super().__init__(sym_var)
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    @property
    def initial(self) -> ca.MX:
        return self._sym_initial

    @property
    def final(self) -> ca.MX:
        return self._sym_final


def create_phase_time_variable(
    phase_def: PhaseDefinition, initial: ConstraintInput = 0.0, final: ConstraintInput = None
) -> TimeVariableImpl:
    """Create time variable for a specific phase."""
    # Validate constraints
    validate_constraint_input_format(initial, f"phase {phase_def.phase_id} initial time")
    validate_constraint_input_format(final, f"phase {phase_def.phase_id} final time")

    # Create symbols
    sym_time = ca.MX.sym(f"t_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]
    sym_t0 = ca.MX.sym(f"t0_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]
    sym_tf = ca.MX.sym(f"tf_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]

    # Set defaults
    if initial is None:
        initial = 0.0

    # Create constraints
    t0_constraint = _BoundaryConstraint(initial)
    tf_constraint = _BoundaryConstraint(final)

    # Store in phase definition
    phase_def.t0_constraint = t0_constraint
    phase_def.tf_constraint = tf_constraint
    phase_def.sym_time = sym_time
    phase_def.sym_time_initial = sym_t0
    phase_def.sym_time_final = sym_tf

    return TimeVariableImpl(sym_time, sym_t0, sym_tf)


def create_phase_state_variable(
    phase_def: PhaseDefinition,
    name: str,
    initial: ConstraintInput = None,
    final: ConstraintInput = None,
    boundary: ConstraintInput = None,
) -> StateVariableImpl:
    """Create a state variable for a specific phase."""
    # Validate inputs
    validate_string_not_empty(name, "State variable name")
    validate_constraint_input_format(initial, f"phase {phase_def.phase_id} state '{name}' initial")
    validate_constraint_input_format(final, f"phase {phase_def.phase_id} state '{name}' final")
    validate_constraint_input_format(
        boundary, f"phase {phase_def.phase_id} state '{name}' boundary"
    )

    # Create symbols
    sym_var = ca.MX.sym(f"{name}_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]
    sym_initial = ca.MX.sym(f"{name}_initial_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]
    sym_final = ca.MX.sym(f"{name}_final_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]

    # Create constraints
    initial_constraint = _BoundaryConstraint(initial) if initial is not None else None
    final_constraint = _BoundaryConstraint(final) if final is not None else None
    boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None

    # Add to phase
    phase_def.add_state(
        name=name,
        symbol=sym_var,
        initial_symbol=sym_initial,
        final_symbol=sym_final,
        initial_constraint=initial_constraint,
        final_constraint=final_constraint,
        boundary_constraint=boundary_constraint,
    )

    return StateVariableImpl(sym_var, sym_initial, sym_final)


def create_phase_control_variable(
    phase_def: PhaseDefinition, name: str, boundary: ConstraintInput = None
) -> ca.MX:
    """Create a control variable for a specific phase."""
    # Validate inputs
    validate_string_not_empty(name, "Control variable name")
    validate_constraint_input_format(
        boundary, f"phase {phase_def.phase_id} control '{name}' boundary"
    )

    # Create symbol
    sym_var = ca.MX.sym(f"{name}_p{phase_def.phase_id}", 1)  # type: ignore[arg-type]

    # Create constraint
    boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None

    # Add to phase
    phase_def.add_control(name=name, symbol=sym_var, boundary_constraint=boundary_constraint)

    return sym_var


def create_static_parameter(
    static_params: StaticParameterState, name: str, boundary: ConstraintInput = None
) -> ca.MX:
    """Create a static parameter that spans across all phases."""
    # Validate inputs
    validate_string_not_empty(name, "Parameter name")
    validate_constraint_input_format(boundary, f"parameter '{name}' boundary")

    # Create symbol
    sym_var = ca.MX.sym(f"param_{name}", 1)  # type: ignore[arg-type]

    # Create constraint
    boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None

    # Add to static parameters
    static_params.add_parameter(name=name, symbol=sym_var, boundary_constraint=boundary_constraint)

    return sym_var


def set_phase_dynamics(
    phase_def: PhaseDefinition,
    dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int | StateVariableImpl],
) -> None:
    """Set dynamics expressions for a specific phase."""
    ordered_state_symbols = phase_def.get_ordered_state_symbols()

    # Validate that all keys are known state variables for this phase
    for state_sym in dynamics_dict.keys():
        found = False
        underlying_sym = _extract_casadi_symbol(state_sym)

        for sym in ordered_state_symbols:
            if underlying_sym is sym:
                found = True
                break

        if not found:
            raise ValueError(
                f"Dynamics provided for undefined state variable in phase {phase_def.phase_id}"
            )

    # Convert expressions
    converted_dict = {}
    for key, value in dynamics_dict.items():
        storage_key = _extract_casadi_symbol(key)

        try:
            if isinstance(value, ca.MX):
                storage_value = value
            elif isinstance(value, StateVariableImpl):
                storage_value = _extract_casadi_symbol(value)
            else:
                storage_value = ca.MX(value)
        except Exception as e:
            if callable(value):
                raise ValueError(
                    f"Dynamics expression appears to be a function {value}. Did you forget to call it?"
                ) from e
            else:
                raise ValueError(
                    f"Cannot convert dynamics expression of type {type(value)} to CasADi MX: {value}"
                ) from e

        converted_dict[storage_key] = storage_value

    phase_def.dynamics_expressions = converted_dict


def set_phase_integral(phase_def: PhaseDefinition, integrand_expr: ca.MX | float | int) -> ca.MX:
    """Add an integral expression for a specific phase."""
    integral_name = f"integral_{len(phase_def.integral_expressions)}_p{phase_def.phase_id}"
    integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

    try:
        if isinstance(integrand_expr, ca.MX):
            pure_expr = integrand_expr
        else:
            pure_expr = ca.MX(integrand_expr)
    except Exception as e:
        if callable(integrand_expr):
            raise ValueError(
                f"Integrand appears to be a function {integrand_expr}. Did you forget to call it?"
            ) from e
        else:
            raise ValueError(
                f"Cannot convert integrand expression of type {type(integrand_expr)} to CasADi MX: {integrand_expr}"
            ) from e

    phase_def.integral_expressions.append(pure_expr)
    phase_def.integral_symbols.append(integral_sym)
    phase_def.num_integrals = len(phase_def.integral_expressions)

    return integral_sym


def set_multiphase_objective(
    multiphase_state: MultiPhaseVariableState, objective_expr: ca.MX | float | int
) -> None:
    """Set the multiphase objective expression."""
    try:
        if isinstance(objective_expr, ca.MX):
            pure_expr = objective_expr
        else:
            pure_expr = ca.MX(objective_expr)
    except Exception as e:
        if callable(objective_expr):
            raise ValueError(
                f"Objective appears to be a function {objective_expr}. Did you forget to call it?"
            ) from e
        else:
            raise ValueError(
                f"Cannot convert objective expression of type {type(objective_expr)} to CasADi MX: {objective_expr}"
            ) from e

    multiphase_state.objective_expression = pure_expr
