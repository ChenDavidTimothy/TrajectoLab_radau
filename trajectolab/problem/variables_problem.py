"""
Variable creation and management functions for optimal control problem definition.
"""

from __future__ import annotations

from typing import Any, cast

import casadi as ca

from .state import ConstraintInput, VariableState, _BoundaryConstraint


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
        """Return underlying MX symbol for CasADi operations."""
        return self._symbolic_var

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy/CasADi function calls by converting to underlying symbol."""
        converted_args = []
        for arg in args:
            if arg is self:
                converted_args.append(self._symbolic_var)
            else:
                converted_args.append(arg)
        return func(*converted_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying MX symbol."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._symbolic_var, name)

    def __hash__(self) -> int:
        return hash(self._symbolic_var)

    def __eq__(self, other: Any) -> bool:
        if hasattr(other, "_symbolic_var"):
            return self._symbolic_var is other._symbolic_var
        return self._symbolic_var is other

    # Simplified arithmetic operators - let CasADi handle conversion
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

    # Simplified comparison operators
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


def create_time_variable(
    state: VariableState,
    initial: ConstraintInput = 0.0,
    final: ConstraintInput = None,
) -> TimeVariableImpl:
    """Create time variable with unified constraint specification."""
    sym_time = ca.MX.sym("t", 1)  # type: ignore[arg-type]
    sym_t0 = ca.MX.sym("t0", 1)  # type: ignore[arg-type]
    sym_tf = ca.MX.sym("tf", 1)  # type: ignore[arg-type]

    if initial is None:
        initial = 0.0

    try:
        t0_constraint = _BoundaryConstraint(initial)
        tf_constraint = _BoundaryConstraint(final)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid time constraint: {e}") from e

    state.t0_constraint = t0_constraint
    state.tf_constraint = tf_constraint
    state.sym_time = sym_time
    state.sym_time_initial = sym_t0
    state.sym_time_final = sym_tf

    return TimeVariableImpl(sym_time, sym_t0, sym_tf)


def create_state_variable(
    state: VariableState,
    name: str,
    initial: ConstraintInput = None,
    final: ConstraintInput = None,
    boundary: ConstraintInput = None,
) -> StateVariableImpl:
    """Create a state variable with unified constraint specification and initial/final properties."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]
    sym_initial = ca.MX.sym(f"{name}_initial", 1)  # type: ignore[arg-type]
    sym_final = ca.MX.sym(f"{name}_final", 1)  # type: ignore[arg-type]

    try:
        initial_constraint = _BoundaryConstraint(initial) if initial is not None else None
        final_constraint = _BoundaryConstraint(final) if final is not None else None
        boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid constraint for state '{name}': {e}") from e

    state.add_state(
        name=name,
        symbol=sym_var,
        initial_symbol=sym_initial,
        final_symbol=sym_final,
        initial_constraint=initial_constraint,
        final_constraint=final_constraint,
        boundary_constraint=boundary_constraint,
    )

    return StateVariableImpl(sym_var, sym_initial, sym_final)


def create_control_variable(
    state: VariableState,
    name: str,
    boundary: ConstraintInput = None,
) -> ca.MX:
    """Create a control variable with path constraint specification."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    try:
        boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid boundary constraint for control '{name}': {e}") from e

    state.add_control(
        name=name,
        symbol=sym_var,
        boundary_constraint=boundary_constraint,
    )

    return sym_var


def create_parameter_variable(
    state: VariableState,
    name: str,
    value: Any,
) -> ca.MX:
    """Create a parameter variable."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]
    state.parameters[name] = value
    return sym_var


def set_dynamics(
    state: VariableState, dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int]
) -> None:
    """Set dynamics expressions with improved error handling."""
    ordered_state_symbols = state.get_ordered_state_symbols()

    # Validate that all keys are known state variables
    for state_sym in dynamics_dict.keys():
        found = False
        underlying_sym = _extract_casadi_symbol(state_sym)

        for sym in ordered_state_symbols:
            if underlying_sym is sym:
                found = True
                break

        if not found:
            raise ValueError("Dynamics provided for undefined state variable")

    # Convert expressions with better error handling
    converted_dict = {}
    for key, value in dynamics_dict.items():
        storage_key = _extract_casadi_symbol(key)

        try:
            # Let CasADi handle the conversion
            if isinstance(value, ca.MX):
                storage_value = value
            else:
                storage_value = ca.MX(value)
        except Exception as e:
            # Provide helpful error messages for common mistakes
            if callable(value):
                raise ValueError(
                    f"Dynamics expression appears to be a function {value}. "
                    f"Did you forget to call it? Use f(x, u) not f."
                ) from e
            else:
                raise ValueError(
                    f"Cannot convert dynamics expression of type {type(value)} to CasADi MX: {value}. "
                    f"Original error: {e}"
                ) from e

        converted_dict[storage_key] = storage_value

    state.dynamics_expressions = converted_dict


def add_integral(state: VariableState, integrand_expr: ca.MX | float | int) -> ca.MX:
    """Add an integral expression with improved error handling."""
    integral_name = f"integral_{len(state.integral_expressions)}"
    integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

    try:
        # Let CasADi handle the conversion
        if isinstance(integrand_expr, ca.MX):
            pure_expr = integrand_expr
        else:
            pure_expr = ca.MX(integrand_expr)
    except Exception as e:
        if callable(integrand_expr):
            raise ValueError(
                f"Integrand appears to be a function {integrand_expr}. "
                f"Did you forget to call it? Use f(x, u) not f."
            ) from e
        else:
            raise ValueError(
                f"Cannot convert integrand expression of type {type(integrand_expr)} to CasADi MX: {integrand_expr}. "
                f"Original error: {e}"
            ) from e

    state.integral_expressions.append(pure_expr)
    state.integral_symbols.append(integral_sym)
    state.num_integrals = len(state.integral_expressions)

    return integral_sym


def set_objective(state: VariableState, objective_expr: ca.MX | float | int) -> None:
    """Set the objective expression with improved error handling."""
    try:
        # Let CasADi handle the conversion
        if isinstance(objective_expr, ca.MX):
            pure_expr = objective_expr
        else:
            pure_expr = ca.MX(objective_expr)
    except Exception as e:
        if callable(objective_expr):
            raise ValueError(
                f"Objective appears to be a function {objective_expr}. "
                f"Did you forget to call it? Use f(x, u) not f."
            ) from e
        else:
            raise ValueError(
                f"Cannot convert objective expression of type {type(objective_expr)} to CasADi MX: {objective_expr}. "
                f"Original error: {e}"
            ) from e

    state.objective_expression = pure_expr
