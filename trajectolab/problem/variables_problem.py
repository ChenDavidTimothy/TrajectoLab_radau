"""
Variable creation and management functions for optimal control problem definition.
"""

from __future__ import annotations

from typing import Any, cast

import casadi as ca

from ..tl_types import CasadiMX, SymExpr
from .state import ConstraintInput, VariableState, _BoundaryConstraint


def _convert_expression_to_pure_symbols(expr: SymExpr) -> SymExpr:
    """Convert expressions containing wrapper objects to pure CasADi symbols."""
    symbolic_var = getattr(expr, "_symbolic_var", None)
    if symbolic_var is not None:
        return symbolic_var
    elif isinstance(expr, ca.MX):
        return expr
    elif isinstance(expr, int | float):
        return expr
    else:
        try:
            return ca.MX(expr)
        except Exception as e:
            # FIXED: Provide better error handling instead of silent fallback
            import inspect

            if inspect.isfunction(expr) or inspect.ismethod(expr):
                raise ValueError(
                    f"Cannot convert function/method {expr} to CasADi expression. "
                    f"Ensure all expressions in dynamics are properly evaluated symbolic expressions."
                ) from e
            else:
                raise ValueError(
                    f"Cannot convert expression of type {type(expr)} to CasADi MX: {expr}. "
                    f"Original error: {e}"
                ) from e


class _SymbolicVariableBase:
    """Base class for symbolic variable wrappers with CasADi integration."""

    def __init__(self, symbolic_var: CasadiMX) -> None:
        self._symbolic_var = symbolic_var

    def __call__(self, other: Any = None) -> CasadiMX:
        if other is None:
            return self._symbolic_var
        raise NotImplementedError("Variable indexing not yet implemented")

    def __casadi_MX__(self) -> CasadiMX:  # noqa: N802
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

    # Arithmetic operators
    def __add__(self, other: Any) -> CasadiMX:
        return self._symbolic_var + _convert_expression_to_pure_symbols(other)

    def __radd__(self, other: Any) -> CasadiMX:
        return _convert_expression_to_pure_symbols(other) + self._symbolic_var

    def __sub__(self, other: Any) -> CasadiMX:
        return self._symbolic_var - _convert_expression_to_pure_symbols(other)

    def __rsub__(self, other: Any) -> CasadiMX:
        return _convert_expression_to_pure_symbols(other) - self._symbolic_var

    def __mul__(self, other: Any) -> CasadiMX:
        return self._symbolic_var * _convert_expression_to_pure_symbols(other)

    def __rmul__(self, other: Any) -> CasadiMX:
        return _convert_expression_to_pure_symbols(other) * self._symbolic_var

    def __truediv__(self, other: Any) -> CasadiMX:
        return self._symbolic_var / _convert_expression_to_pure_symbols(other)

    def __rtruediv__(self, other: Any) -> CasadiMX:
        return _convert_expression_to_pure_symbols(other) / self._symbolic_var

    def __pow__(self, other: Any) -> CasadiMX:
        return self._symbolic_var ** _convert_expression_to_pure_symbols(other)

    def __neg__(self) -> CasadiMX:
        return cast(CasadiMX, -self._symbolic_var)

    # Comparison operators
    def __lt__(self, other: Any) -> CasadiMX:
        return self._symbolic_var < _convert_expression_to_pure_symbols(other)

    def __le__(self, other: Any) -> CasadiMX:
        return self._symbolic_var <= _convert_expression_to_pure_symbols(other)

    def __gt__(self, other: Any) -> CasadiMX:
        return self._symbolic_var > _convert_expression_to_pure_symbols(other)

    def __ge__(self, other: Any) -> CasadiMX:
        return self._symbolic_var >= _convert_expression_to_pure_symbols(other)

    def __ne__(self, other: Any) -> CasadiMX:
        return self._symbolic_var != _convert_expression_to_pure_symbols(other)


class TimeVariableImpl(_SymbolicVariableBase):
    """Implementation of time variable with initial/final properties."""

    def __init__(self, sym_var: CasadiMX, sym_initial: CasadiMX, sym_final: CasadiMX) -> None:
        super().__init__(sym_var)
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    @property
    def initial(self) -> CasadiMX:
        return self._sym_initial

    @property
    def final(self) -> CasadiMX:
        return self._sym_final


class StateVariableImpl(_SymbolicVariableBase):
    """Implementation of state variable with initial/final properties."""

    def __init__(self, sym_var: CasadiMX, sym_initial: CasadiMX, sym_final: CasadiMX) -> None:
        super().__init__(sym_var)
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    @property
    def initial(self) -> CasadiMX:
        return self._sym_initial

    @property
    def final(self) -> CasadiMX:
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
) -> CasadiMX:
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
) -> CasadiMX:
    """Create a parameter variable."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]
    state.parameters[name] = value
    return sym_var


def set_dynamics(
    state: VariableState, dynamics_dict: dict[CasadiMX | StateVariableImpl, SymExpr]
) -> None:
    """Set dynamics expressions."""
    ordered_state_symbols = state.get_ordered_state_symbols()

    for state_sym in dynamics_dict.keys():
        found = False
        underlying_sym = getattr(state_sym, "_symbolic_var", state_sym)

        for sym in ordered_state_symbols:
            if underlying_sym is sym:
                found = True
                break

        if not found:
            raise ValueError("Dynamics provided for undefined state variable")

    converted_dict = {}
    for key, value in dynamics_dict.items():
        storage_key = getattr(key, "_symbolic_var", key)
        storage_value = _convert_expression_to_pure_symbols(value)
        converted_dict[storage_key] = storage_value

    state.dynamics_expressions = converted_dict


def add_integral(state: VariableState, integrand_expr: SymExpr) -> CasadiMX:
    """Add an integral expression."""
    integral_name = f"integral_{len(state.integral_expressions)}"
    integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

    pure_expr = _convert_expression_to_pure_symbols(integrand_expr)
    state.integral_expressions.append(pure_expr)
    state.integral_symbols.append(integral_sym)
    state.num_integrals = len(state.integral_expressions)

    return integral_sym


def set_objective(state: VariableState, objective_expr: SymExpr) -> None:
    """Set the objective expression."""
    pure_expr = _convert_expression_to_pure_symbols(objective_expr)
    state.objective_expression = pure_expr
