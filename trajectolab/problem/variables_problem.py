"""
Variable management functions for optimal control problems.
OPTIMIZED: Uses efficient variable ordering system.
"""

from __future__ import annotations

from typing import Any, cast

import casadi as ca

from ..tl_types import SymExpr, SymType
from .state import VariableState


class TimeVariableImpl:
    """Implementation of time variable with initial/final properties."""

    def __init__(self, sym_var: SymType, sym_initial: SymType, sym_final: SymType) -> None:
        self._sym_var = sym_var
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    def __call__(self, other: Any = None) -> SymType:
        if other is None:
            return self._sym_var
        raise NotImplementedError("Time indexing not yet implemented")

    @property
    def initial(self) -> SymType:
        return self._sym_initial

    @property
    def final(self) -> SymType:
        return self._sym_final

    # Arithmetic operators
    def __add__(self, other: Any) -> SymType:
        return self._sym_var + other

    def __radd__(self, other: Any) -> SymType:
        return other + self._sym_var

    def __sub__(self, other: Any) -> SymType:
        return self._sym_var - other

    def __rsub__(self, other: Any) -> SymType:
        return other - self._sym_var

    def __mul__(self, other: Any) -> SymType:
        return self._sym_var * other

    def __rmul__(self, other: Any) -> SymType:
        return other * self._sym_var

    def __truediv__(self, other: Any) -> SymType:
        return self._sym_var / other

    def __rtruediv__(self, other: Any) -> SymType:
        return other / self._sym_var

    def __pow__(self, other: Any) -> SymType:
        return self._sym_var**other

    def __neg__(self) -> SymType:
        return cast(SymType, -self._sym_var)

    # Comparison operators
    def __lt__(self, other: Any) -> SymType:
        return self._sym_var < other

    def __le__(self, other: Any) -> SymType:
        return self._sym_var <= other

    def __gt__(self, other: Any) -> SymType:
        return self._sym_var > other

    def __ge__(self, other: Any) -> SymType:
        return self._sym_var >= other

    def __eq__(self, other: Any) -> SymType:
        return self._sym_var == other

    def __ne__(self, other: Any) -> SymType:
        return self._sym_var != other


# Internal constraint class for boundaries
class _BoundaryConstraint:
    """Internal class for representing boundary constraints."""

    def __init__(
        self,
        val: SymExpr | None = None,
        lower: float | None = None,
        upper: float | None = None,
        equals: float | None = None,
    ) -> None:
        self.val = val
        self.lower = lower
        self.upper = upper
        self.equals = equals

        if equals is not None:
            self.lower = equals
            self.upper = equals


def create_time_variable(
    state: VariableState,
    initial: float = 0.0,
    final: float | None = None,
    free_final: bool = False,
) -> TimeVariableImpl:
    """Create time variable with bounds."""
    # Create symbolic variables
    sym_time = ca.MX.sym("t", 1)  # type: ignore[arg-type]
    sym_t0 = ca.MX.sym("t0", 1)  # type: ignore[arg-type]
    sym_tf = ca.MX.sym("tf", 1)  # type: ignore[arg-type]

    # Set time bounds
    t0_bounds = (initial, initial)  # Fixed initial time

    if free_final:
        tf_bounds = (0.0, final if final is not None else 1e6)
    else:
        tf_val = final if final is not None else initial
        tf_bounds = (tf_val, tf_val)

    state.t0_bounds = t0_bounds
    state.tf_bounds = tf_bounds

    # Store symbolic variables
    state.sym_time = sym_time
    state.sym_time_initial = sym_t0
    state.sym_time_final = sym_tf

    return TimeVariableImpl(sym_time, sym_t0, sym_tf)


def create_state_variable(
    state: VariableState,
    name: str,
    initial: float | None = None,
    final: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> SymType:
    """Create a state variable using optimized ordering."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Use optimized addition method
    state.add_state_optimized(
        name=name,
        symbol=sym_var,
        initial_constraint=None if initial is None else _BoundaryConstraint(equals=initial),
        final_constraint=None if final is None else _BoundaryConstraint(equals=final),
        lower=lower,
        upper=upper,
    )

    return sym_var


def create_control_variable(
    state: VariableState,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> SymType:
    """Create a control variable using optimized ordering."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Use optimized addition method
    state.add_control_optimized(
        name=name,
        symbol=sym_var,
        lower=lower,
        upper=upper,
    )

    return sym_var


def create_parameter_variable(
    state: VariableState,
    name: str,
    value: Any,
) -> SymType:
    """Create a parameter variable."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Store parameter value
    state.parameters[name] = value

    # Store symbolic variable
    state.sym_parameters[name] = sym_var
    return sym_var


def set_dynamics(state: VariableState, dynamics_dict: dict[SymType, SymExpr]) -> None:
    """Set dynamics expressions."""
    # Verify all keys correspond to defined state variables using optimized access
    ordered_state_symbols = state.get_ordered_state_symbols()

    for state_sym in dynamics_dict.keys():
        found = False
        for sym in ordered_state_symbols:
            if state_sym is sym:
                found = True
                break

        if not found:
            raise ValueError("Dynamics provided for undefined state variable")

    state.dynamics_expressions = dynamics_dict


def add_integral(state: VariableState, integrand_expr: SymExpr) -> SymType:
    """Add an integral expression."""
    integral_name = f"integral_{len(state.integral_expressions)}"
    integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

    state.integral_expressions.append(integrand_expr)
    state.integral_symbols.append(integral_sym)
    state.num_integrals = len(state.integral_expressions)

    return integral_sym


def set_objective(state: VariableState, objective_expr: SymExpr) -> None:
    """Set the objective expression."""
    state.objective_expression = objective_expr
