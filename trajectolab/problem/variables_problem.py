"""
Variable management functions for optimal control problems - UNIFIED CONSTRAINT API.
Implements the new unified constraint specification with initial/final/boundary parameters.
"""

from __future__ import annotations

from typing import Any, cast

import casadi as ca

from ..tl_types import SymExpr, SymType
from .state import ConstraintInput, VariableState, _BoundaryConstraint


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


def create_time_variable(
    state: VariableState,
    initial: ConstraintInput = 0.0,
    final: ConstraintInput = None,
) -> TimeVariableImpl:
    """
    Create time variable with unified constraint specification.

    Args:
        state: Variable state container
        initial: Initial time constraint (default: fixed at 0.0)
            - float/int: Fixed initial time
            - tuple(lower, upper): Range constraint for t0
            - None: Treated as fixed at 0.0
        final: Final time constraint (default: free)
            - float/int: Fixed final time
            - tuple(lower, upper): Range constraint for tf
            - None: Fully free (subject to tf > t0 + epsilon)

    Returns:
        TimeVariableImpl object with initial/final properties
    """
    # Create symbolic variables
    sym_time = ca.MX.sym("t", 1)  # type: ignore[arg-type]
    sym_t0 = ca.MX.sym("t0", 1)  # type: ignore[arg-type]
    sym_tf = ca.MX.sym("tf", 1)  # type: ignore[arg-type]

    # Handle initial time constraint
    if initial is None:
        initial = 0.0  # Default to fixed at 0.0

    try:
        t0_constraint = _BoundaryConstraint(initial)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid initial time constraint: {e}") from e

    # Handle final time constraint
    try:
        tf_constraint = _BoundaryConstraint(final)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid final time constraint: {e}") from e

    # Store constraints in state
    state.t0_constraint = t0_constraint
    state.tf_constraint = tf_constraint

    # Store symbolic variables
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
) -> SymType:
    """
    Create a state variable with unified constraint specification.

    Args:
        state: Variable state container
        name: Variable name
        initial: Initial condition constraint (event constraint at t0)
            - float/int: Fixed value at t0
            - tuple(lower, upper): Range constraint at t0
            - None: No initial constraint
        final: Final condition constraint (event constraint at tf)
            - float/int: Fixed value at tf
            - tuple(lower, upper): Range constraint at tf
            - None: No final constraint
        boundary: Path constraint (applies throughout trajectory)
            - float/int: Fixed value for entire trajectory
            - tuple(lower, upper): Range constraint for entire trajectory
            - None: No path constraint

    Returns:
        CasADi symbolic variable
    """
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Parse constraints
    try:
        initial_constraint = _BoundaryConstraint(initial) if initial is not None else None
        final_constraint = _BoundaryConstraint(final) if final is not None else None
        boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid constraint for state '{name}': {e}") from e

    # Add to unified storage
    state.add_state(
        name=name,
        symbol=sym_var,
        initial_constraint=initial_constraint,
        final_constraint=final_constraint,
        boundary_constraint=boundary_constraint,
    )

    return sym_var


def create_control_variable(
    state: VariableState,
    name: str,
    initial: ConstraintInput = None,
    final: ConstraintInput = None,
    boundary: ConstraintInput = None,
) -> SymType:
    """
    Create a control variable with unified constraint specification.

    Args:
        state: Variable state container
        name: Variable name
        initial: Initial condition constraint (event constraint at t0)
            - float/int: Fixed value at t0
            - tuple(lower, upper): Range constraint at t0
            - None: No initial constraint
        final: Final condition constraint (event constraint at tf)
            - float/int: Fixed value at tf
            - tuple(lower, upper): Range constraint at tf
            - None: No final constraint
        boundary: Path constraint (applies throughout trajectory)
            - float/int: Fixed value for entire trajectory
            - tuple(lower, upper): Range constraint for entire trajectory
            - None: No path constraint (strongly recommended to set bounds)

    Returns:
        CasADi symbolic variable
    """
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Parse constraints
    try:
        initial_constraint = _BoundaryConstraint(initial) if initial is not None else None
        final_constraint = _BoundaryConstraint(final) if final is not None else None
        boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid constraint for control '{name}': {e}") from e

    # Add to unified storage
    state.add_control(
        name=name,
        symbol=sym_var,
        initial_constraint=initial_constraint,
        final_constraint=final_constraint,
        boundary_constraint=boundary_constraint,
    )

    return sym_var


def create_parameter_variable(
    state: VariableState,
    name: str,
    value: Any,
) -> SymType:
    """Create a parameter variable."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Store parameter value (parameters remain as simple dict)
    state.parameters[name] = value

    return sym_var


def set_dynamics(state: VariableState, dynamics_dict: dict[SymType, SymExpr]) -> None:
    """Set dynamics expressions."""
    # Verify all keys correspond to defined state variables
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
