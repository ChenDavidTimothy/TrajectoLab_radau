"""
Variable creation and management functions for optimal control problem definition.
"""

from __future__ import annotations

from typing import Any, cast

import casadi as ca

from ..tl_types import SymExpr, SymType
from .state import ConstraintInput, VariableState, _BoundaryConstraint


def _convert_expression_to_pure_symbols(expr: SymExpr) -> SymExpr:
    """
    Recursively convert expressions containing wrapper objects to pure CasADi symbols.

    This handles cases where StateVariableImpl or TimeVariableImpl objects appear
    in expressions and need to be converted to their underlying MX symbols.
    """
    if hasattr(expr, "_sym_var"):
        # This is a wrapper object - return underlying symbol
        return expr._sym_var
    elif isinstance(expr, ca.MX):
        # For MX expressions, we need to check if they contain wrapper objects
        # CasADi expressions are immutable, so if we got here, it should be pure
        return expr
    elif isinstance(expr, (int, float)):
        # Numeric literals are fine
        return expr
    else:
        # For other types, try to convert to MX
        try:
            return ca.MX(expr)
        except:
            # If conversion fails, return as-is and let CasADi handle the error
            return expr


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

    # CasADi integration - make this object transparent to CasADi operations
    def __casadi_MX__(self) -> SymType:
        """Return underlying MX symbol for CasADi operations."""
        return self._sym_var

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy/CasADi function calls by converting to underlying symbol."""
        # Convert self to underlying symbol in arguments
        converted_args = []
        for arg in args:
            if arg is self:
                converted_args.append(self._sym_var)
            else:
                converted_args.append(arg)
        return func(*converted_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying MX symbol."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._sym_var, name)

    # Make hashable for dictionary use
    def __hash__(self) -> int:
        return hash(self._sym_var)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TimeVariableImpl):
            return self._sym_var is other._sym_var
        return self._sym_var is other

    # Arithmetic operators - ensure they return pure MX expressions
    def __add__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var + other_converted

    def __radd__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted + self._sym_var

    def __sub__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var - other_converted

    def __rsub__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted - self._sym_var

    def __mul__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var * other_converted

    def __rmul__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted * self._sym_var

    def __truediv__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var / other_converted

    def __rtruediv__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted / self._sym_var

    def __pow__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var**other_converted

    def __neg__(self) -> SymType:
        return cast(SymType, -self._sym_var)

    # Comparison operators
    def __lt__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var < other_converted

    def __le__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var <= other_converted

    def __gt__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var > other_converted

    def __ge__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var >= other_converted

    def __ne__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var != other_converted


class StateVariableImpl:
    """Implementation of state variable with initial/final properties."""

    def __init__(self, sym_var: SymType, sym_initial: SymType, sym_final: SymType) -> None:
        self._sym_var = sym_var
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    def __call__(self, other: Any = None) -> SymType:
        if other is None:
            return self._sym_var
        raise NotImplementedError("State indexing not yet implemented")

    @property
    def initial(self) -> SymType:
        return self._sym_initial

    @property
    def final(self) -> SymType:
        return self._sym_final

    # CasADi integration - make this object transparent to CasADi operations
    def __casadi_MX__(self) -> SymType:
        """Return underlying MX symbol for CasADi operations."""
        return self._sym_var

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy/CasADi function calls by converting to underlying symbol."""
        # Convert self to underlying symbol in arguments
        converted_args = []
        for arg in args:
            if arg is self:
                converted_args.append(self._sym_var)
            else:
                converted_args.append(arg)
        return func(*converted_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying MX symbol."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._sym_var, name)

    # Make hashable for dictionary use
    def __hash__(self) -> int:
        return hash(self._sym_var)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StateVariableImpl):
            return self._sym_var is other._sym_var
        return self._sym_var is other

    # Arithmetic operators - ensure they return pure MX expressions
    def __add__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var + other_converted

    def __radd__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted + self._sym_var

    def __sub__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var - other_converted

    def __rsub__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted - self._sym_var

    def __mul__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var * other_converted

    def __rmul__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted * self._sym_var

    def __truediv__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var / other_converted

    def __rtruediv__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return other_converted / self._sym_var

    def __pow__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var**other_converted

    def __neg__(self) -> SymType:
        return cast(SymType, -self._sym_var)

    # Comparison operators
    def __lt__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var < other_converted

    def __le__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var <= other_converted

    def __gt__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var > other_converted

    def __ge__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var >= other_converted

    def __ne__(self, other: Any) -> SymType:
        other_converted = _convert_expression_to_pure_symbols(other)
        return self._sym_var != other_converted


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
) -> StateVariableImpl:
    """
    Create a state variable with unified constraint specification and initial/final properties.

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
        StateVariableImpl object with initial/final properties
    """
    # Create three symbolic variables for this state
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]
    sym_initial = ca.MX.sym(f"{name}_initial", 1)  # type: ignore[arg-type]
    sym_final = ca.MX.sym(f"{name}_final", 1)  # type: ignore[arg-type]

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
    boundary: ConstraintInput = None,  # Remove initial and final parameters
) -> SymType:
    """
    Create a control variable with path constraint specification.

    Args:
        state: Variable state container
        name: Variable name
        boundary: Path constraint (applies throughout trajectory)
            - float/int: Fixed value for entire trajectory
            - tuple(lower, upper): Range constraint for entire trajectory
            - None: No path constraint (strongly recommended to set bounds)

    Returns:
        CasADi symbolic variable
    """
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Parse boundary constraint only
    try:
        boundary_constraint = _BoundaryConstraint(boundary) if boundary is not None else None
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid boundary constraint for control '{name}': {e}") from e

    # Add to unified storage - no initial/final constraints
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
) -> SymType:
    """Create a parameter variable."""
    sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

    # Store parameter value (parameters remain as simple dict)
    state.parameters[name] = value

    return sym_var


def set_dynamics(state: VariableState, dynamics_dict: dict[SymType, SymExpr]) -> None:
    """Set dynamics expressions."""
    # Verify all keys correspond to defined state variables (current state symbols)
    ordered_state_symbols = state.get_ordered_state_symbols()

    for state_sym in dynamics_dict.keys():
        found = False
        # Handle both raw symbols and wrapper objects
        if hasattr(state_sym, "_sym_var"):
            # StateVariableImpl wrapper - get the underlying symbol
            underlying_sym = state_sym._sym_var
        else:
            # Raw symbol
            underlying_sym = state_sym

        for sym in ordered_state_symbols:
            if underlying_sym is sym:
                found = True
                break

        if not found:
            raise ValueError("Dynamics provided for undefined state variable")

    # Convert wrapper objects to underlying symbols for storage
    # AND ensure expressions are pure CasADi expressions
    converted_dict = {}
    for key, value in dynamics_dict.items():
        # Convert key to underlying symbol
        if hasattr(key, "_sym_var"):
            # StateVariableImpl wrapper - use underlying symbol
            storage_key = key._sym_var
        else:
            # Raw symbol
            storage_key = key

        # Convert value to pure expression
        storage_value = _convert_expression_to_pure_symbols(value)

        converted_dict[storage_key] = storage_value

    state.dynamics_expressions = converted_dict


def add_integral(state: VariableState, integrand_expr: SymExpr) -> SymType:
    """Add an integral expression."""
    integral_name = f"integral_{len(state.integral_expressions)}"
    integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

    # Convert expression to pure symbols
    pure_expr = _convert_expression_to_pure_symbols(integrand_expr)

    state.integral_expressions.append(pure_expr)
    state.integral_symbols.append(integral_sym)
    state.num_integrals = len(state.integral_expressions)

    return integral_sym


def set_objective(state: VariableState, objective_expr: SymExpr) -> None:
    """Set the objective expression."""
    # Convert expression to pure symbols
    pure_expr = _convert_expression_to_pure_symbols(objective_expr)
    state.objective_expression = pure_expr
