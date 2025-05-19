"""
Variable management for optimal control problems.
"""

from __future__ import annotations

from typing import Any, cast

import casadi as ca

from ..tl_types import SymExpr, SymType


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


class VariableManager:
    """Manages all variables and expressions for the problem."""

    def __init__(self) -> None:
        # Symbolic variables
        self.sym_states: dict[str, SymType] = {}
        self.sym_controls: dict[str, SymType] = {}
        self.sym_parameters: dict[str, SymType] = {}
        self.sym_time: SymType | None = None
        self.sym_time_initial: SymType | None = None
        self.sym_time_final: SymType | None = None

        # Variable metadata
        self.states: dict[str, dict[str, Any]] = {}
        self.controls: dict[str, dict[str, Any]] = {}
        self.parameters: dict[str, Any] = {}

        # Expressions
        self.dynamics_expressions: dict[SymType, SymExpr] = {}
        self.objective_expression: SymExpr | None = None

        # Integral tracking
        self.integral_expressions: list[SymExpr] = []
        self.integral_symbols: list[SymType] = []
        self.num_integrals: int = 0

        # Time bounds
        self.t0_bounds: tuple[float, float] = (0.0, 0.0)
        self.tf_bounds: tuple[float, float] = (1.0, 1.0)

    def create_time_variable(
        self, initial: float = 0.0, final: float | None = None, free_final: bool = False
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

        self.t0_bounds = t0_bounds
        self.tf_bounds = tf_bounds

        # Store symbolic variables
        self.sym_time = sym_time
        self.sym_time_initial = sym_t0
        self.sym_time_final = sym_tf

        return TimeVariableImpl(sym_time, sym_t0, sym_tf)

    def create_state_variable(
        self,
        name: str,
        initial: float | None = None,
        final: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> SymType:
        """Create a state variable."""
        sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

        # Store metadata
        self.states[name] = {
            "index": len(self.states),
            "initial_constraint": None if initial is None else _BoundaryConstraint(equals=initial),
            "final_constraint": None if final is None else _BoundaryConstraint(equals=final),
            "lower": lower,
            "upper": upper,
        }

        # Store symbolic variable
        self.sym_states[name] = sym_var
        return sym_var

    def create_control_variable(
        self, name: str, lower: float | None = None, upper: float | None = None
    ) -> SymType:
        """Create a control variable."""
        sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

        # Store metadata
        self.controls[name] = {"index": len(self.controls), "lower": lower, "upper": upper}

        # Store symbolic variable
        self.sym_controls[name] = sym_var
        return sym_var

    def create_parameter_variable(self, name: str, value: Any) -> SymType:
        """Create a parameter variable."""
        sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

        # Store parameter value
        self.parameters[name] = value

        # Store symbolic variable
        self.sym_parameters[name] = sym_var
        return sym_var

    def set_dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """Set dynamics expressions."""
        # Verify all keys correspond to defined state variables
        for state_sym in dynamics_dict.keys():
            found = False
            for sym in self.sym_states.values():
                if state_sym is sym:
                    found = True
                    break

            if not found:
                raise ValueError("Dynamics provided for undefined state variable")

        self.dynamics_expressions = dynamics_dict

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        """Add an integral expression."""
        integral_name = f"integral_{len(self.integral_expressions)}"
        integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

        self.integral_expressions.append(integrand_expr)
        self.integral_symbols.append(integral_sym)
        self.num_integrals = len(self.integral_expressions)

        return integral_sym

    def set_objective(self, objective_expr: SymExpr) -> None:
        """Set the objective expression."""
        self.objective_expression = objective_expr
