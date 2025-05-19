"""
Constraint management for optimal control problems.
"""

from __future__ import annotations

from typing import Any

import casadi as ca

from ..tl_types import (
    CasadiMX,
    EventConstraint,
    PathConstraint,
    ProblemParameters,
    SymExpr,
)


class Constraint:
    """Unified constraint class for both path and event constraints."""

    def __init__(
        self,
        val: CasadiMX | float,
        min_val: float | None = None,
        max_val: float | None = None,
        equals: float | None = None,
    ) -> None:
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.equals = equals


class ConstraintManager:
    """Manages constraints for the problem."""

    def __init__(self) -> None:
        self.constraints: list[SymExpr] = []

    def add_constraint(self, constraint_expr: SymExpr) -> None:
        """Add a constraint expression."""
        self.constraints.append(constraint_expr)

    def _is_path_constraint(self, expr: SymExpr, variable_manager: Any) -> bool:
        """Check if constraint is path constraint (depends only on states, controls, time)."""
        # Path constraints only depend on states, controls and time (t)
        # Not on initial/final specific values (t0/tf)
        depends_on_t0_tf = (
            variable_manager.sym_time_initial is not None
            and ca.depends_on(expr, variable_manager.sym_time_initial)  # type: ignore[attr-defined]
        ) or (
            variable_manager.sym_time_final is not None
            and ca.depends_on(expr, variable_manager.sym_time_final)  # type: ignore[attr-defined]
        )
        return not depends_on_t0_tf

    def _symbolic_constraint_to_path_constraint(self, expr: SymExpr) -> PathConstraint:
        """Convert symbolic constraint to PathConstraint."""
        # Handle equality constraints: expr == value
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_EQ", "eq"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs - rhs, equals=0.0)

        # Handle inequality constraints: expr <= value or expr >= value
        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_LE", "le"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs - rhs, max_val=0.0)

        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_GE", "ge"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs - rhs, min_val=0.0)

        # Default case
        return PathConstraint(val=expr, equals=0.0)

    def _symbolic_constraint_to_event_constraint(self, expr: SymExpr) -> EventConstraint:
        """Convert symbolic constraint to EventConstraint."""
        # Handle equality constraints: expr == value
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_EQ", "eq"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs - rhs, equals=0.0)

        # Handle inequality constraints: expr <= value or expr >= value
        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_LE", "le"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs - rhs, max_val=0.0)

        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_GE", "ge"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs - rhs, min_val=0.0)

        # Default case
        return EventConstraint(val=expr, equals=0.0)

    def get_path_constraints_function(self, variable_manager: Any):
        """Get path constraints function for solver."""

        # Filter path constraints
        path_constraints = [
            expr for expr in self.constraints if self._is_path_constraint(expr, variable_manager)
        ]

        # Check for variable bounds
        has_bounds = any(
            variable_manager.states[s].get("lower") is not None
            or variable_manager.states[s].get("upper") is not None
            for s in variable_manager.states
        ) or any(
            variable_manager.controls[c].get("lower") is not None
            or variable_manager.controls[c].get("upper") is not None
            for c in variable_manager.controls
        )

        if not path_constraints and not has_bounds:
            return None

        # Gather symbols in order
        state_syms = [
            variable_manager.sym_states[name]
            for name in sorted(
                variable_manager.sym_states.keys(),
                key=lambda n: variable_manager.states[n]["index"],
            )
        ]
        control_syms = [
            variable_manager.sym_controls[name]
            for name in sorted(
                variable_manager.sym_controls.keys(),
                key=lambda n: variable_manager.controls[n]["index"],
            )
        ]

        def vectorized_path_constraints(
            states_vec: CasadiMX,
            controls_vec: CasadiMX,
            time: CasadiMX,
            params: ProblemParameters,
        ) -> list[PathConstraint]:
            result: list[PathConstraint] = []

            # Create substitution map
            subs_map = {}

            # Map state symbols
            for i, state_sym in enumerate(state_syms):
                subs_map[state_sym] = states_vec[i]

            # Map control symbols
            for i, control_sym in enumerate(control_syms):
                subs_map[control_sym] = controls_vec[i]

            # Map time symbol
            if variable_manager.sym_time is not None:
                subs_map[variable_manager.sym_time] = time

            # Process path constraints
            for expr in path_constraints:
                substituted_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]
                result.append(self._symbolic_constraint_to_path_constraint(substituted_expr))

            # Add state bounds
            for i, name in enumerate(
                sorted(
                    variable_manager.states.keys(),
                    key=lambda n: variable_manager.states[n]["index"],
                )
            ):
                state_def = variable_manager.states[name]
                if state_def.get("lower") is not None:
                    result.append(PathConstraint(val=states_vec[i], min_val=state_def["lower"]))
                if state_def.get("upper") is not None:
                    result.append(PathConstraint(val=states_vec[i], max_val=state_def["upper"]))

            # Add control bounds
            for i, name in enumerate(
                sorted(
                    variable_manager.controls.keys(),
                    key=lambda n: variable_manager.controls[n]["index"],
                )
            ):
                control_def = variable_manager.controls[name]
                if control_def.get("lower") is not None:
                    result.append(PathConstraint(val=controls_vec[i], min_val=control_def["lower"]))
                if control_def.get("upper") is not None:
                    result.append(PathConstraint(val=controls_vec[i], max_val=control_def["upper"]))

            return result

        return vectorized_path_constraints

    def _has_initial_or_final_state_constraints(self, variable_manager: Any) -> bool:
        """Check if there are initial or final state constraints."""
        return any(
            s.get("initial_constraint") is not None or s.get("final_constraint") is not None
            for s in variable_manager.states.values()
        )

    def get_event_constraints_function(self, variable_manager: Any):
        """Get event constraints function for solver."""
        # Filter event constraints
        event_constraints = [
            expr
            for expr in self.constraints
            if not self._is_path_constraint(expr, variable_manager)
        ]

        if not event_constraints and not self._has_initial_or_final_state_constraints(
            variable_manager
        ):
            return None

        # Gather symbols in order
        state_syms = [
            variable_manager.sym_states[name]
            for name in sorted(
                variable_manager.sym_states.keys(),
                key=lambda n: variable_manager.states[n]["index"],
            )
        ]

        def vectorized_event_constraints(
            t0: CasadiMX,
            tf: CasadiMX,
            x0_vec: CasadiMX,
            xf_vec: CasadiMX,
            q: CasadiMX | None,
            params: ProblemParameters,
        ) -> list[EventConstraint]:
            result: list[EventConstraint] = []

            # Create substitution map
            subs_map = {}

            # Map state symbols (default to final)
            for i, state_sym in enumerate(state_syms):
                subs_map[state_sym] = xf_vec[i]

            # Map time symbols
            if variable_manager.sym_time_initial is not None:
                subs_map[variable_manager.sym_time_initial] = t0
            if variable_manager.sym_time_final is not None:
                subs_map[variable_manager.sym_time_final] = tf
            if variable_manager.sym_time is not None:
                subs_map[variable_manager.sym_time] = tf

            # Map integral symbols
            if q is not None and len(variable_manager.integral_symbols) > 0:
                for i, integral_sym in enumerate(variable_manager.integral_symbols):
                    if i < q.shape[0]:
                        subs_map[integral_sym] = q[i]

            # Process event constraints
            for expr in event_constraints:
                substituted_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]
                result.append(self._symbolic_constraint_to_event_constraint(substituted_expr))

            # Add initial state constraints
            for i, name in enumerate(
                sorted(
                    variable_manager.states.keys(),
                    key=lambda n: variable_manager.states[n]["index"],
                )
            ):
                state_def = variable_manager.states[name]
                initial_constraint = state_def.get("initial_constraint")
                if initial_constraint is not None:
                    if initial_constraint.equals is not None:
                        result.append(
                            EventConstraint(val=x0_vec[i], equals=initial_constraint.equals)
                        )
                    else:
                        if initial_constraint.lower is not None:
                            result.append(
                                EventConstraint(val=x0_vec[i], min_val=initial_constraint.lower)
                            )
                        if initial_constraint.upper is not None:
                            result.append(
                                EventConstraint(val=x0_vec[i], max_val=initial_constraint.upper)
                            )

            # Add final state constraints
            for i, name in enumerate(
                sorted(
                    variable_manager.states.keys(),
                    key=lambda n: variable_manager.states[n]["index"],
                )
            ):
                state_def = variable_manager.states[name]
                final_constraint = state_def.get("final_constraint")
                if final_constraint is not None:
                    if final_constraint.equals is not None:
                        result.append(
                            EventConstraint(val=xf_vec[i], equals=final_constraint.equals)
                        )
                    else:
                        if final_constraint.lower is not None:
                            result.append(
                                EventConstraint(val=xf_vec[i], min_val=final_constraint.lower)
                            )
                        if final_constraint.upper is not None:
                            result.append(
                                EventConstraint(val=xf_vec[i], max_val=final_constraint.upper)
                            )

            return result

        return vectorized_event_constraints
