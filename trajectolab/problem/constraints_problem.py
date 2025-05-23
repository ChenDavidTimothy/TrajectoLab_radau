"""
Constraint management functions for optimal control problems - SIMPLIFIED.
Removed ALL legacy code, uses only unified storage system.
"""

from __future__ import annotations

import casadi as ca

from ..tl_types import (
    CasadiMX,
    Constraint,
    EventConstraintsCallable,
    PathConstraintsCallable,
    ProblemParameters,
    SymExpr,
)
from .state import (
    ConstraintState,
    VariableState,
)


def add_constraint(state: ConstraintState, constraint_expr: SymExpr) -> None:
    """Add a constraint expression."""
    state.constraints.append(constraint_expr)


def _is_path_constraint(expr: SymExpr, variable_state: VariableState) -> bool:
    """Check if constraint is path constraint (depends only on states, controls, time)."""
    # Path constraints only depend on states, controls and time (t)
    # Not on initial/final specific values (t0/tf)
    depends_on_t0_tf = (
        variable_state.sym_time_initial is not None
        and ca.depends_on(expr, variable_state.sym_time_initial)  # type: ignore[attr-defined]
    ) or (
        variable_state.sym_time_final is not None
        and ca.depends_on(expr, variable_state.sym_time_final)  # type: ignore[attr-defined]
    )
    return not depends_on_t0_tf


def _symbolic_constraint_to_constraint(expr: SymExpr) -> Constraint:
    """Convert symbolic constraint to unified Constraint."""
    # Handle equality constraints: expr == value
    if (
        isinstance(expr, ca.MX)
        and hasattr(expr, "is_op")
        and expr.is_op(getattr(ca, "OP_EQ", "eq"))
    ):
        lhs = expr.dep(0)
        rhs = expr.dep(1)
        return Constraint(val=lhs - rhs, equals=0.0)

    # Handle inequality constraints: expr <= value or expr >= value
    elif (
        isinstance(expr, ca.MX)
        and hasattr(expr, "is_op")
        and expr.is_op(getattr(ca, "OP_LE", "le"))
    ):
        lhs = expr.dep(0)
        rhs = expr.dep(1)
        return Constraint(val=lhs - rhs, max_val=0.0)

    elif (
        isinstance(expr, ca.MX)
        and hasattr(expr, "is_op")
        and expr.is_op(getattr(ca, "OP_GE", "ge"))
    ):
        lhs = expr.dep(0)
        rhs = expr.dep(1)
        return Constraint(val=lhs - rhs, min_val=0.0)

    # Default case: treat as equality constraint
    return Constraint(val=expr, equals=0.0)


def get_path_constraints_function(
    constraint_state: ConstraintState, variable_state: VariableState
) -> PathConstraintsCallable | None:
    """Get path constraints function for solver using unified storage."""

    # Filter path constraints
    path_constraints = [
        expr for expr in constraint_state.constraints if _is_path_constraint(expr, variable_state)
    ]

    # Check for variable bounds using unified storage
    state_bounds = variable_state.get_state_bounds()
    control_bounds = variable_state.get_control_bounds()

    has_state_bounds = any(lower is not None or upper is not None for lower, upper in state_bounds)
    has_control_bounds = any(
        lower is not None or upper is not None for lower, upper in control_bounds
    )
    has_bounds = has_state_bounds or has_control_bounds

    if not path_constraints and not has_bounds:
        return None

    # Get ordered symbols and names
    state_syms = variable_state.get_ordered_state_symbols()
    control_syms = variable_state.get_ordered_control_symbols()
    state_names = variable_state.get_ordered_state_names()
    control_names = variable_state.get_ordered_control_names()

    def vectorized_path_constraints(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ) -> list[Constraint]:
        result: list[Constraint] = []

        # Create substitution map
        subs_map = {}

        # Map state symbols
        for i, state_sym in enumerate(state_syms):
            subs_map[state_sym] = states_vec[i]

        # Map control symbols
        for i, control_sym in enumerate(control_syms):
            subs_map[control_sym] = controls_vec[i]

        # Map time symbol
        if variable_state.sym_time is not None:
            subs_map[variable_state.sym_time] = time

        # Process path constraints
        for expr in path_constraints:
            substituted_expr = ca.substitute(
                [expr], list(subs_map.keys()), list(subs_map.values())
            )[0]
            result.append(_symbolic_constraint_to_constraint(substituted_expr))

        # Add state bounds using unified storage
        for i, (lower, upper) in enumerate(state_bounds):
            if lower is not None:
                result.append(Constraint(val=states_vec[i], min_val=lower))
            if upper is not None:
                result.append(Constraint(val=states_vec[i], max_val=upper))

        # Add control bounds using unified storage
        for i, (lower, upper) in enumerate(control_bounds):
            if lower is not None:
                result.append(Constraint(val=controls_vec[i], min_val=lower))
            if upper is not None:
                result.append(Constraint(val=controls_vec[i], max_val=upper))

        return result

    return vectorized_path_constraints


def _has_initial_or_final_state_constraints(variable_state: VariableState) -> bool:
    """Check if there are initial or final state constraints using unified storage."""
    initial_constraints = variable_state.get_state_initial_constraints()
    final_constraints = variable_state.get_state_final_constraints()

    return any(c is not None for c in initial_constraints) or any(
        c is not None for c in final_constraints
    )


def get_event_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
) -> EventConstraintsCallable | None:
    """Get event constraints function for solver using unified storage."""
    # Filter event constraints
    event_constraints = [
        expr
        for expr in constraint_state.constraints
        if not _is_path_constraint(expr, variable_state)
    ]

    if not event_constraints and not _has_initial_or_final_state_constraints(variable_state):
        return None

    # Get ordered symbols and names
    state_syms = variable_state.get_ordered_state_symbols()
    state_names = variable_state.get_ordered_state_names()

    def vectorized_event_constraints(
        t0: CasadiMX,
        tf: CasadiMX,
        x0_vec: CasadiMX,
        xf_vec: CasadiMX,
        q: CasadiMX | None,
        params: ProblemParameters,
    ) -> list[Constraint]:
        result: list[Constraint] = []

        # Create substitution map
        subs_map = {}

        # Map state symbols (default to final)
        for i, state_sym in enumerate(state_syms):
            subs_map[state_sym] = xf_vec[i]

        # Map time symbols
        if variable_state.sym_time_initial is not None:
            subs_map[variable_state.sym_time_initial] = t0
        if variable_state.sym_time_final is not None:
            subs_map[variable_state.sym_time_final] = tf
        if variable_state.sym_time is not None:
            subs_map[variable_state.sym_time] = tf

        # Map integral symbols
        if q is not None and len(variable_state.integral_symbols) > 0:
            for i, integral_sym in enumerate(variable_state.integral_symbols):
                if i < q.shape[0]:
                    subs_map[integral_sym] = q[i]

        # Process event constraints
        for expr in event_constraints:
            substituted_expr = ca.substitute(
                [expr], list(subs_map.keys()), list(subs_map.values())
            )[0]
            result.append(_symbolic_constraint_to_constraint(substituted_expr))

        # Add initial state constraints using unified storage
        initial_constraints = variable_state.get_state_initial_constraints()
        for i, constraint in enumerate(initial_constraints):
            if constraint is not None:
                if constraint.equals is not None:
                    result.append(Constraint(val=x0_vec[i], equals=constraint.equals))
                else:
                    if constraint.lower is not None:
                        result.append(Constraint(val=x0_vec[i], min_val=constraint.lower))
                    if constraint.upper is not None:
                        result.append(Constraint(val=x0_vec[i], max_val=constraint.upper))

        # Add final state constraints using unified storage
        final_constraints = variable_state.get_state_final_constraints()
        for i, constraint in enumerate(final_constraints):
            if constraint is not None:
                if constraint.equals is not None:
                    result.append(Constraint(val=xf_vec[i], equals=constraint.equals))
                else:
                    if constraint.lower is not None:
                        result.append(Constraint(val=xf_vec[i], min_val=constraint.lower))
                    if constraint.upper is not None:
                        result.append(Constraint(val=xf_vec[i], max_val=constraint.upper))

        return result

    return vectorized_event_constraints
