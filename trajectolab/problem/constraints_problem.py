"""
Constraint management functions for optimal control problems.
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
from .state import ConstraintState, VariableState


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
    """Get path constraints function for solver with automatic scaling handling."""

    # Filter path constraints
    path_constraints = [
        expr for expr in constraint_state.constraints if _is_path_constraint(expr, variable_state)
    ]

    # Check for variable bounds
    has_bounds = any(
        variable_state.states[s].get("lower") is not None
        or variable_state.states[s].get("upper") is not None
        for s in variable_state.states
    ) or any(
        variable_state.controls[c].get("lower") is not None
        or variable_state.controls[c].get("upper") is not None
        for c in variable_state.controls
    )

    if not path_constraints and not has_bounds:
        return None

    # Gather symbols in order
    state_syms = [
        variable_state.sym_states[name]
        for name in sorted(
            variable_state.sym_states.keys(),
            key=lambda n: variable_state.states[n]["index"],
        )
    ]
    control_syms = [
        variable_state.sym_controls[name]
        for name in sorted(
            variable_state.sym_controls.keys(),
            key=lambda n: variable_state.controls[n]["index"],
        )
    ]

    # State and control names in order (for scaling reference)
    state_names = sorted(
        variable_state.sym_states.keys(), key=lambda n: variable_state.states[n]["index"]
    )
    control_names = sorted(
        variable_state.sym_controls.keys(), key=lambda n: variable_state.controls[n]["index"]
    )

    def vectorized_path_constraints(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ) -> list[Constraint]:
        result: list[Constraint] = []

        # Check if we have scaling info attached to the params
        use_scaling = False
        scaling_obj = None

        if isinstance(params, dict):
            # Special parameters added by apply_path_constraints
            if "_use_scaling" in params:
                use_scaling = bool(params["_use_scaling"])
            if "_scaling_object" in params:
                scaling_obj = params["_scaling_object"]

        # Determine whether to unscale variables before evaluating constraints
        if use_scaling and scaling_obj is not None:
            # UNSCALE states and controls before evaluating constraints
            unscaled_states = ca.MX(states_vec)
            unscaled_controls = ca.MX(controls_vec)

            # Unscale each state
            for i, name in enumerate(state_names):
                if i < states_vec.shape[0]:
                    factor, shift = scaling_obj.get_state_scaling(name)
                    # Only unscale if we have non-default scaling
                    if not (abs(factor - 1.0) < 1e-10 and abs(shift) < 1e-10):
                        unscaled_states[i] = (states_vec[i] - shift) / factor

            # Unscale each control
            for i, name in enumerate(control_names):
                if i < controls_vec.shape[0]:
                    factor, shift = scaling_obj.get_control_scaling(name)
                    # Only unscale if we have non-default scaling
                    if not (abs(factor - 1.0) < 1e-10 and abs(shift) < 1e-10):
                        unscaled_controls[i] = (controls_vec[i] - shift) / factor

            # Create substitution map with unscaled values
            subs_map = {}

            # Map state symbols to unscaled values
            for i, state_sym in enumerate(state_syms):
                subs_map[state_sym] = unscaled_states[i]

            # Map control symbols to unscaled values
            for i, control_sym in enumerate(control_syms):
                subs_map[control_sym] = unscaled_controls[i]

            # Map time symbol
            if variable_state.sym_time is not None:
                subs_map[variable_state.sym_time] = time

        else:
            # Create substitution map with original scaled values
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

        # Add state bounds - these are already correctly defined in scaled or unscaled form
        for i, name in enumerate(
            sorted(
                variable_state.states.keys(),
                key=lambda n: variable_state.states[n]["index"],
            )
        ):
            state_def = variable_state.states[name]
            if state_def.get("lower") is not None:
                result.append(Constraint(val=states_vec[i], min_val=state_def["lower"]))
            if state_def.get("upper") is not None:
                result.append(Constraint(val=states_vec[i], max_val=state_def["upper"]))

        # Add control bounds - these are already correctly defined in scaled or unscaled form
        for i, name in enumerate(
            sorted(
                variable_state.controls.keys(),
                key=lambda n: variable_state.controls[n]["index"],
            )
        ):
            control_def = variable_state.controls[name]
            if control_def.get("lower") is not None:
                result.append(Constraint(val=controls_vec[i], min_val=control_def["lower"]))
            if control_def.get("upper") is not None:
                result.append(Constraint(val=controls_vec[i], max_val=control_def["upper"]))

        return result

    return vectorized_path_constraints


def _has_initial_or_final_state_constraints(variable_state: VariableState) -> bool:
    """Check if there are initial or final state constraints."""
    return any(
        s.get("initial_constraint") is not None or s.get("final_constraint") is not None
        for s in variable_state.states.values()
    )


def get_event_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
) -> EventConstraintsCallable | None:
    """Get event constraints function for solver."""
    # Filter event constraints
    event_constraints = [
        expr
        for expr in constraint_state.constraints
        if not _is_path_constraint(expr, variable_state)
    ]

    if not event_constraints and not _has_initial_or_final_state_constraints(variable_state):
        return None

    # Gather symbols in order
    state_syms = [
        variable_state.sym_states[name]
        for name in sorted(
            variable_state.sym_states.keys(),
            key=lambda n: variable_state.states[n]["index"],
        )
    ]

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

        # Add initial state constraints
        for i, name in enumerate(
            sorted(
                variable_state.states.keys(),
                key=lambda n: variable_state.states[n]["index"],
            )
        ):
            state_def = variable_state.states[name]
            initial_constraint = state_def.get("initial_constraint")
            if initial_constraint is not None:
                if initial_constraint.equals is not None:
                    result.append(Constraint(val=x0_vec[i], equals=initial_constraint.equals))
                else:
                    if initial_constraint.lower is not None:
                        result.append(Constraint(val=x0_vec[i], min_val=initial_constraint.lower))
                    if initial_constraint.upper is not None:
                        result.append(Constraint(val=x0_vec[i], max_val=initial_constraint.upper))

        # Add final state constraints
        for i, name in enumerate(
            sorted(
                variable_state.states.keys(),
                key=lambda n: variable_state.states[n]["index"],
            )
        ):
            state_def = variable_state.states[name]
            final_constraint = state_def.get("final_constraint")
            if final_constraint is not None:
                if final_constraint.equals is not None:
                    result.append(Constraint(val=xf_vec[i], equals=final_constraint.equals))
                else:
                    if final_constraint.lower is not None:
                        result.append(Constraint(val=xf_vec[i], min_val=final_constraint.lower))
                    if final_constraint.upper is not None:
                        result.append(Constraint(val=xf_vec[i], max_val=final_constraint.upper))

        return result

    return vectorized_event_constraints
