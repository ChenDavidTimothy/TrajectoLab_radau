"""
Fixed constraint handling to avoid boolean evaluation of MX expressions.
"""

from __future__ import annotations

import casadi as ca

from ..tl_types import (
    CasadiMX,
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

    # Use CasADi's depends_on function properly
    depends_on_t0_tf = False

    if variable_state.sym_time_initial is not None:
        try:
            depends_on_t0_tf = bool(ca.depends_on(expr, variable_state.sym_time_initial))
        except Exception:
            # If depends_on fails, assume it doesn't depend
            pass

    if variable_state.sym_time_final is not None and not depends_on_t0_tf:
        try:
            depends_on_t0_tf = bool(ca.depends_on(expr, variable_state.sym_time_final))
        except Exception:
            # If depends_on fails, assume it doesn't depend
            pass

    return not depends_on_t0_tf


def _symbolic_constraint_to_constraint(expr: SymExpr):
    """Convert symbolic constraint to unified Constraint."""
    from ..tl_types import Constraint

    # Handle equality constraints: expr == value
    try:
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_EQ", "eq"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return Constraint(val=lhs - rhs, equals=0.0)
    except Exception:
        pass

    # Handle inequality constraints: expr <= value or expr >= value
    try:
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_LE", "le"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return Constraint(val=lhs - rhs, max_val=0.0)
    except Exception:
        pass

    try:
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_GE", "ge"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return Constraint(val=lhs - rhs, min_val=0.0)
    except Exception:
        pass

    # Default case: treat as equality constraint
    from ..tl_types import Constraint

    return Constraint(val=expr, equals=0.0)


def get_path_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
):
    """Get path constraints function for solver."""
    from ..tl_types import Constraint, ProblemParameters

    # Filter path constraints - be more careful with the filtering
    path_constraints = []
    for expr in constraint_state.constraints:
        try:
            if _is_path_constraint(expr, variable_state):
                path_constraints.append(expr)
        except Exception:
            # If we can't determine if it's a path constraint, assume it is
            path_constraints.append(expr)

    # Check for variable bounds
    has_bounds = False
    try:
        has_bounds = any(
            variable_state.states[s].get("lower") is not None
            or variable_state.states[s].get("upper") is not None
            for s in variable_state.states
        ) or any(
            variable_state.controls[c].get("lower") is not None
            or variable_state.controls[c].get("upper") is not None
            for c in variable_state.controls
        )
    except Exception:
        has_bounds = False

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
            subs_map[state_sym] = states_vec[i] if states_vec.size() > i else states_vec

        # Map control symbols
        for i, control_sym in enumerate(control_syms):
            subs_map[control_sym] = controls_vec[i] if controls_vec.size() > i else controls_vec

        # Map time symbol
        if variable_state.sym_time is not None:
            subs_map[variable_state.sym_time] = time

        # Process path constraints
        for expr in path_constraints:
            try:
                substituted_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]
                result.append(_symbolic_constraint_to_constraint(substituted_expr))
            except Exception as e:
                print(f"Warning: Failed to process constraint {expr}: {e}")

        # Add state bounds
        for i, name in enumerate(
            sorted(
                variable_state.states.keys(),
                key=lambda n: variable_state.states[n]["index"],
            )
        ):
            state_def = variable_state.states[name]
            try:
                if state_def.get("lower") is not None:
                    val = states_vec[i] if states_vec.size() > i else states_vec
                    result.append(Constraint(val=val, min_val=state_def["lower"]))
                if state_def.get("upper") is not None:
                    val = states_vec[i] if states_vec.size() > i else states_vec
                    result.append(Constraint(val=val, max_val=state_def["upper"]))
            except Exception as e:
                print(f"Warning: Failed to add bounds for state {name}: {e}")

        # Add control bounds
        for i, name in enumerate(
            sorted(
                variable_state.controls.keys(),
                key=lambda n: variable_state.controls[n]["index"],
            )
        ):
            control_def = variable_state.controls[name]
            try:
                if control_def.get("lower") is not None:
                    val = controls_vec[i] if controls_vec.size() > i else controls_vec
                    result.append(Constraint(val=val, min_val=control_def["lower"]))
                if control_def.get("upper") is not None:
                    val = controls_vec[i] if controls_vec.size() > i else controls_vec
                    result.append(Constraint(val=val, max_val=control_def["upper"]))
            except Exception as e:
                print(f"Warning: Failed to add bounds for control {name}: {e}")

        return result

    return vectorized_path_constraints


def _has_initial_or_final_state_constraints(variable_state: VariableState) -> bool:
    """Check if there are initial or final state constraints."""
    try:
        return any(
            s.get("initial_constraint") is not None or s.get("final_constraint") is not None
            for s in variable_state.states.values()
        )
    except Exception:
        return False


def get_event_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
):
    """Get event constraints function for solver."""
    from ..tl_types import Constraint, ProblemParameters

    # Filter event constraints
    event_constraints = []
    for expr in constraint_state.constraints:
        try:
            if not _is_path_constraint(expr, variable_state):
                event_constraints.append(expr)
        except Exception:
            # If we can't determine, assume it's an event constraint
            event_constraints.append(expr)

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
            try:
                subs_map[state_sym] = xf_vec[i] if xf_vec.size() > i else xf_vec
            except Exception:
                subs_map[state_sym] = xf_vec

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
                try:
                    if i < q.shape[0]:
                        subs_map[integral_sym] = q[i]
                    else:
                        subs_map[integral_sym] = q
                except Exception:
                    subs_map[integral_sym] = q

        # Process event constraints
        for expr in event_constraints:
            try:
                substituted_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]
                result.append(_symbolic_constraint_to_constraint(substituted_expr))
            except Exception as e:
                print(f"Warning: Failed to process event constraint {expr}: {e}")

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
                try:
                    val = x0_vec[i] if x0_vec.size() > i else x0_vec
                    if initial_constraint.equals is not None:
                        result.append(Constraint(val=val, equals=initial_constraint.equals))
                    else:
                        if initial_constraint.lower is not None:
                            result.append(Constraint(val=val, min_val=initial_constraint.lower))
                        if initial_constraint.upper is not None:
                            result.append(Constraint(val=val, max_val=initial_constraint.upper))
                except Exception as e:
                    print(f"Warning: Failed to add initial constraint for {name}: {e}")

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
                try:
                    val = xf_vec[i] if xf_vec.size() > i else xf_vec
                    if final_constraint.equals is not None:
                        result.append(Constraint(val=val, equals=final_constraint.equals))
                    else:
                        if final_constraint.lower is not None:
                            result.append(Constraint(val=val, min_val=final_constraint.lower))
                        if final_constraint.upper is not None:
                            result.append(Constraint(val=val, max_val=final_constraint.upper))
                except Exception as e:
                    print(f"Warning: Failed to add final constraint for {name}: {e}")

        return result

    return vectorized_event_constraints
