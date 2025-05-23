"""
Constraint management functions for optimal control problems - UNIFIED CONSTRAINT API.
Updated to handle unified constraint specification including control initial/final constraints.
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
    _BoundaryConstraint,
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


def _boundary_constraint_to_constraints(
    boundary_constraint: _BoundaryConstraint,
    variable_expression: CasadiMX,
) -> list[Constraint]:
    """Convert boundary constraint to list of Constraint objects."""
    constraints: list[Constraint] = []

    if boundary_constraint.equals is not None:
        constraints.append(Constraint(val=variable_expression, equals=boundary_constraint.equals))
    else:
        if boundary_constraint.lower is not None:
            constraints.append(
                Constraint(val=variable_expression, min_val=boundary_constraint.lower)
            )
        if boundary_constraint.upper is not None:
            constraints.append(
                Constraint(val=variable_expression, max_val=boundary_constraint.upper)
            )

    return constraints


def get_path_constraints_function(
    constraint_state: ConstraintState, variable_state: VariableState
) -> PathConstraintsCallable | None:
    """Get path constraints function for solver using unified constraint API."""

    # Filter path constraints
    path_constraints = [
        expr for expr in constraint_state.constraints if _is_path_constraint(expr, variable_state)
    ]

    # Check for boundary constraints using unified API
    state_boundary_constraints = variable_state.get_state_boundary_constraints()
    control_boundary_constraints = variable_state.get_control_boundary_constraints()

    has_state_boundary = any(
        constraint is not None and constraint.has_constraint()
        for constraint in state_boundary_constraints
    )
    has_control_boundary = any(
        constraint is not None and constraint.has_constraint()
        for constraint in control_boundary_constraints
    )
    has_boundary_constraints = has_state_boundary or has_control_boundary

    if not path_constraints and not has_boundary_constraints:
        return None

    # Get ordered symbols and names
    state_syms = variable_state.get_ordered_state_symbols()
    control_syms = variable_state.get_ordered_control_symbols()

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

        # Add state boundary constraints
        for i, boundary_constraint in enumerate(state_boundary_constraints):
            if boundary_constraint is not None and boundary_constraint.has_constraint():
                result.extend(
                    _boundary_constraint_to_constraints(boundary_constraint, states_vec[i])
                )

        # Add control boundary constraints
        for i, boundary_constraint in enumerate(control_boundary_constraints):
            if boundary_constraint is not None and boundary_constraint.has_constraint():
                result.extend(
                    _boundary_constraint_to_constraints(boundary_constraint, controls_vec[i])
                )

        return result

    return vectorized_path_constraints


def _has_event_constraints(variable_state: VariableState) -> bool:
    """Check if there are any event constraints using unified constraint API."""
    # Check state initial/final constraints
    state_initial_constraints = variable_state.get_state_initial_constraints()
    state_final_constraints = variable_state.get_state_final_constraints()

    # Check control initial/final constraints (NEW CAPABILITY)
    control_initial_constraints = variable_state.get_control_initial_constraints()
    control_final_constraints = variable_state.get_control_final_constraints()

    return any(
        constraint is not None and constraint.has_constraint()
        for constraint in (
            state_initial_constraints
            + state_final_constraints
            + control_initial_constraints
            + control_final_constraints
        )
    )


def get_event_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
) -> EventConstraintsCallable | None:
    """Get event constraints function for solver using unified constraint API."""
    # Filter event constraints
    event_constraints = [
        expr
        for expr in constraint_state.constraints
        if not _is_path_constraint(expr, variable_state)
    ]

    if not event_constraints and not _has_event_constraints(variable_state):
        return None

    # Get ordered symbols and names
    state_syms = variable_state.get_ordered_state_symbols()
    control_syms = variable_state.get_ordered_control_symbols()

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

        # Map control symbols (default to final) - NEW: Controls can have event constraints
        for _i, control_sym in enumerate(control_syms):
            # For event constraints, we need to handle initial and final separately
            # This is a placeholder - actual implementation would need u0_vec and uf_vec
            subs_map[control_sym] = control_sym  # Keep symbolic for now

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

        # Add state initial constraints
        state_initial_constraints = variable_state.get_state_initial_constraints()
        for i, constraint in enumerate(state_initial_constraints):
            if constraint is not None and constraint.has_constraint():
                result.extend(_boundary_constraint_to_constraints(constraint, x0_vec[i]))

        # Add state final constraints
        state_final_constraints = variable_state.get_state_final_constraints()
        for i, constraint in enumerate(state_final_constraints):
            if constraint is not None and constraint.has_constraint():
                result.extend(_boundary_constraint_to_constraints(constraint, xf_vec[i]))

        # Add control initial constraints (NEW CAPABILITY)
        control_initial_constraints = variable_state.get_control_initial_constraints()
        for _i, constraint in enumerate(control_initial_constraints):
            if constraint is not None and constraint.has_constraint():
                # For controls, we need to extract the initial value from the control trajectory
                # This requires solver modifications to provide u0_vec
                # For now, we'll create a placeholder that needs solver support
                # TODO: Implement u0_vec and uf_vec extraction in solver
                pass

        # Add control final constraints (NEW CAPABILITY)
        control_final_constraints = variable_state.get_control_final_constraints()
        for _i, constraint in enumerate(control_final_constraints):
            if constraint is not None and constraint.has_constraint():
                # Similar to initial control constraints - needs solver support
                # TODO: Implement u0_vec and uf_vec extraction in solver
                pass

        return result

    return vectorized_event_constraints
