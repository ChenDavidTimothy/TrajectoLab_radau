# trajectolab/problem/constraints_problem.py
"""
Constraint processing and conversion functions for multiphase path and event constraints.
"""

from __future__ import annotations

from collections.abc import Callable

import casadi as ca

from ..tl_types import Constraint, PhaseID
from .state import (
    MultiPhaseVariableState,
    PhaseDefinition,
    _BoundaryConstraint,
)


def add_phase_path_constraint(
    phase_def: PhaseDefinition, constraint_expr: ca.MX | float | int
) -> None:
    """Add a path constraint expression to a specific phase."""
    if isinstance(constraint_expr, ca.MX):
        phase_def.path_constraints.append(constraint_expr)
    else:
        phase_def.path_constraints.append(ca.MX(constraint_expr))


def add_cross_phase_constraint(
    multiphase_state: MultiPhaseVariableState, constraint_expr: ca.MX | float | int
) -> None:
    """Add a cross-phase constraint expression."""
    if isinstance(constraint_expr, ca.MX):
        multiphase_state.cross_phase_constraints.append(constraint_expr)
    else:
        multiphase_state.cross_phase_constraints.append(ca.MX(constraint_expr))


def _symbolic_constraint_to_constraint(expr: ca.MX) -> Constraint:
    """Convert symbolic constraint to unified Constraint."""

    # Handle CasADi API compatibility - try to get proper operation constants
    try:
        # Attempt to get the actual integer constants from CasADi
        OP_EQ = getattr(ca, "OP_EQ", None)
        OP_LE = getattr(ca, "OP_LE", None)
        OP_GE = getattr(ca, "OP_GE", None)

        # Only proceed with operation checking if we have valid integer constants
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and OP_EQ is not None
            and isinstance(OP_EQ, int)
        ):
            # Handle equality constraints: expr == value
            if expr.is_op(OP_EQ):
                lhs = expr.dep(0)
                rhs = expr.dep(1)
                return Constraint(val=lhs - rhs, equals=0.0)

            # Handle inequality constraints: expr <= value
            elif OP_LE is not None and isinstance(OP_LE, int) and expr.is_op(OP_LE):
                lhs = expr.dep(0)
                rhs = expr.dep(1)
                return Constraint(val=lhs - rhs, max_val=0.0)

            # Handle inequality constraints: expr >= value
            elif OP_GE is not None and isinstance(OP_GE, int) and expr.is_op(OP_GE):
                lhs = expr.dep(0)
                rhs = expr.dep(1)
                return Constraint(val=lhs - rhs, min_val=0.0)

    except (AttributeError, TypeError, NotImplementedError):
        # CasADi API compatibility issue - fall back to safe default
        pass

    # Default case: treat as equality constraint
    # This handles cross-phase constraints like (t0_p2-tf_p1) which should equal zero
    return Constraint(val=expr, equals=0.0)


def _boundary_constraint_to_constraints(
    boundary_constraint: _BoundaryConstraint,
    variable_expression: ca.MX,
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


def get_phase_path_constraints_function(
    phase_def: PhaseDefinition,
) -> Callable[..., list[Constraint]] | None:
    """
    Get path constraints function for a specific phase.

    Path constraints are applied at every collocation point throughout the phase trajectory.
    """
    # Check if phase has any path constraints
    has_path_constraints = bool(phase_def.path_constraints)

    # Check for boundary constraints (these are path constraints)
    state_boundary_constraints = [info.boundary_constraint for info in phase_def.state_info]
    control_boundary_constraints = [info.boundary_constraint for info in phase_def.control_info]

    has_state_boundary = any(
        constraint is not None and constraint.has_constraint()
        for constraint in state_boundary_constraints
    )
    has_control_boundary = any(
        constraint is not None and constraint.has_constraint()
        for constraint in control_boundary_constraints
    )

    # If no path constraints exist, return None
    if not has_path_constraints and not has_state_boundary and not has_control_boundary:
        return None

    # Get ordered symbols for substitution
    state_syms = phase_def.get_ordered_state_symbols()
    control_syms = phase_def.get_ordered_control_symbols()

    def vectorized_path_constraints(
        states_vec: ca.MX,
        controls_vec: ca.MX,
        time: ca.MX,
        static_parameters_vec: ca.MX | None = None,
        static_parameter_symbols: list[ca.MX] | None = None,
    ) -> list[Constraint]:
        """Apply path constraints at a single collocation point."""
        result: list[Constraint] = []

        # Create substitution map for symbolic constraints
        subs_map = {}

        # Map state symbols to current state values
        for i, state_sym in enumerate(state_syms):
            subs_map[state_sym] = states_vec[i]

        # Map control symbols to current control values
        for i, control_sym in enumerate(control_syms):
            subs_map[control_sym] = controls_vec[i]

        # Map time symbol to current time
        if phase_def.sym_time is not None:
            subs_map[phase_def.sym_time] = time

        # Map static parameter symbols to current parameter values
        if static_parameters_vec is not None and static_parameter_symbols is not None:
            for i, param_sym in enumerate(static_parameter_symbols):
                if len(static_parameter_symbols) == 1:
                    subs_map[param_sym] = static_parameters_vec
                else:
                    subs_map[param_sym] = static_parameters_vec[i]

        # Process symbolic path constraints
        for expr in phase_def.path_constraints:
            substituted_expr = ca.substitute(
                [expr], list(subs_map.keys()), list(subs_map.values())
            )[0]
            result.append(_symbolic_constraint_to_constraint(substituted_expr))

        # Add state boundary constraints (applied at every point)
        for i, boundary_constraint in enumerate(state_boundary_constraints):
            if boundary_constraint is not None and boundary_constraint.has_constraint():
                result.extend(
                    _boundary_constraint_to_constraints(boundary_constraint, states_vec[i])
                )

        # Add control boundary constraints (applied at every point)
        for i, boundary_constraint in enumerate(control_boundary_constraints):
            if boundary_constraint is not None and boundary_constraint.has_constraint():
                result.extend(
                    _boundary_constraint_to_constraints(boundary_constraint, controls_vec[i])
                )

        return result

    return vectorized_path_constraints


def get_cross_phase_event_constraints_function(
    multiphase_state: MultiPhaseVariableState,
) -> Callable[..., list[Constraint]] | None:
    """
    Get cross-phase event constraints function for multiphase problems.

    This implements the CGPOPS event constraint structure:
    b_min ≤ b(E^(1), E^(2), ..., E^(P), s) ≤ b_max
    """
    # Check for cross-phase constraints
    has_cross_phase_constraints = bool(multiphase_state.cross_phase_constraints)

    # Check for phase initial/final constraints
    has_phase_event_constraints = False
    for phase_def in multiphase_state.phases.values():
        state_initial_constraints = [info.initial_constraint for info in phase_def.state_info]
        state_final_constraints = [info.final_constraint for info in phase_def.state_info]

        if any(
            constraint is not None and constraint.has_constraint()
            for constraint in (state_initial_constraints + state_final_constraints)
        ):
            has_phase_event_constraints = True
            break

    # If no event constraints exist, return None
    if not has_cross_phase_constraints and not has_phase_event_constraints:
        return None

    def vectorized_cross_phase_event_constraints(
        phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]],  # E^(p) vectors
        static_parameters_vec: ca.MX | None,
    ) -> list[Constraint]:
        """
        Apply cross-phase event constraints.

        Args:
            phase_endpoint_vectors: Dictionary mapping phase_id to endpoint data:
                {'t0': t0, 'tf': tf, 'x0': x0_vec, 'xf': xf_vec, 'q': q_vec}
            static_parameters_vec: Static parameters vector
        """
        result: list[Constraint] = []

        # Create substitution map for cross-phase constraints
        subs_map = {}

        # Map phase variables to endpoint values
        for phase_id, phase_def in multiphase_state.phases.items():
            if phase_id not in phase_endpoint_vectors:
                continue

            endpoint_data = phase_endpoint_vectors[phase_id]

            # Map time symbols
            if phase_def.sym_time_initial is not None:
                subs_map[phase_def.sym_time_initial] = endpoint_data["t0"]
            if phase_def.sym_time_final is not None:
                subs_map[phase_def.sym_time_final] = endpoint_data["tf"]
            if phase_def.sym_time is not None:
                subs_map[phase_def.sym_time] = endpoint_data["tf"]  # Default to final time

            # Map state initial/final symbols
            state_initial_syms = phase_def.get_ordered_state_initial_symbols()
            state_final_syms = phase_def.get_ordered_state_final_symbols()
            state_syms = phase_def.get_ordered_state_symbols()

            x0_vec = endpoint_data["x0"]
            xf_vec = endpoint_data["xf"]

            for i, (sym_initial, sym_final, sym_current) in enumerate(
                zip(state_initial_syms, state_final_syms, state_syms, strict=True)
            ):
                if len(state_syms) == 1:
                    subs_map[sym_initial] = x0_vec
                    subs_map[sym_final] = xf_vec
                    subs_map[sym_current] = xf_vec  # Default to final
                else:
                    subs_map[sym_initial] = x0_vec[i]
                    subs_map[sym_final] = xf_vec[i]
                    subs_map[sym_current] = xf_vec[i]  # Default to final

            # Map integral symbols
            if "q" in endpoint_data and endpoint_data["q"] is not None:
                for i, integral_sym in enumerate(phase_def.integral_symbols):
                    if i < endpoint_data["q"].shape[0]:
                        subs_map[integral_sym] = endpoint_data["q"][i]

        # Map static parameters
        if static_parameters_vec is not None:
            static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
            for i, param_sym in enumerate(static_param_syms):
                if len(static_param_syms) == 1:
                    subs_map[param_sym] = static_parameters_vec
                else:
                    subs_map[param_sym] = static_parameters_vec[i]

        # Process cross-phase constraints
        for expr in multiphase_state.cross_phase_constraints:
            substituted_expr = ca.substitute(
                [expr], list(subs_map.keys()), list(subs_map.values())
            )[0]
            result.append(_symbolic_constraint_to_constraint(substituted_expr))

        # Add phase-specific initial/final constraints
        for phase_id, phase_def in multiphase_state.phases.items():
            if phase_id not in phase_endpoint_vectors:
                continue

            endpoint_data = phase_endpoint_vectors[phase_id]
            x0_vec = endpoint_data["x0"]
            xf_vec = endpoint_data["xf"]

            # Add state initial constraints
            state_initial_constraints = [info.initial_constraint for info in phase_def.state_info]
            for i, constraint in enumerate(state_initial_constraints):
                if constraint is not None and constraint.has_constraint():
                    if len(phase_def.state_info) == 1:
                        result.extend(_boundary_constraint_to_constraints(constraint, x0_vec))
                    else:
                        result.extend(_boundary_constraint_to_constraints(constraint, x0_vec[i]))

            # Add state final constraints
            state_final_constraints = [info.final_constraint for info in phase_def.state_info]
            for i, constraint in enumerate(state_final_constraints):
                if constraint is not None and constraint.has_constraint():
                    if len(phase_def.state_info) == 1:
                        result.extend(_boundary_constraint_to_constraints(constraint, xf_vec))
                    else:
                        result.extend(_boundary_constraint_to_constraints(constraint, xf_vec[i]))

        return result

    return vectorized_cross_phase_event_constraints
