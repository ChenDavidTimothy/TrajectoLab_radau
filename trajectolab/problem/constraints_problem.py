from __future__ import annotations

from collections.abc import Callable

import casadi as ca

from ..tl_types import Constraint, PhaseID
from .state import MultiPhaseVariableState, PhaseDefinition, _BoundaryConstraint


def add_path_constraint(phase_def: PhaseDefinition, constraint_expr: ca.MX | float | int) -> None:
    """Add a path constraint expression to a specific phase."""
    if isinstance(constraint_expr, ca.MX):
        phase_def.path_constraints.append(constraint_expr)
    else:
        phase_def.path_constraints.append(ca.MX(constraint_expr))


def add_event_constraint(
    multiphase_state: MultiPhaseVariableState, constraint_expr: ca.MX | float | int
) -> None:
    """Add an event constraint expression to the multiphase problem."""
    if isinstance(constraint_expr, ca.MX):
        multiphase_state.cross_phase_constraints.append(constraint_expr)
    else:
        multiphase_state.cross_phase_constraints.append(ca.MX(constraint_expr))


def add_cross_phase_constraint(
    multiphase_state: MultiPhaseVariableState, constraint_expr: ca.MX | float | int
) -> None:
    if isinstance(constraint_expr, ca.MX):
        multiphase_state.cross_phase_constraints.append(constraint_expr)
    else:
        multiphase_state.cross_phase_constraints.append(ca.MX(constraint_expr))


def _symbolic_constraint_to_constraint(expr: ca.MX) -> Constraint:
    # converting symbolic constraint to unified Constraint.
    try:
        OP_EQ = getattr(ca, "OP_EQ", None)
        OP_LE = getattr(ca, "OP_LE", None)
        OP_GE = getattr(ca, "OP_GE", None)

        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and OP_EQ is not None
            and isinstance(OP_EQ, int)
        ):
            if expr.is_op(OP_EQ):
                lhs = expr.dep(0)
                rhs = expr.dep(1)
                return Constraint(val=lhs - rhs, equals=0.0)

            elif OP_LE is not None and isinstance(OP_LE, int) and expr.is_op(OP_LE):
                lhs = expr.dep(0)
                rhs = expr.dep(1)
                return Constraint(val=lhs - rhs, max_val=0.0)

            elif OP_GE is not None and isinstance(OP_GE, int) and expr.is_op(OP_GE):
                lhs = expr.dep(0)
                rhs = expr.dep(1)
                return Constraint(val=lhs - rhs, min_val=0.0)

    except (AttributeError, TypeError, NotImplementedError):
        pass

    return Constraint(val=expr, equals=0.0)


def _boundary_constraint_to_constraints(
    boundary_constraint: _BoundaryConstraint, variable_expression: ca.MX
) -> list[Constraint]:
    """SINGLE SOURCE for converting boundary constraint to list of Constraint objects."""
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


def _process_state_boundary_constraints(
    state_boundary_constraints: list[_BoundaryConstraint | None],
    states_vec: ca.MX,
    result: list[Constraint],
) -> None:
    for i, boundary_constraint in enumerate(state_boundary_constraints):
        if boundary_constraint is None or not boundary_constraint.has_constraint():
            continue

        result.extend(_boundary_constraint_to_constraints(boundary_constraint, states_vec[i]))


def _process_control_boundary_constraints(
    control_boundary_constraints: list[_BoundaryConstraint | None],
    controls_vec: ca.MX,
    result: list[Constraint],
) -> None:
    for i, boundary_constraint in enumerate(control_boundary_constraints):
        if boundary_constraint is None or not boundary_constraint.has_constraint():
            continue

        result.extend(_boundary_constraint_to_constraints(boundary_constraint, controls_vec[i]))


def _process_symbolic_path_constraints(
    path_constraints: list[ca.MX],
    subs_map: dict[ca.MX, ca.MX],
    result: list[Constraint],
) -> None:
    for expr in path_constraints:
        substituted_expr = ca.substitute([expr], list(subs_map.keys()), list(subs_map.values()))[0]
        result.append(_symbolic_constraint_to_constraint(substituted_expr))


def _build_substitution_map(
    phase_def: PhaseDefinition,
    states_vec: ca.MX,
    controls_vec: ca.MX,
    time: ca.MX,
    static_parameters_vec: ca.MX | None,
    static_parameter_symbols: list[ca.MX] | None,
    initial_time_variable: ca.MX | None,
    terminal_time_variable: ca.MX | None,
) -> dict[ca.MX, ca.MX]:
    subs_map = {}

    state_syms = phase_def._get_ordered_state_symbols()
    for i, state_sym in enumerate(state_syms):
        subs_map[state_sym] = states_vec[i]

    control_syms = phase_def._get_ordered_control_symbols()
    for i, control_sym in enumerate(control_syms):
        subs_map[control_sym] = controls_vec[i]

    if phase_def.sym_time is not None:
        subs_map[phase_def.sym_time] = time

    if phase_def.sym_time_initial is not None and initial_time_variable is not None:
        subs_map[phase_def.sym_time_initial] = initial_time_variable

    if phase_def.sym_time_final is not None and terminal_time_variable is not None:
        subs_map[phase_def.sym_time_final] = terminal_time_variable

    if static_parameters_vec is not None and static_parameter_symbols is not None:
        for i, param_sym in enumerate(static_parameter_symbols):
            if len(static_parameter_symbols) == 1:
                subs_map[param_sym] = static_parameters_vec
            else:
                subs_map[param_sym] = static_parameters_vec[i]

    return subs_map


def _get_phase_path_constraints_function(
    phase_def: PhaseDefinition,
) -> Callable[..., list[Constraint]] | None:
    """Get path constraints function for a specific phase."""
    has_path_constraints = bool(phase_def.path_constraints)

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

    if not has_path_constraints and not has_state_boundary and not has_control_boundary:
        return None

    def vectorized_path_constraints(
        states_vec: ca.MX,
        controls_vec: ca.MX,
        time: ca.MX,
        static_parameters_vec: ca.MX | None = None,
        static_parameter_symbols: list[ca.MX] | None = None,
        initial_time_variable: ca.MX | None = None,
        terminal_time_variable: ca.MX | None = None,
    ) -> list[Constraint]:
        result: list[Constraint] = []

        subs_map = _build_substitution_map(
            phase_def,
            states_vec,
            controls_vec,
            time,
            static_parameters_vec,
            static_parameter_symbols,
            initial_time_variable,
            terminal_time_variable,
        )

        _process_symbolic_path_constraints(phase_def.path_constraints, subs_map, result)
        _process_state_boundary_constraints(state_boundary_constraints, states_vec, result)
        _process_control_boundary_constraints(control_boundary_constraints, controls_vec, result)

        return result

    return vectorized_path_constraints


def _process_cross_phase_substitution_map(
    multiphase_state: MultiPhaseVariableState,
    phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]],
    static_parameters_vec: ca.MX | None,
) -> dict[ca.MX, ca.MX]:
    subs_map = {}

    for phase_id, phase_def in multiphase_state.phases.items():
        if phase_id not in phase_endpoint_vectors:
            continue

        endpoint_data = phase_endpoint_vectors[phase_id]

        if phase_def.sym_time_initial is not None:
            subs_map[phase_def.sym_time_initial] = endpoint_data["t0"]
        if phase_def.sym_time_final is not None:
            subs_map[phase_def.sym_time_final] = endpoint_data["tf"]
        if phase_def.sym_time is not None:
            subs_map[phase_def.sym_time] = endpoint_data["tf"]

        state_initial_syms = phase_def.get_ordered_state_initial_symbols()
        state_final_syms = phase_def.get_ordered_state_final_symbols()
        state_syms = phase_def._get_ordered_state_symbols()

        x0_vec = endpoint_data["x0"]
        xf_vec = endpoint_data["xf"]

        for i, (sym_initial, sym_final, sym_current) in enumerate(
            zip(state_initial_syms, state_final_syms, state_syms, strict=True)
        ):
            if len(state_syms) == 1:
                subs_map[sym_initial] = x0_vec
                subs_map[sym_final] = xf_vec
                subs_map[sym_current] = xf_vec
            else:
                subs_map[sym_initial] = x0_vec[i]
                subs_map[sym_final] = xf_vec[i]
                subs_map[sym_current] = xf_vec[i]

        if "q" in endpoint_data and endpoint_data["q"] is not None:
            for i, integral_sym in enumerate(phase_def.integral_symbols):
                if i < endpoint_data["q"].shape[0]:
                    subs_map[integral_sym] = endpoint_data["q"][i]

    if static_parameters_vec is not None:
        static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
        for i, param_sym in enumerate(static_param_syms):
            if len(static_param_syms) == 1:
                subs_map[param_sym] = static_parameters_vec
            else:
                subs_map[param_sym] = static_parameters_vec[i]

    return subs_map


def _process_cross_phase_symbolic_constraints(
    multiphase_state: MultiPhaseVariableState,
    subs_map: dict[ca.MX, ca.MX],
    result: list[Constraint],
) -> None:
    for expr in multiphase_state.cross_phase_constraints:
        substituted_expr = ca.substitute([expr], list(subs_map.keys()), list(subs_map.values()))[0]
        constraint = _symbolic_constraint_to_constraint(substituted_expr)
        result.append(constraint)


def _process_phase_initial_boundary_constraints(
    multiphase_state: MultiPhaseVariableState,
    phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]],
    result: list[Constraint],
) -> None:
    for phase_id, phase_def in multiphase_state.phases.items():
        if phase_id not in phase_endpoint_vectors:
            continue

        endpoint_data = phase_endpoint_vectors[phase_id]
        x0_vec = endpoint_data["x0"]

        state_initial_constraints: list[_BoundaryConstraint | None] = [
            info.initial_constraint for info in phase_def.state_info
        ]

        for i, boundary_constraint in enumerate(state_initial_constraints):
            if (
                boundary_constraint is None
                or not boundary_constraint.has_constraint()
                or boundary_constraint.is_symbolic()
            ):
                continue

            if len(phase_def.state_info) == 1:
                result.extend(_boundary_constraint_to_constraints(boundary_constraint, x0_vec))
            else:
                result.extend(_boundary_constraint_to_constraints(boundary_constraint, x0_vec[i]))


def _process_phase_final_boundary_constraints(
    multiphase_state: MultiPhaseVariableState,
    phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]],
    result: list[Constraint],
) -> None:
    for phase_id, phase_def in multiphase_state.phases.items():
        if phase_id not in phase_endpoint_vectors:
            continue

        endpoint_data = phase_endpoint_vectors[phase_id]
        xf_vec = endpoint_data["xf"]

        state_final_constraints: list[_BoundaryConstraint | None] = [
            info.final_constraint for info in phase_def.state_info
        ]

        for i, boundary_constraint in enumerate(state_final_constraints):
            if (
                boundary_constraint is None
                or not boundary_constraint.has_constraint()
                or boundary_constraint.is_symbolic()
            ):
                continue

            if len(phase_def.state_info) == 1:
                result.extend(_boundary_constraint_to_constraints(boundary_constraint, xf_vec))
            else:
                result.extend(_boundary_constraint_to_constraints(boundary_constraint, xf_vec[i]))


def _get_cross_phase_event_constraints_function(
    multiphase_state: MultiPhaseVariableState,
) -> Callable[..., list[Constraint]] | None:
    """Get cross-phase event constraints function for multiphase problems."""
    has_cross_phase_constraints = bool(multiphase_state.cross_phase_constraints)

    has_phase_event_constraints = False
    for phase_def in multiphase_state.phases.values():
        state_initial_constraints = [info.initial_constraint for info in phase_def.state_info]
        state_final_constraints = [info.final_constraint for info in phase_def.state_info]

        # Only count NON-SYMBOLIC constraints (symbolic ones already processed)
        if any(
            constraint is not None and constraint.has_constraint() and not constraint.is_symbolic()
            for constraint in (state_initial_constraints + state_final_constraints)
        ):
            has_phase_event_constraints = True
            break

    if not has_cross_phase_constraints and not has_phase_event_constraints:
        return None

    def vectorized_cross_phase_event_constraints(
        phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]], static_parameters_vec: ca.MX | None
    ) -> list[Constraint]:
        result: list[Constraint] = []

        subs_map = _process_cross_phase_substitution_map(
            multiphase_state, phase_endpoint_vectors, static_parameters_vec
        )

        _process_cross_phase_symbolic_constraints(multiphase_state, subs_map, result)
        _process_phase_initial_boundary_constraints(
            multiphase_state, phase_endpoint_vectors, result
        )
        _process_phase_final_boundary_constraints(multiphase_state, phase_endpoint_vectors, result)

        return result

    return vectorized_cross_phase_event_constraints
