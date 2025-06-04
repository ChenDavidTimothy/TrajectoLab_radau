from __future__ import annotations

from collections.abc import Callable

import casadi as ca

from ..tl_types import Constraint, PhaseID
from .state import MultiPhaseVariableState, PhaseDefinition, _BoundaryConstraint


def add_path_constraint(phase_def: PhaseDefinition, constraint_expr: ca.MX | float | int) -> None:
    """
    Add a path constraint expression to a specific phase.

    Path constraints are applied at every collocation point throughout the phase.
    Leverages existing path constraint processing system.

    Args:
        phase_def: Phase definition to add constraint to
        constraint_expr: Constraint expression (single or multi-variable)
    """
    if isinstance(constraint_expr, ca.MX):
        phase_def.path_constraints.append(constraint_expr)
    else:
        phase_def.path_constraints.append(ca.MX(constraint_expr))


def add_event_constraint(
    multiphase_state: MultiPhaseVariableState, constraint_expr: ca.MX | float | int
) -> None:
    """
    Add an event constraint expression to the multiphase problem.

    Event constraints involving .initial and .final properties are automatically
    processed by the existing symbolic constraint system in validate_multiphase_configuration().
    Supports single-variable and multi-variable boundary constraints.

    Args:
        multiphase_state: Multiphase state to add constraint to
        constraint_expr: Constraint expression involving boundary conditions
    """
    if isinstance(constraint_expr, ca.MX):
        multiphase_state.cross_phase_constraints.append(constraint_expr)
    else:
        multiphase_state.cross_phase_constraints.append(ca.MX(constraint_expr))


def add_cross_phase_constraint(
    multiphase_state: MultiPhaseVariableState, constraint_expr: ca.MX | float | int
) -> None:
    """Add a cross-phase constraint expression."""
    if isinstance(constraint_expr, ca.MX):
        multiphase_state.cross_phase_constraints.append(constraint_expr)
    else:
        multiphase_state.cross_phase_constraints.append(ca.MX(constraint_expr))


def _symbolic_constraint_to_constraint(expr: ca.MX) -> Constraint:
    """SINGLE SOURCE for converting symbolic constraint to unified Constraint."""
    try:
        # Attempt to handle CasADi operation checking for constraints
        OP_EQ = getattr(ca, "OP_EQ", None)
        OP_LE = getattr(ca, "OP_LE", None)
        OP_GE = getattr(ca, "OP_GE", None)

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
    """Process state boundary constraints to reduce nesting."""
    for i, boundary_constraint in enumerate(state_boundary_constraints):
        # Early continue for empty constraints
        if boundary_constraint is None or not boundary_constraint.has_constraint():
            continue

        result.extend(_boundary_constraint_to_constraints(boundary_constraint, states_vec[i]))


def _process_control_boundary_constraints(
    control_boundary_constraints: list[_BoundaryConstraint | None],
    controls_vec: ca.MX,
    result: list[Constraint],
) -> None:
    """Process control boundary constraints to reduce nesting."""
    for i, boundary_constraint in enumerate(control_boundary_constraints):
        # Early continue for empty constraints
        if boundary_constraint is None or not boundary_constraint.has_constraint():
            continue

        result.extend(_boundary_constraint_to_constraints(boundary_constraint, controls_vec[i]))


def _process_symbolic_path_constraints(
    path_constraints: list[ca.MX],
    subs_map: dict[ca.MX, ca.MX],
    result: list[Constraint],
) -> None:
    """Process symbolic path constraints to reduce nesting."""
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
    """Build complete substitution map for path constraints."""
    subs_map = {}

    # Map state symbols to current state values
    state_syms = phase_def.get_ordered_state_symbols()
    for i, state_sym in enumerate(state_syms):
        subs_map[state_sym] = states_vec[i]

    # Map control symbols to current control values
    control_syms = phase_def.get_ordered_control_symbols()
    for i, control_sym in enumerate(control_syms):
        subs_map[control_sym] = controls_vec[i]

    # Map time symbols to current values
    if phase_def.sym_time is not None:
        subs_map[phase_def.sym_time] = time

    # Map phase initial and final time symbols
    if phase_def.sym_time_initial is not None and initial_time_variable is not None:
        subs_map[phase_def.sym_time_initial] = initial_time_variable

    if phase_def.sym_time_final is not None and terminal_time_variable is not None:
        subs_map[phase_def.sym_time_final] = terminal_time_variable

    # Map static parameter symbols to current parameter values
    if static_parameters_vec is not None and static_parameter_symbols is not None:
        for i, param_sym in enumerate(static_parameter_symbols):
            if len(static_parameter_symbols) == 1:
                subs_map[param_sym] = static_parameters_vec
            else:
                subs_map[param_sym] = static_parameters_vec[i]

    return subs_map


def get_phase_path_constraints_function(
    phase_def: PhaseDefinition,
) -> Callable[..., list[Constraint]] | None:
    """
    Get path constraints function for a specific phase.

    Processes both explicit path constraints (from path_constraints()) and
    boundary constraints (from state(boundary=...)).
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

    # Early return if no constraints exist
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
        """Apply path constraints at a single collocation point."""
        result: list[Constraint] = []

        # Build substitution map for symbolic constraints
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

        # Process explicit path constraints (from path_constraints())
        _process_symbolic_path_constraints(phase_def.path_constraints, subs_map, result)

        # Process boundary constraints (from state(boundary=...))
        _process_state_boundary_constraints(state_boundary_constraints, states_vec, result)
        _process_control_boundary_constraints(control_boundary_constraints, controls_vec, result)

        return result

    return vectorized_path_constraints


def _process_cross_phase_substitution_map(
    multiphase_state: MultiPhaseVariableState,
    phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]],
    static_parameters_vec: ca.MX | None,
) -> dict[ca.MX, ca.MX]:
    """Build substitution map for cross-phase constraints."""
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
            subs_map[phase_def.sym_time] = endpoint_data["tf"]

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
                subs_map[sym_current] = xf_vec
            else:
                subs_map[sym_initial] = x0_vec[i]
                subs_map[sym_final] = xf_vec[i]
                subs_map[sym_current] = xf_vec[i]

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

    return subs_map


def _process_cross_phase_symbolic_constraints(
    multiphase_state: MultiPhaseVariableState,
    subs_map: dict[ca.MX, ca.MX],
    result: list[Constraint],
) -> None:
    """Process cross-phase symbolic constraints."""
    for expr in multiphase_state.cross_phase_constraints:
        substituted_expr = ca.substitute([expr], list(subs_map.keys()), list(subs_map.values()))[0]
        constraint = _symbolic_constraint_to_constraint(substituted_expr)
        result.append(constraint)


def _process_phase_initial_boundary_constraints(
    multiphase_state: MultiPhaseVariableState,
    phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]],
    result: list[Constraint],
) -> None:
    """Process phase initial boundary constraints."""
    for phase_id, phase_def in multiphase_state.phases.items():
        if phase_id not in phase_endpoint_vectors:
            continue

        endpoint_data = phase_endpoint_vectors[phase_id]
        x0_vec = endpoint_data["x0"]

        state_initial_constraints: list[_BoundaryConstraint | None] = [
            info.initial_constraint for info in phase_def.state_info
        ]

        for i, boundary_constraint in enumerate(state_initial_constraints):
            # Early continue for empty constraints
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
    """Process phase final boundary constraints."""
    for phase_id, phase_def in multiphase_state.phases.items():
        if phase_id not in phase_endpoint_vectors:
            continue

        endpoint_data = phase_endpoint_vectors[phase_id]
        xf_vec = endpoint_data["xf"]

        state_final_constraints: list[_BoundaryConstraint | None] = [
            info.final_constraint for info in phase_def.state_info
        ]

        for i, boundary_constraint in enumerate(state_final_constraints):
            # Early continue for empty constraints
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


def get_cross_phase_event_constraints_function(
    multiphase_state: MultiPhaseVariableState,
) -> Callable[..., list[Constraint]] | None:
    """
    Get cross-phase event constraints function for multiphase problems.

    Processes both explicit event constraints (from event_constraints()) and
    boundary constraints (from state(initial=..., final=...)).
    """
    # Check for cross-phase constraints (includes event constraints)
    has_cross_phase_constraints = bool(multiphase_state.cross_phase_constraints)

    # Check for phase initial/final constraints - SKIP SYMBOLIC ONES
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

    # Early return if no event constraints exist
    if not has_cross_phase_constraints and not has_phase_event_constraints:
        return None

    def vectorized_cross_phase_event_constraints(
        phase_endpoint_vectors: dict[PhaseID, dict[str, ca.MX]], static_parameters_vec: ca.MX | None
    ) -> list[Constraint]:
        """Apply cross-phase event constraints."""
        result: list[Constraint] = []

        # Build substitution map for cross-phase constraints
        subs_map = _process_cross_phase_substitution_map(
            multiphase_state, phase_endpoint_vectors, static_parameters_vec
        )

        # Process cross-phase constraints (includes event constraints)
        _process_cross_phase_symbolic_constraints(multiphase_state, subs_map, result)

        # Process phase boundary constraints (from state(initial=..., final=...))
        _process_phase_initial_boundary_constraints(
            multiphase_state, phase_endpoint_vectors, result
        )
        _process_phase_final_boundary_constraints(multiphase_state, phase_endpoint_vectors, result)

        return result

    return vectorized_cross_phase_event_constraints
