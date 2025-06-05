from collections.abc import Callable
from dataclasses import dataclass

import casadi as ca

from ..tl_types import PhaseID, ProblemProtocol
from ..utils.constants import MINIMUM_TIME_INTERVAL
from .types_solver import (
    MultiPhaseVariable,
    PhaseVariable,
    _PhaseIntervalBundle,
)


@dataclass
class _BoundConstraint:
    """Unified bound constraint representation."""

    lower: float
    upper: float
    is_fixed: bool

    @classmethod
    def from_bounds(cls, bounds: tuple[float, float]) -> "_BoundConstraint":
        """Create constraint from bounds tuple."""
        lower, upper = bounds
        return cls(lower=lower, upper=upper, is_fixed=(lower == upper))


@dataclass
class _VariableCreationContext:
    """Consolidated context for variable creation operations."""

    opti: ca.Opti
    problem: ProblemProtocol
    phase_id: PhaseID
    num_states: int
    num_controls: int
    num_mesh_intervals: int
    num_integrals: int


@dataclass
class _VariableCreator:
    """Unified variable creation strategy."""

    creator: Callable[[_VariableCreationContext], ca.MX | list[ca.MX]]
    constraint_applier: (
        Callable[[ca.Opti, ca.MX | list[ca.MX], _VariableCreationContext], None] | None
    ) = None


def _apply_bound_constraints(opti: ca.Opti, variable: ca.MX, constraint: _BoundConstraint) -> None:
    """Apply bound constraints to a variable using unified pattern."""
    if constraint.is_fixed:
        opti.subject_to(variable == constraint.lower)
    else:
        if constraint.lower > -1e5:
            opti.subject_to(variable >= constraint.lower)
        if constraint.upper < 1e5:
            opti.subject_to(variable <= constraint.upper)


def _apply_time_constraints(
    opti: ca.Opti, time_variables: list[ca.MX], context: _VariableCreationContext
) -> None:
    """Apply time constraints for initial and terminal time variables."""
    initial_time_var, terminal_time_var = time_variables
    phase_def = context.problem._phases[context.phase_id]

    # Apply bound constraints
    t0_constraint = _BoundConstraint.from_bounds(phase_def.t0_bounds)
    tf_constraint = _BoundConstraint.from_bounds(phase_def.tf_bounds)

    _apply_bound_constraints(opti, initial_time_var, t0_constraint)
    _apply_bound_constraints(opti, terminal_time_var, tf_constraint)

    # Minimum interval prevents singular coordinate transformations
    opti.subject_to(terminal_time_var > initial_time_var + MINIMUM_TIME_INTERVAL)


def _create_time_variables(context: _VariableCreationContext) -> list[ca.MX]:
    """Create initial and terminal time variables for a phase."""
    initial_time_variable = context.opti.variable()
    terminal_time_variable = context.opti.variable()
    return [initial_time_variable, terminal_time_variable]


def _create_state_variables(context: _VariableCreationContext) -> list[ca.MX]:
    """Create global mesh node state variables for a phase."""
    state_variables = []
    for _ in range(context.num_mesh_intervals + 1):
        state_var = context.opti.variable(context.num_states)
        state_variables.append(state_var)
    return state_variables


def _create_control_variables(context: _VariableCreationContext) -> list[ca.MX]:
    """Create control variables for all mesh intervals in a phase."""
    phase_def = context.problem._phases[context.phase_id]
    control_variables = []

    for k in range(context.num_mesh_intervals):
        num_colloc_points = phase_def.collocation_points_per_interval[k]
        control_var = context.opti.variable(context.num_controls, num_colloc_points)
        control_variables.append(control_var)

    return control_variables


def _create_integral_variables(context: _VariableCreationContext) -> ca.MX | None:
    """Create integral variables for a phase."""
    if context.num_integrals > 0:
        return (
            context.opti.variable(context.num_integrals)
            if context.num_integrals > 1
            else context.opti.variable()
        )
    return None


def _create_static_parameter_variables(context: _VariableCreationContext) -> ca.MX:
    """Create static parameter variables shared across all phases."""
    _, _, num_static_params = context.problem._get_total_variable_counts()
    return (
        context.opti.variable(num_static_params)
        if num_static_params > 1
        else context.opti.variable()
    )


def _create_variable_creators() -> dict[str, _VariableCreator]:
    """Create unified variable creators for all variable types."""
    return {
        "time": _VariableCreator(
            creator=_create_time_variables,
            constraint_applier=_apply_time_constraints,
        ),
        "states": _VariableCreator(
            creator=_create_state_variables,
            constraint_applier=None,  # No constraints for global state variables
        ),
        "controls": _VariableCreator(
            creator=_create_control_variables,
            constraint_applier=None,  # No constraints for control variables
        ),
        "integrals": _VariableCreator(
            creator=_create_integral_variables,
            constraint_applier=None,  # No constraints for integral variables
        ),
    }


def _create_phase_variables_unified(
    context: _VariableCreationContext,
) -> dict[str, ca.MX | list[ca.MX]]:
    """Create all phase variables using unified creators."""
    creators = _create_variable_creators()
    variables = {}

    for var_type, creator in creators.items():
        # Create variables
        created_vars = creator.creator(context)
        variables[var_type] = created_vars

        # Apply constraints if needed
        if creator.constraint_applier is not None:
            creator.constraint_applier(context.opti, created_vars, context)

    return variables


def _create_variable_context(
    opti: ca.Opti, problem: ProblemProtocol, phase_id: PhaseID
) -> _VariableCreationContext:
    """Create consolidated context for variable creation."""
    num_states, num_controls = problem._get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    return _VariableCreationContext(
        opti=opti,
        problem=problem,
        phase_id=phase_id,
        num_states=num_states,
        num_controls=num_controls,
        num_mesh_intervals=num_mesh_intervals,
        num_integrals=num_integrals,
    )


def _setup_phase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> PhaseVariable:
    """Setup optimization variables for a single phase using unified approach."""
    context = _create_variable_context(opti, problem, phase_id)
    variables = _create_phase_variables_unified(context)

    # Extract variables with type safety
    time_vars = variables["time"]
    initial_time, terminal_time = time_vars
    state_at_mesh_nodes = variables["states"]
    control_variables = variables["controls"]
    integral_variables = variables["integrals"]

    return PhaseVariable(
        phase_id=phase_id,
        initial_time=initial_time,
        terminal_time=terminal_time,
        state_at_mesh_nodes=state_at_mesh_nodes,
        control_variables=control_variables,
        integral_variables=integral_variables,
    )


def _setup_multiphase_optimization_variables(
    opti: ca.Opti,
    problem: ProblemProtocol,
) -> MultiPhaseVariable:
    """Setup optimization variables for all phases using flattened approach."""
    multiphase_vars = MultiPhaseVariable()

    # Process all phases with flattened loop
    for phase_id in problem._get_phase_ids():
        phase_vars = _setup_phase_optimization_variables(opti, problem, phase_id)
        multiphase_vars.phase_variables[phase_id] = phase_vars

    # Create static parameters if needed
    _, _, num_static_params = problem._get_total_variable_counts()
    if num_static_params > 0:
        static_context = _VariableCreationContext(
            opti=opti,
            problem=problem,
            phase_id=0,  # phase_id not used for static params
            num_states=0,
            num_controls=0,
            num_mesh_intervals=0,
            num_integrals=0,
        )
        multiphase_vars.static_parameters = _create_static_parameter_variables(static_context)

    return multiphase_vars


@dataclass
class _InteriorNodeContext:
    """Consolidated context for interior node processing."""

    opti: ca.Opti
    phase_id: PhaseID
    num_states: int
    num_colloc_nodes: int
    state_at_global_mesh_nodes: list[ca.MX]
    mesh_interval_index: int


def _create_interior_variables(context: _InteriorNodeContext) -> ca.MX:
    """Create interior state variables for mesh interval."""
    num_interior_nodes = context.num_colloc_nodes - 1
    return context.opti.variable(context.num_states, num_interior_nodes)


def _populate_state_columns_with_interior(
    state_columns: list[ca.MX], interior_nodes_var: ca.MX
) -> None:
    """Populate state columns with interior node variables."""
    num_interior_nodes = interior_nodes_var.shape[1]
    for i in range(num_interior_nodes):
        state_columns[i + 1] = interior_nodes_var[:, i]


def _setup_interior_nodes(context: _InteriorNodeContext) -> tuple[list[ca.MX], ca.MX | None]:
    """Setup interior nodes for mesh interval with flattened logic."""
    # Early return for simple case
    if context.num_colloc_nodes <= 1:
        return [], None

    num_interior_nodes = context.num_colloc_nodes - 1
    if num_interior_nodes <= 0:
        return [], None

    # Create state columns structure
    state_columns = [ca.MX(context.num_states, 1) for _ in range(context.num_colloc_nodes + 1)]

    # Set boundary columns
    state_columns[0] = context.state_at_global_mesh_nodes[context.mesh_interval_index]
    state_columns[context.num_colloc_nodes] = context.state_at_global_mesh_nodes[
        context.mesh_interval_index + 1
    ]

    # Create and populate interior variables
    interior_nodes_var = _create_interior_variables(context)
    _populate_state_columns_with_interior(state_columns, interior_nodes_var)

    return state_columns, interior_nodes_var


def setup_phase_interval_state_variables(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    num_states: int,
    num_colloc_nodes: int,
    state_at_global_mesh_nodes: list[ca.MX],
) -> _PhaseIntervalBundle:
    """Setup state variables for single mesh interval with flattened structure."""
    context = _InteriorNodeContext(
        opti=opti,
        phase_id=phase_id,
        num_states=num_states,
        num_colloc_nodes=num_colloc_nodes,
        state_at_global_mesh_nodes=state_at_global_mesh_nodes,
        mesh_interval_index=mesh_interval_index,
    )

    state_columns, interior_nodes_var = _setup_interior_nodes(context)

    # Convert to matrix format
    if state_columns:
        state_matrix = ca.horzcat(*state_columns)
        state_matrix = ca.MX(state_matrix)
    else:
        # Handle simple case without interior nodes
        state_matrix = ca.horzcat(
            state_at_global_mesh_nodes[mesh_interval_index],
            state_at_global_mesh_nodes[mesh_interval_index + 1],
        )
        state_matrix = ca.MX(state_matrix)

    return state_matrix, interior_nodes_var
