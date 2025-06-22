from collections.abc import Callable

import casadi as ca

from ..birkhoff import BirkhoffBasisComponents
from ..mtor_types import Constraint, PhaseID, ProblemProtocol


def _apply_constraint(opti: ca.Opti, constraint: Constraint) -> None:
    if constraint.min_val is not None:
        opti.subject_to(constraint.val >= constraint.min_val)
    if constraint.max_val is not None:
        opti.subject_to(constraint.val <= constraint.max_val)
    if constraint.equals is not None:
        opti.subject_to(constraint.val == constraint.equals)


def _apply_birkhoff_collocation_constraints(
    opti: ca.Opti,
    phase_id: PhaseID,
    state_at_grid_points: ca.MX,
    virtual_variables: list[ca.MX],
    control_variables: list[ca.MX],
    basis_components: BirkhoffBasisComponents,
    initial_state_variable: ca.MX,
    dynamics_function: Callable[..., ca.MX],
    problem: ProblemProtocol | None = None,
    static_parameters_vec: ca.MX | None = None,
) -> None:
    from ..input_validation import _validate_dynamics_output

    num_grid_points = len(basis_components.grid_points)
    num_states = state_at_grid_points.shape[0]

    birkhoff_matrix_a = basis_components.birkhoff_matrix_a

    # Birkhoff a-form constraint: X = x^a * bb + B^a * V
    # where X are states at grid points, V are virtual variables (derivatives)
    bb = ca.DM.ones(num_grid_points, 1)

    # Stack virtual variables into matrix form
    virtual_matrix = ca.horzcat(*virtual_variables)

    # Apply Birkhoff interpolation constraint
    expected_states = ca.repmat(initial_state_variable, 1, num_grid_points) + ca.mtimes(
        virtual_matrix, birkhoff_matrix_a.T
    )
    opti.subject_to(state_at_grid_points == expected_states)

    # Dynamics constraints: V_j = f(X_j, U_j) for each grid point j
    for j in range(num_grid_points):
        state_at_point = state_at_grid_points[:, j]
        control_at_point = control_variables[j]
        virtual_at_point = virtual_variables[j]

        dynamics_rhs = dynamics_function(
            state_at_point, control_at_point, basis_components.grid_points[j], static_parameters_vec
        )

        dynamics_rhs_vector = _validate_dynamics_output(dynamics_rhs, num_states)
        opti.subject_to(virtual_at_point == dynamics_rhs_vector)


def _apply_birkhoff_boundary_constraint(
    opti: ca.Opti,
    phase_id: PhaseID,
    initial_state_variable: ca.MX,
    final_state_variable: ca.MX,
    virtual_variables: list[ca.MX],
    basis_components: BirkhoffBasisComponents,
) -> None:
    # Birkhoff equivalence condition: x^b = x^a + w^B^T * V
    # where w^B are the Birkhoff quadrature weights
    quadrature_weights = ca.DM(basis_components.birkhoff_quadrature_weights)

    virtual_matrix = ca.horzcat(*virtual_variables)
    integral_term = ca.mtimes(virtual_matrix, quadrature_weights)

    opti.subject_to(final_state_variable == initial_state_variable + integral_term)


def _apply_birkhoff_path_constraints(
    opti: ca.Opti,
    phase_id: PhaseID,
    state_at_grid_points: ca.MX,
    control_variables: list[ca.MX],
    basis_components: BirkhoffBasisComponents,
    path_constraints_function: Callable[..., list[Constraint]],
    problem: ProblemProtocol | None = None,
    static_parameters_vec: ca.MX | None = None,
    static_parameter_symbols: list[ca.MX] | None = None,
    initial_time_variable: ca.MX | None = None,
    terminal_time_variable: ca.MX | None = None,
) -> None:
    num_grid_points = len(basis_components.grid_points)

    for j in range(num_grid_points):
        state_at_point = state_at_grid_points[:, j]
        control_at_point = control_variables[j]
        time_at_point = basis_components.grid_points[j]

        path_constraints_result: list[Constraint] | Constraint = path_constraints_function(
            state_at_point,
            control_at_point,
            time_at_point,
            static_parameters_vec,
            static_parameter_symbols,
            initial_time_variable,
            terminal_time_variable,
        )

        constraints_to_apply = (
            path_constraints_result
            if isinstance(path_constraints_result, list)
            else [path_constraints_result]
        )

        for constraint in constraints_to_apply:
            _apply_constraint(opti, constraint)


def _apply_birkhoff_multiphase_cross_phase_event_constraints(
    opti: ca.Opti,
    phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]],
    static_parameters: ca.MX | None,
    problem: ProblemProtocol,
) -> None:
    cross_phase_constraints_function = problem._get_cross_phase_event_constraints_function()
    if cross_phase_constraints_function is None:
        return

    cross_phase_constraints_result: list[Constraint] | Constraint = (
        cross_phase_constraints_function(phase_endpoint_data, static_parameters)
    )

    constraints_to_apply = (
        cross_phase_constraints_result
        if isinstance(cross_phase_constraints_result, list)
        else [cross_phase_constraints_result]
    )

    for constraint in constraints_to_apply:
        _apply_constraint(opti, constraint)
