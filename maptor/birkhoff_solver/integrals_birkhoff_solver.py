from collections.abc import Callable

import casadi as ca

from ..birkhoff import BirkhoffBasisComponents
from ..mtor_types import PhaseID


def _setup_birkhoff_phase_integrals(
    opti: ca.Opti,
    phase_id: PhaseID,
    state_at_grid_points: ca.MX,
    control_variables: list[ca.MX],
    basis_components: BirkhoffBasisComponents,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    integral_integrand_function: Callable[..., ca.MX],
    num_integrals: int,
    accumulated_integral_expressions: list[ca.MX],
    static_parameters_vec: ca.MX | None = None,
) -> None:
    num_grid_points = len(basis_components.grid_points)
    quad_weights = basis_components.birkhoff_quadrature_weights
    grid_points = basis_components.grid_points

    # Time scaling factor from computational domain to physical domain
    time_scaling = (terminal_time_variable - initial_time_variable) / 2.0

    for integral_index in range(num_integrals):
        quad_sum: ca.MX = ca.MX(0)

        for j in range(num_grid_points):
            state_at_point = state_at_grid_points[:, j]
            control_at_point = control_variables[j]

            # Transform from computational time τ ∈ [-1,1] to physical time t ∈ [t^a, t^b]
            physical_time = (terminal_time_variable - initial_time_variable) / 2.0 * grid_points[
                j
            ] + (terminal_time_variable + initial_time_variable) / 2.0

            weight = quad_weights[j]
            integrand_value = integral_integrand_function(
                state_at_point,
                control_at_point,
                physical_time,
                integral_index,
                static_parameters_vec,
            )
            quad_sum += weight * integrand_value

        accumulated_integral_expressions[integral_index] += time_scaling * quad_sum


def _apply_birkhoff_phase_integral_constraints(
    opti: ca.Opti,
    integral_variables: ca.MX,
    accumulated_integral_expressions: list[ca.MX],
    num_integrals: int,
    phase_id: PhaseID,
) -> None:
    if num_integrals == 1:
        opti.subject_to(integral_variables == accumulated_integral_expressions[0])
    else:
        for i in range(num_integrals):
            opti.subject_to(integral_variables[i] == accumulated_integral_expressions[i])
