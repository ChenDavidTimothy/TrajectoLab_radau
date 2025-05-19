"""
Integral setup functions for the direct solver.
"""

import casadi as ca

from ..radau import RadauBasisComponents
from ..tl_types import (
    CasadiMatrix,
    CasadiMX,
    CasadiOpti,
    FloatArray,
    IntegralIntegrandCallable,
    ProblemParameters,
)


def setup_integrals(
    opti: CasadiOpti,
    mesh_interval_index: int,
    state_at_nodes: CasadiMatrix,
    control_variables: CasadiMX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: CasadiMX,
    terminal_time_variable: CasadiMX,
    integral_integrand_function: IntegralIntegrandCallable,
    problem_parameters: ProblemParameters,
    num_integrals: int,
    accumulated_integral_expressions: list[CasadiMX],
) -> None:
    """
    Set up integral calculations for a single mesh interval using quadrature.

    Args:
        opti: CasADi optimization object
        mesh_interval_index: Index of mesh interval
        state_at_nodes: State variables at all nodes in interval
        control_variables: Control variables for interval
        basis_components: Radau basis components
        global_normalized_mesh_nodes: Global mesh nodes
        initial_time_variable: Initial time variable
        terminal_time_variable: Terminal time variable
        integral_integrand_function: Integrand function
        problem_parameters: Problem parameters
        num_integrals: Number of integrals
        accumulated_integral_expressions: List to accumulate integral expressions

    Note:
        This function modifies accumulated_integral_expressions in place.
    """
    num_colloc_nodes = len(basis_components.collocation_nodes)
    colloc_nodes_tau = basis_components.collocation_nodes.flatten()
    quad_weights = basis_components.quadrature_weights.flatten()

    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    tau_to_time_scaling: CasadiMX = (
        (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
    )

    for integral_index in range(num_integrals):
        quad_sum: CasadiMX = ca.MX(0)

        for i_colloc in range(num_colloc_nodes):
            state_at_colloc: CasadiMX = state_at_nodes[:, i_colloc]
            control_at_colloc: CasadiMX = control_variables[:, i_colloc]

            # Calculate physical time at this collocation point
            local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
            global_colloc_tau_val: CasadiMX = (
                global_segment_length / 2 * local_colloc_tau_val
                + (
                    global_normalized_mesh_nodes[mesh_interval_index + 1]
                    + global_normalized_mesh_nodes[mesh_interval_index]
                )
                / 2
            )
            physical_time_at_colloc: CasadiMX = (
                terminal_time_variable - initial_time_variable
            ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

            # Calculate integrand and add to quadrature sum
            weight: float = quad_weights[i_colloc]
            integrand_value: CasadiMX = integral_integrand_function(
                state_at_colloc,
                control_at_colloc,
                physical_time_at_colloc,
                integral_index,
                problem_parameters,
            )
            quad_sum += weight * integrand_value

        accumulated_integral_expressions[integral_index] += tau_to_time_scaling * quad_sum


def apply_integral_constraints(
    opti: CasadiOpti,
    integral_variables: CasadiMX,
    accumulated_integral_expressions: list[CasadiMX],
    num_integrals: int,
) -> None:
    """
    Apply integral constraints to the optimization problem.

    Args:
        opti: CasADi optimization object
        integral_variables: Integral decision variables
        accumulated_integral_expressions: Accumulated integral expressions
        num_integrals: Number of integrals

    Note:
        Constrains integral variables to equal the accumulated quadrature sums.
    """
    if num_integrals == 1:
        opti.subject_to(integral_variables == accumulated_integral_expressions[0])
    else:
        for i in range(num_integrals):
            opti.subject_to(integral_variables[i] == accumulated_integral_expressions[i])
