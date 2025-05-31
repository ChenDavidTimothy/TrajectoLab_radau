# trajectolab/direct_solver/integrals_solver.py
"""
Integral constraint setup using quadrature rules for multiphase direct solver.
"""

from collections.abc import Callable

import casadi as ca

from ..radau import RadauBasisComponents
from ..tl_types import (
    FloatArray,
    PhaseID,
)


def setup_phase_integrals(
    opti: ca.Opti,
    phase_id: PhaseID,
    mesh_interval_index: int,
    state_at_nodes: ca.MX,
    control_variables: ca.MX,
    basis_components: RadauBasisComponents,
    global_normalized_mesh_nodes: FloatArray,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
    integral_integrand_function: Callable[..., ca.MX],
    num_integrals: int,
    accumulated_integral_expressions: list[ca.MX],
    static_parameters_vec: ca.MX | None = None,  # CRITICAL FIX: Add static parameters support
) -> None:
    """
    Set up integral calculations for a single mesh interval within a phase.

    Args:
        opti: CasADi optimization object
        phase_id: Phase identifier
        mesh_interval_index: Index of mesh interval within phase
        state_at_nodes: State variables at all nodes in interval
        control_variables: Control variables for interval
        basis_components: Radau basis components
        global_normalized_mesh_nodes: Global mesh nodes for this phase
        initial_time_variable: Initial time variable for this phase
        terminal_time_variable: Terminal time variable for this phase
        integral_integrand_function: Integrand function for this phase
        num_integrals: Number of integrals for this phase
        accumulated_integral_expressions: List to accumulate integral expressions
        static_parameters_vec: Static parameter variables (CRITICAL FIX: Added parameter)

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

    tau_to_time_scaling: ca.MX = (
        (terminal_time_variable - initial_time_variable) * global_segment_length / 4.0
    )

    for integral_index in range(num_integrals):
        quad_sum: ca.MX = ca.MX(0)

        for i_colloc in range(num_colloc_nodes):
            state_at_colloc: ca.MX = state_at_nodes[:, i_colloc]
            control_at_colloc: ca.MX = control_variables[:, i_colloc]

            # Calculate physical time at this collocation point
            local_colloc_tau_val: float = colloc_nodes_tau[i_colloc]
            global_colloc_tau_val: ca.MX = (
                global_segment_length / 2 * local_colloc_tau_val
                + (
                    global_normalized_mesh_nodes[mesh_interval_index + 1]
                    + global_normalized_mesh_nodes[mesh_interval_index]
                )
                / 2
            )
            physical_time_at_colloc: ca.MX = (
                terminal_time_variable - initial_time_variable
            ) / 2 * global_colloc_tau_val + (terminal_time_variable + initial_time_variable) / 2

            # Calculate integrand and add to quadrature sum
            weight: float = quad_weights[i_colloc]

            # CRITICAL FIX: Pass static parameters to integrand function
            integrand_value: ca.MX = integral_integrand_function(
                state_at_colloc,
                control_at_colloc,
                physical_time_at_colloc,
                integral_index,
                static_parameters_vec,  # CRITICAL FIX: Pass static parameters
            )
            quad_sum += weight * integrand_value

        accumulated_integral_expressions[integral_index] += tau_to_time_scaling * quad_sum


def apply_phase_integral_constraints(
    opti: ca.Opti,
    integral_variables: ca.MX,
    accumulated_integral_expressions: list[ca.MX],
    num_integrals: int,
    phase_id: PhaseID,
) -> None:
    """
    Apply integral constraints for a specific phase.

    Args:
        opti: CasADi optimization object
        integral_variables: Integral decision variables for this phase
        accumulated_integral_expressions: Accumulated integral expressions for this phase
        num_integrals: Number of integrals for this phase
        phase_id: Phase identifier for error messages

    Note:
        Constrains integral variables to equal the accumulated quadrature sums.
    """
    if num_integrals == 1:
        opti.subject_to(integral_variables == accumulated_integral_expressions[0])
    else:
        for i in range(num_integrals):
            opti.subject_to(integral_variables[i] == accumulated_integral_expressions[i])
