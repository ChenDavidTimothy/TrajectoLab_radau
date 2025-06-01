from typing import cast

import casadi as ca

from ..input_validation import set_integral_guess_values, validate_integral_values
from ..tl_types import FloatArray, PhaseID, ProblemProtocol
from .types_solver import MultiPhaseVariableReferences, PhaseVariableReferences


def apply_multiphase_initial_guess(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: ProblemProtocol,
) -> None:
    """
    Apply initial guess to multiphase optimization variables.

    NOTE: Assumes initial guess has been validated by validate_multiphase_initial_guess()
    at the entry point (validate_multiphase_configuration).
    """
    if problem.initial_guess is None:
        return

    ig = problem.initial_guess

    # Apply initial guess for each phase
    for phase_id in problem.get_phase_ids():
        if phase_id in variables.phase_variables:
            phase_vars = variables.phase_variables[phase_id]
            _apply_phase_initial_guess(opti, phase_vars, ig, problem, phase_id)

    # Apply static parameters guess
    _apply_static_parameters_guess(opti, variables, ig, problem)


def _apply_phase_initial_guess(
    opti: ca.Opti,
    phase_vars: PhaseVariableReferences,
    ig,  # MultiPhaseInitialGuess
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> None:
    """Apply initial guess for a single phase."""
    # Get phase information
    num_states, num_controls = problem.get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    # Apply time guesses for this phase
    _apply_phase_time_guesses(opti, phase_vars, ig, phase_id)

    # Apply state guesses for this phase
    _apply_phase_state_guesses(opti, phase_vars, ig, num_states, num_mesh_intervals, phase_id)

    # Apply control guesses for this phase
    _apply_phase_control_guesses(opti, phase_vars, ig, num_controls, num_mesh_intervals, phase_id)

    # Apply integral guesses for this phase
    _apply_phase_integral_guesses(opti, phase_vars, ig, num_integrals, phase_id)


def _apply_phase_time_guesses(
    opti: ca.Opti, phase_vars: PhaseVariableReferences, ig, phase_id: PhaseID
) -> None:
    """Apply initial guess for time variables of a specific phase."""
    if ig.phase_initial_times is not None and phase_id in ig.phase_initial_times:
        opti.set_initial(phase_vars.initial_time, ig.phase_initial_times[phase_id])

    if ig.phase_terminal_times is not None and phase_id in ig.phase_terminal_times:
        opti.set_initial(phase_vars.terminal_time, ig.phase_terminal_times[phase_id])


def _apply_phase_state_guesses(
    opti: ca.Opti,
    phase_vars: PhaseVariableReferences,
    ig,  # MultiPhaseInitialGuess
    num_states: int,
    num_mesh_intervals: int,
    phase_id: PhaseID,
) -> None:
    """Apply initial guess for state variables of a specific phase."""
    if ig.phase_states is None or phase_id not in ig.phase_states:
        return

    phase_states = ig.phase_states[phase_id]

    # Apply global mesh node states
    for k in range(num_mesh_intervals):
        state_guess_k = cast(FloatArray, phase_states[k])

        # Set initial node guess (only once)
        if k == 0:
            opti.set_initial(phase_vars.state_at_mesh_nodes[0], state_guess_k[:, 0])
        # Set terminal node guess
        opti.set_initial(phase_vars.state_at_mesh_nodes[k + 1], state_guess_k[:, -1])

    # Apply interior state node guesses
    for k in range(num_mesh_intervals):
        interior_var = phase_vars.interior_variables[k]
        if interior_var is not None:
            state_guess_k = cast(FloatArray, phase_states[k])
            num_interior_nodes = interior_var.shape[1]

            # Extract interior guess
            interior_guess = state_guess_k[:, 1 : 1 + num_interior_nodes]
            opti.set_initial(interior_var, interior_guess)


def _apply_phase_control_guesses(
    opti: ca.Opti,
    phase_vars: PhaseVariableReferences,
    ig,  # MultiPhaseInitialGuess
    num_controls: int,
    num_mesh_intervals: int,
    phase_id: PhaseID,
) -> None:
    """Apply initial guess for control variables of a specific phase."""
    if ig.phase_controls is None or phase_id not in ig.phase_controls:
        return

    phase_controls = ig.phase_controls[phase_id]

    for k in range(num_mesh_intervals):
        control_guess_k = cast(FloatArray, phase_controls[k])
        opti.set_initial(phase_vars.control_variables[k], control_guess_k)


def _apply_phase_integral_guesses(
    opti: ca.Opti,
    phase_vars: PhaseVariableReferences,
    ig,  # MultiPhaseInitialGuess
    num_integrals: int,
    phase_id: PhaseID,
) -> None:
    """Apply initial guess for integral variables of a specific phase."""
    if (
        ig.phase_integrals is not None
        and phase_id in ig.phase_integrals
        and num_integrals > 0
        and phase_vars.integral_variables is not None
    ):
        phase_integrals = ig.phase_integrals[phase_id]

        # Use centralized validation function
        validate_integral_values(phase_integrals, num_integrals)
        set_integral_guess_values(
            opti, phase_vars.integral_variables, phase_integrals, num_integrals
        )


def _apply_static_parameters_guess(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    ig,  # MultiPhaseInitialGuess
    problem: ProblemProtocol,
) -> None:
    """Apply initial guess for static parameters."""
    if ig.static_parameters is not None and variables.static_parameters is not None:
        total_states, total_controls, num_static_params = problem.get_total_variable_counts()

        if num_static_params > 0:
            opti.set_initial(variables.static_parameters, ig.static_parameters)
