from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import casadi as ca

from ..input_validation import set_integral_guess_values
from ..tl_types import FloatArray, PhaseID, ProblemProtocol
from .types_solver import _MultiPhaseVariable, _PhaseVariable


@dataclass
class _PhaseGuessContext:
    """context for phase guess application."""

    opti: ca.Opti
    phase_vars: _PhaseVariable
    phase_id: PhaseID
    num_states: int
    num_controls: int
    num_mesh_intervals: int
    num_integrals: int
    initial_guess: Any  # MultiPhaseInitialGuess


@dataclass
class _GuessApplicator:
    """Unified guess application strategy."""

    data_getter: Callable[[Any, PhaseID], Any]  # Gets data from initial_guess
    existence_checker: Callable[[Any, PhaseID], bool]  # Checks if data exists
    applicator: Callable[[_PhaseGuessContext, Any], None]  # Applies the guess


def _has_guess_data(guess_dict: dict | None, phase_id: PhaseID) -> bool:
    return guess_dict is not None and phase_id in guess_dict


def _get_guess_data(guess_dict: dict | None, phase_id: PhaseID) -> Any:
    if guess_dict is not None:
        return guess_dict.get(phase_id)
    return None


def _apply_time_initial_guess(context: _PhaseGuessContext, time_value: float) -> None:
    context.opti.set_initial(context.phase_vars.initial_time, time_value)


def _apply_time_terminal_guess(context: _PhaseGuessContext, time_value: float) -> None:
    context.opti.set_initial(context.phase_vars.terminal_time, time_value)


def _apply_state_guesses(context: _PhaseGuessContext, phase_states: list[FloatArray]) -> None:
    # Apply global mesh node states
    for k in range(context.num_mesh_intervals):
        state_guess_k = phase_states[k]

        # Set initial node guess (only once)
        if k == 0:
            context.opti.set_initial(context.phase_vars.state_at_mesh_nodes[0], state_guess_k[:, 0])

        # Set terminal node guess
        context.opti.set_initial(
            context.phase_vars.state_at_mesh_nodes[k + 1], state_guess_k[:, -1]
        )

    # Apply interior state node guesses
    for k in range(context.num_mesh_intervals):
        interior_var = context.phase_vars.interior_variables[k]
        if interior_var is not None:
            state_guess_k = phase_states[k]
            num_interior_nodes = interior_var.shape[1]

            # Extract interior guess
            interior_guess = state_guess_k[:, 1 : 1 + num_interior_nodes]
            context.opti.set_initial(interior_var, interior_guess)


def _apply_control_guesses(context: _PhaseGuessContext, phase_controls: list[FloatArray]) -> None:
    for k in range(context.num_mesh_intervals):
        control_guess_k = phase_controls[k]
        context.opti.set_initial(context.phase_vars.control_variables[k], control_guess_k)


def _apply_integral_guesses(
    context: _PhaseGuessContext, phase_integrals: float | FloatArray
) -> None:
    if context.num_integrals > 0 and context.phase_vars.integral_variables is not None:
        # Use centralized validation function - already validated, just set values
        set_integral_guess_values(
            context.opti,
            context.phase_vars.integral_variables,
            phase_integrals,
            context.num_integrals,
        )


def _create_guess_applicators() -> dict[str, _GuessApplicator]:
    return {
        "initial_times": _GuessApplicator(
            data_getter=lambda ig, phase_id: _get_guess_data(ig.phase_initial_times, phase_id),
            existence_checker=lambda ig, phase_id: _has_guess_data(
                ig.phase_initial_times, phase_id
            ),
            applicator=_apply_time_initial_guess,
        ),
        "terminal_times": _GuessApplicator(
            data_getter=lambda ig, phase_id: _get_guess_data(ig.phase_terminal_times, phase_id),
            existence_checker=lambda ig, phase_id: _has_guess_data(
                ig.phase_terminal_times, phase_id
            ),
            applicator=_apply_time_terminal_guess,
        ),
        "states": _GuessApplicator(
            data_getter=lambda ig, phase_id: _get_guess_data(ig.phase_states, phase_id),
            existence_checker=lambda ig, phase_id: _has_guess_data(ig.phase_states, phase_id),
            applicator=_apply_state_guesses,
        ),
        "controls": _GuessApplicator(
            data_getter=lambda ig, phase_id: _get_guess_data(ig.phase_controls, phase_id),
            existence_checker=lambda ig, phase_id: _has_guess_data(ig.phase_controls, phase_id),
            applicator=_apply_control_guesses,
        ),
        "integrals": _GuessApplicator(
            data_getter=lambda ig, phase_id: _get_guess_data(ig.phase_integrals, phase_id),
            existence_checker=lambda ig, phase_id: _has_guess_data(ig.phase_integrals, phase_id),
            applicator=_apply_integral_guesses,
        ),
    }


def _apply_single_guess_type(context: _PhaseGuessContext, applicator: _GuessApplicator) -> None:
    if applicator.existence_checker(context.initial_guess, context.phase_id):
        guess_data = applicator.data_getter(context.initial_guess, context.phase_id)
        applicator.applicator(context, guess_data)


def _create_phase_guess_context(
    opti: ca.Opti,
    phase_vars: _PhaseVariable,
    initial_guess: Any,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> _PhaseGuessContext:
    num_states, num_controls = problem._get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    return _PhaseGuessContext(
        opti=opti,
        phase_vars=phase_vars,
        phase_id=phase_id,
        num_states=num_states,
        num_controls=num_controls,
        num_mesh_intervals=num_mesh_intervals,
        num_integrals=num_integrals,
        initial_guess=initial_guess,
    )


def _apply_phase_guesses(
    opti: ca.Opti,
    phase_vars: _PhaseVariable,
    initial_guess: Any,
    problem: ProblemProtocol,
    phase_id: PhaseID,
) -> None:
    context = _create_phase_guess_context(opti, phase_vars, initial_guess, problem, phase_id)
    applicators = _create_guess_applicators()

    # Apply all guess types using flattened loop
    for applicator in applicators.values():
        _apply_single_guess_type(context, applicator)


def _apply_static_parameters_guess(
    opti: ca.Opti,
    variables: _MultiPhaseVariable,
    initial_guess: Any,
    problem: ProblemProtocol,
) -> None:
    if initial_guess.static_parameters is not None and variables.static_parameters is not None:
        _, _, num_static_params = problem._get_total_variable_counts()

        if num_static_params > 0:
            opti.set_initial(variables.static_parameters, initial_guess.static_parameters)


def _apply_multiphase_initial_guess(
    opti: ca.Opti,
    variables: _MultiPhaseVariable,
    problem: ProblemProtocol,
) -> None:
    if problem.initial_guess is None:
        return

    initial_guess = problem.initial_guess

    # Apply initial guess for each phase using flattened processing
    for phase_id in problem._get_phase_ids():
        if phase_id in variables.phase_variables:
            phase_vars = variables.phase_variables[phase_id]
            _apply_phase_guesses(opti, phase_vars, initial_guess, problem, phase_id)

    _apply_static_parameters_guess(opti, variables, initial_guess, problem)
