import logging
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..input_validation import _set_integral_guess_values
from ..mtor_types import FloatArray, PhaseID, ProblemProtocol
from .types_birkhoff_solver import _BirkhoffMultiPhaseVariable, _BirkhoffPhaseVariable


logger = logging.getLogger(__name__)


@dataclass
class _BirkhoffPhaseGuessContext:
    opti: ca.Opti
    phase_vars: _BirkhoffPhaseVariable
    phase_id: PhaseID
    num_states: int
    num_controls: int
    num_grid_points: int
    num_integrals: int
    initial_guess: Any


def _apply_birkhoff_state_guesses(
    context: _BirkhoffPhaseGuessContext, phase_states: list[FloatArray]
) -> None:
    # Apply state guesses at mesh boundaries and grid points
    if len(phase_states) > 0:
        first_interval_states = phase_states[0]
        num_provided_states = first_interval_states.shape[0]

        if num_provided_states == context.num_states:
            # Set initial mesh node guess
            if len(context.phase_vars.state_at_mesh_nodes) > 0:
                context.opti.set_initial(
                    context.phase_vars.state_at_mesh_nodes[0], first_interval_states[:, 0]
                )

            # Set final mesh node guess if available
            if len(phase_states) > 0 and len(context.phase_vars.state_at_mesh_nodes) > 1:
                last_interval_states = phase_states[-1]
                context.opti.set_initial(
                    context.phase_vars.state_at_mesh_nodes[-1], last_interval_states[:, -1]
                )
        else:
            logger.info(
                f"Phase {context.phase_id} state guess has {num_provided_states} states, "
                f"expected {context.num_states}. Using CasADi defaults for missing states."
            )


def _apply_birkhoff_control_guesses(
    context: _BirkhoffPhaseGuessContext, phase_controls: list[FloatArray]
) -> None:
    # Apply control guesses at grid points
    if (
        len(phase_controls) > 0
        and len(context.phase_vars.control_variables) == context.num_grid_points
    ):
        first_interval_controls = phase_controls[0]
        num_provided_controls = first_interval_controls.shape[0]

        if num_provided_controls == context.num_controls:
            # Distribute control values across grid points
            for j, control_var in enumerate(context.phase_vars.control_variables):
                if j < first_interval_controls.shape[1]:
                    context.opti.set_initial(control_var, first_interval_controls[:, j])
        else:
            logger.info(
                f"Phase {context.phase_id} control guess has {num_provided_controls} controls, "
                f"expected {context.num_controls}. Using CasADi defaults for missing controls."
            )


def _apply_birkhoff_virtual_guesses(
    context: _BirkhoffPhaseGuessContext, virtual_guess: FloatArray | None
) -> None:
    # Apply virtual variable (derivative) guesses
    if (
        virtual_guess is not None
        and len(context.phase_vars.virtual_variables) == context.num_grid_points
    ):
        if virtual_guess.shape[0] == context.num_states:
            for j, virtual_var in enumerate(context.phase_vars.virtual_variables):
                if j < virtual_guess.shape[1]:
                    context.opti.set_initial(virtual_var, virtual_guess[:, j])
        else:
            logger.info(
                f"Phase {context.phase_id} virtual guess has {virtual_guess.shape[0]} states, "
                f"expected {context.num_states}. Using CasADi defaults."
            )


def _apply_birkhoff_integral_guesses(
    context: _BirkhoffPhaseGuessContext, phase_integrals: float | FloatArray
) -> None:
    if context.num_integrals > 0 and context.phase_vars.integral_variables is not None:
        _set_integral_guess_values(
            context.opti,
            context.phase_vars.integral_variables,
            phase_integrals,
            context.num_integrals,
        )


def _apply_birkhoff_phase_guesses_from_phase_definition(
    opti: ca.Opti,
    phase_vars: _BirkhoffPhaseVariable,
    phase_def: Any,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    grid_points: tuple[float, ...],
) -> None:
    num_states, num_controls = problem._get_phase_variable_counts(phase_id)
    num_grid_points = len(grid_points)
    num_integrals = phase_def.num_integrals

    # Apply time guesses
    if phase_def.guess_initial_time is not None:
        opti.set_initial(phase_vars.initial_time, phase_def.guess_initial_time)

    if phase_def.guess_terminal_time is not None:
        opti.set_initial(phase_vars.terminal_time, phase_def.guess_terminal_time)

    # Apply state guesses
    if phase_def.guess_states is not None:
        context = _BirkhoffPhaseGuessContext(
            opti=opti,
            phase_vars=phase_vars,
            phase_id=phase_id,
            num_states=num_states,
            num_controls=num_controls,
            num_grid_points=num_grid_points,
            num_integrals=num_integrals,
            initial_guess=None,
        )
        _apply_birkhoff_state_guesses(context, phase_def.guess_states)

    # Apply control guesses
    if phase_def.guess_controls is not None:
        context = _BirkhoffPhaseGuessContext(
            opti=opti,
            phase_vars=phase_vars,
            phase_id=phase_id,
            num_states=num_states,
            num_controls=num_controls,
            num_grid_points=num_grid_points,
            num_integrals=num_integrals,
            initial_guess=None,
        )
        _apply_birkhoff_control_guesses(context, phase_def.guess_controls)

    # Apply integral guesses
    if phase_def.guess_integrals is not None:
        context = _BirkhoffPhaseGuessContext(
            opti=opti,
            phase_vars=phase_vars,
            phase_id=phase_id,
            num_states=num_states,
            num_controls=num_controls,
            num_grid_points=num_grid_points,
            num_integrals=num_integrals,
            initial_guess=None,
        )
        _apply_birkhoff_integral_guesses(context, phase_def.guess_integrals)


def _apply_birkhoff_multiphase_initial_guess(
    opti: ca.Opti,
    variables: _BirkhoffMultiPhaseVariable,
    problem: ProblemProtocol,
    grid_points_per_phase: dict[PhaseID, tuple[float, ...]],
) -> None:
    # Apply phase-level guesses
    for phase_id in problem._get_phase_ids():
        if phase_id in variables.phase_variables and phase_id in grid_points_per_phase:
            phase_vars = variables.phase_variables[phase_id]
            phase_def = problem._phases[phase_id]
            grid_points = grid_points_per_phase[phase_id]
            _apply_birkhoff_phase_guesses_from_phase_definition(
                opti, phase_vars, phase_def, problem, phase_id, grid_points
            )

    # Apply static parameter guesses
    if (
        problem._multiphase_state.guess_static_parameters is not None
        and variables.static_parameters is not None
    ):
        param_names = problem._static_parameters.parameter_names
        param_guesses_dict = problem._multiphase_state.guess_static_parameters

        ordered_guesses = []
        for name in param_names:
            if name in param_guesses_dict:
                ordered_guesses.append(param_guesses_dict[name])
            else:
                ordered_guesses.append(0.0)

        opti.set_initial(variables.static_parameters, np.array(ordered_guesses, dtype=np.float64))
