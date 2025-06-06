from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..input_validation import (
    _validate_multiphase_initial_guess_structure,
    validate_array_numerical_integrity,
    validate_integral_values,
)
from ..tl_types import FloatArray, MultiPhaseInitialGuess, PhaseID
from .state import MultiPhaseVariableState


def _validate_phase_exists(phases: dict[PhaseID, Any], phase_id: PhaseID) -> None:
    if phase_id not in phases:
        raise ValueError(f"Phase {phase_id} does not exist in problem")


def _validate_and_convert_arrays(arrays: Sequence[Any], array_type: str) -> list[np.ndarray]:
    return [np.array(arr, dtype=np.float64) for arr in arrays]


def _process_single_or_multi_integral(
    integrals: float | FloatArray, num_integrals: int
) -> float | np.ndarray:
    if num_integrals == 1:
        return integrals
    return np.array(integrals)


def _validate_time_values(time_values: dict[PhaseID, float] | None, time_type: str) -> None:
    if time_values is not None:
        for phase_id, time_val in time_values.items():
            validate_array_numerical_integrity(
                np.array([time_val]), f"Phase {phase_id} {time_type} time"
            )


def _validate_static_parameters(
    static_parameters: FloatArray | None, expected_count: int
) -> np.ndarray | None:
    if static_parameters is None:
        return None

    if expected_count == 0:
        raise ValueError("Problem has no static parameters, but parameter guess was provided")

    params_array = np.array(static_parameters, dtype=np.float64)
    if params_array.size != expected_count:
        raise ValueError(
            f"Static parameters guess must have {expected_count} elements, got {params_array.size}"
        )
    return params_array


def _validate_phase_states(
    phase_states: dict[PhaseID, Sequence[FloatArray]] | None, phases: dict[PhaseID, Any]
) -> dict[PhaseID, list[np.ndarray]]:
    validated_states = {}
    if phase_states is not None:
        for phase_id, states_list in phase_states.items():
            _validate_phase_exists(phases, phase_id)
            validated_states[phase_id] = _validate_and_convert_arrays(states_list, "states")
    return validated_states


def _validate_phase_controls(
    phase_controls: dict[PhaseID, Sequence[FloatArray]] | None, phases: dict[PhaseID, Any]
) -> dict[PhaseID, list[np.ndarray]]:
    validated_controls = {}
    if phase_controls is not None:
        for phase_id, controls_list in phase_controls.items():
            _validate_phase_exists(phases, phase_id)
            validated_controls[phase_id] = _validate_and_convert_arrays(controls_list, "controls")
    return validated_controls


def _validate_phase_integrals(
    phase_integrals: dict[PhaseID, float | FloatArray] | None, phases: dict[PhaseID, Any]
) -> dict[PhaseID, float | np.ndarray]:
    validated_integrals = {}
    if phase_integrals is not None:
        for phase_id, integrals in phase_integrals.items():
            _validate_phase_exists(phases, phase_id)
            phase_def = phases[phase_id]

            validate_integral_values(integrals, phase_def.num_integrals)
            validated_integrals[phase_id] = _process_single_or_multi_integral(
                integrals, phase_def.num_integrals
            )
    return validated_integrals


def _create_validated_initial_guess(
    validated_states: dict[PhaseID, list[np.ndarray]],
    validated_controls: dict[PhaseID, list[np.ndarray]],
    phase_initial_times: dict[PhaseID, float] | None,
    phase_terminal_times: dict[PhaseID, float] | None,
    validated_integrals: dict[PhaseID, float | np.ndarray],
    validated_static_parameters: np.ndarray | None,
) -> MultiPhaseInitialGuess:
    return MultiPhaseInitialGuess(
        phase_states=validated_states if validated_states else None,
        phase_controls=validated_controls if validated_controls else None,
        phase_initial_times=phase_initial_times,
        phase_terminal_times=phase_terminal_times,
        phase_integrals=validated_integrals if validated_integrals else None,
        static_parameters=validated_static_parameters,
    )


class MultiPhaseInitialGuessRequirements:
    def __init__(
        self, phase_requirements: dict[PhaseID, dict[str, Any]], static_parameters_length: int
    ) -> None:
        self.phase_requirements = phase_requirements
        self.static_parameters_length = static_parameters_length

    def __str__(self) -> str:
        req = ["Multiphase Initial Guess Requirements:"]

        for phase_id, phase_req in self.phase_requirements.items():
            req.append(f"  Phase {phase_id}:")

            states_shapes = phase_req.get("states_shapes", [])
            for k, shape in enumerate(states_shapes):
                req.append(f"    States interval {k}: array of shape {shape}")

            controls_shapes = phase_req.get("controls_shapes", [])
            for k, shape in enumerate(controls_shapes):
                req.append(f"    Controls interval {k}: array of shape {shape}")

            if phase_req.get("needs_initial_time", False):
                req.append("    Initial time: float")
            if phase_req.get("needs_terminal_time", False):
                req.append("    Terminal time: float")
            if phase_req.get("integrals_length", 0) > 0:
                req.append(f"    Integrals: array of length {phase_req['integrals_length']}")

        if self.static_parameters_length > 0:
            req.append(f"  Static Parameters: array of length {self.static_parameters_length}")

        return "\n".join(req)


class MultiPhaseSolverInputSummary:
    def __init__(
        self,
        num_phases: int,
        phase_summaries: dict[PhaseID, dict[str, Any]],
        static_parameters_length: int,
        initial_guess_source: str,
    ) -> None:
        self.num_phases = num_phases
        self.phase_summaries = phase_summaries
        self.static_parameters_length = static_parameters_length
        self.initial_guess_source = initial_guess_source

    def __str__(self) -> str:
        summary = ["Multiphase Solver Input Summary:"]
        summary.append(f"  Number of phases: {self.num_phases}")
        summary.append(f"  Initial guess source: {self.initial_guess_source}")

        for phase_id, phase_summary in self.phase_summaries.items():
            summary.append(f"  Phase {phase_id}:")
            summary.append(f"    Mesh intervals: {phase_summary.get('mesh_intervals', 0)}")
            summary.append(f"    Polynomial degrees: {phase_summary.get('polynomial_degrees', [])}")
            summary.append(
                f"    Mesh points: {phase_summary.get('mesh_points_str', 'Not configured')}"
            )

        if self.static_parameters_length > 0:
            summary.append(f"  Static parameters: {self.static_parameters_length}")

        return "\n".join(summary)


def _set_multiphase_initial_guess(
    current_guess_container: list,
    multiphase_state: MultiPhaseVariableState,
    phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
    phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
    phase_initial_times: dict[PhaseID, float] | None = None,
    phase_terminal_times: dict[PhaseID, float] | None = None,
    phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
    static_parameters: FloatArray | None = None,
) -> None:
    validated_states = _validate_phase_states(phase_states, multiphase_state.phases)
    validated_controls = _validate_phase_controls(phase_controls, multiphase_state.phases)
    validated_integrals = _validate_phase_integrals(phase_integrals, multiphase_state.phases)

    expected_static_params = multiphase_state.static_parameters.get_parameter_count()
    validated_static_parameters = _validate_static_parameters(
        static_parameters, expected_static_params
    )

    _validate_time_values(phase_initial_times, "initial")
    _validate_time_values(phase_terminal_times, "terminal")

    current_guess_container[0] = _create_validated_initial_guess(
        validated_states,
        validated_controls,
        phase_initial_times,
        phase_terminal_times,
        validated_integrals,
        validated_static_parameters,
    )


def _can__validate_multiphase_initial_guess(multiphase_state: MultiPhaseVariableState) -> bool:
    return all(phase_def.mesh_configured for phase_def in multiphase_state.phases.values())


def _validate_multiphase_initial_guess(
    current_guess, multiphase_state: MultiPhaseVariableState
) -> None:
    if current_guess is None:
        return

    if not _can__validate_multiphase_initial_guess(multiphase_state):
        raise ValueError(
            "Cannot validate multiphase initial guess: all phases must have mesh configured first."
        )

    from typing import cast

    from ..problem import Problem
    from ..tl_types import ProblemProtocol

    dummy_problem = Problem()
    dummy_problem._multiphase_state = multiphase_state
    _validate_multiphase_initial_guess_structure(
        current_guess, cast(ProblemProtocol, dummy_problem)
    )
