from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from maptor.input_validation import (
    _validate_array_numerical_integrity,
    _validate_integral_values,
)
from maptor.problem.state import MultiPhaseVariableState
from maptor.tl_types import FloatArray, MultiPhaseInitialGuess, NumericArrayLike, PhaseID


def _validate_phase_exists(phases: dict[PhaseID, Any], phase_id: PhaseID) -> None:
    if phase_id not in phases:
        raise ValueError(f"Phase {phase_id} does not exist in problem")


def _validate_and_convert_arrays(arrays: Sequence[Any]) -> list[np.ndarray]:
    return [np.array(arr, dtype=np.float64) for arr in arrays]


def _process_single_or_multi_integral(
    integrals: float | np.ndarray, num_integrals: int
) -> float | np.ndarray:
    if num_integrals == 1:
        return integrals
    return np.array(integrals)


def _validate_time_values(time_values: dict[PhaseID, float] | None, time_type: str) -> None:
    if time_values is not None:
        for phase_id, time_val in time_values.items():
            _validate_array_numerical_integrity(
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
    phase_states: dict[PhaseID, Sequence[NumericArrayLike]] | None, phases: dict[PhaseID, Any]
) -> dict[PhaseID, list[np.ndarray]]:
    validated_states = {}
    if phase_states is not None:
        for phase_id, states_list in phase_states.items():
            _validate_phase_exists(phases, phase_id)
            validated_states[phase_id] = _validate_and_convert_arrays(states_list)
    return validated_states


def _validate_phase_controls(
    phase_controls: dict[PhaseID, Sequence[NumericArrayLike]] | None, phases: dict[PhaseID, Any]
) -> dict[PhaseID, list[np.ndarray]]:
    validated_controls = {}
    if phase_controls is not None:
        for phase_id, controls_list in phase_controls.items():
            _validate_phase_exists(phases, phase_id)
            validated_controls[phase_id] = _validate_and_convert_arrays(controls_list)
    return validated_controls


def _validate_phase_integrals(
    phase_integrals: dict[PhaseID, float | NumericArrayLike] | None, phases: dict[PhaseID, Any]
) -> dict[PhaseID, float | np.ndarray]:
    validated_integrals = {}
    if phase_integrals is not None:
        for phase_id, integrals in phase_integrals.items():
            _validate_phase_exists(phases, phase_id)
            phase_def = phases[phase_id]

            _validate_integral_values(integrals, phase_def.num_integrals)
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


def _set_multiphase_initial_guess(
    current_guess_container: list,
    multiphase_state: MultiPhaseVariableState,
    phase_states: dict[PhaseID, Sequence[NumericArrayLike]] | None = None,
    phase_controls: dict[PhaseID, Sequence[NumericArrayLike]] | None = None,
    phase_initial_times: dict[PhaseID, float] | None = None,
    phase_terminal_times: dict[PhaseID, float] | None = None,
    phase_integrals: dict[PhaseID, float | NumericArrayLike] | None = None,
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
