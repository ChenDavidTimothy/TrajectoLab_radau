from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..input_validation import (
    validate_array_numerical_integrity,
    validate_integral_values,
    validate_multiphase_initial_guess_structure,
)
from ..tl_types import FloatArray, MultiPhaseInitialGuess, PhaseID
from .state import MultiPhaseVariableState


class MultiPhaseInitialGuessRequirements:
    """Requirements for multiphase initial guess specification."""

    def __init__(
        self, phase_requirements: dict[PhaseID, dict[str, any]], static_parameters_length: int
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
    """Summary of multiphase solver input configuration."""

    def __init__(
        self,
        num_phases: int,
        phase_summaries: dict[PhaseID, dict[str, any]],
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


def set_multiphase_initial_guess(
    current_guess_container: list,
    multiphase_state: MultiPhaseVariableState,
    phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
    phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
    phase_initial_times: dict[PhaseID, float] | None = None,
    phase_terminal_times: dict[PhaseID, float] | None = None,
    phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
    static_parameters: FloatArray | None = None,
) -> None:
    """Set initial guess for multiphase variables."""

    # Basic validation - detailed validation deferred until solve time
    validated_phase_states = {}
    if phase_states is not None:
        for phase_id, states_list in phase_states.items():
            if phase_id not in multiphase_state.phases:
                raise ValueError(f"Phase {phase_id} does not exist in problem")
            validated_phase_states[phase_id] = [np.array(s, dtype=np.float64) for s in states_list]

    validated_phase_controls = {}
    if phase_controls is not None:
        for phase_id, controls_list in phase_controls.items():
            if phase_id not in multiphase_state.phases:
                raise ValueError(f"Phase {phase_id} does not exist in problem")
            validated_phase_controls[phase_id] = [
                np.array(c, dtype=np.float64) for c in controls_list
            ]

    validated_phase_integrals = {}
    if phase_integrals is not None:
        for phase_id, integrals in phase_integrals.items():
            if phase_id not in multiphase_state.phases:
                raise ValueError(f"Phase {phase_id} does not exist in problem")
            # SINGLE validation call
            validate_integral_values(integrals, multiphase_state.phases[phase_id].num_integrals)

            phase_def = multiphase_state.phases[phase_id]
            if phase_def.num_integrals == 1:
                validated_phase_integrals[phase_id] = integrals
            else:
                validated_phase_integrals[phase_id] = np.array(integrals)

    validated_static_parameters = None
    if static_parameters is not None:
        num_static_params = multiphase_state.static_parameters.get_parameter_count()
        if num_static_params == 0:
            raise ValueError("Problem has no static parameters, but parameter guess was provided")

        params_array = np.array(static_parameters, dtype=np.float64)
        if params_array.size != num_static_params:
            raise ValueError(
                f"Static parameters guess must have {num_static_params} elements, got {params_array.size}"
            )
        validated_static_parameters = params_array

    # Validate time values
    if phase_initial_times is not None:
        for phase_id, t0 in phase_initial_times.items():
            validate_array_numerical_integrity(np.array([t0]), f"Phase {phase_id} initial time")

    if phase_terminal_times is not None:
        for phase_id, tf in phase_terminal_times.items():
            validate_array_numerical_integrity(np.array([tf]), f"Phase {phase_id} terminal time")

    # Store the validated initial guess
    current_guess_container[0] = MultiPhaseInitialGuess(
        phase_states=validated_phase_states if validated_phase_states else None,
        phase_controls=validated_phase_controls if validated_phase_controls else None,
        phase_initial_times=phase_initial_times,
        phase_terminal_times=phase_terminal_times,
        phase_integrals=validated_phase_integrals if validated_phase_integrals else None,
        static_parameters=validated_static_parameters,
    )


def can_validate_multiphase_initial_guess(multiphase_state: MultiPhaseVariableState) -> bool:
    """Check if we have enough information to validate multiphase initial guess."""
    return all(phase_def.mesh_configured for phase_def in multiphase_state.phases.values())


def get_multiphase_initial_guess_requirements(
    multiphase_state: MultiPhaseVariableState,
) -> MultiPhaseInitialGuessRequirements:
    """Get requirements for multiphase initial guess."""
    phase_requirements = {}

    for phase_id, phase_def in multiphase_state.phases.items():
        if not phase_def.mesh_configured:
            phase_requirements[phase_id] = {
                "states_shapes": [],
                "controls_shapes": [],
                "needs_initial_time": True,
                "needs_terminal_time": True,
                "integrals_length": phase_def.num_integrals,
            }
        else:
            num_states, num_controls = phase_def.get_variable_counts()
            states_shapes = [(num_states, N + 1) for N in phase_def.collocation_points_per_interval]
            controls_shapes = [(num_controls, N) for N in phase_def.collocation_points_per_interval]
            needs_initial_time = phase_def.t0_bounds[0] != phase_def.t0_bounds[1]
            needs_terminal_time = phase_def.tf_bounds[0] != phase_def.tf_bounds[1]

            phase_requirements[phase_id] = {
                "states_shapes": states_shapes,
                "controls_shapes": controls_shapes,
                "needs_initial_time": needs_initial_time,
                "needs_terminal_time": needs_terminal_time,
                "integrals_length": phase_def.num_integrals,
            }

    static_parameters_length = multiphase_state.static_parameters.get_parameter_count()

    return MultiPhaseInitialGuessRequirements(
        phase_requirements=phase_requirements,
        static_parameters_length=static_parameters_length,
    )


def validate_multiphase_initial_guess(
    current_guess, multiphase_state: MultiPhaseVariableState
) -> None:
    """Validate the current multiphase initial guess using CENTRALIZED validation."""
    if current_guess is None:
        return

    if not can_validate_multiphase_initial_guess(multiphase_state):
        raise ValueError(
            "Cannot validate multiphase initial guess: all phases must have mesh configured first."
        )

    # SINGLE comprehensive validation call
    from ..problem import Problem  # Avoid circular import

    dummy_problem = Problem()
    dummy_problem._multiphase_state = multiphase_state
    validate_multiphase_initial_guess_structure(current_guess, dummy_problem)


def get_multiphase_solver_input_summary(
    current_guess, multiphase_state: MultiPhaseVariableState
) -> MultiPhaseSolverInputSummary:
    """Get summary of multiphase solver input configuration."""
    phase_summaries = {}
    initial_guess_source = "no_initial_guess"

    for phase_id, phase_def in multiphase_state.phases.items():
        if not phase_def.mesh_configured:
            phase_summaries[phase_id] = {
                "mesh_intervals": 0,
                "polynomial_degrees": [],
                "mesh_points_str": "mesh_not_configured",
            }
        else:
            phase_summaries[phase_id] = {
                "mesh_intervals": len(phase_def.collocation_points_per_interval),
                "polynomial_degrees": phase_def.collocation_points_per_interval.copy(),
                "mesh_points_str": (
                    np.array2string(phase_def.global_normalized_mesh_nodes, precision=4)
                    if phase_def.global_normalized_mesh_nodes is not None
                    else "None"
                ),
            }

    if current_guess is not None:
        initial_guess_source = "partial_multiphase_provided"

        # Check if guess is complete for all phases
        is_complete = True
        for phase_id, phase_def in multiphase_state.phases.items():
            if not phase_def.mesh_configured:
                is_complete = False
                break

            # Check states guess
            if (
                current_guess.phase_states is None
                or phase_id not in current_guess.phase_states
                or len(current_guess.phase_states[phase_id])
                != len(phase_def.collocation_points_per_interval)
            ):
                is_complete = False
                break

            # Check controls guess
            if (
                current_guess.phase_controls is None
                or phase_id not in current_guess.phase_controls
                or len(current_guess.phase_controls[phase_id])
                != len(phase_def.collocation_points_per_interval)
            ):
                is_complete = False
                break

            # Check integrals guess
            if phase_def.num_integrals > 0 and (
                current_guess.phase_integrals is None
                or phase_id not in current_guess.phase_integrals
            ):
                is_complete = False
                break

        # Check static parameters
        num_static_params = multiphase_state.static_parameters.get_parameter_count()
        if num_static_params > 0 and current_guess.static_parameters is None:
            is_complete = False

        if is_complete:
            initial_guess_source = "complete_multiphase_provided"

    static_parameters_length = multiphase_state.static_parameters.get_parameter_count()

    return MultiPhaseSolverInputSummary(
        num_phases=len(multiphase_state.phases),
        phase_summaries=phase_summaries,
        static_parameters_length=static_parameters_length,
        initial_guess_source=initial_guess_source,
    )


def clear_multiphase_initial_guess(current_guess_container: list) -> None:
    """Clear the current multiphase initial guess."""
    current_guess_container[0] = None
