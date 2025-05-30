# trajectolab/problem/initial_guess_problem.py
"""
Initial guess validation and management for multiphase problems.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..tl_types import FloatArray, MultiPhaseInitialGuess, PhaseID
from .state import MultiPhaseVariableState


class MultiPhaseInitialGuessRequirements:
    """Requirements for multiphase initial guess specification."""

    def __init__(
        self,
        phase_requirements: dict[PhaseID, dict[str, any]],
        static_parameters_length: int,
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
    current_guess_container: list,  # Mutable container to hold current guess
    multiphase_state: MultiPhaseVariableState,
    phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
    phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
    phase_initial_times: dict[PhaseID, float] | None = None,
    phase_terminal_times: dict[PhaseID, float] | None = None,
    phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
    static_parameters: FloatArray | None = None,
) -> None:
    """
    Set initial guess for multiphase variables.

    Validation is deferred until both mesh and initial guess are available for all phases.
    """

    # Basic validation that doesn't require mesh information
    if phase_states is not None:
        for phase_id, states_list in phase_states.items():
            if phase_id not in multiphase_state.phases:
                raise ValueError(f"Phase {phase_id} does not exist in problem")

            for k, state_array in enumerate(states_list):
                if not isinstance(state_array, np.ndarray):
                    raise TypeError(
                        f"State array for phase {phase_id} interval {k} must be numpy array"
                    )
                if state_array.dtype != np.float64:
                    raise ValueError(
                        f"State array for phase {phase_id} interval {k} must have dtype float64"
                    )
                if state_array.ndim != 2:
                    raise ValueError(f"State array for phase {phase_id} interval {k} must be 2D")

    if phase_controls is not None:
        for phase_id, controls_list in phase_controls.items():
            if phase_id not in multiphase_state.phases:
                raise ValueError(f"Phase {phase_id} does not exist in problem")

            for k, control_array in enumerate(controls_list):
                if not isinstance(control_array, np.ndarray):
                    raise TypeError(
                        f"Control array for phase {phase_id} interval {k} must be numpy array"
                    )
                if control_array.dtype != np.float64:
                    raise ValueError(
                        f"Control array for phase {phase_id} interval {k} must have dtype float64"
                    )
                if control_array.ndim != 2:
                    raise ValueError(f"Control array for phase {phase_id} interval {k} must be 2D")

    # Basic validation for integrals
    if phase_integrals is not None:
        for phase_id, integrals in phase_integrals.items():
            if phase_id not in multiphase_state.phases:
                raise ValueError(f"Phase {phase_id} does not exist in problem")

            phase_def = multiphase_state.phases[phase_id]
            if phase_def.num_integrals == 0:
                raise ValueError(
                    f"Phase {phase_id} has no integrals, but integral guess was provided"
                )

            if phase_def.num_integrals == 1:
                if not isinstance(integrals, int | float):
                    raise ValueError(
                        f"For single integral in phase {phase_id}, provide scalar, got {type(integrals)}"
                    )
            else:
                integrals_array = np.array(integrals)
                if integrals_array.size != phase_def.num_integrals:
                    raise ValueError(
                        f"Integral guess for phase {phase_id} must have {phase_def.num_integrals} elements, "
                        f"got {integrals_array.size}"
                    )

    # Basic validation for static parameters
    if static_parameters is not None:
        num_static_params = multiphase_state.static_parameters.get_parameter_count()
        if num_static_params == 0:
            raise ValueError("Problem has no static parameters, but parameter guess was provided")

        params_array = np.array(static_parameters)
        if params_array.size != num_static_params:
            raise ValueError(
                f"Static parameters guess must have {num_static_params} elements, "
                f"got {params_array.size}"
            )

    # Store the initial guess - no mesh-dependent validation yet
    validated_phase_states = {}
    if phase_states is not None:
        for phase_id, states_list in phase_states.items():
            validated_phase_states[phase_id] = [np.array(s, dtype=np.float64) for s in states_list]

    validated_phase_controls = {}
    if phase_controls is not None:
        for phase_id, controls_list in phase_controls.items():
            validated_phase_controls[phase_id] = [
                np.array(c, dtype=np.float64) for c in controls_list
            ]

    validated_phase_integrals = {}
    if phase_integrals is not None:
        for phase_id, integrals in phase_integrals.items():
            phase_def = multiphase_state.phases[phase_id]
            if phase_def.num_integrals == 1:
                validated_phase_integrals[phase_id] = integrals
            else:
                validated_phase_integrals[phase_id] = np.array(integrals)

    validated_static_parameters = None
    if static_parameters is not None:
        validated_static_parameters = np.array(static_parameters, dtype=np.float64)

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
    # All phases must have mesh configured
    for phase_def in multiphase_state.phases.values():
        if not phase_def.mesh_configured:
            return False
    return True


def get_multiphase_initial_guess_requirements(
    multiphase_state: MultiPhaseVariableState,
) -> MultiPhaseInitialGuessRequirements:
    """Get requirements for multiphase initial guess."""
    phase_requirements = {}

    for phase_id, phase_def in multiphase_state.phases.items():
        if not phase_def.mesh_configured:
            # Return requirements that indicate mesh is needed
            phase_requirements[phase_id] = {
                "states_shapes": [],
                "controls_shapes": [],
                "needs_initial_time": True,
                "needs_terminal_time": True,
                "integrals_length": phase_def.num_integrals,
            }
        else:
            # Get variable counts from phase
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
    """
    Validate the current multiphase initial guess.

    This function is called when the solver runs, ensuring validation happens when needed.
    """
    if current_guess is None:
        return

    # Check if we can perform validation
    if not can_validate_multiphase_initial_guess(multiphase_state):
        raise ValueError(
            "Cannot validate multiphase initial guess: all phases must have mesh configured first."
        )

    ig = current_guess

    # Validate each phase
    for phase_id, phase_def in multiphase_state.phases.items():
        num_states, num_controls = phase_def.get_variable_counts()
        num_intervals = len(phase_def.collocation_points_per_interval)

        # Validate states if provided for this phase
        if ig.phase_states is not None and phase_id in ig.phase_states:
            phase_states = ig.phase_states[phase_id]
            if len(phase_states) != num_intervals:
                raise ValueError(
                    f"Phase {phase_id} states guess has {len(phase_states)} arrays, "
                    f"but needs {num_intervals} intervals"
                )

            for k, state_array in enumerate(phase_states):
                expected_shape = (num_states, phase_def.collocation_points_per_interval[k] + 1)
                if state_array.shape != expected_shape:
                    raise ValueError(
                        f"Phase {phase_id} state array {k} has shape {state_array.shape}, expected {expected_shape}"
                    )

        # Validate controls if provided for this phase
        if ig.phase_controls is not None and phase_id in ig.phase_controls:
            phase_controls = ig.phase_controls[phase_id]
            if len(phase_controls) != num_intervals:
                raise ValueError(
                    f"Phase {phase_id} controls guess has {len(phase_controls)} arrays, "
                    f"but needs {num_intervals} intervals"
                )

            for k, control_array in enumerate(phase_controls):
                expected_shape = (num_controls, phase_def.collocation_points_per_interval[k])
                if control_array.shape != expected_shape:
                    raise ValueError(
                        f"Phase {phase_id} control array {k} has shape {control_array.shape}, expected {expected_shape}"
                    )

        # Validate integrals if provided for this phase
        if ig.phase_integrals is not None and phase_id in ig.phase_integrals:
            phase_integrals = ig.phase_integrals[phase_id]
            if phase_def.num_integrals == 0:
                raise ValueError(
                    f"Phase {phase_id} integral guess provided but phase has no integrals"
                )

            if phase_def.num_integrals == 1:
                if not isinstance(phase_integrals, int | float):
                    raise ValueError(f"Phase {phase_id} single integral guess must be scalar")
            else:
                integrals_array = np.array(phase_integrals)
                if integrals_array.size != phase_def.num_integrals:
                    raise ValueError(
                        f"Phase {phase_id} integral guess must have {phase_def.num_integrals} elements, "
                        f"got {integrals_array.size}"
                    )

    # Validate static parameters
    if ig.static_parameters is not None:
        num_static_params = multiphase_state.static_parameters.get_parameter_count()
        if num_static_params == 0:
            raise ValueError(
                "Static parameters guess provided but problem has no static parameters"
            )

        params_array = np.array(ig.static_parameters)
        if params_array.size != num_static_params:
            raise ValueError(
                f"Static parameters guess must have {num_static_params} elements, "
                f"got {params_array.size}"
            )


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
                "mesh_points_str": np.array2string(
                    phase_def.global_normalized_mesh_nodes, precision=4
                )
                if phase_def.global_normalized_mesh_nodes is not None
                else "None",
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
