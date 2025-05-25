"""
Initial guess validation and management with flexible ordering support.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from ..tl_types import FloatArray, InitialGuess
from .state import MeshState, VariableState


class InitialGuessRequirements:
    """Requirements for initial guess specification."""

    def __init__(
        self,
        states_shapes: list[tuple[int, int]],
        controls_shapes: list[tuple[int, int]],
        needs_initial_time: bool,
        needs_terminal_time: bool,
        integrals_length: int,
    ) -> None:
        self.states_shapes = states_shapes
        self.controls_shapes = controls_shapes
        self.needs_initial_time = needs_initial_time
        self.needs_terminal_time = needs_terminal_time
        self.integrals_length = integrals_length

    def __str__(self) -> str:
        req = ["Initial Guess Requirements:"]
        req.append(f"  States: {len(self.states_shapes)} intervals")
        for k, shape in enumerate(self.states_shapes):
            req.append(f"    Interval {k}: array of shape {shape}")
        req.append(f"  Controls: {len(self.controls_shapes)} intervals")
        for k, shape in enumerate(self.controls_shapes):
            req.append(f"    Interval {k}: array of shape {shape}")
        if self.needs_initial_time:
            req.append("  Initial time: float")
        if self.needs_terminal_time:
            req.append("  Terminal time: float")
        if self.integrals_length > 0:
            req.append(f"  Integrals: array of length {self.integrals_length}")
        return "\n".join(req)


class SolverInputSummary:
    """Summary of solver input configuration."""

    def __init__(
        self,
        mesh_intervals: int,
        polynomial_degrees: list[int],
        mesh_points: FloatArray,
        initial_guess_source: str,
        states_guess_shapes: list[tuple[int, int]] | None,
        controls_guess_shapes: list[tuple[int, int]] | None,
        initial_time_guess: float | None,
        terminal_time_guess: float | None,
        integrals_guess_length: int | None,
    ) -> None:
        self.mesh_intervals = mesh_intervals
        self.polynomial_degrees = polynomial_degrees
        self.mesh_points = mesh_points
        self.initial_guess_source = initial_guess_source
        self.states_guess_shapes = states_guess_shapes
        self.controls_guess_shapes = controls_guess_shapes
        self.initial_time_guess = initial_time_guess
        self.terminal_time_guess = terminal_time_guess
        self.integrals_guess_length = integrals_guess_length

    def __str__(self) -> str:
        summary = ["Solver Input Summary:"]
        summary.append(f"  Mesh intervals: {self.mesh_intervals}")
        summary.append(f"  Polynomial degrees: {self.polynomial_degrees}")
        summary.append(f"  Mesh points: {np.array2string(self.mesh_points, precision=4)}")
        summary.append(f"  Initial guess source: {self.initial_guess_source}")
        if self.states_guess_shapes:
            summary.append(f"  States guess shapes: {self.states_guess_shapes}")
        if self.controls_guess_shapes:
            summary.append(f"  Controls guess shapes: {self.controls_guess_shapes}")
        if self.initial_time_guess is not None:
            summary.append(f"  Initial time guess: {self.initial_time_guess}")
        if self.terminal_time_guess is not None:
            summary.append(f"  Terminal time guess: {self.terminal_time_guess}")
        if self.integrals_guess_length is not None:
            summary.append(f"  Integrals guess length: {self.integrals_guess_length}")
        return "\n".join(summary)


def set_initial_guess(
    current_guess_container: list,  # Mutable container to hold current guess
    mesh_state: MeshState,
    variable_state: VariableState,
    states: Sequence[FloatArray] | None = None,
    controls: Sequence[FloatArray] | None = None,
    initial_time: float | None = None,
    terminal_time: float | None = None,
    integrals: float | FloatArray | None = None,
) -> None:
    """
    Set initial guess for variables - FIXED: No longer requires mesh to be configured first.

    Validation is deferred until both mesh and initial guess are available.
    This allows users to call set_initial_guess() and set_mesh() in any order.
    """

    # Basic validation that doesn't require mesh information
    if states is not None:
        for k, state_array in enumerate(states):
            if not isinstance(state_array, np.ndarray):
                raise TypeError(f"State array for interval {k} must be numpy array")
            if state_array.dtype != np.float64:
                raise ValueError(f"State array for interval {k} must have dtype float64")
            if state_array.ndim != 2:
                raise ValueError(f"State array for interval {k} must be 2D")

    if controls is not None:
        for k, control_array in enumerate(controls):
            if not isinstance(control_array, np.ndarray):
                raise TypeError(f"Control array for interval {k} must be numpy array")
            if control_array.dtype != np.float64:
                raise ValueError(f"Control array for interval {k} must have dtype float64")
            if control_array.ndim != 2:
                raise ValueError(f"Control array for interval {k} must be 2D")

    # Basic validation for integrals
    if integrals is not None:
        if variable_state.num_integrals == 0:
            raise ValueError("Problem has no integrals, but integral guess was provided")

        if variable_state.num_integrals == 1:
            if not isinstance(integrals, int | float):
                raise ValueError(f"For single integral, provide scalar, got {type(integrals)}")
        else:
            integrals_array = np.array(integrals)
            if integrals_array.size != variable_state.num_integrals:
                raise ValueError(
                    f"Integral guess must have {variable_state.num_integrals} elements, "
                    f"got {integrals_array.size}"
                )

    # Store the initial guess - no mesh-dependent validation yet
    states_list: list[FloatArray] | None = None
    if states is not None:
        states_list = [np.array(s, dtype=np.float64) for s in states]

    controls_list: list[FloatArray] | None = None
    if controls is not None:
        controls_list = [np.array(c, dtype=np.float64) for c in controls]

    validated_integrals: float | FloatArray | None = None
    if integrals is not None:
        if variable_state.num_integrals == 1:
            validated_integrals = integrals
        else:
            validated_integrals = np.array(integrals)

    current_guess_container[0] = InitialGuess(
        initial_time_variable=initial_time,
        terminal_time_variable=terminal_time,
        states=states_list,
        controls=controls_list,
        integrals=validated_integrals,
    )


def can_validate_initial_guess(mesh_state: MeshState, variable_state: VariableState) -> bool:
    """Check if we have enough information to validate initial guess."""
    return mesh_state.configured


def get_initial_guess_requirements(
    mesh_state: MeshState, variable_state: VariableState
) -> InitialGuessRequirements:
    """Get requirements for initial guess - handles case where mesh isn't configured yet."""
    if not mesh_state.configured:
        # Return requirements that indicate mesh is needed
        return InitialGuessRequirements(
            states_shapes=[],
            controls_shapes=[],
            needs_initial_time=True,
            needs_terminal_time=True,
            integrals_length=variable_state.num_integrals,
        )

    # Get variable counts from unified storage
    num_states, num_controls = variable_state.get_variable_counts()

    states_shapes = [(num_states, N + 1) for N in mesh_state.collocation_points_per_interval]
    controls_shapes = [(num_controls, N) for N in mesh_state.collocation_points_per_interval]

    needs_initial_time = variable_state.t0_bounds[0] != variable_state.t0_bounds[1]
    needs_terminal_time = variable_state.tf_bounds[0] != variable_state.tf_bounds[1]

    return InitialGuessRequirements(
        states_shapes=states_shapes,
        controls_shapes=controls_shapes,
        needs_initial_time=needs_initial_time,
        needs_terminal_time=needs_terminal_time,
        integrals_length=variable_state.num_integrals,
    )


def validate_initial_guess(
    current_guess, mesh_state: MeshState, variable_state: VariableState
) -> None:
    """
    Validate the current initial guess - FIXED: Provides clear error if mesh not configured.

    This function is called when the solver runs, ensuring validation happens when needed.
    """
    if current_guess is None:
        return

    # Check if we can perform validation
    if not can_validate_initial_guess(mesh_state, variable_state):
        raise ValueError(
            "Cannot validate initial guess: mesh must be configured first. "
            "Call problem.set_mesh(polynomial_degrees, mesh_points) before solving."
        )

    ig = current_guess

    # Get variable counts from unified storage
    num_states, num_controls = variable_state.get_variable_counts()
    num_intervals = len(mesh_state.collocation_points_per_interval)

    # Validate states if provided
    if ig.states is not None:
        if len(ig.states) != num_intervals:
            raise ValueError(
                f"States guess has {len(ig.states)} arrays, "
                f"but problem needs {num_intervals} intervals"
            )

        for k, state_array in enumerate(ig.states):
            expected_shape = (num_states, mesh_state.collocation_points_per_interval[k] + 1)
            if state_array.shape != expected_shape:
                raise ValueError(
                    f"State array {k} has shape {state_array.shape}, expected {expected_shape}"
                )

    # Validate controls if provided
    if ig.controls is not None:
        if len(ig.controls) != num_intervals:
            raise ValueError(
                f"Controls guess has {len(ig.controls)} arrays, "
                f"but problem needs {num_intervals} intervals"
            )

        for k, control_array in enumerate(ig.controls):
            expected_shape = (num_controls, mesh_state.collocation_points_per_interval[k])
            if control_array.shape != expected_shape:
                raise ValueError(
                    f"Control array {k} has shape {control_array.shape}, expected {expected_shape}"
                )

    # Validate integrals if provided (already validated in set_initial_guess)
    if ig.integrals is not None:
        if variable_state.num_integrals == 0:
            raise ValueError("Integral guess provided but problem has no integrals")

        if variable_state.num_integrals == 1:
            if not isinstance(ig.integrals, int | float):
                raise ValueError("Single integral guess must be scalar")
        else:
            integrals_array = np.array(ig.integrals)
            if integrals_array.size != variable_state.num_integrals:
                raise ValueError(
                    f"Integral guess must have {variable_state.num_integrals} elements, "
                    f"got {integrals_array.size}"
                )


def get_solver_input_summary(
    current_guess, mesh_state: MeshState, variable_state: VariableState
) -> SolverInputSummary:
    """Get summary of solver input configuration - handles case where mesh isn't configured yet."""

    # If mesh isn't configured, return partial summary
    if not mesh_state.configured:
        return SolverInputSummary(
            mesh_intervals=0,
            polynomial_degrees=[],
            mesh_points=np.array([]),
            initial_guess_source="mesh_not_configured",
            states_guess_shapes=None,
            controls_guess_shapes=None,
            initial_time_guess=None,
            terminal_time_guess=None,
            integrals_guess_length=None,
        )

    states_shapes = None
    controls_shapes = None
    initial_time_guess = None
    terminal_time_guess = None
    integrals_length = None
    source = "no_initial_guess"

    if current_guess is not None:
        source = "partial_user_provided"
        states_shapes = (
            [cast(tuple[int, int], s.shape) for s in current_guess.states]
            if current_guess.states
            else None
        )
        controls_shapes = (
            [cast(tuple[int, int], c.shape) for c in current_guess.controls]
            if current_guess.controls
            else None
        )
        initial_time_guess = current_guess.initial_time_variable
        terminal_time_guess = current_guess.terminal_time_variable

        if current_guess.integrals is not None:
            if isinstance(current_guess.integrals, int | float):
                integrals_length = 1
            else:
                integrals_length = len(np.array(current_guess.integrals))

        # Check if guess is complete
        is_complete = True
        if current_guess.states is None or len(current_guess.states) != len(
            mesh_state.collocation_points_per_interval
        ):
            is_complete = False
        if current_guess.controls is None or len(current_guess.controls) != len(
            mesh_state.collocation_points_per_interval
        ):
            is_complete = False
        if variable_state.num_integrals > 0 and current_guess.integrals is None:
            is_complete = False

        if is_complete:
            source = "complete_user_provided"

    return SolverInputSummary(
        mesh_intervals=len(mesh_state.collocation_points_per_interval),
        polynomial_degrees=mesh_state.collocation_points_per_interval.copy(),
        mesh_points=(
            mesh_state.global_normalized_mesh_nodes.copy()
            if mesh_state.global_normalized_mesh_nodes is not None
            else np.array([])
        ),
        initial_guess_source=source,
        states_guess_shapes=states_shapes,
        controls_guess_shapes=controls_shapes,
        initial_time_guess=initial_time_guess,
        terminal_time_guess=terminal_time_guess,
        integrals_guess_length=integrals_length,
    )


def clear_initial_guess(current_guess_container: list) -> None:
    """Clear the current initial guess."""
    current_guess_container[0] = None
