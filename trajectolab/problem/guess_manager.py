"""
Initial guess management for optimal control problems.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..tl_types import FloatArray, FloatMatrix


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


class InitialGuessManager:
    """Manages initial guess for the problem."""

    def __init__(self) -> None:
        self.current_guess: Any = None

    def clear_guess(self) -> None:
        """Clear the current initial guess."""
        self.current_guess = None

    def set_guess(
        self,
        mesh_manager: Any,
        variable_manager: Any,
        states: Sequence[FloatMatrix] | None = None,
        controls: Sequence[FloatMatrix] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """Set initial guess for variables."""
        if not mesh_manager.configured:
            raise ValueError(
                "Mesh must be configured before setting initial guess. "
                "Call set_mesh(polynomial_degrees, mesh_points) first."
            )

        # Validate dimensions
        num_intervals = len(mesh_manager.collocation_points_per_interval)
        num_states = len(variable_manager.states)
        num_controls = len(variable_manager.controls)

        # Validate states if provided
        states_list: list[FloatMatrix] | None = None
        if states is not None:
            if len(states) != num_intervals:
                raise ValueError(
                    f"If providing states, must provide exactly {num_intervals} arrays, got {len(states)}"
                )

            for k, state_array in enumerate(states):
                expected_shape = (num_states, mesh_manager.collocation_points_per_interval[k] + 1)
                if state_array.shape != expected_shape:
                    raise ValueError(
                        f"State array for interval {k} has shape {state_array.shape}, "
                        f"expected {expected_shape}"
                    )

            states_list = [np.array(s, dtype=np.float64) for s in states]

        # Validate controls if provided
        controls_list: list[FloatMatrix] | None = None
        if controls is not None:
            if len(controls) != num_intervals:
                raise ValueError(
                    f"If providing controls, must provide exactly {num_intervals} arrays, got {len(controls)}"
                )

            for k, control_array in enumerate(controls):
                expected_shape = (num_controls, mesh_manager.collocation_points_per_interval[k])
                if control_array.shape != expected_shape:
                    raise ValueError(
                        f"Control array for interval {k} has shape {control_array.shape}, "
                        f"expected {expected_shape}"
                    )

            controls_list = [np.array(c, dtype=np.float64) for c in controls]

        # Validate integrals if provided
        validated_integrals: float | FloatArray | None = None
        if integrals is not None:
            if variable_manager.num_integrals == 0:
                raise ValueError("Problem has no integrals, but integral guess was provided")

            if variable_manager.num_integrals == 1:
                if not isinstance(integrals, int | float):
                    raise ValueError(f"For single integral, provide scalar, got {type(integrals)}")
                validated_integrals = integrals
            else:
                integrals_array = np.array(integrals)
                if integrals_array.size != variable_manager.num_integrals:
                    raise ValueError(
                        f"Integral guess must have {variable_manager.num_integrals} elements, "
                        f"got {integrals_array.size}"
                    )
                validated_integrals = integrals_array

        # Import here to avoid circular import
        from ..direct_solver import InitialGuess

        self.current_guess = InitialGuess(
            initial_time_variable=initial_time,
            terminal_time_variable=terminal_time,
            states=states_list,
            controls=controls_list,
            integrals=validated_integrals,
        )

    def get_requirements(
        self, mesh_manager: Any, variable_manager: Any
    ) -> InitialGuessRequirements:
        """Get requirements for initial guess."""
        if not mesh_manager.configured:
            raise ValueError("Mesh must be configured before getting initial guess requirements.")

        num_states = len(variable_manager.states)
        num_controls = len(variable_manager.controls)

        states_shapes = [(num_states, N + 1) for N in mesh_manager.collocation_points_per_interval]
        controls_shapes = [(num_controls, N) for N in mesh_manager.collocation_points_per_interval]

        needs_initial_time = variable_manager.t0_bounds[0] != variable_manager.t0_bounds[1]
        needs_terminal_time = variable_manager.tf_bounds[0] != variable_manager.tf_bounds[1]

        return InitialGuessRequirements(
            states_shapes=states_shapes,
            controls_shapes=controls_shapes,
            needs_initial_time=needs_initial_time,
            needs_terminal_time=needs_terminal_time,
            integrals_length=variable_manager.num_integrals,
        )

    def validate_guess(self, mesh_manager: Any, variable_manager: Any) -> None:
        """Validate the current initial guess."""
        if not mesh_manager.configured:
            raise ValueError("Mesh must be configured before validating initial guess")

        if self.current_guess is None:
            return

        ig = self.current_guess

        # Validate states if provided
        if ig.states is not None:
            requirements = self.get_requirements(mesh_manager, variable_manager)

            if len(ig.states) != len(requirements.states_shapes):
                raise ValueError(
                    f"States guess has {len(ig.states)} arrays, "
                    f"but problem needs {len(requirements.states_shapes)}"
                )

            for k, (state_array, expected_shape) in enumerate(
                zip(ig.states, requirements.states_shapes, strict=False)
            ):
                if state_array.shape != expected_shape:
                    raise ValueError(
                        f"State array {k} has shape {state_array.shape}, expected {expected_shape}"
                    )

        # Validate controls if provided
        if ig.controls is not None:
            requirements = self.get_requirements(mesh_manager, variable_manager)

            if len(ig.controls) != len(requirements.controls_shapes):
                raise ValueError(
                    f"Controls guess has {len(ig.controls)} arrays, "
                    f"but problem needs {len(requirements.controls_shapes)}"
                )

            for k, (control_array, expected_shape) in enumerate(
                zip(ig.controls, requirements.controls_shapes, strict=False)
            ):
                if control_array.shape != expected_shape:
                    raise ValueError(
                        f"Control array {k} has shape {control_array.shape}, expected {expected_shape}"
                    )

        # Validate integrals if provided
        if ig.integrals is not None:
            if variable_manager.num_integrals == 0:
                raise ValueError("Integral guess provided but problem has no integrals")

            if variable_manager.num_integrals == 1:
                if not isinstance(ig.integrals, int | float):
                    raise ValueError("Single integral guess must be scalar")
            else:
                integrals_array = np.array(ig.integrals)
                if integrals_array.size != variable_manager.num_integrals:
                    raise ValueError(
                        f"Integral guess must have {variable_manager.num_integrals} elements, "
                        f"got {integrals_array.size}"
                    )

    def get_solver_input_summary(
        self, mesh_manager: Any, variable_manager: Any
    ) -> SolverInputSummary:
        """Get summary of solver input configuration."""
        if not mesh_manager.configured:
            raise ValueError("Mesh must be configured to get solver input summary")

        states_shapes = None
        controls_shapes = None
        initial_time_guess = None
        terminal_time_guess = None
        integrals_length = None
        source = "no_initial_guess"

        if self.current_guess is not None:
            source = "partial_user_provided"
            states_shapes = (
                [cast(tuple[int, int], s.shape) for s in self.current_guess.states]
                if self.current_guess.states
                else None
            )
            controls_shapes = (
                [cast(tuple[int, int], c.shape) for c in self.current_guess.controls]
                if self.current_guess.controls
                else None
            )
            initial_time_guess = self.current_guess.initial_time_variable
            terminal_time_guess = self.current_guess.terminal_time_variable

            if self.current_guess.integrals is not None:
                if isinstance(self.current_guess.integrals, int | float):
                    integrals_length = 1
                else:
                    integrals_length = len(np.array(self.current_guess.integrals))

            # Check if guess is complete
            is_complete = True
            if self.current_guess.states is None or len(self.current_guess.states) != len(
                mesh_manager.collocation_points_per_interval
            ):
                is_complete = False
            if self.current_guess.controls is None or len(self.current_guess.controls) != len(
                mesh_manager.collocation_points_per_interval
            ):
                is_complete = False
            if variable_manager.num_integrals > 0 and self.current_guess.integrals is None:
                is_complete = False

            if is_complete:
                source = "complete_user_provided"

        return SolverInputSummary(
            mesh_intervals=len(mesh_manager.collocation_points_per_interval),
            polynomial_degrees=mesh_manager.collocation_points_per_interval.copy(),
            mesh_points=(
                mesh_manager.global_normalized_mesh_nodes.copy()
                if mesh_manager.global_normalized_mesh_nodes is not None
                else np.array([])
            ),
            initial_guess_source=source,
            states_guess_shapes=states_shapes,
            controls_guess_shapes=controls_shapes,
            initial_time_guess=initial_time_guess,
            terminal_time_guess=terminal_time_guess,
            integrals_guess_length=integrals_length,
        )
