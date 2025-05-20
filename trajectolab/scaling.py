"""
scaling.py - Automatic scaling for optimal control problems in TrajectoLab.

Implements scaling rules from Section 4.12:
- Rule 1: Scale from a control perspective (same scale for all grid points)
- Rule 2: Variable scaling based on bounds or initial guess
- Rule 3: ODE defect scaling matches state variable scaling
"""

from typing import TypeAlias

import numpy as np

from trajectolab.tl_types import CasadiMX, FloatArray


# Type aliases
_ScalingDict: TypeAlias = dict[str, tuple[float, float]]  # (scale_factor, shift)


class Scaling:
    """
    Handles variable and constraint scaling for optimal control problems.
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialize scaling system.

        Args:
            enabled: Whether automatic scaling is enabled (default: True)
        """
        self.state_scaling: _ScalingDict = {}
        self.control_scaling: _ScalingDict = {}
        self.enabled: bool = enabled

    def compute_from_problem(self, problem) -> None:
        """
        Compute scaling factors from a Problem object.

        Args:
            problem: TrajectoLab Problem instance
        """
        # Extract state information
        state_names = list(problem._states.keys())
        state_bounds = {}

        for name, info in problem._states.items():
            state_bounds[name] = {"lower": info.get("lower"), "upper": info.get("upper")}

        # Extract control information
        control_names = list(problem._controls.keys())
        control_bounds = {}

        for name, info in problem._controls.items():
            control_bounds[name] = {"lower": info.get("lower"), "upper": info.get("upper")}

        # Extract initial guesses if available
        state_guesses = None
        control_guesses = None

        if problem.initial_guess is not None:
            if problem.initial_guess.states is not None:
                state_guesses = self._extract_guesses_from_trajectories(
                    problem.initial_guess.states, state_names, len(problem._states)
                )

            if problem.initial_guess.controls is not None:
                control_guesses = self._extract_guesses_from_trajectories(
                    problem.initial_guess.controls, control_names, len(problem._controls)
                )

        # Compute scaling factors
        self.compute_scaling_factors(
            state_names, state_bounds, state_guesses, control_names, control_bounds, control_guesses
        )

    def _extract_guesses_from_trajectories(
        self, trajectories: list[FloatArray], var_names: list[str], num_vars: int
    ) -> dict[str, FloatArray]:
        """
        Extract initial guesses from trajectory arrays.

        Args:
            trajectories: List of trajectory arrays (one per mesh interval)
            var_names: List of variable names
            num_vars: Number of variables

        Returns:
            Dict mapping variable names to arrays of initial guesses
        """
        if not trajectories or len(trajectories) == 0:
            return {}

        # Initialize result dictionary
        result = {}

        # Concatenate values across all intervals
        for i, name in enumerate(var_names):
            if i >= num_vars:
                break

            # Collect values for this variable across all intervals
            all_values = []
            for traj in trajectories:
                if i < traj.shape[0]:  # Make sure the variable exists in this trajectory
                    all_values.extend(traj[i, :].flatten())

            if all_values:
                result[name] = np.array(all_values, dtype=np.float64)

        return result

    def compute_scaling_factors(
        self,
        state_names: list[str],
        state_bounds: dict[str, dict[str, float | None]],
        state_guesses: dict[str, FloatArray] | None = None,
        control_names: list[str] | None = None,
        control_bounds: dict[str, dict[str, float | None]] | None = None,
        control_guesses: dict[str, FloatArray] | None = None,
    ) -> None:
        """Compute scaling factors according to Rules 1 & 2."""
        print("\n=== SCALING FACTORS COMPUTATION ===")
        print(f"Scaling enabled: {self.enabled}")

        if not self.enabled:
            # Set all scaling factors to 1.0 and shifts to 0.0
            for name in state_names:
                self.state_scaling[name] = (1.0, 0.0)
                print(f"  State {name}: Default scaling (1.0, 0.0) - scaling disabled")
            for name in control_names or []:
                self.control_scaling[name] = (1.0, 0.0)
                print(f"  Control {name}: Default scaling (1.0, 0.0) - scaling disabled")
            return

        # Process state variables
        print("\nSTATE VARIABLE SCALING:")
        for name in state_names:
            bounds = state_bounds.get(name, {})
            lower = bounds.get("lower")
            upper = bounds.get("upper")

            print(f"  State {name}:")
            print(f"    Bounds: lower={lower}, upper={upper}")

            # Use bounds if available (Rule 2a)
            if lower is not None and upper is not None and lower < upper:
                print("    Using bounds for scaling (Rule 2a)")
                print(f"    Range: [{lower}, {upper}]")
                scale_factor = 1.0 / (upper - lower)  # Equation (4.250)
                shift = 0.5 - upper / (upper - lower)  # Equation (4.251)
                print(f"    Calculated: scale_factor={scale_factor}, shift={shift}")

            # Otherwise use initial guess to estimate range (Rule 2b)
            elif state_guesses and name in state_guesses and len(state_guesses[name]) > 0:
                guess_array = state_guesses[name]
                min_val = np.min(guess_array)
                max_val = np.max(guess_array)

                print("    Using initial guess for scaling (Rule 2b)")
                print(f"    Guess min={min_val}, max={max_val}")

                if np.isclose(min_val, max_val):
                    # Handle case where all values are nearly identical
                    print("    EDGE CASE: min_val ≈ max_val")
                    if np.abs(min_val) < 1e-10:
                        scale_factor = 1.0
                        shift = 0.0
                        print("    Near zero value, using default scaling")
                    else:
                        # Use magnitude of value as scale factor
                        scale_factor = 1.0 / max(1.0, 2 * np.abs(min_val))
                        shift = 0.0
                        print("    Using magnitude-based scaling")
                else:
                    # Use range from guess values
                    scale_factor = 1.0 / (max_val - min_val)
                    shift = 0.5 - max_val / (max_val - min_val)
                    print(f"    Calculated: scale_factor={scale_factor}, shift={shift}")
            else:
                # Default scaling when no information is available (Rule 2 default)
                print("    No bounds or guess available, using default scaling")
                scale_factor = 1.0
                shift = 0.0
                print(f"    Default: scale_factor={scale_factor}, shift={shift}")

        self.state_scaling[name] = (scale_factor, shift)
        print(f"    FINAL: state_scaling[{name}] = ({scale_factor}, {shift})")

        # Compare to manual scaling if this is altitude or velocity
        if name == "h":
            print(f"    COMPARISON: Manual h_scale = 1e5, auto scale = {scale_factor}")
        elif name == "v":
            print(f"    COMPARISON: Manual v_scale = 1e4, auto scale = {scale_factor}")

        # Process control variables (similar logic as states)
        for name in control_names or []:
            bounds = control_bounds.get(name, {}) if control_bounds else {}
            lower = bounds.get("lower")
            upper = bounds.get("upper")

            if lower is not None and upper is not None and lower < upper:
                scale_factor = 1.0 / (upper - lower)
                shift = 0.5 - upper / (upper - lower)
            elif control_guesses and name in control_guesses and len(control_guesses[name]) > 0:
                guess_array = control_guesses[name]
                min_val = np.min(guess_array)
                max_val = np.max(guess_array)

                if np.isclose(min_val, max_val):
                    if np.abs(min_val) < 1e-10:
                        scale_factor = 1.0
                        shift = 0.0
                    else:
                        scale_factor = 1.0 / max(1.0, 2 * np.abs(min_val))
                        shift = 0.0
                else:
                    scale_factor = 1.0 / (max_val - min_val)
                    shift = 0.5 - max_val / (max_val - min_val)
            else:
                scale_factor = 1.0
                shift = 0.0

            self.control_scaling[name] = (scale_factor, shift)

    def get_state_scaling(self, name: str) -> tuple[float, float]:
        """Get scaling factor and shift for a state variable."""
        return self.state_scaling.get(name, (1.0, 0.0))

    def get_control_scaling(self, name: str) -> tuple[float, float]:
        """Get scaling factor and shift for a control variable."""
        return self.control_scaling.get(name, (1.0, 0.0))

    def get_state_scaling_factor(self, name: str) -> float:
        """Get just the scaling factor for a state variable (for Rule 3)."""
        return self.state_scaling.get(name, (1.0, 0.0))[0]

    def scale_defect_constraint(self, state_name: str, constraint: CasadiMX) -> CasadiMX:
        """
        Scale a defect constraint according to Rule 3.

        Args:
            state_name: Name of the state variable
            constraint: Constraint expression to scale

        Returns:
            Scaled constraint
        """
        # Strict check for enabled flag
        if not self.enabled:
            return constraint

        # Apply scaling to constraint (Rule 3: W_f = V_y)
        factor = self.get_state_scaling_factor(state_name)
        return factor * constraint


def apply_scaling_to_defect_constraints(
    opti,
    scaling: Scaling,
    state_names: list[str],
    state_derivative_at_colloc: CasadiMX,
    state_derivative_rhs_vector: CasadiMX,
    tau_to_time_scaling: float,
    i_colloc: int,
) -> None:
    """
    Apply scaled defect constraints implementing Rule 3.
    """
    num_states = len(state_names)
    print(f"  Applying scaled constraints for {num_states} states:")

    # Apply scaled collocation constraints (Rule 3)
    for i_state in range(num_states):
        state_name = state_names[i_state]
        scale_factor = scaling.get_state_scaling_factor(state_name)

        print(f"    State {i_state} ({state_name}):")
        print(f"      Scale factor: {scale_factor}")

        # Create defect constraint
        defect = (
            state_derivative_at_colloc[i_state, i_colloc]
            - tau_to_time_scaling * state_derivative_rhs_vector[i_state]
        )

        # Scale constraint using corresponding state scale (Rule 3: W_f = V_y)
        scaled_defect = scaling.scale_defect_constraint(state_names[i_state], defect)

        print(f"      Defect type: {type(defect)}")
        print(f"      Is symbolic: {hasattr(defect, 'is_symbolic')}")
        print("      Scaled defect applied to NLP")

        # Apply the scaled constraint
        opti.subject_to(scaled_defect == 0)


def update_scaling_after_mesh_refinement(scaling: Scaling, problem, solution) -> None:
    """
    Update scaling factors after mesh refinement.

    Args:
        scaling: Scaling object
        problem: TrajectoLab Problem
        solution: Solution from last mesh refinement iteration
    """
    if not scaling.enabled:
        return

    # Extract current solution data
    state_guesses = {}
    control_guesses = {}

    # Extract trajectories from solution
    if solution.solved_state_trajectories_per_interval:
        state_guesses = scaling._extract_guesses_from_trajectories(
            solution.solved_state_trajectories_per_interval,
            list(problem._states.keys()),
            len(problem._states),
        )

    if solution.solved_control_trajectories_per_interval:
        control_guesses = scaling._extract_guesses_from_trajectories(
            solution.solved_control_trajectories_per_interval,
            list(problem._controls.keys()),
            len(problem._controls),
        )

    # Recompute scaling with problem bounds and current solution
    scaling.compute_scaling_factors(
        state_names=list(problem._states.keys()),
        state_bounds={
            name: {"lower": info.get("lower"), "upper": info.get("upper")}
            for name, info in problem._states.items()
        },
        state_guesses=state_guesses,
        control_names=list(problem._controls.keys()),
        control_bounds={
            name: {"lower": info.get("lower"), "upper": info.get("upper")}
            for name, info in problem._controls.items()
        },
        control_guesses=control_guesses,
    )
