"""
scaling.py - Comprehensive fix for automatic scaling in TrajectoLab.

This implements a robust, traceable scaling system for optimal control problems.
"""

import logging
from typing import TypeAlias

import numpy as np

from trajectolab.tl_types import CasadiMX, FloatArray, OptimalControlSolution


# Configure dedicated logger for scaling subsystem
scaling_logger = logging.getLogger("trajectolab.scaling")
# By default, the logger level is WARNING - we'll set it in the Scaling constructor

# Type aliases for enhanced readability
_ScalingDict: TypeAlias = dict[str, tuple[float, float]]  # (scale_factor, shift)


class Scaling:
    """
    Handles variable and constraint scaling for optimal control problems.
    """

    def __init__(self, enabled: bool = True, log_level: int = logging.INFO) -> None:
        """
        Initialize scaling system.

        Args:
            enabled: Whether automatic scaling is enabled (default: True)
            log_level: Logging level for scaling operations (default: INFO)
        """
        self.state_scaling: _ScalingDict = {}
        self.control_scaling: _ScalingDict = {}
        self._enabled: bool = enabled  # Use private field with controlled access

        # Set up logging for this instance
        scaling_logger.setLevel(log_level)
        # Only add handler if none exists to avoid duplicate messages
        if not scaling_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            scaling_logger.addHandler(handler)

        # Log initialization
        scaling_logger.info(f"Scaling system initialized with enabled={self._enabled}")

    @property
    def enabled(self) -> bool:
        """Get whether scaling is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set whether scaling is enabled."""
        self._enabled = bool(value)
        scaling_logger.info(f"Scaling enabled set to: {self._enabled}")

        # When disabling scaling, reset factors to neutral values
        if not self._enabled:
            self._reset_scaling_factors()

    def _reset_scaling_factors(self) -> None:
        """Reset all scaling factors to neutral values (1.0, 0.0)."""
        old_state_count = len(self.state_scaling)
        old_control_count = len(self.control_scaling)

        self.state_scaling = {}
        self.control_scaling = {}

        scaling_logger.info(
            f"Reset {old_state_count} state and {old_control_count} control scaling factors to defaults"
        )

    def compute_from_problem(self, problem) -> None:
        """
        Compute scaling factors from a Problem object.

        Args:
            problem: TrajectoLab Problem instance
        """
        if not self._enabled:
            scaling_logger.info("Scaling is disabled - skipping factor computation")
            return

        scaling_logger.info(f"Computing scaling factors from problem '{problem.name}'")

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
        if not self._enabled:
            scaling_logger.info("Scaling is disabled - skipping factor computation")
            return

        scaling_logger.info("Computing scaling factors")

        # Clear existing scaling dictionaries to avoid stale values
        self.state_scaling = {}
        self.control_scaling = {}

        # Process state variables
        scaling_logger.info("Computing state variable scaling factors:")
        for name in state_names:
            bounds = state_bounds.get(name, {})
            lower = bounds.get("lower")
            upper = bounds.get("upper")

            scaling_logger.debug(f"  State {name}: bounds=[{lower}, {upper}]")

            # Use bounds if available (Rule 2a)
            if lower is not None and upper is not None and lower < upper:
                scale_factor = 1.0 / (upper - lower)  # Equation (4.250)
                shift = 0.5 - upper / (upper - lower)  # Equation (4.251)
                scaling_logger.info(f"  State {name}: Using bounds for scaling (Rule 2a)")
                scaling_logger.info(
                    f"    Range: [{lower}, {upper}], factor={scale_factor:.6g}, shift={shift:.6g}"
                )

            # Otherwise use initial guess to estimate range (Rule 2b)
            elif state_guesses and name in state_guesses and len(state_guesses[name]) > 0:
                guess_array = state_guesses[name]
                min_val = float(np.min(guess_array))
                max_val = float(np.max(guess_array))

                scaling_logger.info(f"  State {name}: Using initial guess for scaling (Rule 2b)")
                scaling_logger.info(f"    Guess range: [{min_val:.6g}, {max_val:.6g}]")

                if np.isclose(min_val, max_val, rtol=1e-12, atol=1e-14):
                    # Handle case where all values are nearly identical
                    if np.abs(min_val) < 1e-10:
                        scale_factor = 1.0
                        shift = 0.0
                        scaling_logger.info(
                            "    Near zero values, using default scaling (1.0, 0.0)"
                        )
                    else:
                        # Use magnitude of value as scale factor with safety margin
                        scale_factor = 1.0 / max(1.0, 10.0 * np.abs(min_val))
                        shift = 0.0
                        scaling_logger.info(
                            f"    Using magnitude-based scaling: factor={scale_factor:.6g}"
                        )
                else:
                    # Use range from guess values with safety margin for numerical stability
                    range_margin = 0.1 * (max_val - min_val)  # Add 10% margin
                    safe_min = min_val - range_margin
                    safe_max = max_val + range_margin
                    scale_factor = 1.0 / (safe_max - safe_min)
                    shift = 0.5 - safe_max / (safe_max - safe_min)
                    scaling_logger.info(
                        f"    Calculated with 10% margin: factor={scale_factor:.6g}, shift={shift:.6g}"
                    )
            else:
                # Default scaling when no information is available (Rule 2 default)
                scale_factor = 1.0
                shift = 0.0
                scaling_logger.info(
                    f"  State {name}: No bounds or guess available, using default (1.0, 0.0)"
                )

            # Store state scaling factors with validation
            self.state_scaling[name] = (
                float(scale_factor) if np.isfinite(scale_factor) else 1.0,
                float(shift) if np.isfinite(shift) else 0.0,
            )

            # Compare to manual scaling if this is altitude or velocity (for debugging)
            if name in ("h", "h_scaled"):
                scaling_logger.info(
                    f"    NOTE: Altitude scaling - manual h_scale=1e5, auto={1.0 / scale_factor:.6g}"
                )
            elif name in ("v", "v_scaled"):
                scaling_logger.info(
                    f"    NOTE: Velocity scaling - manual v_scale=1e4, auto={1.0 / scale_factor:.6g}"
                )

        # Process control variables (similar logic as states)
        if control_names:
            scaling_logger.info("Computing control variable scaling factors:")
            for name in control_names:
                bounds = control_bounds.get(name, {}) if control_bounds else {}
                lower = bounds.get("lower")
                upper = bounds.get("upper")

                scaling_logger.debug(f"  Control {name}: bounds=[{lower}, {upper}]")

                if lower is not None and upper is not None and lower < upper:
                    scale_factor = 1.0 / (upper - lower)
                    shift = 0.5 - upper / (upper - lower)
                    scaling_logger.info(
                        f"  Control {name}: Using bounds, factor={scale_factor:.6g}, shift={shift:.6g}"
                    )
                elif control_guesses and name in control_guesses and len(control_guesses[name]) > 0:
                    guess_array = control_guesses[name]
                    min_val = float(np.min(guess_array))
                    max_val = float(np.max(guess_array))

                    scaling_logger.info(
                        f"  Control {name}: Using guess, range=[{min_val:.6g}, {max_val:.6g}]"
                    )

                    if np.isclose(min_val, max_val, rtol=1e-12, atol=1e-14):
                        if np.abs(min_val) < 1e-10:
                            scale_factor = 1.0
                            shift = 0.0
                            scaling_logger.info("    Near zero values, using default (1.0, 0.0)")
                        else:
                            scale_factor = 1.0 / max(1.0, 10.0 * np.abs(min_val))
                            shift = 0.0
                            scaling_logger.info(
                                f"    Using magnitude-based: factor={scale_factor:.6g}"
                            )
                    else:
                        # Use range with safety margin
                        range_margin = 0.1 * (max_val - min_val)
                        safe_min = min_val - range_margin
                        safe_max = max_val + range_margin
                        scale_factor = 1.0 / (safe_max - safe_min)
                        shift = 0.5 - safe_max / (safe_max - safe_min)
                        scaling_logger.info(
                            f"    With 10% margin: factor={scale_factor:.6g}, shift={shift:.6g}"
                        )
                else:
                    scale_factor = 1.0
                    shift = 0.0
                    scaling_logger.info(
                        f"  Control {name}: No data available, using default (1.0, 0.0)"
                    )

                # Store control scaling factors with validation
                self.control_scaling[name] = (
                    float(scale_factor) if np.isfinite(scale_factor) else 1.0,
                    float(shift) if np.isfinite(shift) else 0.0,
                )

    def get_state_scaling(self, name: str) -> tuple[float, float]:
        """
        Get scaling factor and shift for a state variable.

        Args:
            name: State variable name

        Returns:
            Tuple of (scale_factor, shift), defaults to (1.0, 0.0) if not found or scaling disabled
        """
        if not self._enabled:
            return (1.0, 0.0)

        if name not in self.state_scaling:
            scaling_logger.warning(
                f"No scaling factor found for state '{name}', using default (1.0, 0.0)"
            )
            return (1.0, 0.0)

        return self.state_scaling[name]

    def get_control_scaling(self, name: str) -> tuple[float, float]:
        """
        Get scaling factor and shift for a control variable.

        Args:
            name: Control variable name

        Returns:
            Tuple of (scale_factor, shift), defaults to (1.0, 0.0) if not found or scaling disabled
        """
        if not self._enabled:
            return (1.0, 0.0)

        if name not in self.control_scaling:
            scaling_logger.warning(
                f"No scaling factor found for control '{name}', using default (1.0, 0.0)"
            )
            return (1.0, 0.0)

        return self.control_scaling[name]

    def get_state_scaling_factor(self, name: str) -> float:
        """
        Get just the scaling factor for a state variable (for Rule 3).

        Args:
            name: State variable name

        Returns:
            Scaling factor, defaults to 1.0 if not found or scaling disabled
        """
        factor, _ = self.get_state_scaling(name)
        return factor

    def scale_defect_constraint(self, state_name: str, constraint: CasadiMX) -> CasadiMX:
        """
        Scale a defect constraint according to Rule 3.

        Args:
            state_name: Name of the state variable
            constraint: Constraint expression to scale

        Returns:
            Scaled constraint (or original if scaling disabled)
        """
        if not self._enabled:
            return constraint

        # Apply scaling to constraint (Rule 3: W_f = V_y)
        factor = self.get_state_scaling_factor(state_name)
        scaling_logger.debug(
            f"Scaling defect constraint for state '{state_name}' with factor {factor:.6g}"
        )
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

    Args:
        opti: CasADi optimization object
        scaling: Scaling object
        state_names: List of state variable names
        state_derivative_at_colloc: State derivatives at collocation nodes
        state_derivative_rhs_vector: RHS of dynamics equation
        tau_to_time_scaling: Time scaling factor
        i_colloc: Collocation point index
    """
    if not scaling.enabled:
        # Apply unscaled constraint directly
        opti.subject_to(
            state_derivative_at_colloc[:, i_colloc]
            == tau_to_time_scaling * state_derivative_rhs_vector
        )
        scaling_logger.debug(f"Applied unscaled defect constraint at collocation point {i_colloc}")
        return

    num_states = len(state_names)
    scaling_logger.debug(
        f"Applying scaled defect constraints for {num_states} states at collocation point {i_colloc}"
    )

    # Apply scaled collocation constraints
    for i_state in range(num_states):
        if i_state >= len(state_names):
            scaling_logger.warning(
                f"State index {i_state} out of bounds for state_names (len={len(state_names)})"
            )
            continue

        state_name = state_names[i_state]
        factor = scaling.get_state_scaling_factor(state_name)

        # Scale dynamics output to match scaled state derivative
        scaled_dynamics_output = factor * state_derivative_rhs_vector[i_state]

        # Apply constraint with consistent units
        opti.subject_to(
            state_derivative_at_colloc[i_state, i_colloc]
            == tau_to_time_scaling * scaled_dynamics_output
        )

        scaling_logger.debug(
            f"Applied scaled defect constraint for state '{state_name}' with factor {factor:.6g}"
        )


def update_scaling_after_mesh_refinement(scaling: Scaling, problem, solution) -> None:
    """
    Update scaling factors after mesh refinement.

    Args:
        scaling: Scaling object
        problem: TrajectoLab Problem
        solution: Solution from last mesh refinement iteration
    """
    if not scaling.enabled:
        scaling_logger.info("Scaling disabled - skipping post-refinement update")
        return

    scaling_logger.info("Updating scaling factors after mesh refinement")

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

    scaling_logger.info("Scaling factors updated after mesh refinement")


def unscale_solution_values(
    solution: OptimalControlSolution, state_names: list[str], control_names: list[str]
) -> None:
    """
    Apply unscaling to solution values.

    Args:
        solution: Solution object to unscale
        state_names: List of state variable names
        control_names: List of control variable names
    """
    # Configure local logger
    logger = logging.getLogger("trajectolab.scaling.unscale")

    # Check if scaling was used
    if not hasattr(solution, "was_scaled") or not solution.was_scaled:
        logger.info("Solution was not scaled, skipping unscaling")
        return

    # Check if scaling information exists
    if not hasattr(solution, "state_scaling") or not hasattr(solution, "control_scaling"):
        logger.warning("Missing scaling information - cannot unscale")
        return

    # Unscale state trajectories
    logger.info("Unscaling state trajectories")
    if solution.states:
        for i, name in enumerate(state_names):
            if i < len(solution.states) and solution.states[i].size > 0:
                if name in solution.state_scaling:
                    scale_factor, shift = solution.state_scaling[name]
                    logger.info(f"  State {name}: factor={scale_factor:.6g}, shift={shift:.6g}")

                    # Check for division by zero
                    if abs(scale_factor) < 1e-12:
                        logger.warning(f"  State {name}: Near-zero scale factor, using 1.0")
                        scale_factor = 1.0

                    # Sample values before unscaling
                    if solution.states[i].size > 0:
                        before_min = float(np.min(solution.states[i]))
                        before_max = float(np.max(solution.states[i]))
                        logger.info(
                            f"    Before unscaling: range=[{before_min:.6g}, {before_max:.6g}]"
                        )

                    # Unscale: x = (x_scaled - shift) / scale_factor
                    solution.states[i] = (solution.states[i] - shift) / scale_factor

                    # Sample values after unscaling
                    if solution.states[i].size > 0:
                        after_min = float(np.min(solution.states[i]))
                        after_max = float(np.max(solution.states[i]))
                        logger.info(
                            f"    After unscaling: range=[{after_min:.6g}, {after_max:.6g}]"
                        )

                    # Special comparison for h and v
                    if name in ("h", "h_scaled"):
                        logger.info(
                            f"    COMPARISON: Manual h_scale=1e5, auto range=[{after_min:.6g}, {after_max:.6g}]"
                        )
                    elif name in ("v", "v_scaled"):
                        logger.info(
                            f"    COMPARISON: Manual v_scale=1e4, auto range=[{after_min:.6g}, {after_max:.6g}]"
                        )
                else:
                    logger.warning(f"  No scaling info available for state {name}")
            else:
                logger.debug(f"  No trajectory data available for state index {i}")

    # Unscale control trajectories
    logger.info("Unscaling control trajectories")
    if solution.controls:
        for i, name in enumerate(control_names):
            if i < len(solution.controls) and solution.controls[i].size > 0:
                if name in solution.control_scaling:
                    scale_factor, shift = solution.control_scaling[name]
                    logger.info(f"  Control {name}: factor={scale_factor:.6g}, shift={shift:.6g}")

                    # Check for division by zero
                    if abs(scale_factor) < 1e-12:
                        logger.warning(f"  Control {name}: Near-zero scale factor, using 1.0")
                        scale_factor = 1.0

                    # Unscale: u = (u_scaled - shift) / scale_factor
                    solution.controls[i] = (solution.controls[i] - shift) / scale_factor
                    logger.info(f"  Control {name}: Successfully unscaled")
                else:
                    logger.warning(f"  No scaling info available for control {name}")
            else:
                logger.debug(f"  No trajectory data available for control index {i}")

    # Unscale per-interval trajectories if they exist
    logger.info("Unscaling per-interval trajectories")
    if (
        hasattr(solution, "solved_state_trajectories_per_interval")
        and solution.solved_state_trajectories_per_interval
    ):
        for interval_idx, interval_state_data in enumerate(
            solution.solved_state_trajectories_per_interval
        ):
            for i, name in enumerate(state_names):
                if name in solution.state_scaling:
                    scale_factor, shift = solution.state_scaling[name]

                    # Check for division by zero
                    if abs(scale_factor) < 1e-12:
                        logger.warning(
                            f"  State {name} (interval {interval_idx}): Near-zero scale factor, using 1.0"
                        )
                        scale_factor = 1.0

                    # Handle different array dimensions
                    if interval_state_data.ndim == 1:
                        # For 1D arrays, make sure we're within bounds
                        if i == 0:  # Can only unscale if this is the first and only state
                            solution.solved_state_trajectories_per_interval[interval_idx] = (
                                interval_state_data - shift
                            ) / scale_factor
                            logger.debug(f"  Unscaled 1D state data for interval {interval_idx}")
                    elif interval_state_data.ndim == 2 and i < interval_state_data.shape[0]:
                        # For 2D arrays with proper shape
                        solution.solved_state_trajectories_per_interval[interval_idx][i, :] = (
                            interval_state_data[i, :] - shift
                        ) / scale_factor
                        logger.debug(f"  Unscaled state {name} for interval {interval_idx}")

    if (
        hasattr(solution, "solved_control_trajectories_per_interval")
        and solution.solved_control_trajectories_per_interval
    ):
        for interval_idx, interval_control_data in enumerate(
            solution.solved_control_trajectories_per_interval
        ):
            for i, name in enumerate(control_names):
                if name in solution.control_scaling:
                    scale_factor, shift = solution.control_scaling[name]

                    # Check for division by zero
                    if abs(scale_factor) < 1e-12:
                        logger.warning(
                            f"  Control {name} (interval {interval_idx}): Near-zero scale factor, using 1.0"
                        )
                        scale_factor = 1.0

                    # Handle different array dimensions
                    if interval_control_data.ndim == 1:
                        # For 1D arrays, make sure we're within bounds
                        if i == 0:  # Can only unscale if this is the first and only control
                            solution.solved_control_trajectories_per_interval[interval_idx] = (
                                interval_control_data - shift
                            ) / scale_factor
                            logger.debug(f"  Unscaled 1D control data for interval {interval_idx}")
                    elif interval_control_data.ndim == 2 and i < interval_control_data.shape[0]:
                        # For 2D arrays with proper shape
                        solution.solved_control_trajectories_per_interval[interval_idx][i, :] = (
                            interval_control_data[i, :] - shift
                        ) / scale_factor
                        logger.debug(f"  Unscaled control {name} for interval {interval_idx}")

    logger.info("Solution unscaling complete")
