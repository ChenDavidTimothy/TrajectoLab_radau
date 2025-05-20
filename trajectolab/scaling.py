"""
Scaling utilities for optimal control problems based on Rules 1-3 from scale.txt.

Implements variable scaling and matching defect constraint scaling to improve conditioning.
"""

import logging
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from trajectolab.tl_types import FloatArray, FloatMatrix, ProblemProtocol


_ScaleArray: TypeAlias = NDArray[np.float64]
ZERO_TOLERANCE: float = 1e-12

logger = logging.getLogger(__name__)


class ScalingManager:
    """
    Manages scaling for optimal control problems.

    Implements Rules 1-3 from scale.txt:
    - Rule 1: Consistent scaling across grid points
    - Rule 2: Variable scaling based on bounds or initial guess
    - Rule 3: ODE defect scaling equals state variable scaling (W_f = V_y)
    """

    def __init__(self, problem: ProblemProtocol) -> None:
        """Initialize scaling manager with problem data."""
        # Extract problem dimensions
        self.num_states = len(problem._states)
        self.num_controls = len(problem._controls)

        # Initialize scaling arrays (V_y, V_u, r_y, r_u in equation 4.243)
        self.state_scales = np.ones(self.num_states, dtype=np.float64)
        self.state_shifts = np.zeros(self.num_states, dtype=np.float64)
        self.control_scales = np.ones(self.num_controls, dtype=np.float64)
        self.control_shifts = np.zeros(self.num_controls, dtype=np.float64)

        # Defect scaling array (W_f in equation 4.244)
        self.defect_scales = np.ones(self.num_states, dtype=np.float64)

        # Flag indicating if scaling has been computed
        self.is_initialized = False

    def compute_scaling(self, problem: ProblemProtocol, initial_guess=None) -> None:
        """
        Compute variable and defect scaling factors based on the problem definition.

        Implements Rules 1-3 from scale.txt.

        Args:
            problem: The optimal control problem
            initial_guess: Optional initial guess to use for scaling estimation
        """
        logger.info("Computing variable and defect scaling (Rules 1-3)")

        # Rule 2: Variable scaling based on bounds or initial guess
        self._compute_variable_scaling(problem, initial_guess)

        # Rule 3: ODE defect scaling (set equal to state variable scaling)
        self._compute_defect_scaling()

        self.is_initialized = True

        # Log scaling information
        self._log_scaling_info()

    def _log_scaling_info(self) -> None:
        """Log information about scaling factors."""
        logger.info("Scaling factors computed:")
        logger.info(f"  State scales: {self.state_scales}")
        logger.info(f"  State shifts: {self.state_shifts}")
        logger.info(f"  Control scales: {self.control_scales}")
        logger.info(f"  Control shifts: {self.control_shifts}")
        logger.info(f"  Defect scales: {self.defect_scales}")

    def _compute_variable_scaling(self, problem: ProblemProtocol, initial_guess=None) -> None:
        """
        Compute variable scaling based on bounds or initial guess (Rule 2).

        Uses equations (4.250) and (4.251) to normalize variables to [-0.5, +0.5].
        """
        logger.info("Computing variable scaling (Rule 2)")

        # Scale state variables
        for i, (name, props) in enumerate(problem._states.items()):
            lower = props.get("lower")
            upper = props.get("upper")

            if lower is not None and upper is not None and abs(upper - lower) > ZERO_TOLERANCE:
                # Scale based on bounds using equation (4.250) and (4.251)
                self.state_scales[i] = 1.0 / (upper - lower)
                self.state_shifts[i] = 0.5 - upper * self.state_scales[i]
                logger.info(
                    f"  State {name}: Using bounds [{lower}, {upper}] → scale={self.state_scales[i]:.6e}, shift={self.state_shifts[i]:.6e}"
                )
            elif (
                initial_guess is not None
                and hasattr(initial_guess, "states")
                and initial_guess.states is not None
            ):
                # Scale based on initial guess if available
                try:
                    # Estimate bounds from initial guess
                    max_val = -float("inf")
                    min_val = float("inf")
                    for interval_states in initial_guess.states:
                        if i < interval_states.shape[0]:
                            max_val = max(max_val, np.max(interval_states[i]))
                            min_val = min(min_val, np.min(interval_states[i]))

                    if max_val > min_val and abs(max_val - min_val) > ZERO_TOLERANCE:
                        # Apply small padding to estimated bounds to ensure variable stays within range
                        padding = 0.1 * (max_val - min_val)
                        padded_min = min_val - padding
                        padded_max = max_val + padding

                        self.state_scales[i] = 1.0 / (padded_max - padded_min)
                        self.state_shifts[i] = 0.5 - padded_max * self.state_scales[i]
                        logger.info(
                            f"  State {name}: Using initial guess range [{min_val:.6e}, {max_val:.6e}] (padded) → scale={self.state_scales[i]:.6e}, shift={self.state_shifts[i]:.6e}"
                        )
                except (IndexError, AttributeError, ValueError) as e:
                    logger.warning(
                        f"  State {name}: Error extracting bounds from initial guess: {e}"
                    )
            else:
                logger.info(
                    f"  State {name}: No bounds or initial guess, using default scale=1.0, shift=0.0"
                )

        # Scale control variables (similar approach)
        for i, (name, props) in enumerate(problem._controls.items()):
            lower = props.get("lower")
            upper = props.get("upper")

            if lower is not None and upper is not None and abs(upper - lower) > ZERO_TOLERANCE:
                # Scale based on bounds using equation (4.250) and (4.251)
                self.control_scales[i] = 1.0 / (upper - lower)
                self.control_shifts[i] = 0.5 - upper * self.control_scales[i]
                logger.info(
                    f"  Control {name}: Using bounds [{lower}, {upper}] → scale={self.control_scales[i]:.6e}, shift={self.control_shifts[i]:.6e}"
                )
            elif (
                initial_guess is not None
                and hasattr(initial_guess, "controls")
                and initial_guess.controls is not None
            ):
                # Scale based on initial guess if available
                try:
                    max_val = -float("inf")
                    min_val = float("inf")
                    for interval_controls in initial_guess.controls:
                        if i < interval_controls.shape[0]:
                            max_val = max(max_val, np.max(interval_controls[i]))
                            min_val = min(min_val, np.min(interval_controls[i]))

                    if max_val > min_val and abs(max_val - min_val) > ZERO_TOLERANCE:
                        # Apply small padding to estimated bounds
                        padding = 0.1 * (max_val - min_val)
                        padded_min = min_val - padding
                        padded_max = max_val + padding

                        self.control_scales[i] = 1.0 / (padded_max - padded_min)
                        self.control_shifts[i] = 0.5 - padded_max * self.control_scales[i]
                        logger.info(
                            f"  Control {name}: Using initial guess range [{min_val:.6e}, {max_val:.6e}] (padded) → scale={self.control_scales[i]:.6e}, shift={self.control_shifts[i]:.6e}"
                        )
                except (IndexError, AttributeError, ValueError) as e:
                    logger.warning(
                        f"  Control {name}: Error extracting bounds from initial guess: {e}"
                    )
            else:
                logger.info(
                    f"  Control {name}: No bounds or initial guess, using default scale=1.0, shift=0.0"
                )

    def _compute_defect_scaling(self) -> None:
        """
        Compute ODE defect scaling (Rule 3).

        Uses equation (4.248) to set ODE defect scaling equal to state variable scaling.
        """
        logger.info("Computing defect scaling (Rule 3)")

        # Set defect scaling equal to state scaling (Equation 4.248)
        self.defect_scales = self.state_scales.copy()
        logger.info(
            f"  Set defect scales = state scales per equation (4.248): {self.defect_scales}"
        )

    def scale_state(self, state: FloatMatrix) -> FloatMatrix:
        """
        Scale a state matrix using equation (4.243): ỹ = Vy·y + ry.

        Args:
            state: State matrix with shape (num_states, num_points)

        Returns:
            Scaled state matrix
        """
        if not self.is_initialized:
            return state  # Return unscaled if not initialized

        return self.state_scales.reshape(-1, 1) * state + self.state_shifts.reshape(-1, 1)

    def unscale_state(self, scaled_state: FloatMatrix) -> FloatMatrix:
        """
        Unscale a state matrix by inverting equation (4.243).

        Args:
            scaled_state: Scaled state matrix with shape (num_states, num_points)

        Returns:
            Original state matrix
        """
        if not self.is_initialized:
            return scaled_state  # Return as-is if not initialized

        return (scaled_state - self.state_shifts.reshape(-1, 1)) / self.state_scales.reshape(-1, 1)

    def scale_control(self, control: FloatMatrix) -> FloatMatrix:
        """
        Scale a control matrix using equation (4.243): ũ = Vu·u + ru.

        Args:
            control: Control matrix with shape (num_controls, num_points)

        Returns:
            Scaled control matrix
        """
        if not self.is_initialized:
            return control  # Return unscaled if not initialized

        return self.control_scales.reshape(-1, 1) * control + self.control_shifts.reshape(-1, 1)

    def unscale_control(self, scaled_control: FloatMatrix) -> FloatMatrix:
        """
        Unscale a control matrix by inverting equation (4.243).

        Args:
            scaled_control: Scaled control matrix with shape (num_controls, num_points)

        Returns:
            Original control matrix
        """
        if not self.is_initialized:
            return scaled_control  # Return as-is if not initialized

        return (scaled_control - self.control_shifts.reshape(-1, 1)) / self.control_scales.reshape(
            -1, 1
        )

    def scale_defect(self, defect: FloatMatrix) -> FloatMatrix:
        """
        Scale defect constraints using equation (4.244): ζ̃ = Wf·ζ.

        Args:
            defect: Defect constraints with shape (num_states, num_points)

        Returns:
            Scaled defect constraints
        """
        if not self.is_initialized:
            return defect  # Return unscaled if not initialized

        return self.defect_scales.reshape(-1, 1) * defect

    def scale_vector(self, vector: FloatArray, is_state: bool = True) -> FloatArray:
        """
        Scale a state or control vector.

        Args:
            vector: Vector to scale
            is_state: Whether this is a state vector (True) or control vector (False)

        Returns:
            Scaled vector
        """
        if not self.is_initialized:
            return vector

        if is_state:
            return self.state_scales * vector + self.state_shifts
        else:
            return self.control_scales * vector + self.control_shifts

    def unscale_vector(self, scaled_vector: FloatArray, is_state: bool = True) -> FloatArray:
        """
        Unscale a state or control vector.

        Args:
            scaled_vector: Scaled vector
            is_state: Whether this is a state vector (True) or control vector (False)

        Returns:
            Original vector
        """
        if not self.is_initialized:
            return scaled_vector

        if is_state:
            return (scaled_vector - self.state_shifts) / self.state_scales
        else:
            return (scaled_vector - self.control_shifts) / self.control_scales
