"""
Fixed problem-level scaling management for optimal control problems.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..tl_types import FloatArray, InitialGuess
from ..utils.variable_scaling import (
    ProblemScalingInfo,
    VariableScalingInfo,
    compute_scaling_from_bounds,
    create_default_scaling,
    estimate_range_from_initial_guess,
    validate_scaling_info,
)
from .state import VariableState


logger = logging.getLogger(__name__)


def compute_variable_scaling_rule2(
    variable_state: VariableState,
    initial_guess: InitialGuess | None = None,
    enable_scaling: bool = True,
) -> ProblemScalingInfo:
    """
    Compute variable scaling using Rule 2 heuristic.

    Rule 2 implementation:
    1. Estimate largest and smallest value for each variable from:
       a) User-input upper and lower bounds
       b) User-specified initial guesses for variables
    2. Normalize and shift variables using formulas (4.250) and (4.251)
    3. When (a) and (b) provide no information, default scaling to 1

    Args:
        variable_state: Variable state containing variable definitions
        initial_guess: Optional initial guess for fallback range estimation
        enable_scaling: Whether to enable scaling (if False, returns identity scaling)

    Returns:
        Complete scaling information for the problem
    """
    scaling_info = ProblemScalingInfo(scaling_enabled=enable_scaling)

    if not enable_scaling:
        logger.info("Variable scaling disabled by user")
        # Create identity scaling for all variables
        for name in variable_state.states:
            scaling_info.state_scaling[name] = create_default_scaling()
        for name in variable_state.controls:
            scaling_info.control_scaling[name] = create_default_scaling()
        return scaling_info

    logger.info("Computing variable scaling using Rule 2 heuristic")

    # Process state variables
    for name, state_info in variable_state.states.items():
        scaling = _compute_single_variable_scaling(
            name, "state", state_info, initial_guess, variable_state.states
        )
        validate_scaling_info(scaling)
        scaling_info.state_scaling[name] = scaling

    # Process control variables
    for name, control_info in variable_state.controls.items():
        scaling = _compute_single_variable_scaling(
            name, "control", control_info, initial_guess, variable_state.controls
        )
        validate_scaling_info(scaling)
        scaling_info.control_scaling[name] = scaling

    return scaling_info


def _compute_single_variable_scaling(
    var_name: str,
    var_type: str,
    var_info: dict[str, Any],
    initial_guess: InitialGuess | None,
    all_vars: dict[str, dict[str, Any]],
) -> VariableScalingInfo:
    """
    Compute scaling for a single variable using Rule 2.

    Args:
        var_name: Variable name
        var_type: Variable type ("state" or "control")
        var_info: Variable metadata dictionary
        initial_guess: Initial guess (may be None)
        all_vars: All variables of this type (for indexing)

    Returns:
        Scaling information for this variable
    """
    # FIX: Handle both string keys and direct access to bounds
    lower_bound = var_info.get("lower")
    upper_bound = var_info.get("upper")

    # Debug: Print what we found
    logger.debug(f"Processing {var_type} {var_name}: bounds [{lower_bound}, {upper_bound}]")

    # Rule 2, step 1a: Try to use bounds if available
    if lower_bound is not None and upper_bound is not None:
        try:
            scale_weight, shift = compute_scaling_from_bounds(lower_bound, upper_bound)
            logger.debug(
                f"Computed scaling for {var_type} '{var_name}' from bounds "
                f"[{lower_bound}, {upper_bound}]: weight={scale_weight:.3e}, shift={shift:.3f}"
            )
            return VariableScalingInfo(
                scale_weight=scale_weight,
                shift=shift,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scaling_method="bounds",
            )
        except ValueError as e:
            logger.warning(f"Failed to compute scaling from bounds for {var_name}: {e}")

    # Rule 2, step 1b: Try initial guess if bounds not available
    if initial_guess is not None:
        try:
            var_trajectory = _extract_variable_trajectory_from_guess(
                var_name, var_type, initial_guess, all_vars
            )
            if var_trajectory is not None:
                estimated_range = estimate_range_from_initial_guess(var_trajectory)
                scale_weight, shift = compute_scaling_from_bounds(
                    estimated_range[0], estimated_range[1]
                )
                logger.debug(
                    f"Computed scaling for {var_type} '{var_name}' from initial guess "
                    f"range {estimated_range}: weight={scale_weight:.3e}, shift={shift:.3f}"
                )
                return VariableScalingInfo(
                    scale_weight=scale_weight,
                    shift=shift,
                    estimated_range=estimated_range,
                    scaling_method="initial_guess",
                )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to compute scaling from initial guess for {var_name}: {e}")

    # Rule 2, step 3: Default scaling when no information available
    logger.debug(f"Using default scaling for {var_type} '{var_name}' (no bounds or initial guess)")
    return create_default_scaling()


def _extract_variable_trajectory_from_guess(
    var_name: str, var_type: str, initial_guess: InitialGuess, all_vars: dict[str, dict[str, Any]]
) -> FloatArray | None:
    """
    Extract trajectory for a specific variable from initial guess.

    Args:
        var_name: Variable name
        var_type: Variable type ("state" or "control")
        initial_guess: Initial guess containing trajectories
        all_vars: All variables of this type for index lookup

    Returns:
        Flattened trajectory array for the variable, or None if not found
    """
    # Find variable index
    if var_name not in all_vars:
        return None

    var_index = all_vars[var_name]["index"]

    # Extract appropriate trajectories based on variable type
    if var_type == "state":
        trajectories = initial_guess.states
    elif var_type == "control":
        trajectories = initial_guess.controls
    else:
        return None

    if trajectories is None or len(trajectories) == 0:
        return None

    # Concatenate all intervals for this variable
    all_values = []
    for interval_trajectory in trajectories:
        if interval_trajectory.shape[0] > var_index:
            all_values.extend(interval_trajectory[var_index, :].flatten())

    if not all_values:
        return None

    return np.array(all_values, dtype=np.float64)


def apply_scaling_to_initial_guess(
    initial_guess: InitialGuess, scaling_info: ProblemScalingInfo, variable_state: VariableState
) -> InitialGuess:
    """
    Apply scaling transformations to an initial guess.

    Args:
        initial_guess: Original initial guess
        scaling_info: Scaling information for the problem
        variable_state: Variable state for ordering information

    Returns:
        New InitialGuess with scaled values

    Raises:
        ValueError: If initial guess structure is invalid
    """
    if not scaling_info.scaling_enabled:
        return initial_guess

    scaled_guess = InitialGuess(
        initial_time_variable=initial_guess.initial_time_variable,
        terminal_time_variable=initial_guess.terminal_time_variable,
        integrals=initial_guess.integrals,  # Integrals typically don't need scaling
    )

    # Scale state trajectories
    if initial_guess.states is not None:
        scaled_states = []
        state_names = sorted(
            variable_state.states.keys(), key=lambda n: variable_state.states[n]["index"]
        )

        for interval_states in initial_guess.states:
            scaled_interval = np.copy(interval_states)
            for i, name in enumerate(state_names):
                if name in scaling_info.state_scaling:
                    scaling = scaling_info.state_scaling[name]
                    original_values = interval_states[i, :]
                    scaled_values = scaling.scale_weight * original_values + scaling.shift
                    scaled_interval[i, :] = scaled_values
            scaled_states.append(scaled_interval)
        scaled_guess.states = scaled_states

    # Scale control trajectories
    if initial_guess.controls is not None:
        scaled_controls = []
        control_names = sorted(
            variable_state.controls.keys(), key=lambda n: variable_state.controls[n]["index"]
        )

        for interval_controls in initial_guess.controls:
            scaled_interval = np.copy(interval_controls)
            for i, name in enumerate(control_names):
                if name in scaling_info.control_scaling:
                    scaling = scaling_info.control_scaling[name]
                    original_values = interval_controls[i, :]
                    scaled_values = scaling.scale_weight * original_values + scaling.shift
                    scaled_interval[i, :] = scaled_values
            scaled_controls.append(scaled_interval)
        scaled_guess.controls = scaled_controls

    return scaled_guess


def reverse_scaling_on_solution_trajectories(
    scaled_states: list[FloatArray],
    scaled_controls: list[FloatArray],
    scaling_info: ProblemScalingInfo,
    variable_state: VariableState,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """
    Reverse scaling on solution trajectories to get physical values.

    Args:
        scaled_states: Scaled state trajectories
        scaled_controls: Scaled control trajectories
        scaling_info: Scaling information used during optimization
        variable_state: Variable state for ordering information

    Returns:
        Tuple of (physical_states, physical_controls) with original scaling
    """
    if not scaling_info.scaling_enabled:
        return scaled_states, scaled_controls

    # Reverse state scaling
    physical_states = []
    state_names = sorted(
        variable_state.states.keys(), key=lambda n: variable_state.states[n]["index"]
    )

    for scaled_trajectory in scaled_states:
        physical_trajectory = np.copy(scaled_trajectory)
        for i, name in enumerate(state_names):
            if name in scaling_info.state_scaling:
                scaling = scaling_info.state_scaling[name]
                scaled_values = scaled_trajectory[i]
                physical_values = (scaled_values - scaling.shift) / scaling.scale_weight
                physical_trajectory[i] = physical_values
        physical_states.append(physical_trajectory)

    # Reverse control scaling
    physical_controls = []
    control_names = sorted(
        variable_state.controls.keys(), key=lambda n: variable_state.controls[n]["index"]
    )

    for scaled_trajectory in scaled_controls:
        physical_trajectory = np.copy(scaled_trajectory)
        for i, name in enumerate(control_names):
            if name in scaling_info.control_scaling:
                scaling = scaling_info.control_scaling[name]
                scaled_values = scaled_trajectory[i]
                physical_values = (scaled_values - scaling.shift) / scaling.scale_weight
                physical_trajectory[i] = physical_values
        physical_controls.append(scaled_trajectory)

    return physical_states, physical_controls
