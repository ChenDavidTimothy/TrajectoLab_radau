"""
Utilities for CasADi integration and conversion.
"""

from typing import cast

import casadi as ca
import numpy as np

from trajectolab.tl_types import FloatArray, ProblemParameters


def convert_casadi_to_numpy(
    casadi_dynamics_func,
    state: FloatArray,
    control: FloatArray,
    time: float,
    params: ProblemParameters,
) -> FloatArray:
    """
    Convert CasADi dynamics function call to NumPy arrays.

    Args:
        casadi_dynamics_func: CasADi dynamics function
        state: State vector as NumPy array
        control: Control vector as NumPy array
        time: Time scalar
        params: Problem parameters dictionary

    Returns:
        Dynamics output as NumPy array
    """
    # Convert to CasADi
    state_dm = ca.DM(state)
    control_dm = ca.DM(control)
    time_dm = ca.DM([time])

    # Call dynamics
    result_casadi = casadi_dynamics_func(state_dm, control_dm, time_dm, params)

    # Convert back to NumPy
    if isinstance(result_casadi, ca.DM):
        result_np = np.array(result_casadi.full(), dtype=np.float64).flatten()
    else:
        # Handle array of MX objects
        dm_result = ca.DM(len(result_casadi), 1)
        for i, item in enumerate(result_casadi):
            dm_result[i] = ca.evalf(item)
        result_np = np.array(dm_result.full(), dtype=np.float64).flatten()

    return cast(FloatArray, result_np)


def validate_casadi_expression(expr) -> bool:
    """
    Validate that an expression is a proper CasADi type.

    Args:
        expr: Expression to validate

    Returns:
        True if valid CasADi expression
    """
    return isinstance(expr, ca.MX | ca.DM | ca.SX)


def extract_casadi_value(casadi_obj, target_shape: tuple[int, int] | None = None) -> FloatArray:
    """
    Extract numerical value from CasADi object and ensure proper shape.

    Args:
        casadi_obj: CasADi object to extract value from
        target_shape: Optional target shape to reshape to

    Returns:
        NumPy array with extracted values
    """
    # Convert to numpy array
    if hasattr(casadi_obj, "to_DM"):
        np_array = np.array(casadi_obj.to_DM(), dtype=np.float64)
    else:
        np_array = np.array(casadi_obj, dtype=np.float64)

    # Reshape if target shape provided
    if target_shape is not None:
        expected_rows, expected_cols = target_shape

        # Handle empty arrays
        if expected_rows == 0:
            return np.empty((0, expected_cols), dtype=np.float64)

        # Ensure 2D shape
        if np_array.ndim == 1:
            if len(np_array) == expected_rows:
                np_array = np_array.reshape(expected_rows, 1)
            else:
                np_array = np_array.reshape(1, -1)

        # Transpose if dimensions are swapped
        if np_array.shape[0] != expected_rows and np_array.shape[1] == expected_rows:
            np_array = np_array.T

    return cast(FloatArray, np_array)
