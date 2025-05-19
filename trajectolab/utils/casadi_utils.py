"""
Utilities for CasADi integration and conversion.

This module provides safe conversion functions between CasADi and NumPy types,
with comprehensive error handling and type validation.
"""

import logging
from typing import Any, TypeAlias, cast

import casadi as ca
import numpy as np

from trajectolab.tl_types import CasadiDM, CasadiFunction, CasadiMX, FloatArray, ProblemParameters


logger = logging.getLogger(__name__)

# Type aliases for better type safety
_CasadiExpression: TypeAlias = CasadiMX | CasadiDM | ca.SX
_CasadiObject: TypeAlias = CasadiMX | CasadiDM | CasadiFunction | ca.SX | Any
_CasadiResult: TypeAlias = CasadiMX | CasadiDM | list[CasadiMX] | tuple[CasadiMX, ...]


class CasadiConversionError(Exception):
    """Raised when CasADi conversion fails."""


def convert_casadi_to_numpy(
    casadi_dynamics_func: CasadiFunction,
    state: FloatArray,
    control: FloatArray,
    time: float,
    params: ProblemParameters,
) -> FloatArray:
    """
    Convert CasADi dynamics function call to NumPy arrays.

    Args:
        casadi_dynamics_func: CasADi dynamics function to evaluate
        state: State vector as NumPy array
        control: Control vector as NumPy array
        time: Time scalar
        params: Problem parameters dictionary

    Returns:
        Dynamics output as NumPy array

    Raises:
        CasadiConversionError: If conversion fails
        ValueError: If inputs have invalid shapes or types
    """
    # Validate inputs
    if not callable(casadi_dynamics_func):
        raise ValueError("casadi_dynamics_func must be callable")

    if not isinstance(state, np.ndarray) or state.dtype != np.float64:
        raise ValueError("state must be float64 NumPy array")

    if not isinstance(control, np.ndarray) or control.dtype != np.float64:
        raise ValueError("control must be float64 NumPy array")

    if not isinstance(time, int | float):
        raise ValueError("time must be numeric scalar")

    try:
        # Convert to CasADi
        state_dm = ca.DM(state)
        control_dm = ca.DM(control)
        time_dm = ca.DM([float(time)])

        # Call dynamics
        result_casadi = casadi_dynamics_func(state_dm, control_dm, time_dm, params)

        # Convert back to NumPy
        if isinstance(result_casadi, ca.DM):
            result_np = np.array(result_casadi.full(), dtype=np.float64).flatten()
        elif isinstance(result_casadi, list | tuple):
            # Handle array of CasADi objects
            if not result_casadi:
                raise CasadiConversionError("Empty result from dynamics function")

            dm_result = ca.DM(len(result_casadi), 1)
            for i, item in enumerate(result_casadi):
                try:
                    dm_result[i] = ca.evalf(item)
                except Exception as e:
                    raise CasadiConversionError(f"Failed to evaluate item {i}: {e}") from e

            result_np = np.array(dm_result.full(), dtype=np.float64).flatten()
        else:
            # Try direct conversion
            try:
                result_np = np.array(result_casadi, dtype=np.float64).flatten()
            except Exception as e:
                raise CasadiConversionError(
                    f"Unsupported result type {type(result_casadi)}: {e}"
                ) from e

        return cast(FloatArray, result_np)

    except CasadiConversionError:
        raise
    except Exception as e:
        raise CasadiConversionError(f"CasADi dynamics evaluation failed: {e}") from e


def validate_casadi_expression(expr: _CasadiObject) -> bool:
    """
    Validate that an expression is a proper CasADi type.

    Args:
        expr: Expression to validate

    Returns:
        True if valid CasADi expression, False otherwise
    """
    return isinstance(expr, ca.MX | ca.DM | ca.SX)


def extract_casadi_value(
    casadi_obj: _CasadiObject, target_shape: tuple[int, int] | None = None
) -> FloatArray:
    """
    Extract numerical value from CasADi object and ensure proper shape.

    Args:
        casadi_obj: CasADi object to extract value from
        target_shape: Optional target shape to reshape to

    Returns:
        NumPy array with extracted values

    Raises:
        CasadiConversionError: If extraction fails
        ValueError: If target_shape is invalid
    """
    if casadi_obj is None:
        raise ValueError("casadi_obj cannot be None")

    try:
        # Convert to numpy array
        if hasattr(casadi_obj, "to_DM"):
            np_array = np.array(casadi_obj.to_DM(), dtype=np.float64)
        elif hasattr(casadi_obj, "full"):
            np_array = np.array(casadi_obj.full(), dtype=np.float64)
        else:
            np_array = np.array(casadi_obj, dtype=np.float64)

        # Reshape if target shape provided
        if target_shape is not None:
            expected_rows, expected_cols = target_shape

            if expected_rows < 0 or expected_cols < 0:
                raise ValueError(f"Invalid target shape: {target_shape}")

            # Handle empty arrays
            if expected_rows == 0:
                return np.empty((0, expected_cols), dtype=np.float64)

            # Ensure 2D shape
            if np_array.ndim == 1:
                if len(np_array) == expected_rows:
                    np_array = np_array.reshape(expected_rows, 1)
                elif len(np_array) == expected_cols:
                    np_array = np_array.reshape(1, expected_cols)
                else:
                    np_array = np_array.reshape(1, -1)

            # Transpose if dimensions are swapped
            if (
                np_array.shape[0] != expected_rows
                and np_array.shape[1] == expected_rows
                and np_array.shape[0] == expected_cols
            ):
                np_array = np_array.T

            # Validate final shape
            if np_array.shape != (expected_rows, expected_cols):
                logger.warning(f"Could not reshape to target {target_shape}, got {np_array.shape}")

        return cast(FloatArray, np_array)

    except Exception as e:
        raise CasadiConversionError(f"Failed to extract CasADi value: {e}") from e
