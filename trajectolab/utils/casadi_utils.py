"""
Utility functions for CasADi integration, conversion, and numerical validation.
FIXED: Fully compatible with optimized dynamics interface (ca.MX return type).
"""

import logging
from typing import cast

import casadi as ca
import numpy as np

from trajectolab.exceptions import DataIntegrityError
from trajectolab.tl_types import FloatArray


logger = logging.getLogger(__name__)


class CasadiConversionError(Exception):
    """Raised when CasADi conversion fails."""


def convert_casadi_to_numpy(
    casadi_dynamics_func: ca.Function,
    state: FloatArray,
    control: FloatArray,
    time: float,
) -> FloatArray:
    """
    Convert CasADi dynamics function call to NumPy arrays with enhanced validation.
    FIXED: Handles both optimized interface (ca.MX) and legacy interface (list[ca.MX]).
    """
    # Guard clause: Validate function
    if not callable(casadi_dynamics_func):
        raise DataIntegrityError(
            "casadi_dynamics_func must be callable", "Invalid function passed to CasADi converter"
        )

    # Guard clause: Validate state input
    if not isinstance(state, np.ndarray) or state.dtype != np.float64:
        raise DataIntegrityError(
            f"state must be float64 NumPy array, got {type(state)} with dtype {getattr(state, 'dtype', 'N/A')}",
            "Invalid state data for CasADi conversion",
        )

    # Critical: Check for NaN/Inf in state
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        raise DataIntegrityError(
            "state array contains NaN or Inf values", "Numerical corruption in state input"
        )

    # Guard clause: Validate control input
    if not isinstance(control, np.ndarray) or control.dtype != np.float64:
        raise DataIntegrityError(
            f"control must be float64 NumPy array, got {type(control)} with dtype {getattr(control, 'dtype', 'N/A')}",
            "Invalid control data for CasADi conversion",
        )

    # Critical: Check for NaN/Inf in control
    if np.any(np.isnan(control)) or np.any(np.isinf(control)):
        raise DataIntegrityError(
            "control array contains NaN or Inf values", "Numerical corruption in control input"
        )

    # Guard clause: Validate time
    if not isinstance(time, int | float):
        raise DataIntegrityError(
            f"time must be numeric scalar, got {type(time)}",
            "Invalid time value for CasADi conversion",
        )

    if np.isnan(time) or np.isinf(time):
        raise DataIntegrityError(
            f"time value is invalid: {time}", "Numerical corruption in time input"
        )

    try:
        # Convert to CasADi
        state_dm = ca.DM(state)
        control_dm = ca.DM(control)
        time_dm = ca.DM([float(time)])

        # Call dynamics
        result_casadi = casadi_dynamics_func(state_dm, control_dm, time_dm)

        # FIXED: Handle both optimized and legacy interfaces
        if isinstance(result_casadi, ca.DM):
            # Direct DM result - convert to numpy
            result_np = np.array(result_casadi.full(), dtype=np.float64).flatten()
        elif isinstance(result_casadi, ca.MX):
            # MX result (new optimized interface) - evaluate and convert
            result_dm = ca.evalf(result_casadi)
            if isinstance(result_dm, ca.DM):
                result_np = np.array(result_dm.full(), dtype=np.float64).flatten()
            else:
                # Fallback: evaluate each element
                num_states = result_casadi.shape[0]
                result_np = np.array(
                    [float(ca.evalf(result_casadi[i])) for i in range(num_states)], dtype=np.float64
                )
        elif isinstance(result_casadi, list | tuple):
            # Legacy list interface - handle array of CasADi objects
            if not result_casadi:
                raise CasadiConversionError("Empty result from dynamics function")

            result_values = []
            for i, item in enumerate(result_casadi):
                try:
                    if isinstance(item, ca.MX):
                        result_values.append(float(ca.evalf(item)))
                    elif isinstance(item, ca.DM):
                        result_values.append(float(item))
                    else:
                        result_values.append(float(item))
                except Exception as e:
                    raise CasadiConversionError(f"Failed to evaluate item {i}: {e}") from e

            result_np = np.array(result_values, dtype=np.float64)
        else:
            # Try direct conversion
            try:
                result_np = np.array(result_casadi, dtype=np.float64).flatten()
            except Exception as e:
                raise CasadiConversionError(
                    f"Unsupported result type {type(result_casadi)}: {e}"
                ) from e

        # Critical: Validate output for NaN/Inf
        if np.any(np.isnan(result_np)) or np.any(np.isinf(result_np)):
            raise DataIntegrityError(
                "CasADi dynamics output contains NaN or Inf values",
                "Numerical corruption in dynamics result",
            )

        return cast(FloatArray, result_np)

    except CasadiConversionError:
        raise
    except DataIntegrityError:
        raise  # Re-raise TrajectoLab-specific errors
    except Exception as e:
        raise CasadiConversionError(f"CasADi dynamics evaluation failed: {e}") from e


def convert_casadi_dynamics_result_to_numpy(
    dynamics_result: ca.MX | list[ca.MX], num_states: int
) -> FloatArray:
    """
    ADDED: Direct conversion helper for dynamics results.

    Converts either the new optimized ca.MX result or legacy list[ca.MX] result
    to a numpy array for use in adaptive algorithms.
    """
    if isinstance(dynamics_result, ca.MX):
        # New optimized interface - ca.MX result
        if dynamics_result.shape[0] == num_states and dynamics_result.shape[1] == 1:
            # Correct shape - convert directly
            state_deriv_np = np.array(
                [float(ca.evalf(dynamics_result[i])) for i in range(num_states)], dtype=np.float64
            )
        elif dynamics_result.shape[0] == 1 and dynamics_result.shape[1] == num_states:
            # Transposed - need to transpose first
            state_deriv_np = np.array(
                [float(ca.evalf(dynamics_result[0, i])) for i in range(num_states)],
                dtype=np.float64,
            )
        elif dynamics_result.shape[0] == num_states:
            # Column vector or row vector - handle both
            state_deriv_np = np.array(
                [float(ca.evalf(dynamics_result[i])) for i in range(num_states)], dtype=np.float64
            )
        else:
            raise ValueError(
                f"Unexpected dynamics result shape: {dynamics_result.shape}, expected ({num_states}, 1) or compatible"
            )
    elif isinstance(dynamics_result, list):
        # Legacy interface - list[ca.MX] (backward compatibility)
        if len(dynamics_result) != num_states:
            raise ValueError(
                f"Dynamics list length {len(dynamics_result)} != expected states {num_states}"
            )
        state_deriv_np = np.array(
            [float(ca.evalf(expr)) for expr in dynamics_result], dtype=np.float64
        )
    else:
        raise ValueError(f"Unsupported dynamics result type: {type(dynamics_result)}")

    # Validate result
    if np.any(np.isnan(state_deriv_np)) or np.any(np.isinf(state_deriv_np)):
        raise DataIntegrityError(
            "Converted dynamics result contains NaN or Inf values",
            "Numerical corruption in dynamics conversion",
        )

    return cast(FloatArray, state_deriv_np)
