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
    """Convert CasADi dynamics function call to NumPy arrays with enhanced validation.

    Handles both optimized interface (ca.MX) and legacy interface (list[ca.MX]) to maintain
    backward compatibility while supporting performance optimizations.

    Args:
        casadi_dynamics_func: CasADi function for dynamics evaluation
        state: Current state vector
        control: Current control vector
        time: Current time

    Returns:
        FloatArray: State derivatives as NumPy array

    Raises:
        DataIntegrityError: Input corruption or numerical issues detected
        CasadiConversionError: CasADi evaluation or conversion failed
    """
    # Validation at external boundary prevents propagation of corrupted data
    if not callable(casadi_dynamics_func):
        raise DataIntegrityError(
            "casadi_dynamics_func must be callable", "Invalid function passed to CasADi converter"
        )

    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        raise DataIntegrityError(
            "state array contains NaN or Inf values", "Numerical corruption in state input"
        )

    if np.any(np.isnan(control)) or np.any(np.isinf(control)):
        raise DataIntegrityError(
            "control array contains NaN or Inf values", "Numerical corruption in control input"
        )

    if np.isnan(time) or np.isinf(time):
        raise DataIntegrityError(
            f"time value is invalid: {time}", "Numerical corruption in time input"
        )

    try:
        state_dm = ca.DM(state)
        control_dm = ca.DM(control)
        time_dm = ca.DM([float(time)])

        result_casadi = casadi_dynamics_func(state_dm, control_dm, time_dm)

        # Handle both optimized (direct MX) and legacy (list) interfaces
        if isinstance(result_casadi, ca.DM):
            result_np = cast(FloatArray, np.array(result_casadi.full(), dtype=np.float64).flatten())
        elif isinstance(result_casadi, ca.MX):
            # Optimized interface - direct MX evaluation
            result_dm = ca.evalf(result_casadi)
            if isinstance(result_dm, ca.DM):
                result_np = cast(FloatArray, np.array(result_dm.full(), dtype=np.float64).flatten())
            else:
                # Fallback for complex MX expressions
                num_states = result_casadi.shape[0]
                result_np = cast(
                    FloatArray,
                    np.array(
                        [float(ca.evalf(result_casadi[i])) for i in range(num_states)],
                        dtype=np.float64,
                    ),
                )
        elif isinstance(result_casadi, list | tuple):
            # Legacy interface - list of CasADi objects
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

            result_np = cast(FloatArray, np.array(result_values, dtype=np.float64))
        else:
            try:
                result_np = cast(FloatArray, np.array(result_casadi, dtype=np.float64).flatten())
            except Exception as e:
                raise CasadiConversionError(
                    f"Unsupported result type {type(result_casadi)}: {e}"
                ) from e

        # Validation at external boundary catches CasADi numerical corruption
        if np.any(np.isnan(result_np)) or np.any(np.isinf(result_np)):
            raise DataIntegrityError(
                "CasADi dynamics output contains NaN or Inf values",
                "Numerical corruption in dynamics result",
            )

        return result_np

    except CasadiConversionError:
        raise
    except DataIntegrityError:
        raise
    except Exception as e:
        raise CasadiConversionError(f"CasADi dynamics evaluation failed: {e}") from e
