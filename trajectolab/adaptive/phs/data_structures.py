"""
Core data structures for the PHS adaptive algorithm.
"""

from dataclasses import dataclass
from typing import Any, cast

import casadi as ca
import numpy as np

from trajectolab.tl_types import _FloatArray, _FloatMatrix

__all__ = [
    "AdaptiveParameters",
    "IntervalSimulationBundle",
    "PRefineResult",
    "HRefineResult",
    "PReduceResult",
    "NumPyDynamicsAdapter",
]


@dataclass
class AdaptiveParameters:
    """Parameters controlling the adaptive mesh refinement algorithm."""

    error_tolerance: float
    max_iterations: int
    min_polynomial_degree: int
    max_polynomial_degree: int
    ode_solver_tolerance: float = 1e-7
    num_error_sim_points: int = 50


@dataclass
class IntervalSimulationBundle:
    """Holds results from forward/backward simulations for error estimation."""

    forward_simulation_local_tau_evaluation_points: _FloatArray | None = None
    state_trajectory_from_forward_simulation: _FloatMatrix | None = None
    nlp_state_trajectory_evaluated_at_forward_simulation_points: _FloatMatrix | None = None
    backward_simulation_local_tau_evaluation_points: _FloatArray | None = None
    state_trajectory_from_backward_simulation: _FloatMatrix | None = None
    nlp_state_trajectory_evaluated_at_backward_simulation_points: _FloatMatrix | None = None
    are_forward_and_backward_simulations_successful: bool = True

    def __post_init__(self) -> None:
        """Ensure all ndarray fields are properly formatted as 2D arrays."""
        for field_name, field_val in self.__dict__.items():
            if not isinstance(field_val, np.ndarray) or field_val is None:
                continue
            # Ensure 2D arrays
            if field_val.ndim == 1:
                setattr(self, field_name, field_val.reshape(1, -1))
            elif field_val.ndim > 2:
                raise ValueError(
                    f"Field {field_name} must be 1D or 2D array, got {field_val.ndim}D."
                )


@dataclass
class PRefineResult:
    """Result of polynomial degree refinement."""

    actual_Nk_to_use: int
    was_p_successful: bool
    unconstrained_target_Nk: int


@dataclass
class HRefineResult:
    """Result of h-refinement."""

    collocation_nodes_for_new_subintervals: list[int]
    num_new_subintervals: int


@dataclass
class PReduceResult:
    """Result of p-reduction."""

    new_num_collocation_nodes: int
    was_reduction_applied: bool


class NumPyDynamicsAdapter:
    """Adapter to use CasADi dynamics with NumPy arrays for simulation."""

    def __init__(self, casadi_dynamics_func: Any, problem_parameters: dict[str, Any]):
        self.casadi_dynamics_func = casadi_dynamics_func
        self.problem_parameters = problem_parameters

    def __call__(self, state: _FloatArray, control: _FloatArray, time: float) -> _FloatArray:
        """Convert NumPy arrays to CasADi, call dynamics, convert back to NumPy."""
        # Convert inputs to CasADi
        state_dm = ca.DM(state)
        control_dm = ca.DM(control)
        time_dm = ca.DM([time])

        # Call CasADi dynamics function
        result_casadi = self.casadi_dynamics_func(
            state_dm, control_dm, time_dm, self.problem_parameters
        )

        # Convert result back to NumPy
        if isinstance(result_casadi, ca.DM):
            result_np = np.array(result_casadi.full(), dtype=np.float64).flatten()
        else:
            # Handle array of MX objects
            dm_result = ca.DM(len(result_casadi), 1)
            for i, item in enumerate(result_casadi):
                dm_result[i] = ca.evalf(item)
            result_np = np.array(dm_result.full(), dtype=np.float64).flatten()

        return cast(_FloatArray, result_np)


def _extract_and_prepare_array(
    casadi_value: Any, expected_rows: int, expected_cols: int
) -> _FloatMatrix:
    """Extracts numerical value from CasADi and ensures correct 2D shape."""
    # Convert to numpy array
    if hasattr(casadi_value, "to_DM"):
        np_array = np.array(casadi_value.to_DM(), dtype=np.float64)
    else:
        np_array = np.array(casadi_value, dtype=np.float64)

    # Handle empty arrays for states/controls
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

    # Handle dimension mismatch as best we can
    if np_array.shape != (expected_rows, expected_cols):
        squeezed = np_array.squeeze()
        if squeezed.ndim == 1 and expected_rows == 1 and len(squeezed) == expected_cols:
            np_array = squeezed.reshape(1, expected_cols)

    return np_array
