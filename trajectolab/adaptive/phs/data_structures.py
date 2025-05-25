"""
Data structures and parameter containers for the PHS adaptive algorithm.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from trajectolab.tl_types import FloatArray, ODESolverCallable


__all__ = [
    "AdaptiveParameters",
    "HRefineResult",
    "IntervalSimulationBundle",
    "PReduceResult",
    "PRefineResult",
    "extract_and_prepare_array",
]

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParameters:
    """Parameters controlling the adaptive mesh refinement algorithm."""

    error_tolerance: float
    max_iterations: int
    min_polynomial_degree: int
    max_polynomial_degree: int
    ode_solver_tolerance: float = 1e-7
    num_error_sim_points: int = 50
    ode_method: str = "RK45"
    ode_max_step: float | None = None
    ode_solver: ODESolverCallable | None = None

    def get_ode_solver(self) -> ODESolverCallable:
        """Get the configured ODE solver function."""
        if self.ode_solver is not None:
            return self.ode_solver

        # Create solver with user's simple parameters
        from scipy.integrate import solve_ivp

        def configured_solver(fun, t_span, y0, t_eval=None, **kwargs):
            kwargs.setdefault("method", self.ode_method)
            kwargs.setdefault("rtol", self.ode_solver_tolerance)
            kwargs.setdefault("atol", self.ode_solver_tolerance * 1e-2)

            if self.ode_max_step is not None:
                kwargs.setdefault("max_step", self.ode_max_step)

            return solve_ivp(fun, t_span, y0, t_eval=t_eval, **kwargs)

        return configured_solver


@dataclass
class IntervalSimulationBundle:
    """Holds results from forward/backward simulations for error estimation."""

    forward_simulation_local_tau_evaluation_points: FloatArray | None = None
    state_trajectory_from_forward_simulation: FloatArray | None = None
    nlp_state_trajectory_evaluated_at_forward_simulation_points: FloatArray | None = None
    backward_simulation_local_tau_evaluation_points: FloatArray | None = None
    state_trajectory_from_backward_simulation: FloatArray | None = None
    nlp_state_trajectory_evaluated_at_backward_simulation_points: FloatArray | None = None
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


def extract_and_prepare_array(
    casadi_value: Any, expected_rows: int, expected_cols: int
) -> FloatArray:
    """
    Extracts numerical value from CasADi and ensures correct 2D shape.
    Updated to use unified FloatArray type.
    """
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
        else:
            logger.warning(
                f"Array shape mismatch: got {np_array.shape}, expected ({expected_rows}, {expected_cols})"
            )

    return np_array
