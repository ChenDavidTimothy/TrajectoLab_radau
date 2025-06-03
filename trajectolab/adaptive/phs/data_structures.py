import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from trajectolab.tl_types import FloatArray, ODESolverCallable, PhaseID


__all__ = [
    "AdaptiveParameters",
    "HRefineResult",
    "MultiphaseAdaptiveState",
    "PReduceResult",
    "PRefineResult",
    "ensure_2d_array",
]

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParameters:
    """Parameters controlling the multiphase adaptive mesh refinement algorithm."""

    error_tolerance: float
    max_iterations: int
    min_polynomial_degree: int
    max_polynomial_degree: int
    ode_solver_tolerance: float = 1e-7
    num_error_sim_points: int = 50
    ode_method: str = "RK45"
    ode_max_step: float | None = None
    ode_atol_factor: float = 1e-2
    ode_solver: ODESolverCallable | None = None

    def get_ode_solver(self) -> ODESolverCallable:
        """Get the configured ODE solver function."""
        if self.ode_solver is not None:
            return self.ode_solver

        from scipy.integrate import solve_ivp

        def configured_solver(fun, t_span, y0, t_eval=None, **kwargs):
            print("\n🔧 INSIDE configured_solver function")
            print(f"   Input kwargs: {kwargs}")
            print("   User settings to apply:")
            print(f"     ode_method: {self.ode_method}")
            print(f"     ode_solver_tolerance: {self.ode_solver_tolerance}")
            print(f"     ode_atol_factor: {self.ode_atol_factor}")
            print(f"     ode_max_step: {self.ode_max_step}")

            # FIXED: Explicitly set user parameters instead of using setdefault
            # Force user settings to be applied - this ensures user has complete control
            kwargs["method"] = self.ode_method
            kwargs["rtol"] = self.ode_solver_tolerance
            kwargs["atol"] = self.ode_solver_tolerance * self.ode_atol_factor

            if self.ode_max_step is not None:
                kwargs["max_step"] = self.ode_max_step

            print(f"   Modified kwargs to pass to scipy: {kwargs}")

            # Check if our invalid values are present
            if kwargs.get("method") == "this is gibberish":
                print("   🚨 About to call scipy with INVALID method - should fail!")
            if kwargs.get("rtol", 0) < 0:
                print("   🚨 About to call scipy with NEGATIVE rtol - should fail!")
            if kwargs.get("atol", 0) < 0:
                print("   🚨 About to call scipy with NEGATIVE atol - should fail!")
            if kwargs.get("max_step") is not None and kwargs.get("max_step") < 0:
                print("   🚨 About to call scipy with NEGATIVE max_step - should fail!")

            print("   Calling scipy.integrate.solve_ivp now...")

            return solve_ivp(fun, t_span, y0, t_eval=t_eval, **kwargs)

        return configured_solver


@dataclass
class MultiphaseAdaptiveState:
    """Tracks adaptive refinement state across all phases in unified NLP."""

    phase_polynomial_degrees: dict[PhaseID, list[int]]
    phase_mesh_points: dict[PhaseID, FloatArray]
    phase_converged: dict[PhaseID, bool]
    iteration: int = 0
    most_recent_unified_solution: Any = None  # OptimalControlSolution

    def all_phases_converged(self) -> bool:
        """Check if all phases have converged."""
        return all(self.phase_converged.values()) if self.phase_converged else False

    def get_phase_ids(self) -> list[PhaseID]:
        """Get ordered list of phase IDs."""
        return sorted(self.phase_polynomial_degrees.keys())

    def configure_problem_meshes(self, problem: Any) -> None:  # ProblemProtocol
        """Configure all phase meshes in the unified problem."""
        for phase_id in self.get_phase_ids():
            if phase_id in problem._phases:
                phase_def = problem._phases[phase_id]
                phase_def.collocation_points_per_interval = self.phase_polynomial_degrees[phase_id]
                phase_def.global_normalized_mesh_nodes = self.phase_mesh_points[phase_id]


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


def ensure_2d_array(casadi_value: Any, expected_rows: int, expected_cols: int) -> FloatArray:
    """
     Simple array conversion without over-engineering.
    Converts CasADi value to 2D numpy array with expected shape.
    """
    # Convert to numpy array
    if hasattr(casadi_value, "to_DM"):
        np_array = np.array(casadi_value.to_DM(), dtype=np.float64)
    else:
        np_array = np.array(casadi_value, dtype=np.float64)

    # Handle empty arrays
    if expected_rows == 0:
        return np.empty((0, expected_cols), dtype=np.float64)

    # Ensure 2D and correct shape
    if np_array.ndim == 1:
        if len(np_array) == expected_rows * expected_cols:
            np_array = np_array.reshape(expected_rows, expected_cols)
        elif len(np_array) == expected_rows:
            np_array = np_array.reshape(expected_rows, 1)
        else:
            np_array = np_array.reshape(1, -1)

    # Transpose if dimensions are swapped
    if np_array.shape[0] != expected_rows and np_array.shape[1] == expected_rows:
        np_array = np_array.T

    return np_array
