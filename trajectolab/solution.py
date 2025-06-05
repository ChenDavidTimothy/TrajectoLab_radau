"""
Solution interface for optimal control problem results.

This module provides the Solution class with a clean, bundled API design
that eliminates fragmentation and provides logical information grouping.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .tl_types import FloatArray, OptimalControlSolution, PhaseID, ProblemProtocol


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Solution:
    """Clean, bundled interface for multiphase optimal control problem solutions."""

    def __init__(
        self,
        raw_solution: OptimalControlSolution | None,
        problem: ProblemProtocol | None,
        auto_summary: bool = True,
    ) -> None:
        """
        Initialize solution wrapper from raw multiphase optimization results.

        Args:
            raw_solution: Raw optimization results from solver
            problem: Problem protocol instance
            auto_summary: Whether to automatically display comprehensive summary (default: True)
        """
        # Store raw data for internal use and direct CasADi access
        self._raw_solution = raw_solution
        self._problem = problem

        # Store raw CasADi objects for advanced users
        self.raw_solution = raw_solution.raw_solution if raw_solution else None
        self.opti = raw_solution.opti_object if raw_solution else None

        # Build variable name mappings for dictionary access
        if problem is not None:
            self._phase_state_names = {}
            self._phase_control_names = {}
            for phase_id in problem._get_phase_ids():
                self._phase_state_names[phase_id] = problem._get_phase_ordered_state_names(phase_id)
                self._phase_control_names[phase_id] = problem._get_phase_ordered_control_names(
                    phase_id
                )
        else:
            self._phase_state_names = {}
            self._phase_control_names = {}

        if auto_summary:
            self._show_comprehensive_summary()

    def _show_comprehensive_summary(self) -> None:
        try:
            from .summary import print_comprehensive_solution_summary

            print_comprehensive_solution_summary(self)
        except ImportError as e:
            logger.warning(f"Could not import comprehensive summary: {e}")
        except Exception as e:
            logger.warning(f"Error in comprehensive summary: {e}")

    @property
    def status(self) -> dict[str, Any]:
        """
        Complete solution status information.

        Returns:
            Dictionary containing:
            - success: Whether optimization succeeded
            - message: Solver status message
            - objective: Objective function value
            - total_mission_time: Total time across all phases
        """
        if self._raw_solution is None:
            return {
                "success": False,
                "message": "No solution available",
                "objective": float("nan"),
                "total_mission_time": float("nan"),
            }

        # Calculate total mission time
        if self._raw_solution.phase_initial_times and self._raw_solution.phase_terminal_times:
            earliest_start = min(self._raw_solution.phase_initial_times.values())
            latest_end = max(self._raw_solution.phase_terminal_times.values())
            total_time = latest_end - earliest_start
        else:
            total_time = float("nan")

        return {
            "success": self._raw_solution.success,
            "message": self._raw_solution.message,
            "objective": self._raw_solution.objective
            if self._raw_solution.objective is not None
            else float("nan"),
            "total_mission_time": total_time,
        }

    @property
    def phases(self) -> dict[PhaseID, dict[str, Any]]:
        """
        Complete phase information for all phases.

        Returns:
            Dictionary mapping phase IDs to phase data containing:
            - times: {initial, final, duration}
            - variables: {state_names, control_names, num_states, num_controls}
            - mesh: {polynomial_degrees, mesh_nodes, num_intervals}
            - time_arrays: {states, controls}
            - integrals: Integral values for this phase
        """
        if self._raw_solution is None:
            return {}

        phases_data = {}

        for phase_id in self._get_phase_ids():
            # Time information
            initial_time = self._raw_solution.phase_initial_times.get(phase_id, float("nan"))
            final_time = self._raw_solution.phase_terminal_times.get(phase_id, float("nan"))
            duration = (
                final_time - initial_time
                if not (np.isnan(initial_time) or np.isnan(final_time))
                else float("nan")
            )

            # Variable information
            state_names = self._phase_state_names.get(phase_id, [])
            control_names = self._phase_control_names.get(phase_id, [])

            # Mesh information
            polynomial_degrees = self._raw_solution.phase_mesh_intervals.get(phase_id, [])
            mesh_nodes = self._raw_solution.phase_mesh_nodes.get(
                phase_id, np.array([], dtype=np.float64)
            )

            # Time arrays
            time_states = self._raw_solution.phase_time_states.get(
                phase_id, np.array([], dtype=np.float64)
            )
            time_controls = self._raw_solution.phase_time_controls.get(
                phase_id, np.array([], dtype=np.float64)
            )

            # Integrals
            integrals = self._raw_solution.phase_integrals.get(phase_id, None)

            phases_data[phase_id] = {
                "times": {"initial": initial_time, "final": final_time, "duration": duration},
                "variables": {
                    "state_names": state_names.copy(),
                    "control_names": control_names.copy(),
                    "num_states": len(state_names),
                    "num_controls": len(control_names),
                },
                "mesh": {
                    "polynomial_degrees": polynomial_degrees.copy() if polynomial_degrees else [],
                    "mesh_nodes": mesh_nodes.copy()
                    if mesh_nodes.size > 0
                    else np.array([], dtype=np.float64),
                    "num_intervals": len(polynomial_degrees) if polynomial_degrees else 0,
                },
                "time_arrays": {"states": time_states.copy(), "controls": time_controls.copy()},
                "integrals": integrals,
            }

        return phases_data

    @property
    def parameters(self) -> dict[str, Any] | None:
        """
        Static parameter information.

        Returns:
            Dictionary containing:
            - values: Parameter values array
            - names: Parameter names (if available)
            - count: Number of parameters
            Returns None if no static parameters.
        """
        if self._raw_solution is None or self._raw_solution.static_parameters is None:
            return None

        # Try to get parameter names if available
        param_names = None
        if self._problem is not None and hasattr(self._problem, "_static_parameters"):
            try:
                static_params = self._problem._static_parameters
                if hasattr(static_params, "parameter_names"):
                    param_names = static_params.parameter_names.copy()
            except (AttributeError, IndexError):
                pass

        return {
            "values": self._raw_solution.static_parameters.copy(),
            "names": param_names,
            "count": len(self._raw_solution.static_parameters),
        }

    @property
    def adaptive(self) -> dict[str, Any] | None:
        """
        Adaptive algorithm information (if adaptive solver was used).

        Returns:
            Dictionary containing:
            - converged: Whether algorithm converged
            - iterations: Number of iterations performed
            - target_tolerance: Target error tolerance
            - phase_converged: Per-phase convergence status
            - final_errors: Final error estimates per phase
            - gamma_factors: Normalization factors per phase
            Returns None if fixed mesh solver was used.
        """
        if self._raw_solution is None or self._raw_solution.adaptive_data is None:
            return None

        adaptive_data = self._raw_solution.adaptive_data

        return {
            "converged": adaptive_data.converged,
            "iterations": adaptive_data.total_iterations,
            "target_tolerance": adaptive_data.target_tolerance,
            "phase_converged": adaptive_data.phase_converged.copy(),
            "final_errors": {
                phase_id: errors.copy()
                for phase_id, errors in adaptive_data.final_phase_error_estimates.items()
            },
            "gamma_factors": {
                phase_id: factors.copy() if factors is not None else None
                for phase_id, factors in adaptive_data.phase_gamma_factors.items()
            },
        }

    def _get_phase_ids(self) -> list[PhaseID]:
        """Get sorted list of phase IDs."""
        if self._raw_solution is None:
            return []
        return sorted(self._raw_solution.phase_initial_times.keys())

    def __getitem__(self, key: str | tuple[PhaseID, str]) -> FloatArray:
        """
        Dictionary-style access to solution variables and time arrays.

        Args:
            key: Either a variable name (searches all phases) or (phase_id, variable_name) tuple

        Returns:
            FloatArray containing the requested data

        Raises:
            KeyError: If the variable name is not found

        Examples:
            >>> # Access by variable name (searches all phases)
            >>> altitude_data = solution["altitude"]  # Returns first phase with "altitude"
            >>>
            >>> # Access by phase and variable name
            >>> altitude_phase1 = solution[(1, "altitude")]
            >>> thrust_phase2 = solution[(2, "thrust")]
            >>>
            >>> # Access time arrays
            >>> time_states_p1 = solution[(1, "time_states")]
            >>> time_controls_p2 = solution[(2, "time_controls")]
        """
        if not self.status["success"]:
            logger.warning("Cannot access variable '%s': Solution not successful", key)
            return np.array([], dtype=np.float64)

        if isinstance(key, tuple):
            return self._get_by_tuple_key(key)
        elif isinstance(key, str):
            return self._get_by_string_key(key)
        else:
            raise KeyError(
                f"Invalid key type: {type(key)}. Use string or (phase_id, variable_name) tuple"
            )

    def _get_by_tuple_key(self, key: tuple[PhaseID, str]) -> FloatArray:
        if len(key) != 2:
            raise KeyError("Tuple key must have exactly 2 elements: (phase_id, variable_name)")

        phase_id, var_name = key

        if phase_id not in self._get_phase_ids():
            raise KeyError(f"Phase {phase_id} not found in solution")

        if var_name == "time_states":
            return self._raw_solution.phase_time_states.get(
                phase_id, np.array([], dtype=np.float64)
            )
        elif var_name == "time_controls":
            return self._raw_solution.phase_time_controls.get(
                phase_id, np.array([], dtype=np.float64)
            )

        if phase_id in self._phase_state_names and var_name in self._phase_state_names[phase_id]:
            var_index = self._phase_state_names[phase_id].index(var_name)
            if phase_id in self._raw_solution.phase_states and var_index < len(
                self._raw_solution.phase_states[phase_id]
            ):
                return self._raw_solution.phase_states[phase_id][var_index]

        if (
            phase_id in self._phase_control_names
            and var_name in self._phase_control_names[phase_id]
        ):
            var_index = self._phase_control_names[phase_id].index(var_name)
            if phase_id in self._raw_solution.phase_controls and var_index < len(
                self._raw_solution.phase_controls[phase_id]
            ):
                return self._raw_solution.phase_controls[phase_id][var_index]

        raise KeyError(f"Variable '{var_name}' not found in phase {phase_id}")

    def _get_by_string_key(self, key: str) -> FloatArray:
        for phase_id in self._get_phase_ids():
            try:
                return self[(phase_id, key)]
            except KeyError:
                continue

        all_vars = []
        for phase_id in self._get_phase_ids():
            phase_vars = (
                self._phase_state_names.get(phase_id, [])
                + self._phase_control_names.get(phase_id, [])
                + ["time_states", "time_controls"]
            )
            all_vars.extend([f"({phase_id}, '{var}')" for var in phase_vars])

        raise KeyError(f"Variable '{key}' not found in any phase. Available: {all_vars}")

    def __contains__(self, key: str | tuple[PhaseID, str]) -> bool:
        """
        Check if a variable exists in the solution.

        Args:
            key: Either a variable name or (phase_id, variable_name) tuple

        Returns:
            True if variable exists, False otherwise

        Examples:
            >>> if "altitude" in solution:
            ...     altitude = solution["altitude"]
            >>> if (1, "altitude") in solution:
            ...     altitude_p1 = solution[(1, "altitude")]
        """
        try:
            self[key]
            return True
        except KeyError:
            return False

    def plot(
        self,
        phase_id: PhaseID | None = None,
        *variable_names: str,
        figsize: tuple[float, float] = (12.0, 8.0),
        show_phase_boundaries: bool = True,
    ) -> None:
        """
        Plot multiphase trajectories with interval coloring and phase boundaries.

        Args:
            phase_id: Specific phase to plot (None plots all phases)
            *variable_names: Optional specific variable names to plot
            figsize: Figure size for each window
            show_phase_boundaries: Whether to show vertical lines at phase boundaries

        Examples:
            >>> solution.plot()  # Plot all phases with interval colors
            >>> solution.plot(1)  # Plot only phase 1
            >>> solution.plot(phase_id=None, "position", "velocity")  # Specific variables
            >>> solution.plot(1, "thrust")  # Specific variable for specific phase
        """
        from .plot import plot_multiphase_solution

        plot_multiphase_solution(self, phase_id, variable_names, figsize, show_phase_boundaries)

    def summary(self, comprehensive: bool = True) -> None:
        """
        Print solution summary.

        Args:
            comprehensive: If True (default), show exhaustive summary.
                          If False, show simple summary.

        Examples:
            >>> solution.summary()  # Comprehensive summary (default)
            >>> solution.summary(comprehensive=False)  # Simple summary
        """
        if comprehensive:
            try:
                from .summary import print_comprehensive_solution_summary

                print_comprehensive_solution_summary(self)
            except ImportError as e:
                logger.warning(f"Could not import comprehensive summary: {e}")
            except Exception as e:
                logger.warning(f"Error in comprehensive summary: {e}")
        else:
            # Simple summary using new API
            print(f"Solution Status: {self.status['success']}")
            print(f"Objective: {self.status['objective']:.6e}")
            print(f"Total Mission Time: {self.status['total_mission_time']:.6f}")
            print(f"Phases: {len(self.phases)}")
            if self.adaptive:
                print(f"Adaptive: Converged in {self.adaptive['iterations']} iterations")
