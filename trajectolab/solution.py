"""
Solution interface for optimal control problem results.

This module provides the Solution class that wraps optimization results
in a user-friendly interface with trajectory access capabilities.
Comprehensive summary functionality is delegated to summary.py for separation of concerns.
"""

import logging
from typing import TypeAlias

import numpy as np

from .tl_types import FloatArray, OptimalControlSolution, PhaseID, ProblemProtocol


logger = logging.getLogger(__name__)

_TrajectoryTuple: TypeAlias = tuple[FloatArray, FloatArray]


class Solution:
    """User-friendly interface for multiphase optimal control problem solutions."""

    # Type hints for key attributes to ensure proper type inference
    success: bool
    objective: float
    _problem: ProblemProtocol | None

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
        if raw_solution is not None:
            self.success = raw_solution.success
            self.message = raw_solution.message

            # Core solution data
            self.objective = (
                raw_solution.objective if raw_solution.objective is not None else float("nan")
            )

            # Multiphase solution data
            self.phase_initial_times = raw_solution.phase_initial_times
            self.phase_terminal_times = raw_solution.phase_terminal_times
            self.phase_time_states = raw_solution.phase_time_states
            self.phase_states = raw_solution.phase_states
            self.phase_time_controls = raw_solution.phase_time_controls
            self.phase_controls = raw_solution.phase_controls
            self.phase_integrals = raw_solution.phase_integrals
            self.static_parameters = raw_solution.static_parameters

            # Raw solver data
            self.raw_solution = raw_solution.raw_solution
            self.opti = raw_solution.opti_object

            # Mesh information per phase
            self.phase_mesh_intervals = raw_solution.phase_mesh_intervals
            self.phase_mesh_nodes = raw_solution.phase_mesh_nodes

            # Per-interval solution data per phase
            self.phase_solved_state_trajectories_per_interval = (
                raw_solution.phase_solved_state_trajectories_per_interval
            )
            self.phase_solved_control_trajectories_per_interval = (
                raw_solution.phase_solved_control_trajectories_per_interval
            )
        else:
            self.success = False
            self.message = "No multiphase solution"
            self.objective = float("nan")

            # Initialize empty multiphase data
            self.phase_initial_times = {}
            self.phase_terminal_times = {}
            self.phase_time_states = {}
            self.phase_states = {}
            self.phase_time_controls = {}
            self.phase_controls = {}
            self.phase_integrals = {}
            self.static_parameters = None

            self.raw_solution = None
            self.opti = None
            self.phase_mesh_intervals = {}
            self.phase_mesh_nodes = {}
            self.phase_solved_state_trajectories_per_interval = {}
            self.phase_solved_control_trajectories_per_interval = {}

        if problem is not None:
            self._problem = problem
            # Store phase-specific variable names
            self._phase_state_names = {}
            self._phase_control_names = {}
            for phase_id in problem.get_phase_ids():
                self._phase_state_names[phase_id] = problem.get_phase_ordered_state_names(phase_id)
                self._phase_control_names[phase_id] = problem.get_phase_ordered_control_names(
                    phase_id
                )
        else:
            self._problem = None
            self._phase_state_names = {}
            self._phase_control_names = {}

        # Automatically show comprehensive summary by default
        if auto_summary:
            self._show_comprehensive_summary()

    def _show_comprehensive_summary(self) -> None:
        """Automatically display comprehensive solution summary."""
        try:
            from .summary import print_comprehensive_solution_summary
            print_comprehensive_solution_summary(self)
        except ImportError as e:
            logger.warning(f"Could not import comprehensive summary: {e}")
            # Fallback to simple summary
            self._print_simple_summary()
        except Exception as e:
            logger.warning(f"Error in comprehensive summary: {e}")
            # Fallback to simple summary
            self._print_simple_summary()

    def get_phase_ids(self) -> list[PhaseID]:
        """Get list of phase IDs in the solution."""
        return sorted(self.phase_initial_times.keys())

    def get_phase_initial_time(self, phase_id: PhaseID) -> float:
        """Get initial time for a specific phase."""
        if phase_id not in self.phase_initial_times:
            raise ValueError(f"Phase {phase_id} not found in solution")
        return self.phase_initial_times[phase_id]

    def get_phase_final_time(self, phase_id: PhaseID) -> float:
        """Get final time for a specific phase."""
        if phase_id not in self.phase_terminal_times:
            raise ValueError(f"Phase {phase_id} not found in solution")
        return self.phase_terminal_times[phase_id]

    def get_phase_duration(self, phase_id: PhaseID) -> float:
        """Get duration of a specific phase."""
        return self.get_phase_final_time(phase_id) - self.get_phase_initial_time(phase_id)

    def get_total_mission_time(self) -> float:
        """Get total mission time across all phases."""
        if not self.phase_initial_times or not self.phase_terminal_times:
            return float("nan")

        earliest_start = min(self.phase_initial_times.values())
        latest_end = max(self.phase_terminal_times.values())
        return latest_end - earliest_start

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
        if not self.success:
            logger.warning("Cannot access variable '%s': Solution not successful", key)
            return np.array([], dtype=np.float64)

        # Handle tuple access: (phase_id, variable_name)
        if isinstance(key, tuple):
            return self._get_by_tuple_key(key)

        # Handle string access: search all phases for variable name
        elif isinstance(key, str):
            return self._get_by_string_key(key)

        else:
            raise KeyError(
                f"Invalid key type: {type(key)}. Use string or (phase_id, variable_name) tuple"
            )

    def _get_by_tuple_key(self, key: tuple[PhaseID, str]) -> FloatArray:
        """Extracted helper method for tuple-based access."""
        if len(key) != 2:
            raise KeyError("Tuple key must have exactly 2 elements: (phase_id, variable_name)")

        phase_id, var_name = key

        if phase_id not in self.get_phase_ids():
            raise KeyError(f"Phase {phase_id} not found in solution")

        # Handle time arrays
        if var_name == "time_states":
            return self.phase_time_states.get(phase_id, np.array([], dtype=np.float64))
        elif var_name == "time_controls":
            return self.phase_time_controls.get(phase_id, np.array([], dtype=np.float64))

        # Handle state variables
        if phase_id in self._phase_state_names and var_name in self._phase_state_names[phase_id]:
            var_index = self._phase_state_names[phase_id].index(var_name)
            if phase_id in self.phase_states and var_index < len(self.phase_states[phase_id]):
                return self.phase_states[phase_id][var_index]

        # Handle control variables
        if (
            phase_id in self._phase_control_names
            and var_name in self._phase_control_names[phase_id]
        ):
            var_index = self._phase_control_names[phase_id].index(var_name)
            if phase_id in self.phase_controls and var_index < len(self.phase_controls[phase_id]):
                return self.phase_controls[phase_id][var_index]

        raise KeyError(f"Variable '{var_name}' not found in phase {phase_id}")

    def _get_by_string_key(self, key: str) -> FloatArray:
        """Extracted helper method for string-based access."""
        # Search for variable in all phases
        for phase_id in self.get_phase_ids():
            try:
                return self[(phase_id, key)]
            except KeyError:
                continue

        # Variable not found in any phase
        all_vars = []
        for phase_id in self.get_phase_ids():
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

    @property
    def phase_state_names(self) -> dict[PhaseID, list[str]]:
        """Get dictionary mapping phase IDs to state variable names."""
        return {phase_id: names.copy() for phase_id, names in self._phase_state_names.items()}

    @property
    def phase_control_names(self) -> dict[PhaseID, list[str]]:
        """Get dictionary mapping phase IDs to control variable names."""
        return {phase_id: names.copy() for phase_id, names in self._phase_control_names.items()}

    @property
    def all_variable_names(self) -> list[str]:
        """Get list of all unique variable names across all phases."""
        all_vars = set()
        for phase_id in self.get_phase_ids():
            all_vars.update(self._phase_state_names.get(phase_id, []))
            all_vars.update(self._phase_control_names.get(phase_id, []))
        return sorted(all_vars)

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
        # Import here to avoid circular imports
        from .plot import plot_multiphase_solution

        plot_multiphase_solution(
            self, phase_id, variable_names, figsize, show_phase_boundaries
        )

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
                self._print_simple_summary()
            except Exception as e:
                logger.warning(f"Error in comprehensive summary: {e}")
                self._print_simple_summary()
        else:
            self._print_simple_summary()

    def _print_simple_summary(self) -> None:
        """Print simple solution summary (fallback)."""
        print(f"Multiphase Solution Status: {'Success' if self.success else 'Failed'}")

        if self.success:
            print(f"  Objective: {self.objective}")
            print(f"  Total Mission Time: {self.get_total_mission_time():.6f}")
            print(f"  Number of Phases: {len(self.get_phase_ids())}")

            for phase_id in self.get_phase_ids():
                print(f"  Phase {phase_id}:")
                print(
                    f"    Time: {self.get_phase_initial_time(phase_id):.6f} → {self.get_phase_final_time(phase_id):.6f}"
                )
                print(f"    Duration: {self.get_phase_duration(phase_id):.6f}")
                print(
                    f"    States: {len(self._phase_state_names.get(phase_id, []))} {self._phase_state_names.get(phase_id, [])}"
                )
                print(
                    f"    Controls: {len(self._phase_control_names.get(phase_id, []))} {self._phase_control_names.get(phase_id, [])}"
                )
                if phase_id in self.phase_mesh_intervals:
                    print(f"    Mesh intervals: {len(self.phase_mesh_intervals[phase_id])}")

            if self.static_parameters is not None:
                print(f"  Static Parameters: {len(self.static_parameters)}")
        else:
            print(f"  Message: {self.message}")
