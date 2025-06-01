import logging
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure as MplFigure

from .tl_types import FloatArray, OptimalControlSolution, PhaseID, ProblemProtocol


logger = logging.getLogger(__name__)

_TrajectoryTuple: TypeAlias = tuple[FloatArray, FloatArray]


class Solution:
    """User-friendly interface for multiphase optimal control problem solutions."""

    # Type hints for key attributes to ensure proper type inference
    success: bool
    objective: float  # Always a float - NaN for failed solutions

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
        """Initialize solution wrapper from raw multiphase optimization results."""
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
        Plot multiphase trajectories with smart layout and phase visualization.

        Args:
            phase_id: Specific phase to plot (None plots all phases)
            *variable_names: Optional specific variable names to plot
            figsize: Figure size for each window
            show_phase_boundaries: Whether to show vertical lines at phase boundaries

        Examples:
            >>> solution.plot()  # Plot all phases, separate windows for states/controls
            >>> solution.plot(1)  # Plot only phase 1
            >>> solution.plot(phase_id=None, "position", "velocity")  # Specific variables, all phases
            >>> solution.plot(1, "thrust")  # Specific variable for specific phase
        """
        if not self.success:
            logger.warning("Cannot plot: Solution not successful")
            return

        if phase_id is not None:
            # Plot specific phase
            if phase_id not in self.get_phase_ids():
                raise ValueError(f"Phase {phase_id} not found in solution")
            self._plot_single_phase(phase_id, variable_names, figsize)
        else:
            # Plot all phases
            if variable_names:
                self._plot_multiphase_variables(variable_names, figsize, show_phase_boundaries)
            else:
                self._plot_multiphase_default(figsize, show_phase_boundaries)

    def _plot_single_phase(
        self, phase_id: PhaseID, variable_names: tuple[str, ...], figsize: tuple[float, float]
    ) -> None:
        """Plot trajectories for a single phase."""
        phase_state_names = self._phase_state_names.get(phase_id, [])
        phase_control_names = self._phase_control_names.get(phase_id, [])

        if variable_names:
            # Plot specific variables
            self._create_variable_plot(
                f"Phase {phase_id} Variables", [(phase_id, var) for var in variable_names], figsize
            )
        else:
            # Plot all variables for this phase
            figures_created = []

            if phase_state_names:
                fig = self._create_variable_plot(
                    f"Phase {phase_id} States",
                    [(phase_id, var) for var in phase_state_names],
                    figsize,
                    show_immediately=False,
                )
                figures_created.append(fig)

            if phase_control_names:
                fig = self._create_variable_plot(
                    f"Phase {phase_id} Controls",
                    [(phase_id, var) for var in phase_control_names],
                    figsize,
                    show_immediately=False,
                )
                figures_created.append(fig)

            # Show all figures
            for fig in figures_created:
                plt.figure(fig.number)
                plt.show(block=False)

            if figures_created:
                plt.figure(figures_created[-1].number)
                plt.show()

    def _plot_multiphase_variables(
        self,
        variable_names: tuple[str, ...],
        figsize: tuple[float, float],
        show_phase_boundaries: bool,
    ) -> None:
        """Plot specific variables across all phases."""
        # Find which phases have each variable
        phase_var_pairs = []
        for var_name in variable_names:
            for phase_id in self.get_phase_ids():
                if (phase_id, var_name) in self:
                    phase_var_pairs.append((phase_id, var_name))

        if not phase_var_pairs:
            logger.warning("None of the requested variables found in any phase")
            return

        self._create_multiphase_variable_plot(
            "Multiphase Variables", phase_var_pairs, figsize, show_phase_boundaries
        )

    def _plot_multiphase_default(
        self, figsize: tuple[float, float], show_phase_boundaries: bool
    ) -> None:
        """Plot all variables with states and controls in separate windows."""
        figures_created = []

        # Collect all unique state variables across phases
        all_state_vars = set()
        all_control_vars = set()

        for phase_id in self.get_phase_ids():
            all_state_vars.update(self._phase_state_names.get(phase_id, []))
            all_control_vars.update(self._phase_control_names.get(phase_id, []))

        # Plot states
        if all_state_vars:
            state_pairs = []
            for var_name in sorted(all_state_vars):
                for phase_id in self.get_phase_ids():
                    if (phase_id, var_name) in self:
                        state_pairs.append((phase_id, var_name))

            if state_pairs:
                fig = self._create_multiphase_variable_plot(
                    "Multiphase States",
                    state_pairs,
                    figsize,
                    show_phase_boundaries,
                    show_immediately=False,
                )
                figures_created.append(fig)

        # Plot controls
        if all_control_vars:
            control_pairs = []
            for var_name in sorted(all_control_vars):
                for phase_id in self.get_phase_ids():
                    if (phase_id, var_name) in self:
                        control_pairs.append((phase_id, var_name))

            if control_pairs:
                fig = self._create_multiphase_variable_plot(
                    "Multiphase Controls",
                    control_pairs,
                    figsize,
                    show_phase_boundaries,
                    show_immediately=False,
                )
                figures_created.append(fig)

        # Show all figures
        for fig in figures_created:
            plt.figure(fig.number)
            plt.show(block=False)

        if figures_created:
            plt.figure(figures_created[-1].number)
            plt.show()

    def _create_variable_plot(
        self,
        title: str,
        phase_var_pairs: list[tuple[PhaseID, str]],
        figsize: tuple[float, float],
        show_immediately: bool = True,
    ) -> MplFigure:
        """Create a plot for specific phase-variable pairs."""
        if not phase_var_pairs:
            return plt.figure()

        # Group by variable name for subplots
        var_groups = {}
        for phase_id, var_name in phase_var_pairs:
            if var_name not in var_groups:
                var_groups[var_name] = []
            var_groups[var_name].append(phase_id)

        num_vars = len(var_groups)
        if num_vars == 0:
            return plt.figure()

        # subplot layout determination
        rows, cols = self._determine_subplot_layout(num_vars)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False)
        fig.suptitle(title)

        if num_vars == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()

        # Plot each variable
        for i, (var_name, phase_ids) in enumerate(var_groups.items()):
            ax = axes[i]

            for phase_id in phase_ids:
                try:
                    time_data = self[
                        (
                            phase_id,
                            "time_states"
                            if self._is_state_variable(phase_id, var_name)
                            else "time_controls",
                        )
                    ]
                    var_data = self[(phase_id, var_name)]

                    if len(time_data) > 0 and len(var_data) > 0:
                        ax.plot(time_data, var_data, label=f"Phase {phase_id}", linewidth=1.5)

                except KeyError:
                    continue

            ax.set_ylabel(var_name)
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused subplots
        for i in range(num_vars, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if show_immediately:
            plt.show()

        return fig

    def _create_multiphase_variable_plot(
        self,
        title: str,
        phase_var_pairs: list[tuple[PhaseID, str]],
        figsize: tuple[float, float],
        show_phase_boundaries: bool,
        show_immediately: bool = True,
    ) -> MplFigure:
        """Create a multiphase plot with phase boundaries."""
        fig = self._create_variable_plot(title, phase_var_pairs, figsize, show_immediately=False)

        if show_phase_boundaries and len(self.get_phase_ids()) > 1:
            # Add phase boundary lines
            for ax in fig.get_axes():
                if ax.get_visible():
                    for phase_id in self.get_phase_ids()[:-1]:  # Exclude last phase
                        final_time = self.get_phase_final_time(phase_id)
                        ax.axvline(
                            final_time,
                            color="red",
                            linestyle="--",
                            alpha=0.7,
                            label="Phase Boundary" if phase_id == self.get_phase_ids()[0] else "",
                        )

                    # Update legend if phase boundaries were added
                    handles, labels = ax.get_legend_handles_labels()
                    if "Phase Boundary" in labels:
                        ax.legend()

        if show_immediately:
            plt.show()

        return fig

    def _determine_subplot_layout(self, num_plots: int) -> tuple[int, int]:
        """
        mathematical approach to subplot layout.

        Eliminates hardcoded conditional logic with clean mathematical solution.
        """
        if num_plots <= 1:
            return (1, 1)

        # Pure mathematical approach - always works
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))
        return (rows, cols)

    def _is_state_variable(self, phase_id: PhaseID, var_name: str) -> bool:
        """Check if a variable is a state variable in the given phase."""
        return phase_id in self._phase_state_names and var_name in self._phase_state_names[phase_id]

    def summary(self) -> None:
        """Print a summary of the multiphase solution results."""
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
