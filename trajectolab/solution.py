"""
Solution interface for optimal control problem results.

This module provides the Solution class that wraps optimization results
in a user-friendly interface with plotting capabilities and trajectory access.
"""

import logging
from typing import TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D

from .tl_types import FloatArray, OptimalControlSolution, ProblemProtocol


logger = logging.getLogger(__name__)

_TrajectoryTuple: TypeAlias = tuple[FloatArray, FloatArray]


class Solution:
    """User-friendly interface for optimal control problem solutions."""

    # Type hints for key attributes to ensure proper type inference
    success: bool
    objective: float  # Always a float - NaN for failed solutions
    initial_time: float | None
    final_time: float | None

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
        """Initialize solution wrapper from raw optimization results."""
        if raw_solution is not None:
            self.success = raw_solution.success
            self.message = raw_solution.message
            self.initial_time = raw_solution.initial_time_variable
            self.final_time = raw_solution.terminal_time_variable
            self.objective = (
                raw_solution.objective if raw_solution.objective is not None else float("nan")
            )
            self.integrals = raw_solution.integrals
            self.time_states = raw_solution.time_states
            self.states = raw_solution.states
            self.time_controls = raw_solution.time_controls
            self.controls = raw_solution.controls
            self.raw_solution = raw_solution.raw_solution
            self.opti = raw_solution.opti_object
            self.mesh_intervals = raw_solution.num_collocation_nodes_per_interval
            self.mesh_nodes = raw_solution.global_normalized_mesh_nodes
        else:
            self.success = False
            self.message = "No solution"
            self.initial_time = None
            self.final_time = None
            self.objective = float("nan")
            self.integrals = None
            self.time_states = np.array([], dtype=np.float64)
            self.states = []
            self.time_controls = np.array([], dtype=np.float64)
            self.controls = []
            self.raw_solution = None
            self.opti = None
            self.mesh_intervals = []
            self.mesh_nodes = None

        if problem is not None:
            self._state_names = problem.get_ordered_state_names()
            self._control_names = problem.get_ordered_control_names()
        else:
            self._state_names = []
            self._control_names = []

    def __getitem__(self, key: str) -> FloatArray:
        """
        Dictionary-style access to solution variables and time arrays.

        Args:
            key: Variable name, "time_states", or "time_controls"

        Returns:
            FloatArray containing the requested data

        Raises:
            KeyError: If the variable name is not found

        Examples:
            >>> altitude_data = solution["altitude"]
            >>> thrust_data = solution["thrust"]
            >>> time_states = solution["time_states"]
            >>> time_controls = solution["time_controls"]
        """
        if not self.success:
            logger.warning("Cannot access variable '%s': Solution not successful", key)
            return np.array([], dtype=np.float64)

        # Handle time arrays
        if key == "time_states":
            return self.time_states
        elif key == "time_controls":
            return self.time_controls

        # Handle state variables
        if key in self._state_names:
            var_index = self._state_names.index(key)
            return self.states[var_index]

        # Handle control variables
        if key in self._control_names:
            var_index = self._control_names.index(key)
            return self.controls[var_index]

        # Variable not found
        available_vars = self._state_names + self._control_names + ["time_states", "time_controls"]
        raise KeyError(f"Variable '{key}' not found. Available variables: {available_vars}")

    def __contains__(self, key: str) -> bool:
        """
        Check if a variable exists in the solution.

        Args:
            key: Variable name to check

        Returns:
            True if variable exists, False otherwise

        Examples:
            >>> if "altitude" in solution:
            ...     altitude = solution["altitude"]
        """
        if key in ["time_states", "time_controls"]:
            return True
        return key in self._state_names or key in self._control_names

    @property
    def state_names(self) -> list[str]:
        """Get list of available state variable names."""
        return self._state_names.copy()

    @property
    def control_names(self) -> list[str]:
        """Get list of available control variable names."""
        return self._control_names.copy()

    @property
    def variable_names(self) -> list[str]:
        """Get list of all available variable names including time arrays."""
        return self._state_names + self._control_names + ["time_states", "time_controls"]

    def plot(self, *variable_names: str, figsize: tuple[float, float] = (10.0, 8.0)) -> None:
        """
        Plot trajectories with smart layout and windowing.

        Args:
            *variable_names: Optional specific variable names to plot. If none provided,
                all states and controls are plotted in separate windows.
            figsize: Figure size for each window

        Raises:
            ValueError: If attempting to plot time arrays directly or unknown variables

        Examples:
            >>> solution.plot()  # Separate windows for states and controls
            >>> solution.plot("position", "velocity")  # Specific variables in one window
            >>> solution.plot("thrust", "altitude")  # Mixed state/control variables

        Note:
            - States and controls are automatically separated into different windows when no
              specific variables are requested
            - Smart layout uses optimal subplot arrangements (up to 3x3 per window)
            - Additional windows are created when more than 9 variables per type
        """
        if not self.success:
            logger.warning("Cannot plot: Solution not successful")
            return

        # Validate requested variables
        if variable_names:
            self._validate_plot_variables(list(variable_names))
            # Plot specific variables in organized windows
            self._plot_specific_variables(list(variable_names), figsize)
        else:
            # Default behavior: separate windows for states and controls
            self._plot_default_layout(figsize)

    def _validate_plot_variables(self, variable_names: list[str]) -> None:
        """
        Validate variables for plotting.

        Args:
            variable_names: List of variable names to validate

        Raises:
            ValueError: If time arrays or unknown variables are requested
        """
        # Check for time array plotting attempts
        time_arrays = [name for name in variable_names if name in ["time_states", "time_controls"]]
        if time_arrays:
            raise ValueError(
                f"Cannot plot time arrays directly: {time_arrays}. "
                "Time arrays are automatically used as x-axis for trajectory plots."
            )

        # Check for unknown variables
        unknown_vars = [name for name in variable_names if name not in self]
        if unknown_vars:
            available_vars = self._state_names + self._control_names
            raise ValueError(
                f"Unknown variables: {unknown_vars}. Available variables: {available_vars}"
            )

    def _plot_default_layout(self, figsize: tuple[float, float]) -> None:
        """Plot all variables with states and controls in separate windows."""
        figures_created = []

        if self._state_names:
            figures_created.extend(
                self._create_variable_windows(
                    self._state_names, "States", "state", figsize, show_immediately=False
                )
            )

        if self._control_names:
            figures_created.extend(
                self._create_variable_windows(
                    self._control_names, "Controls", "control", figsize, show_immediately=False
                )
            )

        if not self._state_names and not self._control_names:
            logger.info("No variables to plot")
            return

        # Show all figures simultaneously
        for fig in figures_created:
            plt.figure(fig.number)
            plt.show(block=False)

        # Make the last figure blocking so program doesn't exit immediately in scripts
        if figures_created:
            plt.figure(figures_created[-1].number)
            plt.show()

    def _plot_specific_variables(
        self, variable_names: list[str], figsize: tuple[float, float]
    ) -> None:
        """Plot specific requested variables in organized windows."""
        self._create_variable_windows(
            variable_names, "Variables", "mixed", figsize, show_immediately=True
        )
        # For specific variables, show immediately since there's typically only one window

    def _create_variable_windows(
        self,
        variables: list[str],
        window_title: str,
        var_type: str,
        figsize: tuple[float, float],
        show_immediately: bool = True,
    ) -> list[MplFigure]:
        """
        Create optimally-laid-out windows for variable groups.

        Args:
            variables: List of variable names to plot
            window_title: Base title for windows
            var_type: Type of variables ("state", "control", or "mixed")
            figsize: Figure size for each window
            show_immediately: Whether to show figures immediately or return them for later display

        Returns:
            List of created figures
        """
        max_plots_per_window = 9  # 3x3 grid maximum
        num_variables = len(variables)

        if num_variables == 0:
            return []

        figures_created = []

        # Create windows for groups of variables
        window_count = 1
        for start_idx in range(0, num_variables, max_plots_per_window):
            end_idx = min(start_idx + max_plots_per_window, num_variables)
            window_variables = variables[start_idx:end_idx]

            # Determine window title
            if num_variables <= max_plots_per_window:
                title = window_title
            else:
                title = f"{window_title} ({window_count})"

            fig = self._create_single_window(
                window_variables, title, var_type, figsize, show_immediately
            )
            figures_created.append(fig)
            window_count += 1

        return figures_created

    def _create_single_window(
        self,
        variables: list[str],
        title: str,
        var_type: str,
        figsize: tuple[float, float],
        show_immediately: bool = True,
    ) -> MplFigure:
        """
        Create a single window with optimal subplot layout.

        Args:
            variables: Variables to plot in this window
            title: Window title
            var_type: Variable type for determining plot style
            figsize: Figure size
            show_immediately: Whether to show the figure immediately

        Returns:
            Created matplotlib figure
        """
        num_plots = len(variables)
        if num_plots == 0:
            return plt.figure()

        # Determine optimal layout
        rows, cols = self._determine_subplot_layout(num_plots)

        # Create figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        fig.suptitle(title)

        # Handle single subplot case
        if num_plots == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()

        # Get colors for mesh intervals
        colors = self._get_interval_colors()

        # Plot each variable
        for i, var_name in enumerate(variables):
            self._plot_single_variable(axes[i], var_name, self._get_variable_type(var_name), colors)

        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        # Set x-label on bottom plots
        bottom_row_start = (rows - 1) * cols
        for i in range(bottom_row_start, min(bottom_row_start + cols, num_plots)):
            axes[i].set_xlabel("Time")

        # Add mesh legend if applicable
        if colors is not None and len(colors) > 0:
            self._add_mesh_legend(fig, colors)

        plt.tight_layout()

        if show_immediately:
            plt.show()

        return fig

    def _determine_subplot_layout(self, num_plots: int) -> tuple[int, int]:
        """
        Determine optimal (rows, cols) layout for given number of plots.

        Args:
            num_plots: Number of plots to arrange

        Returns:
            Tuple of (rows, columns) for optimal layout
        """
        if num_plots == 1:
            return (1, 1)
        elif num_plots == 2:
            return (2, 1)
        elif num_plots == 3:
            return (2, 2)
        elif num_plots == 4:
            return (2, 2)
        elif num_plots in [5, 6]:
            return (3, 2)
        elif num_plots in [7, 8, 9]:
            return (3, 3)
        else:
            # Fallback for edge cases (shouldn't occur with max 9 per window)
            rows = int(np.ceil(np.sqrt(num_plots)))
            cols = int(np.ceil(num_plots / rows))
            return (rows, cols)

    def _get_variable_type(self, var_name: str) -> str:
        """
        Determine if variable is state or control.

        Args:
            var_name: Variable name

        Returns:
            "state" or "control"
        """
        if var_name in self._state_names:
            return "state"
        elif var_name in self._control_names:
            return "control"
        else:
            # Fallback (shouldn't occur after validation)
            return "state"

    def _plot_single_variable(
        self, ax: MplAxes, name: str, var_type: str, colors: list | np.ndarray | None
    ) -> None:
        """Plot single variable with correct mathematical representation."""
        if name not in self:
            logger.warning("Variable '%s' not found for plotting", name)
            return

        # Get time and values using dictionary access
        if var_type == "state":
            time_array = self["time_states"]
            values_array = self[name]
        else:
            time_array = self["time_controls"]
            values_array = self[name]

        if time_array.size == 0:
            return

        if var_type == "control":
            self._plot_control_step_function(ax, time_array, values_array, colors)
        else:
            self._plot_state_linear(ax, time_array, values_array, colors)

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    def _plot_control_step_function(
        self,
        ax: MplAxes,
        time_array: FloatArray,
        values_array: FloatArray,
        colors: list | np.ndarray | None,
    ) -> None:
        """Plot control trajectory as step function (piecewise constant)."""
        if len(time_array) == 0:
            return

        extended_times = np.copy(time_array)
        extended_values = np.copy(values_array)

        if len(time_array) > 0:
            if self.final_time is not None and self.final_time > time_array[-1]:
                extended_times = np.append(extended_times, self.final_time)
                extended_values = np.append(extended_values, values_array[-1])
            elif len(time_array) > 1:
                dt = time_array[-1] - time_array[-2]
                extended_times = np.append(extended_times, time_array[-1] + dt * 0.1)
                extended_values = np.append(extended_values, values_array[-1])
            else:
                extended_times = np.append(extended_times, time_array[-1] + 1.0)
                extended_values = np.append(extended_values, values_array[-1])

        if colors is None or len(colors) == 0:
            ax.step(extended_times, extended_values, where="post", color="b", linewidth=1.5)
            ax.plot(time_array, values_array, "bo", markersize=4)
        else:
            intervals = self._get_mesh_intervals()
            if len(intervals) > 0:
                for k, (t_start, t_end) in enumerate(intervals):
                    mask = (time_array >= t_start - 1e-10) & (time_array <= t_end + 1e-10)
                    if np.any(mask):
                        interval_times = time_array[mask]
                        interval_values = values_array[mask]

                        if len(interval_times) > 0 and interval_times[-1] < t_end - 1e-10:
                            interval_times = np.append(interval_times, t_end)
                            interval_values = np.append(interval_values, interval_values[-1])

                        color = colors[k % len(colors)]
                        ax.step(
                            interval_times,
                            interval_values,
                            where="post",
                            color=color,
                            linewidth=1.5,
                        )
                        ax.plot(
                            time_array[mask], values_array[mask], "o", color=color, markersize=4
                        )
            else:
                ax.step(extended_times, extended_values, where="post", color="b", linewidth=1.5)
                ax.plot(time_array, values_array, "bo", markersize=4)

    def _plot_state_linear(
        self,
        ax: MplAxes,
        time_array: FloatArray,
        values_array: FloatArray,
        colors: list | np.ndarray | None,
    ) -> None:
        """Plot state trajectory with linear interpolation (smooth curves)."""
        if len(time_array) == 0:
            return

        if colors is None or len(colors) == 0:
            ax.plot(time_array, values_array, "b.-", linewidth=1.5, markersize=3)
        else:
            intervals = self._get_mesh_intervals()
            if len(intervals) > 0:
                for k, (t_start, t_end) in enumerate(intervals):
                    mask = (time_array >= t_start - 1e-10) & (time_array <= t_end + 1e-10)
                    if np.any(mask):
                        ax.plot(
                            time_array[mask],
                            values_array[mask],
                            color=colors[k % len(colors)],
                            marker=".",
                            linestyle="-",
                            linewidth=1.5,
                            markersize=7,
                        )
            else:
                ax.plot(time_array, values_array, "b.-", linewidth=1.5, markersize=3)

    def _get_interval_colors(self) -> FloatArray | None:
        """Get colors for mesh intervals."""
        if self.mesh_nodes is None or len(self.mesh_nodes) <= 1:
            return None

        num_intervals = len(self.mesh_nodes) - 1
        colormap = plt.get_cmap("viridis")
        color_values = np.linspace(0, 1, num_intervals, dtype=np.float64)
        colors = colormap(color_values)
        return cast(FloatArray, colors)

    def _get_mesh_intervals(self) -> list[tuple[float, float]]:
        """Get mesh interval boundaries in physical time."""
        if self.mesh_nodes is None or self.initial_time is None or self.final_time is None:
            return []

        alpha = (self.final_time - self.initial_time) / 2.0
        alpha_0 = (self.final_time + self.initial_time) / 2.0
        mesh_phys = alpha * self.mesh_nodes + alpha_0

        return [(mesh_phys[i], mesh_phys[i + 1]) for i in range(len(mesh_phys) - 1)]

    def _add_mesh_legend(self, fig: MplFigure, colors: np.ndarray) -> None:
        """Add mesh interval legend."""
        if not self.mesh_intervals:
            return

        handles = [
            Line2D([0], [0], color=colors[k % len(colors)], lw=2)
            for k in range(len(self.mesh_intervals))
        ]
        labels = [
            f"Interval {k} (N={self.mesh_intervals[k]})" for k in range(len(self.mesh_intervals))
        ]

        fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

    def summary(self) -> None:
        """Print a summary of the solution results."""
        print(f"Solution Status: {'Success' if self.success else 'Failed'}")
        if self.success:
            print(f"  Objective: {self.objective}")
            print(f"  Time: {self.initial_time} → {self.final_time}")
            print(f"  States: {len(self._state_names)} {self._state_names}")
            print(f"  Controls: {len(self._control_names)} {self._control_names}")
            if self.mesh_nodes is not None:
                print(f"  Mesh intervals: {len(self.mesh_nodes) - 1}")
        else:
            print(f"  Message: {self.message}")
