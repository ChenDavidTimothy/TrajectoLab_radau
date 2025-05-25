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

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
        """Initialize solution wrapper from raw optimization results."""
        if raw_solution is not None:
            self.success = raw_solution.success
            self.message = raw_solution.message
            self.initial_time = raw_solution.initial_time_variable
            self.final_time = raw_solution.terminal_time_variable
            self.objective = raw_solution.objective
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
            self.objective = None
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

    def plot(self, figsize: tuple[float, float] = (10.0, 8.0)) -> None:
        """Plot all state and control trajectories."""
        if not self.success:
            logger.warning("Cannot plot: Solution not successful")
            return

        num_states = len(self._state_names)
        num_controls = len(self._control_names)
        total_plots = num_states + num_controls

        if total_plots == 0:
            logger.info("No variables to plot")
            return

        fig, axes = plt.subplots(total_plots, 1, figsize=figsize, sharex=True)
        if total_plots == 1:
            axes = [axes]

        colors = self._get_interval_colors()

        for i, name in enumerate(self._state_names):
            self._plot_single_variable(axes[i], name, "state", colors)

        for i, name in enumerate(self._control_names):
            self._plot_single_variable(axes[num_states + i], name, "control", colors)

        axes[-1].set_xlabel("Time")
        if colors is not None and len(colors) > 0:
            self._add_mesh_legend(fig, colors)
        plt.tight_layout()
        plt.show()

    def plot_states(
        self, names: list[str] | None = None, figsize: tuple[float, float] = (10.0, 8.0)
    ) -> None:
        """Plot specific state variables."""
        names = names or self._state_names
        self._plot_variables(names, "state", figsize)

    def plot_controls(
        self, names: list[str] | None = None, figsize: tuple[float, float] = (10.0, 8.0)
    ) -> None:
        """Plot specific control variables."""
        names = names or self._control_names
        self._plot_variables(names, "control", figsize)

    def _plot_variables(
        self, names: list[str], var_type: str, figsize: tuple[float, float]
    ) -> None:
        """Plot list of variables of same type."""
        if not self.success or not names:
            reason = "solution not successful" if not self.success else "no variables specified"
            logger.warning("Cannot plot %ss: %s", var_type, reason)
            return

        fig, axes = plt.subplots(len(names), 1, figsize=figsize, sharex=True)
        if len(names) == 1:
            axes = [axes]

        colors = self._get_interval_colors()

        for i, name in enumerate(names):
            self._plot_single_variable(axes[i], name, var_type, colors)

        axes[-1].set_xlabel("Time")
        if colors is not None and len(colors) > 0:
            self._add_mesh_legend(fig, colors)
        plt.tight_layout()
        plt.show()

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
