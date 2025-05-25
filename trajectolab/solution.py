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

from .tl_types import FloatArray, OptimalControlSolution, ProblemProtocol, SymType


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


# Type aliases
_VariableIdentifier: TypeAlias = str | int | SymType
_TrajectoryTuple: TypeAlias = tuple[FloatArray, FloatArray]


class Solution:
    """
    User-friendly interface for optimal control problem solutions.

    The Solution class provides easy access to optimization results including
    trajectories, objective values, and built-in plotting capabilities. It wraps
    the raw solver output in a convenient interface.

    Attributes:
        success: Whether the optimization was successful
        message: Status message from the solver
        initial_time: Initial time value (t0)
        final_time: Final time value (tf)
        objective: Objective function value at the solution
        integrals: Values of integral expressions (if any)
        time_states: Time points for state trajectories
        states: List of state variable trajectories
        time_controls: Time points for control trajectories
        controls: List of control variable trajectories
        mesh_intervals: Polynomial degrees used in final mesh
        mesh_nodes: Normalized mesh node locations

    Note:
        Solutions are typically created by solve_fixed_mesh() or solve_adaptive().
        Direct instantiation is not recommended.
    """

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
        """Initialize solution wrapper from raw optimization results."""
        # DIRECT FIELD ACCESS - No wrapper forwarding
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
            # Default empty solution
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

        # Extract variable names for trajectory access
        if problem is not None:
            self._state_names = problem.get_ordered_state_names()
            self._control_names = problem.get_ordered_control_names()

            # Create symbol→name mapping for symbolic variable support
            self._sym_to_name: dict[SymType, str] = {}
            state_syms = problem.get_ordered_state_symbols()
            control_syms = problem.get_ordered_control_symbols()

            for name, sym in zip(self._state_names, state_syms, strict=False):
                self._sym_to_name[sym] = name
            for name, sym in zip(self._control_names, control_syms, strict=False):
                self._sym_to_name[sym] = name
        else:
            self._state_names = []
            self._control_names = []
            self._sym_to_name = {}

    # ========================================================================
    # TRAJECTORY ACCESS
    # ========================================================================

    def get_trajectory(self, identifier: _VariableIdentifier) -> _TrajectoryTuple:
        """
        Get time and value trajectory for any variable.

        Args:
            identifier: Variable identifier, can be:
                - Variable name as string (e.g., "position")
                - Variable index as integer
                - Symbolic variable from problem definition
                - StateVariableImpl or TimeVariableImpl wrapper objects

        Returns:
            Tuple of (time_array, values_array) where:
                - time_array: 1D array of time points
                - values_array: 1D array of variable values

        Example:
            >>> # Get trajectory by name
            >>> time, pos = solution.get_trajectory("position")
            >>>
            >>> # Get trajectory by symbolic variable
            >>> x = problem.state("position")
            >>> time, pos = solution.get_trajectory(x)
            >>>
            >>> # Plot trajectory
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(time, pos)
            >>> plt.xlabel("Time")
            >>> plt.ylabel("Position")
        """
        if not self.success:
            logger.warning("Cannot get trajectory: Solution not successful")
            empty = np.array([], dtype=np.float64)
            return empty, empty

        # Resolve identifier to name and type
        var_name, var_type = self._resolve_variable(identifier)
        if var_name is None:
            logger.warning("Variable '%s' not found in solution", identifier)
            empty = np.array([], dtype=np.float64)
            return empty, empty

        # Get data directly from stored fields
        if var_type == "state":
            time_array = self.time_states
            var_index = self._state_names.index(var_name)
            values_array = self.states[var_index]
        else:  # control
            time_array = self.time_controls
            var_index = self._control_names.index(var_name)
            values_array = self.controls[var_index]

        return time_array, values_array

    def _resolve_variable(self, identifier: _VariableIdentifier) -> tuple[str | None, str | None]:
        """Resolve variable identifier to (name, type)."""
        # Handle wrapper objects (StateVariableImpl, TimeVariableImpl)
        if hasattr(identifier, "_sym_var"):
            # This is a wrapper object - get the underlying symbol
            underlying_sym = identifier._sym_var
            if underlying_sym in self._sym_to_name:
                var_name = self._sym_to_name[underlying_sym]
            else:
                return None, None
        # Handle symbolic variables
        elif hasattr(identifier, "is_symbolic") or hasattr(identifier, "is_constant"):
            sym_identifier = cast(SymType, identifier)
            if sym_identifier in self._sym_to_name:
                var_name = self._sym_to_name[sym_identifier]
            else:
                return None, None
        else:
            var_name = str(identifier)

        # Determine type
        if var_name in self._state_names:
            return var_name, "state"
        elif var_name in self._control_names:
            return var_name, "control"
        else:
            return None, None

    # ========================================================================
    # PLOTTING INTERFACE
    # ========================================================================

    def plot(self, figsize: tuple[float, float] = (10.0, 8.0)) -> None:
        """
        Plot all state and control trajectories.

        Creates a multi-panel plot showing all state and control variables
        over time, with automatic mesh interval coloring if available.

        Args:
            figsize: Figure size as (width, height) in inches

        Example:
            >>> solution.plot()                    # Default size
            >>> solution.plot(figsize=(12, 10))   # Custom size
        """
        if not self.success:
            logger.warning("Cannot plot: Solution not successful")
            return

        num_states = len(self._state_names)
        num_controls = len(self._control_names)
        total_plots = num_states + num_controls

        if total_plots == 0:
            logger.info("No variables to plot")
            return

        # Create subplots
        fig, axes = plt.subplots(total_plots, 1, figsize=figsize, sharex=True)
        if total_plots == 1:
            axes = [axes]

        # Get colors for mesh intervals
        colors = self._get_interval_colors()

        # Plot states
        for i, name in enumerate(self._state_names):
            self._plot_single_variable(axes[i], name, "state", colors)

        # Plot controls
        for i, name in enumerate(self._control_names):
            self._plot_single_variable(axes[num_states + i], name, "control", colors)

        # Finalize
        axes[-1].set_xlabel("Time")
        if colors is not None and len(colors) > 0:
            self._add_mesh_legend(fig, colors)
        plt.tight_layout()
        plt.show()

    def plot_states(
        self, names: list[str] | None = None, figsize: tuple[float, float] = (10.0, 8.0)
    ) -> None:
        """
        Plot specific state variables.

        Args:
            names: List of state variable names to plot. If None, plots all states.
            figsize: Figure size as (width, height) in inches

        Example:
            >>> solution.plot_states(["position", "velocity"])
            >>> solution.plot_states()  # Plot all states
        """
        names = names or self._state_names
        self._plot_variables(names, "state", figsize)

    def plot_controls(
        self, names: list[str] | None = None, figsize: tuple[float, float] = (10.0, 8.0)
    ) -> None:
        """
        Plot specific control variables.

        Args:
            names: List of control variable names to plot. If None, plots all controls.
            figsize: Figure size as (width, height) in inches

        Example:
            >>> solution.plot_controls(["thrust", "steering"])
            >>> solution.plot_controls()  # Plot all controls
        """
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
        """
        Plot single variable with correct mathematical representation.

        States: Linear interpolation (smooth curves)
        Controls: Step functions (piecewise constant)
        """
        time_array, values_array = self.get_trajectory(name)

        if time_array.size == 0:
            return

        # Branch based on variable type for correct mathematical representation
        if var_type == "control":
            self._plot_control_step_function(ax, time_array, values_array, colors)
        else:  # state
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
        """
        Plot control trajectory as step function (piecewise constant).

        Each control value is held constant until the next time point,
        following the mathematical interpretation of discrete control optimization.
        """
        if len(time_array) == 0:
            return

        # Prepare extended arrays for proper step function visualization
        extended_times = np.copy(time_array)
        extended_values = np.copy(values_array)

        # Extend final control value to make it visible in step plot
        if len(time_array) > 0:
            if self.final_time is not None and self.final_time > time_array[-1]:
                # Extend to problem final time
                extended_times = np.append(extended_times, self.final_time)
                extended_values = np.append(extended_values, values_array[-1])
            elif len(time_array) > 1:
                # Extend by small amount based on time spacing
                dt = time_array[-1] - time_array[-2]
                extended_times = np.append(extended_times, time_array[-1] + dt * 0.1)
                extended_values = np.append(extended_values, values_array[-1])
            else:
                # Single point case
                extended_times = np.append(extended_times, time_array[-1] + 1.0)
                extended_values = np.append(extended_values, values_array[-1])

        # Plot step function with appropriate coloring
        if colors is None or len(colors) == 0:
            # Simple step function without mesh interval coloring
            ax.step(extended_times, extended_values, where="post", color="b", linewidth=1.5)
            ax.plot(time_array, values_array, "bo", markersize=4)  # Mark actual control points
        else:
            # Step function with mesh interval coloring
            intervals = self._get_mesh_intervals()
            if len(intervals) > 0:
                for k, (t_start, t_end) in enumerate(intervals):
                    # Find control points in this mesh interval
                    mask = (time_array >= t_start - 1e-10) & (time_array <= t_end + 1e-10)
                    if np.any(mask):
                        interval_times = time_array[mask]
                        interval_values = values_array[mask]

                        # Extend last control in interval to interval boundary
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
                # Fallback to simple step function
                ax.step(extended_times, extended_values, where="post", color="b", linewidth=1.5)
                ax.plot(time_array, values_array, "bo", markersize=4)

    def _plot_state_linear(
        self,
        ax: MplAxes,
        time_array: FloatArray,
        values_array: FloatArray,
        colors: list | np.ndarray | None,
    ) -> None:
        """
        Plot state trajectory with linear interpolation (smooth curves).

        Maintains existing behavior for states, which are continuous functions
        in the mathematical formulation.
        """
        if len(time_array) == 0:
            return

        # Preserve existing state plotting implementation
        if colors is None or len(colors) == 0:
            ax.plot(time_array, values_array, "b.-", linewidth=1.5, markersize=3)
        else:
            # Plot by mesh intervals with colors
            intervals = self._get_mesh_intervals()
            if len(intervals) > 0:
                for k, (t_start, t_end) in enumerate(intervals):
                    # Find state points in this mesh interval
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
                # Fallback to simple linear plot
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

        # Convert normalized mesh to physical time
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

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    def summary(self) -> None:
        """
        Print a summary of the solution results.

        Displays key information including success status, objective value,
        time horizon, variable counts, and mesh configuration.

        Example:
            >>> solution.summary()
            Solution Status: Success
              Objective: 2.345e-02
              Time: 0.000 → 3.142
              States: 3
              Controls: 2
              Mesh intervals: 5
        """
        print(f"Solution Status: {'Success' if self.success else 'Failed'}")
        if self.success:
            print(f"  Objective: {self.objective}")
            print(f"  Time: {self.initial_time} → {self.final_time}")
            print(f"  States: {len(self._state_names)}")
            print(f"  Controls: {len(self._control_names)}")
            if self.mesh_nodes is not None:
                print(f"  Mesh intervals: {len(self.mesh_nodes) - 1}")
        else:
            print(f"  Message: {self.message}")
