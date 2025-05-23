"""
Solution interface - SIMPLIFIED.
Removed ALL wrapper forwarding, uses direct field access with integrated plotting.
"""

from typing import TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D

from .tl_types import FloatArray, OptimalControlSolution, ProblemProtocol, SymType


# Type aliases
_VariableIdentifier: TypeAlias = str | int | SymType
_TrajectoryTuple: TypeAlias = tuple[FloatArray, FloatArray]


class Solution:
    """SIMPLIFIED solution interface with direct field access and integrated plotting."""

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
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
    # TRAJECTORY ACCESS - Direct from stored data
    # ========================================================================

    def get_trajectory(self, identifier: _VariableIdentifier) -> _TrajectoryTuple:
        """Get trajectory for any variable (state or control)."""
        if not self.success:
            print("Warning: Solution not successful")
            empty = np.array([], dtype=np.float64)
            return empty, empty

        # Resolve identifier to name and type
        var_name, var_type = self._resolve_variable(identifier)
        if var_name is None:
            print(f"Warning: Variable '{identifier}' not found")
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
        # Handle symbolic variables
        if hasattr(identifier, "is_symbolic") or hasattr(identifier, "is_constant"):
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
    # INTEGRATED PLOTTING - No separate module needed
    # ========================================================================

    def plot(self, figsize: tuple[float, float] = (10.0, 8.0)) -> None:
        """Plot all states and controls."""
        if not self.success:
            print("Cannot plot: Solution not successful")
            return

        num_states = len(self._state_names)
        num_controls = len(self._control_names)
        total_plots = num_states + num_controls

        if total_plots == 0:
            print("Nothing to plot")
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
        """Plot specific states."""
        names = names or self._state_names
        self._plot_variables(names, "state", figsize)

    def plot_controls(
        self, names: list[str] | None = None, figsize: tuple[float, float] = (10.0, 8.0)
    ) -> None:
        """Plot specific controls."""
        names = names or self._control_names
        self._plot_variables(names, "control", figsize)

    def _plot_variables(
        self, names: list[str], var_type: str, figsize: tuple[float, float]
    ) -> None:
        """Plot list of variables of same type."""
        if not self.success or not names:
            print(f"Cannot plot {var_type}s")
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
        """Plot single variable with mesh interval coloring."""
        time_array, values_array = self.get_trajectory(name)

        if time_array.size == 0:
            return

        # Simple plot if no mesh intervals
        if colors is None or len(colors) == 0:
            ax.plot(time_array, values_array, "b.-", linewidth=1.5, markersize=3)
        else:
            # Plot by intervals with colors
            intervals = self._get_mesh_intervals()
            if len(intervals) > 0:
                for k, (t_start, t_end) in enumerate(intervals):
                    # Find points in this interval
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
                # Fallback to simple plot
                ax.plot(time_array, values_array, "b.-", linewidth=1.5, markersize=3)

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

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
        if self.mesh_intervals is None:
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
        """Print solution summary."""
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
