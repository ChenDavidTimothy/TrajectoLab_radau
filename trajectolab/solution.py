"""
solution.py

Defines classes to store and interact with the solution
of an optimal control problem.
"""

from collections.abc import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D

from .direct_solver import OptimalControlSolution
from .tl_types import FloatArray, IntArray, ProblemProtocol, SymType


class IntervalData:
    """Holds data specific to a single interval of the solution."""

    def __init__(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        Nk: int | None = None,
        time_states_segment: FloatArray | None = None,
        states_segment: Sequence[FloatArray] | None = None,
        time_controls_segment: FloatArray | None = None,
        controls_segment: Sequence[FloatArray] | None = None,
    ) -> None:
        self.t_start: float | None = t_start
        self.t_end: float | None = t_end
        self.Nk: int | None = Nk

        self.time_states_segment: FloatArray = (
            time_states_segment
            if time_states_segment is not None
            else np.array([], dtype=np.float64)
        )
        self.states_segment: list[FloatArray] = (
            list(states_segment) if states_segment is not None else []
        )
        self.time_controls_segment: FloatArray = (
            time_controls_segment
            if time_controls_segment is not None
            else np.array([], dtype=np.float64)
        )
        self.controls_segment: list[FloatArray] = (
            list(controls_segment) if controls_segment is not None else []
        )


class Solution:
    """Processes and provides access to the solution of an optimal control problem."""

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
        self._problem: ProblemProtocol | None = problem
        self._raw_solution: OptimalControlSolution | None = raw_solution

        self._state_names: list[str] = []
        if problem and hasattr(problem, "_states") and hasattr(problem._states, "keys"):
            self._state_names = list(problem._states.keys())

        self._control_names: list[str] = []
        if problem and hasattr(problem, "_controls") and hasattr(problem._controls, "keys"):
            self._control_names = list(problem._controls.keys())

        # Map to store symbolic variables to their names
        self._sym_state_map: dict[SymType, str] = {}
        self._sym_control_map: dict[SymType, str] = {}

        # If the problem has symbolic variables, create mappings
        if problem and hasattr(problem, "_sym_states"):
            for name, sym in problem._sym_states.items():
                self._sym_state_map[sym] = name

        if problem and hasattr(problem, "_sym_controls"):
            for name, sym in problem._sym_controls.items():
                self._sym_control_map[sym] = name

        self.success: bool = False
        self.message: str = "No solution data"
        self.initial_time: float | None = None
        self.final_time: float | None = None
        self.objective: float | None = None
        self.integrals: FloatArray | None = None

        self._time_states: FloatArray | None = None
        self._states: list[FloatArray] | None = None
        self._time_controls: FloatArray | None = None
        self._controls: list[FloatArray] | None = None

        self._polynomial_degrees: IntArray | None = None
        self._mesh_points_normalized: FloatArray | None = None
        self._mesh_points_time: FloatArray | None = None

        if raw_solution:
            self._extract_solution_data(raw_solution)

    def _extract_solution_data(self, raw_solution: OptimalControlSolution) -> None:
        self.success = bool(getattr(raw_solution, "success", False))
        self.message = str(getattr(raw_solution, "message", "Unknown"))

        initial_time_val = getattr(raw_solution, "initial_time_variable", None)
        self.initial_time = float(initial_time_val) if initial_time_val is not None else None

        final_time_val = getattr(raw_solution, "terminal_time_variable", None)
        self.final_time = float(final_time_val) if final_time_val is not None else None

        objective_val = getattr(raw_solution, "objective", None)
        self.objective = float(objective_val) if objective_val is not None else None

        integrals_val = getattr(raw_solution, "integrals", None)
        self.integrals = (
            np.asarray(integrals_val, dtype=np.float64) if integrals_val is not None else None
        )

        time_states_val = getattr(raw_solution, "time_states", None)
        self._time_states = (
            np.asarray(time_states_val, dtype=np.float64) if time_states_val is not None else None
        )

        states_val = getattr(raw_solution, "states", None)
        if states_val is not None:
            self._states = [np.asarray(s, dtype=np.float64) for s in states_val]

        time_controls_val = getattr(raw_solution, "time_controls", None)
        self._time_controls = (
            np.asarray(time_controls_val, dtype=np.float64)
            if time_controls_val is not None
            else None
        )

        controls_val = getattr(raw_solution, "controls", None)
        if controls_val is not None:
            self._controls = [np.asarray(c, dtype=np.float64) for c in controls_val]

        poly_degrees_val = getattr(raw_solution, "num_collocation_nodes_per_interval", None)
        if poly_degrees_val is not None:
            self._polynomial_degrees = np.asarray(poly_degrees_val, dtype=np.int_)

        mesh_norm_val = getattr(raw_solution, "global_normalized_mesh_nodes", None)
        if mesh_norm_val is not None:
            self._mesh_points_normalized = np.asarray(mesh_norm_val, dtype=np.float64)

        if (
            self.initial_time is not None
            and self.final_time is not None
            and self._mesh_points_normalized is not None
        ):
            alpha = (self.final_time - self.initial_time) / 2.0
            alpha_0 = (self.final_time + self.initial_time) / 2.0
            self._mesh_points_time = alpha * self._mesh_points_normalized + alpha_0

    @property
    def num_intervals(self) -> int:
        if self._polynomial_degrees is not None:
            return len(self._polynomial_degrees)
        if self._mesh_points_normalized is not None:
            return max(0, len(self._mesh_points_normalized) - 1)
        return 0

    @property
    def polynomial_degrees(self) -> IntArray | None:
        return self._polynomial_degrees

    @property
    def mesh_points(self) -> FloatArray | None:
        return self._mesh_points_time

    @property
    def mesh_points_normalized(self) -> FloatArray | None:
        return self._mesh_points_normalized

    def _get_state_index(self, state_name_or_index: str | int) -> int | None:
        if isinstance(state_name_or_index, int):
            return (
                state_name_or_index if 0 <= state_name_or_index < len(self._state_names) else None
            )
        try:
            return self._state_names.index(state_name_or_index)
        except ValueError:
            return None

    def _get_control_index(self, control_name_or_index: str | int) -> int | None:
        if isinstance(control_name_or_index, int):
            return (
                control_name_or_index
                if 0 <= control_name_or_index < len(self._control_names)
                else None
            )
        try:
            return self._control_names.index(control_name_or_index)
        except ValueError:
            return None

    def get_state_trajectory(
        self, state_name_or_index: str | int | SymType
    ) -> tuple[FloatArray, FloatArray]:
        """
        Get state trajectory data.

        Args:
            state_name_or_index: State name, index, or symbolic variable

        Returns:
            Tuple of (time_array, value_array)
        """
        # Handle symbolic variable case
        if hasattr(state_name_or_index, "is_symbolic") or (
            hasattr(state_name_or_index, "is_constant")
            and not (isinstance(state_name_or_index, int | str | float))
        ):
            sym_var = state_name_or_index
            if sym_var in self._sym_state_map:
                state_name_or_index = self._sym_state_map[sym_var]
            else:
                # Search through the problem if the map is not populated
                if self._problem and hasattr(self._problem, "_sym_states"):
                    for name, var in self._problem._sym_states.items():
                        if var is sym_var:
                            state_name_or_index = name
                            self._sym_state_map[sym_var] = name
                            break

        # Proceed with standard lookup
        index = self._get_state_index(state_name_or_index)
        empty_arr = np.array([], dtype=np.float64)
        time_arr = self._time_states if self._time_states is not None else empty_arr

        if index is None or self._states is None or not (0 <= index < len(self._states)):
            return time_arr, empty_arr
        return time_arr, self._states[index]

    def get_control_trajectory(
        self, control_name_or_index: str | int | SymType
    ) -> tuple[FloatArray, FloatArray]:
        """
        Get control trajectory data.

        Args:
            control_name_or_index: Control name, index, or symbolic variable

        Returns:
            Tuple of (time_array, value_array)
        """
        # Handle symbolic variable case
        if hasattr(control_name_or_index, "is_symbolic") or (
            hasattr(control_name_or_index, "is_constant")
            and not (isinstance(control_name_or_index, int | str | float))
        ):
            sym_var = control_name_or_index
            if sym_var in self._sym_control_map:
                control_name_or_index = self._sym_control_map[sym_var]
            else:
                # Search through the problem if the map is not populated
                if self._problem and hasattr(self._problem, "_sym_controls"):
                    for name, var in self._problem._sym_controls.items():
                        if var is sym_var:
                            control_name_or_index = name
                            self._sym_control_map[sym_var] = name
                            break

        # Proceed with standard lookup
        index = self._get_control_index(control_name_or_index)
        empty_arr = np.array([], dtype=np.float64)
        time_arr = self._time_controls if self._time_controls is not None else empty_arr

        if index is None or self._controls is None or not (0 <= index < len(self._controls)):
            return time_arr, empty_arr
        return time_arr, self._controls[index]

    def interpolate_state(
        self, state_name_or_index: str | int | SymType, time_point: float | FloatArray
    ) -> float | FloatArray | None:
        """
        Interpolate state value at specified time point(s).

        Args:
            state_name_or_index: State name, index, or symbolic variable
            time_point: Time point(s) to interpolate at

        Returns:
            Interpolated state value(s)
        """
        # Handle symbolic variable case
        if hasattr(state_name_or_index, "is_symbolic") or (
            hasattr(state_name_or_index, "is_constant")
            and not (isinstance(state_name_or_index, int | str | float))
        ):
            sym_var = state_name_or_index
            if sym_var in self._sym_state_map:
                state_name_or_index = self._sym_state_map[sym_var]
            else:
                # Search through the problem if the map is not populated
                if self._problem and hasattr(self._problem, "_sym_states"):
                    for name, var in self._problem._sym_states.items():
                        if var is sym_var:
                            state_name_or_index = name
                            self._sym_state_map[sym_var] = name
                            break

        time, values = self.get_state_trajectory(state_name_or_index)
        if time.size == 0 or values.size == 0:
            return None
        return np.interp(time_point, time, values)

    def interpolate_control(
        self, control_name_or_index: str | int | SymType, time_point: float | FloatArray
    ) -> float | FloatArray | None:
        """
        Interpolate control value at specified time point(s).

        Args:
            control_name_or_index: Control name, index, or symbolic variable
            time_point: Time point(s) to interpolate at

        Returns:
            Interpolated control value(s)
        """
        # Handle symbolic variable case
        if hasattr(control_name_or_index, "is_symbolic") or (
            hasattr(control_name_or_index, "is_constant")
            and not (isinstance(control_name_or_index, int | str | float))
        ):
            sym_var = control_name_or_index
            if sym_var in self._sym_control_map:
                control_name_or_index = self._sym_control_map[sym_var]
            else:
                # Search through the problem if the map is not populated
                if self._problem and hasattr(self._problem, "_sym_controls"):
                    for name, var in self._problem._sym_controls.items():
                        if var is sym_var:
                            control_name_or_index = name
                            self._sym_control_map[sym_var] = name
                            break

        time, values = self.get_control_trajectory(control_name_or_index)
        if time.size == 0 or values.size == 0:
            return None
        return np.interp(time_point, time, values)

    def get_data_for_interval(self, interval_index: int) -> IntervalData | None:
        if not (0 <= interval_index < self.num_intervals):
            return None
        if self._mesh_points_time is None or self._polynomial_degrees is None:
            return None

        interval_t_start = self._mesh_points_time[interval_index]
        interval_t_end = self._mesh_points_time[interval_index + 1]
        nk_interval = self._polynomial_degrees[interval_index]

        time_states_segment, states_segment = self._extract_segment_data(
            self._time_states, self._states, interval_t_start, interval_t_end
        )
        time_controls_segment, controls_segment = self._extract_segment_data(
            self._time_controls, self._controls, interval_t_start, interval_t_end
        )

        return IntervalData(
            t_start=interval_t_start,
            t_end=interval_t_end,
            Nk=nk_interval,
            time_states_segment=time_states_segment,
            states_segment=states_segment,
            time_controls_segment=time_controls_segment,
            controls_segment=controls_segment,
        )

    def get_all_interval_data(self) -> list[IntervalData | None]:
        return [self.get_data_for_interval(i) for i in range(self.num_intervals)]

    def _extract_segment_data(
        self,
        time_array: FloatArray | None,
        data_arrays: list[FloatArray] | None,
        interval_t_start: float,
        interval_t_end: float,
        epsilon: float = 1e-9,
    ) -> tuple[FloatArray, list[FloatArray]]:
        empty_time_segment = np.array([], dtype=np.float64)
        num_data_arrays = len(data_arrays) if data_arrays else 0
        empty_data_segments = [np.array([], dtype=np.float64) for _ in range(num_data_arrays)]

        if time_array is None or not data_arrays or time_array.size == 0:
            return empty_time_segment, empty_data_segments

        sort_indices = np.argsort(time_array)
        sorted_time = time_array[sort_indices]

        start_idx = np.searchsorted(sorted_time, interval_t_start - epsilon, side="left")
        end_idx = np.searchsorted(sorted_time, interval_t_end + epsilon, side="right")

        if start_idx >= end_idx:
            return empty_time_segment, empty_data_segments

        actual_indices_in_interval = sort_indices[start_idx:end_idx]
        time_segment = time_array[actual_indices_in_interval]
        extracted_data_segments: list[FloatArray] = []

        for data_array_item in data_arrays:
            if data_array_item.size == time_array.size:
                extracted_data_segments.append(data_array_item[actual_indices_in_interval])
            else:
                extracted_data_segments.append(np.array([], dtype=np.float64))
        return time_segment, extracted_data_segments

    def plot(self, figsize: tuple[float, float] = (10.0, 8.0)) -> None:
        if not self.success:
            print("Cannot plot: Solution not successful")
            return

        num_states = len(self._state_names)
        num_controls = len(self._control_names)
        num_rows = num_states + num_controls

        if num_rows == 0:
            print("Cannot plot: No states or controls to plot.")
            return

        fig, axes_obj = plt.subplots(num_rows, 1, figsize=figsize, sharex=True)
        axes_list: list[MplAxes]
        if num_rows == 1:
            axes_list = [axes_obj]  # axes_obj is a single Axes object
        else:
            axes_list = list(axes_obj)  # axes_obj is a numpy array of Axes objects

        num_intervals = self.num_intervals
        if num_intervals == 0:
            if self._time_states is not None and self._states is not None:
                for i, name in enumerate(self._state_names):
                    if i < len(axes_list):
                        axes_list[i].plot(
                            self._time_states, self._states[i], marker=".", linestyle="-"
                        )
                        axes_list[i].set_ylabel(name)
                        axes_list[i].grid(True)
            if self._time_controls is not None and self._controls is not None:
                for i, name in enumerate(self._control_names):
                    ax_idx = num_states + i
                    if ax_idx < len(axes_list):
                        axes_list[ax_idx].plot(
                            self._time_controls, self._controls[i], marker=".", linestyle="-"
                        )
                        axes_list[ax_idx].set_ylabel(name)
                        axes_list[ax_idx].grid(True)
            if axes_list:
                axes_list[-1].set_xlabel("Time")
            plt.tight_layout()
            plt.show()
            return

        colors = plt.get_cmap("viridis")(np.linspace(0, 1, num_intervals))
        all_interval_data = self.get_all_interval_data()
        plot_row_idx = 0

        for state_idx, name in enumerate(self._state_names):
            ax = axes_list[plot_row_idx]
            for k, interval_data in enumerate(all_interval_data):
                if (
                    interval_data
                    and state_idx < len(interval_data.states_segment)
                    and interval_data.states_segment[state_idx].size > 0
                ):
                    ax.plot(
                        interval_data.time_states_segment,
                        interval_data.states_segment[state_idx],
                        color=colors[k % len(colors)],
                        marker=".",
                        linestyle="-",
                    )
            ax.set_ylabel(name)
            ax.grid(True)
            plot_row_idx += 1

        for control_idx, name in enumerate(self._control_names):
            ax = axes_list[plot_row_idx]
            for k, interval_data in enumerate(all_interval_data):
                if (
                    interval_data
                    and control_idx < len(interval_data.controls_segment)
                    and interval_data.controls_segment[control_idx].size > 0
                ):
                    ax.plot(
                        interval_data.time_controls_segment,
                        interval_data.controls_segment[control_idx],
                        color=colors[k % len(colors)],
                        marker=".",
                        linestyle="-",
                    )
            ax.set_ylabel(name)
            ax.grid(True)
            plot_row_idx += 1

        if self.polynomial_degrees is not None and num_intervals > 0:
            handles = [
                Line2D([0], [0], color=colors[k % len(colors)], lw=2) for k in range(num_intervals)
            ]
            labels = [
                f"Interval {k} (Nk={self.polynomial_degrees[k]})" for k in range(num_intervals)
            ]
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        axes_list[-1].set_xlabel("Time")
        rect_val: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.85 if num_intervals > 0 else 1.0,
            0.96,
        )
        plt.tight_layout(rect=rect_val)
        plt.show()

    def plot_states(
        self,
        state_names: Iterable[str | SymType] | None = None,
        figsize: tuple[float, float] = (10.0, 8.0),
    ) -> None:
        """
        Plot specific state trajectories.

        Args:
            state_names: Names or symbolic variables of states to plot
            figsize: Figure size
        """
        # Convert symbolic variables to names
        processed_names: list[str] = []
        if state_names is not None:
            for state in state_names:
                if hasattr(state, "is_symbolic") or (
                    hasattr(state, "is_constant") and not isinstance(state, int | str | float)
                ):
                    sym_var = state
                    if sym_var in self._sym_state_map:
                        processed_names.append(self._sym_state_map[sym_var])
                    else:
                        # Search through problem
                        if self._problem and hasattr(self._problem, "_sym_states"):
                            for name, var in self._problem._sym_states.items():
                                if var is sym_var:
                                    processed_names.append(name)
                                    self._sym_state_map[sym_var] = name
                                    break
                elif isinstance(state, str):
                    processed_names.append(state)

            self._plot_variables(
                variable_type="state",
                variable_names_to_plot=processed_names,
                all_variable_names=self._state_names,
                get_index_func=self._get_state_index,
                figsize=figsize,
            )
        else:
            self._plot_variables(
                variable_type="state",
                variable_names_to_plot=None,
                all_variable_names=self._state_names,
                get_index_func=self._get_state_index,
                figsize=figsize,
            )

    def plot_controls(
        self,
        control_names: Iterable[str | SymType] | None = None,
        figsize: tuple[float, float] = (10.0, 8.0),
    ) -> None:
        """
        Plot specific control trajectories.

        Args:
            control_names: Names or symbolic variables of controls to plot
            figsize: Figure size
        """
        # Convert symbolic variables to names
        processed_names: list[str] = []
        if control_names is not None:
            for control in control_names:
                if hasattr(control, "is_symbolic") or (
                    hasattr(control, "is_constant") and not isinstance(control, int | str | float)
                ):
                    sym_var = control
                    if sym_var in self._sym_control_map:
                        processed_names.append(self._sym_control_map[sym_var])
                    else:
                        # Search through problem
                        if self._problem and hasattr(self._problem, "_sym_controls"):
                            for name, var in self._problem._sym_controls.items():
                                if var is sym_var:
                                    processed_names.append(name)
                                    self._sym_control_map[sym_var] = name
                                    break
                elif isinstance(control, str):
                    processed_names.append(control)

            self._plot_variables(
                variable_type="control",
                variable_names_to_plot=processed_names,
                all_variable_names=self._control_names,
                get_index_func=self._get_control_index,
                figsize=figsize,
            )
        else:
            self._plot_variables(
                variable_type="control",
                variable_names_to_plot=None,
                all_variable_names=self._control_names,
                get_index_func=self._get_control_index,
                figsize=figsize,
            )

    def _plot_variables(
        self,
        variable_type: str,
        variable_names_to_plot: Iterable[str] | None,
        all_variable_names: list[str],
        get_index_func: Callable[[str | int], int | None],
        figsize: tuple[float, float],
    ) -> None:
        if not self.success:
            print(f"Cannot plot {variable_type}s: Solution not successful")
            return

        names_to_plot_list: list[str]
        if variable_names_to_plot is None:
            names_to_plot_list = all_variable_names
        else:
            names_to_plot_list = [
                name for name in variable_names_to_plot if name in all_variable_names
            ]

        if not names_to_plot_list:
            print(f"No valid {variable_type} names provided or available to plot.")
            return

        num_vars_to_plot = len(names_to_plot_list)
        fig, axes_obj = plt.subplots(num_vars_to_plot, 1, figsize=figsize, sharex=True)
        axes_list: list[MplAxes]
        if num_vars_to_plot == 1:
            axes_list = [axes_obj]
        else:
            axes_list = list(axes_obj)

        num_intervals = self.num_intervals
        all_interval_data = self.get_all_interval_data()

        colors = plt.get_cmap("viridis")(np.linspace(0, 1, max(1, num_intervals)))

        for i, name in enumerate(names_to_plot_list):
            ax = axes_list[i]
            var_idx = get_index_func(name)
            if var_idx is None:
                continue

            if num_intervals == 0:
                time_traj: FloatArray | None = None
                value_traj: FloatArray | None = None
                if (
                    variable_type == "state"
                    and self._time_states is not None
                    and self._states is not None
                    and var_idx < len(self._states)
                ):
                    time_traj, value_traj = self._time_states, self._states[var_idx]
                elif (
                    variable_type == "control"
                    and self._time_controls is not None
                    and self._controls is not None
                    and var_idx < len(self._controls)
                ):
                    time_traj, value_traj = self._time_controls, self._controls[var_idx]

                if time_traj is not None and value_traj is not None and time_traj.size > 0:
                    ax.plot(time_traj, value_traj, marker=".", linestyle="-")
            else:
                for k, interval_data_opt in enumerate(all_interval_data):
                    if interval_data_opt is None:
                        continue
                    interval_data = interval_data_opt

                    time_segment: FloatArray | None = None
                    value_segment: FloatArray | None = None

                    if variable_type == "state":
                        time_segment = interval_data.time_states_segment
                        if var_idx < len(interval_data.states_segment):
                            value_segment = interval_data.states_segment[var_idx]
                    elif variable_type == "control":
                        time_segment = interval_data.time_controls_segment
                        if var_idx < len(interval_data.controls_segment):
                            value_segment = interval_data.controls_segment[var_idx]

                    if (
                        time_segment is not None
                        and value_segment is not None
                        and time_segment.size > 0
                    ):
                        ax.plot(
                            time_segment,
                            value_segment,
                            color=colors[k % len(colors)],
                            marker=".",
                            linestyle="-",
                        )
            ax.set_ylabel(name)
            ax.grid(True)

        if num_intervals > 0 and self.polynomial_degrees is not None:
            handles = [
                Line2D([0], [0], color=colors[k % len(colors)], lw=2) for k in range(num_intervals)
            ]
            labels = [
                f"Interval {k} (Nk={self.polynomial_degrees[k]})" for k in range(num_intervals)
            ]
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        axes_list[-1].set_xlabel("Time")
        rect_val: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.85 if num_intervals > 0 else 1.0,
            0.96,
        )
        plt.tight_layout(rect=rect_val)
        plt.show()

    def get_symbolic_trajectory(self, var: SymType) -> tuple[FloatArray, FloatArray]:
        """
        Get the trajectory for a symbolic variable.

        Args:
            var: Symbolic variable

        Returns:
            Tuple of (time_array, value_array)
        """
        # First check if the variable is in our maps
        if var in self._sym_state_map:
            return self.get_state_trajectory(self._sym_state_map[var])
        elif var in self._sym_control_map:
            return self.get_control_trajectory(self._sym_control_map[var])

        # Identify the variable by searching the problem
        var_name = None
        var_type = None

        # Check state variables
        if self._problem:
            for name, sym in self._problem._sym_states.items():
                if var is sym:
                    var_name = name
                    var_type = "state"
                    # Update our map for future use
                    self._sym_state_map[var] = name
                    break

            # Check control variables
            if var_name is None:
                for name, sym in self._problem._sym_controls.items():
                    if var is sym:
                        var_name = name
                        var_type = "control"
                        # Update our map for future use
                        self._sym_control_map[var] = name
                        break

        # Return trajectory based on type
        if var_type == "state":
            return self.get_state_trajectory(var_name)
        elif var_type == "control":
            return self.get_control_trajectory(var_name)
        else:
            empty_arr = np.array([], dtype=np.float64)
            return empty_arr, empty_arr

    def interpolate_symbolic(
        self, var: SymType, time_point: float | FloatArray
    ) -> float | FloatArray | None:
        """
        Interpolate the value of a symbolic variable at given time points.

        Args:
            var: Symbolic variable
            time_point: Time point(s) to interpolate at

        Returns:
            Interpolated value(s)
        """
        # First check if the variable is in our maps
        if var in self._sym_state_map:
            return self.interpolate_state(self._sym_state_map[var], time_point)
        elif var in self._sym_control_map:
            return self.interpolate_control(self._sym_control_map[var], time_point)

        # Identify the variable by searching the problem
        var_name = None
        var_type = None

        # Check state variables
        if self._problem:
            for name, sym in self._problem._sym_states.items():
                if var is sym:
                    var_name = name
                    var_type = "state"
                    # Update our map for future use
                    self._sym_state_map[var] = name
                    break

            # Check control variables
            if var_name is None:
                for name, sym in self._problem._sym_controls.items():
                    if var is sym:
                        var_name = name
                        var_type = "control"
                        # Update our map for future use
                        self._sym_control_map[var] = name
                        break

        # Interpolate based on type
        if var_type == "state":
            return self.interpolate_state(var_name, time_point)
        elif var_type == "control":
            return self.interpolate_control(var_name, time_point)
        else:
            return None
