from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from numpy.typing import NDArray


class IntervalData:
    def __init__(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        Nk: Optional[int] = None,
        time_states_segment: Optional[NDArray[np.float64]] = None,
        states_segment: Optional[List[NDArray[np.float64]]] = None,
        time_controls_segment: Optional[NDArray[np.float64]] = None,
        controls_segment: Optional[List[NDArray[np.float64]]] = None,
    ):
        self.t_start: Optional[float] = t_start
        self.t_end: Optional[float] = t_end
        self.Nk: Optional[int] = Nk
        self.time_states_segment: NDArray[np.float64] = (
            time_states_segment if time_states_segment is not None else np.array([])
        )
        self.states_segment: List[NDArray[np.float64]] = (
            states_segment if states_segment is not None else []
        )
        self.time_controls_segment: NDArray[np.float64] = (
            time_controls_segment if time_controls_segment is not None else np.array([])
        )
        self.controls_segment: List[NDArray[np.float64]] = (
            controls_segment if controls_segment is not None else []
        )


class Solution:
    def __init__(self, solution: Any, problem: Any):
        self._problem = problem
        self._solution = solution
        self._state_names: List[str] = list(problem._states.keys()) if problem else []
        self._control_names: List[str] = list(problem._controls.keys()) if problem else []

        # Extract basic solution properties
        self.success: bool = False
        self.message: str = "No solution data"
        self.initial_time: Optional[float] = None
        self.final_time: Optional[float] = None
        self.objective: Optional[float] = None
        self.integrals: Optional[Union[float, List[float]]] = None

        # Extract trajectories
        self._time_states: Optional[NDArray[np.float64]] = None
        self._states: Optional[List[NDArray[np.float64]]] = None
        self._time_controls: Optional[NDArray[np.float64]] = None
        self._controls: Optional[List[NDArray[np.float64]]] = None

        # Extract mesh information
        self._polynomial_degrees: Optional[List[int]] = None
        self._mesh_points_normalized: Optional[NDArray[np.float64]] = None
        self._mesh_points_time: Optional[NDArray[np.float64]] = None

        if solution:
            self._extract_solution_data(solution)

    def _extract_solution_data(self, solution: Any) -> None:
        # Basic properties
        self.success = getattr(solution, "success", False)
        self.message = getattr(solution, "message", "Unknown")
        self.initial_time = getattr(solution, "initial_time_variable", None)
        self.final_time = getattr(solution, "terminal_time_variable", None)
        self.objective = getattr(solution, "objective", None)
        self.integrals = getattr(solution, "integrals", None)

        # Trajectories
        self._time_states = getattr(solution, "time_states", None)
        self._states = getattr(solution, "states", None)
        self._time_controls = getattr(solution, "time_controls", None)
        self._controls = getattr(solution, "controls", None)

        # Mesh information
        self._polynomial_degrees = getattr(solution, "num_collocation_nodes_per_interval", None)
        self._mesh_points_normalized = getattr(solution, "global_normalized_mesh_nodes", None)

        # Calculate mesh points in time domain
        if (
            self.initial_time is not None
            and self.final_time is not None
            and self._mesh_points_normalized is not None
        ):
            alpha = (self.final_time - self.initial_time) / 2.0
            alpha_0 = (self.final_time + self.initial_time) / 2.0
            self._mesh_points_time = alpha * np.array(self._mesh_points_normalized) + alpha_0

    @property
    def num_intervals(self) -> int:
        if self._polynomial_degrees is not None:
            return len(self._polynomial_degrees)
        elif self._mesh_points_normalized is not None:
            return len(self._mesh_points_normalized) - 1
        return 0

    @property
    def polynomial_degrees(self) -> Optional[List[int]]:
        return self._polynomial_degrees

    @property
    def mesh_points(self) -> Optional[NDArray[np.float64]]:
        return self._mesh_points_time

    @property
    def mesh_points_normalized(self) -> Optional[NDArray[np.float64]]:
        return self._mesh_points_normalized

    def get_state_trajectory(
        self, state_name_or_index: Union[str, int]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        index = self._get_state_index(state_name_or_index)
        if index is None or self._states is None or self._time_states is None:
            return np.array([]), np.array([])
        if index >= len(self._states):
            return np.array([]), np.array([])
        return self._time_states, self._states[index]

    def get_control_trajectory(
        self, control_name_or_index: Union[str, int]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        index = self._get_control_index(control_name_or_index)
        if index is None or self._controls is None or self._time_controls is None:
            return np.array([]), np.array([])
        if index >= len(self._controls):
            return np.array([]), np.array([])
        return self._time_controls, self._controls[index]

    def interpolate_state(
        self, state_name_or_index: Union[str, int], time_point: float
    ) -> Optional[float]:
        time, values = self.get_state_trajectory(state_name_or_index)
        if len(time) == 0 or len(values) == 0:
            return None
        return float(np.interp(time_point, time, values))

    def interpolate_control(
        self, control_name_or_index: Union[str, int], time_point: float
    ) -> Optional[float]:
        time, values = self.get_control_trajectory(control_name_or_index)
        if len(time) == 0 or len(values) == 0:
            return None
        return float(np.interp(time_point, time, values))

    def get_data_for_interval(self, interval_index: int) -> Optional[IntervalData]:
        if not (0 <= interval_index < self.num_intervals):
            return None
        if self._mesh_points_time is None:
            return None

        interval_t_start = self._mesh_points_time[interval_index]
        interval_t_end = self._mesh_points_time[interval_index + 1]
        nk_interval = -1
        if self._polynomial_degrees is not None and interval_index < len(self._polynomial_degrees):
            nk_interval = self._polynomial_degrees[interval_index]

        # Extract segment data
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

    def get_all_interval_data(self) -> List[Optional[IntervalData]]:
        return [self.get_data_for_interval(i) for i in range(self.num_intervals)]

    def _extract_segment_data(
        self,
        time_array: Optional[NDArray[np.float64]],
        data_arrays: Optional[List[NDArray[np.float64]]],
        interval_t_start: float,
        interval_t_end: float,
        epsilon: float = 1e-9,
    ) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:
        if time_array is None or not data_arrays or len(time_array) == 0:
            return np.array([]), [
                np.array([]) for _ in range(len(data_arrays) if data_arrays else 1)
            ]

        sort_indices = np.argsort(time_array)
        sorted_time = time_array[sort_indices]

        idx_in_interval = np.where(
            (sorted_time >= interval_t_start - epsilon) & (sorted_time <= interval_t_end + epsilon)
        )[0]

        if len(idx_in_interval) == 0:
            return np.array([]), [np.array([]) for _ in range(len(data_arrays))]

        time_segment = sorted_time[idx_in_interval]
        data_segments: List[NDArray[np.float64]] = []

        for data_array in data_arrays:
            if len(data_array) == len(time_array):
                sorted_data = data_array[sort_indices]
                data_segments.append(sorted_data[idx_in_interval])
            else:
                data_segments.append(np.array([]))

        return time_segment, data_segments

    def _get_state_index(self, state_name_or_index: Union[str, int]) -> Optional[int]:
        if isinstance(state_name_or_index, int):
            return (
                state_name_or_index if 0 <= state_name_or_index < len(self._state_names) else None
            )
        elif isinstance(state_name_or_index, str):
            try:
                return self._state_names.index(state_name_or_index)
            except ValueError:
                return None
        return None

    def _get_control_index(self, control_name_or_index: Union[str, int]) -> Optional[int]:
        if isinstance(control_name_or_index, int):
            return (
                control_name_or_index
                if 0 <= control_name_or_index < len(self._control_names)
                else None
            )
        elif isinstance(control_name_or_index, str):
            try:
                return self._control_names.index(control_name_or_index)
            except ValueError:
                return None
        return None

    def plot(self, figsize: Tuple[float, float] = (10, 8)) -> None:
        if not self.success:
            print("Cannot plot: Solution not successful")
            return

        num_states = len(self._state_names)
        num_controls = len(self._control_names)

        num_rows = num_states + num_controls
        if num_rows == 0:
            return

        fig, axes = plt.subplots(num_rows, 1, figsize=figsize, sharex=True)
        if num_rows == 1:
            axes = [axes]

        # Get color map for intervals
        num_intervals = self.num_intervals
        colors = cm.viridis(np.linspace(0, 1, num_intervals))

        all_interval_data = self.get_all_interval_data()

        row = 0

        # Plot states by interval
        for _i, name in enumerate(self._state_names):
            state_idx = self._get_state_index(name)
            if state_idx is not None:
                for k, interval_data in enumerate(all_interval_data):
                    if interval_data is not None and state_idx < len(interval_data.states_segment):
                        if interval_data.states_segment[state_idx].size > 0:
                            axes[row].plot(
                                interval_data.time_states_segment,
                                interval_data.states_segment[state_idx],
                                color=colors[k],
                                marker=".",
                                linestyle="-",
                            )
            axes[row].set_ylabel(f"{name}")
            axes[row].grid(True)
            row += 1

        # Plot controls by interval
        for _i, name in enumerate(self._control_names):
            control_idx = self._get_control_index(name)
            if control_idx is not None:
                for k, interval_data in enumerate(all_interval_data):
                    if interval_data is not None and control_idx < len(
                        interval_data.controls_segment
                    ):
                        if interval_data.controls_segment[control_idx].size > 0:
                            axes[row].plot(
                                interval_data.time_controls_segment,
                                interval_data.controls_segment[control_idx],
                                color=colors[k],
                                marker=".",
                                linestyle="-",
                            )
            axes[row].set_ylabel(f"{name}")
            axes[row].grid(True)
            row += 1

        # Add legend
        if self.polynomial_degrees:
            handles: List[Line2D] = []
            labels: List[str] = []
            for k in range(num_intervals):
                handles.append(Line2D([0], [0], color=colors[k], lw=2))
                labels.append(f"Interval {k} (Nk={self.polynomial_degrees[k]})")
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        axes[-1].set_xlabel("Time")
        plt.tight_layout(rect=(0, 0, 0.85, 0.96))
        plt.show()

    def plot_states(
        self, state_names: Optional[List[str]] = None, figsize: Tuple[float, float] = (10, 8)
    ) -> None:
        if not self.success:
            print("Cannot plot: Solution not successful")
            return

        if state_names is None:
            state_names = self._state_names

        if not state_names:
            return

        valid_states = [name for name in state_names if name in self._state_names]

        fig, axes = plt.subplots(len(valid_states), 1, figsize=figsize, sharex=True)
        if len(valid_states) == 1:
            axes = [axes]

        # Get color map for intervals
        num_intervals = self.num_intervals
        colors = cm.viridis(np.linspace(0, 1, num_intervals))

        all_interval_data = self.get_all_interval_data()

        for i, name in enumerate(valid_states):
            state_idx = self._get_state_index(name)
            if state_idx is not None:
                for k, interval_data in enumerate(all_interval_data):
                    if interval_data is not None and state_idx < len(interval_data.states_segment):
                        if interval_data.states_segment[state_idx].size > 0:
                            axes[i].plot(
                                interval_data.time_states_segment,
                                interval_data.states_segment[state_idx],
                                color=colors[k],
                                marker=".",
                                linestyle="-",
                            )
            axes[i].set_ylabel(f"{name}")
            axes[i].grid(True)

        # Add legend
        if self.polynomial_degrees:
            handles: List[Line2D] = []
            labels: List[str] = []
            for k in range(num_intervals):
                handles.append(Line2D([0], [0], color=colors[k], lw=2))
                labels.append(f"Interval {k} (Nk={self.polynomial_degrees[k]})")
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        axes[-1].set_xlabel("Time")
        plt.tight_layout(rect=(0, 0, 0.85, 0.96))
        plt.show()

    def plot_controls(
        self, control_names: Optional[List[str]] = None, figsize: Tuple[float, float] = (10, 8)
    ) -> None:
        if not self.success:
            print("Cannot plot: Solution not successful")
            return

        if control_names is None:
            control_names = self._control_names

        if not control_names:
            return

        valid_controls = [name for name in control_names if name in self._control_names]

        fig, axes = plt.subplots(len(valid_controls), 1, figsize=figsize, sharex=True)
        if len(valid_controls) == 1:
            axes = [axes]

        # Get color map for intervals
        num_intervals = self.num_intervals
        colors = cm.viridis(np.linspace(0, 1, num_intervals))

        all_interval_data = self.get_all_interval_data()

        for i, name in enumerate(valid_controls):
            control_idx = self._get_control_index(name)
            if control_idx is not None:
                for k, interval_data in enumerate(all_interval_data):
                    if interval_data is not None and control_idx < len(
                        interval_data.controls_segment
                    ):
                        if interval_data.controls_segment[control_idx].size > 0:
                            axes[i].plot(
                                interval_data.time_controls_segment,
                                interval_data.controls_segment[control_idx],
                                color=colors[k],
                                marker=".",
                                linestyle="-",
                            )
            axes[i].set_ylabel(f"{name}")
            axes[i].grid(True)

        # Add legend
        if self.polynomial_degrees:
            handles: List[Line2D] = []
            labels: List[str] = []
            for k in range(num_intervals):
                handles.append(Line2D([0], [0], color=colors[k], lw=2))
                labels.append(f"Interval {k} (Nk={self.polynomial_degrees[k]})")
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        axes[-1].set_xlabel("Time")
        plt.tight_layout(rect=(0, 0, 0.85, 0.96))
        plt.show()

    def to_csv(self, filename: str) -> None:
        if not self.success:
            print("Cannot export: Solution not successful")
            return

        # Combine state and control data
        data: Dict[str, Any] = {}

        # Add time
        if self._time_states is not None and len(self._time_states) > 0:
            data["time"] = self._time_states

        # Add states
        for i, name in enumerate(self._state_names):
            if self._states is not None and i < len(self._states):
                data[name] = self._states[i]

        # Add controls (needs interpolation to match state times)
        for i, name in enumerate(self._control_names):
            if (
                self._controls is not None
                and i < len(self._controls)
                and self._time_controls is not None
            ):
                if self._time_states is not None and len(self._time_states) > 0:
                    data[name] = np.interp(
                        self._time_states, self._time_controls, self._controls[i]
                    )

        # Create dataframe and save
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def save(self, filename: str) -> None:
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "Solution":
        import pickle

        with open(filename, "rb") as f:
            return pickle.load(f)
