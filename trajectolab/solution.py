"""
solution.py

Defines classes to store and interact with the solution
of an optimal control problem.
"""

from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D

from .scaling import ScalingFactors
from .tl_types import FloatArray, IntArray, OptimalControlSolution, ProblemProtocol, SymType


# --- Type Aliases ---
_VariableIdentifier: TypeAlias = str | int | SymType
_TrajectoryTuple: TypeAlias = tuple[FloatArray, FloatArray]
_SymbolMap: TypeAlias = dict[SymType, str]
_InterpolationResult: TypeAlias = float | FloatArray | None
_PlotVariableNames: TypeAlias = Iterable[str | SymType] | None


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


class _Solution:
    """Processes and provides access to the solution of an optimal control problem."""

    def __init__(
        self, raw_solution: OptimalControlSolution | None, problem: ProblemProtocol | None
    ) -> None:
        self._problem: ProblemProtocol | None = problem
        self._raw_solution: OptimalControlSolution | None = raw_solution

        # Original state names from problem
        self._state_names: list[str] = []
        if problem and hasattr(problem, "_states") and hasattr(problem._states, "keys"):
            self._state_names = list(problem._states.keys())

        self._control_names: list[str] = []
        if problem and hasattr(problem, "_controls") and hasattr(problem._controls, "keys"):
            self._control_names = list(problem._controls.keys())

        # Map to store symbolic variables to their names (original implementation)
        self._sym_state_map: _SymbolMap = {}
        self._sym_control_map: _SymbolMap = {}

        # If the problem has symbolic variables, create mappings (original implementation)
        if problem and hasattr(problem, "_sym_states"):
            for name, sym in problem._sym_states.items():
                self._sym_state_map[sym] = name

        if problem and hasattr(problem, "_sym_controls"):
            for name, sym in problem._sym_controls.items():
                self._sym_control_map[sym] = name

        # *** AUTO-SCALING: Enhanced symbolic mapping for physical variables ***
        self._physical_symbol_to_name_map: dict[str, str] = {}  # Maps str(symbol) to physical name

        # *** Auto-scaling related fields ***
        self._auto_scaling_enabled = False
        self._scaling_factors: dict[str, dict[str, float]] = {}
        self._physical_to_scaled_map: dict[str, str] = {}
        self._scaled_to_physical_map: dict[str, str] = {}
        self._physical_symbols: dict[str, SymType] = {}
        self._physical_state_names: list[str] = []
        self._physical_control_names: list[str] = []

        # Initialize core solution data
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
            # *** Extract auto-scaling information ***
            self._extract_scaling_information(raw_solution)

    def _extract_scaling_information(self, raw_solution: OptimalControlSolution) -> None:
        """
        Extract auto-scaling information from the raw solution.

        Args:
            raw_solution: The raw solution containing scaling information
        """
        # Extract auto-scaling flag
        if hasattr(raw_solution, "auto_scaling_enabled"):
            self._auto_scaling_enabled = raw_solution.auto_scaling_enabled

        # Extract scaling factors
        if hasattr(raw_solution, "scaling_factors"):
            self._scaling_factors = raw_solution.scaling_factors

        # Extract variable mappings
        if hasattr(raw_solution, "physical_to_scaled_map"):
            self._physical_to_scaled_map = raw_solution.physical_to_scaled_map

        if hasattr(raw_solution, "scaled_to_physical_map"):
            self._scaled_to_physical_map = raw_solution.scaled_to_physical_map

        if hasattr(raw_solution, "physical_symbols"):
            self._physical_symbols = raw_solution.physical_symbols

        # If auto-scaling is enabled, build physical variable name lists and symbol mappings
        if self._auto_scaling_enabled:
            self._build_physical_variable_lists()
            self._build_physical_symbol_mappings()

    def _build_physical_variable_lists(self) -> None:
        """Build lists of physical variable names for auto-scaling."""
        # Extract physical state names from scaled names
        self._physical_state_names = []
        for scaled_name in self._state_names:
            if scaled_name.endswith("_scaled") and scaled_name in self._scaled_to_physical_map:
                physical_name = self._scaled_to_physical_map[scaled_name]
                self._physical_state_names.append(physical_name)

        # Extract physical control names from scaled names
        self._physical_control_names = []
        for scaled_name in self._control_names:
            if scaled_name.endswith("_scaled") and scaled_name in self._scaled_to_physical_map:
                physical_name = self._scaled_to_physical_map[scaled_name]
                self._physical_control_names.append(physical_name)

    def _build_physical_symbol_mappings(self) -> None:
        """
        Build mapping from physical symbol string representations to variable names.
        This helps resolve auto-scaling symbolic variables.
        """
        for physical_name, physical_symbol in self._physical_symbols.items():
            try:
                # Use string representation as key for lookup
                symbol_str = str(physical_symbol)
                self._physical_symbol_to_name_map[symbol_str] = physical_name
            except Exception as e:
                print(
                    f"Warning: Could not create string mapping for physical symbol '{physical_name}': {e}"
                )

    def _extract_solution_data(self, raw_solution: OptimalControlSolution) -> None:
        """Extract solution data from raw solution (original implementation)."""
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

    def _unscale_values(self, scaled_values: FloatArray, physical_var_name: str) -> FloatArray:
        """
        Unscale values from scaled space to physical space.

        Args:
            scaled_values: Values in scaled space
            physical_var_name: Name of the physical variable

        Returns:
            Values in physical space
        """
        if not self._auto_scaling_enabled or physical_var_name not in self._scaling_factors:
            return scaled_values

        # Convert scaling factors dict to ScalingFactors object
        sf_dict = self._scaling_factors[physical_var_name]
        factors = ScalingFactors(v=sf_dict["v"], r=sf_dict["r"], rule=str(sf_dict["rule"]))

        return unscale_values(scaled_values, factors)

    def _resolve_variable_for_trajectory_access(
        self, identifier: _VariableIdentifier, variable_type: str
    ) -> tuple[_VariableIdentifier, str | None]:
        """
        Resolve variable identifier for trajectory access, handling auto-scaling.

        Args:
            identifier: Variable identifier (name, index, or symbolic)
            variable_type: "state" or "control"

        Returns:
            Tuple of (actual_identifier_for_lookup, physical_name_for_unscaling)
        """
        if not self._auto_scaling_enabled or not isinstance(identifier, str):
            return identifier, None

        var_name = identifier

        # Case 1: User requests a physical variable name (e.g., "h")
        # We need to find the corresponding scaled name for lookup
        if var_name in self._physical_to_scaled_map:
            scaled_name = self._physical_to_scaled_map[var_name]
            return (
                scaled_name,
                var_name,
            )  # Return scaled name for lookup, physical name for unscaling

        # Case 2: User requests a scaled variable name directly (e.g., "h_scaled")
        # We can use it directly for lookup, and get physical name for unscaling
        if var_name.endswith("_scaled") and var_name in self._scaled_to_physical_map:
            physical_name = self._scaled_to_physical_map[var_name]
            return (
                var_name,
                physical_name,
            )  # Return scaled name for lookup, physical name for unscaling

        # Case 3: Variable not related to auto-scaling
        return identifier, None

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

    def _get_variable_index(self, identifier: str | int, names_list: list[str]) -> int | None:
        """Get index for a variable by string name or integer index."""
        if isinstance(identifier, int):
            return identifier if 0 <= identifier < len(names_list) else None
        try:
            return names_list.index(identifier)
        except ValueError:
            return None

    def _resolve_symbolic_variable(
        self, sym_var: SymType, symbol_map: _SymbolMap, problem_attr: str
    ) -> str | None:
        """Resolve a symbolic variable to its string name."""
        if sym_var in symbol_map:
            return symbol_map[sym_var]

        # Search through the problem if the map is not populated
        if self._problem and hasattr(self._problem, problem_attr):
            attrs = getattr(self._problem, problem_attr)
            # Cast to ensure proper typing since we know these are symbol dictionaries
            if isinstance(attrs, dict):
                for name_any, var_any in attrs.items():
                    # Cast the items to ensure correct types
                    name = cast(str, name_any)
                    var = cast(SymType, var_any)
                    if var is sym_var:
                        symbol_map[sym_var] = name
                        return name
        return None

    def _resolve_physical_symbolic_variable(self, sym_var: SymType) -> str | None:
        """
        Resolve a physical symbolic variable (auto-scaling expression) to its physical name.

        Args:
            sym_var: Physical symbolic variable (expression like (scaled_var - r) / v)

        Returns:
            Physical variable name if found, None otherwise
        """
        if not self._auto_scaling_enabled:
            return None

        try:
            # Convert to string and look up in our mapping
            symbol_str = str(sym_var)
            return self._physical_symbol_to_name_map.get(symbol_str)
        except Exception:
            return None

    def _is_symbolic_variable(self, variable: _VariableIdentifier) -> bool:
        """Check if a variable identifier is a symbolic variable."""
        return hasattr(variable, "is_symbolic") or (
            hasattr(variable, "is_constant") and not isinstance(variable, int | str | float)
        )

    def _get_trajectory(
        self, identifier: _VariableIdentifier, variable_type: str
    ) -> _TrajectoryTuple:
        """
        Unified trajectory extraction for states and controls.

        Args:
            identifier: Variable name, index, or symbolic variable
            variable_type: Either "state" or "control"

        Returns:
            Tuple of (time_array, value_array)
        """
        empty_arr = np.array([], dtype=np.float64)

        # Select appropriate data based on variable type
        if variable_type == "state":
            time_arr = self._time_states if self._time_states is not None else empty_arr
            values_list = self._states
            names_list = self._state_names
            symbol_map = self._sym_state_map
            problem_attr = "_sym_states"
        elif variable_type == "control":
            time_arr = self._time_controls if self._time_controls is not None else empty_arr
            values_list = self._controls
            names_list = self._control_names
            symbol_map = self._sym_control_map
            problem_attr = "_sym_controls"
        else:
            raise ValueError(f"Invalid variable_type: {variable_type}")

        # Handle symbolic variable case
        if self._is_symbolic_variable(identifier):
            sym_var = cast(SymType, identifier)

            # First try regular symbolic resolution
            var_name = self._resolve_symbolic_variable(sym_var, symbol_map, problem_attr)

            # If that fails and auto-scaling is enabled, try physical symbol resolution
            if var_name is None and self._auto_scaling_enabled:
                var_name = self._resolve_physical_symbolic_variable(sym_var)

            if var_name is None:
                print(f"Warning: Variable '{sym_var}' not found")
                return time_arr, empty_arr
            identifier = var_name

        # Now identifier is guaranteed to be str | int
        string_or_int_identifier = cast(str | int, identifier)
        index = self._get_variable_index(string_or_int_identifier, names_list)

        if index is None or values_list is None or not (0 <= index < len(values_list)):
            return time_arr, empty_arr
        return time_arr, values_list[index]

    def _get_state_trajectory(self, identifier: _VariableIdentifier) -> _TrajectoryTuple:
        """
        Get state trajectory data with automatic unscaling.

        Args:
            identifier: Variable name, index, or symbolic variable

        Returns:
            Tuple of (time_array, value_array) in physical space if auto-scaling enabled
        """
        # Resolve the identifier for trajectory access
        lookup_identifier, physical_name = self._resolve_variable_for_trajectory_access(
            identifier, "state"
        )

        # Get the trajectory using the resolved identifier
        time_arr, values = self._get_trajectory(lookup_identifier, "state")

        # Unscale if needed
        if physical_name is not None:
            unscaled_values = self._unscale_values(values, physical_name)
            return time_arr, unscaled_values

        return time_arr, values

    def _get_control_trajectory(self, identifier: _VariableIdentifier) -> _TrajectoryTuple:
        """
        Get control trajectory data with automatic unscaling.

        Args:
            identifier: Variable name, index, or symbolic variable

        Returns:
            Tuple of (time_array, value_array) in physical space if auto-scaling enabled
        """
        # Resolve the identifier for trajectory access
        lookup_identifier, physical_name = self._resolve_variable_for_trajectory_access(
            identifier, "control"
        )

        # Get the trajectory using the resolved identifier
        time_arr, values = self._get_trajectory(lookup_identifier, "control")

        # Unscale if needed
        if physical_name is not None:
            unscaled_values = self._unscale_values(values, physical_name)
            return time_arr, unscaled_values

        return time_arr, values

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
    ) -> tuple[FloatArray, Sequence[FloatArray]]:
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

    def _resolve_variable_names(
        self, variables: _PlotVariableNames, variable_type: str
    ) -> list[str]:
        """Resolve variable names from a mix of strings and symbolic variables."""
        processed_names: list[str] = []
        if variables is None:
            return processed_names

        for var in variables:
            if self._is_symbolic_variable(var):
                sym_var = cast(SymType, var)
                if variable_type == "state":
                    resolved_name = self._resolve_symbolic_variable(
                        sym_var, self._sym_state_map, "_sym_states"
                    )
                    # Try physical symbol resolution if regular resolution fails
                    if resolved_name is None and self._auto_scaling_enabled:
                        resolved_name = self._resolve_physical_symbolic_variable(sym_var)
                elif variable_type == "control":
                    resolved_name = self._resolve_symbolic_variable(
                        sym_var, self._sym_control_map, "_sym_controls"
                    )
                    # Try physical symbol resolution if regular resolution fails
                    if resolved_name is None and self._auto_scaling_enabled:
                        resolved_name = self._resolve_physical_symbolic_variable(sym_var)
                else:
                    resolved_name = None

                if resolved_name is not None:
                    processed_names.append(resolved_name)
            elif isinstance(var, str):
                processed_names.append(var)

        return processed_names

    def _plot_variables(
        self,
        variable_type: str,
        variable_names_to_plot: Iterable[str] | None,
        figsize: tuple[float, float],
    ) -> None:
        """
        Unified plotting function for states and controls.

        Args:
            variable_type: Either "state" or "control"
            variable_names_to_plot: Names to plot (None means all)
            figsize: Figure size
        """
        if not self.success:
            print(f"Cannot plot {variable_type}s: Solution not successful")
            return

        # Select appropriate data based on variable type
        if variable_type == "state":
            all_names = self._state_names

            def get_index_func(name):
                return self._get_variable_index(name, self._state_names)
        elif variable_type == "control":
            all_names = self._control_names

            def get_index_func(name):
                return self._get_variable_index(name, self._control_names)
        else:
            raise ValueError(f"Invalid variable_type: {variable_type}")

        # Determine which variables to plot
        names_to_plot_list: list[str]
        if variable_names_to_plot is None:
            names_to_plot_list = all_names
        else:
            names_to_plot_list = [name for name in variable_names_to_plot if name in all_names]

        if not names_to_plot_list:
            print(f"No valid {variable_type} names provided or available to plot.")
            return

        # Create subplots
        num_vars_to_plot = len(names_to_plot_list)
        fig, axes_obj = plt.subplots(num_vars_to_plot, 1, figsize=figsize, sharex=True)
        axes_list: list[MplAxes]
        if num_vars_to_plot == 1:
            axes_list = [axes_obj]
        else:
            axes_list = list(axes_obj)

        # Get interval data and colors
        num_intervals = self.num_intervals
        all_interval_data = self.get_all_interval_data()
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, max(1, num_intervals)))

        # Plot each variable
        for i, name in enumerate(names_to_plot_list):
            ax = axes_list[i]
            var_idx = get_index_func(name)
            if var_idx is None:
                continue

            if num_intervals == 0:
                # Simple case - no intervals
                time_traj, value_traj = self._get_trajectory(name, variable_type)
                if time_traj.size > 0 and value_traj.size > 0:
                    ax.plot(time_traj, value_traj, marker=".", linestyle="-")
            else:
                # Plot by intervals with different colors
                for k, interval_data_opt in enumerate(all_interval_data):
                    if interval_data_opt is None:
                        continue
                    interval_data = interval_data_opt

                    if variable_type == "state":
                        time_segment = interval_data.time_states_segment
                        value_segment = (
                            interval_data.states_segment[var_idx]
                            if var_idx < len(interval_data.states_segment)
                            else None
                        )
                    else:  # control
                        time_segment = interval_data.time_controls_segment
                        value_segment = (
                            interval_data.controls_segment[var_idx]
                            if var_idx < len(interval_data.controls_segment)
                            else None
                        )

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

        # Add legend if we have intervals
        if num_intervals > 0 and self.polynomial_degrees is not None:
            handles = [
                Line2D([0], [0], color=colors[k % len(colors)], lw=2) for k in range(num_intervals)
            ]
            labels = [
                f"Interval {k} (Nk={self.polynomial_degrees[k]})" for k in range(num_intervals)
            ]
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        # Set x-axis label and layout
        axes_list[-1].set_xlabel("Time")
        rect_val: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.85 if num_intervals > 0 else 1.0,
            0.96,
        )
        plt.tight_layout(rect=rect_val)
        plt.show()

    def plot(self, figsize: tuple[float, float] = (10.0, 8.0)) -> None:
        """Plot all states and controls."""
        if not self.success:
            print("Cannot plot: Solution not successful")
            return

        num_states = len(self._state_names)
        num_controls = len(self._control_names)
        num_rows = num_states + num_controls

        if num_rows == 0:
            print("Cannot plot: No states or controls to plot.")
            return

        # Create subplots
        fig, axes_obj = plt.subplots(num_rows, 1, figsize=figsize, sharex=True)
        axes_list: list[MplAxes]
        if num_rows == 1:
            axes_list = [axes_obj]
        else:
            axes_list = list(axes_obj)

        # Get interval data and colors
        num_intervals = self.num_intervals
        all_interval_data = self.get_all_interval_data()
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, max(1, num_intervals)))

        plot_row_idx = 0

        # Plot states
        for state_idx, name in enumerate(self._state_names):
            ax = axes_list[plot_row_idx]
            if num_intervals == 0:
                # Simple case - no intervals
                if self._time_states is not None and self._states is not None:
                    ax.plot(self._time_states, self._states[state_idx], marker=".", linestyle="-")
            else:
                # Plot by intervals with different colors
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

        # Plot controls
        for control_idx, name in enumerate(self._control_names):
            ax = axes_list[plot_row_idx]
            if num_intervals == 0:
                # Simple case - no intervals
                if self._time_controls is not None and self._controls is not None:
                    ax.plot(
                        self._time_controls, self._controls[control_idx], marker=".", linestyle="-"
                    )
            else:
                # Plot by intervals with different colors
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

        # Add legend if we have intervals
        if self.polynomial_degrees is not None and num_intervals > 0:
            handles = [
                Line2D([0], [0], color=colors[k % len(colors)], lw=2) for k in range(num_intervals)
            ]
            labels = [
                f"Interval {k} (Nk={self.polynomial_degrees[k]})" for k in range(num_intervals)
            ]
            fig.legend(handles, labels, loc="upper right", title="Mesh Intervals")

        # Set x-axis label and layout
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
        state_names: _PlotVariableNames = None,
        figsize: tuple[float, float] = (10.0, 8.0),
    ) -> None:
        """Plot specific state trajectories."""
        processed_names = self._resolve_variable_names(state_names, "state")

        if state_names is not None:
            # Filter to only include valid names
            processed_names = [name for name in processed_names if name in self._state_names]
            self._plot_variables("state", processed_names, figsize)
        else:
            self._plot_variables("state", None, figsize)

    def plot_controls(
        self,
        control_names: _PlotVariableNames = None,
        figsize: tuple[float, float] = (10.0, 8.0),
    ) -> None:
        """Plot specific control trajectories."""
        processed_names = self._resolve_variable_names(control_names, "control")

        if control_names is not None:
            # Filter to only include valid names
            processed_names = [name for name in processed_names if name in self._control_names]
            self._plot_variables("control", processed_names, figsize)
        else:
            self._plot_variables("control", None, figsize)

    def get_trajectory(self, identifier: _VariableIdentifier) -> _TrajectoryTuple:
        """
        Get trajectory data for any variable (state or control) with automatic unscaling.

        Args:
            identifier: Variable identifier - string name or symbolic variable

        Returns:
            Tuple of (time_array, value_array) in PHYSICAL UNITS (auto-unscaled)
        """
        if not self.success:
            print("Warning: Solution was not successful, returning empty trajectory")
            empty_arr = np.array([], dtype=np.float64)
            return empty_arr, empty_arr

        # Handle both symbolic variables AND strings with the same logic
        return self._get_unified_trajectory(identifier)

    def _get_unified_trajectory(self, identifier: _VariableIdentifier) -> _TrajectoryTuple:
        """
        Unified trajectory getter that handles both symbolic variables and strings.
        Enhanced for auto-scaling support.
        """
        # Case 1: Symbolic variable - enhanced resolution for auto-scaling
        if self._is_symbolic_variable(identifier):
            sym_var = cast(SymType, identifier)

            # First try regular symbolic maps
            if sym_var in self._sym_state_map:
                return self._get_state_trajectory(self._sym_state_map[sym_var])
            elif sym_var in self._sym_control_map:
                return self._get_control_trajectory(self._sym_control_map[sym_var])

            # Try physical symbol resolution for auto-scaling
            if self._auto_scaling_enabled:
                physical_name = self._resolve_physical_symbolic_variable(sym_var)
                if physical_name is not None:
                    # Check if it's a state or control
                    if physical_name in self._physical_state_names:
                        return self._get_state_trajectory(physical_name)
                    elif physical_name in self._physical_control_names:
                        return self._get_control_trajectory(physical_name)

            # Search in problem if not in maps (fallback)
            var_name = self._resolve_symbolic_variable(sym_var, self._sym_state_map, "_sym_states")
            if var_name is not None:
                return self._get_state_trajectory(var_name)

            var_name = self._resolve_symbolic_variable(
                sym_var, self._sym_control_map, "_sym_controls"
            )
            if var_name is not None:
                return self._get_control_trajectory(var_name)

        # Case 2: String variable - use same search pattern
        elif isinstance(identifier, str):
            var_name = identifier

            # Check states first (handles both direct names and auto-scaling)
            if var_name in self._state_names or (
                self._auto_scaling_enabled
                and var_name in self._physical_to_scaled_map
                and self._physical_to_scaled_map[var_name] in self._state_names
            ):
                return self._get_state_trajectory(var_name)

            # Check controls second (handles both direct names and auto-scaling)
            if var_name in self._control_names or (
                self._auto_scaling_enabled
                and var_name in self._physical_to_scaled_map
                and self._physical_to_scaled_map[var_name] in self._control_names
            ):
                return self._get_control_trajectory(var_name)

        # Not found - return empty arrays
        print(f"Warning: Variable '{identifier}' not found")
        empty_arr = np.array([], dtype=np.float64)
        return empty_arr, empty_arr

    # *** Methods for auto-scaling information access ***
    def get_scaling_info(self) -> dict[str, Any]:
        """
        Get scaling information for analysis and debugging.
        Returns:
            Dictionary containing scaling factors and variable mappings
        """
        return {
            "auto_scaling_enabled": self._auto_scaling_enabled,
            "scaling_factors": self._scaling_factors,
            "physical_to_scaled_map": self._physical_to_scaled_map,
            "scaled_to_physical_map": self._scaled_to_physical_map,
            "physical_state_names": self._physical_state_names,
            "physical_control_names": self._physical_control_names,
        }

    def print_scaling_summary(self) -> None:
        """Print a summary of the scaling information."""
        if not self._auto_scaling_enabled:
            print("Auto-scaling was not used for this solution.")
            return

        print("\n--- Auto-Scaling Summary ---")
        print(f"Auto-scaling enabled: {self._auto_scaling_enabled}")
        print(f"Number of scaled variables: {len(self._scaling_factors)}")

        if self._scaling_factors:
            print("\nScaling factors:")
            for var_name, sf_info in sorted(self._scaling_factors.items()):
                rule = sf_info.get("rule", "Unknown")
                v_factor = sf_info.get("v", 1.0)
                r_factor = sf_info.get("r", 0.0)
                print(
                    f"  {var_name:<12s} | Rule: {rule:<28s} | v: {v_factor:.3e} | r: {r_factor:.3f}"
                )

        print(f"\nPhysical states available: {self._physical_state_names}")
        print(f"Physical controls available: {self._physical_control_names}")
        print("Note: All trajectories returned are automatically unscaled to physical units.")
