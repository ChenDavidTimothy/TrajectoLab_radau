# solution_processor.py
import numpy as np
from typing import List, Optional, Union, Tuple, Any

class IntervalData:
    """Class representing data for a specific mesh interval."""
    def __init__(self, t_start=None, t_end=None, Nk=None,
                 time_states_segment=None, states_segment=None,
                 time_controls_segment=None, controls_segment=None):
        self.t_start = t_start
        self.t_end = t_end
        self.Nk = Nk
        self.time_states_segment = time_states_segment if time_states_segment is not None else np.array([])
        self.states_segment = states_segment if states_segment is not None else []
        self.time_controls_segment = time_controls_segment if time_controls_segment is not None else np.array([])
        self.controls_segment = controls_segment if controls_segment is not None else []

class SolutionProcessor:
    """
    Processes the solution from the optimal control solver
    and provides easy access to common solution components and time-scaled data.
    """
    def __init__(self, solution):
        """
        Initializes the SolutionProcessor with the solution data.

        Args:
            solution: The OptimalControlSolution object returned by the solver.
        """
        # Initialize with default values
        self._solution = solution
        self.initial_time_variable = None
        self.terminal_time_variable = None
        self.adaptive_message = "N/A"
        self.nlp_success = False
        self.objective = None
        self.integrals = None
        self._time_states = None
        self._states = None
        self._time_controls = None
        self._controls = None
        self._Nk_list = None
        self._mesh_nodes_tau_global = None
        self._mesh_nodes_time_domain = None
        
        if solution is None:
            return

        # Helper function to safely extract attributes
        def get_attr(attr_name, default=None):
            return getattr(solution, attr_name, default)
        
        # Extract common solution components
        self.initial_time_variable = get_attr('initial_time_variable')
        self.terminal_time_variable = get_attr('terminal_time_variable')
        self.adaptive_message = get_attr('message', "N/A")
        self.nlp_success = get_attr('success', False)
        self.objective = get_attr('objective')
        self.integrals = get_attr('integrals')

        # Extract trajectory data
        self._time_states = get_attr('time_states')
        self._states = get_attr('states')
        self._time_controls = get_attr('time_controls')
        self._controls = get_attr('controls')

        # Extract mesh information
        self._Nk_list = get_attr('num_collocation_nodes_per_interval')
        mesh_nodes = get_attr('global_normalized_mesh_nodes')
        self._mesh_nodes_tau_global = np.asarray(mesh_nodes) if mesh_nodes is not None else None
        
        # Calculate mesh nodes in time domain
        self._mesh_nodes_time_domain = self._calculate_mesh_nodes_time_domain()

    def _calculate_mesh_nodes_time_domain(self) -> Optional[np.ndarray]:
        """
        Calculates mesh node locations in the actual time domain [initial_time_variable, terminal_time_variable].
        """
        if self.initial_time_variable is None or self.terminal_time_variable is None or self._mesh_nodes_tau_global is None:
            return None

        alpha = (self.terminal_time_variable - self.initial_time_variable) / 2.0
        alpha_0 = (self.terminal_time_variable + self.initial_time_variable) / 2.0
        return alpha * self._mesh_nodes_tau_global + alpha_0

    @property
    def time_states(self) -> Optional[np.ndarray]:
        """Time vector for state trajectories (dense output)."""
        return self._time_states

    @property
    def states(self) -> Optional[List[np.ndarray]]:
        """List of state trajectories (each state is a NumPy array, dense output)."""
        return self._states

    @property
    def time_controls(self) -> Optional[np.ndarray]:
        """Time vector for control trajectories (dense output)."""
        return self._time_controls

    @property
    def controls(self) -> Optional[List[np.ndarray]]:
        """List of control trajectories (each control is a NumPy array, dense output)."""
        return self._controls

    @property
    def num_states(self) -> int:
        """Number of states."""
        return len(self._states) if self._states is not None else 0

    @property
    def num_controls(self) -> int:
        """Number of controls."""
        return len(self._controls) if self._controls is not None else 0

    @property
    def num_collocation_nodes_per_interval(self) -> Optional[List[int]]:
        """List of polynomial degrees (number of collocation points) per interval."""
        return self._Nk_list

    @property
    def global_normalized_mesh_nodes(self) -> Optional[np.ndarray]:
        """Mesh nodes in the normalized global tau domain [-1, 1]."""
        return self._mesh_nodes_tau_global

    @property
    def mesh_nodes_time_domain(self) -> Optional[np.ndarray]:
        """Mesh nodes in the actual time domain [initial_time_variable, terminal_time_variable]."""
        return self._mesh_nodes_time_domain

    @property
    def num_intervals(self) -> int:
        """Number of mesh intervals."""
        if self._Nk_list is not None:
            return len(self._Nk_list)
        elif self._mesh_nodes_tau_global is not None and len(self._mesh_nodes_tau_global) > 1:
            return len(self._mesh_nodes_tau_global) - 1
        return 0

    def get_state_trajectory(self, state_index: int) -> Optional[np.ndarray]:
        """
        Retrieves a specific state trajectory.

        Args:
            state_index: The 0-based index of the state.

        Returns:
            A NumPy array for the requested state trajectory, or None if unavailable.
        """
        if self._states and 0 <= state_index < self.num_states:
            return self._states[state_index]
        return None

    def get_control_trajectory(self, control_index: int) -> Optional[np.ndarray]:
        """
        Retrieves a specific control trajectory.

        Args:
            control_index: The 0-based index of the control.

        Returns:
            A NumPy array for the requested control trajectory, or None if unavailable.
        """
        if self._controls and 0 <= control_index < self.num_controls:
            return self._controls[control_index]
        return None

    def _extract_segment_data(self, time_array, data_arrays, interval_t_start, interval_t_end, epsilon=1e-9):
        """
        Helper function to extract time and data segments for a specific interval.
        """
        if time_array is None or not data_arrays or len(time_array) == 0:
            return np.array([]), [np.array([]) for _ in range(len(data_arrays) if data_arrays else 1)]
            
        sort_indices = np.argsort(time_array)
        sorted_time = time_array[sort_indices]
        
        idx_in_interval = np.where(
            (sorted_time >= interval_t_start - epsilon) &
            (sorted_time <= interval_t_end + epsilon)
        )[0]
        
        if len(idx_in_interval) == 0:
            return np.array([]), [np.array([]) for _ in range(len(data_arrays))]
            
        time_segment = sorted_time[idx_in_interval]
        data_segments = []
        
        for data_array in data_arrays:
            if len(data_array) == len(time_array):
                sorted_data = data_array[sort_indices]
                data_segments.append(sorted_data[idx_in_interval])
            else:
                data_segments.append(np.array([]))
                
        return time_segment, data_segments

    def get_data_for_interval(self, interval_index: int) -> Optional[IntervalData]:
        """
        Extracts state, control, and time data corresponding to a specific mesh interval.
        This is useful for interval-wise plotting or analysis.

        Args:
            interval_index: The 0-based index of the mesh interval.

        Returns:
            An IntervalData object containing data for the specified interval,
            or None if data is insufficient or index is out of bounds.
        """
        if not (self.num_intervals > 0 and 0 <= interval_index < self.num_intervals):
            return None
        if self._mesh_nodes_time_domain is None or self._Nk_list is None:
            return None

        interval_t_start = self._mesh_nodes_time_domain[interval_index]
        interval_t_end = self._mesh_nodes_time_domain[interval_index + 1]
        nk_interval = self._Nk_list[interval_index] if interval_index < len(self._Nk_list) else -1

        # Extract state data
        time_states_segment, states_segment = self._extract_segment_data(
            self._time_states, 
            self._states if self._states else [], 
            interval_t_start, 
            interval_t_end
        )
        
        # Extract control data
        time_controls_segment, controls_segment = self._extract_segment_data(
            self._time_controls, 
            self._controls if self._controls else [], 
            interval_t_start, 
            interval_t_end
        )

        return IntervalData(
            t_start=interval_t_start,
            t_end=interval_t_end,
            Nk=nk_interval,
            time_states_segment=time_states_segment,
            states_segment=states_segment,
            time_controls_segment=time_controls_segment,
            controls_segment=controls_segment
        )

    def get_all_interval_data(self) -> List[Optional[IntervalData]]:
        """
        Retrieves data for all mesh intervals.

        Returns:
            A list of IntervalData objects, where each object contains data for an interval
            as returned by get_data_for_interval().
        """
        if self.num_intervals == 0:
            return []
        return [self.get_data_for_interval(i) for i in range(self.num_intervals)]

    def summary(self) -> str:
        """
        Provides a string summary of the solution, similar to the logs in the example scripts.
        """
        if self._solution is None:
            return "--- Solution Data Not Available ---"

        lines = []
        if "Adaptive mesh converged" in self.adaptive_message:
             lines.append("--- Adaptive Refinement Succeeded (Converged to Tolerance) ---")
        elif "Reached max iterations" in self.adaptive_message and self.nlp_success:
             lines.append("--- Adaptive Refinement Completed (Max Iterations Reached, Last NLP Succeeded) ---")
        elif self.nlp_success:
             lines.append("--- Adaptive Refinement Completed (Last NLP Succeeded, Check Message for Details) ---")
        else:
             lines.append("--- Adaptive Refinement Result (Process or Last NLP Failed) ---")

        lines.append(f"  Overall Adaptive Process Message: {self.adaptive_message}")
        lines.append(f"  Last NLP Solve Success: {self.nlp_success}")

        if self.initial_time_variable is not None: lines.append(f"  Final initial_time_variable: {self.initial_time_variable:.4f}")
        if self.terminal_time_variable is not None: lines.append(f"  Final terminal_time_variable: {self.terminal_time_variable:.4f}")
        if self.objective is not None: lines.append(f"  Final Objective: {self.objective:.4f}")

        integrals_val = self.integrals
        if integrals_val is not None:
            if isinstance(integrals_val, (np.ndarray, list)) and np.array(integrals_val).size > 0:
                lines.append(f"  Final Integrals: {np.array2string(np.array(integrals_val), precision=4)}")
            elif isinstance(integrals_val, (float, int)):
                lines.append(f"  Final Integrals: {integrals_val:.4f}")
            else:
                lines.append(f"  Final Integrals: {integrals_val}")

        if self._Nk_list is not None: lines.append(f"  Final num_collocation_nodes_per_interval: {self._Nk_list}")

        if self._mesh_nodes_tau_global is not None:
            lines.append(f"  Final global_normalized_mesh_nodes: {np.array2string(self._mesh_nodes_tau_global, precision=4)}")

        if self._mesh_nodes_time_domain is not None:
            lines.append(f"  Mesh nodes in actual time domain: {np.array2string(self._mesh_nodes_time_domain, precision=4)}")
        else:
            lines.append(f"  Mesh nodes in actual time domain: Not available")

        return "\n".join(lines)

    def __repr__(self) -> str:
        if self._solution is None:
            return "<SolutionProcessor (No Data)>"

        status = "Successful NLP" if self.nlp_success else "Failed NLP"
        obj_str = f"{self.objective:.4f}" if self.objective is not None else "N/A"
        tf_str = f"{self.terminal_time_variable:.2f}" if self.terminal_time_variable is not None else "N/A"
        t0_str = f"{self.initial_time_variable:.2f}" if self.initial_time_variable is not None else "N/A"

        return (f"<SolutionProcessor initial_time_variable={t0_str}, terminal_time_variable={tf_str}, "
                f"Objective={obj_str}, Status='{status}', "
                f"Intervals={self.num_intervals}>")