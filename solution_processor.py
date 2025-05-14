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
        if solution is None:
            # Create empty processor with default values
            self._solution = None
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
            return

        self._solution = solution

        # Pre-assign frequently used top-level items
        self.initial_time_variable = solution.initial_time_variable
        self.terminal_time_variable = solution.terminal_time_variable
        self.adaptive_message = solution.message if hasattr(solution, 'message') else "N/A"
        # 'success' in the solution typically refers to the NLP success of the *last* iteration.
        # The overall adaptive process success is better judged by the message.
        self.nlp_success = solution.success if hasattr(solution, 'success') else False
        self.objective = solution.objective if hasattr(solution, 'objective') else None
        self.integrals = solution.integrals if hasattr(solution, 'integrals') else None

        self._time_states = solution.time_states if hasattr(solution, 'time_states') else None
        self._states = solution.states if hasattr(solution, 'states') else None
        self._time_controls = solution.time_controls if hasattr(solution, 'time_controls') else None
        self._controls = solution.controls if hasattr(solution, 'controls') else None

        self._Nk_list = solution.num_collocation_nodes_per_interval if hasattr(solution, 'num_collocation_nodes_per_interval') else None
        _mesh_nodes_tau_global = solution.global_normalized_mesh_nodes if hasattr(solution, 'global_normalized_mesh_nodes') else None
        self._mesh_nodes_tau_global = np.asarray(_mesh_nodes_tau_global) if _mesh_nodes_tau_global is not None else None

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
        print(f"Warning: State trajectory for index {state_index} not available.")
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
        print(f"Warning: Control trajectory for index {control_index} not available.")
        return None

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
            # print(f"Warning: Interval index {interval_index} is out of bounds (0-{self.num_intervals-1}).")
            return None
        if self._mesh_nodes_time_domain is None or self._Nk_list is None:
            # print("Warning: Mesh information incomplete for interval data extraction.")
            return None


        interval_t_start = self._mesh_nodes_time_domain[interval_index]
        interval_t_end = self._mesh_nodes_time_domain[interval_index + 1]
        # Handle case where num_collocation_nodes_per_interval might be shorter than number of intervals inferred from mesh_nodes
        nk_interval = self._Nk_list[interval_index] if interval_index < len(self._Nk_list) else -1 # -1 indicates Nk unknown


        epsilon = 1e-9 # For robust boundary checks with floating point times

        states_segment_list = []
        time_states_segment_array = np.array([])
        if self._time_states is not None and self._states is not None and len(self._time_states) > 0 :
            sort_indices_states = np.argsort(self._time_states)
            sorted_time_states = self._time_states[sort_indices_states]

            idx_states_in_interval = np.where(
                (sorted_time_states >= interval_t_start - epsilon) &
                (sorted_time_states <= interval_t_end + epsilon)
            )[0]

            if len(idx_states_in_interval) > 0:
                time_states_segment_array = sorted_time_states[idx_states_in_interval]
                for state_traj_idx in range(self.num_states):
                    state_traj = self._states[state_traj_idx]
                    if len(state_traj) == len(self._time_states):
                        sorted_state_traj = state_traj[sort_indices_states]
                        states_segment_list.append(sorted_state_traj[idx_states_in_interval])
                    else:
                        # print(f"Warning: State {state_traj_idx} length mismatch for interval {interval_index}.")
                        states_segment_list.append(np.array([])) # Fallback for inconsistent length
            else: # No time points in interval
                 for _ in range(self.num_states): states_segment_list.append(np.array([]))
        else: # No state data available
            for _ in range(self.num_states if self.num_states > 0 else 1): states_segment_list.append(np.array([]))


        controls_segment_list = []
        time_controls_segment_array = np.array([])
        if self._time_controls is not None and self._controls is not None and len(self._time_controls) > 0:
            sort_indices_controls = np.argsort(self._time_controls)
            sorted_time_controls = self._time_controls[sort_indices_controls]

            idx_controls_in_interval = np.where(
                (sorted_time_controls >= interval_t_start - epsilon) &
                (sorted_time_controls <= interval_t_end + epsilon)
            )[0]

            if len(idx_controls_in_interval) > 0:
                time_controls_segment_array = sorted_time_controls[idx_controls_in_interval]
                for control_traj_idx in range(self.num_controls):
                    control_traj = self._controls[control_traj_idx]
                    if len(control_traj) == len(self._time_controls):
                        sorted_control_traj = control_traj[sort_indices_controls]
                        controls_segment_list.append(sorted_control_traj[idx_controls_in_interval])
                    else:
                        # print(f"Warning: Control {control_traj_idx} length mismatch for interval {interval_index}.")
                        controls_segment_list.append(np.array([]))
            else: # No time points in interval
                for _ in range(self.num_controls): controls_segment_list.append(np.array([]))
        else: # No control data available
            for _ in range(self.num_controls if self.num_controls > 0 else 1): controls_segment_list.append(np.array([]))


        return IntervalData(
            t_start=interval_t_start,
            t_end=interval_t_end,
            Nk=nk_interval,
            time_states_segment=time_states_segment_array,
            states_segment=states_segment_list,
            time_controls_segment=time_controls_segment_array,
            controls_segment=controls_segment_list
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
        if self._solution is None: # Handle case where SolutionProcessor was initialized with empty dict
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