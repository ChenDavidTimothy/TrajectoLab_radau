import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .tl_types import FloatArray, OptimalControlSolution, PhaseID, ProblemProtocol


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Solution:
    """Clean, bundled interface for multiphase optimal control problem solutions.

    The Solution class provides a comprehensive interface for accessing optimization
    results, including trajectories, solver diagnostics, mesh information, and
    adaptive refinement data.

    Examples:
        Basic solution usage with two-phase Schwartz problem:

        >>> problem = mtor.Problem("Two-Phase Schwartz Problem")
        >>> # ... define two-phase problem with x0, x1 states and u control ...
        >>> solution = mtor.solve_adaptive(problem)
        >>> if solution.status["success"]:
        ...     print(f"Objective: {solution.status['objective']:.6f}")
        ...     solution.plot()

        Accessing trajectory data from multiphase problem:

        >>> # Get mission-wide trajectories (auto-concatenates phases)
        >>> x0_complete = solution["x0"]        # Both phases combined
        >>> x1_complete = solution["x1"]        # Both phases combined
        >>> u_complete = solution["u"]          # Both phases combined
        >>>
        >>> # Get phase-specific data
        >>> x0_phase1 = solution[(1, "x0")]     # Phase 1 only
        >>> x1_phase2 = solution[(2, "x1")]     # Phase 2 only
        >>>
        >>> # Get time arrays
        >>> time_states = solution["time_states"]
        >>> time_controls = solution["time_controls"]

        Working with multiphase results:

        >>> # Examine all phases in Schwartz problem
        >>> for phase_id, phase_data in solution.phases.items():
        ...     duration = phase_data["times"]["duration"]
        ...     states = phase_data["variables"]["state_names"]  # ["x0", "x1"]
        ...     controls = phase_data["variables"]["control_names"]  # ["u"]
        ...     print(f"Phase {phase_id}: {duration:.3f}s, states: {states}")

        Adaptive algorithm analysis:

        >>> if solution.adaptive:
        ...     print(f"Converged: {solution.adaptive['converged']}")
        ...     print(f"Iterations: {solution.adaptive['iterations']}")
        ...     for phase_id, errors in solution.adaptive['final_errors'].items():
        ...         print(f"Phase {phase_id} max error: {max(errors):.2e}")
    """

    def __init__(
        self,
        raw_solution: OptimalControlSolution | None,
        problem: ProblemProtocol | None,
        auto_summary: bool = True,
    ) -> None:
        """
        Initialize solution wrapper from raw multiphase optimization results.

        Args:
            raw_solution: Raw optimization results from solver
            problem: Problem protocol instance
            auto_summary: Whether to automatically display comprehensive summary (default: True)
        """
        # Store raw data for internal use and direct CasADi access
        self._raw_solution = raw_solution
        self._problem = problem

        # Store raw CasADi objects for advanced users
        self.raw_solution = raw_solution.raw_solution if raw_solution else None
        self.opti = raw_solution.opti_object if raw_solution else None

        # Build variable name mappings for dictionary access
        if problem is not None:
            self._phase_state_names = {}
            self._phase_control_names = {}
            for phase_id in problem._get_phase_ids():
                self._phase_state_names[phase_id] = problem._get_phase_ordered_state_names(phase_id)
                self._phase_control_names[phase_id] = problem._get_phase_ordered_control_names(
                    phase_id
                )
        else:
            self._phase_state_names = {}
            self._phase_control_names = {}

        if auto_summary:
            self._show_comprehensive_summary()

    def _show_comprehensive_summary(self) -> None:
        try:
            from .summary import print_comprehensive_solution_summary

            print_comprehensive_solution_summary(self)
        except ImportError as e:
            logger.warning(f"Could not import comprehensive summary: {e}")
        except Exception as e:
            logger.warning(f"Error in comprehensive summary: {e}")

    @property
    def status(self) -> dict[str, Any]:
        """
        Complete solution status and optimization results.

        Provides comprehensive information about the optimization outcome,
        including success status, objective value, and mission duration.
        This is typically the first property to check after solving.

        Returns:
            Dictionary containing:

            - **success** (bool): Whether optimization succeeded
            - **message** (str): Solver status message with details
            - **objective** (float): Final objective function value
            - **total_mission_time** (float): Total time across all phases

        Examples:
            Check if two-phase Schwartz optimization succeeded:

            >>> solution = mtor.solve_adaptive(schwartz_problem)
            >>> if solution.status["success"]:
            ...     print("Optimization successful!")
            ...     print(f"Optimal objective: {solution.status['objective']:.6f}")
            ... else:
            ...     print(f"Optimization failed: {solution.status['message']}")

            Get mission duration for multiphase problem:

            >>> if solution.status["success"]:
            ...     total_time = solution.status["total_mission_time"]  # Phase 1 + Phase 2
            ...     objective = solution.status["objective"]           # 5*(x0_final² + x1_final²)
            ...     print(f"Total mission time: {total_time:.3f} seconds")
            ...     print(f"Final cost: {objective:.6e}")

            Handle optimization failures:

            >>> status = solution.status
            >>> if not status["success"]:
            ...     print("Two-phase Schwartz optimization failed!")
            ...     print(f"Reason: {status['message']}")
            ...     print("Try different initial guess or tighter tolerances")
        """
        if self._raw_solution is None:
            return {
                "success": False,
                "message": "No solution available",
                "objective": float("nan"),
                "total_mission_time": float("nan"),
            }

        # Calculate total mission time
        if self._raw_solution.phase_initial_times and self._raw_solution.phase_terminal_times:
            earliest_start = min(self._raw_solution.phase_initial_times.values())
            latest_end = max(self._raw_solution.phase_terminal_times.values())
            total_time = latest_end - earliest_start
        else:
            total_time = float("nan")

        return {
            "success": self._raw_solution.success,
            "message": self._raw_solution.message,
            "objective": self._raw_solution.objective
            if self._raw_solution.objective is not None
            else float("nan"),
            "total_mission_time": total_time,
        }

    @property
    def phases(self) -> dict[PhaseID, dict[str, Any]]:
        """
        Comprehensive information for all phases in the optimal control problem.

        Provides detailed data about each phase including timing, variables,
        mesh configuration, and time arrays. Essential for understanding
        multiphase mission structure and accessing phase-specific data.

        Returns:
            Dictionary mapping phase IDs to phase data containing:

            - **times** (dict): Phase timing information

              - initial (float): Phase start time
              - final (float): Phase end time
              - duration (float): Phase duration

            - **variables** (dict): Variable information

              - state_names (list): Ordered state variable names
              - control_names (list): Ordered control variable names
              - num_states (int): Number of state variables
              - num_controls (int): Number of control variables

            - **mesh** (dict): Mesh configuration details

              - polynomial_degrees (list): Polynomial degree per interval
              - mesh_nodes (FloatArray): Normalized mesh node locations
              - num_intervals (int): Total number of mesh intervals

            - **time_arrays** (dict): Time coordinate arrays

              - states (FloatArray): Time points for state trajectories
              - controls (FloatArray): Time points for control trajectories

            - **integrals** (float | FloatArray | None): Integral values for this phase

        Examples:
            Examine all phases in two-phase Schwartz problem:

            >>> for phase_id, phase_data in solution.phases.items():
            ...     times = phase_data["times"]
            ...     variables = phase_data["variables"]
            ...     print(f"Phase {phase_id}:")
            ...     print(f"  Duration: {times['duration']:.3f} seconds")
            ...     print(f"  States: {variables['state_names']}")      # ["x0", "x1"]
            ...     print(f"  Controls: {variables['control_names']}")  # ["u"]

            Get specific phase information from Schwartz problem:

            >>> phase_1 = solution.phases[1]
            >>> first_phase_duration = phase_1["times"]["duration"]      # Phase 1: 0.0 to 1.0
            >>> num_states = phase_1["variables"]["num_states"]          # 2 states (x0, x1)
            >>> mesh_intervals = phase_1["mesh"]["num_intervals"]        # Number of intervals
            >>> print(f"Phase 1: {first_phase_duration:.1f}s, {num_states} states, {mesh_intervals} intervals")

            Check mesh refinement results for each phase:

            >>> for phase_id, phase_data in solution.phases.items():
            ...     mesh = phase_data["mesh"]
            ...     degrees = mesh["polynomial_degrees"]    # e.g., [6, 6] for two intervals
            ...     intervals = mesh["num_intervals"]       # e.g., 2
            ...     print(f"Phase {phase_id}: {intervals} intervals, degrees {degrees}")

            Access time arrays for Schwartz problem analysis:

            >>> phase_2 = solution.phases[2]
            >>> state_times = phase_2["time_arrays"]["states"]     # Time points for x0, x1
            >>> control_times = phase_2["time_arrays"]["controls"] # Time points for u
            >>> print(f"Phase 2: {len(state_times)} state points, {len(control_times)} control points")
        """
        if self._raw_solution is None:
            return {}

        phases_data = {}

        for phase_id in self._get_phase_ids():
            # Time information
            initial_time = self._raw_solution.phase_initial_times.get(phase_id, float("nan"))
            final_time = self._raw_solution.phase_terminal_times.get(phase_id, float("nan"))
            duration = (
                final_time - initial_time
                if not (np.isnan(initial_time) or np.isnan(final_time))
                else float("nan")
            )

            # Variable information
            state_names = self._phase_state_names.get(phase_id, [])
            control_names = self._phase_control_names.get(phase_id, [])

            # Mesh information
            polynomial_degrees = self._raw_solution.phase_mesh_intervals.get(phase_id, [])
            mesh_nodes = self._raw_solution.phase_mesh_nodes.get(
                phase_id, np.array([], dtype=np.float64)
            )

            # Time arrays
            time_states = self._raw_solution.phase_time_states.get(
                phase_id, np.array([], dtype=np.float64)
            )
            time_controls = self._raw_solution.phase_time_controls.get(
                phase_id, np.array([], dtype=np.float64)
            )

            # Integrals
            integrals = self._raw_solution.phase_integrals.get(phase_id, None)

            phases_data[phase_id] = {
                "times": {"initial": initial_time, "final": final_time, "duration": duration},
                "variables": {
                    "state_names": state_names.copy(),
                    "control_names": control_names.copy(),
                    "num_states": len(state_names),
                    "num_controls": len(control_names),
                },
                "mesh": {
                    "polynomial_degrees": polynomial_degrees.copy() if polynomial_degrees else [],
                    "mesh_nodes": mesh_nodes.copy()
                    if mesh_nodes.size > 0
                    else np.array([], dtype=np.float64),
                    "num_intervals": len(polynomial_degrees) if polynomial_degrees else 0,
                },
                "time_arrays": {"states": time_states.copy(), "controls": time_controls.copy()},
                "integrals": integrals,
            }

        return phases_data

    @property
    def parameters(self) -> dict[str, Any] | None:
        """
        Static parameter optimization results and information.

        Provides access to optimized static parameters that remain constant
        throughout the mission but are optimized by the solver. Returns None
        if no static parameters were defined in the problem.

        Returns:
            Dictionary containing parameter information, or None if no parameters:

            - **values** (FloatArray): Optimized parameter values
            - **names** (list[str] | None): Parameter names if available
            - **count** (int): Number of static parameters

        Examples:
            Check if Schwartz problem has static parameters:

            >>> if solution.parameters is not None:
            ...     param_info = solution.parameters
            ...     print(f"Found {param_info['count']} static parameters")
            ...     print(f"Values: {param_info['values']}")
            ... else:
            ...     print("No static parameters in Schwartz problem")

            Access specific parameter values (if Schwartz problem had parameters):

            >>> params = solution.parameters
            >>> if params and params["names"]:
            ...     for name, value in zip(params["names"], params["values"]):
            ...         print(f"{name}: {value:.6f}")
            ... else:
            ...     # Parameters exist but no names available
            ...     for i, value in enumerate(params["values"]):
            ...         print(f"Parameter {i}: {value:.6f}")

            Use parameters in Schwartz problem analysis:

            >>> if solution.parameters:
            ...     # Example: if Schwartz had optimal final time as parameter
            ...     optimal_param = solution.parameters["values"][0]
            ...     print(f"Optimal parameter: {optimal_param:.6f}")
        """
        if self._raw_solution is None or self._raw_solution.static_parameters is None:
            return None

        # Try to get parameter names if available
        param_names = None
        if self._problem is not None and hasattr(self._problem, "_static_parameters"):
            try:
                static_params = self._problem._static_parameters
                if hasattr(static_params, "parameter_names"):
                    param_names = static_params.parameter_names.copy()
            except (AttributeError, IndexError):
                pass

        return {
            "values": self._raw_solution.static_parameters.copy(),
            "names": param_names,
            "count": len(self._raw_solution.static_parameters),
        }

    @property
    def adaptive(self) -> dict[str, Any] | None:
        """
        Adaptive mesh refinement algorithm results and convergence information.

        Provides detailed information about the adaptive algorithm's performance,
        convergence status, and final error estimates. Only available when using
        the adaptive solver (solve_adaptive). Returns None for fixed mesh solutions.

        Returns:
            Dictionary containing adaptive algorithm data, or None if fixed mesh was used:

            - **converged** (bool): Whether algorithm achieved target tolerance
            - **iterations** (int): Number of refinement iterations performed
            - **target_tolerance** (float): Target error tolerance specified
            - **phase_converged** (dict): Per-phase convergence status mapping
            - **final_errors** (dict): Final error estimates per phase (list of floats)
            - **gamma_factors** (dict): Normalization factors used per phase

        Examples:
            Check adaptive convergence for Schwartz problem:

            >>> if solution.adaptive:
            ...     adaptive_info = solution.adaptive
            ...     if adaptive_info["converged"]:
            ...         print(f"Schwartz problem converged in {adaptive_info['iterations']} iterations")
            ...         print(f"Target tolerance: {adaptive_info['target_tolerance']:.1e}")
            ...     else:
            ...         print("Schwartz problem did not converge within iteration limit")
            ... else:
            ...     print("Fixed mesh solution - no adaptive data available")

            Analyze per-phase convergence for two-phase Schwartz:

            >>> if solution.adaptive:
            ...     for phase_id, converged in solution.adaptive["phase_converged"].items():
            ...         status = "✓" if converged else "✗"
            ...         phase_name = "elliptical constraint" if phase_id == 1 else "final approach"
            ...         print(f"Phase {phase_id} ({phase_name}): {status}")

            Examine final error estimates for Schwartz phases:

            >>> if solution.adaptive:
            ...     for phase_id, errors in solution.adaptive["final_errors"].items():
            ...         max_error = max(errors) if errors else 0.0
            ...         avg_error = np.mean(errors) if errors else 0.0
            ...         print(f"Phase {phase_id}: max_error={max_error:.2e}, avg_error={avg_error:.2e}")

            Get refinement statistics for complex Schwartz dynamics:

            >>> if solution.adaptive:
            ...     total_iterations = solution.adaptive["iterations"]
            ...     target_tol = solution.adaptive["target_tolerance"]
            ...     converged = solution.adaptive["converged"]
            ...     print(f"Schwartz adaptive refinement: {total_iterations} iterations")
            ...     print(f"Target: {target_tol:.1e}, Converged: {converged}")
        """
        if self._raw_solution is None or self._raw_solution.adaptive_data is None:
            return None

        adaptive_data = self._raw_solution.adaptive_data

        return {
            "converged": adaptive_data.converged,
            "iterations": adaptive_data.total_iterations,
            "target_tolerance": adaptive_data.target_tolerance,
            "phase_converged": adaptive_data.phase_converged.copy(),
            "final_errors": {
                phase_id: errors.copy()
                for phase_id, errors in adaptive_data.final_phase_error_estimates.items()
            },
            "gamma_factors": {
                phase_id: factors.copy() if factors is not None else None
                for phase_id, factors in adaptive_data.phase_gamma_factors.items()
            },
        }

    def _get_phase_ids(self) -> list[PhaseID]:
        if self._raw_solution is None:
            return []
        return sorted(self._raw_solution.phase_initial_times.keys())

    def __getitem__(self, key: str | tuple[PhaseID, str]) -> FloatArray:
        if not self.status["success"]:
            logger.warning("Cannot access variable '%s': Solution not successful", key)
            return np.array([], dtype=np.float64)

        if isinstance(key, tuple):
            return self._get_by_tuple_key(key)
        elif isinstance(key, str):
            return self._get_by_string_key(key)
        else:
            raise KeyError(
                f"Invalid key type: {type(key)}. Use string or (phase_id, variable_name) tuple"
            )

    def _get_by_tuple_key(self, key: tuple[PhaseID, str]) -> FloatArray:
        if len(key) != 2:
            raise KeyError("Tuple key must have exactly 2 elements: (phase_id, variable_name)")

        # Explicit None check for mypy type safety
        if self._raw_solution is None:
            return np.array([], dtype=np.float64)

        phase_id, var_name = key

        if phase_id not in self._get_phase_ids():
            raise KeyError(f"Phase {phase_id} not found in solution")

        if var_name == "time_states":
            return self._raw_solution.phase_time_states.get(
                phase_id, np.array([], dtype=np.float64)
            )
        elif var_name == "time_controls":
            return self._raw_solution.phase_time_controls.get(
                phase_id, np.array([], dtype=np.float64)
            )

        if phase_id in self._phase_state_names and var_name in self._phase_state_names[phase_id]:
            var_index = self._phase_state_names[phase_id].index(var_name)
            if phase_id in self._raw_solution.phase_states and var_index < len(
                self._raw_solution.phase_states[phase_id]
            ):
                return self._raw_solution.phase_states[phase_id][var_index]

        if (
            phase_id in self._phase_control_names
            and var_name in self._phase_control_names[phase_id]
        ):
            var_index = self._phase_control_names[phase_id].index(var_name)
            if phase_id in self._raw_solution.phase_controls and var_index < len(
                self._raw_solution.phase_controls[phase_id]
            ):
                return self._raw_solution.phase_controls[phase_id][var_index]

        raise KeyError(f"Variable '{var_name}' not found in phase {phase_id}")

    def _get_by_string_key(self, key: str) -> FloatArray:
        matching_arrays = []

        for phase_id in self._get_phase_ids():
            try:
                phase_data = self[(phase_id, key)]
                matching_arrays.append(phase_data)
            except KeyError:
                continue

        if not matching_arrays:
            all_vars = []
            for phase_id in self._get_phase_ids():
                phase_vars = (
                    self._phase_state_names.get(phase_id, [])
                    + self._phase_control_names.get(phase_id, [])
                    + ["time_states", "time_controls"]
                )
                all_vars.extend([f"({phase_id}, '{var}')" for var in phase_vars])

            raise KeyError(f"Variable '{key}' not found in any phase. Available: {all_vars}")

        if len(matching_arrays) == 1:
            return matching_arrays[0]

        return np.concatenate(matching_arrays, dtype=np.float64)

    def __contains__(self, key: str | tuple[PhaseID, str]) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def plot(
        self,
        phase_id: PhaseID | None = None,
        *variable_names: str,
        figsize: tuple[float, float] = (12.0, 8.0),
        show_phase_boundaries: bool = True,
    ) -> None:
        """
        Plot multiphase trajectories with professional formatting and phase boundaries.

        Creates comprehensive trajectory plots with automatic layout, interval coloring,
        and phase boundary indicators. Supports both single-phase and multiphase
        visualization with customizable variable selection.

        Args:
            phase_id: Specific phase to plot (None plots all phases)
            *variable_names: Optional specific variable names to plot
            figsize: Figure size for each plotting window
            show_phase_boundaries: Whether to show vertical lines at phase boundaries

        Examples:
            Plot all Schwartz variables for both phases:

            >>> solution.plot()  # Complete Schwartz solution visualization

            Plot specific Schwartz phase only:

            >>> solution.plot(phase_id=1)  # Only phase 1 (elliptical constraint region)

            Plot specific Schwartz variables across both phases:

            >>> solution.plot(phase_id=None, "x0", "x1", "u")

            Plot specific variables for specific Schwartz phase:

            >>> solution.plot(1, "x0", "x1")  # Phase 1 states only

            Customize Schwartz plot appearance:

            >>> solution.plot(
            ...     figsize=(16, 10),           # Larger figure for detailed view
            ...     show_phase_boundaries=True  # Show transition at t=1.0
            ... )

            Plot with custom Schwartz analysis:

            >>> # Plot solution then add elliptical constraint visualization
            >>> solution.plot(1, "x0", "x1")
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Add elliptical constraint boundary to phase portrait
            >>> plt.figure(1)  # Access the x0-x1 plot
            >>> theta = np.linspace(0, 2*np.pi, 100)
            >>> ellipse_x = 1 + 0.333 * np.cos(theta)  # Ellipse center and size
            >>> ellipse_y = 0.4 + 0.3 * np.sin(theta)
            >>> plt.plot(ellipse_x, ellipse_y, 'r--', label='Constraint boundary')
            >>> plt.legend()
            >>> plt.show()

        Note:
            - Automatically handles both state and control variables
            - Uses distinct colors for different mesh intervals in adaptive solutions
            - Shows phase boundaries as vertical lines in multiphase problems
            - Creates separate subplots for each variable with proper scaling
            - Includes time units and variable labels automatically
        """
        from .plot import plot_multiphase_solution

        plot_multiphase_solution(self, phase_id, variable_names, figsize, show_phase_boundaries)

    def summary(self, comprehensive: bool = True) -> None:
        """
        Display detailed solution summary with optimization results and diagnostics.

        Prints a comprehensive overview of the solution including solver status,
        objective value, phase information, mesh details, and adaptive algorithm
        results if applicable. Essential for solution validation and analysis.

        Args:
            comprehensive: If True (default), show exhaustive summary with all details.
                          If False, show concise summary with key information only.

        Examples:
            Display full Schwartz solution summary (default):

            >>> solution.summary()
            # Prints comprehensive summary including:
            # - Solver status and objective value (5*(x0_final² + x1_final²))
            # - Phase-by-phase breakdown (Phase 1: 0.0-1.0s, Phase 2: 1.0-2.9s)
            # - Mesh configuration and refinement details
            # - Adaptive algorithm convergence (if solve_adaptive used)
            # - Variable counts (2 states, 1 control per phase)

            Display concise Schwartz summary:

            >>> solution.summary(comprehensive=False)
            # Prints brief summary with:
            # - Success status and objective
            # - Total mission time (2.9 seconds)
            # - Number of phases (2)
            # - Adaptive convergence status

            Use in Schwartz validation workflow:

            >>> solution = mtor.solve_adaptive(schwartz_problem)
            >>>
            >>> # Always check summary first
            >>> solution.summary()
            >>>
            >>> # Then proceed with Schwartz-specific analysis if successful
            >>> if solution.status["success"]:
            ...     # Check final state against elliptical constraint
            ...     final_x0 = solution["x0"][-1]
            ...     final_x1 = solution["x1"][-1]
            ...     solution.plot()
            ... else:
            ...     print("Schwartz solution failed - check summary for details")

            Suppress automatic summary:

            >>> # Solve without automatic summary display
            >>> solution = mtor.solve_fixed_mesh(schwartz_problem, show_summary=False)
            >>>
            >>> # Display summary manually when needed
            >>> solution.summary(comprehensive=True)

        Note:
            - Comprehensive summary includes mesh refinement details and error estimates
            - Concise summary focuses on key results for quick validation
            - Automatically adapts display based on single-phase vs multiphase problems
            - Shows adaptive algorithm performance when applicable
        """
        if comprehensive:
            try:
                from .summary import print_comprehensive_solution_summary

                print_comprehensive_solution_summary(self)
            except ImportError as e:
                logger.warning(f"Could not import comprehensive summary: {e}")
            except Exception as e:
                logger.warning(f"Error in comprehensive summary: {e}")
        else:
            # Simple summary
            print(f"Solution Status: {self.status['success']}")
            print(f"Objective: {self.status['objective']:.6e}")
            print(f"Total Mission Time: {self.status['total_mission_time']:.6f}")
            print(f"Phases: {len(self.phases)}")
            if self.adaptive:
                print(f"Adaptive: Converged in {self.adaptive['iterations']} iterations")
