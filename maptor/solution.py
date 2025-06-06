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
    adaptive refinement data. It supports both single-phase and multiphase problems
    with a unified API design.

    Examples:
        Basic solution usage:

        >>> solution = mtor.solve_fixed_mesh(problem)
        >>> if solution.status["success"]:
        ...     print(f"Objective: {solution.status['objective']:.6f}")
        ...     solution.plot()

        Accessing trajectory data:

        >>> # Get specific variable data
        >>> altitude = solution["altitude"]  # Auto-finds in any phase
        >>> thrust_p1 = solution[(1, "thrust")]  # Specific phase
        >>>
        >>> # Get time arrays
        >>> time_states = solution[(1, "time_states")]
        >>> time_controls = solution[(1, "time_controls")]

        Working with multiphase results:

        >>> # Examine all phases
        >>> for phase_id, phase_data in solution.phases.items():
        ...     duration = phase_data["times"]["duration"]
        ...     states = phase_data["variables"]["state_names"]
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
            Check if optimization succeeded:

            >>> solution = mtor.solve_fixed_mesh(problem)
            >>> if solution.status["success"]:
            ...     print("Optimization successful!")
            ...     print(f"Optimal objective: {solution.status['objective']:.6f}")
            ... else:
            ...     print(f"Optimization failed: {solution.status['message']}")

            Get mission duration:

            >>> if solution.status["success"]:
            ...     total_time = solution.status["total_mission_time"]
            ...     objective = solution.status["objective"]
            ...     print(f"Mission completed in {total_time:.3f} seconds")
            ...     print(f"Final cost: {objective:.6e}")

            Handle optimization failures:

            >>> status = solution.status
            >>> if not status["success"]:
            ...     print("Optimization failed!")
            ...     print(f"Reason: {status['message']}")
            ...     print("Try different initial guess or solver options")
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
            Examine all phases in a multiphase mission:

            >>> for phase_id, phase_data in solution.phases.items():
            ...     times = phase_data["times"]
            ...     variables = phase_data["variables"]
            ...     print(f"Phase {phase_id}:")
            ...     print(f"  Duration: {times['duration']:.3f} seconds")
            ...     print(f"  States: {variables['state_names']}")
            ...     print(f"  Controls: {variables['control_names']}")

            Get specific phase information:

            >>> phase_1 = solution.phases[1]
            >>> launch_duration = phase_1["times"]["duration"]
            >>> num_states = phase_1["variables"]["num_states"]
            >>> mesh_intervals = phase_1["mesh"]["num_intervals"]
            >>> print(f"Launch phase: {launch_duration:.1f}s, {num_states} states, {mesh_intervals} intervals")

            Check mesh refinement results:

            >>> for phase_id, phase_data in solution.phases.items():
            ...     mesh = phase_data["mesh"]
            ...     degrees = mesh["polynomial_degrees"]
            ...     intervals = mesh["num_intervals"]
            ...     print(f"Phase {phase_id}: {intervals} intervals, degrees {degrees}")

            Access time arrays for custom analysis:

            >>> phase_2 = solution.phases[2]
            >>> state_times = phase_2["time_arrays"]["states"]
            >>> control_times = phase_2["time_arrays"]["controls"]
            >>> print(f"State points: {len(state_times)}, Control points: {len(control_times)}")
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
            Check if problem has static parameters:

            >>> if solution.parameters is not None:
            ...     param_info = solution.parameters
            ...     print(f"Found {param_info['count']} static parameters")
            ...     print(f"Values: {param_info['values']}")
            ... else:
            ...     print("No static parameters in this problem")

            Access specific parameter values:

            >>> params = solution.parameters
            >>> if params and params["names"]:
            ...     for name, value in zip(params["names"], params["values"]):
            ...         print(f"{name}: {value:.6f}")
            ... else:
            ...     # Parameters exist but no names available
            ...     for i, value in enumerate(params["values"]):
            ...         print(f"Parameter {i}: {value:.6f}")

            Use parameters in post-processing:

            >>> if solution.parameters:
            ...     optimal_mass = solution.parameters["values"][0]
            ...     optimal_thrust = solution.parameters["values"][1]
            ...     print(f"Optimal design: mass={optimal_mass:.1f}, thrust={optimal_thrust:.1f}")
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
            Check adaptive convergence status:

            >>> if solution.adaptive:
            ...     adaptive_info = solution.adaptive
            ...     if adaptive_info["converged"]:
            ...         print(f"Converged in {adaptive_info['iterations']} iterations")
            ...         print(f"Target tolerance: {adaptive_info['target_tolerance']:.1e}")
            ...     else:
            ...         print("Did not converge within iteration limit")
            ... else:
            ...     print("Fixed mesh solution - no adaptive data available")

            Analyze per-phase convergence:

            >>> if solution.adaptive:
            ...     for phase_id, converged in solution.adaptive["phase_converged"].items():
            ...         status = "✓" if converged else "✗"
            ...         print(f"Phase {phase_id}: {status}")

            Examine final error estimates:

            >>> if solution.adaptive:
            ...     for phase_id, errors in solution.adaptive["final_errors"].items():
            ...         max_error = max(errors) if errors else 0.0
            ...         avg_error = np.mean(errors) if errors else 0.0
            ...         print(f"Phase {phase_id}: max_error={max_error:.2e}, avg_error={avg_error:.2e}")

            Get refinement statistics:

            >>> if solution.adaptive:
            ...     total_iterations = solution.adaptive["iterations"]
            ...     target_tol = solution.adaptive["target_tolerance"]
            ...     converged = solution.adaptive["converged"]
            ...     print(f"Adaptive refinement: {total_iterations} iterations")
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
        """Get sorted list of phase IDs."""
        if self._raw_solution is None:
            return []
        return sorted(self._raw_solution.phase_initial_times.keys())

    def __getitem__(self, key: str | tuple[PhaseID, str]) -> FloatArray:
        """
        Dictionary-style access to solution variables and time arrays.

        Provides convenient access to trajectory data using either variable names
        (searches all phases) or explicit (phase_id, variable_name) tuples.
        This is the primary method for extracting solution data for analysis.

        Args:
            key: Either a variable name (searches all phases) or (phase_id, variable_name) tuple

        Returns:
            FloatArray containing the requested trajectory or time data

        Raises:
            KeyError: If the variable name is not found in any phase

        Examples:
            Access variables by name (auto-search all phases):

            >>> # Get trajectory data from any phase containing this variable
            >>> altitude_data = solution["altitude"]
            >>> velocity_data = solution["velocity"]
            >>> thrust_data = solution["thrust"]

            Access variables from specific phases:

            >>> # Get data from specific phases (useful for multiphase problems)
            >>> launch_altitude = solution[(1, "altitude")]  # Phase 1 altitude
            >>> orbit_velocity = solution[(2, "velocity")]   # Phase 2 velocity
            >>> descent_thrust = solution[(3, "thrust")]     # Phase 3 thrust

            Access time coordinate arrays:

            >>> # Get time points for state and control trajectories
            >>> state_times = solution[(1, "time_states")]    # Time points for states
            >>> control_times = solution[(1, "time_controls")] # Time points for controls

            Use in plotting and analysis:

            >>> # Extract data for custom plotting
            >>> import matplotlib.pyplot as plt
            >>> t = solution[(1, "time_states")]
            >>> x = solution[(1, "position")]
            >>> v = solution[(1, "velocity")]
            >>>
            >>> plt.figure(figsize=(12, 4))
            >>> plt.subplot(1, 2, 1)
            >>> plt.plot(t, x)
            >>> plt.xlabel("Time (s)")
            >>> plt.ylabel("Position")
            >>>
            >>> plt.subplot(1, 2, 2)
            >>> plt.plot(t, v)
            >>> plt.xlabel("Time (s)")
            >>> plt.ylabel("Velocity")
            >>> plt.show()

            Handle missing variables gracefully:

            >>> try:
            ...     fuel_data = solution["fuel_mass"]
            >>> except KeyError:
            ...     print("Fuel mass not found in solution")
            ...     # List available variables
            ...     for phase_id, phase_info in solution.phases.items():
            ...         vars_list = phase_info["variables"]["state_names"] + phase_info["variables"]["control_names"]
            ...         print(f"Phase {phase_id} variables: {vars_list}")
        """
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
        for phase_id in self._get_phase_ids():
            try:
                return self[(phase_id, key)]
            except KeyError:
                continue

        all_vars = []
        for phase_id in self._get_phase_ids():
            phase_vars = (
                self._phase_state_names.get(phase_id, [])
                + self._phase_control_names.get(phase_id, [])
                + ["time_states", "time_controls"]
            )
            all_vars.extend([f"({phase_id}, '{var}')" for var in phase_vars])

        raise KeyError(f"Variable '{key}' not found in any phase. Available: {all_vars}")

    def __contains__(self, key: str | tuple[PhaseID, str]) -> bool:
        """
        Check if a variable exists in the solution.

        Convenient method to verify variable availability before attempting
        to access it, preventing KeyError exceptions.

        Args:
            key: Either a variable name or (phase_id, variable_name) tuple

        Returns:
            True if variable exists, False otherwise

        Examples:
            Check variable availability before access:

            >>> if "altitude" in solution:
            ...     altitude_data = solution["altitude"]
            ...     print(f"Altitude range: {altitude_data.min():.1f} to {altitude_data.max():.1f}")
            ... else:
            ...     print("Altitude not available in this solution")

            Check phase-specific variables:

            >>> if (1, "thrust") in solution:
            ...     thrust_data = solution[(1, "thrust")]
            ...     max_thrust = thrust_data.max()
            ...     print(f"Maximum thrust in phase 1: {max_thrust:.2f}")
            ... else:
            ...     print("Thrust not available in phase 1")

            Validate multiple variables:

            >>> required_vars = ["position", "velocity", "acceleration"]
            >>> available_vars = [var for var in required_vars if var in solution]
            >>> missing_vars = [var for var in required_vars if var not in solution]
            >>>
            >>> if missing_vars:
            ...     print(f"Missing variables: {missing_vars}")
            ... else:
            ...     print("All required variables available")
        """
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
            Plot all variables for all phases:

            >>> solution.plot()  # Comprehensive plot with all variables

            Plot specific phase only:

            >>> solution.plot(phase_id=1)  # Only phase 1 trajectories

            Plot specific variables across all phases:

            >>> solution.plot(phase_id=None, "altitude", "velocity", "thrust")

            Plot specific variables for specific phase:

            >>> solution.plot(1, "position", "velocity")  # Phase 1, specific variables

            Customize plot appearance:

            >>> solution.plot(
            ...     figsize=(16, 10),           # Larger figure
            ...     show_phase_boundaries=False  # Hide phase boundaries
            ... )

            Plot with custom analysis:

            >>> # Plot solution then add custom annotations
            >>> solution.plot(1, "altitude", "velocity")
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Add custom annotations to the current plot
            >>> plt.figure(1)  # Access the altitude plot
            >>> plt.axhline(y=10000, color='red', linestyle='--', label='Target altitude')
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
            Display full solution summary (default):

            >>> solution.summary()
            # Prints comprehensive summary including:
            # - Solver status and objective value
            # - Phase-by-phase breakdown with timing
            # - Mesh configuration and refinement details
            # - Adaptive algorithm convergence (if applicable)
            # - Variable counts and trajectory statistics

            Display concise summary:

            >>> solution.summary(comprehensive=False)
            # Prints brief summary with:
            # - Success status and objective
            # - Total mission time
            # - Number of phases
            # - Adaptive convergence status

            Use in solution validation workflow:

            >>> solution = mtor.solve_adaptive(problem)
            >>>
            >>> # Always check summary first
            >>> solution.summary()
            >>>
            >>> # Then proceed with detailed analysis if successful
            >>> if solution.status["success"]:
            ...     # Extract specific data
            ...     trajectory_data = solution["altitude"]
            ...     solution.plot()
            ... else:
            ...     print("Solution failed - check summary for details")

            Suppress automatic summary:

            >>> # Solve without automatic summary display
            >>> solution = mtor.solve_fixed_mesh(problem, show_summary=False)
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
            # Simple summary using new API
            print(f"Solution Status: {self.status['success']}")
            print(f"Objective: {self.status['objective']:.6e}")
            print(f"Total Mission Time: {self.status['total_mission_time']:.6f}")
            print(f"Phases: {len(self.phases)}")
            if self.adaptive:
                print(f"Adaptive: Converged in {self.adaptive['iterations']} iterations")
