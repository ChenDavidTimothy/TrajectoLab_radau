import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .mtor_types import FloatArray, OptimalControlSolution, PhaseID, ProblemProtocol


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Solution:
    """
    Optimal control solution with comprehensive data access and analysis capabilities.

    Provides unified interface for accessing optimization results, trajectories,
    solver diagnostics, mesh information, and adaptive refinement data. Supports
    both single-phase and multiphase problems with automatic data concatenation.

    **Data Access Patterns:**

    **Mission-wide access (concatenates all phases):**
    - `solution["variable_name"]` - Variable across all phases
    - `solution["time_states"]` - State time points across all phases
    - `solution["time_controls"]` - Control time points across all phases

    **Phase-specific access:**
    - `solution[(phase_id, "variable_name")]` - Variable in specific phase
    - `solution[(phase_id, "time_states")]` - State times in specific phase
    - `solution[(phase_id, "time_controls")]` - Control times in specific phase

    **Existence checking:**
    - `"variable_name" in solution` - Check mission-wide variable
    - `(phase_id, "variable") in solution` - Check phase-specific variable

    Examples:
        Basic solution workflow:

        >>> solution = mtor.solve_adaptive(problem)
        >>> if solution.status["success"]:
        ...     print(f"Objective: {solution.status['objective']:.6f}")
        ...     solution.plot()

        Mission-wide data access:

        >>> altitude_all = solution["altitude"]       # All phases concatenated
        >>> velocity_all = solution["velocity"]       # All phases concatenated
        >>> state_times_all = solution["time_states"] # All phase state times

        Phase-specific data access:

        >>> altitude_p1 = solution[(1, "altitude")]   # Phase 1 only
        >>> velocity_p2 = solution[(2, "velocity")]   # Phase 2 only
        >>> state_times_p1 = solution[(1, "time_states")]

        Data extraction patterns:

        >>> # Final/initial values
        >>> final_altitude = solution["altitude"][-1]
        >>> initial_velocity = solution["velocity"][0]
        >>> final_mass_p1 = solution[(1, "mass")][-1]
        >>>
        >>> # Extrema
        >>> max_altitude = max(solution["altitude"])
        >>> min_thrust_p2 = min(solution[(2, "thrust")])

        Variable existence checking:

        >>> if "altitude" in solution:
        ...     altitude_data = solution["altitude"]
        >>> if (2, "thrust") in solution:
        ...     thrust_p2 = solution[(2, "thrust")]

        Phase information access:

        >>> for phase_id, phase_data in solution.phases.items():
        ...     duration = phase_data["times"]["duration"]
        ...     state_names = phase_data["variables"]["state_names"]

        Solution validation:

        >>> status = solution.status
        >>> if status["success"]:
        ...     objective = status["objective"]
        ...     mission_time = status["total_mission_time"]
        ... else:
        ...     print(f"Failed: {status['message']}")
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

        Provides comprehensive optimization outcome information including
        success status, objective value, and mission timing. Essential
        for solution validation and performance assessment.

        Returns:
            Dictionary containing complete status information:

            - **success** (bool): Optimization success status
            - **message** (str): Detailed solver status message
            - **objective** (float): Final objective function value
            - **total_mission_time** (float): Sum of all phase durations

        Examples:
            Success checking:

            >>> if solution.status["success"]:
            ...     print("Optimization successful")

            Objective extraction:

            >>> objective = solution.status["objective"]
            >>> mission_time = solution.status["total_mission_time"]

            Error handling:

            >>> status = solution.status
            >>> if not status["success"]:
            ...     print(f"Failed: {status['message']}")
            ...     print(f"Objective: {status['objective']}")  # May be NaN

            Status inspection:

            >>> print(f"Success: {solution.status['success']}")
            >>> print(f"Message: {solution.status['message']}")
            >>> print(f"Objective: {solution.status['objective']:.6e}")
            >>> print(f"Mission time: {solution.status['total_mission_time']:.3f}")
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
        Comprehensive phase information and data organization.

        Provides detailed data for each phase including timing, variables,
        mesh configuration, and trajectory arrays. Essential for understanding
        multiphase structure and accessing phase-specific information.

        Returns:
            Dictionary mapping phase IDs to phase data:

            **Phase data structure:**

            - **times** (dict): Phase timing
                - initial (float): Phase start time
                - final (float): Phase end time
                - duration (float): Phase duration
            - **variables** (dict): Variable information
                - state_names (list): State variable names
                - control_names (list): Control variable names
                - num_states (int): Number of states
                - num_controls (int): Number of controls
            - **mesh** (dict): Mesh configuration
                - polynomial_degrees (list): Polynomial degree per interval
                - mesh_nodes (FloatArray): Mesh node locations
                - num_intervals (int): Total intervals
            - **time_arrays** (dict): Time coordinates
                - states (FloatArray): State time points
                - controls (FloatArray): Control time points
            - **integrals** (float | FloatArray | None): Integral values

        Examples:
            Phase iteration:

            >>> for phase_id, phase_data in solution.phases.items():
            ...     print(f"Phase {phase_id}")

            Timing information:

            >>> phase_1 = solution.phases[1]
            >>> duration = phase_1["times"]["duration"]
            >>> start_time = phase_1["times"]["initial"]
            >>> end_time = phase_1["times"]["final"]

            Variable information:

            >>> variables = solution.phases[1]["variables"]
            >>> state_names = variables["state_names"]     # ["x", "y", "vx", "vy"]
            >>> control_names = variables["control_names"] # ["thrust_x", "thrust_y"]
            >>> num_states = variables["num_states"]       # 4
            >>> num_controls = variables["num_controls"]   # 2

            Mesh information:

            >>> mesh = solution.phases[1]["mesh"]
            >>> degrees = mesh["polynomial_degrees"]       # [6, 8, 6]
            >>> intervals = mesh["num_intervals"]          # 3
            >>> nodes = mesh["mesh_nodes"]                 # [-1, -0.5, 0.5, 1]

            Time arrays:

            >>> time_arrays = solution.phases[1]["time_arrays"]
            >>> state_times = time_arrays["states"]        # State time coordinates
            >>> control_times = time_arrays["controls"]    # Control time coordinates

            Integral values:

            >>> integrals = solution.phases[1]["integrals"]
            >>> if isinstance(integrals, float):
            ...     single_integral = integrals             # Single integral
            >>> else:
            ...     multiple_integrals = integrals          # Array of integrals
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

        Provides access to optimized static parameters with comprehensive
        parameter information. Returns None if no parameters were defined.

        Returns:
            Parameter information dictionary or None:

            - **values** (FloatArray): Optimized parameter values
            - **names** (list[str] | None): Parameter names if available
            - **count** (int): Number of static parameters

        Examples:
            Parameter existence check:

            >>> if solution.parameters is not None:
            ...     print("Problem has static parameters")

            Parameter access:

            >>> params = solution.parameters
            >>> if params:
            ...     values = params["values"]        # [500.0, 1500.0, 0.1]
            ...     count = params["count"]          # 3
            ...     names = params["names"]          # ["mass", "thrust", "drag"] or None

            Named parameter access:

            >>> params = solution.parameters
            >>> if params and params["names"]:
            ...     for name, value in zip(params["names"], params["values"]):
            ...         print(f"{name}: {value:.6f}")

            Unnamed parameter access:

            >>> params = solution.parameters
            >>> if params:
            ...     for i, value in enumerate(params["values"]):
            ...         print(f"Parameter {i}: {value:.6f}")

            No parameters case:

            >>> if solution.parameters is None:
            ...     print("No static parameters in problem")
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
        Provides comprehensive adaptive algorithm performance data including
        convergence status, error estimates, refinement statistics, and complete
        iteration-by-iteration benchmarking data. Only available for adaptive
        solver solutions.

        Returns:
            Adaptive algorithm data dictionary or None:

            - **converged** (bool): Algorithm convergence status
            - **iterations** (int): Refinement iterations performed
            - **target_tolerance** (float): Target error tolerance
            - **phase_converged** (dict): Per-phase convergence status
            - **final_errors** (dict): Final error estimates per phase
            - **gamma_factors** (dict): Normalization factors per phase
            - **iteration_history** (dict): Complete benchmarking data per iteration

        Examples:
            Basic adaptive solution check:

            >>> if solution.adaptive:
            ...     print("Adaptive solution available")

            Convergence assessment:

            >>> adaptive_info = solution.adaptive
            >>> if adaptive_info:
            ...     converged = adaptive_info["converged"]
            ...     iterations = adaptive_info["iterations"]
            ...     tolerance = adaptive_info["target_tolerance"]

            Research benchmarking access:

            >>> if solution.adaptive and "iteration_history" in solution.adaptive:
            ...     history = solution.adaptive["iteration_history"]
            ...     for iteration, data in history.items():
            ...         error = data["max_error_all_phases"]
            ...         points = data["total_collocation_points"]
            ...         print(f"Iteration {iteration}: error={error:.3e}, points={points}")

            Per-phase convergence:

            >>> if solution.adaptive:
            ...     for phase_id, converged in solution.adaptive["phase_converged"].items():
            ...         status = "✓" if converged else "✗"
            ...         print(f"Phase {phase_id}: {status}")

            Error analysis:

            >>> if solution.adaptive:
            ...     for phase_id, errors in solution.adaptive["final_errors"].items():
            ...         max_error = max(errors) if errors else 0.0
            ...         print(f"Phase {phase_id} max error: {max_error:.2e}")

            Fixed mesh solution:

            >>> if solution.adaptive is None:
            ...     print("Fixed mesh solution - no adaptive data")
        """
        if self._raw_solution is None or self._raw_solution.adaptive_data is None:
            return None

        adaptive_data = self._raw_solution.adaptive_data

        result = {
            "converged": adaptive_data.converged,
            "iterations": adaptive_data.total_iterations,
            "target_tolerance": adaptive_data.target_tolerance,
            "phase_converged": adaptive_data.phase_converged,
            "final_errors": adaptive_data.final_phase_error_estimates,
            "gamma_factors": adaptive_data.phase_gamma_factors,
        }

        # Add iteration history for benchmarking if available
        if hasattr(adaptive_data, "iteration_history") and adaptive_data.iteration_history:
            # Convert IterationData objects to dictionaries for API consistency
            iteration_history = {}
            for iteration, data in adaptive_data.iteration_history.items():
                iteration_history[iteration] = {
                    "iteration": data.iteration,
                    "phase_error_estimates": data.phase_error_estimates,
                    "phase_collocation_points": data.phase_collocation_points,
                    "phase_mesh_intervals": data.phase_mesh_intervals,
                    "phase_polynomial_degrees": data.phase_polynomial_degrees,
                    "phase_mesh_nodes": data.phase_mesh_nodes,
                    "refinement_strategy": data.refinement_strategy,
                    "total_collocation_points": data.total_collocation_points,
                    "max_error_all_phases": data.max_error_all_phases,
                    "convergence_status": data.convergence_status,
                }
            result["iteration_history"] = iteration_history

        return result

    def _get_phase_ids(self) -> list[PhaseID]:
        if self._raw_solution is None:
            return []
        return sorted(self._raw_solution.phase_initial_times.keys())

    def __getitem__(self, key: str | tuple[PhaseID, str]) -> FloatArray:
        if not self.status["success"]:
            raise RuntimeError(
                f"Cannot access variable '{key}': Solution failed with message: {self.status['message']}"
            )

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
            raise RuntimeError("Cannot access variable: No solution data available")

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
        Plot solution trajectories with comprehensive customization options.

        Creates trajectory plots with automatic formatting, phase boundaries,
        and flexible variable selection. Supports both single-phase and
        multiphase visualization with professional styling.

        Args:
            phase_id: Phase selection:
                - None: Plot all phases (default)
                - int: Plot specific phase only

            variable_names: Variable selection:
                - Empty: Plot all variables
                - Specified: Plot only named variables

            figsize: Figure size tuple (width, height)

            show_phase_boundaries: Display vertical lines at phase transitions

        Examples:
            Basic plotting:

            >>> solution.plot()  # All variables, all phases

            Specific phase:

            >>> solution.plot(phase_id=1)  # Phase 1 only

            Selected variables:

            >>> solution.plot(phase_id=None, "altitude", "velocity", "thrust")

            Custom formatting:

            >>> solution.plot(
            ...     figsize=(16, 10),
            ...     show_phase_boundaries=True
            ... )

            Phase-specific variables:

            >>> solution.plot(1, "x_position", "y_position")  # Phase 1 positions

            No phase boundaries:

            >>> solution.plot(show_phase_boundaries=False)
        """
        from .plot import plot_multiphase_solution

        plot_multiphase_solution(self, phase_id, variable_names, figsize, show_phase_boundaries)

    def summary(self, comprehensive: bool = True) -> None:
        """
        Display solution summary with comprehensive details and diagnostics.

        Prints detailed overview including solver status, phase information,
        mesh details, and adaptive algorithm results. Essential for solution
        validation and performance analysis.

        Args:
            comprehensive: Summary detail level:
                - True: Full detailed summary (default)
                - False: Concise key information only

        Examples:
            Full summary:

            >>> solution.summary()  # Comprehensive details

            Concise summary:

            >>> solution.summary(comprehensive=False)  # Key information only

            Manual summary control:

            >>> # Solve without automatic summary
            >>> solution = mtor.solve_adaptive(problem, show_summary=False)
            >>> # Display summary when needed
            >>> solution.summary()

            Conditional summary:

            >>> if solution.status["success"]:
            ...     solution.summary()
            ... else:
            ...     solution.summary(comprehensive=False)  # Brief failure info
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

    # ============================================================================
    # BENCHMARKING AND RESEARCH COMPARISON
    # ============================================================================

    def get_benchmark_table(self, phase_id: PhaseID | None = None) -> dict[str, list]:
        """
        Extract adaptive refinement metrics for research benchmarking comparison.

        Provides iteration-by-iteration performance data in format compatible with
        CGPOPS, GPOPS-II, and other pseudospectral method comparisons. Data captured
        represents actual algorithm performance with complete research integrity.

        Args:
            phase_id: Phase selection for analysis:
                - None: Combined multiphase analysis (default)
                - int: Single phase analysis

        Returns:
            Benchmark metrics dictionary with keys:
                - **mesh_iteration**: Iteration numbers [1, 2, 3, ...]
                - **estimated_error**: Maximum error estimates per iteration
                - **collocation_points**: Total collocation points per iteration
                - **mesh_intervals**: Number of mesh intervals per iteration
                - **polynomial_degrees**: Polynomial degree distributions per iteration
                - **refinement_strategy**: h/p refinement decisions per iteration

        Raises:
            ValueError: If no adaptive iteration history available or invalid phase_id

        Examples:
            Complete multiphase benchmarking:

            >>> solution = mtor.solve_adaptive(problem, error_tolerance=1e-6)
            >>> benchmark = solution.get_benchmark_table()
            >>> print("Iteration | Error | Collocation Points")
            >>> for i in range(len(benchmark["mesh_iteration"])):
            ...     iter_num = benchmark["mesh_iteration"][i]
            ...     error = benchmark["estimated_error"][i]
            ...     points = benchmark["collocation_points"][i]
            ...     print(f"{iter_num:9d} | {error:.3e} | {points:15d}")

            Single phase analysis:

            >>> phase1_data = solution.get_benchmark_table(phase_id=1)
            >>> convergence_rate = phase1_data["estimated_error"]

            Export to research paper table:

            >>> import pandas as pd
            >>> benchmark = solution.get_benchmark_table()
            >>> df = pd.DataFrame(benchmark)
            >>> df.to_latex("maptor_performance_table.tex")

            Compare with literature:

            >>> # CGPOPS benchmark comparison
            >>> cgpops_errors = [2.463e-3, 9.891e-5, 3.559e-6, 3.287e-7, 8.706e-8]
            >>> maptor_errors = benchmark["estimated_error"]
            >>> performance_ratio = [m/c for m, c in zip(maptor_errors, cgpops_errors)]
        """
        if not self.adaptive or "iteration_history" not in self.adaptive:
            raise ValueError(
                "No adaptive iteration history available for benchmarking. "
                "Use solve_adaptive() to generate iteration data."
            )

        history = self.adaptive["iteration_history"]

        if not history:
            raise ValueError("Adaptive iteration history is empty")

        if phase_id is not None and not isinstance(phase_id, int):
            raise ValueError(f"phase_id must be integer or None, got {type(phase_id)}")

        # Validate phase_id exists in data
        if phase_id is not None:
            first_iteration = next(iter(history.values()))
            if phase_id not in first_iteration["phase_error_estimates"]:
                available_phases = list(first_iteration["phase_error_estimates"].keys())
                raise ValueError(
                    f"Phase {phase_id} not found. Available phases: {available_phases}"
                )

        mesh_iterations = []
        estimated_errors = []
        collocation_points = []
        mesh_intervals = []
        polynomial_degrees = []
        refinement_strategies = []

        for iteration in sorted(history.keys()):
            data = history[iteration]
            mesh_iterations.append(iteration + 1)  # 1-indexed for research convention

            if phase_id is not None:
                # Single phase benchmarking
                phase_errors = data["phase_error_estimates"][phase_id]
                max_error = max(phase_errors) if phase_errors else float("inf")
                colloc_points = data["phase_collocation_points"][phase_id]
                intervals = data["phase_mesh_intervals"][phase_id]
                degrees = data["phase_polynomial_degrees"][phase_id].copy()
                strategy = data["refinement_strategy"].get(phase_id, {})
            else:
                # Multiphase combined benchmarking
                max_error = data["max_error_all_phases"]
                colloc_points = data["total_collocation_points"]
                intervals = sum(data["phase_mesh_intervals"].values())
                degrees = []
                for phase_degrees in data["phase_polynomial_degrees"].values():
                    degrees.extend(phase_degrees)
                # Combine all phase strategies
                strategy = {}
                for phase_strategy in data["refinement_strategy"].values():
                    strategy.update(phase_strategy)

            estimated_errors.append(max_error)
            collocation_points.append(colloc_points)
            mesh_intervals.append(intervals)
            polynomial_degrees.append(degrees)
            refinement_strategies.append(strategy)

        return {
            "mesh_iteration": mesh_iterations,
            "estimated_error": estimated_errors,
            "collocation_points": collocation_points,
            "mesh_intervals": mesh_intervals,
            "polynomial_degrees": polynomial_degrees,
            "refinement_strategy": refinement_strategies,
        }

    def plot_mesh_refinement_history(
        self,
        phase_id: PhaseID,
        figsize: tuple[float, float] = (12, 6),
        transform_domain: tuple[float, float] | None = None,
        show_strategy: bool = True,
    ) -> None:
        """
        Visualize adaptive mesh refinement history for research analysis.

        Creates mesh point distribution evolution plot showing both mesh interval
        boundaries (red circles) and interior collocation points (black dots).

        Args:
            phase_id: Phase to visualize
            figsize: Figure dimensions (width, height)
            transform_domain: Transform from [-1,1] to physical domain (min, max)
            show_strategy: Display h/p refinement strategy annotations
        """
        if not self.adaptive or "iteration_history" not in self.adaptive:
            raise ValueError("No adaptive iteration history available for plotting")

        history = self.adaptive["iteration_history"]
        if not history:
            raise ValueError("Adaptive iteration history is empty")

        first_iteration = next(iter(history.values()))
        if phase_id not in first_iteration["phase_mesh_nodes"]:
            available_phases = list(first_iteration["phase_mesh_nodes"].keys())
            raise ValueError(f"Phase {phase_id} not found. Available phases: {available_phases}")

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            from maptor.radau import _compute_radau_collocation_components
        except ImportError:
            raise ImportError("matplotlib required for mesh refinement plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Strategy colors
        strategy_colors = {"h": "red", "p": "blue", "none": "gray"}

        for iteration in sorted(history.keys()):
            data = history[iteration]
            mesh_nodes = data["phase_mesh_nodes"][phase_id].copy()
            polynomial_degrees = data["phase_polynomial_degrees"][phase_id]

            # Transform domain if requested
            if transform_domain is not None:
                domain_min, domain_max = transform_domain
                mesh_nodes = domain_min + (mesh_nodes + 1) * (domain_max - domain_min) / 2

            y_position = iteration + 1

            # Plot mesh interval boundaries (red circles)
            ax.scatter(
                mesh_nodes,
                [y_position] * len(mesh_nodes),
                s=60,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidth=2,
            )

            # Plot interior collocation points (black dots)
            collocation_points = []
            for interval_idx in range(len(polynomial_degrees)):
                degree = polynomial_degrees[interval_idx]
                if degree > 0:
                    radau_components = _compute_radau_collocation_components(degree)
                    radau_points = radau_components.collocation_nodes

                    # Transform to current interval
                    interval_start = mesh_nodes[interval_idx]
                    interval_end = mesh_nodes[interval_idx + 1]
                    interval_colloc_points = (
                        interval_start + (radau_points + 1) * (interval_end - interval_start) / 2
                    )
                    collocation_points.extend(interval_colloc_points)

            if collocation_points:
                ax.scatter(
                    collocation_points,
                    [y_position] * len(collocation_points),
                    s=25,
                    marker="o",
                    color="black",
                )

        # Formatting
        domain_label = "Mesh Point Location"
        if transform_domain is not None:
            domain_label += f" [{transform_domain[0]}, {transform_domain[1]}]"

        ax.set_xlabel(domain_label)
        ax.set_ylabel("Mesh Iteration")
        ax.set_title(f"MAPTOR Mesh Refinement History - Phase {phase_id}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, len(history) + 0.5)

        # FIX: Set proper y-tick labels to show iteration numbers
        iterations = sorted(history.keys())
        y_positions = [iter_num + 1 for iter_num in iterations]  # iter 0->y=1, iter 1->y=2, etc
        y_labels = [f"Iter {iter_num}" for iter_num in iterations]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)

        # Simple legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="red",
                markersize=8,
                linewidth=2,
                label="Mesh boundaries",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markersize=6,
                linewidth=0,
                label="Collocation points",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.show()

    def export_benchmark_comparison(
        self,
        filename: str,
        phase_id: PhaseID | None = None,
        format: str = "latex",
    ) -> None:
        """
        Export benchmarking data for research publication comparison.

        Creates formatted tables compatible with academic publications,
        facilitating direct comparison with CGPOPS, GPOPS-II, and other
        pseudospectral optimal control methods.

        Args:
            filename: Output filename (extension determines format if not specified)
            phase_id: Phase to export (None for multiphase analysis)
            format: Export format ("latex", "csv", "json")

        Examples:
            LaTeX table for publication:

            >>> solution.export_benchmark_comparison(
            ...     "maptor_vs_cgpops.tex",
            ...     phase_id=1,
            ...     format="latex"
            ... )

            CSV data for analysis:

            >>> solution.export_benchmark_comparison(
            ...     "refinement_data.csv",
            ...     format="csv"
            ... )

            JSON for web visualization:

            >>> solution.export_benchmark_comparison(
            ...     "adaptive_metrics.json",
            ...     format="json"
            ... )
        """
        benchmark_data = self.get_benchmark_table(phase_id=phase_id)

        if format == "latex":
            _export_latex_table(benchmark_data, filename, phase_id)
        elif format == "csv":
            _export_csv_table(benchmark_data, filename)
        elif format == "json":
            _export_json_table(benchmark_data, filename)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'latex', 'csv', or 'json'")


def _export_latex_table(benchmark_data: dict, filename: str, phase_id: PhaseID | None) -> None:
    """Export benchmark data as LaTeX table for research publications."""
    phase_label = f"Phase {phase_id}" if phase_id is not None else "Multiphase"
    latex_content = f"""% MAPTOR Benchmark Results - {phase_label}
\\begin{{table}}[h]
\\centering
\\caption{{MAPTOR Performance on Example Problem Using hp-adaptive refinement}}
\\label{{tab:maptor_performance}}
\\begin{{tabular}}{{|c|c|c|}}
\\hline
Mesh Iteration & Estimated Error & Number of Collocation Points \\\\
\\hline
"""
    for i in range(len(benchmark_data["mesh_iteration"])):
        iteration = benchmark_data["mesh_iteration"][i]
        error = benchmark_data["estimated_error"][i]
        points = benchmark_data["collocation_points"][i]
        latex_content += f"{iteration} & {error:.3e} & {points} \\\\\n"
    latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
    with open(filename, "w") as f:
        f.write(latex_content)


def _export_csv_table(benchmark_data: dict, filename: str) -> None:
    """Export benchmark data as CSV for data analysis."""
    import csv

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["mesh_iteration", "estimated_error", "collocation_points", "mesh_intervals"]
        )
        for i in range(len(benchmark_data["mesh_iteration"])):
            writer.writerow(
                [
                    benchmark_data["mesh_iteration"][i],
                    benchmark_data["estimated_error"][i],
                    benchmark_data["collocation_points"][i],
                    benchmark_data["mesh_intervals"][i],
                ]
            )


def _export_json_table(benchmark_data: dict, filename: str) -> None:
    """Export benchmark data as JSON for web applications."""
    import json

    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for key, value in benchmark_data.items():
        if hasattr(value[0], "tolist"):  # numpy array
            json_data[key] = [v.tolist() if hasattr(v, "tolist") else v for v in value]
        else:
            json_data[key] = value
    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)
