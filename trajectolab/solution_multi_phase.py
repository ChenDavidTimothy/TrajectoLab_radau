"""
Multi-phase solution wrapper with visualization and analysis capabilities.

This module provides the MultiPhaseSolution class that wraps MultiPhaseOptimalControlSolution
with comprehensive analysis, visualization, and data access methods specifically designed
for multi-phase optimal control problems.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .exceptions import ConfigurationError
from .solution import Solution
from .tl_types import (
    MultiPhaseOptimalControlSolution,
    MultiPhaseProblemProtocol,
    PhaseEndpointVector,
)


# Library logger
logger = logging.getLogger(__name__)


class MultiPhaseSolution:
    """
    Multi-phase solution wrapper with visualization and analysis capabilities.

    Provides a comprehensive interface for analyzing and visualizing multi-phase
    optimal control solutions, maintaining the CGPOPS mathematical structure
    while offering intuitive access to solution data and analysis tools.

    The class wraps MultiPhaseOptimalControlSolution and provides:
    - Individual phase solution access
    - Multi-phase trajectory visualization
    - Phase transition analysis
    - Continuity and discontinuity assessment
    - Global parameter and constraint analysis
    - Export capabilities for further analysis

    Args:
        solution_data: Multi-phase optimal control solution data
        problem: Multi-phase problem protocol for context

    Example:
        >>> # After solving multi-phase problem
        >>> solution = solve_multi_phase_fixed_mesh(mp_problem)
        >>>
        >>> # Analyze solution
        >>> if solution.success:
        ...     print(f"Optimal objective: {solution.objective:.6f}")
        ...     print(f"Global parameters: {solution.global_parameters}")
        ...
        ...     # Plot all phases
        ...     solution.plot_phases()
        ...
        ...     # Analyze phase transitions
        ...     transitions = solution.analyze_phase_transitions()
        ...     print(f"Max discontinuity: {transitions['max_state_discontinuity']:.6e}")
        ...
        ...     # Access individual phases
        ...     phase_0 = solution.get_phase_solution(0)
        ...     phase_0.plot()
    """

    def __init__(
        self,
        solution_data: MultiPhaseOptimalControlSolution,
        problem: MultiPhaseProblemProtocol,
    ) -> None:
        """Initialize multi-phase solution wrapper."""
        self.solution_data = solution_data
        self.problem = problem

        # Validate solution data
        if solution_data is None:
            raise ConfigurationError(
                "Solution data cannot be None", "Multi-phase solution initialization error"
            )

        if solution_data.phase_count != problem.get_phase_count():
            raise ConfigurationError(
                f"Solution phase count ({solution_data.phase_count}) doesn't match "
                f"problem phase count ({problem.get_phase_count()})",
                "Multi-phase solution initialization error",
            )

        # Cache frequently accessed properties
        self._phase_solutions_cache: list[Solution] | None = None
        self._transition_analysis_cache: dict[str, Any] | None = None

        logger.debug(
            "Initialized MultiPhaseSolution: %d phases, success=%s", self.phase_count, self.success
        )

    # ========================================================================
    # BASIC PROPERTIES - Direct access to solution data
    # ========================================================================

    @property
    def success(self) -> bool:
        """Whether the multi-phase solve was successful."""
        return self.solution_data.success

    @property
    def message(self) -> str:
        """Solver message describing the solution status."""
        return self.solution_data.message

    @property
    def objective(self) -> float | None:
        """Global objective function value."""
        return self.solution_data.objective

    @property
    def phase_count(self) -> int:
        """Number of phases in the solution."""
        return self.solution_data.phase_count

    @property
    def global_parameters(self) -> dict[str, float]:
        """Global static parameters and their optimized values."""
        return dict(self.solution_data.global_parameters)

    @property
    def phase_endpoints(self) -> list[PhaseEndpointVector]:
        """Phase endpoint vectors E^(p) for all phases."""
        return list(self.solution_data.phase_endpoints)

    @property
    def solve_time(self) -> float | None:
        """Total solve time (if available)."""
        return self.solution_data.solve_time

    @property
    def nlp_iterations(self) -> int | None:
        """Number of NLP iterations (if available)."""
        return self.solution_data.nlp_iterations

    # ========================================================================
    # PHASE SOLUTION ACCESS - Individual phase extraction
    # ========================================================================

    def get_phase_solution(self, phase_index: int) -> Solution:
        """
        Get solution for specific phase wrapped in Solution class.

        Args:
            phase_index: Zero-based phase index

        Returns:
            Solution object for the specified phase

        Raises:
            ConfigurationError: If phase index is out of bounds

        Example:
            >>> phase_0_solution = solution.get_phase_solution(0)
            >>> phase_0_solution.plot()
            >>> print(f"Phase 0 final time: {phase_0_solution.final_time}")
        """
        if not (0 <= phase_index < self.phase_count):
            raise ConfigurationError(
                f"Phase index {phase_index} out of range [0, {self.phase_count})",
                "Multi-phase solution phase access error",
            )

        # Use cached solutions if available
        if self._phase_solutions_cache is None:
            self._phase_solutions_cache = self._create_phase_solution_wrappers()

        return self._phase_solutions_cache[phase_index]

    def get_all_phase_solutions(self) -> list[Solution]:
        """
        Get all phase solutions as list of Solution objects.

        Returns:
            List of Solution objects, one per phase

        Example:
            >>> phase_solutions = solution.get_all_phase_solutions()
            >>> for i, phase_sol in enumerate(phase_solutions):
            ...     print(f"Phase {i} final time: {phase_sol.final_time}")
        """
        if self._phase_solutions_cache is None:
            self._phase_solutions_cache = self._create_phase_solution_wrappers()

        return list(self._phase_solutions_cache)

    def _create_phase_solution_wrappers(self) -> list[Solution]:
        """Create Solution wrappers for all phase solutions."""
        logger.debug("Creating phase solution wrappers")

        phase_solutions = []
        for phase_idx, phase_solution_data in enumerate(self.solution_data.phase_solutions):
            try:
                # Create wrapper for individual phase
                # Note: This requires the individual phase problem, which we can get from the multi-phase problem
                phase_problem = self.problem.phases[phase_idx]
                phase_solution = Solution(phase_solution_data, phase_problem)
                phase_solutions.append(phase_solution)

            except Exception as e:
                logger.warning("Failed to create wrapper for phase %d: %s", phase_idx, str(e))

                # Create a minimal wrapper for failed phase
                class MinimalSolution:
                    def __init__(self, data):
                        self.solution_data = data
                        self.success = data.success
                        self.message = data.message

                    def plot(self):
                        print(f"Plotting not available for phase {phase_idx}: {self.message}")

                phase_solutions.append(MinimalSolution(phase_solution_data))

        return phase_solutions

    # ========================================================================
    # MULTI-PHASE VISUALIZATION - Comprehensive plotting capabilities
    # ========================================================================

    def plot_phases(
        self,
        phase_indices: list[int] | None = None,
        states: list[str] | None = None,
        controls: list[str] | None = None,
        show_transitions: bool = True,
        figsize: tuple[float, float] = (12, 8),
    ) -> None:
        """
        Plot trajectories across multiple phases.

        Creates comprehensive multi-phase trajectory plots showing state and control
        evolution across all phases with phase transition markers.

        Args:
            phase_indices: List of phase indices to plot (None for all phases)
            states: List of state names to plot (None for all states)
            controls: List of control names to plot (None for all controls)
            show_transitions: Whether to show phase transition markers
            figsize: Figure size (width, height)

        Example:
            >>> # Plot all phases
            >>> solution.plot_phases()
            >>>
            >>> # Plot specific phases and variables
            >>> solution.plot_phases(
            ...     phase_indices=[0, 2],
            ...     states=['position', 'velocity'],
            ...     controls=['thrust'],
            ...     show_transitions=True
            ... )
        """
        if not self.success:
            logger.warning("Cannot plot failed solution")
            print(f"Solution failed: {self.message}")
            return

        logger.debug("Creating multi-phase trajectory plot")

        try:
            # Determine phases to plot
            if phase_indices is None:
                phase_indices = list(range(self.phase_count))
            else:
                # Validate phase indices
                for idx in phase_indices:
                    if not (0 <= idx < self.phase_count):
                        raise ConfigurationError(
                            f"Phase index {idx} out of range [0, {self.phase_count})",
                            "Multi-phase plotting error",
                        )

            # Get phase solutions to plot
            phases_to_plot = [self.get_phase_solution(i) for i in phase_indices]

            # Determine variables to plot
            if states is None and controls is None:
                # Plot all available variables from first successful phase
                for phase_sol in phases_to_plot:
                    if phase_sol.success and hasattr(phase_sol, "solution_data"):
                        try:
                            states = self.problem.phases[
                                phase_indices[phases_to_plot.index(phase_sol)]
                            ].get_ordered_state_names()
                            controls = self.problem.phases[
                                phase_indices[phases_to_plot.index(phase_sol)]
                            ].get_ordered_control_names()
                            break
                        except Exception:
                            continue

            # Create multi-phase plot
            self._create_multi_phase_plot(
                phases_to_plot, phase_indices, states, controls, show_transitions, figsize
            )

        except Exception as e:
            logger.error("Multi-phase plotting failed: %s", str(e))
            print(f"Multi-phase plotting failed: {e}")

    def _create_multi_phase_plot(
        self,
        phase_solutions: list[Solution],
        phase_indices: list[int],
        states: list[str] | None,
        controls: list[str] | None,
        show_transitions: bool,
        figsize: tuple[float, float],
    ) -> None:
        """Create the actual multi-phase plot."""
        # Calculate subplot layout
        num_plots = 0
        if states:
            num_plots += len(states)
        if controls:
            num_plots += len(controls)

        if num_plots == 0:
            print("No variables specified for plotting")
            return

        # Create figure and subplots
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(phase_indices)))

        # Plot states
        if states:
            for state_idx, state_name in enumerate(states):
                ax = axes[plot_idx]

                for phase_idx, (phase_sol, color) in enumerate(
                    zip(phase_solutions, colors, strict=False)
                ):
                    if phase_sol.success and hasattr(phase_sol, "solution_data"):
                        try:
                            self._plot_phase_variable(
                                ax,
                                phase_sol,
                                "state",
                                state_idx,
                                state_name,
                                phase_indices[phase_idx],
                                color,
                                show_transitions,
                            )
                        except Exception as e:
                            logger.debug(
                                "Failed to plot state %s for phase %d: %s",
                                state_name,
                                phase_indices[phase_idx],
                                str(e),
                            )

                ax.set_ylabel(state_name)
                ax.grid(True, alpha=0.3)
                ax.legend()
                plot_idx += 1

        # Plot controls
        if controls:
            for control_idx, control_name in enumerate(controls):
                ax = axes[plot_idx]

                for phase_idx, (phase_sol, color) in enumerate(
                    zip(phase_solutions, colors, strict=False)
                ):
                    if phase_sol.success and hasattr(phase_sol, "solution_data"):
                        try:
                            self._plot_phase_variable(
                                ax,
                                phase_sol,
                                "control",
                                control_idx,
                                control_name,
                                phase_indices[phase_idx],
                                color,
                                show_transitions,
                            )
                        except Exception as e:
                            logger.debug(
                                "Failed to plot control %s for phase %d: %s",
                                control_name,
                                phase_indices[phase_idx],
                                str(e),
                            )

                ax.set_ylabel(control_name)
                ax.grid(True, alpha=0.3)
                ax.legend()
                plot_idx += 1

        # Set common labels
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Multi-Phase Trajectory ({len(phase_indices)} phases)")
        plt.tight_layout()
        plt.show()

    def _plot_phase_variable(
        self,
        ax,
        phase_solution: Solution,
        var_type: str,
        var_index: int,
        var_name: str,
        phase_index: int,
        color,
        show_transitions: bool,
    ) -> None:
        """Plot a single variable for a single phase."""
        try:
            if var_type == "state":
                if (
                    hasattr(phase_solution.solution_data, "time_states")
                    and hasattr(phase_solution.solution_data, "states")
                    and var_index < len(phase_solution.solution_data.states)
                ):
                    time_data = phase_solution.solution_data.time_states
                    var_data = phase_solution.solution_data.states[var_index]
                    ax.plot(
                        time_data, var_data, color=color, label=f"Phase {phase_index}", linewidth=2
                    )

            elif var_type == "control":
                if (
                    hasattr(phase_solution.solution_data, "time_controls")
                    and hasattr(phase_solution.solution_data, "controls")
                    and var_index < len(phase_solution.solution_data.controls)
                ):
                    time_data = phase_solution.solution_data.time_controls
                    var_data = phase_solution.solution_data.controls[var_index]
                    ax.plot(
                        time_data,
                        var_data,
                        color=color,
                        label=f"Phase {phase_index}",
                        linestyle="--",
                        linewidth=2,
                    )

            # Add phase transition markers
            if show_transitions:
                if hasattr(phase_solution.solution_data, "initial_time_variable"):
                    ax.axvline(
                        phase_solution.solution_data.initial_time_variable,
                        color=color,
                        alpha=0.5,
                        linestyle=":",
                    )
                if hasattr(phase_solution.solution_data, "terminal_time_variable"):
                    ax.axvline(
                        phase_solution.solution_data.terminal_time_variable,
                        color=color,
                        alpha=0.5,
                        linestyle=":",
                    )

        except Exception as e:
            logger.debug("Variable plotting failed for %s %s: %s", var_type, var_name, str(e))

    # ========================================================================
    # PHASE TRANSITION ANALYSIS - Comprehensive transition analysis
    # ========================================================================

    def analyze_phase_transitions(self) -> dict[str, Any]:
        """
        Analyze continuity and discontinuities at phase boundaries.

        Performs comprehensive analysis of phase transitions including:
        - State continuity/discontinuity analysis
        - Time continuity analysis
        - Constraint violation analysis
        - Jump magnitude quantification

        Returns:
            Dictionary containing comprehensive transition analysis

        Example:
            >>> analysis = solution.analyze_phase_transitions()
            >>> print(f"Max state discontinuity: {analysis['max_state_discontinuity']:.6e}")
            >>> print(f"Continuous transitions: {analysis['continuous_transitions']}")
            >>> print(f"Discontinuous transitions: {analysis['discontinuous_transitions']}")
        """
        if self._transition_analysis_cache is not None:
            return dict(self._transition_analysis_cache)

        logger.debug("Analyzing phase transitions")

        try:
            # Use built-in analysis from solution data
            if hasattr(self.solution_data, "analyze_phase_continuity"):
                analysis = self.solution_data.analyze_phase_continuity()
            else:
                # Perform custom analysis
                analysis = self._perform_phase_transition_analysis()

            # Cache the results
            self._transition_analysis_cache = analysis

            logger.debug("Phase transition analysis completed")
            return dict(analysis)

        except Exception as e:
            logger.error("Phase transition analysis failed: %s", str(e))
            return {
                "error": str(e),
                "max_state_discontinuity": 0.0,
                "max_time_gap": 0.0,
                "continuous_transitions": [],
                "discontinuous_transitions": [],
            }

    def _perform_phase_transition_analysis(self) -> dict[str, Any]:
        """Perform custom phase transition analysis."""
        analysis = {
            "max_state_discontinuity": 0.0,
            "max_time_gap": 0.0,
            "continuous_transitions": [],
            "discontinuous_transitions": [],
            "transition_details": {},
        }

        if self.phase_count < 2:
            return analysis

        # Analyze each phase transition
        for i in range(self.phase_count - 1):
            try:
                phase_i_sol = self.get_phase_solution(i)
                phase_j_sol = self.get_phase_solution(i + 1)

                transition_key = f"phase_{i}_to_{i + 1}"
                transition_analysis = self._analyze_single_transition(phase_i_sol, phase_j_sol)

                analysis["transition_details"][transition_key] = transition_analysis

                # Update global statistics
                if transition_analysis["state_discontinuity_magnitude"] is not None:
                    analysis["max_state_discontinuity"] = max(
                        analysis["max_state_discontinuity"],
                        transition_analysis["state_discontinuity_magnitude"],
                    )

                if transition_analysis["time_gap"] is not None:
                    analysis["max_time_gap"] = max(
                        analysis["max_time_gap"], abs(transition_analysis["time_gap"])
                    )

                # Classify transition
                if transition_analysis["is_continuous"]:
                    analysis["continuous_transitions"].append((i, i + 1))
                else:
                    analysis["discontinuous_transitions"].append((i, i + 1))

            except Exception as e:
                logger.warning("Failed to analyze transition %d->%d: %s", i, i + 1, str(e))

        return analysis

    def _analyze_single_transition(self, phase_i: Solution, phase_j: Solution) -> dict[str, Any]:
        """Analyze a single phase transition."""
        analysis = {
            "state_discontinuity_magnitude": None,
            "time_gap": None,
            "is_continuous": False,
            "state_discontinuity_vector": None,
        }

        try:
            # Analyze state continuity
            if (
                phase_i.success
                and phase_j.success
                and hasattr(phase_i.solution_data, "states")
                and hasattr(phase_j.solution_data, "states")
            ):
                if (
                    phase_i.solution_data.states
                    and phase_j.solution_data.states
                    and len(phase_i.solution_data.states) > 0
                    and len(phase_j.solution_data.states) > 0
                ):
                    # Get final state of phase i and initial state of phase j
                    final_state_i = np.array([state[-1] for state in phase_i.solution_data.states])
                    initial_state_j = np.array([state[0] for state in phase_j.solution_data.states])

                    # Calculate discontinuity
                    state_discontinuity = final_state_i - initial_state_j
                    analysis["state_discontinuity_vector"] = state_discontinuity
                    analysis["state_discontinuity_magnitude"] = np.linalg.norm(state_discontinuity)

            # Analyze time continuity
            if hasattr(phase_i.solution_data, "terminal_time_variable") and hasattr(
                phase_j.solution_data, "initial_time_variable"
            ):
                if (
                    phase_i.solution_data.terminal_time_variable is not None
                    and phase_j.solution_data.initial_time_variable is not None
                ):
                    analysis["time_gap"] = (
                        phase_j.solution_data.initial_time_variable
                        - phase_i.solution_data.terminal_time_variable
                    )

            # Determine if transition is continuous
            state_continuous = (
                analysis["state_discontinuity_magnitude"] is None
                or analysis["state_discontinuity_magnitude"] < 1e-6
            )
            time_continuous = analysis["time_gap"] is None or abs(analysis["time_gap"]) < 1e-6

            analysis["is_continuous"] = state_continuous and time_continuous

        except Exception as e:
            logger.debug("Single transition analysis failed: %s", str(e))

        return analysis

    # ========================================================================
    # SOLUTION SUMMARY AND REPORTING
    # ========================================================================

    def get_solution_summary(self) -> dict[str, Any]:
        """
        Get comprehensive solution summary.

        Returns detailed summary including solution status, objective value,
        phase information, global parameters, and transition analysis.

        Returns:
            Dictionary containing comprehensive solution summary

        Example:
            >>> summary = solution.get_solution_summary()
            >>> print(f"Objective: {summary['objective']}")
            >>> print(f"Phase count: {summary['phase_count']}")
            >>> print(f"Global parameters: {summary['global_parameters']}")
        """
        logger.debug("Creating solution summary")

        summary = {
            # Basic solution information
            "success": self.success,
            "message": self.message,
            "objective": self.objective,
            "phase_count": self.phase_count,
            # Global information
            "global_parameters": self.global_parameters,
            # Timing information
            "solve_time": self.solve_time,
            "nlp_iterations": self.nlp_iterations,
            # Phase information
            "phases": [],
            # Transition analysis
            "transition_analysis": None,
        }

        # Add phase-specific information
        try:
            for phase_idx in range(self.phase_count):
                phase_sol = self.get_phase_solution(phase_idx)
                phase_info = {
                    "index": phase_idx,
                    "success": phase_sol.success,
                    "message": getattr(phase_sol, "message", "No message"),
                    "initial_time": getattr(phase_sol.solution_data, "initial_time_variable", None),
                    "terminal_time": getattr(
                        phase_sol.solution_data, "terminal_time_variable", None
                    ),
                }

                # Add variable information if available
                if hasattr(phase_sol.solution_data, "states"):
                    phase_info["num_states"] = (
                        len(phase_sol.solution_data.states) if phase_sol.solution_data.states else 0
                    )
                if hasattr(phase_sol.solution_data, "controls"):
                    phase_info["num_controls"] = (
                        len(phase_sol.solution_data.controls)
                        if phase_sol.solution_data.controls
                        else 0
                    )

                summary["phases"].append(phase_info)

        except Exception as e:
            logger.warning("Failed to create phase information in summary: %s", str(e))

        # Add transition analysis
        try:
            summary["transition_analysis"] = self.analyze_phase_transitions()
        except Exception as e:
            logger.warning("Failed to add transition analysis to summary: %s", str(e))

        return summary

    def print_solution_summary(self) -> None:
        """
        Print formatted solution summary to console.

        Displays a human-readable summary of the multi-phase solution including
        key results, phase information, and transition analysis.

        Example:
            >>> solution.print_solution_summary()
            Multi-Phase Solution Summary
            ============================
            Status: Successful
            Objective: 1.234567e+02
            Phases: 3
            Global Parameters: {'gravity': 9.81, 'mass': 1000.0}
            ...
        """
        summary = self.get_solution_summary()

        print("\nMulti-Phase Solution Summary")
        print("=" * 50)
        print(f"Status: {'Successful' if summary['success'] else 'Failed'}")

        if not summary["success"]:
            print(f"Error: {summary['message']}")
        else:
            print(
                f"Objective: {summary['objective']:.6e}"
                if summary["objective"] is not None
                else "Objective: N/A"
            )

        print(f"Phases: {summary['phase_count']}")

        if summary["global_parameters"]:
            print(f"Global Parameters: {summary['global_parameters']}")

        if summary["solve_time"] is not None:
            print(f"Solve Time: {summary['solve_time']:.3f} seconds")

        if summary["nlp_iterations"] is not None:
            print(f"NLP Iterations: {summary['nlp_iterations']}")

        # Phase information
        print("\nPhase Details:")
        print("-" * 30)
        for phase_info in summary["phases"]:
            status = "✓" if phase_info["success"] else "✗"
            t0 = phase_info.get("initial_time", "N/A")
            tf = phase_info.get("terminal_time", "N/A")
            print(
                f"Phase {phase_info['index']}: {status} [{t0:.3f}, {tf:.3f}]"
                if isinstance(t0, (int, float)) and isinstance(tf, (int, float))
                else f"Phase {phase_info['index']}: {status} [N/A, N/A]"
            )

        # Transition analysis
        if summary["transition_analysis"] and summary["phase_count"] > 1:
            trans = summary["transition_analysis"]
            print("\nPhase Transitions:")
            print("-" * 30)
            print(f"Max State Discontinuity: {trans.get('max_state_discontinuity', 0.0):.6e}")
            print(f"Max Time Gap: {trans.get('max_time_gap', 0.0):.6e}")
            print(f"Continuous Transitions: {len(trans.get('continuous_transitions', []))}")
            print(f"Discontinuous Transitions: {len(trans.get('discontinuous_transitions', []))}")

    # ========================================================================
    # EXPORT AND ANALYSIS METHODS
    # ========================================================================

    def export_to_dict(self) -> dict[str, Any]:
        """
        Export solution data to dictionary format.

        Creates a comprehensive dictionary representation of the solution
        suitable for serialization, analysis, or external processing.

        Returns:
            Dictionary containing complete solution data

        Example:
            >>> solution_dict = solution.export_to_dict()
            >>> import json
            >>> with open('solution.json', 'w') as f:
            ...     json.dump(solution_dict, f, indent=2, default=str)
        """
        logger.debug("Exporting solution to dictionary")

        try:
            export_data = {
                "meta": {
                    "success": self.success,
                    "message": self.message,
                    "objective": self.objective,
                    "phase_count": self.phase_count,
                    "solve_time": self.solve_time,
                    "nlp_iterations": self.nlp_iterations,
                },
                "global_parameters": self.global_parameters,
                "phases": [],
                "phase_endpoints": [],
                "transition_analysis": self.analyze_phase_transitions(),
            }

            # Export phase solutions
            for phase_idx in range(self.phase_count):
                try:
                    phase_sol = self.get_phase_solution(phase_idx)
                    phase_data = {
                        "index": phase_idx,
                        "success": phase_sol.success,
                        "message": getattr(phase_sol, "message", ""),
                        "initial_time": getattr(
                            phase_sol.solution_data, "initial_time_variable", None
                        ),
                        "terminal_time": getattr(
                            phase_sol.solution_data, "terminal_time_variable", None
                        ),
                    }

                    # Add trajectory data if available
                    if hasattr(phase_sol.solution_data, "time_states"):
                        phase_data["time_states"] = phase_sol.solution_data.time_states.tolist()
                    if hasattr(phase_sol.solution_data, "states"):
                        phase_data["states"] = (
                            [state.tolist() for state in phase_sol.solution_data.states]
                            if phase_sol.solution_data.states
                            else []
                        )
                    if hasattr(phase_sol.solution_data, "time_controls"):
                        phase_data["time_controls"] = phase_sol.solution_data.time_controls.tolist()
                    if hasattr(phase_sol.solution_data, "controls"):
                        phase_data["controls"] = (
                            [control.tolist() for control in phase_sol.solution_data.controls]
                            if phase_sol.solution_data.controls
                            else []
                        )

                    export_data["phases"].append(phase_data)

                except Exception as e:
                    logger.warning("Failed to export phase %d: %s", phase_idx, str(e))
                    export_data["phases"].append({"index": phase_idx, "error": str(e)})

            # Export phase endpoints
            for endpoint in self.phase_endpoints:
                endpoint_data = {
                    "phase_index": endpoint.phase_index,
                    "initial_state": endpoint.initial_state.tolist()
                    if endpoint.initial_state is not None
                    else None,
                    "initial_time": endpoint.initial_time,
                    "final_state": endpoint.final_state.tolist()
                    if endpoint.final_state is not None
                    else None,
                    "final_time": endpoint.final_time,
                    "integrals": endpoint.integrals.tolist()
                    if endpoint.integrals is not None
                    else None,
                }
                export_data["phase_endpoints"].append(endpoint_data)

            logger.debug("Solution export completed")
            return export_data

        except Exception as e:
            logger.error("Solution export failed: %s", str(e))
            return {
                "meta": {"success": False, "error": str(e)},
                "global_parameters": {},
                "phases": [],
                "phase_endpoints": [],
                "transition_analysis": {},
            }

    def __repr__(self) -> str:
        """String representation of multi-phase solution."""
        status = "Successful" if self.success else "Failed"
        obj_str = f", obj={self.objective:.6e}" if self.objective is not None else ""
        return f"MultiPhaseSolution({status}, {self.phase_count} phases{obj_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Multi-Phase Solution: {self.phase_count} phases",
            f"Status: {'Successful' if self.success else 'Failed'}",
        ]

        if self.success and self.objective is not None:
            lines.append(f"Objective: {self.objective:.6e}")

        if self.global_parameters:
            lines.append(f"Global Parameters: {len(self.global_parameters)}")

        return "\n".join(lines)
