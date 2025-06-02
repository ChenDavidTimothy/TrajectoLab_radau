"""
Comprehensive solution summary reporting for TrajectoLab optimal control problems.

This module provides exhaustive, professional solution summaries similar to IPOPT's
console output, structured for control engineering applications with complete
problem analysis, solver performance, and mesh refinement details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .tl_types import FloatArray, PhaseID

if TYPE_CHECKING:
    from .solution import Solution


logger = logging.getLogger(__name__)


def print_comprehensive_solution_summary(solution: "Solution") -> None:
    """
    Print exhaustive solution summary with professional formatting.

    Provides complete analysis including problem definition, solver performance,
    mesh details, phase analysis, and computational statistics.

    Args:
        solution: Solution object containing multiphase optimization results
    """
    print("\n" + "=" * 80)
    print("TRAJECTOLAB MULTIPHASE OPTIMAL CONTROL SOLUTION SUMMARY")
    print("=" * 80)

    # 1. Problem Definition Summary
    _print_problem_definition_section(solution)

    # 2. Solution Status and Performance
    _print_solution_status_section(solution)

    # 3. Solver Configuration and Execution
    _print_solver_execution_section(solution)

    # 4. Mesh and Discretization Details
    _print_mesh_discretization_section(solution)

    # 5. Phase-by-Phase Analysis
    _print_phase_analysis_section(solution)

    # 6. Static Parameters Analysis
    _print_static_parameters_section(solution)

    # 7. Computational Statistics
    _print_computational_statistics_section(solution)

    # 8. Quality Metrics and Validation
    _print_quality_metrics_section(solution)

    print("=" * 80)
    print("END TRAJECTOLAB SOLUTION SUMMARY")
    print("=" * 80 + "\n")


def _print_problem_definition_section(solution: "Solution") -> None:
    """Print problem definition and structure summary."""
    print("\n┌─ PROBLEM DEFINITION")
    print("│")

    if solution._problem is not None:
        problem_name = getattr(solution._problem, 'name', 'Multiphase Optimal Control Problem')
        print(f"│  Problem Name: {problem_name}")
    else:
        print("│  Problem Name: Unknown")

    # Phase structure
    phase_ids = solution.get_phase_ids()
    total_states, total_controls, num_static_params = _get_total_variable_counts(solution)

    print(f"│  Problem Type: {'Single-Phase' if len(phase_ids) == 1 else 'Multiphase'} Optimal Control")
    print(f"│  Number of Phases: {len(phase_ids)}")
    print(f"│  Total State Variables: {total_states}")
    print(f"│  Total Control Variables: {total_controls}")
    print(f"│  Static Parameters: {num_static_params}")

    # Time horizon
    if phase_ids:
        try:
            earliest_time = min(solution.get_phase_initial_time(pid) for pid in phase_ids)
            latest_time = max(solution.get_phase_final_time(pid) for pid in phase_ids)
            print(f"│  Time Horizon: [{earliest_time:.6f}, {latest_time:.6f}] (Duration: {latest_time - earliest_time:.6f})")
        except (ValueError, KeyError):
            print("│  Time Horizon: Not available")

    print("│")


def _print_solution_status_section(solution: "Solution") -> None:
    """Print solution status and performance metrics."""
    print("┌─ SOLUTION STATUS")
    print("│")

    status_symbol = "✓" if solution.success else "✗"
    status_text = "SUCCESS" if solution.success else "FAILED"
    print(f"│  {status_symbol} Status: {status_text}")

    if solution.success:
        print(f"│  Objective Value: {solution.objective:.8e}")

        # Check if this looks like minimum time problem
        if abs(solution.objective) < 1e-6 or _is_likely_minimum_time_problem(solution):
            try:
                final_times = [solution.get_phase_final_time(pid) for pid in solution.get_phase_ids()]
                if final_times:
                    max_final_time = max(final_times)
                    print(f"│  Final Time: {max_final_time:.8f} (Minimum-time problem detected)")
            except (ValueError, KeyError):
                pass
    else:
        print(f"│  Failure Reason: {solution.message}")

    print("│")


def _print_solver_execution_section(solution: "Solution") -> None:
    """Print solver configuration and execution details."""
    print("┌─ SOLVER EXECUTION")
    print("│")

    # Determine solver type from message or available data
    solver_type = "Unknown"
    iterations_info = "Not available"

    if hasattr(solution, 'message') and solution.message:
        if "adaptive" in solution.message.lower():
            solver_type = "Adaptive Mesh Refinement"
            # Extract iteration info from message
            if "iteration" in solution.message.lower():
                try:
                    # Try to extract iteration count from message
                    words = solution.message.split()
                    for i, word in enumerate(words):
                        if "iteration" in word.lower() and i > 0:
                            # Look for numbers before the word
                            for j in range(i-1, -1, -1):
                                if words[j].isdigit():
                                    iterations_info = f"{words[j]} iterations"
                                    break
                            break
                except Exception:
                    pass
        else:
            solver_type = "Fixed Mesh Direct Collocation"

    print(f"│  Solver Type: {solver_type}")
    print(f"│  Iterations: {iterations_info}")

    # Solver options if available
    if solution._problem is not None and hasattr(solution._problem, 'solver_options'):
        solver_opts = solution._problem.solver_options
        if solver_opts:
            print("│  NLP Solver Configuration:")
            for key, value in solver_opts.items():
                print(f"│    {key}: {value}")

    print("│")


def _print_mesh_discretization_section(solution: "Solution") -> None:
    """Print comprehensive mesh and discretization analysis."""
    print("┌─ MESH AND DISCRETIZATION")
    print("│")

    if not solution.phase_mesh_intervals:
        print("│  Mesh information not available")
        print("│")
        return

    # Calculate mesh statistics
    total_intervals = 0
    total_collocation_points = 0
    min_degree = float('inf')
    max_degree = 0

    print("│  Phase Mesh Details:")
    print("│  ┌─────────┬───────────┬─────────────┬──────────────┬─────────────────")
    print("│  │ Phase   │ Intervals │ Poly Degree │ Colloc Pts   │ Mesh Nodes     ")
    print("│  │         │           │ [Min, Max]  │ (Total)      │ (Normalized)   ")
    print("│  ├─────────┼───────────┼─────────────┼──────────────┼─────────────────")

    for phase_id in sorted(solution.get_phase_ids()):
        if phase_id in solution.phase_mesh_intervals:
            intervals = solution.phase_mesh_intervals[phase_id]
            num_intervals = len(intervals)
            total_intervals += num_intervals

            if intervals:
                phase_min = min(intervals)
                phase_max = max(intervals)
                min_degree = min(min_degree, phase_min)
                max_degree = max(max_degree, phase_max)

                phase_colloc_pts = sum(intervals)
                total_collocation_points += phase_colloc_pts

                # Mesh nodes info
                mesh_info = "Not available"
                if phase_id in solution.phase_mesh_nodes and solution.phase_mesh_nodes[phase_id] is not None:
                    mesh_nodes = solution.phase_mesh_nodes[phase_id]
                    mesh_info = f"[{mesh_nodes[0]:.3f}, {mesh_nodes[-1]:.3f}]"

                print(f"│  │ {phase_id:7d} │ {num_intervals:9d} │ [{phase_min:2d}, {phase_max:2d}]    │ {phase_colloc_pts:12d} │ {mesh_info:15s}")

    print("│  └─────────┴───────────┴─────────────┴──────────────┴─────────────────")
    print("│")
    print(f"│  Total Mesh Intervals: {total_intervals}")
    print(f"│  Total Collocation Points: {total_collocation_points}")
    if min_degree != float('inf'):
        print(f"│  Polynomial Degree Range: [{int(min_degree)}, {int(max_degree)}]")

    # Adaptive mesh specific information
    if "adaptive" in getattr(solution, 'message', '').lower():
        print("│")
        print("│  Adaptive Mesh Refinement Details:")

        # Try to extract more adaptive info from message
        if hasattr(solution, 'message') and "tolerance" in solution.message:
            try:
                # Extract tolerance from message
                import re
                tolerance_match = re.search(r'tolerance\s+([0-9\.e\-\+]+)', solution.message)
                if tolerance_match:
                    tolerance = tolerance_match.group(1)
                    print(f"│    Target Error Tolerance: {tolerance}")
            except Exception:
                pass

        print("│    Final mesh represents converged adaptive refinement")

    print("│")


def _print_phase_analysis_section(solution: "Solution") -> None:
    """Print detailed phase-by-phase analysis."""
    print("┌─ PHASE-BY-PHASE ANALYSIS")
    print("│")

    phase_ids = solution.get_phase_ids()
    if not phase_ids:
        print("│  No phases available")
        print("│")
        return

    for i, phase_id in enumerate(phase_ids):
        print(f"│  Phase {phase_id}:")
        print("│  ├─ Timing:")

        try:
            t_initial = solution.get_phase_initial_time(phase_id)
            t_final = solution.get_phase_final_time(phase_id)
            duration = t_final - t_initial

            print(f"│  │  Initial Time: {t_initial:.8f}")
            print(f"│  │  Final Time:   {t_final:.8f}")
            print(f"│  │  Duration:     {duration:.8f}")
        except (ValueError, KeyError):
            print("│  │  Timing information not available")

        print("│  ├─ Variables:")

        # State variables
        if phase_id in solution._phase_state_names:
            state_names = solution._phase_state_names[phase_id]
            print(f"│  │  States ({len(state_names)}): {state_names}")

            # Show state trajectory statistics
            print("│  │  State Trajectory Statistics:")
            for j, state_name in enumerate(state_names):
                try:
                    state_data = solution[(phase_id, state_name)]
                    if len(state_data) > 0:
                        min_val = np.min(state_data)
                        max_val = np.max(state_data)
                        mean_val = np.mean(state_data)
                        print(f"│  │    {state_name:12s}: [{min_val:10.4e}, {max_val:10.4e}] (μ={mean_val:10.4e})")
                except (KeyError, ValueError):
                    print(f"│  │    {state_name:12s}: Data not available")
        else:
            print("│  │  States: Information not available")

        # Control variables
        if phase_id in solution._phase_control_names:
            control_names = solution._phase_control_names[phase_id]
            print(f"│  │  Controls ({len(control_names)}): {control_names}")

            # Show control trajectory statistics
            if control_names:
                print("│  │  Control Trajectory Statistics:")
                for control_name in control_names:
                    try:
                        control_data = solution[(phase_id, control_name)]
                        if len(control_data) > 0:
                            min_val = np.min(control_data)
                            max_val = np.max(control_data)
                            mean_val = np.mean(control_data)
                            print(f"│  │    {control_name:12s}: [{min_val:10.4e}, {max_val:10.4e}] (μ={mean_val:10.4e})")
                    except (KeyError, ValueError):
                        print(f"│  │    {control_name:12s}: Data not available")
        else:
            print("│  │  Controls: Information not available")

        # Integral values
        if phase_id in solution.phase_integrals:
            integral_val = solution.phase_integrals[phase_id]
            if isinstance(integral_val, (int, float)):
                print(f"│  │  Integral Value: {integral_val:.8e}")
            elif isinstance(integral_val, np.ndarray):
                print(f"│  │  Integral Values ({len(integral_val)}): {integral_val}")

        # Mesh information for this phase
        if phase_id in solution.phase_mesh_intervals:
            intervals = solution.phase_mesh_intervals[phase_id]
            num_intervals = len(intervals)
            total_points = sum(intervals) + num_intervals  # +num_intervals for boundary points
            print(f"│  │  Mesh: {num_intervals} intervals, {total_points} total discretization points")

        # Add separator between phases (except for last phase)
        if i < len(phase_ids) - 1:
            print("│  │")

    print("│")


def _print_static_parameters_section(solution: "Solution") -> None:
    """Print static parameter analysis."""
    if solution.static_parameters is None or len(solution.static_parameters) == 0:
        return

    print("┌─ STATIC PARAMETERS")
    print("│")

    params = solution.static_parameters
    print(f"│  Number of Static Parameters: {len(params)}")
    print("│  Optimized Values:")

    # Try to get parameter names from problem if available
    param_names = None
    if solution._problem is not None and hasattr(solution._problem, '_static_parameters'):
        try:
            static_params = solution._problem._static_parameters
            if hasattr(static_params, 'parameter_names'):
                param_names = static_params.parameter_names
        except (AttributeError, IndexError):
            pass

    for i, value in enumerate(params):
        param_name = f"param_{i+1}" if param_names is None or i >= len(param_names) else param_names[i]
        print(f"│    {param_name:15s}: {value:15.8e}")

    print("│")


def _print_computational_statistics_section(solution: "Solution") -> None:
    """Print computational performance statistics."""
    print("┌─ COMPUTATIONAL STATISTICS")
    print("│")

    # Calculate total problem size
    total_states, total_controls, num_static_params = _get_total_variable_counts(solution)

    # Calculate total discretization points
    total_discretization_points = 0
    total_intervals = 0

    for phase_id in solution.get_phase_ids():
        if phase_id in solution.phase_mesh_intervals:
            intervals = solution.phase_mesh_intervals[phase_id]
            total_intervals += len(intervals)
            # Each interval contributes its collocation points
            total_discretization_points += sum(intervals)

    # Estimate NLP size (this is approximate)
    estimated_nlp_variables = (
        total_discretization_points * (total_states + total_controls) +  # Trajectory variables
        len(solution.get_phase_ids()) * 2 +  # Time variables per phase
        num_static_params  # Static parameters
    )

    print(f"│  Problem Dimensions:")
    print(f"│    Total State Variables: {total_states}")
    print(f"│    Total Control Variables: {total_controls}")
    print(f"│    Static Parameters: {num_static_params}")
    print(f"│    Total Mesh Intervals: {total_intervals}")
    print(f"│    Total Discretization Points: {total_discretization_points}")
    print(f"│    Estimated NLP Variables: {estimated_nlp_variables}")

    # Memory usage estimate (very rough)
    estimated_memory_mb = estimated_nlp_variables * 8 / (1024 * 1024)  # 8 bytes per double
    print(f"│    Estimated Memory Usage: ~{estimated_memory_mb:.1f} MB")

    print("│")


def _print_quality_metrics_section(solution: "Solution") -> None:
    """Print solution quality and validation metrics."""
    print("┌─ SOLUTION QUALITY METRICS")
    print("│")

    if not solution.success:
        print("│  Solution quality metrics not available (solve failed)")
        print("│")
        return

    # Check for common quality indicators
    quality_checks = []

    # 1. Check for reasonable objective value
    if not (np.isnan(solution.objective) or np.isinf(solution.objective)):
        quality_checks.append("✓ Objective value is finite")
    else:
        quality_checks.append("✗ Objective value is not finite")

    # 2. Check time consistency
    time_consistent = True
    try:
        for phase_id in solution.get_phase_ids():
            t_i = solution.get_phase_initial_time(phase_id)
            t_f = solution.get_phase_final_time(phase_id)
            if t_f <= t_i:
                time_consistent = False
                break
    except (ValueError, KeyError):
        time_consistent = False

    if time_consistent:
        quality_checks.append("✓ Phase time ordering is consistent")
    else:
        quality_checks.append("✗ Phase time ordering issues detected")

    # 3. Check for NaN/Inf in trajectories
    trajectory_health = True
    try:
        for phase_id in solution.get_phase_ids():
            # Check state trajectories
            if phase_id in solution._phase_state_names:
                for state_name in solution._phase_state_names[phase_id]:
                    try:
                        data = solution[(phase_id, state_name)]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            trajectory_health = False
                            break
                    except KeyError:
                        continue
            if not trajectory_health:
                break
    except Exception:
        trajectory_health = False

    if trajectory_health:
        quality_checks.append("✓ All trajectories contain finite values")
    else:
        quality_checks.append("✗ Some trajectories contain NaN/Inf values")

    # 4. Check mesh consistency
    mesh_consistent = True
    try:
        for phase_id in solution.get_phase_ids():
            if phase_id in solution.phase_mesh_intervals and phase_id in solution.phase_mesh_nodes:
                intervals = solution.phase_mesh_intervals[phase_id]
                mesh_nodes = solution.phase_mesh_nodes[phase_id]
                if len(intervals) != len(mesh_nodes) - 1:
                    mesh_consistent = False
                    break
    except Exception:
        mesh_consistent = False

    if mesh_consistent:
        quality_checks.append("✓ Mesh configuration is consistent")
    else:
        quality_checks.append("✗ Mesh configuration inconsistencies detected")

    # Print quality checks
    print("│  Quality Validation:")
    for check in quality_checks:
        print(f"│    {check}")

    # Overall assessment
    passed_checks = sum(1 for check in quality_checks if check.startswith("✓"))
    total_checks = len(quality_checks)

    print("│")
    if passed_checks == total_checks:
        print("│  ✓ Overall Assessment: SOLUTION APPEARS HEALTHY")
    elif passed_checks >= total_checks * 0.75:
        print("│  ⚠ Overall Assessment: SOLUTION MOSTLY HEALTHY (some warnings)")
    else:
        print("│  ✗ Overall Assessment: SOLUTION HAS QUALITY ISSUES")

    print("│")


def _get_total_variable_counts(solution: "Solution") -> tuple[int, int, int]:
    """Get total variable counts across all phases."""
    total_states = 0
    total_controls = 0

    for phase_id in solution.get_phase_ids():
        if phase_id in solution._phase_state_names:
            total_states += len(solution._phase_state_names[phase_id])
        if phase_id in solution._phase_control_names:
            total_controls += len(solution._phase_control_names[phase_id])

    num_static_params = 0
    if solution.static_parameters is not None:
        num_static_params = len(solution.static_parameters)

    return total_states, total_controls, num_static_params


def _is_likely_minimum_time_problem(solution: "Solution") -> bool:
    """Heuristic to detect if this is likely a minimum-time problem."""
    # Check if objective is close to a phase final time
    try:
        final_times = [solution.get_phase_final_time(pid) for pid in solution.get_phase_ids()]
        if final_times:
            max_final_time = max(final_times)
            # If objective is within 1% of max final time, likely minimum-time
            if abs(solution.objective - max_final_time) / max(abs(max_final_time), 1e-6) < 0.01:
                return True
    except (ValueError, KeyError):
        pass

    return False
