"""
Solution data presentation for TrajectoLab optimal control problems.

This module provides factual presentation of optimization results without
analysis or interpretation. Presents raw solver output data for user evaluation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .solution import Solution


logger = logging.getLogger(__name__)


def print_comprehensive_solution_summary(solution: Solution) -> None:
    """
    Present factual solution data without analysis or interpretation.

    Args:
        solution: Solution object containing multiphase optimization results
    """
    print("\n" + "=" * 80)
    print("TRAJECTOLAB SOLUTION DATA")
    print("=" * 80)

    # 1. Problem Structure
    _print_problem_structure_section(solution)

    # 2. Solution Status
    _print_solution_status_section(solution)

    # 3. Solver Information
    _print_solver_information_section(solution)

    # 4. Phase Data
    _print_phase_data_section(solution)

    # 5. Static Parameters
    _print_static_parameters_section(solution)

    # 6. Mesh Configuration
    _print_mesh_configuration_section(solution)

    # 7. Adaptive Algorithm Data
    _print_adaptive_algorithm_data_section(solution)

    print("=" * 80)
    print("END SOLUTION DATA")
    print("=" * 80 + "\n")


def _print_problem_structure_section(solution: Solution) -> None:
    """Present basic problem structure data."""
    print("\n┌─ PROBLEM STRUCTURE")
    print("│")

    # Problem name
    if solution._problem is not None:
        problem_name = getattr(solution._problem, "name", "Optimal Control Problem")
        print(f"│  Name: {problem_name}")
    else:
        print("│  Name: Not available")

    # Basic counts
    phase_ids = solution.get_phase_ids()
    print(f"│  Phases: {len(phase_ids)}")

    # Variable counts
    total_states = 0
    total_controls = 0
    for phase_id in phase_ids:
        if phase_id in solution._phase_state_names:
            total_states += len(solution._phase_state_names[phase_id])
        if phase_id in solution._phase_control_names:
            total_controls += len(solution._phase_control_names[phase_id])

    print(f"│  Total State Variables: {total_states}")
    print(f"│  Total Control Variables: {total_controls}")

    # Static parameters
    num_static_params = 0
    if solution.static_parameters is not None:
        num_static_params = len(solution.static_parameters)
    print(f"│  Static Parameters: {num_static_params}")

    print("│")


def _print_solution_status_section(solution: Solution) -> None:
    """Present raw solution status data."""
    print("┌─ SOLUTION STATUS")
    print("│")

    print(f"│  Success: {solution.success}")
    print(f"│  Message: {solution.message}")

    if solution.success:
        print(f"│  Objective: {solution.objective:.12e}")
    else:
        print("│  Objective: Not available")

    print("│")


def _print_solver_information_section(solution: Solution) -> None:
    """Present solver configuration data."""
    print("┌─ SOLVER INFORMATION")
    print("│")

    # Solver options if available
    if solution._problem is not None and hasattr(solution._problem, "solver_options"):
        solver_opts = solution._problem.solver_options
        if solver_opts:
            print("│  NLP Solver Options:")
            for key, value in solver_opts.items():
                print(f"│    {key}: {value}")
        else:
            print("│  NLP Solver Options: Default")
    else:
        print("│  NLP Solver Options: Not available")

    print("│")


def _print_phase_data_section(solution: Solution) -> None:
    """Present phase data without analysis."""
    print("┌─ PHASE DATA")
    print("│")

    phase_ids = solution.get_phase_ids()
    if not phase_ids:
        print("│  No phases available")
        print("│")
        return

    for phase_id in phase_ids:
        print(f"│  Phase {phase_id}:")

        # Timing data
        try:
            t_initial = solution.get_phase_initial_time(phase_id)
            t_final = solution.get_phase_final_time(phase_id)
            duration = t_final - t_initial
            print(f"│    Initial Time: {t_initial:.12e}")
            print(f"│    Final Time: {t_final:.12e}")
            print(f"│    Duration: {duration:.12e}")
        except (ValueError, KeyError):
            print("│    Timing: Not available")

        # Variable names
        if phase_id in solution._phase_state_names:
            state_names = solution._phase_state_names[phase_id]
            print(f"│    State Variables ({len(state_names)}): {state_names}")
        else:
            print("│    State Variables: Not available")

        if phase_id in solution._phase_control_names:
            control_names = solution._phase_control_names[phase_id]
            print(f"│    Control Variables ({len(control_names)}): {control_names}")
        else:
            print("│    Control Variables: Not available")

        # Integral values (raw data)
        if phase_id in solution.phase_integrals:
            integral_val = solution.phase_integrals[phase_id]
            if isinstance(integral_val, int | float):
                print(f"│    Integral: {integral_val:.12e}")
            else:
                print(f"│    Integrals: {integral_val}")
        else:
            print("│    Integral: Not available")

        print("│")


def _print_static_parameters_section(solution: Solution) -> None:
    """Present static parameter data."""
    if solution.static_parameters is None or len(solution.static_parameters) == 0:
        return

    print("┌─ STATIC PARAMETERS")
    print("│")

    params = solution.static_parameters
    print(f"│  Count: {len(params)}")

    # Parameter names if available
    param_names = None
    if solution._problem is not None and hasattr(solution._problem, "_static_parameters"):
        try:
            static_params = solution._problem._static_parameters
            if hasattr(static_params, "parameter_names"):
                param_names = static_params.parameter_names
        except (AttributeError, IndexError):
            pass

    print("│  Values:")
    for i, value in enumerate(params):
        param_name = (
            f"param_{i + 1}" if param_names is None or i >= len(param_names) else param_names[i]
        )
        print(f"│    {param_name}: {value:.12e}")

    print("│")


def _print_mesh_configuration_section(solution: Solution) -> None:
    """Present mesh configuration data."""
    print("┌─ MESH CONFIGURATION")
    print("│")

    if not solution.phase_mesh_intervals:
        print("│  Mesh data not available")
        print("│")
        return

    # Mesh table
    print("│  Phase Mesh Data:")
    print("│  ┌─────────┬───────────┬─────────────┬──────────────")
    print("│  │ Phase   │ Intervals │ Poly Degrees│ Mesh Bounds  ")
    print("│  ├─────────┼───────────┼─────────────┼──────────────")

    total_intervals = 0
    for phase_id in sorted(solution.get_phase_ids()):
        if phase_id in solution.phase_mesh_intervals:
            intervals = solution.phase_mesh_intervals[phase_id]
            num_intervals = len(intervals)
            total_intervals += num_intervals

            # Polynomial degrees
            if intervals:
                degrees_str = f"{intervals}"
            else:
                degrees_str = "[]"

            # Mesh bounds
            mesh_bounds = "Not available"
            if (
                phase_id in solution.phase_mesh_nodes
                and solution.phase_mesh_nodes[phase_id] is not None
            ):
                mesh_nodes = solution.phase_mesh_nodes[phase_id]
                mesh_bounds = f"[{mesh_nodes[0]:.3f}, {mesh_nodes[-1]:.3f}]"

            print(
                f"│  │ {phase_id:7d} │ {num_intervals:9d} │ {degrees_str:11s} │ {mesh_bounds:12s}"
            )

    print("│  └─────────┴───────────┴─────────────┴──────────────")
    print(f"│  Total Intervals: {total_intervals}")
    print("│")


def _print_adaptive_algorithm_data_section(solution: Solution) -> None:
    """Present adaptive algorithm data if available."""
    # Check if adaptive data is stored in solution
    if not hasattr(solution, "adaptive_data") or solution.adaptive_data is None:
        return

    adaptive_data = solution.adaptive_data

    print("┌─ ADAPTIVE ALGORITHM DATA")
    print("│")

    # Basic adaptive information
    print(f"│  Target Tolerance: {adaptive_data.target_tolerance:.3e}")
    print(f"│  Total Iterations: {adaptive_data.total_iterations}")
    print(f"│  Converged: {adaptive_data.converged}")

    # Phase convergence status
    print("│  Phase Convergence:")
    for phase_id in sorted(adaptive_data.phase_converged.keys()):
        converged = adaptive_data.phase_converged[phase_id]
        print(f"│    Phase {phase_id}: {converged}")

    # Final error estimates per phase
    print("│  Final Error Estimates:")
    for phase_id in sorted(adaptive_data.final_phase_error_estimates.keys()):
        errors = adaptive_data.final_phase_error_estimates[phase_id]
        print(f"│    Phase {phase_id}:")
        for k, error in enumerate(errors):
            if np.isnan(error) or np.isinf(error):
                error_str = f"{error}"
            else:
                error_str = f"{error:.3e}"
            print(f"│      Interval {k}: {error_str}")

    # Maximum error per phase
    print("│  Maximum Error Per Phase:")
    for phase_id in sorted(adaptive_data.final_phase_error_estimates.keys()):
        errors = adaptive_data.final_phase_error_estimates[phase_id]
        finite_errors = [e for e in errors if not (np.isnan(e) or np.isinf(e))]
        if finite_errors:
            max_error = max(finite_errors)
            print(f"│    Phase {phase_id}: {max_error:.3e}")
        else:
            print(f"│    Phase {phase_id}: No finite errors")

    # Global maximum error
    all_finite_errors = []
    for errors in adaptive_data.final_phase_error_estimates.values():
        all_finite_errors.extend([e for e in errors if not (np.isnan(e) or np.isinf(e))])

    if all_finite_errors:
        global_max = max(all_finite_errors)
        print(f"│  Global Maximum Error: {global_max:.3e}")
    else:
        print("│  Global Maximum Error: No finite errors")

    print("│")
