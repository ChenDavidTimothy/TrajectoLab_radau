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
    phases = solution.phases
    print(f"│  Phases: {len(phases)}")

    # Variable counts
    total_states = 0
    total_controls = 0
    for phase_data in phases.values():
        total_states += phase_data["variables"]["num_states"]
        total_controls += phase_data["variables"]["num_controls"]

    print(f"│  Total State Variables: {total_states}")
    print(f"│  Total Control Variables: {total_controls}")

    # Static parameters
    parameters = solution.parameters
    num_static_params = parameters["count"] if parameters else 0
    print(f"│  Static Parameters: {num_static_params}")

    print("│")


def _print_solution_status_section(solution: Solution) -> None:
    """Present raw solution status data."""
    print("┌─ SOLUTION STATUS")
    print("│")

    status = solution.status
    print(f"│  Success: {status['success']}")
    print(f"│  Message: {status['message']}")

    if status["success"]:
        print(f"│  Objective: {status['objective']:.12e}")
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

    phases = solution.phases
    if not phases:
        print("│  No phases available")
        print("│")
        return

    for phase_id, phase_data in phases.items():
        print(f"│  Phase {phase_id}:")

        # Timing data
        times = phase_data["times"]
        t_initial = times["initial"]
        t_final = times["final"]
        duration = times["duration"]

        if not np.isnan(t_initial):
            print(f"│    Initial Time: {t_initial:.12e}")
            print(f"│    Final Time: {t_final:.12e}")
            print(f"│    Duration: {duration:.12e}")
        else:
            print("│    Timing: Not available")

        # Variable names
        variables = phase_data["variables"]
        state_names = variables["state_names"]
        control_names = variables["control_names"]

        print(f"│    State Variables ({len(state_names)}): {state_names}")
        print(f"│    Control Variables ({len(control_names)}): {control_names}")

        # Integral values (raw data)
        integrals = phase_data["integrals"]
        if integrals is not None:
            if isinstance(integrals, int | float):
                print(f"│    Integral: {integrals:.12e}")
            else:
                print(f"│    Integrals: {integrals}")
        else:
            print("│    Integral: Not available")

        print("│")


def _print_static_parameters_section(solution: Solution) -> None:
    """Present static parameter data."""
    parameters = solution.parameters
    if parameters is None or parameters["count"] == 0:
        return

    print("┌─ STATIC PARAMETERS")
    print("│")

    print(f"│  Count: {parameters['count']}")

    param_names = parameters["names"]
    param_values = parameters["values"]

    print("│  Values:")
    for i, value in enumerate(param_values):
        param_name = (
            f"param_{i + 1}" if param_names is None or i >= len(param_names) else param_names[i]
        )
        print(f"│    {param_name}: {value:.12e}")

    print("│")


def _print_mesh_configuration_section(solution: Solution) -> None:
    """Present mesh configuration data."""
    print("┌─ MESH CONFIGURATION")
    print("│")

    phases = solution.phases
    if not phases:
        print("│  Mesh data not available")
        print("│")
        return

    # Mesh table
    print("│  Phase Mesh Data:")
    print("│  ┌─────────┬───────────┬─────────────┬──────────────")
    print("│  │ Phase   │ Intervals │ Poly Degrees│ Mesh Bounds  ")
    print("│  ├─────────┼───────────┼─────────────┼──────────────")

    total_intervals = 0
    for phase_id in sorted(phases.keys()):
        phase_data = phases[phase_id]
        mesh_data = phase_data["mesh"]

        intervals = mesh_data["polynomial_degrees"]
        num_intervals = mesh_data["num_intervals"]
        total_intervals += num_intervals

        # Polynomial degrees
        degrees_str = f"{intervals}" if intervals else "[]"

        # Mesh bounds
        mesh_nodes = mesh_data["mesh_nodes"]
        if mesh_nodes is not None and len(mesh_nodes) > 0:
            mesh_bounds = f"[{mesh_nodes[0]:.3f}, {mesh_nodes[-1]:.3f}]"
        else:
            mesh_bounds = "Not available"

        print(f"│  │ {phase_id:7d} │ {num_intervals:9d} │ {degrees_str:11s} │ {mesh_bounds:12s}")

    print("│  └─────────┴───────────┴─────────────┴──────────────")
    print(f"│  Total Intervals: {total_intervals}")
    print("│")


def _print_adaptive_algorithm_data_section(solution: Solution) -> None:
    """Present adaptive algorithm data if available."""
    adaptive = solution.adaptive
    if adaptive is None:
        return

    print("┌─ ADAPTIVE ALGORITHM DATA")
    print("│")

    # Basic adaptive information
    print(f"│  Target Tolerance: {adaptive['target_tolerance']:.3e}")
    print(f"│  Total Iterations: {adaptive['iterations']}")
    print(f"│  Converged: {adaptive['converged']}")

    # Phase convergence status
    print("│  Phase Convergence:")
    for phase_id in sorted(adaptive["phase_converged"].keys()):
        converged = adaptive["phase_converged"][phase_id]
        print(f"│    Phase {phase_id}: {converged}")

    # Final error estimates per phase
    print("│  Final Error Estimates:")
    for phase_id in sorted(adaptive["final_errors"].keys()):
        errors = adaptive["final_errors"][phase_id]
        print(f"│    Phase {phase_id}:")
        for k, error in enumerate(errors):
            if np.isnan(error) or np.isinf(error):
                error_str = f"{error}"
            else:
                error_str = f"{error:.3e}"
            print(f"│      Interval {k}: {error_str}")

    # Maximum error per phase
    print("│  Maximum Error Per Phase:")
    for phase_id in sorted(adaptive["final_errors"].keys()):
        errors = adaptive["final_errors"][phase_id]
        finite_errors = [e for e in errors if not (np.isnan(e) or np.isinf(e))]
        if finite_errors:
            max_error = max(finite_errors)
            print(f"│    Phase {phase_id}: {max_error:.3e}")
        else:
            print(f"│    Phase {phase_id}: No finite errors")

    # Global maximum error
    all_finite_errors = []
    for errors in adaptive["final_errors"].values():
        all_finite_errors.extend([e for e in errors if not (np.isnan(e) or np.isinf(e))])

    if all_finite_errors:
        global_max = max(all_finite_errors)
        print(f"│  Global Maximum Error: {global_max:.3e}")
    else:
        print("│  Global Maximum Error: No finite errors")

    print("│")
