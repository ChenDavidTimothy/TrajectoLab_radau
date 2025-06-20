from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from .mtor_types import PhaseID


if TYPE_CHECKING:
    from .solution import Solution


def get_benchmark_table(solution: Solution, phase_id: PhaseID | None = None) -> dict[str, list]:
    """Extract adaptive refinement metrics for research benchmarking comparison."""
    if not solution.adaptive or "iteration_history" not in solution.adaptive:
        raise ValueError(
            "No adaptive iteration history available for benchmarking. "
            "Use solve_adaptive() to generate iteration data."
        )

    history = solution.adaptive["iteration_history"]

    if not history:
        raise ValueError("Adaptive iteration history is empty")

    if phase_id is not None and not isinstance(phase_id, int):
        raise ValueError(f"phase_id must be integer or None, got {type(phase_id)}")

    # Validate phase_id exists in data
    if phase_id is not None:
        first_iteration = next(iter(history.values()))
        if phase_id not in first_iteration["phase_error_estimates"]:
            available_phases = list(first_iteration["phase_error_estimates"].keys())
            raise ValueError(f"Phase {phase_id} not found. Available phases: {available_phases}")

    mesh_iterations = []
    estimated_errors = []
    collocation_points = []
    mesh_intervals = []
    polynomial_degrees = []
    refinement_strategies = []

    for iteration in sorted(history.keys()):
        data = history[iteration]
        # Keep iteration numbering as-is (0, 1, 2, ...)
        mesh_iterations.append(iteration)

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
    solution: Solution,
    phase_id: PhaseID,
    figsize: tuple[float, float] = (12, 6),
    transform_domain: tuple[float, float] | None = None,
    show_strategy: bool = True,
) -> None:
    """Visualize adaptive mesh refinement history for research analysis."""
    if not solution.adaptive or "iteration_history" not in solution.adaptive:
        raise ValueError("No adaptive iteration history available for plotting")

    history = solution.adaptive["iteration_history"]
    if not history:
        raise ValueError("Adaptive iteration history is empty")

    first_iteration = next(iter(history.values()))
    if phase_id not in first_iteration["phase_mesh_nodes"]:
        available_phases = list(first_iteration["phase_mesh_nodes"].keys())
        raise ValueError(f"Phase {phase_id} not found. Available phases: {available_phases}")

    try:
        import matplotlib.pyplot as plt

        from maptor.radau import _compute_radau_collocation_components
    except ImportError:
        raise ImportError("matplotlib required for mesh refinement plotting")

    fig, ax = plt.subplots(figsize=figsize)

    for iteration in sorted(history.keys()):
        data = history[iteration]
        mesh_nodes = data["phase_mesh_nodes"][phase_id].copy()
        polynomial_degrees = data["phase_polynomial_degrees"][phase_id]

        # Transform domain if requested
        if transform_domain is not None:
            domain_min, domain_max = transform_domain
            mesh_nodes = domain_min + (mesh_nodes + 1) * (domain_max - domain_min) / 2

        # Use iteration number directly (0, 1, 2, ...)
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
    ax.set_ylabel("Mesh State")
    ax.set_title(f"MAPTOR Mesh Refinement History - Phase {phase_id}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, len(history) + 0.5)

    # Set proper y-tick labels
    iterations = sorted(history.keys())
    y_positions = [iter_num + 1 for iter_num in iterations]
    y_labels = []
    for iter_num in iterations:
        if iter_num == 0:
            y_labels.append("Initial")
        else:
            y_labels.append(f"Iter {iter_num}")

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
    solution: Solution,
    filename: str,
    phase_id: PhaseID | None = None,
    format: str = "latex",
) -> None:
    """Export benchmarking data for research publication comparison."""
    benchmark_data = get_benchmark_table(solution, phase_id=phase_id)

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
Mesh State & Estimated Error & Number of Collocation Points \\\\
\\hline
"""
    for i in range(len(benchmark_data["mesh_iteration"])):
        iteration = benchmark_data["mesh_iteration"][i]
        error = benchmark_data["estimated_error"][i]
        points = benchmark_data["collocation_points"][i]

        # Format iteration label for LaTeX
        if iteration == 0:
            iter_label = "Initial"
        else:
            iter_label = f"Iter {iteration}"

        # Format error for LaTeX
        if np.isnan(error):
            error_str = "N/A"
        else:
            error_str = f"{error:.3e}"

        latex_content += f"{iter_label} & {error_str} & {points} \\\\\n"
    latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
    with open(filename, "w") as f:
        f.write(latex_content)


def _export_csv_table(benchmark_data: dict, filename: str) -> None:
    """Export benchmark data as CSV for data analysis."""
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
    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for key, value in benchmark_data.items():
        if hasattr(value[0], "tolist"):  # numpy array
            json_data[key] = [v.tolist() if hasattr(v, "tolist") else v for v in value]
        else:
            json_data[key] = value
    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)
