import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure as MplFigure

from .tl_types import FloatArray, PhaseID


if TYPE_CHECKING:
    from .solution import Solution


logger = logging.getLogger(__name__)


def plot_multiphase_solution(
    solution: "Solution",
    phase_id: PhaseID | None = None,
    variable_names: tuple[str, ...] = (),
    figsize: tuple[float, float] = (12.0, 8.0),
    show_phase_boundaries: bool = True,
) -> None:
    """
    Plot multiphase trajectories with interval coloring and phase boundaries.

    Args:
        solution: Solution object containing multiphase trajectory data
        phase_id: Specific phase to plot (None plots all phases)
        variable_names: Optional specific variable names to plot
        figsize: Figure size for each window
        show_phase_boundaries: Whether to show vertical lines at phase boundaries

    Examples:
        >>> plot_multiphase_solution(solution)  # Plot all phases
        >>> plot_multiphase_solution(solution, 1)  # Plot only phase 1
        >>> plot_multiphase_solution(solution, None, ("position", "velocity"))
    """
    if not solution.status["success"]:
        logger.warning("Cannot plot: Solution not successful")
        return

    if phase_id is not None:
        if phase_id not in solution.phases:
            raise ValueError(f"Phase {phase_id} not found in solution")
        _plot_single_phase(solution, phase_id, variable_names, figsize)
    else:
        if variable_names:
            _plot_multiphase_variables(solution, variable_names, figsize, show_phase_boundaries)
        else:
            _plot_multiphase_default(solution, figsize, show_phase_boundaries)


def _plot_single_phase(
    solution: "Solution",
    phase_id: PhaseID,
    variable_names: tuple[str, ...],
    figsize: tuple[float, float],
) -> None:
    phase_data = solution.phases[phase_id]
    state_names = phase_data["variables"]["state_names"]
    control_names = phase_data["variables"]["control_names"]

    if variable_names:
        _create_variable_plot(
            solution,
            f"Phase {phase_id} Variables",
            [(phase_id, var) for var in variable_names],
            figsize,
        )
    else:
        figures_created = []

        if state_names:
            fig = _create_variable_plot(
                solution,
                f"Phase {phase_id} States",
                [(phase_id, var) for var in state_names],
                figsize,
                show_immediately=False,
            )
            figures_created.append(fig)

        if control_names:
            fig = _create_variable_plot(
                solution,
                f"Phase {phase_id} Controls",
                [(phase_id, var) for var in control_names],
                figsize,
                show_immediately=False,
            )
            figures_created.append(fig)

        for fig in figures_created:
            plt.figure(fig.number)
            plt.show(block=False)

        if figures_created:
            plt.figure(figures_created[-1].number)
            plt.show()


def _plot_multiphase_variables(
    solution: "Solution",
    variable_names: tuple[str, ...],
    figsize: tuple[float, float],
    show_phase_boundaries: bool,
) -> None:
    phase_var_pairs = []
    for var_name in variable_names:
        for phase_id in solution.phases.keys():
            if (phase_id, var_name) in solution:
                phase_var_pairs.append((phase_id, var_name))

    if not phase_var_pairs:
        logger.warning("None of the requested variables found in any phase")
        return

    _create_multiphase_variable_plot(
        solution, "Multiphase Variables", phase_var_pairs, figsize, show_phase_boundaries
    )


def _plot_multiphase_default(
    solution: "Solution", figsize: tuple[float, float], show_phase_boundaries: bool
) -> None:
    figures_created = []

    all_state_vars = set()
    all_control_vars = set()

    for phase_data in solution.phases.values():
        all_state_vars.update(phase_data["variables"]["state_names"])
        all_control_vars.update(phase_data["variables"]["control_names"])

    if all_state_vars:
        state_pairs = []
        for var_name in sorted(all_state_vars):
            for phase_id in solution.phases.keys():
                if (phase_id, var_name) in solution:
                    state_pairs.append((phase_id, var_name))

        if state_pairs:
            fig = _create_multiphase_variable_plot(
                solution,
                "Multiphase States",
                state_pairs,
                figsize,
                show_phase_boundaries,
                show_immediately=False,
            )
            figures_created.append(fig)

    if all_control_vars:
        control_pairs = []
        for var_name in sorted(all_control_vars):
            for phase_id in solution.phases.keys():
                if (phase_id, var_name) in solution:
                    control_pairs.append((phase_id, var_name))

        if control_pairs:
            fig = _create_multiphase_variable_plot(
                solution,
                "Multiphase Controls",
                control_pairs,
                figsize,
                show_phase_boundaries,
                show_immediately=False,
            )
            figures_created.append(fig)

    for fig in figures_created:
        plt.figure(fig.number)
        plt.show(block=False)

    if figures_created:
        plt.figure(figures_created[-1].number)
        plt.show()


def _create_variable_plot(
    solution: "Solution",
    title: str,
    phase_var_pairs: list[tuple[PhaseID, str]],
    figsize: tuple[float, float],
    show_immediately: bool = True,
) -> MplFigure:
    if not phase_var_pairs:
        return plt.figure()

    var_groups: dict[str, list[PhaseID]] = {}
    for phase_id, var_name in phase_var_pairs:
        if var_name not in var_groups:
            var_groups[var_name] = []
        var_groups[var_name].append(phase_id)

    num_vars = len(var_groups)
    if num_vars == 0:
        return plt.figure()

    rows, cols = _determine_subplot_layout(num_vars)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False)
    fig.suptitle(title)

    if num_vars == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for i, (var_name, phase_ids) in enumerate(var_groups.items()):
        ax = axes[i]

        for phase_id in phase_ids:
            try:
                _plot_single_variable_with_intervals(solution, ax, phase_id, var_name)
            except KeyError:
                continue

        ax.set_ylabel(var_name)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for i in range(num_vars, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if show_immediately:
        plt.show()

    return fig


def _create_multiphase_variable_plot(
    solution: "Solution",
    title: str,
    phase_var_pairs: list[tuple[PhaseID, str]],
    figsize: tuple[float, float],
    show_phase_boundaries: bool,
    show_immediately: bool = True,
) -> MplFigure:
    fig = _create_variable_plot(solution, title, phase_var_pairs, figsize, show_immediately=False)

    if show_phase_boundaries and len(solution.phases) > 1:
        phase_ids = sorted(solution.phases.keys())
        for ax in fig.get_axes():
            if ax.get_visible():
                for phase_id in phase_ids[:-1]:
                    final_time = solution.phases[phase_id]["times"]["final"]
                    ax.axvline(
                        final_time,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2,
                        label="Phase Boundary" if phase_id == phase_ids[0] else "",
                    )

                handles, labels = ax.get_legend_handles_labels()
                if "Phase Boundary" in labels:
                    ax.legend()

    if show_immediately:
        plt.show()

    return fig


def _plot_single_variable_with_intervals(
    solution: "Solution", ax: MplAxes, phase_id: PhaseID, var_name: str
) -> None:
    phase_data = solution.phases[phase_id]

    if var_name in phase_data["variables"]["state_names"]:
        var_type = "state"
        time_data = solution[(phase_id, "time_states")]
        var_data = solution[(phase_id, var_name)]
    elif var_name in phase_data["variables"]["control_names"]:
        var_type = "control"
        time_data = solution[(phase_id, "time_controls")]
        var_data = solution[(phase_id, var_name)]
    else:
        return

    if len(time_data) == 0:
        return

    interval_colors = _get_phase_interval_colors(solution, phase_id)
    interval_boundaries = _get_phase_mesh_intervals(solution, phase_id)

    if interval_colors is None or len(interval_boundaries) == 0:
        if var_type == "control":
            _plot_control_step_function_simple(ax, time_data, var_data, f"Phase {phase_id}")
        else:
            _plot_state_linear_simple(ax, time_data, var_data, f"Phase {phase_id}")
    else:
        if var_type == "control":
            _plot_control_step_function_intervals(
                ax, time_data, var_data, interval_boundaries, interval_colors, phase_id
            )
        else:
            _plot_state_linear_intervals(
                ax, time_data, var_data, interval_boundaries, interval_colors, phase_id
            )


def _plot_control_step_function_intervals(
    ax: MplAxes,
    time_array: FloatArray,
    values_array: FloatArray,
    intervals: list[tuple[float, float]],
    colors: np.ndarray,
    phase_id: PhaseID,
) -> None:
    if len(time_array) == 0:
        return

    extended_times = np.copy(time_array)
    extended_values = np.copy(values_array)

    if len(intervals) > 0:
        final_time = intervals[-1][1]
        if len(time_array) > 0 and time_array[-1] < final_time - 1e-10:
            extended_times = np.append(extended_times, final_time)
            extended_values = np.append(extended_values, values_array[-1])
    elif len(time_array) > 1:
        dt = time_array[-1] - time_array[-2]
        extended_times = np.append(extended_times, time_array[-1] + dt * 0.1)
        extended_values = np.append(extended_values, values_array[-1])

    for k, (t_start, t_end) in enumerate(intervals):
        mask = (time_array >= t_start - 1e-10) & (time_array <= t_end + 1e-10)
        if not np.any(mask):
            continue

        interval_times = time_array[mask]
        interval_values = values_array[mask]

        if len(interval_times) > 0 and interval_times[-1] < t_end - 1e-10:
            interval_times = np.append(interval_times, t_end)
            interval_values = np.append(interval_values, interval_values[-1])

        color = colors[k % len(colors)]

        ax.step(
            interval_times,
            interval_values,
            where="post",
            color=color,
            linewidth=1.5,
            label=f"Phase {phase_id} Int {k + 1}" if k == 0 else "",
        )

        ax.plot(time_array[mask], values_array[mask], "o", color=color, markersize=4)


def _plot_state_linear_intervals(
    ax: MplAxes,
    time_array: FloatArray,
    values_array: FloatArray,
    intervals: list[tuple[float, float]],
    colors: np.ndarray,
    phase_id: PhaseID,
) -> None:
    if len(time_array) == 0:
        return

    for k, (t_start, t_end) in enumerate(intervals):
        mask = (time_array >= t_start - 1e-10) & (time_array <= t_end + 1e-10)
        if not np.any(mask):
            continue

        color = colors[k % len(colors)]

        ax.plot(
            time_array[mask],
            values_array[mask],
            color=color,
            marker=".",
            linestyle="-",
            linewidth=1.5,
            markersize=7,
            label=f"Phase {phase_id} Int {k + 1}" if k == 0 else "",
        )


def _plot_control_step_function_simple(
    ax: MplAxes, time_array: FloatArray, values_array: FloatArray, label: str
) -> None:
    extended_times = np.copy(time_array)
    extended_values = np.copy(values_array)

    if len(time_array) > 1:
        dt = time_array[-1] - time_array[-2]
        extended_times = np.append(extended_times, time_array[-1] + dt * 0.1)
        extended_values = np.append(extended_values, values_array[-1])

    ax.step(extended_times, extended_values, where="post", linewidth=1.5, label=label)
    ax.plot(time_array, values_array, "o", markersize=4)


def _plot_state_linear_simple(
    ax: MplAxes, time_array: FloatArray, values_array: FloatArray, label: str
) -> None:
    ax.plot(time_array, values_array, ".-", linewidth=1.5, markersize=3, label=label)


def _get_phase_interval_colors(solution: "Solution", phase_id: PhaseID) -> np.ndarray | None:
    phase_data = solution.phases[phase_id]
    num_intervals = phase_data["mesh"]["num_intervals"]

    if num_intervals <= 1:
        return None

    colormap = plt.get_cmap("viridis")
    color_values = np.linspace(0, 1, num_intervals, dtype=np.float64)
    colors = colormap(color_values)
    return colors


def _get_phase_mesh_intervals(solution: "Solution", phase_id: PhaseID) -> list[tuple[float, float]]:
    phase_data = solution.phases[phase_id]

    mesh_nodes = phase_data["mesh"]["mesh_nodes"]
    initial_time = phase_data["times"]["initial"]
    terminal_time = phase_data["times"]["final"]

    if (
        mesh_nodes is None
        or len(mesh_nodes) == 0
        or np.isnan(initial_time)
        or np.isnan(terminal_time)
    ):
        return []

    # Convert normalized mesh nodes to physical time
    alpha = (terminal_time - initial_time) / 2.0
    alpha_0 = (terminal_time + initial_time) / 2.0
    mesh_phys = alpha * mesh_nodes + alpha_0

    return [(mesh_phys[i], mesh_phys[i + 1]) for i in range(len(mesh_phys) - 1)]


def _determine_subplot_layout(num_plots: int) -> tuple[int, int]:
    if num_plots <= 1:
        return (1, 1)

    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))
    return (rows, cols)
