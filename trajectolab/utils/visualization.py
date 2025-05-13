"""
Visualization utilities for TrajectoLab solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

from .solution import SolutionProcessor

def plot_solution_with_processor(solution_processor: SolutionProcessor, title_suffix: str = ""):
    """
    Plots the state and control trajectories using the SolutionProcessor,
    coloring each mesh interval differently.

    Args:
        solution_processor: An instance of SolutionProcessor.
        title_suffix: A string to append to the plot titles.
    """
    if not solution_processor.nlp_success: # Check NLP success for plotting
        print("Plotting skipped: NLP solve failed or no solution data.")
        return

    if solution_processor.num_intervals == 0:
        print("Plotting skipped: No mesh intervals found in the solution.")
        return

    num_intervals = solution_processor.num_intervals
    colors = plt.cm.viridis(np.linspace(0, 1, num_intervals)) if num_intervals > 0 else ['blue']

    all_interval_data = solution_processor.get_all_interval_data()

    # --- Plot States ---
    if solution_processor.num_states > 0:
        fig_states, axes_states = plt.subplots(solution_processor.num_states, 1, sharex=True, figsize=(10, 2 + 2 * solution_processor.num_states))
        if solution_processor.num_states == 1:
            axes_states = [axes_states] # Make it iterable

        for i in range(solution_processor.num_states):
            axes_states[i].set_ylabel(f"State x{i+1}")
            axes_states[i].grid(True, which='both', linestyle='--', linewidth=0.5)

            for k, interval_data in enumerate(all_interval_data):
                if interval_data and len(interval_data['states_segment']) > i and interval_data['states_segment'][i].size > 0:
                    axes_states[i].plot(interval_data['time_states_segment'],
                                        interval_data['states_segment'][i],
                                        color=colors[k], marker='.', linestyle='-')
                                        # Labeling for legend handled below to avoid duplicates

        axes_states[-1].set_xlabel("Time (s)")
        fig_states.suptitle(f"State Trajectories by Mesh Interval{title_suffix}", fontsize=14)

        if num_intervals > 0 and solution_processor.num_states > 0 and solution_processor.num_collocation_nodes_per_interval:
            handles, labels = [], []
            for k in range(num_intervals):
                handles.append(plt.Line2D([0], [0], color=colors[k], lw=2))
                labels.append(f"Int {k} (Nk={solution_processor.num_collocation_nodes_per_interval[k]})") # Use num_collocation_nodes_per_interval from processor
            fig_states.legend(handles, labels, loc='upper right', title="Mesh Intervals")
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    # --- Plot Controls ---
    if solution_processor.num_controls > 0:
        fig_controls, axes_controls = plt.subplots(solution_processor.num_controls, 1, sharex=True, figsize=(10, 2 + 2 * solution_processor.num_controls))
        if solution_processor.num_controls == 1:
            axes_controls = [axes_controls] # Make it iterable

        for i in range(solution_processor.num_controls):
            axes_controls[i].set_ylabel(f"Control u{i+1}")
            axes_controls[i].grid(True, which='both', linestyle='--', linewidth=0.5)

            for k, interval_data in enumerate(all_interval_data):
                if interval_data and len(interval_data['controls_segment']) > i and interval_data['controls_segment'][i].size > 0 :
                    axes_controls[i].plot(interval_data['time_controls_segment'],
                                          interval_data['controls_segment'][i],
                                          color=colors[k], marker='.', linestyle='-')

        axes_controls[-1].set_xlabel("Time (s)")
        fig_controls.suptitle(f"Control Trajectories by Mesh Interval{title_suffix}", fontsize=14)
        if num_intervals > 0 and solution_processor.num_controls > 0 and solution_processor.num_collocation_nodes_per_interval:
            handles, labels = [], []
            for k in range(num_intervals):
                handles.append(plt.Line2D([0], [0], color=colors[k], lw=2))
                labels.append(f"Int {k} (Nk={solution_processor.num_collocation_nodes_per_interval[k]})")
            fig_controls.legend(handles, labels, loc='upper right', title="Mesh Intervals")
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    plt.show()