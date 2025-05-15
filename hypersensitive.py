# hypersensitive.py
# Example problem definition for the PHS-Adaptive mesh refinement.

from typing import Any, List, Union  # For type hinting problem functions

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

# Assuming phs_adaptive.py, solver_radau.py, and radau_pseudospectral_basis.py are accessible
# either in the same directory or in the Python path.
from phs_adaptive import (
    AdaptiveParameters,
    DefaultGuessValues,
    InitialGuess,
    OptimalControlProblem,
    run_phs_adaptive_mesh_refinement,
)
from solution_processor import SolutionProcessor  # Import the new class


# --- Hypersensitive Problem Definition ---
def hypersensitive_dynamics(
    states: Union[np.ndarray, ca.MX],
    controls: Union[np.ndarray, ca.MX],
    time: Union[float, ca.MX],
    params: Any,
) -> List[Union[float, ca.MX]]:
    """
    Dynamics function for the hypersensitive problem.
    dx/dt = -x^3 + u
    """
    x = states[0] if isinstance(states, (np.ndarray, list, tuple, ca.DM)) else states
    u = controls[0] if isinstance(controls, (np.ndarray, list, tuple, ca.DM)) else controls

    # Handle CasADi MX potentially being (1,1) shape but conceptually scalar
    if isinstance(x, ca.MX) and x.shape != (1, 1) and x.is_scalar():
        x = x[0, 0]
    if isinstance(u, ca.MX) and u.shape != (1, 1) and u.is_scalar():
        u = u[0, 0]

    return [-(x**3) + u]


def hypersensitive_objective(
    initial_time_variable: Union[float, ca.MX],
    terminal_time_variable: Union[float, ca.MX],
    x0: Union[np.ndarray, ca.MX],
    xf: Union[np.ndarray, ca.MX],
    integral_decision_variables: Union[None, float, np.ndarray, ca.MX],
    params: Any,
) -> Union[float, ca.MX]:
    """
    Objective function for the hypersensitive problem.
    Minimize J = q[0] (integral of 0.5*(x^2 + u^2))
    """
    if integral_decision_variables is None:
        raise ValueError(
            "integral_decision_variables (integral value) is None in objective function."
        )
    # integral_decision_variables can be scalar or array-like depending on num_integrals
    return (
        integral_decision_variables[0]
        if isinstance(integral_decision_variables, (list, np.ndarray, ca.DM))
        and len(integral_decision_variables) > 0
        else integral_decision_variables
    )


def hypersensitive_integrand(
    states: Union[np.ndarray, ca.MX],
    controls: Union[np.ndarray, ca.MX],
    time: Union[float, ca.MX],
    integral_idx: int,
    params: Any,
) -> Union[float, ca.MX]:
    """
    Integrand for the hypersensitive problem.
    L = 0.5 * (x^2 + u^2)
    """
    x = states[0] if isinstance(states, (np.ndarray, list, tuple, ca.DM)) else states
    u = controls[0] if isinstance(controls, (np.ndarray, list, tuple, ca.DM)) else controls

    if isinstance(x, ca.MX) and x.shape != (1, 1) and x.is_scalar():
        x = x[0, 0]
    if isinstance(u, ca.MX) and u.shape != (1, 1) and u.is_scalar():
        u = u[0, 0]

    if integral_idx == 0:
        return 0.5 * (x**2 + u**2)
    return 0  # Should not happen if num_integrals is 1


def event_constraints_function(initial_time_variable, terminal_time_variable, x0, xf, q, p):
    from rpm_solver import EventConstraint

    return [
        EventConstraint(val=x0[0], equals=1.5),  # x(initial_time_variable) = 1.5
        EventConstraint(val=xf[0], equals=1.0),  # x(terminal_time_variable) = 1.0
    ]


TF_hypersensitive: float = 40.0

initial_guess = InitialGuess(
    initial_time_variable=0.0,
    terminal_time_variable=TF_hypersensitive,
    states=[
        np.array([[1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7]]),
        np.array([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]]),
        np.array([[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, 1.0]]),
    ],
    controls=[
        np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, -0.1]]),
        np.array([[-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]]),
    ],
    integrals=[1.0],
)

initial_problem_def_hypersensitive = OptimalControlProblem(
    num_states=1,
    num_controls=1,
    num_integrals=1,
    collocation_points_per_interval=[8, 8, 8],  # Initial Nk list
    global_normalized_mesh_nodes=[-1.0, -1 / 3, 1 / 3, 1.0],  # Initial mesh
    dynamics_function=hypersensitive_dynamics,
    objective_function=hypersensitive_objective,
    integral_integrand_function=hypersensitive_integrand,
    t0_bounds=[0.0, 0.0],
    tf_bounds=[TF_hypersensitive, TF_hypersensitive],
    problem_parameters={},  # No specific parameters for this simple problem
    initial_guess=initial_guess,
    default_initial_guess_values=DefaultGuessValues(state=0.0, control=0.0, integral=0.0),
    event_constraints_function=event_constraints_function,
    solver_options={
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 200,
    },
)

adaptive_params_hypersensitive = AdaptiveParameters(
    epsilon_tol=1e-3,
    M_max_iterations=30,
    N_min_poly_degree=4,
    N_max_poly_degree=8,
    ode_solver_tol=1e-7,
    num_error_sim_points=40,
)


def plot_solution_with_processor(solution_processor: SolutionProcessor):
    """
    Plots the state and control trajectories using the SolutionProcessor,
    coloring each mesh interval differently.

    Args:
        solution_processor: An instance of SolutionProcessor.
    """
    if not solution_processor.nlp_success:  # Check NLP success for plotting
        print("Plotting skipped: Last NLP solve failed or no solution data.")
        return

    if solution_processor.num_intervals == 0:
        print("Plotting skipped: No mesh intervals found in the solution.")
        return

    num_intervals = solution_processor.num_intervals
    colors = plt.cm.viridis(np.linspace(0, 1, num_intervals)) if num_intervals > 0 else ["blue"]

    all_interval_data = solution_processor.get_all_interval_data()

    # --- Plot States ---
    if solution_processor.num_states > 0:
        fig_states, axes_states = plt.subplots(
            solution_processor.num_states,
            1,
            sharex=True,
            figsize=(10, 2 + 2 * solution_processor.num_states),
        )
        if solution_processor.num_states == 1:
            axes_states = [axes_states]  # Make it iterable

        for i in range(solution_processor.num_states):
            axes_states[i].set_ylabel(f"State x{i+1}")
            axes_states[i].grid(True, which="both", linestyle="--", linewidth=0.5)

            for k, interval_data in enumerate(all_interval_data):
                if (
                    interval_data
                    and len(interval_data.states_segment) > i
                    and interval_data.states_segment[i].size > 0
                ):
                    axes_states[i].plot(
                        interval_data.time_states_segment,
                        interval_data.states_segment[i],
                        color=colors[k],
                        marker=".",
                        linestyle="-",
                    )
                    # Labeling for legend handled below to avoid duplicates

        axes_states[-1].set_xlabel("Time (s)")
        fig_states.suptitle(
            "State Trajectories by Mesh Interval (using SolutionProcessor)", fontsize=14
        )

        if (
            num_intervals > 0
            and solution_processor.num_states > 0
            and solution_processor.num_collocation_nodes_per_interval
        ):
            handles, labels = [], []
            for k in range(num_intervals):
                handles.append(plt.Line2D([0], [0], color=colors[k], lw=2))
                labels.append(
                    f"Int {k} (Nk={solution_processor.num_collocation_nodes_per_interval[k]})"
                )  # Use num_collocation_nodes_per_interval from processor
            fig_states.legend(handles, labels, loc="upper right", title="Mesh Intervals")
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    # --- Plot Controls ---
    if solution_processor.num_controls > 0:
        fig_controls, axes_controls = plt.subplots(
            solution_processor.num_controls,
            1,
            sharex=True,
            figsize=(10, 2 + 2 * solution_processor.num_controls),
        )
        if solution_processor.num_controls == 1:
            axes_controls = [axes_controls]  # Make it iterable

        for i in range(solution_processor.num_controls):
            axes_controls[i].set_ylabel(f"Control u{i+1}")
            axes_controls[i].grid(True, which="both", linestyle="--", linewidth=0.5)

            for k, interval_data in enumerate(all_interval_data):
                if (
                    interval_data
                    and len(interval_data.controls_segment) > i
                    and interval_data.controls_segment[i].size > 0
                ):
                    axes_controls[i].plot(
                        interval_data.time_controls_segment,
                        interval_data.controls_segment[i],
                        color=colors[k],
                        marker=".",
                        linestyle="-",
                    )  # Original used steps-post for some controls. Adjust if needed.

        axes_controls[-1].set_xlabel("Time (s)")
        fig_controls.suptitle(
            "Control Trajectories by Mesh Interval (using SolutionProcessor)", fontsize=14
        )
        if (
            num_intervals > 0
            and solution_processor.num_controls > 0
            and solution_processor.num_collocation_nodes_per_interval
        ):
            handles, labels = [], []
            for k in range(num_intervals):
                handles.append(plt.Line2D([0], [0], color=colors[k], lw=2))
                labels.append(
                    f"Int {k} (Nk={solution_processor.num_collocation_nodes_per_interval[k]})"
                )
            fig_controls.legend(handles, labels, loc="upper right", title="Mesh Intervals")
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    plt.show()


if __name__ == "__main__":
    print(
        "Running PHS-Adaptive Mesh Refinement for Hypersensitive Problem from hypersensitive.py..."
    )

    # Run the adaptive mesh refinement
    final_solution = run_phs_adaptive_mesh_refinement(
        initial_problem_def_hypersensitive, adaptive_params_hypersensitive
    )

    # --- Process the solution using SolutionProcessor ---
    if final_solution:  # Check if a solution was returned
        processor = SolutionProcessor(final_solution)

        # --- Print Solution Summary using the processor ---
        print("\n--- Solution Summary (from SolutionProcessor) ---")
        print(processor.summary())
        print(f"Processor representation: {processor}")

        # --- Example: Accessing specific data via processor ---
        print("\n--- Further details from SolutionProcessor ---")
        if processor.nlp_success:
            print(f"Number of states: {processor.num_states}")
            state0_traj = processor.get_state_trajectory(0)
            if state0_traj is not None:
                print(f"First 3 points of State 0: {state0_traj[:3]}")

            print(f"Number of controls: {processor.num_controls}")
            control0_traj = processor.get_control_trajectory(0)
            if control0_traj is not None:
                print(f"First 3 points of Control 0: {control0_traj[:3]}")

            print(f"Number of mesh intervals: {processor.num_intervals}")
            if processor.num_intervals > 0:
                first_interval_data = processor.get_data_for_interval(0)
                if first_interval_data:
                    print(
                        f"Data for Interval 0: t_start={first_interval_data.t_start:.2f}, t_end={first_interval_data.t_end:.2f}, Nk={first_interval_data.Nk}"
                    )
                    if (
                        first_interval_data.states_segment
                        and len(first_interval_data.states_segment[0]) > 0
                    ):
                        print(
                            f"  State 0 segment in interval 0 (first 3 pts): {first_interval_data.states_segment[0][:3]}"
                        )
                    else:
                        print("  State 0 segment in interval 0: No data points")

        # --- Plot the results using the processor ---
        if processor.nlp_success:  # Plot only if the last NLP solve was successful
            print("\nAttempting to plot solution using SolutionProcessor...")
            plot_solution_with_processor(processor)
        else:
            print(
                "\nSkipping plots because the adaptive refinement or the last NLP solve failed (checked via processor)."
            )

    else:
        print("\n--- Adaptive Refinement Failed (No solution returned) ---")
        # You could still instantiate SolutionProcessor with an empty solution if you want consistent error handling
        # processor = SolutionProcessor(None)
        # print(processor.summary()) # Would indicate no data
        print("Skipping plots and further processing.")
