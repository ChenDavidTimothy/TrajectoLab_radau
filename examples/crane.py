# --- Main script for CST2 Reactor Problem (Unscaled) ---
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


def main_cst2_reactor_unscaled():
    print("\n--- Running CST2 Reactor Problem (Unscaled) ---")
    # Constants from the problem description
    p_rho = 0.01
    c1 = 2.83374
    c2 = -0.80865
    c3 = 0.71265
    c4 = 17.2656
    c5 = 27.0756

    problem = tl.Problem("CST2_Reactor_Unscaled")

    # Time
    t_final_cst2 = 9.0
    t = problem.time(initial=0.0, final=t_final_cst2)

    # States
    x1 = problem.state("x1", initial=0.0, final=10.0)
    x2 = problem.state("x2", initial=22.0, final=14.0)
    x3 = problem.state("x3", initial=0.0, final=0.0)
    x4 = problem.state("x4", initial=0, final=2.5, lower=-2.5, upper=2.5)
    x5 = problem.state("x5", initial=-1, final=0.0, lower=-1.0, upper=1.0)
    x6 = problem.state("x6", initial=0.0, final=0.0)

    # Controls
    u1 = problem.control("u1", lower=-c1, upper=c1)
    u2 = problem.control("u2", lower=c2, upper=c3)

    # Dynamics
    problem.dynamics(
        {
            x1: x4,
            x2: x5,
            x3: x6,
            x4: u1 + c4 * x3,
            x5: u2,
            x6: -(u1 + c5 * x3 + 2.0 * x5 * x6) / x2,
        }
    )

    # Objective function
    integrand_cst2 = 0.5 * (x3**2 + x6**2 + p_rho * (u1**2 + u2**2))
    integral_var_cst2 = problem.add_integral(integrand_cst2)
    problem.minimize(integral_var_cst2)

    # Mesh configuration (same as scaled version for comparison)
    fixed_polynomial_degrees = [6, 6, 6]
    fixed_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]
    problem.set_mesh(fixed_polynomial_degrees, fixed_mesh_points)

    # Initial Guess Generation (physical values)
    num_states = 6
    num_controls = 2

    initial_states_physical = {
        "x1": (0.0, 10.0),
        "x2": (22.0, 14.0),
        "x3": (0.0, 0.0),
        "x4": (-1.0, 2.3),
        "x5": (0.0, 0.0),
        "x6": (0.0, 0.0),
    }
    # The order of states/controls in the guess list must match how TrajectoLab expects them
    # Usually, this is the order of definition if not using named dictionaries.
    # For tl.Problem, set_initial_guess uses list of lists/arrays based on definition order.

    states_physical_guess_intervals = []
    controls_physical_guess_intervals = []

    # For tl.Problem, the states and controls in the guess must be in the order they were defined.
    # We can get this order from problem._states_ordered and problem._controls_ordered if they exist and are public,
    # or maintain it manually. Assuming definition order: x1, x2, x3, x4, x5, x6 and u1, u2.

    # State initial guesses (physical)
    s_names_ordered_for_guess = ["x1", "x2", "x3", "x4", "x5", "x6"]
    c_names_ordered_for_guess = ["u1", "u2"]

    initial_controls_physical_midpoints = {
        "u1": (-c1 + c1) / 2.0,  # 0.0
        "u2": (c2 + c3) / 2.0,
    }

    for N_poly_degree in fixed_polynomial_degrees:
        tau_points_states = np.linspace(-1, 1, N_poly_degree + 1)
        current_interval_states_guess = np.zeros((num_states, N_poly_degree + 1))
        for i, s_name in enumerate(s_names_ordered_for_guess):
            s_init, s_final = initial_states_physical[s_name]
            current_interval_states_guess[i, :] = (
                s_init + (s_final - s_init) * (tau_points_states + 1) / 2
            )
        states_physical_guess_intervals.append(current_interval_states_guess)

        current_interval_controls_guess = np.zeros((num_controls, N_poly_degree))
        for i, c_name in enumerate(c_names_ordered_for_guess):
            current_interval_controls_guess[i, :] = initial_controls_physical_midpoints[c_name]
        controls_physical_guess_intervals.append(current_interval_controls_guess)

    problem.set_initial_guess(
        states=states_physical_guess_intervals,  # Physical guesses
        controls=controls_physical_guess_intervals,  # Physical guesses
        initial_time=0.0,
        terminal_time=t_final_cst2,
        integrals=0.1,  # Guess for the integral value
    )

    print("Solving CST2 Reactor (Unscaled) with fixed mesh...")
    nlp_max_iter = 2000
    print(f"NLP max iterations: {nlp_max_iter}")
    fixed_solution = tl.solve_adaptive(
        problem,
        error_tolerance=5e-7,
        nlp_options={
            "ipopt.print_level": 5,  # Increased print level
            "ipopt.sb": "yes",
            "print_time": 1,
            "ipopt.max_iter": nlp_max_iter,
            "ipopt.tol": 1e-6,
            "ipopt.constr_viol_tol": 1e-6,
        },
    )

    if fixed_solution.success:
        fixed_solution.plot()
        # For unscaled problem, fixed_solution.objective is the direct physical objective
        physical_objective_cst2 = fixed_solution.objective
        print("CST2 (Unscaled) Fixed mesh solution successful!")
        print(f"  Physical Objective (J_phys):      {physical_objective_cst2:.8f}")

        reference_objective_cst2 = 0.0375194596
        print(f"  Reference Objective (J*):         {reference_objective_cst2:.8f}")
        error_percentage = (
            abs(physical_objective_cst2 - reference_objective_cst2) / reference_objective_cst2 * 100
        )
        print(f"  Error from reference:             {error_percentage:.4f}%")

        # Plotting (optional, can be adapted from the scaled version)
        # For simplicity, we'll reuse the plotting structure if needed,
        # but the variables are directly from fixed_solution.get_trajectory(symbol)

        # Example for plotting x1 and u1:
        time_plot, x1_sol = fixed_solution.get_trajectory(x1)  # Use direct symbol x1
        _, u1_sol = fixed_solution.get_trajectory(u1)  # Use direct symbol u1

        # Fetch all trajectories for plotting
        traj_x1 = fixed_solution.get_trajectory(x1)[1]
        traj_x2 = fixed_solution.get_trajectory(x2)[1]
        traj_x3 = fixed_solution.get_trajectory(x3)[1]
        traj_x4 = fixed_solution.get_trajectory(x4)[1]
        traj_x5 = fixed_solution.get_trajectory(x5)[1]
        time_plot_x6, traj_x6 = fixed_solution.get_trajectory(
            x6
        )  # Get time from last state for consistency

        time_plot_u1, traj_u1 = fixed_solution.get_trajectory(u1)
        _, traj_u2 = fixed_solution.get_trajectory(u2)

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 2, 1)
        plt.plot(time_plot_x6, traj_x1, label="x1")
        plt.title("x1")
        plt.grid(True)
        plt.legend()
        plt.subplot(3, 2, 2)
        plt.plot(time_plot_x6, traj_x2, label="x2")
        plt.title("x2")
        plt.grid(True)
        plt.legend()
        plt.subplot(3, 2, 3)
        plt.plot(time_plot_x6, traj_x3, label="x3")
        plt.title("x3")
        plt.grid(True)
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.plot(time_plot_x6, traj_x4, label="x4")
        plt.hlines(
            [-2.5, 2.5], xmin=time_plot_x6[0], xmax=time_plot_x6[-1], colors="r", linestyles="--"
        )
        plt.title("x4")
        plt.grid(True)
        plt.legend()
        plt.subplot(3, 2, 5)
        plt.plot(time_plot_x6, traj_x5, label="x5")
        plt.hlines(
            [-1.0, 1.0], xmin=time_plot_x6[0], xmax=time_plot_x6[-1], colors="r", linestyles="--"
        )
        plt.title("x5")
        plt.grid(True)
        plt.legend()
        plt.subplot(3, 2, 6)
        plt.plot(time_plot_x6, traj_x6, label="x6")
        plt.title("x6")
        plt.grid(True)
        plt.legend()
        plt.suptitle("CST2 Reactor States (Unscaled)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time_plot_u1, traj_u1, label="u1")
        plt.hlines(
            [-c1, c1], xmin=time_plot_u1[0], xmax=time_plot_u1[-1], colors="r", linestyles="--"
        )
        plt.title("u1")
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(time_plot_u1, traj_u2, label="u2")
        plt.hlines(
            [c2, c3], xmin=time_plot_u1[0], xmax=time_plot_u1[-1], colors="r", linestyles="--"
        )
        plt.title("u2")
        plt.grid(True)
        plt.legend()
        plt.suptitle("CST2 Reactor Controls (Unscaled)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    else:
        print(f"CST2 (Unscaled) Fixed mesh solution failed: {fixed_solution.message}")


# To run this new problem, modify the main execution block:
if __name__ == "__main__":
    # main_hypersensitive()
    # main_cst2_reactor() # Run the autoscaled version
    main_cst2_reactor_unscaled()  # Run the unscaled version
