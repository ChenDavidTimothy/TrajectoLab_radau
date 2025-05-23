# --- Main script for HIV Immunology Model ---
import numpy as np

import trajectolab as tl


def main_hiv_immunology():
    print("\n--- Running HIV Immunology Model ---")
    # Constants from Table 10.17
    s1 = 2.0
    s2 = 1.5
    mu = 0.002
    k_const = 2.5e-4  # Renamed from k
    c_const = 0.007  # Renamed from c
    g = 30.0
    b1 = 14.0
    b2 = 1.0
    A1 = 2.5e5
    A2 = 75.0

    problem = tl.Problem("HIV_Immunology_Model")

    # Time
    t_final_hiv = 50.0
    t = problem.time(initial=0.0, final=t_final_hiv)

    # States (Differential Variables)
    T = problem.state("T", initial=400.0, boundary=(0, 1200))  #
    V = problem.state("V", initial=3.0, boundary=(0.05, 5))  #

    # Controls (Algebraic Variables)
    u1 = problem.control("u1", boundary=(0, 0.02))  #
    u2 = problem.control("u2", boundary=(0, 0.9))  #

    # System Dynamics (Differential-algebraic equations)
    T_dot = s1 - (s2 * V) / (b1 + V) - mu * T - k_const * V * T + u1 * T  #
    V_dot = (g * (1 - u2) * V) / (b2 + V) - c_const * V * T  #

    problem.dynamics(
        {
            T: T_dot,
            V: V_dot,
        }
    )

    # Objective Function
    integrand_hiv = T - (A1 * u1**2 + A2 * u2**2)  #
    integral_var_hiv = problem.add_integral(integrand_hiv)
    problem.minimize(-integral_var_hiv)  #

    # Mesh configuration
    fixed_polynomial_degrees = [10, 10, 10]
    fixed_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]
    problem.set_mesh(fixed_polynomial_degrees, fixed_mesh_points)

    # Initial Guess Generation
    num_states = 2
    num_controls = 2

    initial_states_physical = {
        "T": (400.0, 600.0),  # (initial_T, guessed_final_T near mid-bound)
        "V": (3.0, 0.05),  # (initial_V, guessed_final_V near lower bound)
    }
    s_names_ordered_for_guess = ["T", "V"]

    initial_controls_physical_midpoints = {
        "u1": (0.0 + 0.02) / 2.0,
        "u2": (0.0 + 0.9) / 2.0,
    }
    c_names_ordered_for_guess = ["u1", "u2"]

    states_physical_guess_intervals = []
    controls_physical_guess_intervals = []

    for N_poly_degree in fixed_polynomial_degrees:
        tau_points_states = np.linspace(-1, 1, N_poly_degree + 1)
        current_interval_states_guess = np.zeros((num_states, N_poly_degree + 1))
        for i, s_name in enumerate(s_names_ordered_for_guess):
            s_init, s_final_guess = initial_states_physical[s_name]
            current_interval_states_guess[i, :] = (
                s_init + (s_final_guess - s_init) * (tau_points_states + 1) / 2
            )
        states_physical_guess_intervals.append(current_interval_states_guess)

        num_control_points_in_interval = N_poly_degree
        current_interval_controls_guess = np.zeros((num_controls, num_control_points_in_interval))
        for i, c_name in enumerate(c_names_ordered_for_guess):
            current_interval_controls_guess[i, :] = initial_controls_physical_midpoints[c_name]
        controls_physical_guess_intervals.append(current_interval_controls_guess)

    integral_guess_hiv = 25000.0

    problem.set_initial_guess(
        states=states_physical_guess_intervals,
        controls=controls_physical_guess_intervals,
        initial_time=0.0,  #
        terminal_time=t_final_hiv,  #
        integrals=integral_guess_hiv,
    )

    print("Solving HIV Immunology Model with fixed mesh...")
    nlp_max_iter = 3000
    print(f"NLP max iterations: {nlp_max_iter}")

    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 5,
            "ipopt.sb": "yes",
            "print_time": 1,
            "ipopt.max_iter": nlp_max_iter,
            "ipopt.tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-4,
        },
    )

    if solution.success:
        physical_objective_hiv = -solution.objective
        print("HIV Immunology Model Fixed mesh solution successful!")
        print(f"  Physical Objective (J_phys):    {physical_objective_hiv:.4f}")

        reference_objective_hiv = 29514.4477  #
        print(f"  Reference Objective (J*):       {reference_objective_hiv:.4f}")
        error_percentage = (
            abs(physical_objective_hiv - reference_objective_hiv) / reference_objective_hiv * 100
        )
        print(f"  Error from reference:           {error_percentage:.4f}%")

        solution.plot()

    else:
        print(f"HIV Immunology Model Fixed mesh solution failed: {solution.message}")


if __name__ == "__main__":
    main_hiv_immunology()
