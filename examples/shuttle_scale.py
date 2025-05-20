import logging
import time

import casadi as ca
import numpy as np

import trajectolab as tl


# Configure logging to see scaling information
logging.basicConfig(level=logging.INFO)


def create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0):
    """
    Create the Space Shuttle reentry optimal control problem.

    Args:
        heating_constraint: Upper bound on aerodynamic heating (BTU/ft²/sec) or None for no constraint
        bank_angle_min: Minimum bank angle in degrees (default: -90°)

    Returns:
        A tuple containing the TrajectoLab Problem instance and a dictionary of symbolic variables.
    """
    problem = tl.Problem("Space Shuttle Reentry Trajectory")

    # Define constants
    a0 = -0.20704
    a1 = 0.029244
    b0 = 0.07854
    b1 = -0.61592e-2
    b2 = 0.621408e-3
    c0 = 1.0672181
    c1 = -0.19213774e-1
    c2 = 0.21286289e-3
    c3 = -0.10117249e-5
    mu = 0.14076539e17
    Re = 20902900
    S = 2690
    rho0 = 0.002378
    hr = 23800
    g0 = 32.174
    weight = 203000
    mass = weight / g0
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    # Original script uses manual scaling factors
    h_scale = 1e5
    v_scale = 1e4

    t = problem.time(initial=0.0, free_final=True)

    # Note: Original problem uses manual scaling (h_scale, v_scale)
    # We'll keep the same state definitions for consistency
    h_scaled = problem.state(
        "h_scaled",
        initial=260000.0 / h_scale,
        lower=0.0,
        final=80000.0 / h_scale,
    )
    phi = problem.state("phi", initial=0.0 * deg2rad)
    theta = problem.state(
        "theta",
        initial=0.0 * deg2rad,
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
    )
    v_scaled = problem.state(
        "v_scaled",
        initial=25600.0 / v_scale,
        lower=1.0 / v_scale,
        final=2500.0 / v_scale,
    )
    gamma = problem.state(
        "gamma",
        initial=-1.0 * deg2rad,
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
        final=-5.0 * deg2rad,
    )
    psi = problem.state("psi", initial=90.0 * deg2rad)

    alpha = problem.control("alpha", lower=-90.0 * deg2rad, upper=90.0 * deg2rad)
    beta = problem.control("beta", lower=bank_angle_min * deg2rad, upper=1.0 * deg2rad)

    symbolic_vars = {
        "t": t,
        "h_scaled": h_scaled,
        "phi": phi,
        "theta": theta,
        "v_scaled": v_scaled,
        "gamma": gamma,
        "psi": psi,
        "alpha": alpha,
        "beta": beta,
    }

    # Un-scale the variables for dynamics
    h_actual = h_scaled * h_scale
    v_actual = v_scaled * v_scale

    # Define problem dynamics, same as original
    eps = 1e-10
    r = Re + h_actual
    rho = rho0 * ca.exp(-h_actual / hr)
    g = mu / (r * r)
    alpha_deg_calc = alpha * rad2deg
    CL = a0 + a1 * alpha_deg_calc
    CD = b0 + b1 * alpha_deg_calc + b2 * alpha_deg_calc * alpha_deg_calc
    q_dyn = 0.5 * rho * v_actual * v_actual
    L = q_dyn * CL * S
    D = q_dyn * CD * S
    qr = 17700 * ca.sqrt(rho) * (0.0001 * v_actual) ** 3.07
    qa = c0 + c1 * alpha_deg_calc + c2 * alpha_deg_calc**2 + c3 * alpha_deg_calc**3
    q_heat = qa * qr

    problem.dynamics(
        {
            h_scaled: (v_actual * ca.sin(gamma)) / h_scale,
            phi: (v_actual / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
            theta: (v_actual / r) * ca.cos(gamma) * ca.cos(psi),
            v_scaled: (-(D / mass) - g * ca.sin(gamma)) / v_scale,
            gamma: (L / (mass * v_actual + eps)) * ca.cos(beta)
            + ca.cos(gamma) * ((v_actual / r) - (g / (v_actual + eps))),
            psi: (1 / (mass * v_actual * ca.cos(gamma) + eps)) * L * ca.sin(beta)
            + (v_actual / (r * (ca.cos(theta) + eps)))
            * ca.cos(gamma)
            * ca.sin(psi)
            * ca.sin(theta),
        }
    )

    if heating_constraint is not None:
        problem.subject_to(q_heat <= heating_constraint)

    problem.minimize(-theta)

    return problem, symbolic_vars


def prepare_initial_guess(problem, polynomial_degrees, deg2rad, initial_terminal_time=2000.0):
    """Create a consistent initial guess for the problem."""
    states_guess = []
    controls_guess = []
    h_scale = 1e5
    v_scale = 1e4
    h0, v0 = 260000.0 / h_scale, 25600.0 / v_scale
    phi0, theta0 = 0.0, 0.0
    gamma0, psi0 = -1.0 * deg2rad, 90.0 * deg2rad
    hf, vf = 80000.0 / h_scale, 2500.0 / v_scale
    gammaF = -5.0 * deg2rad

    for N in polynomial_degrees:  # N is the polynomial degree for an interval
        t_param = np.linspace(0, 1, N + 1)  # Legendre-Gauss-Lobatto points for state eval
        h_vals = h0 + (hf - h0) * t_param
        phi_vals = phi0 * np.ones(N + 1)
        theta_vals = theta0 * np.ones(N + 1)
        v_vals = v0 + (vf - v0) * t_param
        gamma_vals = gamma0 + (gammaF - gamma0) * t_param
        psi_vals = psi0 * np.ones(N + 1)
        state_array = np.vstack([h_vals, phi_vals, theta_vals, v_vals, gamma_vals, psi_vals])
        states_guess.append(state_array)

        # For controls, N points are needed (collocation points within interval)
        alpha_vals = np.zeros(N)  # Assuming N collocation points for control
        beta_vals = -45.0 * deg2rad * np.ones(N)
        control_array = np.vstack([alpha_vals, beta_vals])
        controls_guess.append(control_array)

    problem.set_initial_guess(
        states=states_guess, controls=controls_guess, terminal_time=initial_terminal_time
    )


def solve_with_scaling(enable_scaling=True):
    """Solve the shuttle problem with or without scaling."""
    # Setup
    deg2rad = np.pi / 180.0
    heating_limit = 70.0  # BTU/ft²/sec
    bank_min = -90.0  # degrees

    # Common mesh settings
    num_intervals = 15
    polynomial_degrees = [20] * num_intervals
    mesh_points = np.linspace(-1.0, 1.0, num_intervals + 1)

    # Literature values for comparison
    lit_J = 5.9587608e-1  # radians
    lit_tf = 2.0085881e3  # seconds

    # Create problem
    problem, symbolic_vars = create_shuttle_reentry_problem(
        heating_constraint=heating_limit, bank_angle_min=bank_min
    )

    # Set mesh
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Prepare initial guess
    prepare_initial_guess(problem, polynomial_degrees, deg2rad, initial_terminal_time=lit_tf)

    # NLP solver options - same as original
    solver_options = {
        "ipopt.max_iter": 2000,
        "ipopt.mumps_pivtol": 5e-7,
        "ipopt.mumps_mem_percent": 50000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-7,
        "ipopt.print_level": 5,  # Show more info for timing comparison
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.tol": 1e-8,
    }

    # Solve with or without scaling
    print(f"\nSolving with scaling = {enable_scaling}")
    start_time = time.time()

    solution = tl.solve_fixed_mesh(
        problem,
        polynomial_degrees=polynomial_degrees,
        mesh_points=mesh_points,
        nlp_options=solver_options,
        enable_scaling=enable_scaling,
    )

    solve_time = time.time() - start_time

    # Analyze results
    if solution.success:
        final_time = solution.final_time
        final_theta_rad = -solution.objective  # Objective is -theta
        final_theta_deg = final_theta_rad * 180.0 / np.pi

        print("\nSolution Results:")
        print(f"  Success: {solution.success}")
        print(f"  Solve time: {solve_time:.2f} seconds")
        print(f"  Final latitude: {final_theta_deg:.4f}°")
        print(f"  Objective (J*): {final_theta_rad:.7e} radians")
        print(f"  Final time (t_F*): {final_time:.7e} seconds")

        # Compare with literature values
        J_abs_lit = abs(lit_J)
        tf_abs_lit = abs(lit_tf)
        J_diff = abs(final_theta_rad - lit_J) / J_abs_lit * 100
        tf_diff = abs(final_time - lit_tf) / tf_abs_lit * 100

        print("\nComparison with literature values:")
        print(f"  Literature J*: {lit_J:.7e}")
        print(f"  Literature t_F*: {lit_tf:.7e}")
        print(f"  J* difference: {J_diff:.4f}%")
        print(f"  t_F* difference: {tf_diff:.4f}%")

        return solution, solve_time
    else:
        print(f"\nSolution Failed: {solution.message}")
        return solution, solve_time


def main():
    """Main function to run scaling comparison."""
    print("=== SHUTTLE REENTRY PROBLEM - SCALING COMPARISON ===")

    # Solve without scaling
    non_scaled_solution, non_scaled_time = solve_with_scaling(enable_scaling=False)

    # Solve with scaling
    scaled_solution, scaled_time = solve_with_scaling(enable_scaling=True)

    # Compare results if both succeeded
    if non_scaled_solution.success and scaled_solution.success:
        print("\n=== SCALING COMPARISON SUMMARY ===")
        print(f"Without scaling: {non_scaled_time:.2f} seconds")
        print(f"With scaling: {scaled_time:.2f} seconds")
        print(f"Speedup: {non_scaled_time / scaled_time:.2f}x")

        # Compare objective values
        non_scaled_obj = -non_scaled_solution.objective
        scaled_obj = -scaled_solution.objective
        obj_diff = abs(non_scaled_obj - scaled_obj) / abs(non_scaled_obj) * 100

        print(f"Objective difference: {obj_diff:.6f}%")

        # Compare solution quality (terminal time)
        non_scaled_tf = non_scaled_solution.final_time
        scaled_tf = scaled_solution.final_time
        tf_diff = abs(non_scaled_tf - scaled_tf) / abs(non_scaled_tf) * 100

        print(f"Terminal time difference: {tf_diff:.6f}%")


if __name__ == "__main__":
    main()
