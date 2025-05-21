import sys
import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


def create_unscaled_shuttle_problem(heating_constraint=None, bank_angle_min=-90.0):
    """
    Create the Space Shuttle reentry problem WITHOUT manual scaling.
    Uses actual units directly for altitude and velocity.
    """
    problem = tl.Problem("Space Shuttle Reentry Trajectory", use_scaling=True)

    # Define constants (same as original)
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

    # No manual scaling
    t = problem.time(initial=0.0, free_final=True)

    # State variables with ACTUAL units (not scaled)
    h = problem.state(
        "h",
        initial=260000.0,  # Actual altitude in feet
        lower=0.0,
        final=80000.0,  # Actual altitude in feet
    )
    phi = problem.state("phi", initial=0.0 * deg2rad)
    theta = problem.state(
        "theta",
        initial=0.0 * deg2rad,
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
    )
    v = problem.state(
        "v",
        initial=25600.0,  # Actual velocity in ft/s
        lower=1.0,
        final=2500.0,  # Actual velocity in ft/s
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
        "h": h,
        "phi": phi,
        "theta": theta,
        "v": v,
        "gamma": gamma,
        "psi": psi,
        "alpha": alpha,
        "beta": beta,
    }

    # Using actual units in dynamics
    eps = 1e-10
    r = Re + h
    rho = rho0 * ca.exp(-h / hr)
    g = mu / (r * r)
    alpha_deg_calc = alpha * rad2deg
    CL = a0 + a1 * alpha_deg_calc
    CD = b0 + b1 * alpha_deg_calc + b2 * alpha_deg_calc * alpha_deg_calc
    q_dyn = 0.5 * rho * v * v
    L = q_dyn * CL * S
    D = q_dyn * CD * S
    qr = 17700 * ca.sqrt(rho) * (0.0001 * v) ** 3.07
    qa = c0 + c1 * alpha_deg_calc + c2 * alpha_deg_calc**2 + c3 * alpha_deg_calc**3
    q_heat = qa * qr

    problem.dynamics(
        {
            h: v * ca.sin(gamma),
            phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
            theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
            v: -(D / mass) - g * ca.sin(gamma),
            gamma: (L / (mass * v + eps)) * ca.cos(beta)
            + ca.cos(gamma) * ((v / r) - (g / (v + eps))),
            psi: (1 / (mass * v * ca.cos(gamma) + eps)) * L * ca.sin(beta)
            + (v / (r * (ca.cos(theta) + eps))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
        }
    )

    if heating_constraint is not None:
        problem.subject_to(q_heat <= heating_constraint)

    problem.minimize(-theta)

    return problem, symbolic_vars


def prepare_unscaled_initial_guess(
    problem, polynomial_degrees, deg2rad, initial_terminal_time=2000.0
):
    """Create initial guess without manual scaling factors."""
    states_guess = []
    controls_guess = []
    h0, v0 = 260000.0, 25600.0  # Actual values
    phi0, theta0 = 0.0, 0.0
    gamma0, psi0 = -1.0 * deg2rad, 90.0 * deg2rad
    hf, vf = 80000.0, 2500.0  # Actual values

    for N in polynomial_degrees:
        t_param = np.linspace(0, 1, N + 1)
        h_vals = h0 + (hf - h0) * t_param
        phi_vals = phi0 * np.ones(N + 1)
        theta_vals = theta0 * np.ones(N + 1)
        v_vals = v0 + (vf - v0) * t_param
        gamma_vals = gamma0 + (-5.0 * deg2rad - gamma0) * t_param
        psi_vals = psi0 * np.ones(N + 1)
        state_array = np.vstack([h_vals, phi_vals, theta_vals, v_vals, gamma_vals, psi_vals])
        states_guess.append(state_array)

        # Control guess
        alpha_vals = np.zeros(N)
        beta_vals = -45.0 * deg2rad * np.ones(N)
        control_array = np.vstack([alpha_vals, beta_vals])
        controls_guess.append(control_array)

    problem.set_initial_guess(
        states=states_guess, controls=controls_guess, terminal_time=initial_terminal_time
    )


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
    h_scale = 1e5
    v_scale = 1e4

    t = problem.time(initial=0.0, free_final=True)

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

    h_actual = h_scaled * h_scale
    v_actual = v_scaled * v_scale
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


def run_fair_comparison():
    """Run a fair comparison with and without automatic scaling."""
    # First, print the environment and setup
    print("\n==== SCALING DIAGNOSTICS ENVIRONMENT ====")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"CasADi version: {ca.__version__}")
    print(f"TrajectoLab version: {tl.__version__}")
    print("\n==== TEST CONFIGURATION ====")
    print("Will run 4 tests:")
    print("1. Fixed mesh with manual scaling (original)")
    print("2. Fixed mesh with automatic scaling")
    print("3. Adaptive mesh with manual scaling (original)")
    print("4. Adaptive mesh with automatic scaling")
    print("\nEach test will be analyzed for scaling factor calculation and application")
    """Run a fair comparison with and without automatic scaling."""
    results = {}

    # Example configuration from shuttle.py (example 10.137)
    bank_min = -90
    heating_limit = None
    deg2rad = np.pi / 180.0

    # Common solver options
    fixed_mesh_options = {
        "ipopt.max_iter": 2000,
        "ipopt.mumps_pivtol": 5e-7,
        "ipopt.mumps_mem_percent": 50000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-7,
        "ipopt.print_level": 0,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.tol": 1e-8,
    }

    print("\n=== FIXED MESH WITH MANUAL SCALING (ORIGINAL) ===")
    # Import original function with manual scaling

    # 1. ORIGINAL PROBLEM WITH MANUAL SCALING (use_scaling=False)
    problem1, sym_vars1 = create_shuttle_reentry_problem(
        heating_constraint=heating_limit,
        bank_angle_min=bank_min,
    )
    # Disable automatic scaling
    problem1.use_scaling = False
    print(
        f"Scaling status: problem1.use_scaling = {problem1.use_scaling}, "
        f"problem1._scaling.enabled = {problem1._scaling.enabled}"
    )

    # Set up mesh
    num_intervals = 15
    polynomial_degrees = [20] * num_intervals
    mesh_points = np.linspace(-1.0, 1.0, num_intervals + 1)
    problem1.set_mesh(polynomial_degrees, mesh_points)

    # Use original initial guess
    prepare_initial_guess(problem1, polynomial_degrees, deg2rad, initial_terminal_time=2000.0)

    # Solve
    start_time = time.time()
    solution1 = tl.solve_fixed_mesh(
        problem1,
        nlp_options=fixed_mesh_options,
    )
    solve_time1 = time.time() - start_time

    results["Fixed Mesh - Manual Scaling"] = {
        "solution": solution1,
        "sym_vars": sym_vars1,
        "solve_time": solve_time1,
    }

    print(f"  Success: {solution1.success}")
    if solution1.success:
        print(f"  Objective (final latitude): {-solution1.objective:.5f} radians")
        print(f"  Final time: {solution1.final_time:.2f} seconds")
        print(f"  Solve time: {solve_time1:.2f} seconds")

    print("\n=== FIXED MESH WITH AUTOMATIC SCALING ===")
    problem2, sym_vars2 = create_unscaled_shuttle_problem(
        heating_constraint=heating_limit,
        bank_angle_min=bank_min,
    )
    # Ensure automatic scaling is enabled using property
    problem2.use_scaling = True
    print(
        f"Scaling status: problem2.use_scaling = {problem2.use_scaling}, "
        f"problem2._scaling.enabled = {problem2._scaling.enabled}"
    )

    # Set up identical mesh
    problem2.set_mesh(polynomial_degrees, mesh_points)

    # Use initial guess without manual scaling
    prepare_unscaled_initial_guess(
        problem2, polynomial_degrees, deg2rad, initial_terminal_time=2000.0
    )

    # Solve
    start_time = time.time()
    solution2 = tl.solve_fixed_mesh(
        problem2,
        nlp_options=fixed_mesh_options,
    )
    solve_time2 = time.time() - start_time

    results["Fixed Mesh - Automatic Scaling"] = {
        "solution": solution2,
        "sym_vars": sym_vars2,
        "solve_time": solve_time2,
    }

    print(f"  Success: {solution2.success}")
    if solution2.success:
        print(f"  Objective (final latitude): {-solution2.objective:.5f} radians")
        print(f"  Final time: {solution2.final_time:.2f} seconds")
        print(f"  Solve time: {solve_time2:.2f} seconds")

    print("\n=== ADAPTIVE MESH WITH MANUAL SCALING (ORIGINAL) ===")

    # 3. ORIGINAL PROBLEM WITH MANUAL SCALING (use_scaling=False)
    problem3, sym_vars3 = create_shuttle_reentry_problem(
        heating_constraint=heating_limit,
        bank_angle_min=bank_min,
    )
    # Disable automatic scaling
    problem3.use_scaling = False

    # Set up initial mesh for adaptive
    initial_num_intervals = 9
    initial_poly_degree = 6
    initial_polynomial_degrees = [initial_poly_degree] * initial_num_intervals
    initial_mesh_points = np.linspace(-1.0, 1.0, initial_num_intervals + 1)
    problem3.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # Use original initial guess
    prepare_initial_guess(
        problem3, initial_polynomial_degrees, deg2rad, initial_terminal_time=2000.0
    )

    # Solve with adaptive mesh
    start_time = time.time()
    solution3 = tl.solve_adaptive(
        problem3,
        error_tolerance=1e-5,
        max_iterations=10,
        min_polynomial_degree=4,
        max_polynomial_degree=10,
        nlp_options=adaptive_mesh_options,
    )
    solve_time3 = time.time() - start_time

    results["Adaptive Mesh - Manual Scaling"] = {
        "solution": solution3,
        "sym_vars": sym_vars3,
        "solve_time": solve_time3,
    }

    print(f"  Success: {solution3.success}")
    if solution3.success:
        print(f"  Objective (final latitude): {-solution3.objective:.5f} radians")
        print(f"  Final time: {solution3.final_time:.2f} seconds")
        print(f"  Solve time: {solve_time3:.2f} seconds")
        print(f"  Final mesh intervals: {len(solution3.mesh_points) - 1}")
        print(f"  Final polynomial degrees: {solution3.polynomial_degrees}")

    print("\n=== ADAPTIVE MESH WITH AUTOMATIC SCALING ===")
    # 4. UNSCALED PROBLEM WITH AUTOMATIC SCALING (use_scaling=True)
    problem4, sym_vars4 = create_unscaled_shuttle_problem(
        heating_constraint=heating_limit,
        bank_angle_min=bank_min,
    )
    # Ensure automatic scaling is enabled
    problem4.use_scaling = True

    # Set up identical initial mesh
    problem4.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # Use unscaled initial guess
    prepare_unscaled_initial_guess(
        problem4, initial_polynomial_degrees, deg2rad, initial_terminal_time=2000.0
    )

    # Solve with adaptive mesh
    start_time = time.time()
    solution4 = tl.solve_adaptive(
        problem4,
        error_tolerance=1e-5,
        max_iterations=10,
        min_polynomial_degree=4,
        max_polynomial_degree=10,
        nlp_options=adaptive_mesh_options,
    )
    solve_time4 = time.time() - start_time

    results["Adaptive Mesh - Automatic Scaling"] = {
        "solution": solution4,
        "sym_vars": sym_vars4,
        "solve_time": solve_time4,
    }

    print(f"  Success: {solution4.success}")
    if solution4.success:
        print(f"  Objective (final latitude): {-solution4.objective:.5f} radians")
        print(f"  Final time: {solution4.final_time:.2f} seconds")
        print(f"  Solve time: {solve_time4:.2f} seconds")
        print(f"  Final mesh intervals: {len(solution4.mesh_points) - 1}")
        print(f"  Final polynomial degrees: {solution4.polynomial_degrees}")

    return results


def plot_fair_comparison(results):
    """Create comparison plots for the solutions."""
    # Plot state trajectories
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("State Variables Comparison - Manual vs. Automatic Scaling", fontsize=16)

    line_styles = {
        "Fixed Mesh - Manual Scaling": ("blue", "-"),
        "Fixed Mesh - Automatic Scaling": ("blue", "--"),
        "Adaptive Mesh - Manual Scaling": ("red", "-"),
        "Adaptive Mesh - Automatic Scaling": ("red", "--"),
    }

    # Define which variables to plot for each approach
    # Note: different variable names between manually scaled and automatically scaled versions
    variable_mapping = {
        "Fixed Mesh - Manual Scaling": [
            ("h_scaled", 0, 0, "Altitude"),
            ("phi", 0, 1, "Longitude (rad)"),
            ("theta", 1, 0, "Latitude (rad)"),
            ("v_scaled", 1, 1, "Velocity"),
            ("gamma", 2, 0, "Flight Path Angle (rad)"),
            ("psi", 2, 1, "Azimuth (rad)"),
        ],
        "Fixed Mesh - Automatic Scaling": [
            ("h", 0, 0, "Altitude"),
            ("phi", 0, 1, "Longitude (rad)"),
            ("theta", 1, 0, "Latitude (rad)"),
            ("v", 1, 1, "Velocity"),
            ("gamma", 2, 0, "Flight Path Angle (rad)"),
            ("psi", 2, 1, "Azimuth (rad)"),
        ],
        "Adaptive Mesh - Manual Scaling": [
            ("h_scaled", 0, 0, "Altitude"),
            ("phi", 0, 1, "Longitude (rad)"),
            ("theta", 1, 0, "Latitude (rad)"),
            ("v_scaled", 1, 1, "Velocity"),
            ("gamma", 2, 0, "Flight Path Angle (rad)"),
            ("psi", 2, 1, "Azimuth (rad)"),
        ],
        "Adaptive Mesh - Automatic Scaling": [
            ("h", 0, 0, "Altitude"),
            ("phi", 0, 1, "Longitude (rad)"),
            ("theta", 1, 0, "Latitude (rad)"),
            ("v", 1, 1, "Velocity"),
            ("gamma", 2, 0, "Flight Path Angle (rad)"),
            ("psi", 2, 1, "Azimuth (rad)"),
        ],
    }

    # Set up the titles once
    for row in range(3):
        for col in range(2):
            for _, mapping in variable_mapping.items():
                for _, r, c, title in mapping:
                    if r == row and c == col:
                        axs[row, col].set_title(title)
                        axs[row, col].grid(True)
                        break

    # Plot each trajectory
    for label, result_dict in results.items():
        solution = result_dict["solution"]
        sym_vars = result_dict["sym_vars"]

        if solution.success and label in variable_mapping:
            color, linestyle = line_styles[label]

            for var_name, row, col, _ in variable_mapping[label]:
                if var_name in sym_vars:
                    time_vals, var_vals = solution.get_symbolic_trajectory(sym_vars[var_name])

                    # For altitude and velocity, we need to convert to consistent units for comparison
                    if var_name == "h":
                        # Convert to scale used in manually scaled version (h/1e5)
                        var_vals = var_vals / 1e5
                    elif var_name == "v":
                        # Convert to scale used in manually scaled version (v/1e4)
                        var_vals = var_vals / 1e4

                    axs[row, col].plot(
                        time_vals, var_vals, color=color, linestyle=linestyle, label=label
                    )

    # Add legend to the first subplot
    axs[0, 0].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot control variables
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle("Control Variables Comparison", fontsize=16)

    control_mapping = {"alpha": (0, "Angle of Attack (rad)"), "beta": (1, "Bank Angle (rad)")}

    for label, result_dict in results.items():
        solution = result_dict["solution"]
        sym_vars = result_dict["sym_vars"]

        if solution.success:
            color, linestyle = line_styles[label]

            for control_name, (idx, title) in control_mapping.items():
                if control_name in sym_vars:
                    time_vals, control_vals = solution.get_symbolic_trajectory(
                        sym_vars[control_name]
                    )
                    axs2[idx].plot(
                        time_vals, control_vals, color=color, linestyle=linestyle, label=label
                    )
                    axs2[idx].set_title(title)
                    axs2[idx].grid(True)

    axs2[0].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Create a summary table
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    ax3.axis("tight")
    ax3.axis("off")

    # Compile the data for the table
    table_data = []
    for label in [
        "Fixed Mesh - Manual Scaling",
        "Fixed Mesh - Automatic Scaling",
        "Adaptive Mesh - Manual Scaling",
        "Adaptive Mesh - Automatic Scaling",
    ]:
        if label in results:
            result = results[label]
            solution = result["solution"]

            if solution.success:
                row = [
                    label,
                    f"{-solution.objective:.5f}",
                    f"{solution.final_time:.2f}",
                    f"{result['solve_time']:.2f}",
                ]

                # Add mesh details for adaptive solutions
                if "Adaptive" in label:
                    row.append(f"{len(solution.mesh_points) - 1}")
                    poly_range = (
                        f"{min(solution.polynomial_degrees)}-{max(solution.polynomial_degrees)}"
                    )
                    row.append(poly_range)
                else:
                    row.append("15")  # Fixed mesh intervals
                    row.append("20")  # Fixed polynomial degree
            else:
                row = [label, "N/A", "N/A", "N/A", "N/A", "N/A"]

            table_data.append(row)

    headers = [
        "Configuration",
        "Final Latitude (rad)",
        "Final Time (s)",
        "Solve Time (s)",
        "Intervals",
        "Poly. Degree",
    ]

    ax3.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    ax3.set_title("Numerical Comparison Summary", fontsize=16)

    plt.tight_layout()
    plt.show()


# Run the fair comparison and generate plots
results = run_fair_comparison()
plot_fair_comparison(results)
