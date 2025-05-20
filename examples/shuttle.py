import casadi as ca
import numpy as np

import trajectolab as tl


def create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0):
    """
    Create the Space Shuttle reentry optimal control problem.

    Args:
        heating_constraint: Upper bound on aerodynamic heating (BTU/ft²/sec) or None for no constraint
        bank_angle_min: Minimum bank angle in degrees (default: -90°)

    Returns:
        TrajectoLab Problem instance
    """
    # Create problem
    problem = tl.Problem("Space Shuttle Reentry Trajectory")

    # Define constants
    # Aerodynamic coefficients - exactly as in Table 10.31
    a0 = -0.20704
    a1 = 0.029244
    b0 = 0.07854
    b1 = -0.61592e-2
    b2 = 0.621408e-3
    c0 = 1.0672181
    c1 = -0.19213774e-1
    c2 = 0.21286289e-3
    c3 = -0.10117249e-5

    # Physical constants - exactly as in Table 10.31
    mu = 0.14076539e17  # Gravitational parameter
    Re = 20902900  # Earth radius (ft)
    S = 2690  # Reference area (ft²)
    rho0 = 0.002378  # Sea-level density (slug/ft³)
    hr = 23800  # Scale height (ft)
    g0 = 32.174  # Reference gravity (ft/s²)

    # Vehicle parameters
    weight = 203000  # Weight (lb)
    mass = weight / g0  # Mass (slug)

    # Conversion factors
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    # Scaling factors (from Julia implementation)
    h_scale = 1e5  # Altitude scaling factor
    v_scale = 1e4  # Velocity scaling factor

    # Define time with free final time
    t = problem.time(initial=0.0, free_final=True)

    # Define states with initial conditions and bounds - using scaled values
    # Note: Initial values and bounds are now scaled by h_scale and v_scale
    h_scaled = problem.state(
        "h_scaled",
        initial=260000.0 / h_scale,  # 2.6 in scaled units
        lower=0.0,
        final=80000.0 / h_scale,
    )  # 0.8 in scaled units

    phi = problem.state("phi", initial=0.0 * deg2rad)  # Longitude (rad)

    theta = problem.state(
        "theta",
        initial=0.0 * deg2rad,  # Latitude (rad)
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
    )

    v_scaled = problem.state(
        "v_scaled",
        initial=25600.0 / v_scale,  # 2.56 in scaled units
        lower=1.0 / v_scale,  # 0.0001 in scaled units
        final=2500.0 / v_scale,
    )  # 0.25 in scaled units

    gamma = problem.state(
        "gamma",
        initial=-1.0 * deg2rad,  # Flight path angle (rad)
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
        final=-5.0 * deg2rad,
    )

    psi = problem.state("psi", initial=90.0 * deg2rad)  # Azimuth (rad)

    # Define controls with bounds
    alpha = problem.control(
        "alpha", lower=-90.0 * deg2rad, upper=90.0 * deg2rad
    )  # Angle of attack (rad)

    beta = problem.control(
        "beta", lower=bank_angle_min * deg2rad, upper=1.0 * deg2rad
    )  # Bank angle (rad)

    # Define intermediate variables for dynamics
    # Unscaled altitude and velocity - needed for physics calculations
    h = h_scaled * h_scale
    v = v_scaled * v_scale

    # Add a small epsilon to avoid division by zero
    eps = 1e-10

    # Radius from Earth center
    r = Re + h

    # Atmospheric density
    rho = rho0 * ca.exp(-h / hr)

    # Gravitational acceleration
    g = mu / (r * r)

    # Convert alpha to degrees for coefficient calculations (α̂ = (180/π)α)
    alpha_deg = alpha * rad2deg

    # Aerodynamic coefficients
    CL = a0 + a1 * alpha_deg
    CD = b0 + b1 * alpha_deg + b2 * alpha_deg * alpha_deg

    # Aerodynamic forces
    q_dyn = 0.5 * rho * v * v  # Dynamic pressure
    L = q_dyn * CL * S  # Lift force
    D = q_dyn * CD * S  # Drag force

    # Heating calculation
    qr = 17700 * ca.sqrt(rho) * (0.0001 * v) ** 3.07
    qa = c0 + c1 * alpha_deg + c2 * alpha_deg**2 + c3 * alpha_deg**3
    q = qa * qr

    # Define dynamics using scaled variables
    # Note: We need to scale the derivatives of h and v appropriately
    problem.dynamics(
        {
            h_scaled: (v * ca.sin(gamma)) / h_scale,
            phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / (ca.cos(theta) + eps),
            theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
            v_scaled: (-(D / mass) - g * ca.sin(gamma)) / v_scale,
            gamma: (L / (mass * v + eps)) * ca.cos(beta)
            + ca.cos(gamma) * ((v / r) - (g / (v + eps))),
            psi: (1 / (mass * v * ca.cos(gamma) + eps)) * L * ca.sin(beta)
            + (v / (r * (ca.cos(theta) + eps))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
        }
    )

    # Add heating constraint if specified - Equation 10.1055
    if heating_constraint is not None:
        problem.subject_to(q <= heating_constraint)

    # Objective: Maximize final latitude (crossrange) - Equation (8.8)
    problem.minimize(-theta)  # Negative since we're maximizing

    return problem


def prepare_initial_guess(problem, polynomial_degrees, deg2rad):
    """
    Prepare initial guess for the problem, using scaled variables.

    Args:
        problem: The problem to prepare guess for
        polynomial_degrees: List of polynomial degrees for each mesh interval
        deg2rad: Conversion factor from degrees to radians
    """
    states_guess = []
    controls_guess = []

    # Scale factors
    h_scale = 1e5
    v_scale = 1e4

    # Initial conditions (scaled)
    h0, v0 = 260000.0 / h_scale, 25600.0 / v_scale  # 2.6, 2.56 in scaled units
    phi0, theta0 = 0.0, 0.0
    gamma0, psi0 = -1.0 * np.pi / 180.0, 90.0 * np.pi / 180.0

    # Final conditions (scaled)
    hf, vf = 80000.0 / h_scale, 2500.0 / v_scale  # 0.8, 0.25 in scaled units
    gammaF = -5.0 * np.pi / 180.0

    for N in polynomial_degrees:
        # Create linearly spaced points
        t = np.linspace(0, 1, N + 1)

        # Linear interpolation for states (using scaled values)
        h_vals = h0 + (hf - h0) * t
        phi_vals = phi0 * np.ones(N + 1)
        theta_vals = theta0 * np.ones(N + 1)
        v_vals = v0 + (vf - v0) * t
        gamma_vals = gamma0 + (gammaF - gamma0) * t
        psi_vals = psi0 * np.ones(N + 1)

        # Stack states
        state_array = np.vstack([h_vals, phi_vals, theta_vals, v_vals, gamma_vals, psi_vals])
        states_guess.append(state_array)

        # Simple zero controls
        alpha_vals = np.zeros(N)
        beta_vals = -45.0 * np.pi / 180.0 * np.ones(N)

        # Stack controls
        control_array = np.vstack([alpha_vals, beta_vals])
        controls_guess.append(control_array)

    # Set with a reasonable final time
    problem.set_initial_guess(states=states_guess, controls=controls_guess, terminal_time=2000.0)


def solve_with_fixed_mesh(
    problem,
    example_name,
    example_num,
    bank_min,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    """Solve the shuttle reentry problem with fixed mesh and compare with literature"""
    # Define mesh with simpler and more uniform configuration
    # This is more similar to the Julia implementation
    num_intervals = 15  # Similar to Julia's approach
    polynomial_degrees = [20] * num_intervals  # Use uniform degree
    mesh_points = np.linspace(-1.0, 1.0, num_intervals + 1)  # Uniform mesh

    # Set mesh
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Conversion factor
    deg2rad = np.pi / 180.0

    # Set initial guess
    prepare_initial_guess(problem, polynomial_degrees, deg2rad)

    # Solve with fixed mesh
    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min}°, 1°]"
    print(f"\nSolving Example {example_num}: {example_name} (Fixed Mesh)")
    print(f"Parameters: {bank_str}, {heat_str}")

    solution = tl.solve_fixed_mesh(
        problem,
        polynomial_degrees=polynomial_degrees,
        mesh_points=mesh_points,
        nlp_options={
            "ipopt.max_iter": 2000,
            "ipopt.mumps_pivtol": 5e-7,
            "ipopt.mumps_mem_percent": 50000,
            "ipopt.linear_solver": "mumps",
            "ipopt.constr_viol_tol": 1e-7,
            "ipopt.print_level": 5,
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.tol": 1e-8,
        },
    )

    # Analyze solution
    analyze_solution(
        solution,
        example_name,
        example_num,
        "Fixed Mesh",
        bank_min,
        heating_limit,
        literature_J,
        literature_tf,
    )

    return solution


def analyze_solution(
    solution,
    example_name,
    example_num,
    method,
    bank_min,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    """Analyze the solution and print results comparing with literature values"""
    if solution.success:
        # Extract final values - remember to unscale h and v
        final_time = solution.final_time
        final_theta = solution.interpolate_state("theta", final_time)
        final_theta_rad = float(final_theta)
        final_theta_deg = final_theta_rad * 180 / np.pi

        # Format for scientific notation matching the literature
        J_formatted = f"{final_theta_rad:.7e}".replace("e-0", "e-").replace("e+0", "e+")
        tf_formatted = f"{final_time:.7e}".replace("e+0", "e+")

        heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
        bank_str = f"β ∈ [{bank_min}°, 1°]"

        print(f"\nExample {example_num}: {example_name} ({method})")
        print(f"Parameters: {bank_str}, {heat_str}")
        print("Optimal Results:")
        print(f"  J* = {J_formatted}  (final latitude in radians)")
        print(f"  t_F* = {tf_formatted}  (final time in seconds)")
        print(f"  Final latitude: {final_theta_deg:.4f}°")

        # Compare with literature if provided
        if literature_J is not None and literature_tf is not None:
            J_diff = abs(final_theta_rad - literature_J) / literature_J * 100
            tf_diff = abs(final_time - literature_tf) / literature_tf * 100

            print("\nComparison with literature values:")
            print(f"  Literature J* = {literature_J:.7e}")
            print(f"  Literature t_F* = {literature_tf:.7e}")
            print(f"  J* difference: {J_diff:.4f}%")
            print(f"  t_F* difference: {tf_diff:.4f}%")

        # Print mesh information for adaptive solutions
        if method == "Adaptive Mesh" and solution.polynomial_degrees is not None:
            print("\nFinal mesh details:")
            print(f"  Polynomial degrees: {solution.polynomial_degrees}")
            if solution.mesh_points is not None:
                print(f"  Number of mesh intervals: {len(solution.mesh_points) - 1}")

        return True
    else:
        print(f"\nExample {example_num} Solution ({method}) Failed:")
        print(f"  Reason: {solution.message}")
        return False


def plot_solution(solution):
    """
    Plot the solution using matplotlib, remembering to unscale the variables.

    Args:
        solution: TrajectoLab solution
    """
    import matplotlib.pyplot as plt

    # Scale factors
    h_scale = 1e5
    v_scale = 1e4

    # Get trajectories using the appropriate API methods
    # Each call returns (time_array, value_array)
    time_h, h_scaled_vals = solution.get_state_trajectory("h_scaled")
    time_v, v_scaled_vals = solution.get_state_trajectory("v_scaled")
    time_phi, phi_vals = solution.get_state_trajectory("phi")
    time_theta, theta_vals = solution.get_state_trajectory("theta")
    time_gamma, gamma_vals = solution.get_state_trajectory("gamma")
    time_psi, psi_vals = solution.get_state_trajectory("psi")

    # Get control trajectories
    time_alpha, alpha_vals = solution.get_control_trajectory("alpha")
    time_beta, beta_vals = solution.get_control_trajectory("beta")

    # Unscale altitude and velocity
    h_vals = h_scaled_vals * h_scale
    v_vals = v_scaled_vals * v_scale

    # Convert to degrees for plotting
    phi_deg = phi_vals * 180 / np.pi
    theta_deg = theta_vals * 180 / np.pi
    gamma_deg = gamma_vals * 180 / np.pi
    psi_deg = psi_vals * 180 / np.pi
    alpha_deg = alpha_vals * 180 / np.pi
    beta_deg = beta_vals * 180 / np.pi

    # Create figure for state variables
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Plot altitude
    axs[0, 0].plot(time_h, h_vals / 1e5)  # Plot in units of 100,000 ft
    axs[0, 0].set_title("Altitude (100,000 ft)")
    axs[0, 0].grid(True)

    # Plot velocity
    axs[0, 1].plot(time_v, v_vals / 1e3)  # Plot in units of 1,000 ft/s
    axs[0, 1].set_title("Velocity (1,000 ft/s)")
    axs[0, 1].grid(True)

    # Plot longitude
    axs[1, 0].plot(time_phi, phi_deg)
    axs[1, 0].set_title("Longitude (deg)")
    axs[1, 0].grid(True)

    # Plot flight path angle
    axs[1, 1].plot(time_gamma, gamma_deg)
    axs[1, 1].set_title("Flight Path Angle (deg)")
    axs[1, 1].grid(True)

    # Plot latitude
    axs[2, 0].plot(time_theta, theta_deg)
    axs[2, 0].set_title("Latitude (deg)")
    axs[2, 0].grid(True)

    # Plot azimuth
    axs[2, 1].plot(time_psi, psi_deg)
    axs[2, 1].set_title("Azimuth (deg)")
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Create figure for control variables
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot angle of attack
    axs[0].plot(time_alpha, alpha_deg)
    axs[0].set_title("Angle of Attack (deg)")
    axs[0].grid(True)

    # Plot bank angle
    axs[1].plot(time_beta, beta_deg)
    axs[1].set_title("Bank Angle (deg)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Optionally, create a 3D plot of the trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(phi_deg, theta_deg, h_vals / 1e5)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Altitude (100,000 ft)")
    ax.set_title("Space Shuttle Reentry Trajectory")
    plt.show()


def main():
    """
    Simplified main function - just solve one example to verify scaling helps
    """
    # Literature values from the book
    lit_J_ex137 = 5.9587608e-1
    lit_tf_ex137 = 2.0085881e3

    # Create problem with scaling
    problem = create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0)

    # Solve it with fixed mesh
    solution = solve_with_fixed_mesh(
        problem, "SHUTTLE MAXIMUM CROSSRANGE", "10.137", -90, None, lit_J_ex137, lit_tf_ex137
    )

    # Plot if solution is successful
    if solution.success:
        plot_solution(solution)


if __name__ == "__main__":
    main()
