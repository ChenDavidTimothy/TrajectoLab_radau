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

    # Define time with free final time
    t = problem.time(initial=0.0, free_final=True)

    # Define states with initial conditions and bounds - exactly as in Example 10.137
    h = problem.state("h", initial=260000.0, lower=0.0, final=80000.0)  # Altitude (ft)
    phi = problem.state("phi", initial=0.0 * deg2rad)  # Longitude (rad)
    theta = problem.state(
        "theta",
        initial=0.0 * deg2rad,  # Latitude (rad)
        lower=-89.0 * deg2rad,
        upper=89.0 * deg2rad,
    )
    v = problem.state("v", initial=25600.0, lower=1.0, final=2500.0)  # Velocity (ft/s)
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

    # Define dynamics exactly as in Equations 10.1050-10.1054
    problem.dynamics(
        {
            h: v * ca.sin(gamma),
            phi: (v / r) * ca.cos(gamma) * ca.sin(psi) / ca.cos(theta),
            theta: (v / r) * ca.cos(gamma) * ca.cos(psi),
            v: -(D / mass) - g * ca.sin(gamma),
            gamma: (L / (mass * v)) * ca.cos(beta) + ca.cos(gamma) * ((v / r) - (g / v)),
            psi: (1 / (mass * v * ca.cos(gamma))) * L * ca.sin(beta)
            + (v / (r * ca.cos(theta))) * ca.cos(gamma) * ca.sin(psi) * ca.sin(theta),
        }
    )

    # Add heating constraint if specified - Equation 10.1055
    if heating_constraint is not None:
        problem.subject_to(q <= heating_constraint)

    # Objective: Maximize final latitude (crossrange) - Equation (8.8)
    problem.minimize(-theta)  # Negative since we're maximizing

    return problem


def prepare_initial_guess(problem, polynomial_degrees, deg2rad):
    """Create an initial guess for the shuttle reentry problem"""
    states_guess = []
    controls_guess = []

    for N in polynomial_degrees:
        # Create state guess arrays with proper dimensions
        h_guess = np.linspace(260000, 80000, N + 1)
        phi_guess = np.zeros(N + 1)
        theta_guess = np.linspace(0, 0.6, N + 1)  # Guess around expected final value
        v_guess = np.linspace(25600, 2500, N + 1)
        gamma_guess = np.linspace(-1 * deg2rad, -5 * deg2rad, N + 1)
        psi_guess = np.ones(N + 1) * 90 * deg2rad

        # Stack state guesses
        state_array = np.vstack([h_guess, phi_guess, theta_guess, v_guess, gamma_guess, psi_guess])
        states_guess.append(state_array)

        # Control guesses
        alpha_guess = np.zeros(N)  # Start with zero angle of attack
        beta_guess = np.ones(N) * (-45 * deg2rad)  # Middle of bank angle range

        # Stack control guesses
        control_array = np.vstack([alpha_guess, beta_guess])
        controls_guess.append(control_array)

    # Set initial guess with a reasonable final time - close to expected values ~2000s
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
    # Define mesh with higher precision for accuracy
    polynomial_degrees = [18, 18, 18]  # High degrees for complex dynamics
    mesh_points = np.array([-1.0, -0.5, 0.0, 1.0])

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
            "ipopt.print_level": 1,  # Reduced output
            "ipopt.max_iter": 5000,
            "ipopt.tol": 1e-7,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "limited-memory",
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


def solve_with_adaptive_mesh(
    problem,
    example_name,
    example_num,
    bank_min,
    heating_limit=None,
    literature_J=None,
    literature_tf=None,
):
    """Solve the shuttle reentry problem with adaptive mesh and compare with literature"""
    # Define initial mesh for adaptive solution
    initial_polynomial_degrees = [8, 8, 8]  # Start with lower degree
    initial_mesh_points = np.array([-1.0, -0.3, 0.3, 1.0])

    # Set initial mesh
    problem.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # Conversion factor
    deg2rad = np.pi / 180.0

    # Set initial guess
    prepare_initial_guess(problem, initial_polynomial_degrees, deg2rad)

    # Solve with adaptive mesh
    heat_str = f"q_U = {heating_limit}" if heating_limit is not None else "q_U = ∞"
    bank_str = f"β ∈ [{bank_min}°, 1°]"
    print(f"\nSolving Example {example_num}: {example_name} (Adaptive Mesh)")
    print(f"Parameters: {bank_str}, {heat_str}")

    solution = tl.solve_adaptive(
        problem,
        initial_polynomial_degrees=initial_polynomial_degrees,
        initial_mesh_points=initial_mesh_points,
        error_tolerance=1e-3,
        max_iterations=15,
        min_polynomial_degree=4,
        max_polynomial_degree=12,
        nlp_options={
            "ipopt.print_level": 1,  # Reduced output
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-6,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "limited-memory",
        },
    )

    # Analyze solution
    analyze_solution(
        solution,
        example_name,
        example_num,
        "Adaptive Mesh",
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
        # Extract final values
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


def main():
    """
    Solve all three examples from the literature with both fixed and adaptive mesh:
    - Example 10.137: Maximum Crossrange with bank angle -90° ≤ β ≤ 1°, no heating constraint
    - Example 10.138: Maximum Crossrange with restricted bank angle -70° ≤ β ≤ 1°
    - Example 10.139: Maximum Crossrange with heating constraint q_U = 70 BTU/ft²/sec
    """
    # Literature values from the images
    lit_J_ex137 = 5.9587608e-1
    lit_tf_ex137 = 2.0085881e3

    lit_J_ex138 = 5.9574673e-1
    lit_tf_ex138 = 2.0346546e3

    lit_J_ex139 = 5.3451536e-1
    lit_tf_ex139 = 2.1986660e3

    # Example 10.137: Maximum Crossrange
    # Fixed Mesh
    problem_ex137 = create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0)
    solution_ex137_fixed = solve_with_fixed_mesh(
        problem_ex137, "SHUTTLE MAXIMUM CROSSRANGE", "10.137", -90, None, lit_J_ex137, lit_tf_ex137
    )

    # Adaptive Mesh
    problem_ex137 = create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-90.0)
    solution_ex137_adaptive = solve_with_adaptive_mesh(
        problem_ex137, "SHUTTLE MAXIMUM CROSSRANGE", "10.137", -90, None, lit_J_ex137, lit_tf_ex137
    )

    # Example 10.138: Maximum Crossrange with Control Bound
    # Fixed Mesh
    problem_ex138 = create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-70.0)
    solution_ex138_fixed = solve_with_fixed_mesh(
        problem_ex138,
        "SHUTTLE MAXIMUM CROSSRANGE WITH CONTROL BOUND",
        "10.138",
        -70,
        None,
        lit_J_ex138,
        lit_tf_ex138,
    )

    # Adaptive Mesh
    problem_ex138 = create_shuttle_reentry_problem(heating_constraint=None, bank_angle_min=-70.0)
    solution_ex138_adaptive = solve_with_adaptive_mesh(
        problem_ex138,
        "SHUTTLE MAXIMUM CROSSRANGE WITH CONTROL BOUND",
        "10.138",
        -70,
        None,
        lit_J_ex138,
        lit_tf_ex138,
    )

    # Example 10.139: Maximum Crossrange with Heat Limit
    # Fixed Mesh
    problem_ex139 = create_shuttle_reentry_problem(heating_constraint=70.0, bank_angle_min=-90.0)
    solution_ex139_fixed = solve_with_fixed_mesh(
        problem_ex139,
        "SHUTTLE MAXIMUM CROSSRANGE WITH HEAT LIMIT",
        "10.139",
        -90,
        70,
        lit_J_ex139,
        lit_tf_ex139,
    )

    # Adaptive Mesh
    problem_ex139 = create_shuttle_reentry_problem(heating_constraint=70.0, bank_angle_min=-90.0)
    solution_ex139_adaptive = solve_with_adaptive_mesh(
        problem_ex139,
        "SHUTTLE MAXIMUM CROSSRANGE WITH HEAT LIMIT",
        "10.139",
        -90,
        70,
        lit_J_ex139,
        lit_tf_ex139,
    )

    # Summary of all results
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    print("\nExample 10.137: SHUTTLE MAXIMUM CROSSRANGE")
    print(f"  Literature:    J* = {lit_J_ex137:.7e}, t_F* = {lit_tf_ex137:.7e}")
    if solution_ex137_fixed.success:
        j_fixed = solution_ex137_fixed.interpolate_state("theta", solution_ex137_fixed.final_time)
        tf_fixed = solution_ex137_fixed.final_time
        print(f"  Fixed Mesh:    J* = {float(j_fixed):.7e}, t_F* = {tf_fixed:.7e}")
    if solution_ex137_adaptive.success:
        j_adaptive = solution_ex137_adaptive.interpolate_state(
            "theta", solution_ex137_adaptive.final_time
        )
        tf_adaptive = solution_ex137_adaptive.final_time
        print(f"  Adaptive Mesh: J* = {float(j_adaptive):.7e}, t_F* = {tf_adaptive:.7e}")

    print("\nExample 10.138: SHUTTLE MAXIMUM CROSSRANGE WITH CONTROL BOUND")
    print(f"  Literature:    J* = {lit_J_ex138:.7e}, t_F* = {lit_tf_ex138:.7e}")
    if solution_ex138_fixed.success:
        j_fixed = solution_ex138_fixed.interpolate_state("theta", solution_ex138_fixed.final_time)
        tf_fixed = solution_ex138_fixed.final_time
        print(f"  Fixed Mesh:    J* = {float(j_fixed):.7e}, t_F* = {tf_fixed:.7e}")
    if solution_ex138_adaptive.success:
        j_adaptive = solution_ex138_adaptive.interpolate_state(
            "theta", solution_ex138_adaptive.final_time
        )
        tf_adaptive = solution_ex138_adaptive.final_time
        print(f"  Adaptive Mesh: J* = {float(j_adaptive):.7e}, t_F* = {tf_adaptive:.7e}")

    print("\nExample 10.139: SHUTTLE MAXIMUM CROSSRANGE WITH HEAT LIMIT")
    print(f"  Literature:    J* = {lit_J_ex139:.7e}, t_F* = {lit_tf_ex139:.7e}")
    if solution_ex139_fixed.success:
        j_fixed = solution_ex139_fixed.interpolate_state("theta", solution_ex139_fixed.final_time)
        tf_fixed = solution_ex139_fixed.final_time
        print(f"  Fixed Mesh:    J* = {float(j_fixed):.7e}, t_F* = {tf_fixed:.7e}")
    if solution_ex139_adaptive.success:
        j_adaptive = solution_ex139_adaptive.interpolate_state(
            "theta", solution_ex139_adaptive.final_time
        )
        tf_adaptive = solution_ex139_adaptive.final_time
        print(f"  Adaptive Mesh: J* = {float(j_adaptive):.7e}, t_F* = {tf_adaptive:.7e}")


if __name__ == "__main__":
    main()
