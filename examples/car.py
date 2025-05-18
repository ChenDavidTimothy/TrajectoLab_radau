import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from trajectolab import FixedMesh, InitialGuess, PHSAdaptive, Problem, RadauDirectSolver, solve


# Create the car race problem
problem = Problem("Car Race - Time Optimal")

# Define time variable (free final time for time-optimal control)
t = problem.time(initial=0.0, free_final=True)

# Define states with boundary conditions
pos = problem.state("position", initial=0.0, final=1.0)  # start at 0, finish line at 1
speed = problem.state("speed", initial=0.0)  # start from standstill

# Define control with bounds (throttle between 0 and 1)
throttle = problem.control("throttle", lower=0.0, upper=1.0)

# Define system dynamics
# pos_dot = speed
# speed_dot = throttle - speed (throttle minus drag proportional to speed)
problem.dynamics({pos: speed, speed: throttle - speed})

# Path constraint: speed limit varies with position
# limit(pos) = 1 - sin(2*pi*pos)/2
speed_limit = 1 - ca.sin(2 * ca.pi * pos) / 2
problem.subject_to(speed <= speed_limit)

# Objective: minimize race time (time-optimal control)
problem.minimize(t.final)

# Set initial guesses to help convergence
problem.set_initial_guess(speed, 1.0)
problem.set_initial_guess(throttle, 0.5)

# Create initial guess for time


initial_guess = InitialGuess(
    initial_time_variable=0.0,
    terminal_time_variable=1.0,  # Initial guess for final time
)
problem.initial_guess = initial_guess

# Configure the adaptive solver
solver = RadauDirectSolver(
    mesh_method=PHSAdaptive(
        error_tolerance=1e-7,  # Error tolerance for mesh refinement
        max_iterations=30,  # Maximum adaptive iterations
        min_polynomial_degree=4,  # Minimum polynomial degree per interval
        max_polynomial_degree=16,  # Maximum polynomial degree per interval
        ode_solver_tolerance=1e-7,  # ODE solver tolerance for error estimation
        num_error_sim_points=50,  # Points for error simulation
        initial_polynomial_degrees=[6, 8, 10],  # Starting polynomial degrees
        initial_mesh_points=[-1.0, -0.33, 0.33, 1.0],  # Starting with 3 intervals
    ),
    nlp_options={
        "ipopt.print_level": 2,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
        "print_time": 0,
    },
)

# Solve the problem
print("Solving car race problem with adaptive mesh refinement...")
solution = solve(problem, solver)

# Analyze and display results
if solution.success:
    print("\n🏁 Race completed successfully!")
    print(f"Optimal lap time: {solution.final_time:.6f} time units")
    print(f"Final mesh intervals: {solution.num_intervals}")
    print(f"Polynomial degrees per interval: {solution.polynomial_degrees}")
    print(f"Objective value: {solution.objective:.6f}")

    # Get trajectories for analysis
    t_vals, pos_vals = solution.get_symbolic_trajectory(pos)
    t_speed, speed_vals = solution.get_symbolic_trajectory(speed)
    t_throttle, throttle_vals = solution.get_symbolic_trajectory(throttle)

    # Calculate speed limit along the trajectory for verification
    speed_limit_vals = 1 - np.sin(2 * np.pi * pos_vals) / 2

    # Custom plotting to match the original CasADi example
    plt.figure(figsize=(14, 10))

    # Main trajectory plot
    plt.subplot(2, 2, 1)
    plt.plot(t_speed, speed_vals, "b-", linewidth=2, label="Speed")
    plt.plot(t_vals, pos_vals, "g-", linewidth=2, label="Position")
    plt.plot(t_vals, speed_limit_vals, "r--", linewidth=2, label="Speed limit")
    plt.step(t_throttle, throttle_vals, "k-", where="post", linewidth=1.5, label="Throttle")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Car Race - Optimal Trajectories")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Speed vs position plot
    plt.subplot(2, 2, 2)
    plt.plot(pos_vals, speed_vals, "b-", linewidth=2, label="Actual speed")
    pos_fine = np.linspace(0, 1, 1000)
    speed_limit_fine = 1 - np.sin(2 * np.pi * pos_fine) / 2
    plt.plot(pos_fine, speed_limit_fine, "r--", linewidth=2, label="Speed limit")
    plt.xlabel("Position")
    plt.ylabel("Speed")
    plt.title("Speed Profile vs Position")
    plt.legend()
    plt.grid(True)

    # Control profile
    plt.subplot(2, 2, 3)
    plt.step(t_throttle, throttle_vals, "k-", where="post", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Throttle")
    plt.title("Optimal Control (Throttle)")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    # Mesh information
    plt.subplot(2, 2, 4)
    if solution.mesh_points is not None and solution.polynomial_degrees is not None:
        mesh_centers = []
        for i in range(len(solution.polynomial_degrees)):
            center = (solution.mesh_points[i] + solution.mesh_points[i + 1]) / 2
            mesh_centers.append(center)

        plt.bar(
            mesh_centers,
            solution.polynomial_degrees,
            width=np.diff(solution.mesh_points),
            alpha=0.7,
            edgecolor="black",
        )
        plt.xlabel("Time")
        plt.ylabel("Polynomial Degree")
        plt.title(f"Adaptive Mesh ({solution.num_intervals} intervals)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Verification: check constraint violations
    max_speed_violation = np.max(speed_vals - speed_limit_vals)
    print("\nConstraint verification:")
    print(f"Maximum speed limit violation: {max_speed_violation:.8f}")

    if max_speed_violation > 1e-6:
        print("⚠️  Warning: Speed limit constraint may be violated!")
    else:
        print("✅ All constraints satisfied")

    # Plot the standard TrajectoLab visualization
    print("\nShowing TrajectoLab standard plot...")
    solution.plot(figsize=(12, 8))

else:
    print(f"❌ Solution failed: {solution.message}")

    # If adaptive fails, try fixed mesh as fallback
    print("\nTrying with fixed mesh as fallback...")
    fixed_solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=[10] * 10,  # 10 intervals with degree 10 each
            mesh_points=np.linspace(-1, 1, 11),
        ),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000, "ipopt.tol": 1e-6},
    )

    fallback_solution = solve(problem, fixed_solver)
    if fallback_solution.success:
        print("✅ Fallback solution successful!")
        print(f"Optimal lap time: {fallback_solution.final_time:.6f}")
        fallback_solution.plot()
    else:
        print(f"❌ Fallback also failed: {fallback_solution.message}")
