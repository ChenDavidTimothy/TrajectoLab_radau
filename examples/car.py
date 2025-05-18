"""
Car Race Optimal Control Problem
=================================

Time-optimal control of a race car with position-dependent speed limits.

This example demonstrates:
- Free final time problem (time-optimal control)
- Path constraints (position-dependent speed limits)
- Adaptive mesh refinement with explicit configuration
- Custom plotting and constraint verification

Compatible with the current TrajectoLab design (explicit mesh control).
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from trajectolab import FixedMesh, PHSAdaptive, Problem, RadauDirectSolver, solve


def create_car_race_problem() -> Problem:
    """Create the car race optimal control problem."""
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

    return problem, pos, speed, throttle


def solve_car_race_adaptive() -> None:
    """Solve car race with adaptive mesh refinement."""
    print("="*80)
    print("CAR RACE - ADAPTIVE MESH REFINEMENT")
    print("="*80)

    # Create problem
    problem, pos, speed, throttle = create_car_race_problem()

    # EXPLICITLY set initial mesh for adaptive algorithm
    initial_polynomial_degrees = [6, 8, 10]
    initial_mesh_points = np.array([-1.0, -0.33, 0.33, 1.0])

    # Set mesh explicitly on problem
    problem.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # OPTIONALLY provide initial guess for first iteration
    # Create simple linear initial guess
    states_guess = []
    controls_guess = []

    for N_k in initial_polynomial_degrees:
        # State guess: position from 0 to 1, speed from 0 to 0.5
        tau_points = np.linspace(-1, 1, N_k + 1)
        pos_values = 0.5 * (tau_points + 1)  # Linear from 0 to 1
        speed_values = 0.25 * (tau_points + 1)  # Linear from 0 to 0.5

        state_array = np.array([pos_values, speed_values])
        states_guess.append(state_array)

        # Control guess: moderate constant throttle
        control_array = np.full((1, N_k), 0.5, dtype=np.float64)
        controls_guess.append(control_array)

    # Set initial guess via problem (optional)
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        terminal_time=2.0,  # Initial guess for final time
    )

    # Configure the adaptive solver
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            # Required parameters - must be explicitly provided
            initial_polynomial_degrees=initial_polynomial_degrees,
            initial_mesh_points=initial_mesh_points,
            # Optional initial guess (uses what we set on problem above)
            initial_guess=problem.initial_guess,
            # Optional algorithm parameters
            error_tolerance=1e-7,
            max_iterations=30,
            min_polynomial_degree=4,
            max_polynomial_degree=16,
            ode_solver_tolerance=1e-7,
            num_error_sim_points=50,
        ),
        nlp_options={
            "ipopt.print_level": 2,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-8,
            "ipopt.acceptable_tol": 1e-6,
            "print_time": 0,
        },
    )

    # Show solver input summary
    summary = problem.get_solver_input_summary()
    print("Solver Input Summary:")
    print(summary)

    # Solve the problem
    print("\nSolving car race problem with adaptive mesh refinement...")
    solution = solve(problem, solver)

    # Analyze and display results
    if solution.success:
        print(f"\n🏁 Race completed successfully!")
        print(f"Optimal lap time: {solution.final_time:.6f} time units")
        print(f"Final mesh intervals: {solution.num_intervals}")
        print(f"Polynomial degrees per interval: {solution.polynomial_degrees}")
        print(f"Objective value: {solution.objective:.6f}")

        # Get trajectories for analysis using symbolic variables
        t_vals, pos_vals = solution.get_symbolic_trajectory(pos)
        t_speed, speed_vals = solution.get_symbolic_trajectory(speed)
        t_throttle, throttle_vals = solution.get_symbolic_trajectory(throttle)

        # Verify constraint satisfaction
        verify_constraints(pos_vals, speed_vals)

        # Create custom plots
        plot_car_race_results(t_vals, pos_vals, t_speed, speed_vals, t_throttle, throttle_vals, solution)

        # Show standard TrajectoLab plot
        print("\nShowing TrajectoLab standard plot...")
        solution.plot(figsize=(12, 8))

    else:
        print(f"❌ Solution failed: {solution.message}")

        # Try fixed mesh as fallback
        try_fixed_mesh_fallback(problem)


def solve_car_race_fixed_mesh() -> None:
    """Solve car race with fixed mesh."""
    print("="*80)
    print("CAR RACE - FIXED MESH")
    print("="*80)

    # Create problem
    problem, pos, speed, throttle = create_car_race_problem()

    # EXPLICITLY set fixed mesh
    polynomial_degrees = [10, 12, 15, 12, 10]
    mesh_points = np.array([-1.0, -0.5, -0.1, 0.3, 0.7, 1.0])

    # Set mesh explicitly on problem
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Create detailed initial guess
    states_guess = []
    controls_guess = []

    for N_k in polynomial_degrees:
        # More sophisticated initial guess
        tau_points = np.linspace(-1, 1, N_k + 1)
        global_time = (tau_points + 1) / 2  # Map to [0, 1]

        # Position: quadratic profile from 0 to 1
        pos_values = global_time**1.5
        # Speed: build up then taper off
        speed_values = 0.8 * global_time * (1 - 0.5 * global_time)

        state_array = np.array([pos_values, speed_values])
        states_guess.append(state_array)

        # Control: higher at start, moderate later
        tau_control = np.linspace(-1, 1, N_k)
        global_time_control = (tau_control + 1) / 2
        throttle_values = 0.8 * (1 - 0.3 * global_time_control)
        control_array = throttle_values.reshape(1, -1)
        controls_guess.append(control_array)

    # Set initial guess
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        terminal_time=1.8,
    )

    # Create fixed mesh solver
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=polynomial_degrees,
            mesh_points=mesh_points,
            # Could also provide initial_guess here, but using problem.set_initial_guess()
        ),
        nlp_options={
            "ipopt.print_level": 2,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-8,
        },
    )

    # Solve
    print("Solving car race problem with fixed mesh...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"\n🏁 Fixed mesh race completed!")
        print(f"Optimal lap time: {solution.final_time:.6f} time units")
        print(f"Objective value: {solution.objective:.6f}")

        # Get trajectories
        t_vals, pos_vals = solution.get_symbolic_trajectory(pos)
        t_speed, speed_vals = solution.get_symbolic_trajectory(speed)
        t_throttle, throttle_vals = solution.get_symbolic_trajectory(throttle)

        # Verify and plot
        verify_constraints(pos_vals, speed_vals)
        plot_car_race_results(t_vals, pos_vals, t_speed, speed_vals, t_throttle, throttle_vals, solution)
        solution.plot()
    else:
        print(f"❌ Fixed mesh solution failed: {solution.message}")


def try_fixed_mesh_fallback(problem: Problem) -> None:
    """Try fixed mesh as fallback when adaptive fails."""
    print("\nTrying with fixed mesh as fallback...")

    # Reset mesh for fixed solver
    fallback_degrees = [10] * 10
    fallback_mesh = np.linspace(-1, 1, 11)
    problem.set_mesh(fallback_degrees, fallback_mesh)

    # Use simple initial guess
    problem.set_initial_guess(terminal_time=2.5)

    fixed_solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=fallback_degrees,
            mesh_points=fallback_mesh,
        ),
        nlp_options={
            "ipopt.print_level": 2,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-6
        },
    )

    fallback_solution = solve(problem, fixed_solver)
    if fallback_solution.success:
        print("✅ Fallback solution successful!")
        print(f"Optimal lap time: {fallback_solution.final_time:.6f}")
        fallback_solution.plot()
    else:
        print(f"❌ Fallback also failed: {fallback_solution.message}")


def verify_constraints(pos_vals: np.ndarray, speed_vals: np.ndarray) -> None:
    """Verify that speed limit constraints are satisfied."""
    # Calculate speed limit along the trajectory
    speed_limit_vals = 1 - np.sin(2 * np.pi * pos_vals) / 2

    # Check constraint violations
    violations = speed_vals - speed_limit_vals
    max_violation = np.max(violations)

    print("\nConstraint verification:")
    print(f"Maximum speed limit violation: {max_violation:.8f}")

    if max_violation > 1e-6:
        print("⚠️  Warning: Speed limit constraint may be violated!")
        num_violations = np.sum(violations > 1e-6)
        print(f"Number of points with violations > 1e-6: {num_violations}")
    else:
        print("✅ All constraints satisfied")


def plot_car_race_results(
    t_vals: np.ndarray,
    pos_vals: np.ndarray,
    t_speed: np.ndarray,
    speed_vals: np.ndarray,
    t_throttle: np.ndarray,
    throttle_vals: np.ndarray,
    solution: object
) -> None:
    """Create custom plots for car race results."""
    # Calculate speed limit for plotting
    speed_limit_vals = 1 - np.sin(2 * np.pi * pos_vals) / 2

    # Create comprehensive plot
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
    plt.grid(True, alpha=0.3)

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
    plt.grid(True, alpha=0.3)

    # Control profile
    plt.subplot(2, 2, 3)
    plt.step(t_throttle, throttle_vals, "k-", where="post", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Throttle")
    plt.title("Optimal Control (Throttle)")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)

    # Mesh information (if available)
    plt.subplot(2, 2, 4)
    if hasattr(solution, 'mesh_points') and hasattr(solution, 'polynomial_degrees'):
        if solution.mesh_points is not None and solution.polynomial_degrees is not None:
            # Convert mesh points from [-1,1] to physical time
            physical_mesh_points = np.interp(
                solution.mesh_points,
                [-1, 1],
                [t_vals[0], t_vals[-1]]
            )

            mesh_centers = []
            for i in range(len(solution.polynomial_degrees)):
                center = (physical_mesh_points[i] + physical_mesh_points[i + 1]) / 2
                mesh_centers.append(center)

            widths = np.diff(physical_mesh_points)

            plt.bar(
                mesh_centers,
                solution.polynomial_degrees,
                width=widths,
                alpha=0.7,
                edgecolor="black",
            )
            plt.xlabel("Time")
            plt.ylabel("Polynomial Degree")
            plt.title(f"Mesh Structure ({solution.num_intervals} intervals)")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Mesh information\nnot available",
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Mesh Information")
    else:
        plt.text(0.5, 0.5, "Fixed mesh\n(no refinement data)",
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Mesh Information")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run car race examples."""
    print("TrajectoLab Car Race Example")
    print("=" * 40)
    print("Demonstrates time-optimal control with path constraints")
    print("and adaptive mesh refinement.\n")

    # Solve with adaptive mesh
    solve_car_race_adaptive()

    # Solve with fixed mesh for comparison
    solve_car_race_fixed_mesh()

    print("\n" + "=" * 40)
    print("Car race examples completed!")


if __name__ == "__main__":
    main()
