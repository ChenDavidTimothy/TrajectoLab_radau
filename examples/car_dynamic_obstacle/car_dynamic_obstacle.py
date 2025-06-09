#!/usr/bin/env python3
"""
Complete Dynamic Obstacle Avoidance Example with Real-Time Animation

This file demonstrates:
1. Time-dependent obstacle avoidance optimization using MAPTOR
2. Real-time accurate animation of the solution
3. Constraint verification and visualization

The vehicle must navigate from (0,0) to (20,20) while avoiding a moving obstacle
that follows a complex waypoint trajectory.
"""

import casadi as ca
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

import maptor as mtor


def create_challenging_obstacle_position(current_time):
    """Create symbolic obstacle position with complex multi-segment trajectory."""
    # Define waypoints: [x, y, time]
    waypoints = np.array(
        [
            [10.0, 15.0, 0.0],  # Start: upper left
            [15.0, 10.0, 3.0],  # Move to center, crossing vehicle path
            [12.0, 8.0, 5.0],  # Slow down near path
            [8.0, 6.0, 8.0],  # Block lower route
            [5.0, 12.0, 10.0],  # Move up to block alternative
            [25.0, 25.0, 15.0],  # Exit area quickly
        ]
    )

    # Clamp time to waypoint bounds
    t_min, t_max = waypoints[0, 2], waypoints[-1, 2]
    t_clamped = ca.fmax(t_min, ca.fmin(t_max, current_time))

    # Initialize with final waypoint (fallback)
    obs_x = waypoints[-1, 0]
    obs_y = waypoints[-1, 1]

    # Multi-segment interpolation with conditional logic
    for i in range(len(waypoints) - 1):
        t1, t2 = waypoints[i, 2], waypoints[i + 1, 2]
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        in_segment = ca.logic_and(t_clamped >= t1, t_clamped < t2)
        alpha = (t_clamped - t1) / (t2 - t1)
        x_interp = (1 - alpha) * wp1[0] + alpha * wp2[0]
        y_interp = (1 - alpha) * wp1[1] + alpha * wp2[1]
        obs_x = ca.if_else(in_segment, x_interp, obs_x)
        obs_y = ca.if_else(in_segment, y_interp, obs_y)

    return obs_x, obs_y


def create_obstacle_trajectory_numpy(time_array):
    """Create obstacle trajectory using same waypoints as symbolic version."""
    waypoints = np.array(
        [
            [10.0, 15.0, 0.0],  # Start: upper left
            [15.0, 10.0, 3.0],  # Move to center, crossing vehicle path
            [12.0, 8.0, 5.0],  # Slow down near path
            [8.0, 6.0, 8.0],  # Block lower route
            [5.0, 12.0, 10.0],  # Move up to block alternative
            [25.0, 25.0, 15.0],  # Exit area quickly
        ]
    )

    obs_x = np.zeros_like(time_array)
    obs_y = np.zeros_like(time_array)

    for i, t in enumerate(time_array):
        # Clamp time to waypoint bounds
        t_clamped = np.clip(t, waypoints[0, 2], waypoints[-1, 2])

        # Find segment and interpolate
        if t_clamped >= waypoints[-1, 2]:
            obs_x[i] = waypoints[-1, 0]
            obs_y[i] = waypoints[-1, 1]
        else:
            for j in range(len(waypoints) - 1):
                t1, t2 = waypoints[j, 2], waypoints[j + 1, 2]
                if t1 <= t_clamped < t2:
                    alpha = (t_clamped - t1) / (t2 - t1)
                    obs_x[i] = (1 - alpha) * waypoints[j, 0] + alpha * waypoints[j + 1, 0]
                    obs_y[i] = (1 - alpha) * waypoints[j, 1] + alpha * waypoints[j + 1, 1]
                    break

    return obs_x, obs_y


def create_vehicle_triangle(x, y, theta, size=1.0):
    """Create triangle vertices representing vehicle orientation."""
    # Triangle pointing in heading direction
    front = np.array([x + size * np.cos(theta), y + size * np.sin(theta)])
    left_rear = np.array(
        [
            x - 0.5 * size * np.cos(theta) - 0.5 * size * np.sin(theta),
            y - 0.5 * size * np.sin(theta) + 0.5 * size * np.cos(theta),
        ]
    )
    right_rear = np.array(
        [
            x - 0.5 * size * np.cos(theta) + 0.5 * size * np.sin(theta),
            y - 0.5 * size * np.sin(theta) - 0.5 * size * np.cos(theta),
        ]
    )

    return np.array([front, left_rear, right_rear])


def animate_obstacle_avoidance(solution, save_filename="obstacle_avoidance_simple.mp4"):
    """
    Create a real-time animation with simplified and robust logic,
    including the vehicle's safety circle.

    Args:
        solution: MAPTOR solution object containing trajectory data.
        save_filename: Output video filename.
    """
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # 1. EXTRACT AND CLEAN DATA
    time_states = solution["time_states"]
    x_vehicle = solution["x_position"]
    y_vehicle = solution["y_position"]
    theta_vehicle = solution["heading"]

    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    x_sol = x_vehicle[unique_indices]
    y_sol = y_vehicle[unique_indices]
    theta_sol = theta_vehicle[unique_indices]

    # 2. CREATE ANIMATION GRID AND INTERPOLATE
    final_time = time_sol[-1]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    x_anim = np.interp(animation_time, time_sol, x_sol)
    y_anim = np.interp(animation_time, time_sol, y_sol)
    theta_anim = np.interp(animation_time, time_sol, theta_sol)

    obs_x_anim, obs_y_anim = create_obstacle_trajectory_numpy(animation_time)

    # 3. SET UP THE PLOT
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 25)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Dynamic Obstacle Avoidance")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    # Plot static elements
    ax.scatter(0, 0, c="g", s=100, marker="s", label="Start", zorder=5)
    ax.scatter(20, 20, c="b", s=150, marker="*", label="Goal", zorder=5)
    ax.plot(x_sol, y_sol, "b-", alpha=0.3, label="Vehicle Path")
    ax.plot(
        *create_obstacle_trajectory_numpy(time_sol),
        "r-",
        alpha=0.3,
        label="Obstacle Path",
    )

    # Define radii from the problem
    vehicle_radius = 1.5  # <-- ADDED: Vehicle radius for the circle
    obstacle_radius = 2.5

    # Initialize animated artists
    vehicle_triangle = Polygon([[0, 0], [0, 0], [0, 0]], facecolor="blue", alpha=0.8)
    ax.add_patch(vehicle_triangle)

    # <-- ADDED: Create the safety circle for the vehicle
    safety_circle = Circle(
        (0, 0),
        vehicle_radius,
        fill=False,
        edgecolor="blue",
        linestyle="--",
        alpha=0.6,
    )
    ax.add_patch(safety_circle)

    obstacle_circle = Circle((0, 0), obstacle_radius, facecolor="red", alpha=0.7)
    ax.add_patch(obstacle_circle)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    # 4. DEFINE THE ANIMATION LOOP
    def animate(frame):
        """This function is called for each frame of the animation."""
        # Update vehicle position and orientation
        triangle_verts = create_vehicle_triangle(
            x_anim[frame], y_anim[frame], theta_anim[frame], size=1.8
        )
        vehicle_triangle.set_xy(triangle_verts)

        # <-- ADDED: Update the safety circle's position to follow the vehicle
        safety_circle.center = (x_anim[frame], y_anim[frame])

        # Update obstacle position
        obstacle_circle.center = (obs_x_anim[frame], obs_y_anim[frame])

        # Update time display
        time_text.set_text(f"Time: {animation_time[frame]:.2f}s")

        # <-- MODIFIED: Return the safety_circle to be redrawn
        return vehicle_triangle, safety_circle, obstacle_circle, time_text

    # 5. CREATE AND SAVE THE ANIMATION
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=1000 / fps,  # ms per frame
        blit=True,
    )

    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps)
        print(f"✓ Animation saved successfully to {save_filename}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


def solve_obstacle_avoidance_problem():
    """Set up and solve the dynamic obstacle avoidance optimization problem."""
    print("Setting up dynamic obstacle avoidance problem...")

    # Problem setup
    problem = mtor.Problem("Challenging Dynamic Obstacle Avoidance")
    phase = problem.set_phase(1)

    # Variables
    t = phase.time(initial=0.0)
    x = phase.state("x_position", initial=0.0, final=20.0)
    y = phase.state("y_position", initial=0.0, final=20.0)
    theta = phase.state("heading", initial=0.0)
    v = phase.state("velocity", initial=1.0, boundary=(0.5, 8.0))
    delta = phase.control("steering_angle", boundary=(-0.5, 0.5))
    a = phase.control("acceleration", boundary=(-3.0, 3.0))

    # Parameters and Dynamics
    L = 2.5  # Wheelbase (m)
    vehicle_radius = 1.5  # Vehicle safety radius (m)
    obstacle_radius = 2.5  # Obstacle radius (m)

    # Bicycle dynamics model
    phase.dynamics(
        {
            x: v * ca.cos(theta),
            y: v * ca.sin(theta),
            theta: v * ca.tan(delta) / L,
            v: a,
        }
    )

    # Constraints
    obs_x, obs_y = create_challenging_obstacle_position(t)
    distance_squared = (x - obs_x) ** 2 + (y - obs_y) ** 2
    min_separation = vehicle_radius + obstacle_radius

    # Obstacle avoidance constraint
    phase.path_constraints(distance_squared >= min_separation**2)

    # Boundary constraints
    phase.path_constraints(x >= -5.0, x <= 25.0, y >= -5.0, y <= 25.0)

    # Objective: minimize time
    problem.minimize(t.final)

    # Mesh and initial guess
    phase.mesh([6, 6, 6], [-1.0, -1 / 3, 1 / 3, 1.0])

    # Create initial guess
    states_guess, controls_guess = [], []
    for N in [6, 6, 6]:
        tau = np.linspace(0, 1, N + 1)
        states_guess.append(
            np.array(
                [
                    20.0 * tau,  # Linear x trajectory
                    20.0 * tau,  # Linear y trajectory
                    np.ones(N + 1) * 0.785,  # Constant heading ~45°
                    np.ones(N + 1) * 3.0,  # Constant velocity
                ]
            )
        )
        controls_guess.append(np.array([np.zeros(N), np.zeros(N)]))  # Zero steering/acceleration

    problem.guess(
        phase_states={1: states_guess},
        phase_controls={1: controls_guess},
        phase_terminal_times={1: 12.0},
    )

    print("Solving optimization problem...")
    # Solve with adaptive mesh refinement
    solution = mtor.solve_adaptive(
        problem,
        error_tolerance=1e-4,
        max_iterations=20,
        min_polynomial_degree=3,
        max_polynomial_degree=9,
        nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000},
    )

    return solution


def main():
    """Main execution function."""
    print("=== Dynamic Obstacle Avoidance with Real-Time Animation ===\n")

    # Solve the optimization problem
    solution = solve_obstacle_avoidance_problem()

    # Check solution status
    if solution.status["success"]:
        print("✓ Optimal solution found!")
        final_time = solution.phases[1]["times"]["final"]
        objective = solution.status["objective"]
        print(f"  Optimal time: {final_time:.3f} seconds")
        print(f"  Objective value: {objective:.6f}")

        # Create and display animation
        print("\nCreating real-time animation...")
        try:
            anim = animate_obstacle_avoidance(solution, "obstacle_avoidance_solution.mp4")

            print("\n=== Animation Controls ===")
            print("- Close the plot window to continue")
            print("- Video file: obstacle_avoidance_solution.mp4")

            plt.show()

        except ValueError as e:
            if "duplicates" in str(e).lower():
                print(f"Animation error (time data issue): {e}")
                print("This can happen with certain mesh configurations.")
                print("Displaying static solution plot instead...")
            else:
                print(f"Animation error: {e}")
            solution.plot()
            plt.show()

        except Exception as e:
            print(f"Unexpected animation error: {e}")
            print("Displaying basic solution plot instead...")
            solution.plot()
            plt.show()

    else:
        print("✗ Optimization failed!")
        print(f"  Status: {solution.status['message']}")
        print("  Try adjusting initial guess or solver parameters")


if __name__ == "__main__":
    main()
