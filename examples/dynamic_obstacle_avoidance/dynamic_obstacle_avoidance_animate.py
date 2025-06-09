from pathlib import Path

# Import the solution by running the main problem
import dynamic_obstacle_avoidance as dynamic_obstacle_avoidance
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


def create_obstacle_trajectory_numpy(time_array):
    """Create obstacle trajectory using same waypoints as symbolic version.

    Args:
        time_array: NumPy array of time points

    Returns:
        Tuple of (obs_x, obs_y) NumPy arrays for obstacle positions
    """
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
    """Create triangle vertices representing vehicle orientation.

    Args:
        x: Vehicle x position
        y: Vehicle y position
        theta: Vehicle heading angle
        size: Triangle size scale factor

    Returns:
        NumPy array of triangle vertices
    """
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


def animate_obstacle_avoidance(solution, save_filename="obstacle_avoidance.mp4"):
    """Create real-time animation of the dynamic obstacle avoidance solution.

    Args:
        solution: MAPTOR solution object containing trajectory data
        save_filename: Output video filename

    Returns:
        matplotlib animation object
    """
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # Extract and clean data
    time_states = solution["time_states"]
    x_vehicle = solution["x_position"]
    y_vehicle = solution["y_position"]
    theta_vehicle = solution["heading"]

    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    x_sol = x_vehicle[unique_indices]
    y_sol = y_vehicle[unique_indices]
    theta_sol = theta_vehicle[unique_indices]

    # Create animation grid and interpolate
    final_time = time_sol[-1]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    x_anim = np.interp(animation_time, time_sol, x_sol)
    y_anim = np.interp(animation_time, time_sol, y_sol)
    theta_anim = np.interp(animation_time, time_sol, theta_sol)

    obs_x_anim, obs_y_anim = create_obstacle_trajectory_numpy(animation_time)

    # Set up the plot
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
    vehicle_radius = 1.5
    obstacle_radius = 2.5

    # Initialize animated artists
    vehicle_triangle = Polygon([[0, 0], [0, 0], [0, 0]], facecolor="blue", alpha=0.8)
    ax.add_patch(vehicle_triangle)

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

    def animate(frame):
        """Animation function called for each frame."""
        # Update vehicle position and orientation
        triangle_verts = create_vehicle_triangle(
            x_anim[frame], y_anim[frame], theta_anim[frame], size=1.8
        )
        vehicle_triangle.set_xy(triangle_verts)

        # Update safety circle position
        safety_circle.center = (x_anim[frame], y_anim[frame])

        # Update obstacle position
        obstacle_circle.center = (obs_x_anim[frame], obs_y_anim[frame])

        # Update time display
        time_text.set_text(f"Time: {animation_time[frame]:.2f}s")

        return vehicle_triangle, safety_circle, obstacle_circle, time_text

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=1000 / fps,
        blit=True,
    )

    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps)
        print(f"Animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


if __name__ == "__main__":
    # Use the solution from the imported module
    solution = dynamic_obstacle_avoidance.solution

    if solution.status["success"]:
        print("Creating animation...")

        # Save mp4 in the same directory as this script
        script_dir = Path(__file__).parent
        output_file = script_dir / "car_obstacle_avoidance.mp4"

        anim = animate_obstacle_avoidance(solution, str(output_file))
        plt.show()
    else:
        print("Cannot animate: solution failed")
