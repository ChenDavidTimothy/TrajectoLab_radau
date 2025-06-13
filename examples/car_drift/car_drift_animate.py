from pathlib import Path

# Import the solution from the main problem file
import car_drift
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def create_vehicle_triangle(x, y, theta, size=0.5):
    """Create triangle representing vehicle with heading orientation."""
    # Triangle pointing in heading direction
    front = np.array([x + size * np.cos(theta), y + size * np.sin(theta)])
    left_rear = np.array(
        [
            x - 0.5 * size * np.cos(theta) - 0.3 * size * np.sin(theta),
            y - 0.5 * size * np.sin(theta) + 0.3 * size * np.cos(theta),
        ]
    )
    right_rear = np.array(
        [
            x - 0.5 * size * np.cos(theta) + 0.3 * size * np.sin(theta),
            y - 0.5 * size * np.sin(theta) - 0.3 * size * np.cos(theta),
        ]
    )

    return np.array([front, left_rear, right_rear])


def animate_car_drift(solution, save_filename="car_drift.mp4"):
    """Create animation of complex vehicle dynamics solution."""

    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # Extract solution data
    time_states = solution["time_states"]
    x_vehicle = solution["x_position"]
    y_vehicle = solution["y_position"]
    heading = solution["heading"]
    velocity = solution["velocity"]
    sideslip = solution["sideslip_angle"]
    yaw_rate = solution["yaw_rate"]

    # Remove duplicate time points for smooth animation
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    x_sol = x_vehicle[unique_indices]
    y_sol = y_vehicle[unique_indices]
    theta_sol = heading[unique_indices]
    v_sol = velocity[unique_indices]
    beta_sol = sideslip[unique_indices]
    gamma_sol = yaw_rate[unique_indices]

    # Create smooth animation timeline
    final_time = time_sol[-1]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    # Interpolate solution data
    x_anim = np.interp(animation_time, time_sol, x_sol)
    y_anim = np.interp(animation_time, time_sol, y_sol)
    theta_anim = np.interp(animation_time, time_sol, theta_sol)
    v_anim = np.interp(animation_time, time_sol, v_sol)
    beta_anim = np.interp(animation_time, time_sol, beta_sol)
    gamma_anim = np.interp(animation_time, time_sol, gamma_sol)

    # Set up the plot
    fig, (ax_main, ax_states) = plt.subplots(1, 2, figsize=(16, 8))

    # Main trajectory plot
    ax_main.set_xlim(-1, 11)
    ax_main.set_ylim(-1, 11)
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title("Complex Vehicle Dynamics - Trajectory")
    ax_main.set_xlabel("X Position (m)")
    ax_main.set_ylabel("Y Position (m)")

    # Plot static elements
    ax_main.scatter(0, 0, c="g", s=100, marker="s", label="Start", zorder=5)
    ax_main.scatter(10, 10, c="r", s=150, marker="*", label="Goal", zorder=5)
    ax_main.plot(x_sol, y_sol, "b-", alpha=0.4, linewidth=2, label="Planned Path")

    # Vehicle states plot
    ax_states.set_xlim(0, final_time)
    ax_states.set_ylim(-2, 20)
    ax_states.grid(True, alpha=0.3)
    ax_states.set_title("Vehicle States")
    ax_states.set_xlabel("Time (s)")
    ax_states.set_ylabel("Value")

    # Plot state trajectories
    ax_states.plot(time_sol, v_sol, "b-", label="Velocity (m/s)", alpha=0.7)
    ax_states.plot(time_sol, beta_sol * 10, "r-", label="Sideslip (×10 rad)", alpha=0.7)
    ax_states.plot(time_sol, gamma_sol * 5, "g-", label="Yaw Rate (×5 rad/s)", alpha=0.7)
    ax_states.legend()

    # Initialize animated elements
    vehicle_triangle = Polygon(
        [[0, 0], [0, 0], [0, 0]], facecolor="blue", edgecolor="darkblue", alpha=0.8, linewidth=2
    )
    ax_main.add_patch(vehicle_triangle)

    # Current position marker
    (current_pos,) = ax_main.plot([], [], "ro", markersize=8, label="Current Position")

    # State indicator lines
    v_line = ax_states.axvline(0, color="blue", linestyle="--", alpha=0.8)

    # Info text
    info_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax_main.legend(loc="lower right")

    def animate(frame):
        """Animation function called for each frame."""

        # Update vehicle triangle position and orientation
        triangle_verts = create_vehicle_triangle(
            x_anim[frame], y_anim[frame], theta_anim[frame], size=0.8
        )
        vehicle_triangle.set_xy(triangle_verts)

        # Update current position marker
        current_pos.set_data([x_anim[frame]], [y_anim[frame]])

        # Update state indicator line
        v_line.set_xdata([animation_time[frame], animation_time[frame]])

        # Update info text
        info_text.set_text(
            f"Time: {animation_time[frame]:.2f}s\n"
            f"Position: ({x_anim[frame]:.1f}, {y_anim[frame]:.1f})\n"
            f"Velocity: {v_anim[frame]:.1f} m/s\n"
            f"Heading: {theta_anim[frame] * 180 / np.pi:.1f}°\n"
            f"Sideslip: {beta_anim[frame] * 180 / np.pi:.1f}°\n"
            f"Yaw Rate: {gamma_anim[frame]:.2f} rad/s"
        )

        return vehicle_triangle, current_pos, v_line, info_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True, repeat=True
    )

    # Save animation
    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps, bitrate=1800)
        print(f"Animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


if __name__ == "__main__":
    # Use the solution from the imported module
    solution = car_drift.solution

    if solution.status["success"]:
        print("Creating car drift animation...")

        # Save mp4 in the same directory as this script
        script_dir = Path(__file__).parent
        output_file = script_dir / "car_drift.mp4"

        anim = animate_car_drift(solution, str(output_file))
        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Error: {solution.status['message']}")
