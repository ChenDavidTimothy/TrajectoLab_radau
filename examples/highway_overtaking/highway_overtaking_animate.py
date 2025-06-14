from pathlib import Path

import highway_overtaking
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def create_vehicle_rectangle(x, y, theta, length=3.0, width=1.5):
    """Create rectangle representing vehicle with proper orientation."""
    # Vehicle corners in local frame
    corners_local = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )

    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Transform to global frame
    corners_global = corners_local @ rotation_matrix.T + np.array([x, y])
    return corners_global


def create_obstacle_trajectories_numpy(time_array):
    """Create obstacle trajectories using waypoints from main problem file."""
    waypoints_1 = highway_overtaking.OBSTACLE_1_WAYPOINTS
    waypoints_2 = highway_overtaking.OBSTACLE_2_WAYPOINTS

    obs_x_1 = np.zeros_like(time_array)
    obs_y_1 = np.zeros_like(time_array)
    obs_x_2 = np.zeros_like(time_array)
    obs_y_2 = np.zeros_like(time_array)

    for i, t in enumerate(time_array):
        # Obstacle 1 trajectory
        t_clamped_1 = np.clip(t, waypoints_1[0, 2], waypoints_1[-1, 2])
        if t_clamped_1 >= waypoints_1[-1, 2]:
            obs_x_1[i] = waypoints_1[-1, 0]
            obs_y_1[i] = waypoints_1[-1, 1]
        else:
            alpha_1 = (t_clamped_1 - waypoints_1[0, 2]) / (waypoints_1[-1, 2] - waypoints_1[0, 2])
            obs_x_1[i] = (1 - alpha_1) * waypoints_1[0, 0] + alpha_1 * waypoints_1[-1, 0]
            obs_y_1[i] = (1 - alpha_1) * waypoints_1[0, 1] + alpha_1 * waypoints_1[-1, 1]

        # Obstacle 2 trajectory
        t_clamped_2 = np.clip(t, waypoints_2[0, 2], waypoints_2[-1, 2])
        if t_clamped_2 >= waypoints_2[-1, 2]:
            obs_x_2[i] = waypoints_2[-1, 0]
            obs_y_2[i] = waypoints_2[-1, 1]
        else:
            alpha_2 = (t_clamped_2 - waypoints_2[0, 2]) / (waypoints_2[-1, 2] - waypoints_2[0, 2])
            obs_x_2[i] = (1 - alpha_2) * waypoints_2[0, 0] + alpha_2 * waypoints_2[-1, 0]
            obs_y_2[i] = (1 - alpha_2) * waypoints_2[0, 1] + alpha_2 * waypoints_2[-1, 1]

    return obs_x_1, obs_y_1, obs_x_2, obs_y_2


def animate_highway_overtaking(solution, save_filename="highway_overtaking.mp4"):
    """Animate highway overtaking scenario with vertical orientation."""
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # Extract and clean data
    time_states = solution["time_states"]
    x_vehicle = solution["x_position"]
    y_vehicle = solution["y_position"]
    theta_vehicle = solution["heading"]
    u_vehicle = solution["longitudinal_velocity"]
    v_vehicle = solution["lateral_velocity"]
    omega_vehicle = solution["yaw_rate"]

    # Remove duplicate time points
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    x_sol = x_vehicle[unique_indices]
    y_sol = y_vehicle[unique_indices]
    theta_sol = theta_vehicle[unique_indices]
    u_sol = u_vehicle[unique_indices]
    v_sol = v_vehicle[unique_indices]
    omega_sol = omega_vehicle[unique_indices]

    # Create animation grid
    final_time = time_sol[-1]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    # Interpolate trajectories
    x_anim = np.interp(animation_time, time_sol, x_sol)
    y_anim = np.interp(animation_time, time_sol, y_sol)
    theta_anim = np.interp(animation_time, time_sol, theta_sol)
    u_anim = np.interp(animation_time, time_sol, u_sol)
    v_anim = np.interp(animation_time, time_sol, v_sol)
    omega_anim = np.interp(animation_time, time_sol, omega_sol)

    # Create obstacle trajectories
    obs_x_1_anim, obs_y_1_anim, obs_x_2_anim, obs_y_2_anim = create_obstacle_trajectories_numpy(
        animation_time
    )

    # Set up figure with highway-oriented layout
    fig = plt.figure(figsize=(16, 12))

    # Main highway view (larger, vertical orientation)
    ax_main = plt.subplot(2, 3, (1, 4))
    ax_main.set_xlim(3, 14)
    ax_main.set_ylim(-2, 22)
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.2)
    ax_main.set_title("Highway Overtaking Scenario", fontsize=14, fontweight="bold")
    ax_main.set_xlabel("Lateral Position (m)")
    ax_main.set_ylabel("Longitudinal Position (m)")

    # Draw highway lanes
    lane_width = 3.5
    center_line = 8.5
    right_lane_center = center_line + lane_width / 2
    left_lane_center = center_line - lane_width / 2

    # Lane boundaries
    ax_main.axvline(x=center_line - lane_width, color="yellow", linewidth=3, label="Highway Edge")
    ax_main.axvline(x=center_line, color="white", linestyle="--", linewidth=2, label="Lane Divider")
    ax_main.axvline(x=center_line + lane_width, color="yellow", linewidth=3, label="Highway Edge")

    # Lane markings (dashed lines every few meters)
    for y_mark in range(0, 21, 4):
        ax_main.plot(
            [center_line, center_line], [y_mark, y_mark + 2], "w--", linewidth=2, alpha=0.7
        )

    # Velocity plot
    ax_vel = plt.subplot(2, 3, 2)
    ax_vel.set_xlim(0, final_time)
    ax_vel.set_ylim(min(u_sol.min(), v_sol.min()) - 2, max(u_sol.max(), v_sol.max()) + 2)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_title("Vehicle Velocities")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")

    # Lateral position plot
    ax_lat = plt.subplot(2, 3, 3)
    ax_lat.set_xlim(0, final_time)
    ax_lat.set_ylim(x_sol.min() - 1, x_sol.max() + 1)
    ax_lat.grid(True, alpha=0.3)
    ax_lat.set_title("Lateral Position")
    ax_lat.set_xlabel("Time (s)")
    ax_lat.set_ylabel("X Position (m)")

    # Control inputs plot
    ax_ctrl = plt.subplot(2, 3, 5)
    if "acceleration" in solution.__dict__ or hasattr(solution, "acceleration"):
        accel = (
            solution["acceleration"] if "acceleration" in solution else np.zeros_like(time_sol[:-1])
        )
        steering = (
            solution["steering_angle"]
            if "steering_angle" in solution
            else np.zeros_like(time_sol[:-1])
        )
        time_controls = solution["time_controls"] if "time_controls" in solution else time_sol[:-1]

        ax_ctrl.set_xlim(0, final_time)
        ax_ctrl.set_ylim(min(accel.min(), steering.min()) - 1, max(accel.max(), steering.max()) + 1)
        ax_ctrl.plot(time_controls, accel, "b-", label="Acceleration", linewidth=2)
        ax_ctrl.plot(time_controls, steering, "r-", label="Steering", linewidth=2)
        ax_ctrl.legend()
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.set_title("Control Inputs")
    ax_ctrl.set_xlabel("Time (s)")

    # Yaw rate plot
    ax_yaw = plt.subplot(2, 3, 6)
    ax_yaw.set_xlim(0, final_time)
    ax_yaw.set_ylim(omega_sol.min() - 0.2, omega_sol.max() + 0.2)
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.set_title("Yaw Rate")
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.set_ylabel("Yaw Rate (rad/s)")

    # Plot static elements
    ax_main.scatter(
        *highway_overtaking.AGENT_START, c="green", s=150, marker="s", label="Start", zorder=10
    )
    ax_main.scatter(
        *highway_overtaking.AGENT_END, c="blue", s=200, marker="*", label="Goal", zorder=10
    )
    ax_main.plot(x_sol, y_sol, "b-", alpha=0.4, label="Agent Path", linewidth=3)

    # Plot obstacle paths
    obs1_x_full, obs1_y_full, obs2_x_full, obs2_y_full = create_obstacle_trajectories_numpy(
        time_sol
    )
    ax_main.plot(obs1_x_full, obs1_y_full, "r--", alpha=0.5, label="Obstacle 1 Path", linewidth=2)
    ax_main.plot(
        obs2_x_full,
        obs2_y_full,
        "orange",
        linestyle="--",
        alpha=0.5,
        label="Obstacle 2 Path",
        linewidth=2,
    )

    # Plot velocity and other trajectories
    ax_vel.plot(time_sol, u_sol, "b-", label="Longitudinal", linewidth=2)
    ax_vel.plot(time_sol, v_sol, "r-", label="Lateral", linewidth=2)
    ax_vel.legend()

    ax_lat.plot(time_sol, x_sol, "g-", linewidth=2)
    ax_lat.axhline(y=right_lane_center, color="gray", linestyle=":", alpha=0.7, label="Right Lane")
    ax_lat.axhline(y=left_lane_center, color="gray", linestyle=":", alpha=0.7, label="Left Lane")
    ax_lat.legend()

    ax_yaw.plot(time_sol, omega_sol, "purple", linewidth=2)

    # Initialize animated elements
    agent_vehicle = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], facecolor="blue", alpha=0.9)
    ax_main.add_patch(agent_vehicle)

    obstacle_1 = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], facecolor="red", alpha=0.8)
    ax_main.add_patch(obstacle_1)

    obstacle_2 = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], facecolor="orange", alpha=0.8)
    ax_main.add_patch(obstacle_2)

    # Time and state displays
    time_text = ax_main.text(
        0.02,
        0.95,
        "",
        transform=ax_main.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    state_text = ax_main.text(
        0.02,
        0.02,
        "",
        transform=ax_main.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
    )

    # Time markers on plots
    (time_marker_vel,) = ax_vel.plot([], [], "ko", markersize=8)
    (time_marker_lat,) = ax_lat.plot([], [], "ko", markersize=8)
    (time_marker_yaw,) = ax_yaw.plot([], [], "ko", markersize=8)

    ax_main.legend(loc="upper right", fontsize=10)

    def animate(frame):
        """Animation function called for each frame."""
        current_time = animation_time[frame]

        # Update agent vehicle
        agent_corners = create_vehicle_rectangle(x_anim[frame], y_anim[frame], theta_anim[frame])
        agent_vehicle.set_xy(agent_corners)

        # Update obstacles
        obs1_corners = create_vehicle_rectangle(
            obs_x_1_anim[frame], obs_y_1_anim[frame], np.pi / 2, 3.5, 1.8
        )
        obstacle_1.set_xy(obs1_corners)

        obs2_corners = create_vehicle_rectangle(
            obs_x_2_anim[frame], obs_y_2_anim[frame], -np.pi / 2, 3.5, 1.8
        )
        obstacle_2.set_xy(obs2_corners)

        # Update displays
        time_text.set_text(f"Time: {current_time:.2f}s")

        speed = np.sqrt(u_anim[frame] ** 2 + v_anim[frame] ** 2)
        lane_position = "Right" if x_anim[frame] > center_line else "Left"

        state_info = (
            f"Lane: {lane_position}\n"
            f"Speed: {speed:.1f} m/s\n"
            f"Y-pos: {y_anim[frame]:.1f} m\n"
            f"Lateral-v: {v_anim[frame]:.1f} m/s"
        )
        state_text.set_text(state_info)

        # Update time markers
        time_marker_vel.set_data([current_time], [np.interp(current_time, time_sol, u_sol)])
        time_marker_lat.set_data([current_time], [np.interp(current_time, time_sol, x_sol)])
        time_marker_yaw.set_data([current_time], [np.interp(current_time, time_sol, omega_sol)])

        return (
            agent_vehicle,
            obstacle_1,
            obstacle_2,
            time_text,
            state_text,
            time_marker_vel,
            time_marker_lat,
            time_marker_yaw,
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()

    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps, bitrate=1800)
        print(f"Highway overtaking animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


if __name__ == "__main__":
    # Use the solution from the imported module
    solution = highway_overtaking.solution

    if solution.status["success"]:
        print("Creating highway overtaking animation...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "highway_overtaking.mp4"

        anim = animate_highway_overtaking(solution, str(output_file))

        # Print overtaking behavior summary
        x_traj = solution["x_position"]
        y_traj = solution["y_position"]
        u_traj = solution["longitudinal_velocity"]

        print("\nOvertaking Behavior Summary:")
        print(f"  Lateral range: [{x_traj.min():.1f}, {x_traj.max():.1f}] m")
        print(f"  Max speed: {u_traj.max():.1f} m/s")
        print(f"  Lane changes: {'Yes' if (x_traj.max() - x_traj.min()) > 2.0 else 'No'}")
        print(f"  Distance traveled: {y_traj[-1] - y_traj[0]:.1f} m")

        center_line = 8.5
        lane_changes = np.sum(np.diff(np.sign(x_traj - center_line)) != 0)
        print(f"  Total lane changes: {lane_changes}")

        plt.show()

    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
