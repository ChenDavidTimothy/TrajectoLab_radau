from pathlib import Path

import highway_overtaking
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.interpolate import CubicSpline


def _extract_initial_guess_trajectory():
    """Extract initial guess trajectory for visualization."""
    right_lane = highway_overtaking.RIGHT_LANE_CENTER
    left_lane = highway_overtaking.LEFT_LANE_CENTER

    waypoints_t = np.array([0.0, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 1.0])
    waypoints_x = np.array(
        [
            right_lane,
            right_lane,
            (right_lane + left_lane) / 2,
            left_lane,
            left_lane,
            (right_lane + left_lane) / 2,
            right_lane,
            right_lane,
        ]
    )
    waypoints_y = np.array([0.0, 5.0, 10.0, 15.0, 25.0, 30.0, 33.0, 35.0])

    spline_x = CubicSpline(waypoints_t, waypoints_x, bc_type="clamped")
    spline_y = CubicSpline(waypoints_t, waypoints_y, bc_type="clamped")

    t_eval = np.linspace(0, 1, 100)
    x_guess = spline_x(t_eval)
    y_guess = spline_y(t_eval)

    # Scale to actual time
    final_time = 18.0  # From problem.guess terminal time
    t_scaled = t_eval * final_time

    return t_scaled, x_guess, y_guess


def create_vehicle_rectangle(x, y, theta, length=3.0, width=1.5):
    """Create rectangle representing vehicle with proper orientation."""
    corners_local = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

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
    """Animate highway overtaking scenario - highway view only."""
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # Extract and clean data
    time_states = solution["time_states"]
    x_vehicle = solution["x_position"]
    y_vehicle = solution["y_position"]
    theta_vehicle = solution["heading"]
    u_vehicle = solution["longitudinal_velocity"]
    v_vehicle = solution["lateral_velocity"]

    # Remove duplicate time points
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    x_sol = x_vehicle[unique_indices]
    y_sol = y_vehicle[unique_indices]
    theta_sol = theta_vehicle[unique_indices]
    u_sol = u_vehicle[unique_indices]
    v_sol = v_vehicle[unique_indices]

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

    # Create obstacle trajectories
    obs_x_1_anim, obs_y_1_anim, obs_x_2_anim, obs_y_2_anim = create_obstacle_trajectories_numpy(
        animation_time
    )

    # Set up figure - highway view only
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(
        highway_overtaking.HIGHWAY_LEFT_BOUNDARY - 1, highway_overtaking.HIGHWAY_RIGHT_BOUNDARY + 1
    )
    ax.set_ylim(highway_overtaking.HIGHWAY_BOTTOM - 2, highway_overtaking.HIGHWAY_TOP + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("Highway Overtaking", fontsize=16, fontweight="bold")
    ax.set_xlabel("Lateral Position (m)", fontsize=12)
    ax.set_ylabel("Longitudinal Position (m)", fontsize=12)

    # Draw highway infrastructure
    center_line = highway_overtaking.HIGHWAY_CENTER
    lane_width = highway_overtaking.LANE_WIDTH

    # Highway boundaries
    ax.axvline(
        x=highway_overtaking.HIGHWAY_LEFT_BOUNDARY,
        color="yellow",
        linewidth=4,
    )
    ax.axvline(x=center_line, color="white", linestyle="--", linewidth=2)
    ax.axvline(x=highway_overtaking.HIGHWAY_RIGHT_BOUNDARY, color="yellow", linewidth=4)

    # Lane center guidelines (faint)
    ax.axvline(
        x=highway_overtaking.RIGHT_LANE_CENTER, color="gray", linestyle=":", alpha=0.5, linewidth=1
    )
    ax.axvline(
        x=highway_overtaking.LEFT_LANE_CENTER, color="gray", linestyle=":", alpha=0.5, linewidth=1
    )

    # Lane markings (dashed center line segments)
    y_range = np.arange(highway_overtaking.HIGHWAY_BOTTOM, highway_overtaking.HIGHWAY_TOP, 6)
    for y_mark in y_range:
        ax.plot([center_line, center_line], [y_mark, y_mark + 3], "w--", linewidth=2, alpha=0.8)

    # Plot static elements
    ax.scatter(
        *highway_overtaking.AGENT_START,
        c="green",
        s=200,
        marker="s",
        zorder=10,
        edgecolor="black",
    )
    ax.scatter(
        *highway_overtaking.AGENT_END,
        c="blue",
        s=250,
        marker="*",
        zorder=10,
        edgecolor="black",
    )

    # Plot trajectories
    ax.plot(x_sol, y_sol, "b-", alpha=0.4, linewidth=3)

    # Plot obstacle paths
    obs1_x_full, obs1_y_full, obs2_x_full, obs2_y_full = create_obstacle_trajectories_numpy(
        time_sol
    )
    ax.plot(obs1_x_full, obs1_y_full, "r--", alpha=0.5, linewidth=2)
    ax.plot(
        obs2_x_full,
        obs2_y_full,
        "orange",
        linestyle="--",
        alpha=0.5,
        linewidth=2,
    )

    # Plot initial guess trajectory
    guess_times, guess_x, guess_y = _extract_initial_guess_trajectory()
    ax.plot(guess_x, guess_y, "g:", alpha=0.7, linewidth=2)

    # Initialize animated elements
    agent_vehicle = Polygon(
        [[0, 0], [0, 0], [0, 0], [0, 0]], facecolor="blue", edgecolor="navy", alpha=0.9, linewidth=2
    )
    ax.add_patch(agent_vehicle)

    obstacle_1 = Polygon(
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        facecolor="red",
        edgecolor="darkred",
        alpha=0.8,
        linewidth=2,
    )
    ax.add_patch(obstacle_1)

    obstacle_2 = Polygon(
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        facecolor="orange",
        edgecolor="darkorange",
        alpha=0.8,
        linewidth=2,
    )
    ax.add_patch(obstacle_2)

    # Information display
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        verticalalignment="top",
    )

    def animate(frame):
        """Animation function called for each frame."""
        current_time = animation_time[frame]

        # Update agent vehicle
        agent_corners = create_vehicle_rectangle(
            x_anim[frame], y_anim[frame], theta_anim[frame], 4.0, 2.0
        )
        agent_vehicle.set_xy(agent_corners)

        # Update obstacles
        obs1_corners = create_vehicle_rectangle(
            obs_x_1_anim[frame], obs_y_1_anim[frame], np.pi / 2, 4.0, 2.0
        )
        obstacle_1.set_xy(obs1_corners)

        obs2_corners = create_vehicle_rectangle(
            obs_x_2_anim[frame], obs_y_2_anim[frame], -np.pi / 2, 4.0, 2.0
        )
        obstacle_2.set_xy(obs2_corners)

        # Update information display
        speed = np.sqrt(u_anim[frame] ** 2 + v_anim[frame] ** 2)

        # Determine current lane
        if abs(x_anim[frame] - highway_overtaking.RIGHT_LANE_CENTER) < abs(
            x_anim[frame] - highway_overtaking.LEFT_LANE_CENTER
        ):
            current_lane = "Right"
        else:
            current_lane = "Left"

        info_display = (
            f"Time: {current_time:.1f}s\n"
            f"Lane: {current_lane}\n"
            f"Speed: {speed:.1f} m/s\n"
            f"Position: ({x_anim[frame]:.1f}, {y_anim[frame]:.1f})"
        )
        info_text.set_text(info_display)

        return agent_vehicle, obstacle_1, obstacle_2, info_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()

    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps, bitrate=2000)
        print(f"Highway overtaking animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


if __name__ == "__main__":
    solution = highway_overtaking.solution

    if solution.status["success"]:
        print("Creating highway overtaking animation...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "highway_overtaking.mp4"

        anim = animate_highway_overtaking(solution, str(output_file))

        # Print summary
        x_traj = solution["x_position"]
        print("\nOvertaking Summary:")
        print(f"  Mission time: {solution.status['total_mission_time']:.1f}s")
        print(
            f"  Max lateral deviation: {abs(x_traj - highway_overtaking.RIGHT_LANE_CENTER).max():.1f}m"
        )
        print(f"  Final position: ({x_traj[-1]:.1f}, {solution['y_position'][-1]:.1f})")

        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
