from pathlib import Path

import manipulator_2dof
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Colors
# ============================================================================

COLORS = {
    "primary_red": "#991b1b",
    "background_dark": "#2d2d2d",
    "text_light": "#e5e7eb",
    "agent_blue": "#3b82f6",
    "obstacle_green": "#10b981",
    "obstacle_orange": "#f59e0b",
    "lane_guides": "#6b7280",
}


# ============================================================================
# Geometry Creation Functions
# ============================================================================


def _create_manipulator_geometry(q1, q2, l1=0.5, l2=0.4):
    joint1_pos = np.array([0.0, 0.0])
    joint2_pos = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    end_effector_pos = np.array(
        [l1 * np.cos(q1) + l2 * np.cos(q1 + q2), l1 * np.sin(q1) + l2 * np.sin(q1 + q2)]
    )

    return joint1_pos, joint2_pos, end_effector_pos


def _create_workspace_boundary(l1=0.5, l2=0.4, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    max_reach = l1 + l2
    min_reach = abs(l1 - l2)

    outer_boundary_x = max_reach * np.cos(theta)
    outer_boundary_y = max_reach * np.sin(theta)
    inner_boundary_x = min_reach * np.cos(theta)
    inner_boundary_y = min_reach * np.sin(theta)

    return (outer_boundary_x, outer_boundary_y), (inner_boundary_x, inner_boundary_y)


# ============================================================================
# Animation Function
# ============================================================================


def animate_manipulator_2dof(solution, save_filename="manipulator_2dof.mp4"):
    """
    Animate 2DOF manipulator trajectory with real-time duration.

    Args:
        solution: MAPTOR solution object
        save_filename: Output video filename

    Returns:
        matplotlib animation object
    """
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    time_states = solution["time_states"]
    q1_traj = solution["q1"]
    q2_traj = solution["q2"]
    q1_dot_traj = solution["q1_dot"]
    q2_dot_traj = solution["q2_dot"]

    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    q1_sol = q1_traj[unique_indices]
    q2_sol = q2_traj[unique_indices]
    q1_dot_sol = q1_dot_traj[unique_indices]
    q2_dot_sol = q2_dot_traj[unique_indices]

    final_time = solution.status["total_mission_time"]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    q1_anim = np.interp(animation_time, time_sol, q1_sol)
    q2_anim = np.interp(animation_time, time_sol, q2_sol)
    q1_dot_anim = np.interp(animation_time, time_sol, q1_dot_sol)
    q2_dot_anim = np.interp(animation_time, time_sol, q2_dot_sol)

    time_controls = solution["time_controls"]
    unique_control_indices = np.unique(time_controls, return_index=True)[1]
    time_control_sol = time_controls[unique_control_indices]
    tau1_sol = solution["tau1"][unique_control_indices]
    tau2_sol = solution["tau2"][unique_control_indices]
    tau1_anim = np.interp(animation_time, time_control_sol, tau1_sol)
    tau2_anim = np.interp(animation_time, time_control_sol, tau2_sol)

    plt.style.use("dark_background")
    fig, (ax_main, ax_torques) = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        facecolor=COLORS["background_dark"],
        gridspec_kw={"width_ratios": [2, 1]},
    )

    ax_main.set_facecolor(COLORS["background_dark"])
    workspace_limit = 1.2
    ax_main.set_xlim(-workspace_limit, workspace_limit)
    ax_main.set_ylim(-workspace_limit, workspace_limit)
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title("2DOF Manipulator Motion", color=COLORS["text_light"], fontsize=14)
    ax_main.set_xlabel("X Position (m)", color=COLORS["text_light"])
    ax_main.set_ylabel("Y Position (m)", color=COLORS["text_light"])
    ax_main.tick_params(colors=COLORS["text_light"])

    ax_torques.set_facecolor(COLORS["background_dark"])
    ax_torques.set_xlim(0, final_time)
    torque_min = min(min(tau1_sol), min(tau2_sol)) - 2
    torque_max = max(max(tau1_sol), max(tau2_sol)) + 2
    ax_torques.set_ylim(torque_min, torque_max)
    ax_torques.grid(True, alpha=0.3)
    ax_torques.set_title("Joint Torques", color=COLORS["text_light"], fontsize=12)
    ax_torques.set_xlabel("Time (s)", color=COLORS["text_light"])
    ax_torques.set_ylabel("Torque (N⋅m)", color=COLORS["text_light"])
    ax_torques.tick_params(colors=COLORS["text_light"])

    ax_torques.plot(
        time_control_sol,
        tau1_sol,
        color=COLORS["agent_blue"],
        linewidth=2,
        alpha=0.7,
        label="τ₁ (Joint 1)",
    )
    ax_torques.plot(
        time_control_sol,
        tau2_sol,
        color=COLORS["obstacle_orange"],
        linewidth=2,
        alpha=0.7,
        label="τ₂ (Joint 2)",
    )
    ax_torques.legend(
        facecolor=COLORS["background_dark"],
        edgecolor=COLORS["text_light"],
        labelcolor=COLORS["text_light"],
    )

    outer_boundary, inner_boundary = _create_workspace_boundary()
    ax_main.plot(
        outer_boundary[0],
        outer_boundary[1],
        color=COLORS["lane_guides"],
        linewidth=1,
        alpha=0.3,
        linestyle="--",
    )
    ax_main.plot(
        inner_boundary[0],
        inner_boundary[1],
        color=COLORS["lane_guides"],
        linewidth=1,
        alpha=0.3,
        linestyle="--",
    )

    joint1_marker = ax_main.scatter(
        [],
        [],
        s=150,
        c=COLORS["primary_red"],
        marker="o",
        zorder=10,
        edgecolor=COLORS["text_light"],
        linewidth=2,
    )
    joint2_marker = ax_main.scatter(
        [],
        [],
        s=120,
        c=COLORS["agent_blue"],
        marker="o",
        zorder=10,
        edgecolor=COLORS["text_light"],
        linewidth=2,
    )
    end_effector_marker = ax_main.scatter(
        [],
        [],
        s=100,
        c=COLORS["obstacle_green"],
        marker="s",
        zorder=10,
        edgecolor=COLORS["text_light"],
        linewidth=2,
    )

    (link1_line,) = ax_main.plot(
        [], [], color=COLORS["agent_blue"], linewidth=6, solid_capstyle="round"
    )
    (link2_line,) = ax_main.plot(
        [], [], color=COLORS["obstacle_orange"], linewidth=6, solid_capstyle="round"
    )

    (end_effector_trail,) = ax_main.plot(
        [], [], color=COLORS["obstacle_green"], linewidth=2, alpha=0.8, label="End-effector path"
    )

    (tau1_marker,) = ax_torques.plot([], [], "o", color=COLORS["agent_blue"], markersize=8)
    (tau2_marker,) = ax_torques.plot([], [], "o", color=COLORS["obstacle_orange"], markersize=8)

    state_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        fontsize=10,
        color=COLORS["text_light"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": COLORS["background_dark"], "alpha": 0.8},
        verticalalignment="top",
    )

    ax_main.legend(
        loc="upper right",
        facecolor=COLORS["background_dark"],
        edgecolor=COLORS["text_light"],
        labelcolor=COLORS["text_light"],
    )

    def animate(frame):
        current_time = animation_time[frame]

        joint1_pos, joint2_pos, end_effector_pos = _create_manipulator_geometry(
            q1_anim[frame], q2_anim[frame]
        )

        joint1_marker.set_offsets([joint1_pos])
        joint2_marker.set_offsets([joint2_pos])
        end_effector_marker.set_offsets([end_effector_pos])

        link1_line.set_data([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]])
        link2_line.set_data(
            [joint2_pos[0], end_effector_pos[0]], [joint2_pos[1], end_effector_pos[1]]
        )

        trail_frames = min(frame + 1, int(2.0 * fps))
        trail_start = max(0, frame + 1 - trail_frames)
        trail_positions = [
            _create_manipulator_geometry(q1_anim[i], q2_anim[i])[2]
            for i in range(trail_start, frame + 1)
        ]
        if trail_positions:
            trail_x = [pos[0] for pos in trail_positions]
            trail_y = [pos[1] for pos in trail_positions]
            end_effector_trail.set_data(trail_x, trail_y)

        tau1_marker.set_data([current_time], [tau1_anim[frame]])
        tau2_marker.set_data([current_time], [tau2_anim[frame]])

        state_info = (
            f"Time: {current_time:.2f}s / {final_time:.2f}s\n"
            f"Joint 1: {q1_anim[frame]:.3f} rad ({np.degrees(q1_anim[frame]):+6.1f}°)\n"
            f"Joint 2: {q2_anim[frame]:.3f} rad ({np.degrees(q2_anim[frame]):+6.1f}°)\n"
            f"Joint 1 velocity: {q1_dot_anim[frame]:+6.3f} rad/s\n"
            f"Joint 2 velocity: {q2_dot_anim[frame]:+6.3f} rad/s\n"
            f"End-effector: ({end_effector_pos[0]:+5.3f}, {end_effector_pos[1]:+5.3f}) m\n"
            f"Torques: τ₁={tau1_anim[frame]:+6.2f}, τ₂={tau2_anim[frame]:+6.2f} N⋅m"
        )
        state_text.set_text(state_info)

        return (
            joint1_marker,
            joint2_marker,
            end_effector_marker,
            link1_line,
            link2_line,
            end_effector_trail,
            tau1_marker,
            tau2_marker,
            state_text,
        )

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()

    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps, bitrate=2000)
        print(f"Animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    solution = manipulator_2dof.solution

    if solution.status["success"]:
        print("Creating 2DOF manipulator animation...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "manipulator_2dof.mp4"

        anim = animate_manipulator_2dof(solution, str(output_file))

        q1_traj = solution["q1"]
        q2_traj = solution["q2"]
        tau1_traj = solution["tau1"]
        tau2_traj = solution["tau2"]

        print("\nManipulator Performance Summary:")
        print(
            f"  Initial configuration: q1={np.degrees(q1_traj[0]):.1f}°, q2={np.degrees(q2_traj[0]):.1f}°"
        )
        print(
            f"  Final configuration: q1={np.degrees(q1_traj[-1]):.1f}°, q2={np.degrees(q2_traj[-1]):.1f}°"
        )
        print(f"  Max joint 1 torque: {max(abs(tau1_traj)):.2f} N⋅m")
        print(f"  Max joint 2 torque: {max(abs(tau2_traj)):.2f} N⋅m")
        print(f"  Motion time: {solution.status['total_mission_time']:.3f} s")
        print(f"  Video duration: {solution.status['total_mission_time']:.3f} s (real-time)")

        initial_ee = np.array(
            [
                0.5 * np.cos(np.pi / 2) + 0.4 * np.cos(np.pi / 2 + 0),
                0.5 * np.sin(np.pi / 2) + 0.4 * np.sin(np.pi / 2 + 0),
            ]
        )
        final_ee = np.array(
            [
                0.5 * np.cos(q1_traj[-1]) + 0.4 * np.cos(q1_traj[-1] + q2_traj[-1]),
                0.5 * np.sin(q1_traj[-1]) + 0.4 * np.sin(q1_traj[-1] + q2_traj[-1]),
            ]
        )
        ee_distance = np.linalg.norm(final_ee - initial_ee)
        print(f"  End-effector travel distance: {ee_distance:.3f} m")

        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
