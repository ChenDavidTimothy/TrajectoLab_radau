from pathlib import Path

# Import the solution by running the main problem
import manipulator_3dof
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
    "blue": "#3b82f6",
    "green": "#10b981",
    "orange": "#f59e0b",
    "grey": "#6b7280",
    "box_color": "#dc2626",  # Red for the mass box
}


# ============================================================================
# 3D Geometry Creation Functions
# ============================================================================


def _create_manipulator_3d_geometry(q1, q2, q3, l1=0.3, l2=0.4, l3=0.3):
    """Create 3D manipulator geometry from joint angles."""
    # Base joint (origin)
    base_pos = np.array([0.0, 0.0, 0.0])

    # Joint 1 position (top of base link)
    joint1_pos = np.array([0.0, 0.0, l1])

    # Joint 2 position (end of upper arm)
    joint2_pos = joint1_pos + np.array(
        [l2 * np.cos(q2) * np.cos(q1), l2 * np.cos(q2) * np.sin(q1), l2 * np.sin(q2)]
    )

    # End effector position (end of forearm)
    end_effector_pos = joint2_pos + np.array(
        [l3 * np.cos(q2 + q3) * np.cos(q1), l3 * np.cos(q2 + q3) * np.sin(q1), l3 * np.sin(q2 + q3)]
    )

    return base_pos, joint1_pos, joint2_pos, end_effector_pos


def _create_box_wireframe(center_pos, box_size=0.12):
    """Create simple wireframe box at end effector position."""
    # Box dimensions (larger for visibility)
    half_size = box_size / 2

    # Create 8 vertices of the cube
    vertices = (
        np.array(
            [
                [-half_size, -half_size, -half_size],  # 0
                [half_size, -half_size, -half_size],  # 1
                [half_size, half_size, -half_size],  # 2
                [-half_size, half_size, -half_size],  # 3
                [-half_size, -half_size, half_size],  # 4
                [half_size, -half_size, half_size],  # 5
                [half_size, half_size, half_size],  # 6
                [-half_size, half_size, half_size],  # 7
            ]
        )
        + center_pos
    )

    # Define edges for wireframe
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # Bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # Top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # Vertical edges
    ]

    return vertices, edges


def _create_workspace_boundary_3d(l1=0.3, l2=0.4, l3=0.3, num_points=50):
    """Create 3D workspace boundary surfaces."""
    max_reach = l2 + l3
    min_reach = abs(l2 - l3)

    # Create spherical coordinates
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta = np.linspace(0, np.pi, num_points // 2)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Outer workspace boundary (shifted up by l1)
    x_outer = max_reach * np.sin(theta_grid) * np.cos(phi_grid)
    y_outer = max_reach * np.sin(theta_grid) * np.sin(phi_grid)
    z_outer = l1 + max_reach * np.cos(theta_grid)

    # Inner workspace boundary (only upper hemisphere)
    theta_inner = np.linspace(0, np.pi / 2, num_points // 4)
    phi_inner_grid, theta_inner_grid = np.meshgrid(phi, theta_inner)
    x_inner = min_reach * np.sin(theta_inner_grid) * np.cos(phi_inner_grid)
    y_inner = min_reach * np.sin(theta_inner_grid) * np.sin(phi_inner_grid)
    z_inner = l1 + min_reach * np.cos(theta_inner_grid)

    return (x_outer, y_outer, z_outer), (x_inner, y_inner, z_inner)


def _create_ground_plane(size=1.0, num_points=10):
    """Create ground plane grid."""
    x = np.linspace(-size, size, num_points)
    y = np.linspace(-size, size, num_points)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros_like(x_grid)
    return x_grid, y_grid, z_grid


# ============================================================================
# Animation Function
# ============================================================================


def animate_manipulator_3dof(solution, save_filename="manipulator_3dof.mp4"):
    """
    Animate 3DOF manipulator trajectory with realistic mass box visualization.

    Args:
        solution: MAPTOR solution object
        save_filename: Output video filename

    Returns:
        matplotlib animation object
    """
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # Extract solution data
    time_states = solution["time_states"]
    q1_traj = solution["q1"]
    q2_traj = solution["q2"]
    q3_traj = solution["q3"]
    q1_dot_traj = solution["q1_dot"]
    q2_dot_traj = solution["q2_dot"]
    q3_dot_traj = solution["q3_dot"]

    # Remove duplicate time points
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    q1_sol = q1_traj[unique_indices]
    q2_sol = q2_traj[unique_indices]
    q3_sol = q3_traj[unique_indices]
    q1_dot_sol = q1_dot_traj[unique_indices]
    q2_dot_sol = q2_dot_traj[unique_indices]
    q3_dot_sol = q3_dot_traj[unique_indices]

    # Real-time animation setup
    final_time = solution.status["total_mission_time"]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    # Interpolate trajectories
    q1_anim = np.interp(animation_time, time_sol, q1_sol)
    q2_anim = np.interp(animation_time, time_sol, q2_sol)
    q3_anim = np.interp(animation_time, time_sol, q3_sol)
    q1_dot_anim = np.interp(animation_time, time_sol, q1_dot_sol)
    q2_dot_anim = np.interp(animation_time, time_sol, q2_dot_sol)
    q3_dot_anim = np.interp(animation_time, time_sol, q3_dot_sol)

    # Control data interpolation
    time_controls = solution["time_controls"]
    unique_control_indices = np.unique(time_controls, return_index=True)[1]
    time_control_sol = time_controls[unique_control_indices]
    tau1_sol = solution["tau1"][unique_control_indices]
    tau2_sol = solution["tau2"][unique_control_indices]
    tau3_sol = solution["tau3"][unique_control_indices]
    tau1_anim = np.interp(animation_time, time_control_sol, tau1_sol)
    tau2_anim = np.interp(animation_time, time_control_sol, tau2_sol)
    tau3_anim = np.interp(animation_time, time_control_sol, tau3_sol)

    # Setup figure and axes
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10), facecolor=COLORS["background_dark"])

    # 3D manipulator view (left, larger)
    ax_main = fig.add_subplot(121, projection="3d", facecolor=COLORS["background_dark"])

    # Torque plots (right, smaller)
    ax_torques = fig.add_subplot(122, facecolor=COLORS["background_dark"])

    # Configure 3D plot
    workspace_limit = 1.0
    ax_main.set_xlim([-workspace_limit, workspace_limit])
    ax_main.set_ylim([-workspace_limit, workspace_limit])
    ax_main.set_zlim([0, workspace_limit])
    ax_main.set_xlabel("X (m)", color=COLORS["text_light"])
    ax_main.set_ylabel("Y (m)", color=COLORS["text_light"])
    ax_main.set_zlabel("Z (m)", color=COLORS["text_light"])
    ax_main.set_title("3DOF Manipulator with 1kg Mass Box", color=COLORS["text_light"], fontsize=14)
    ax_main.tick_params(colors=COLORS["text_light"])
    ax_main.view_init(elev=20, azim=45)

    # Configure torque plot
    ax_torques.set_facecolor(COLORS["background_dark"])
    ax_torques.set_xlim(0, final_time)
    torque_min = min(min(tau1_sol), min(tau2_sol), min(tau3_sol)) - 2
    torque_max = max(max(tau1_sol), max(tau2_sol), max(tau3_sol)) + 2
    ax_torques.set_ylim(torque_min, torque_max)
    ax_torques.grid(True, alpha=0.3)
    ax_torques.set_title("Joint Torques", color=COLORS["text_light"], fontsize=12)
    ax_torques.set_xlabel("Time (s)", color=COLORS["text_light"])
    ax_torques.set_ylabel("Torque (N⋅m)", color=COLORS["text_light"])
    ax_torques.tick_params(colors=COLORS["text_light"])

    # Plot torque trajectories
    ax_torques.plot(
        time_control_sol,
        tau1_sol,
        color=COLORS["blue"],
        linewidth=2,
        alpha=0.7,
        label="τ₁ (Base Joint)",
    )
    ax_torques.plot(
        time_control_sol,
        tau2_sol,
        color=COLORS["green"],
        linewidth=2,
        alpha=0.7,
        label="τ₂ (Shoulder Joint)",
    )
    ax_torques.plot(
        time_control_sol,
        tau3_sol,
        color=COLORS["orange"],
        linewidth=2,
        alpha=0.7,
        label="τ₃ (Elbow Joint)",
    )
    ax_torques.legend(
        facecolor=COLORS["background_dark"],
        edgecolor=COLORS["text_light"],
        labelcolor=COLORS["text_light"],
    )

    # Add workspace boundaries
    outer_boundary, inner_boundary = _create_workspace_boundary_3d()
    ax_main.plot_surface(
        outer_boundary[0],
        outer_boundary[1],
        outer_boundary[2],
        alpha=0.1,
        color=COLORS["grey"],
    )
    ax_main.plot_surface(
        inner_boundary[0],
        inner_boundary[1],
        inner_boundary[2],
        alpha=0.05,
        color=COLORS["grey"],
    )

    # Add ground plane
    ground_x, ground_y, ground_z = _create_ground_plane()
    ax_main.plot_surface(ground_x, ground_y, ground_z, alpha=0.2, color=COLORS["text_light"])

    # Initialize animated elements
    # Joint markers (using plot for proper 3D updates)
    (base_marker,) = ax_main.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["primary_red"],
        markersize=12,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
        zorder=10,
    )
    (joint1_marker,) = ax_main.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["blue"],
        markersize=10,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
        zorder=10,
    )
    (joint2_marker,) = ax_main.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["green"],
        markersize=8,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
        zorder=10,
    )

    # Links
    (base_link_line,) = ax_main.plot([], [], [], color=COLORS["primary_red"], linewidth=8)
    (upper_arm_line,) = ax_main.plot([], [], [], color=COLORS["blue"], linewidth=6)
    (forearm_line,) = ax_main.plot([], [], [], color=COLORS["green"], linewidth=4)

    # Mass box visualization (wireframe for reliability)
    box_lines = []
    for _ in range(12):  # 12 edges for a cube wireframe
        (line,) = ax_main.plot([], [], [], color=COLORS["box_color"], linewidth=3, alpha=0.9)
        box_lines.append(line)

    # End-effector trail
    (end_effector_trail,) = ax_main.plot(
        [], [], [], color=COLORS["box_color"], linewidth=3, alpha=0.8, label="Mass box path"
    )

    # Torque markers
    (tau1_marker,) = ax_torques.plot([], [], "o", color=COLORS["blue"], markersize=8)
    (tau2_marker,) = ax_torques.plot([], [], "o", color=COLORS["green"], markersize=8)
    (tau3_marker,) = ax_torques.plot([], [], "o", color=COLORS["orange"], markersize=8)

    # State information text
    state_text = ax_main.text2D(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        fontsize=10,
        color=COLORS["text_light"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": COLORS["background_dark"], "alpha": 0.8},
        verticalalignment="top",
    )

    def animate(frame):
        current_time = animation_time[frame]

        # Get manipulator geometry
        base_pos, joint1_pos, joint2_pos, end_effector_pos = _create_manipulator_3d_geometry(
            q1_anim[frame], q2_anim[frame], q3_anim[frame]
        )

        # Update joint markers
        base_marker.set_data_3d([base_pos[0]], [base_pos[1]], [base_pos[2]])
        joint1_marker.set_data_3d([joint1_pos[0]], [joint1_pos[1]], [joint1_pos[2]])
        joint2_marker.set_data_3d([joint2_pos[0]], [joint2_pos[1]], [joint2_pos[2]])

        # Update links
        base_link_line.set_data_3d(
            [base_pos[0], joint1_pos[0]], [base_pos[1], joint1_pos[1]], [base_pos[2], joint1_pos[2]]
        )
        upper_arm_line.set_data_3d(
            [joint1_pos[0], joint2_pos[0]],
            [joint1_pos[1], joint2_pos[1]],
            [joint1_pos[2], joint2_pos[2]],
        )
        forearm_line.set_data_3d(
            [joint2_pos[0], end_effector_pos[0]],
            [joint2_pos[1], end_effector_pos[1]],
            [joint2_pos[2], end_effector_pos[2]],
        )

        # Update mass box wireframe at end effector
        vertices, edges = _create_box_wireframe(end_effector_pos)
        for i, edge in enumerate(edges):
            start_vertex = vertices[edge[0]]
            end_vertex = vertices[edge[1]]
            box_lines[i].set_data_3d(
                [start_vertex[0], end_vertex[0]],
                [start_vertex[1], end_vertex[1]],
                [start_vertex[2], end_vertex[2]],
            )

        # Update end-effector trail (2-second window)
        trail_frames = min(frame + 1, int(2.0 * fps))
        trail_start = max(0, frame + 1 - trail_frames)
        trail_positions = [
            _create_manipulator_3d_geometry(q1_anim[i], q2_anim[i], q3_anim[i])[3]
            for i in range(trail_start, frame + 1)
        ]
        if trail_positions:
            trail_x = [pos[0] for pos in trail_positions]
            trail_y = [pos[1] for pos in trail_positions]
            trail_z = [pos[2] for pos in trail_positions]
            end_effector_trail.set_data_3d(trail_x, trail_y, trail_z)

        # Update torque markers
        tau1_marker.set_data([current_time], [tau1_anim[frame]])
        tau2_marker.set_data([current_time], [tau2_anim[frame]])
        tau3_marker.set_data([current_time], [tau3_anim[frame]])

        # Update state information
        state_info = (
            f"Time: {current_time:.2f}s / {final_time:.2f}s\n"
            f"Base (q₁): {q1_anim[frame]:.3f} rad ({np.degrees(q1_anim[frame]):+6.1f}°)\n"
            f"Shoulder (q₂): {q2_anim[frame]:.3f} rad ({np.degrees(q2_anim[frame]):+6.1f}°)\n"
            f"Elbow (q₃): {q3_anim[frame]:.3f} rad ({np.degrees(q3_anim[frame]):+6.1f}°)\n"
            f"Joint Velocities: {q1_dot_anim[frame]:+5.2f}, {q2_dot_anim[frame]:+5.2f}, {q3_dot_anim[frame]:+5.2f} rad/s\n"
            f"Mass Box Position: ({end_effector_pos[0]:+5.3f}, {end_effector_pos[1]:+5.3f}, {end_effector_pos[2]:+5.3f}) m\n"
            f"Mass Box Load: 1.0 kg\n"
            f"Joint Torques: Base={tau1_anim[frame]:+5.1f}, Shoulder={tau2_anim[frame]:+5.1f}, Elbow={tau3_anim[frame]:+5.1f} N⋅m"
        )
        state_text.set_text(state_info)

        return (
            base_marker,
            joint1_marker,
            joint2_marker,
            *box_lines,
            base_link_line,
            upper_arm_line,
            forearm_line,
            end_effector_trail,
            tau1_marker,
            tau2_marker,
            tau3_marker,
            state_text,
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()

    # Save animation
    try:
        anim.save(save_filename, writer="ffmpeg", fps=fps, bitrate=3000)
        print(f"Animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    solution = manipulator_3dof.solution

    if solution.status["success"]:
        print("Creating 3DOF manipulator animation with realistic mass box...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "manipulator_3dof.mp4"

        anim = animate_manipulator_3dof(solution, str(output_file))

        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
