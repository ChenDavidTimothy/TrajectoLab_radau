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
    "box_color": "#dc2626",  # Red for the mass box
    "obstacle_color": "#8b5cf6",  # Purple for obstacle
    "safety_color": "#fbbf24",  # Yellow for safety boundary
}


# ============================================================================
# 3D Geometry Creation Functions
# ============================================================================


def _create_manipulator_3d_geometry(
    q1, q2, q3, l1=manipulator_3dof.l1, l2=manipulator_3dof.l2, l3=manipulator_3dof.l3
):
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


def _create_box_wireframe(center_pos, box_size=0.08):
    """Create simple wireframe box at end effector position."""
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


def _create_sphere_wireframe(center, radius, num_meridians=16, num_parallels=12):
    """Create wireframe sphere for obstacle visualization."""
    vertices = []
    edges = []

    # Generate vertices using spherical coordinates
    for i in range(num_parallels + 1):
        phi = np.pi * i / num_parallels  # Latitude angle
        for j in range(num_meridians):
            theta = 2 * np.pi * j / num_meridians  # Longitude angle

            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate meridian edges (longitude lines)
    for j in range(num_meridians):
        for i in range(num_parallels):
            start_idx = i * num_meridians + j
            end_idx = (i + 1) * num_meridians + j
            edges.append([start_idx, end_idx])

    # Generate parallel edges (latitude lines)
    for i in range(1, num_parallels):  # Skip poles
        for j in range(num_meridians):
            start_idx = i * num_meridians + j
            end_idx = i * num_meridians + ((j + 1) % num_meridians)
            edges.append([start_idx, end_idx])

    return vertices, edges


# ============================================================================
# Animation Function
# ============================================================================


def animate_manipulator_3dof(solution, save_filename="manipulator_3dof.mp4"):
    """
    Animate 3DOF manipulator trajectory with obstacle avoidance.

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

    # Remove duplicate time points
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    q1_sol = q1_traj[unique_indices]
    q2_sol = q2_traj[unique_indices]
    q3_sol = q3_traj[unique_indices]

    # Real-time animation setup
    final_time = solution.status["total_mission_time"]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    # Interpolate trajectories
    q1_anim = np.interp(animation_time, time_sol, q1_sol)
    q2_anim = np.interp(animation_time, time_sol, q2_sol)
    q3_anim = np.interp(animation_time, time_sol, q3_sol)

    # Setup figure with single 3D plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 10), facecolor=COLORS["background_dark"])
    ax = fig.add_subplot(111, projection="3d", facecolor=COLORS["background_dark"])

    # Configure 3D plot with equal aspect ratios
    workspace_limit = 0.8
    ax.set_xlim([-workspace_limit, workspace_limit])
    ax.set_ylim([-workspace_limit, workspace_limit])
    ax.set_zlim([0, workspace_limit * 2])
    ax.set_xlabel("X (m)", color=COLORS["text_light"])
    ax.set_ylabel("Y (m)", color=COLORS["text_light"])
    ax.set_zlabel("Z (m)", color=COLORS["text_light"])
    ax.set_title(
        "3DOF Manipulator with Obstacle Avoidance", color=COLORS["text_light"], fontsize=16
    )
    ax.tick_params(colors=COLORS["text_light"])
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratios for true sphere

    # Create static obstacle visualization using imported parameters
    obstacle_center = np.array(
        [
            manipulator_3dof.OBSTACLE_CENTER_X,
            manipulator_3dof.OBSTACLE_CENTER_Y,
            manipulator_3dof.OBSTACLE_CENTER_Z,
        ]
    )

    # Create obstacle sphere wireframe
    obstacle_vertices, obstacle_edges = _create_sphere_wireframe(
        obstacle_center, manipulator_3dof.OBSTACLE_RADIUS, num_meridians=16, num_parallels=12
    )

    # Create safety boundary sphere wireframe
    safety_radius = manipulator_3dof.OBSTACLE_RADIUS + manipulator_3dof.SAFETY_MARGIN
    safety_vertices, safety_edges = _create_sphere_wireframe(
        obstacle_center, safety_radius, num_meridians=12, num_parallels=8
    )

    # Add static obstacle sphere wireframe
    for edge in obstacle_edges:
        start_vertex = obstacle_vertices[edge[0]]
        end_vertex = obstacle_vertices[edge[1]]
        ax.plot(
            [start_vertex[0], end_vertex[0]],
            [start_vertex[1], end_vertex[1]],
            [start_vertex[2], end_vertex[2]],
            color=COLORS["obstacle_color"],
            linewidth=2,
            alpha=0.9,
        )

    # Add static safety boundary wireframe
    for edge in safety_edges:
        start_vertex = safety_vertices[edge[0]]
        end_vertex = safety_vertices[edge[1]]
        ax.plot(
            [start_vertex[0], end_vertex[0]],
            [start_vertex[1], end_vertex[1]],
            [start_vertex[2], end_vertex[2]],
            color=COLORS["safety_color"],
            linewidth=1,
            alpha=0.5,
            linestyle="--",
        )

    # Initialize animated elements
    (base_marker,) = ax.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["primary_red"],
        markersize=15,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
        zorder=10,
    )
    (joint1_marker,) = ax.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["blue"],
        markersize=12,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
        zorder=10,
    )
    (joint2_marker,) = ax.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["green"],
        markersize=10,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
        zorder=10,
    )

    # Links
    (base_link_line,) = ax.plot([], [], [], color=COLORS["primary_red"], linewidth=10)
    (upper_arm_line,) = ax.plot([], [], [], color=COLORS["blue"], linewidth=8)
    (forearm_line,) = ax.plot([], [], [], color=COLORS["green"], linewidth=6)

    # Mass box visualization
    box_lines = []
    for _ in range(12):  # 12 edges for a cube wireframe
        (line,) = ax.plot([], [], [], color=COLORS["box_color"], linewidth=4, alpha=0.9)
        box_lines.append(line)

    # End-effector trail
    (end_effector_trail,) = ax.plot([], [], [], color=COLORS["box_color"], linewidth=3, alpha=0.8)

    def animate(frame):
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

        # Update end-effector trail (1.5-second window)
        trail_frames = min(frame + 1, int(1.5 * fps))
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

        return (
            base_marker,
            joint1_marker,
            joint2_marker,
            *box_lines,
            base_link_line,
            upper_arm_line,
            forearm_line,
            end_effector_trail,
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
        print("Creating 3DOF manipulator animation with obstacle avoidance...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "manipulator_3dof.mp4"

        anim = animate_manipulator_3dof(solution, str(output_file))

        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
