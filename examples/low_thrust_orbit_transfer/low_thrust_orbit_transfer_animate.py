from pathlib import Path

import low_thrust_orbit_transfer
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# MAPTOR Color Palette (from highway_overtaking_animate.py)
# ============================================================================

COLORS = {
    "primary_red": "#991b1b",
    "primary_red_dark": "#7f1d1d",
    "primary_red_light": "#f87171",
    "background_dark": "#2d2d2d",
    "text_light": "#e5e7eb",
    "agent_blue": "#3b82f6",
    "obstacle_green": "#10b981",
    "obstacle_orange": "#f59e0b",
    "road_markings": "#e5e7eb",
    "lane_guides": "#6b7280",
}


# ============================================================================
# Physical Constants (from low_thrust_orbit_transfer.py)
# ============================================================================

MU = 1.407645794e16  # ft³/s²
RE = 20925662.73  # ft
P_SCALE = 1e7
L_SCALE = np.pi
T_SCALE = 1e4


# ============================================================================
# Orbital Mechanics Conversion Functions
# ============================================================================


def modified_equinoctial_to_cartesian(p_scaled, f, g, h, k, L_scaled):
    """Convert modified equinoctial elements to Cartesian position vector."""
    # Convert to physical units
    p = p_scaled * P_SCALE
    L = L_scaled * L_SCALE

    # Orbital mechanics calculations
    q = 1.0 + f * np.cos(L) + g * np.sin(L)
    r = p / q
    alpha2 = h * h - k * k
    s2 = 1 + h * h + k * k

    # Position components
    r1 = r / s2 * (np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L))
    r2 = r / s2 * (np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L))
    r3 = 2 * r / s2 * (h * np.sin(L) - k * np.cos(L))

    return np.array([r1, r2, r3])


def generate_orbit_points(p_scaled, f, g, h, k, num_points=100):
    """Generate points along an orbit for visualization."""
    L_values = np.linspace(0, 2 * np.pi, num_points)
    orbit_points = []

    for L in L_values:
        L_scaled = L / L_SCALE
        pos = modified_equinoctial_to_cartesian(p_scaled, f, g, h, k, L_scaled)
        orbit_points.append(pos)

    return np.array(orbit_points)


def calculate_thrust_direction(solution_data, frame_idx):
    """Calculate thrust direction from control variables."""
    u1 = solution_data["u1"][frame_idx]
    u2 = solution_data["u2"][frame_idx]
    u3 = solution_data["u3"][frame_idx]

    # Normalize to ensure unit vector
    magnitude = np.sqrt(u1**2 + u2**2 + u3**2)
    if magnitude > 1e-10:
        return np.array([u1, u2, u3]) / magnitude
    return np.array([0, 0, 0])


# ============================================================================
# Animation Setup
# ============================================================================


def animate_low_thrust_orbit_transfer(solution, save_filename="low_thrust_orbit_transfer.mp4"):
    """Create promotional animation for low thrust orbit transfer."""
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    print("Creating MAPTOR orbital mechanics animation...")

    # Extract solution data - states and controls have different time arrays
    time_states = solution["time_states"]
    time_controls = solution["time_controls"]

    # State variables
    p_scaled_traj = solution["p_scaled"]
    f_traj = solution["f"]
    g_traj = solution["g"]
    h_traj = solution["h"]
    k_traj = solution["k"]
    L_scaled_traj = solution["L_scaled"]
    mass_traj = solution["w"]

    # Control variables
    u1_traj = solution["u1"]
    u2_traj = solution["u2"]
    u3_traj = solution["u3"]

    # Remove duplicates from state timeline
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]

    # Interpolate controls to state timeline (MAPTOR uses different discretization points)
    u1_interp = np.interp(time_sol, time_controls, u1_traj)
    u2_interp = np.interp(time_sol, time_controls, u2_traj)
    u3_interp = np.interp(time_sol, time_controls, u3_traj)

    solution_data = {
        "p_scaled": p_scaled_traj[unique_indices],
        "f": f_traj[unique_indices],
        "g": g_traj[unique_indices],
        "h": h_traj[unique_indices],
        "k": k_traj[unique_indices],
        "L_scaled": L_scaled_traj[unique_indices],
        "mass": mass_traj[unique_indices],
        "u1": u1_interp,
        "u2": u2_interp,
        "u3": u3_interp,
    }

    # Animation parameters - convert scaled time to physical time
    final_time_scaled = solution.status["total_mission_time"]  # Scaled time units
    final_time_physical = final_time_scaled * T_SCALE  # Physical seconds
    fps = 30
    animation_duration_seconds = 15  # Make 15-second animation for LinkedIn
    total_frames = animation_duration_seconds * fps
    animation_indices = np.linspace(0, len(time_sol) - 1, total_frames, dtype=int)

    # Generate trajectory points
    trajectory_points = []
    for idx in animation_indices:
        pos = modified_equinoctial_to_cartesian(
            solution_data["p_scaled"][idx],
            solution_data["f"][idx],
            solution_data["g"][idx],
            solution_data["h"][idx],
            solution_data["k"][idx],
            solution_data["L_scaled"][idx],
        )
        trajectory_points.append(pos)
    trajectory_points = np.array(trajectory_points)

    # Generate initial and final orbits
    initial_orbit = generate_orbit_points(
        solution_data["p_scaled"][0],
        solution_data["f"][0],
        solution_data["g"][0],
        solution_data["h"][0],
        solution_data["k"][0],
    )

    final_orbit = generate_orbit_points(
        solution_data["p_scaled"][-1],
        solution_data["f"][-1],
        solution_data["g"][-1],
        solution_data["h"][-1],
        solution_data["k"][-1],
    )

    return _create_orbital_animation(
        trajectory_points,
        initial_orbit,
        final_orbit,
        solution_data,
        animation_indices,
        fps,
        total_frames,
        save_filename,
        final_time_physical,
    )


def _create_orbital_animation(
    trajectory_points,
    initial_orbit,
    final_orbit,
    solution_data,
    animation_indices,
    fps,
    total_frames,
    save_filename,
    final_time_physical,
):
    """Create the 3D orbital animation with MAPTOR styling."""

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS["background_dark"])

    # Create 3D subplot
    ax = fig.add_subplot(111, projection="3d", facecolor=COLORS["background_dark"])
    ax.set_facecolor(COLORS["background_dark"])

    # Set axis properties
    max_radius = (
        max(
            np.max(np.linalg.norm(initial_orbit, axis=1)),
            np.max(np.linalg.norm(final_orbit, axis=1)),
        )
        * 1.1
    )
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([-max_radius, max_radius])

    # Remove axis for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    # Draw Earth with proper 3D depth - no manual z-order
    earth_radius = RE
    u = np.linspace(0, 2 * np.pi, 30)  # Fewer points for performance
    v = np.linspace(0, np.pi, 30)
    earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
    earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
    earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        earth_x, earth_y, earth_z, color=COLORS["agent_blue"], alpha=0.9
    )  # Solid Earth for proper occlusion

    # Ensure equal aspect ratio for spherical Earth
    ax.set_box_aspect([1, 1, 1])

    # Draw initial orbit (ghost) - let matplotlib handle depth
    ax.plot(
        initial_orbit[:, 0],
        initial_orbit[:, 1],
        initial_orbit[:, 2],
        color=COLORS["lane_guides"],
        alpha=0.4,
        linewidth=2,
        linestyle="--",
        label="Initial Orbit",
    )

    # Draw final orbit (target) - let matplotlib handle depth
    ax.plot(
        final_orbit[:, 0],
        final_orbit[:, 1],
        final_orbit[:, 2],
        color=COLORS["obstacle_green"],
        alpha=0.6,
        linewidth=3,
        label="Target Orbit",
    )

    # Initialize animated elements - let matplotlib handle 3D depth
    (trajectory_line,) = ax.plot(
        [], [], [], color=COLORS["primary_red"], linewidth=4, alpha=0.9, label="Transfer Trajectory"
    )

    (spacecraft_point,) = ax.plot(
        [],
        [],
        [],
        "o",
        color=COLORS["primary_red_light"],
        markersize=12,
        markeredgecolor=COLORS["text_light"],
        markeredgewidth=2,
    )

    # Add title and mission info
    title_text = fig.suptitle(
        "MAPTOR: Low Thrust Orbit Transfer Optimization",
        fontsize=20,
        color=COLORS["text_light"],
        y=0.95,
    )

    # Mission statistics
    initial_mass = solution_data["mass"][0]
    final_mass = solution_data["mass"][-1]
    fuel_efficiency = (final_mass / initial_mass) * 100
    final_time_hours = final_time_physical / 3600  # final_time_physical is in seconds

    info_text = ax.text2D(
        0.02,
        0.98,
        f"Mission Duration: {final_time_hours:.1f} hours\n"
        f"Fuel Efficiency: {fuel_efficiency:.1f}%\n"
        f"Advanced Pseudospectral Method",
        transform=ax.transAxes,
        fontsize=12,
        color=COLORS["text_light"],
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=COLORS["background_dark"],
            alpha=0.8,
            edgecolor=COLORS["primary_red"],
        ),
    )

    # Progress indicator
    progress_text = ax.text2D(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=14,
        color=COLORS["primary_red_light"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=COLORS["background_dark"],
            alpha=0.8,
            edgecolor=COLORS["primary_red"],
        ),
    )

    def animate(frame):
        current_idx = animation_indices[frame]

        # Update trajectory - PERMANENT trail (show all points from start to current)
        trail_points = trajectory_points[: frame + 1]  # All points from start to current

        if len(trail_points) > 1:
            trajectory_line.set_data_3d(trail_points[:, 0], trail_points[:, 1], trail_points[:, 2])

        # Update spacecraft position
        current_pos = trajectory_points[frame]
        spacecraft_point.set_data_3d([current_pos[0]], [current_pos[1]], [current_pos[2]])

        # Update progress
        progress = (frame / total_frames) * 100
        current_mass = solution_data["mass"][current_idx]
        mass_used = ((initial_mass - current_mass) / initial_mass) * 100

        progress_text.set_text(f"Progress: {progress:.0f}%\nPropellant Used: {mass_used:.1f}%")

        return trajectory_line, spacecraft_point, progress_text

    # Set viewing angle for dramatic effect
    ax.view_init(elev=15, azim=45)

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()

    try:
        anim.save(
            save_filename, writer="ffmpeg", fps=fps, bitrate=3000, extra_args=["-vcodec", "libx264"]
        )
        print(f"Professional orbital animation saved to {Path(save_filename).resolve()}")
    except Exception as e:
        print(f"Could not save video file ({e}). Displaying animation instead.")

    return anim


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    solution = low_thrust_orbit_transfer.solution

    if solution.status["success"]:
        print("Creating professional MAPTOR orbital mechanics animation for LinkedIn...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "low_thrust_orbit_transfer_promo.mp4"

        anim = animate_low_thrust_orbit_transfer(solution, str(output_file))

        # Print mission summary for LinkedIn post
        final_time_hours = (
            solution.status["total_mission_time"] * T_SCALE
        ) / 3600  # Convert scaled time to hours
        initial_mass = solution["w"][0]
        final_mass = solution["w"][-1]
        fuel_efficiency = (final_mass / initial_mass) * 100

        print("\n=== LINKEDIN POST CONTENT ===")
        print("🚀 MAPTOR: Advanced Trajectory Optimization")
        print(f"⏱️ Mission: {final_time_hours:.1f} hour orbit transfer")
        print(f"⚡ Fuel Efficiency: {fuel_efficiency:.1f}%")
        print("🎯 Method: Legendre-Gauss-Radau Pseudospectral")
        print("💡 Features: Adaptive mesh, gravitational harmonics")

        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
