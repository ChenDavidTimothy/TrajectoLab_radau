from pathlib import Path

import cartpole_swingup
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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
}


# ============================================================================
# Geometry Creation Functions
# ============================================================================


def _create_cart_rectangle(x_pos, cart_width=0.3, cart_height=0.2):
    """Create cart rectangle centered at x_pos."""
    corners = np.array(
        [
            [x_pos - cart_width / 2, -cart_height / 2],
            [x_pos + cart_width / 2, -cart_height / 2],
            [x_pos + cart_width / 2, cart_height / 2],
            [x_pos - cart_width / 2, cart_height / 2],
        ]
    )
    return corners


def _create_pole_line(x_pos, theta, pole_length=0.5):
    """Create pole line from cart position at given angle."""
    # Pole goes from cart center to end of pole
    # theta=0 is upright, positive theta tilts left from cart perspective
    pole_end_x = x_pos - pole_length * np.sin(theta)
    pole_end_y = pole_length * np.cos(theta)

    return np.array([[x_pos, 0], [pole_end_x, pole_end_y]])


# ============================================================================
# Animation Function
# ============================================================================


def animate_cartpole_swingup(solution, save_filename="cartpole_swingup.mp4"):
    """
    Animate cartpole swingup trajectory.

    Args:
        solution: MAPTOR solution object
        save_filename: Output video filename

    Returns:
        matplotlib animation object
    """
    if not solution.status["success"]:
        raise ValueError("Cannot animate a failed solution.")

    # Extract and clean data
    time_states = solution["time_states"]
    x_cart = solution["x"]
    theta_pole = solution["theta"]
    x_dot_cart = solution["x_dot"]
    force = solution["F"]

    # Remove duplicate time points
    unique_indices = np.unique(time_states, return_index=True)[1]
    time_sol = time_states[unique_indices]
    x_sol = x_cart[unique_indices]
    theta_sol = theta_pole[unique_indices]
    x_dot_sol = x_dot_cart[unique_indices]

    # Animation parameters - real-time duration
    final_time = solution.status["total_mission_time"]
    fps = 30
    total_frames = int(final_time * fps)
    animation_time = np.linspace(0, final_time, total_frames)

    # Interpolate trajectories
    x_anim = np.interp(animation_time, time_sol, x_sol)
    theta_anim = np.interp(animation_time, time_sol, theta_sol)
    x_dot_anim = np.interp(animation_time, time_sol, x_dot_sol)

    # Interpolate force for display
    time_controls = solution["time_controls"]
    unique_control_indices = np.unique(time_controls, return_index=True)[1]
    time_control_sol = time_controls[unique_control_indices]
    force_sol = force[unique_control_indices]
    force_anim = np.interp(animation_time, time_control_sol, force_sol)

    # Setup plot
    plt.style.use("dark_background")
    fig, (ax_main, ax_force) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        facecolor=COLORS["background_dark"],
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Main animation plot
    ax_main.set_facecolor(COLORS["background_dark"])
    ax_main.set_xlim(-2, 2)
    ax_main.set_ylim(-1, 1)
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title("Cartpole Swingup", color=COLORS["text_light"], fontsize=14)
    ax_main.set_xlabel("Position (m)", color=COLORS["text_light"])
    ax_main.set_ylabel("Height (m)", color=COLORS["text_light"])
    ax_main.tick_params(colors=COLORS["text_light"])

    # Force plot
    ax_force.set_facecolor(COLORS["background_dark"])
    ax_force.set_xlim(0, final_time)
    ax_force.set_ylim(min(force_sol) - 1, max(force_sol) + 1)
    ax_force.grid(True, alpha=0.3)
    ax_force.set_title("Applied Force", color=COLORS["text_light"], fontsize=12)
    ax_force.set_xlabel("Time (s)", color=COLORS["text_light"])
    ax_force.set_ylabel("Force (N)", color=COLORS["text_light"])
    ax_force.tick_params(colors=COLORS["text_light"])

    # Plot complete force trajectory
    ax_force.plot(
        time_control_sol,
        force_sol,
        color=COLORS["agent_blue"],
        linewidth=2,
        alpha=0.7,
        label="Force trajectory",
    )
    ax_force.legend(
        facecolor=COLORS["background_dark"],
        edgecolor=COLORS["text_light"],
        labelcolor=COLORS["text_light"],
    )

    # Ground line
    ax_main.axhline(y=-0.5, color=COLORS["text_light"], linewidth=2, alpha=0.5)

    # Initialize animated elements
    cart_patch = Rectangle(
        (0, 0), 0, 0, facecolor=COLORS["agent_blue"], edgecolor=COLORS["text_light"], linewidth=2
    )
    ax_main.add_patch(cart_patch)

    (pole_line,) = ax_main.plot([], [], color=COLORS["primary_red"], linewidth=4)
    (pole_end_dot,) = ax_main.plot([], [], "o", color=COLORS["obstacle_orange"], markersize=8)

    # Trajectory trails
    (cart_trail_line,) = ax_main.plot(
        [], [], color=COLORS["obstacle_green"], linewidth=2, alpha=0.6, label="Cart trajectory"
    )
    (pole_trail_line,) = ax_main.plot(
        [], [], color=COLORS["obstacle_orange"], linewidth=2, alpha=0.8, label="Pole end trajectory"
    )

    # Force indicator (current time marker)
    (force_marker,) = ax_force.plot([], [], "o", color=COLORS["primary_red"], markersize=8)

    # State text display
    state_text = ax_main.text(
        0.02,
        0.95,
        "",
        transform=ax_main.transAxes,
        fontsize=10,
        color=COLORS["text_light"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": COLORS["background_dark"], "alpha": 0.8},
    )

    ax_main.legend(
        loc="upper right",
        facecolor=COLORS["background_dark"],
        edgecolor=COLORS["text_light"],
        labelcolor=COLORS["text_light"],
    )

    def animate(frame):
        current_time = animation_time[frame]

        # Update cart position
        cart_corners = _create_cart_rectangle(x_anim[frame])
        cart_patch.set_xy(cart_corners[0])  # Bottom-left corner
        cart_patch.set_width(cart_corners[1, 0] - cart_corners[0, 0])
        cart_patch.set_height(cart_corners[2, 1] - cart_corners[1, 1])

        # Update pole
        pole_coords = _create_pole_line(x_anim[frame], theta_anim[frame])
        pole_line.set_data(pole_coords[:, 0], pole_coords[:, 1])

        # Pole end dot
        pole_end_dot.set_data([pole_coords[1, 0]], [pole_coords[1, 1]])

        # Update trails
        # Cart trail (along ground)
        cart_trail_x = x_anim[: frame + 1]
        cart_trail_y = np.zeros_like(cart_trail_x)
        cart_trail_line.set_data(cart_trail_x, cart_trail_y)

        # Pole end trail (actual swinging path)
        pole_trail_x = x_anim[: frame + 1] - 0.5 * np.sin(theta_anim[: frame + 1])
        pole_trail_y = 0.5 * np.cos(theta_anim[: frame + 1])
        pole_trail_line.set_data(pole_trail_x, pole_trail_y)

        # Update force marker
        force_marker.set_data([current_time], [force_anim[frame]])

        # Update state information
        state_info = (
            f"Time: {current_time:.2f}s\n"
            f"Cart position: {x_anim[frame]:.3f} m\n"
            f"Pole angle: {theta_anim[frame]:.3f} rad ({np.degrees(theta_anim[frame]):.1f}°)\n"
            f"Cart velocity: {x_dot_anim[frame]:.3f} m/s\n"
            f"Applied force: {force_anim[frame]:.2f} N"
        )
        state_text.set_text(state_info)

        return (
            cart_patch,
            pole_line,
            pole_end_dot,
            cart_trail_line,
            pole_trail_line,
            force_marker,
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
    solution = cartpole_swingup.solution

    if solution.status["success"]:
        print("Creating cartpole swingup animation...")

        script_dir = Path(__file__).parent
        output_file = script_dir / "cartpole_swingup.mp4"

        anim = animate_cartpole_swingup(solution, str(output_file))

        # Print cartpole-specific verification
        x_traj = solution["x"]
        theta_traj = solution["theta"]
        force_traj = solution["F"]

        print("\nCartpole Performance Summary:")
        print(f"  Initial pole angle: {theta_traj[0]:.3f} rad ({np.degrees(theta_traj[0]):.1f}°)")
        print(f"  Final pole angle: {theta_traj[-1]:.3f} rad ({np.degrees(theta_traj[-1]):.1f}°)")
        print(f"  Final cart position: {x_traj[-1]:.6f} m")
        print(f"  Max cart displacement: {max(abs(x_traj)):.3f} m")
        print(f"  Max control force: {max(abs(force_traj)):.2f} N")
        print(f"  Swingup time: {solution.status['total_mission_time']:.3f} s")
        print(f"  Video duration: {solution.status['total_mission_time']:.3f} s (real-time)")

        plt.show()
    else:
        print("Cannot animate: solution failed")
        print(f"Failure message: {solution.status['message']}")
