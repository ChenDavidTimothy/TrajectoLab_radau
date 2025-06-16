# drone_animation.py
from pathlib import Path

# Import drone solution - drone.py is single source of truth
import drone
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class DroneAnimator:
    """3D drone animation using detailed eVTOL-style geometry from drone.py parameters."""

    def __init__(self, solution):
        self.solution = solution

        # Extract EXACT parameters from drone.py
        self.m = drone.m
        self.l_arm = drone.l_arm  # 0.19 m - this will be our evtol_radius equivalent
        self.K_T = drone.K_T
        self.omega_max = drone.omega_max

        # EXACT scaling factors from drone.py
        self.POS_SCALE = drone.POS_SCALE
        self.VEL_SCALE = drone.VEL_SCALE
        self.ANG_SCALE = drone.ANG_SCALE
        self.OMEGA_M_SCALE = drone.OMEGA_M_SCALE

        # Adapt eVTOL parameters to drone.py
        self.evtol_radius = self.l_arm  # Use arm length as size reference

        # Color palette from highway_overtaking_animate.py
        self.COLORS = {
            "primary_red": "#991b1b",
            "primary_red_light": "#f87171",
            "background_dark": "#2d2d2d",
            "text_light": "#e5e7eb",
            "agent_blue": "#3b82f6",
            "obstacle_green": "#10b981",
            "lane_guides": "#6b7280",
        }

    def _create_rotation_matrix(self, phi, theta, psi):
        """Create rotation matrix matching drone.py convention."""
        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])

        R_y = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        )

        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

        return R_z @ R_y @ R_x

    def update_drone_shape(self, x, y, z, phi, theta, psi, u_C, v_C, w_C):
        """
        Create detailed drone geometry adapted from eVTOL example.
        Uses drone.py parameters as single source of truth.
        """
        # Scale geometry based on drone.py arm length
        evtol_size = 2 * self.evtol_radius  # Total span
        arm_length = evtol_size / 2  # Individual arm length
        rotor_radius = arm_length / 3
        body_radius = arm_length / 2
        body_height = arm_length / 3

        # Create octagonal body (eVTOL style)
        n_sides = 8
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        body_points = []
        for angle in angles:
            body_points.append([body_radius * np.cos(angle), body_radius * np.sin(angle), 0])

        # Tapered body design
        taper_ratio = 0.8
        body_points_top = [
            [x[0] * taper_ratio, x[1] * taper_ratio, -body_height / 2] for x in body_points
        ]
        body_points_bottom = [
            [x[0] * taper_ratio, x[1] * taper_ratio, body_height / 2] for x in body_points
        ]

        # Arm dimensions
        arm_width = arm_length / 10
        arm_height = arm_width / 2

        # Start building complete geometry
        all_vertices = []
        all_vertices.extend(body_points_top)
        all_vertices.extend(body_points_bottom)
        faces = []

        def create_arm_vertices(start_point, end_point, width, height):
            """Create detailed arm geometry."""
            direction = np.array(end_point) - np.array(start_point)
            length = np.linalg.norm(direction)
            direction = direction / length

            up = np.array([0, 0, 1])
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)

            vertices = []
            # Bottom vertices
            vertices.append(start_point - right * width / 2 - up * height / 2)
            vertices.append(start_point + right * width / 2 - up * height / 2)
            vertices.append(end_point + right * (width / 2 * 0.7) - up * height / 2)
            vertices.append(end_point - right * (width / 2 * 0.7) - up * height / 2)

            # Top vertices
            vertices.append(start_point - right * width / 2 + up * height / 2)
            vertices.append(start_point + right * width / 2 + up * height / 2)
            vertices.append(end_point + right * (width / 2 * 0.7) + up * height / 2)
            vertices.append(end_point - right * (width / 2 * 0.7) + up * height / 2)

            return vertices

        # X-configuration arm positions (matching drone.py motor layout)
        arm_positions = [
            (
                [0, 0, 0],
                [arm_length * np.cos(np.pi / 4), arm_length * np.sin(np.pi / 4), 0],
            ),  # Front-right
            (
                [0, 0, 0],
                [arm_length * np.cos(3 * np.pi / 4), arm_length * np.sin(3 * np.pi / 4), 0],
            ),  # Front-left
            (
                [0, 0, 0],
                [arm_length * np.cos(-np.pi / 4), arm_length * np.sin(-np.pi / 4), 0],
            ),  # Rear-right
            (
                [0, 0, 0],
                [arm_length * np.cos(-3 * np.pi / 4), arm_length * np.sin(-3 * np.pi / 4), 0],
            ),  # Rear-left
        ]

        # Create arms
        arm_vertex_offset = len(all_vertices)
        for i, (start, end) in enumerate(arm_positions):
            arm_vertices = create_arm_vertices(
                np.array(start), np.array(end), arm_width, arm_height
            )
            all_vertices.extend(arm_vertices)

            idx = arm_vertex_offset + i * 8

            # Define arm faces
            arm_faces = [
                [idx, idx + 1, idx + 2, idx + 3],  # Bottom
                [idx + 4, idx + 5, idx + 6, idx + 7],  # Top
                [idx, idx + 1, idx + 5, idx + 4],  # Front
                [idx + 2, idx + 3, idx + 7, idx + 6],  # Back
                [idx, idx + 3, idx + 7, idx + 4],  # Left
                [idx + 1, idx + 2, idx + 6, idx + 5],  # Right
            ]
            faces.extend(arm_faces)

        # Create detailed rotors
        rotors = []
        n_rotor_points = 16
        rotor_height = arm_height / 2

        for start, end in arm_positions:
            x_center, y_center = end[0], end[1]

            hub_radius = rotor_radius * 0.2
            for j in range(n_rotor_points):
                rotor_angle = j * 2 * np.pi / n_rotor_points
                rotors.extend(
                    [
                        # Rotor disc points
                        [
                            x_center + rotor_radius * np.cos(rotor_angle),
                            y_center + rotor_radius * np.sin(rotor_angle),
                            rotor_height,
                        ],
                        [
                            x_center + rotor_radius * np.cos(rotor_angle),
                            y_center + rotor_radius * np.sin(rotor_angle),
                            -rotor_height,
                        ],
                        # Hub points
                        [
                            x_center + hub_radius * np.cos(rotor_angle),
                            y_center + hub_radius * np.sin(rotor_angle),
                            rotor_height * 1.5,
                        ],
                        [
                            x_center + hub_radius * np.cos(rotor_angle),
                            y_center + hub_radius * np.sin(rotor_angle),
                            -rotor_height * 1.5,
                        ],
                    ]
                )

        all_vertices.extend(rotors)

        # Body faces
        faces.append(list(range(n_sides, 2 * n_sides)))  # Top face
        faces.append(list(range(n_sides)))  # Bottom face

        # Body side faces
        for i in range(n_sides):
            faces.append([i, (i + 1) % n_sides, ((i + 1) % n_sides) + n_sides, i + n_sides])

        # Rotor faces
        rotor_vertex_offset = len(all_vertices) - len(rotors)
        points_per_rotor = n_rotor_points * 4

        for rotor_idx in range(4):
            current_rotor_idx = rotor_vertex_offset + rotor_idx * points_per_rotor

            for j in range(n_rotor_points):
                next_j = (j + 1) % n_rotor_points
                # Rotor disc faces
                faces.append(
                    [
                        current_rotor_idx + j * 2,
                        current_rotor_idx + next_j * 2,
                        current_rotor_idx + next_j * 2 + 1,
                        current_rotor_idx + j * 2 + 1,
                    ]
                )
                # Hub faces
                hub_start = current_rotor_idx + n_rotor_points * 2
                faces.append(
                    [
                        hub_start + j * 2,
                        hub_start + next_j * 2,
                        hub_start + next_j * 2 + 1,
                        hub_start + j * 2 + 1,
                    ]
                )

        # Transform all vertices
        all_vertices = np.array(all_vertices)
        R = self._create_rotation_matrix(phi, theta, psi)
        transformed_points = (R @ all_vertices.T).T + np.array([x, y, z])

        # Create final face list
        transformed_faces = [[transformed_points[idx] for idx in face] for face in faces]

        # Create front direction indicator
        front_direction = np.array(
            [[0, 0, 0], [arm_length * np.cos(np.pi / 4), arm_length * np.sin(np.pi / 4), 0]]
        )
        transformed_front = (R @ front_direction.T).T + np.array([x, y, z])

        return transformed_faces, transformed_front

    def animate_drone_flight(self, save_filename="drone_flight.mp4"):
        """Create drone flight animation with detailed eVTOL-style geometry."""
        if not self.solution.status["success"]:
            raise ValueError("Cannot animate failed solution")

        # Extract solution data with bounds checking
        time_states = self.solution["time_states"]
        n_points = len(time_states)

        print(
            f"Solution data: {n_points} points, time range: {time_states[0]:.3f} to {time_states[-1]:.3f}s"
        )

        # Convert ALL scaled values to physical
        X_physical = self.solution["X_scaled"] * self.POS_SCALE
        Y_physical = self.solution["Y_scaled"] * self.POS_SCALE
        Z_physical = self.solution["Z_scaled"] * self.POS_SCALE

        phi_physical = self.solution["phi_scaled"] * self.ANG_SCALE
        theta_physical = self.solution["theta_scaled"] * self.ANG_SCALE
        psi_physical = self.solution["psi_scaled"] * self.ANG_SCALE

        X_dot_physical = self.solution["X_dot_scaled"] * self.VEL_SCALE
        Y_dot_physical = self.solution["Y_dot_scaled"] * self.VEL_SCALE
        Z_dot_physical = self.solution["Z_dot_scaled"] * self.VEL_SCALE

        omega1_physical = self.solution["omega1_scaled"] * self.OMEGA_M_SCALE
        omega2_physical = self.solution["omega2_scaled"] * self.OMEGA_M_SCALE
        omega3_physical = self.solution["omega3_scaled"] * self.OMEGA_M_SCALE
        omega4_physical = self.solution["omega4_scaled"] * self.OMEGA_M_SCALE

        # CRITICAL FIX: Ensure arrays are same length
        min_length = min(
            len(X_physical),
            len(Y_physical),
            len(Z_physical),
            len(phi_physical),
            len(theta_physical),
            len(psi_physical),
            len(omega1_physical),
            len(omega2_physical),
            len(omega3_physical),
            len(omega4_physical),
        )

        print(
            f"Array lengths - X:{len(X_physical)}, phi:{len(phi_physical)}, omega1:{len(omega1_physical)}"
        )
        print(f"Using minimum length: {min_length}")

        # Truncate all arrays to same length
        X_physical = X_physical[:min_length]
        Y_physical = Y_physical[:min_length]
        Z_physical = Z_physical[:min_length]
        phi_physical = phi_physical[:min_length]
        theta_physical = theta_physical[:min_length]
        psi_physical = psi_physical[:min_length]
        X_dot_physical = X_dot_physical[:min_length]
        Y_dot_physical = Y_dot_physical[:min_length]
        Z_dot_physical = Z_dot_physical[:min_length]
        omega1_physical = omega1_physical[:min_length]
        omega2_physical = omega2_physical[:min_length]
        omega3_physical = omega3_physical[:min_length]
        omega4_physical = omega4_physical[:min_length]
        time_states = time_states[:min_length]

        # Animation parameters
        fps = 20
        total_time = time_states[-1]
        total_frames = min(100, min_length)  # Cap frames and ensure <= data points

        print(f"Animation: {total_frames} frames, {min_length} data points")

        # Setup 3D plot
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16, 12), facecolor=self.COLORS["background_dark"])
        ax = fig.add_subplot(111, projection="3d", facecolor=self.COLORS["background_dark"])

        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        ax.set_zlim(0, 25)

        ax.set_xlabel("X Position (m)", color=self.COLORS["text_light"], fontsize=12)
        ax.set_ylabel("Y Position (m)", color=self.COLORS["text_light"], fontsize=12)
        ax.set_zlabel("Z Position (m)", color=self.COLORS["text_light"], fontsize=12)
        ax.set_title(
            "DJI Mavic 3 Detailed Quadrotor Flight", color=self.COLORS["text_light"], fontsize=16
        )

        # Style
        ax.tick_params(colors=self.COLORS["text_light"])
        ax.grid(True, color=self.COLORS["lane_guides"], alpha=0.3)

        # Waypoints
        ax.scatter(
            5,
            5,
            5,
            c=self.COLORS["obstacle_green"],
            s=300,
            marker="s",
            label="Start",
            edgecolor=self.COLORS["text_light"],
            linewidth=2,
        )
        ax.scatter(
            20,
            20,
            20,
            c=self.COLORS["agent_blue"],
            s=400,
            marker="*",
            label="Target",
            edgecolor=self.COLORS["text_light"],
            linewidth=2,
        )

        # Flight path
        ax.plot(
            X_physical,
            Y_physical,
            Z_physical,
            color=self.COLORS["primary_red_light"],
            linewidth=3,
            alpha=0.8,
            label="Flight Path",
        )

        # Detailed drone visualization
        drone_mesh = Poly3DCollection(
            [],
            facecolor=self.COLORS["agent_blue"],
            alpha=0.9,
            edgecolor=self.COLORS["text_light"],
            linewidth=0.5,
        )
        ax.add_collection3d(drone_mesh)

        # Front direction indicator
        (front_line,) = ax.plot([], [], [], color="yellow", linewidth=3, alpha=0.9)

        # Trail
        trail_length = min(30, min_length // 3)
        (trail_line,) = ax.plot(
            [], [], [], color=self.COLORS["primary_red"], linewidth=4, alpha=0.9
        )

        # Info display
        info_text = ax.text2D(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=10,
            color=self.COLORS["text_light"],
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.COLORS["background_dark"],
                alpha=0.9,
                edgecolor=self.COLORS["primary_red"],
            ),
        )

        def animate(frame):
            # CRITICAL FIX: Proper index calculation with guaranteed bounds
            if total_frames <= 1:
                idx = 0
            else:
                # Map frame to data index, ensuring we never exceed bounds
                idx = int((frame / total_frames) * min_length)
                idx = min(idx, min_length - 1)  # Absolute guarantee of bounds

            # Current state (all physical units, bounds-checked)
            position = [X_physical[idx], Y_physical[idx], Z_physical[idx]]
            attitude = [phi_physical[idx], theta_physical[idx], psi_physical[idx]]
            velocities = [X_dot_physical[idx], Y_dot_physical[idx], Z_dot_physical[idx]]
            motor_speeds = [
                omega1_physical[idx],
                omega2_physical[idx],
                omega3_physical[idx],
                omega4_physical[idx],
            ]

            # Update detailed drone mesh
            faces, front_nose_line = self.update_drone_shape(
                position[0],
                position[1],
                position[2],
                attitude[0],
                attitude[1],
                attitude[2],
                velocities[0],
                velocities[1],
                velocities[2],
            )
            drone_mesh.set_verts(faces)
            front_line.set_data_3d(
                front_nose_line[:, 0], front_nose_line[:, 1], front_nose_line[:, 2]
            )

            # Update trail with bounds checking
            trail_start = max(0, idx - trail_length)
            trail_end = min(idx + 1, min_length)
            trail_x = X_physical[trail_start:trail_end]
            trail_y = Y_physical[trail_start:trail_end]
            trail_z = Z_physical[trail_start:trail_end]
            trail_line.set_data_3d(trail_x, trail_y, trail_z)

            # Calculate metrics
            current_time = time_states[idx]
            speed_3d = np.linalg.norm(velocities)
            distance_to_target = np.linalg.norm(np.array(position) - [20, 20, 20])

            # Motor info
            motor_rpm = [omega * 60 / (2 * np.pi) for omega in motor_speeds]
            avg_rpm = np.mean(motor_rpm)

            # Thrust calculation
            total_thrust = self.K_T * sum(omega**2 for omega in motor_speeds)
            thrust_to_weight = total_thrust / (self.m * 9.81)

            # Update display
            info_text.set_text(
                f"TIME: {current_time:.2f}s / {total_time:.2f}s\n"
                f"POS: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})m\n"
                f"ATT: R{np.degrees(attitude[0]):+.0f}° P{np.degrees(attitude[1]):+.0f}° Y{np.degrees(attitude[2]):+.0f}°\n"
                f"VEL: {speed_3d:.1f}m/s\n"
                f"DIST: {distance_to_target:.1f}m to target\n"
                f"RPM: {avg_rpm:.0f} avg ({motor_rpm[0]:.0f},{motor_rpm[1]:.0f},{motor_rpm[2]:.0f},{motor_rpm[3]:.0f})\n"
                f"T/W: {thrust_to_weight:.2f}\n"
                f"IDX: {idx}/{min_length - 1} FRAME: {frame}/{total_frames - 1}"
            )

            return drone_mesh, front_line, trail_line, info_text

        # Setup view
        ax.legend(loc="upper left")
        ax.view_init(elev=30, azim=45)

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames, interval=1000 / fps, blit=False, repeat=True
        )

        # Save with error handling
        try:
            print(f"Saving animation with {total_frames} frames...")
            anim.save(save_filename, writer="ffmpeg", fps=fps, bitrate=2000)
            print(f"✓ Animation saved to {Path(save_filename).resolve()}")
        except Exception as e:
            print(f"✗ Could not save video: {e}")
            print("Displaying animation instead...")

        return anim


def animate_drone_flight(save_filename="drone_flight_animation.mp4"):
    """Main function to create detailed drone flight animation."""
    solution = drone.solution

    if not solution.status["success"]:
        print("✗ Cannot animate: drone solution failed")
        print(f"Failure message: {solution.status['message']}")
        return None

    # Solution summary
    final_time = solution.status["total_mission_time"]
    print("\n✓ Drone Flight Solution Summary:")
    print(f"Flight time: {final_time:.2f} seconds")
    print(f"Objective: {solution.status['objective']:.6f}")

    # Position accuracy check
    X_final = solution["X_scaled"][-1] * drone.POS_SCALE
    Y_final = solution["Y_scaled"][-1] * drone.POS_SCALE
    Z_final = solution["Z_scaled"][-1] * drone.POS_SCALE

    target_error = np.sqrt((X_final - 20.0) ** 2 + (Y_final - 20.0) ** 2 + (Z_final - 20.0) ** 2)
    print(f"Final position: ({X_final:.2f}, {Y_final:.2f}, {Z_final:.2f})m")
    print(f"Target error: {target_error:.3f}m")

    print("\n🎬 Creating detailed 3D drone animation...")
    animator = DroneAnimator(solution)
    anim = animator.animate_drone_flight(save_filename)

    return anim


if __name__ == "__main__":
    anim = animate_drone_flight()
    if anim:
        import matplotlib.pyplot as plt

        plt.show()
