import casadi as ca
import numpy as np

import maptor as mtor


# ============================================================================
# OBSTACLE TRAJECTORY DEFINITION (PRESERVED FROM ORIGINAL)
# ============================================================================

OBSTACLE_WAYPOINTS = np.array(
    [
        [5.0, 5.0, 0.0],
        [12.0, 12.0, 3.0],
        [15.0, 15.0, 6.0],
        [20.0, 20.0, 12.0],
    ]
)

# Create module-level interpolants for obstacle trajectory
_times = OBSTACLE_WAYPOINTS[:, 2]
_x_coords = OBSTACLE_WAYPOINTS[:, 0]
_y_coords = OBSTACLE_WAYPOINTS[:, 1]

_x_interpolant = ca.interpolant("obs_x_interp", "linear", [_times], _x_coords)
_y_interpolant = ca.interpolant("obs_y_interp", "linear", [_times], _y_coords)


def obstacle_position(current_time):
    """Get obstacle position at given time with clamped interpolation."""
    t_clamped = ca.fmax(_times[0], ca.fmin(_times[-1], current_time))
    return _x_interpolant(t_clamped), _y_interpolant(t_clamped)


# ============================================================================
# VEHICLE PHYSICAL PARAMETERS (ENHANCED BICYCLE MODEL)
# ============================================================================

# Vehicle inertial properties
m = 1500.0  # Vehicle mass (kg)
I_z = 2500.0  # Yaw moment of inertia (kg*m^2)

# Vehicle geometry
l_f = 1.2  # Distance from CG to front axle (m)
l_r = 1.3  # Distance from CG to rear axle (m)

# Tire parameters (using k_f, k_r notation from mathematical model)
k_f = 100000.0  # Front tire cornering stiffness (N/rad)
k_r = 100000.0  # Rear tire cornering stiffness (N/rad)

# Safety parameters
vehicle_radius = 1.5  # Vehicle safety radius (m)
obstacle_radius = 2.5  # Obstacle radius (m)
min_separation = vehicle_radius + obstacle_radius

# Numerical safety parameters
u_min = 0.1  # Minimum absolute longitudinal velocity for safe division (m/s)


# ============================================================================
# PROBLEM SETUP
# ============================================================================

problem = mtor.Problem("Enhanced Dynamic Bicycle Model - Obstacle Avoidance")
phase = problem.set_phase(1)


# ============================================================================
# STATE VARIABLES (MATHEMATICAL MODEL FROM IMAGE)
# ============================================================================

t = phase.time(initial=0.0)

# Global position states (inertial frame)
x = phase.state("x_position", initial=0.0, final=20.0)
y = phase.state("y_position", initial=0.0, final=20.0)

# Vehicle orientation (heading angle φ in image)
phi = phase.state("heading", initial=np.pi / 4.0)

# Vehicle velocity states (body frame - u,v,ω from image)
u = phase.state("longitudinal_velocity", initial=10.0, boundary=(-20.0, 20.0))
v = phase.state("lateral_velocity", initial=0.0, boundary=(-15.0, 15.0))
omega = phase.state("yaw_rate", initial=0.0, boundary=(-3.0, 3.0))


# ============================================================================
# CONTROL VARIABLES (a, δ from mathematical model)
# ============================================================================

a = phase.control("acceleration", boundary=(-8.0, 8.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))


# ============================================================================
# ENHANCED BICYCLE MODEL IMPLEMENTATION (EQUATIONS FROM IMAGE)
# ============================================================================

# Safe division for longitudinal velocity (preserves sign for reverse motion)
u_safe = ca.if_else(ca.fabs(u) < u_min, ca.if_else(u >= 0, u_min, -u_min), u)

# Explicit sideslip angle calculations (Equations 3c, 3d from image)
alpha_f = (v + l_f * omega) / u_safe - delta  # Front tire sideslip angle
alpha_r = (v - l_r * omega) / u_safe  # Rear tire sideslip angle

# Linear tire force model (Equations 3c, 3d from image)
F_Y1 = k_f * alpha_f  # Front lateral tire force
F_Y2 = k_r * alpha_r  # Rear lateral tire force

# Complete bicycle model dynamics (Equation 1 from image)
phase.dynamics(
    {
        # Global position kinematics (body to inertial frame transformation)
        x: u * ca.cos(phi) - v * ca.sin(phi),
        y: u * ca.sin(phi) + v * ca.cos(phi),
        # Vehicle orientation (yaw rate integration)
        phi: omega,
        # Longitudinal dynamics (with lateral coupling and tire force)
        u: a + v * omega - (F_Y1 * ca.sin(delta)) / m,
        # Lateral dynamics (with longitudinal coupling)
        v: -u * omega + (F_Y1 * ca.cos(delta) + F_Y2) / m,
        # Yaw dynamics (tire force moments about CG)
        omega: (l_f * F_Y1 * ca.cos(delta) - l_r * F_Y2) / I_z,
    }
)


# ============================================================================
# COLLISION AVOIDANCE CONSTRAINTS
# ============================================================================

obs_x, obs_y = obstacle_position(t)
distance_squared = (x - obs_x) ** 2 + (y - obs_y) ** 2

phase.path_constraints(distance_squared >= min_separation**2)


# ============================================================================
# WORKSPACE BOUNDS
# ============================================================================

phase.path_constraints(x >= -10.0, x <= 25.0, y >= -10.0, y <= 25.0)


# ============================================================================
# OBJECTIVE FUNCTION (MINIMUM TIME)
# ============================================================================

problem.minimize(t.final)


# ============================================================================
# MESH CONFIGURATION
# ============================================================================

phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])


# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-1,
    max_iterations=30,
    min_polynomial_degree=4,
    max_polynomial_degree=12,
    ode_solver_tolerance=1e-2,
    nlp_options={
        "ipopt.print_level": 5,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-4,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.linear_solver": "mumps",
        "ipopt.mu_strategy": "adaptive",
    },
)


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

if solution.status["success"]:
    print(f"Enhanced bicycle model objective: {solution.status['objective']:.6f}")
    print(f"Mission time: {solution.status['total_mission_time']:.6f} seconds")

    # Final state verification
    x_final = solution[(1, "x_position")][-1]
    y_final = solution[(1, "y_position")][-1]
    u_final = solution[(1, "longitudinal_velocity")][-1]
    v_final = solution[(1, "lateral_velocity")][-1]
    omega_final = solution[(1, "yaw_rate")][-1]

    print("Final state verification:")
    print(f"  Position: ({x_final:.2f}, {y_final:.2f}) m")
    print("  Target position: (20.0, 20.0) m")
    print(f"  Position error: {np.sqrt((x_final - 20.0) ** 2 + (y_final - 20.0) ** 2):.3f} m")
    print(f"  Final velocities: u = {u_final:.2f} m/s, v = {v_final:.2f} m/s")
    print(f"  Final yaw rate: ω = {omega_final:.2f} rad/s")

    # Enhanced dynamic behavior analysis
    u_traj = solution[(1, "longitudinal_velocity")]
    v_traj = solution[(1, "lateral_velocity")]
    omega_traj = solution[(1, "yaw_rate")]

    print("\nEnhanced dynamic behavior analysis:")
    print(f"  Longitudinal velocity range: [{u_traj.min():.2f}, {u_traj.max():.2f}] m/s")
    print(f"  Max lateral velocity: {abs(v_traj).max():.2f} m/s")
    print(f"  Max yaw rate: {abs(omega_traj).max():.2f} rad/s")
    print(f"  Reverse motion: {'Yes' if u_traj.min() < 0 else 'No'}")
    print(f"  Significant lateral motion: {'Yes' if abs(v_traj).max() > 2.0 else 'No'}")

    # Calculate sideslip angles for analysis (using final trajectory)
    time_states = solution[(1, "time_states")]
    slip_angles = []
    for i in range(len(u_traj)):
        if abs(u_traj[i]) > 0.1:
            # Body frame slip angle β = arctan(v/u)
            slip_angle = np.arctan2(v_traj[i], u_traj[i]) * 180 / np.pi
            slip_angles.append(abs(slip_angle))

    if slip_angles:
        print(f"  Max body slip angle: {max(slip_angles):.1f}°")
        print(f"  High sideslip behavior: {'Yes' if max(slip_angles) > 10 else 'No'}")

    solution.plot()

else:
    print(f"Enhanced bicycle model optimization failed: {solution.status['message']}")
