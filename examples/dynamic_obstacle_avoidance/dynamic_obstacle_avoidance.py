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

# Create module-level interpolants for efficiency and reliability
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
# VEHICLE PHYSICAL PARAMETERS (DYNAMIC BICYCLE MODEL)
# ============================================================================

# Tire parameters
C_alpha_f = 100000.0  # Front tire cornering stiffness (N/rad)
C_alpha_r = 100000.0  # Rear tire cornering stiffness (N/rad)

# Vehicle geometry
L_f = 1.2  # Distance from CG to front axle (m)
L_r = 1.3  # Distance from CG to rear axle (m)

# Vehicle inertial properties
m = 1500.0  # Vehicle mass (kg)
I_z = 2500.0  # Yaw moment of inertia (kg*m^2)

# Safety parameters
vehicle_radius = 1.5  # Vehicle safety radius (m)
obstacle_radius = 2.5  # Obstacle radius (m)
min_separation = vehicle_radius + obstacle_radius

# Numerical safety parameters
v_x_min = 0.1  # Minimum absolute longitudinal velocity for division safety (m/s)

# ============================================================================
# PROBLEM SETUP
# ============================================================================

problem = mtor.Problem("Dynamic Obstacle Avoidance - Dynamic Bicycle Model")
phase = problem.set_phase(1)

# ============================================================================
# STATE VARIABLES (DYNAMIC BICYCLE MODEL)
# ============================================================================

t = phase.time(initial=0.0)

# Global position states
x = phase.state("x_position", initial=0.0, final=20.0)
y = phase.state("y_position", initial=0.0, final=20.0)

# Vehicle orientation
theta = phase.state("heading", initial=np.pi / 4.0, final=np.pi / 9.0)

# Vehicle velocity states
v_x = phase.state("longitudinal_velocity", initial=10.0, boundary=(-20.0, 20.0))
v_y = phase.state("lateral_velocity", initial=0.0, boundary=(-15.0, 15.0))

# Vehicle angular velocity
r = phase.state("yaw_rate", initial=0.0, boundary=(-3.0, 3.0))

# ============================================================================
# CONTROL VARIABLES
# ============================================================================

delta_f = phase.control("steering_angle", boundary=(-0.5, 0.5))
a_x = phase.control("longitudinal_acceleration", boundary=(-8.0, 8.0))

# ============================================================================
# DYNAMIC BICYCLE MODEL IMPLEMENTATION
# ============================================================================

# Division-by-zero protection for longitudinal velocity
v_x_safe = ca.if_else(ca.fabs(v_x) < v_x_min, ca.if_else(v_x >= 0, v_x_min, -v_x_min), v_x)

# Dynamic bicycle model coefficients with division protection
A = -(C_alpha_f * ca.cos(delta_f) + C_alpha_r) / (m * v_x_safe)
B = (-L_f * C_alpha_f * ca.cos(delta_f) + L_r * C_alpha_r) / (I_z * v_x_safe)
C = (-L_f * C_alpha_f * ca.cos(delta_f) + L_r * C_alpha_r) / (m * v_x_safe)
D = -(L_f**2 * C_alpha_f * ca.cos(delta_f) + L_r**2 * C_alpha_r) / (I_z * v_x_safe)
E = (C_alpha_f * ca.cos(delta_f)) / m
F = (L_f * C_alpha_f * ca.cos(delta_f)) / I_z

# Complete dynamic bicycle model dynamics
phase.dynamics(
    {
        x: v_x * ca.cos(theta) - v_y * ca.sin(theta),
        y: v_x * ca.sin(theta) + v_y * ca.cos(theta),
        theta: r,
        v_x: a_x,
        v_y: A * v_y + C * r + E * delta_f,
        r: B * v_y + D * r + F * delta_f,
    }
)

# ============================================================================
# COLLISION AVOIDANCE CONSTRAINTS (PRESERVED FROM ORIGINAL)
# ============================================================================

obs_x, obs_y = obstacle_position(t)
distance_squared = (x - obs_x) ** 2 + (y - obs_y) ** 2

phase.path_constraints(distance_squared >= min_separation**2)

# ============================================================================
# WORKSPACE BOUNDS (PRESERVED FROM ORIGINAL)
# ============================================================================

# phase.path_constraints(x >= -10.0, x <= 30.0, y >= -10.0, y <= 30.0)

# ============================================================================
# OBJECTIVE FUNCTION (PRESERVED FROM ORIGINAL)
# ============================================================================

problem.minimize(t.final)

# ============================================================================
# MESH CONFIGURATION (PRESERVED FROM ORIGINAL)
# ============================================================================

phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

# ============================================================================
# INITIAL GUESS FOR DYNAMIC MODEL
# ============================================================================


# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-2,
    max_iterations=30,
    min_polynomial_degree=4,
    max_polynomial_degree=12,
    ode_solver_tolerance=1e-4,
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
    print(f"Dynamic bicycle model objective: {solution.status['objective']:.6f}")
    print(f"Mission time: {solution.status['total_mission_time']:.6f} seconds")

    # Final state verification
    x_final = solution[(1, "x_position")][-1]
    y_final = solution[(1, "y_position")][-1]
    v_x_final = solution[(1, "longitudinal_velocity")][-1]
    v_y_final = solution[(1, "lateral_velocity")][-1]

    print("Final state verification:")
    print(f"  Position: ({x_final:.2f}, {y_final:.2f}) m")
    print("  Target position: (20.0, 20.0) m")
    print(f"  Position error: {np.sqrt((x_final - 20.0) ** 2 + (y_final - 20.0) ** 2):.3f} m")
    print(f"  Final velocities: v_x = {v_x_final:.2f} m/s, v_y = {v_y_final:.2f} m/s")

    # Dynamic behavior analysis
    v_x_traj = solution[(1, "longitudinal_velocity")]
    v_y_traj = solution[(1, "lateral_velocity")]
    r_traj = solution[(1, "yaw_rate")]

    print("Dynamic behavior analysis:")
    print(f"  Longitudinal velocity range: [{v_x_traj.min():.2f}, {v_x_traj.max():.2f}] m/s")
    print(f"  Max lateral velocity: {abs(v_y_traj).max():.2f} m/s")
    print(f"  Max yaw rate: {abs(r_traj).max():.2f} rad/s")
    print(f"  Reverse motion: {'Yes' if v_x_traj.min() < 0 else 'No'}")
    print(f"  Significant drift: {'Yes' if abs(v_y_traj).max() > 2.0 else 'No'}")

    solution.plot()

else:
    print(f"Optimization failed: {solution.status['message']}")
