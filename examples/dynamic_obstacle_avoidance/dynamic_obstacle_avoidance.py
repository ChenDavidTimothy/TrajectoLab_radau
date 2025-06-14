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
# VEHICLE PHYSICAL PARAMETERS (FROM GE ET AL., 2021, TABLE I)
# ============================================================================

# Vehicle inertial properties
m = 1412.0  # Vehicle mass (kg)
I_z = 1536.7  # Yaw moment of inertia (kg*m^2)

# Vehicle geometry
l_f = 1.06  # Distance from CG to front axle (m)
l_r = 1.85  # Distance from CG to rear axle (m)

# Using conventional positive stiffness values with F = -k*alpha
k_f = 128916.0  # Front axle equivalent sideslip stiffness (N/rad)
k_r = 85944.0  # Rear axle equivalent sideslip stiffness (N/rad)

# Safety parameters
vehicle_radius = 1.5  # Vehicle safety radius (m)
obstacle_radius = 2.5  # Obstacle radius (m)
min_separation = vehicle_radius + obstacle_radius

# Numerical safety parameters
u_min = 0.05  # Minimum longitudinal velocity allowed (m/s)


# ============================================================================
# PROBLEM SETUP
# ============================================================================

problem = mtor.Problem("Dynamic Bicycle Model - Forward Motion Only")
phase = problem.set_phase(1)


# ============================================================================
# STATE VARIABLES (X = [x, y, φ, u, v, ω])
# ============================================================================

t = phase.time(initial=0.0)

# Global position states (inertial frame)
x = phase.state("x_position", initial=0.0, final=20.0)
y = phase.state("y_position", initial=0.0, final=20.0)

# Vehicle orientation (heading angle φ)
phi = phase.state("heading", initial=np.pi / 3)

# Vehicle velocity states (body frame - u, v, ω)
# --- FIX 1: CONSTRAIN STATE TO BE FORWARD-ONLY ---
# This guides the solver to the correct region of the solution space.
u = phase.state("longitudinal_velocity", initial=5.0, boundary=(u_min, 20.0))
v = phase.state("lateral_velocity", initial=0.0, boundary=(-15.0, 15.0))
omega = phase.state("yaw_rate", initial=0.0, boundary=(-3.0, 3.0))


# ============================================================================
# CONTROL VARIABLES (U = [a, δ])
# ============================================================================

a = phase.control("acceleration", boundary=(-8.0, 8.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))


# ============================================================================
# DYNAMIC BICYCLE MODEL IMPLEMENTATION (EQUATION (1) FROM GE ET AL., 2021)
# ============================================================================

# --- FIX 2: REINSTATE SAFEGUARD FOR ROBUSTNESS ---
# This `u_safe` logic prevents division by zero even if the solver makes
# a poor intermediate guess for `u` at a collocation point.
u_safe = ca.if_else(u < u_min, u_min, u)

# Sideslip angle calculations (implied by Eq. 3a, 3b)
alpha_f = (v + l_f * omega) / u_safe - delta  # Front tire sideslip angle
alpha_r = (v - l_r * omega) / u_safe  # Rear tire sideslip angle

# Tire force model using standard physics convention (F = -k*alpha)
F_Y1 = -k_f * alpha_f  # Front lateral tire force
F_Y2 = -k_r * alpha_r  # Rear lateral tire force

# Complete bicycle model dynamics
phase.dynamics(
    {
        # Global position kinematics (body to inertial frame transformation)
        x: u * ca.cos(phi) - v * ca.sin(phi),
        y: u * ca.sin(phi) + v * ca.cos(phi),
        # Vehicle orientation (yaw rate integration)
        phi: omega,
        # Longitudinal dynamics
        u: a + v * omega - (F_Y1 * ca.sin(delta)) / m,
        # Lateral dynamics
        v: -u * omega + (F_Y1 * ca.cos(delta) + F_Y2) / m,
        # Yaw dynamics
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

phase.path_constraints(x >= -40.0, x <= 40.0, y >= -40.0, y <= 40.0)


# ============================================================================
# OBJECTIVE FUNCTION (MINIMUM TIME)
# ============================================================================

control_effort = phase.add_integral(a**2 + delta**2)


problem.minimize(t.final + 0.01 * control_effort)


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
    min_polynomial_degree=3,
    max_polynomial_degree=10,
    ode_solver_tolerance=1e-1,
    nlp_options={
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1e-8,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 5,
    },
)

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

if solution.status["success"]:
    print(f"Forward-only model objective: {solution.status['objective']:.6f}")
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

    solution.plot()

else:
    print(f"Optimization failed: {solution.status['message']}")
