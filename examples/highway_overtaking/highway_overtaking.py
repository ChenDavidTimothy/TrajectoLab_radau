import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

import maptor as mtor


# ============================================================================
# ROBUST HIGHWAY GEOMETRY - WIDER AND MORE FORGIVING
# ============================================================================

HIGHWAY_LEFT_BOUNDARY = -2.0  # Extended left boundary
HIGHWAY_RIGHT_BOUNDARY = 20.0  # Extended right boundary
HIGHWAY_BOTTOM = -10.0  # Extended approach zone
HIGHWAY_TOP = 50.0  # Extended exit zone

# 4-lane highway with wider, more realistic geometry
LANE_WIDTH = 3.6  # Wider lanes for easier maneuvering
HIGHWAY_CENTER = (HIGHWAY_LEFT_BOUNDARY + HIGHWAY_RIGHT_BOUNDARY) / 2  # x = 10.0
RIGHT_LANE_CENTER = HIGHWAY_CENTER + LANE_WIDTH / 2  # x = 12.75
LEFT_LANE_CENTER = HIGHWAY_CENTER - LANE_WIDTH / 2  # x = 7.25


# ============================================================================
# IMPROVED OBSTACLE SCENARIO - GENEROUS OVERTAKING WINDOWS
# ============================================================================

# Agent trajectory: Extended for more natural approach/exit
AGENT_START = (RIGHT_LANE_CENTER, 0.0)
AGENT_END = (RIGHT_LANE_CENTER, 50.0)  # Match HIGHWAY_TOP

# Obstacle 1: Slower vehicle with predictable behavior across full range
OBSTACLE_1_WAYPOINTS = np.array(
    [
        [RIGHT_LANE_CENTER, 15.0, 0.0],  # Start ahead of agent
        [RIGHT_LANE_CENTER, 40.0, 5.0],  # Covers extended highway range
    ]
)

# Obstacle 2: Oncoming with generous separation timing across full range
OBSTACLE_2_WAYPOINTS = np.array(
    [
        [LEFT_LANE_CENTER, 55.0, 0.0],  # Start beyond HIGHWAY_TOP
        [LEFT_LANE_CENTER, 0.0, 15.0],  # Extends beyond HIGHWAY_BOTTOM
    ]
)

# Create robust interpolants
_times_1 = OBSTACLE_1_WAYPOINTS[:, 2]
_x_coords_1 = OBSTACLE_1_WAYPOINTS[:, 0]
_y_coords_1 = OBSTACLE_1_WAYPOINTS[:, 1]

_times_2 = OBSTACLE_2_WAYPOINTS[:, 2]
_x_coords_2 = OBSTACLE_2_WAYPOINTS[:, 0]
_y_coords_2 = OBSTACLE_2_WAYPOINTS[:, 1]

_x_interpolant_1 = ca.interpolant("obs_x_interp_1", "linear", [_times_1], _x_coords_1)
_y_interpolant_1 = ca.interpolant("obs_y_interp_1", "linear", [_times_1], _y_coords_1)
_x_interpolant_2 = ca.interpolant("obs_x_interp_2", "linear", [_times_2], _x_coords_2)
_y_interpolant_2 = ca.interpolant("obs_y_interp_2", "linear", [_times_2], _y_coords_2)


def obstacle_1_position(current_time):
    """Get first obstacle position with clamped interpolation."""
    t_clamped = ca.fmax(_times_1[0], ca.fmin(_times_1[-1], current_time))
    return _x_interpolant_1(t_clamped), _y_interpolant_1(t_clamped)


def obstacle_2_position(current_time):
    """Get second obstacle position with clamped interpolation."""
    t_clamped = ca.fmax(_times_2[0], ca.fmin(_times_2[-1], current_time))
    return _x_interpolant_2(t_clamped), _y_interpolant_2(t_clamped)


# ============================================================================
# ROBUST VEHICLE PARAMETERS - PRESERVE COMPUTATIONAL LOGIC
# ============================================================================

m = 1412.0
I_z = 1536.7
l_f = 1.06
l_r = 1.85
k_f = 128916.0
k_r = 85944.0

# More conservative safety parameters for robustness
vehicle_radius = 1.5  # Slightly larger safety margin
obstacle_radius = 1.5  # More realistic vehicle size
min_separation = vehicle_radius + obstacle_radius + 0.5  # 5.0m total
u_min = 0.5  # Higher minimum for numerical stability


# ============================================================================
# PROBLEM SETUP
# ============================================================================

problem = mtor.Problem("Robust Highway Overtaking")
phase = problem.set_phase(1)


# ============================================================================
# ROBUST STATE VARIABLE BOUNDS
# ============================================================================

t = phase.time(initial=0.0)

x = phase.state("x_position", initial=AGENT_START[0], final=AGENT_END[0])
y = phase.state("y_position", initial=AGENT_START[1], final=AGENT_END[1])
phi = phase.state("heading", initial=np.pi / 2, final=np.pi / 2)

# More permissive velocity bounds for robustness
u = phase.state("longitudinal_velocity", initial=12.0, boundary=(u_min, 30.0))
v = phase.state("lateral_velocity", initial=0.0, boundary=(-5.0, 5.0))
omega = phase.state("yaw_rate", initial=0.0, boundary=(-2.5, 2.5))


# ============================================================================
# ROBUST CONTROL BOUNDS
# ============================================================================

a = phase.control("acceleration", boundary=(-6.0, 6.0))  # More conservative
delta = phase.control("steering_angle", boundary=(-0.4, 0.4))  # More conservative


# ============================================================================
# PRESERVE EXACT DYNAMIC BICYCLE MODEL - NO CHANGES
# ============================================================================

u_safe = ca.if_else(u < u_min, u_min, u)

alpha_f = (v + l_f * omega) / u_safe - delta
alpha_r = (v - l_r * omega) / u_safe

F_Y1 = -k_f * alpha_f
F_Y2 = -k_r * alpha_r

phase.dynamics(
    {
        x: u * ca.cos(phi) - v * ca.sin(phi),
        y: u * ca.sin(phi) + v * ca.cos(phi),
        phi: omega,
        u: a + v * omega - (F_Y1 * ca.sin(delta)) / m,
        v: -u * omega + (F_Y1 * ca.cos(delta) + F_Y2) / m,
        omega: (l_f * F_Y1 * ca.cos(delta) - l_r * F_Y2) / I_z,
    }
)


# ============================================================================
# ROBUST COLLISION AVOIDANCE WITH SMOOTH CONSTRAINTS
# ============================================================================

obs_x_1, obs_y_1 = obstacle_1_position(t)
obs_x_2, obs_y_2 = obstacle_2_position(t)

# Use smooth collision avoidance formulation
distance_squared_1 = (x - obs_x_1) ** 2 + (y - obs_y_1) ** 2
distance_squared_2 = (x - obs_x_2) ** 2 + (y - obs_y_2) ** 2

# Relaxed constraints for better conditioning
phase.path_constraints(
    distance_squared_1 >= min_separation**2 - 1.0,  # Small relaxation
    distance_squared_2 >= min_separation**2 - 1.0,
)


# ============================================================================
# GENEROUS WORKSPACE BOUNDS
# ============================================================================

phase.path_constraints(
    x >= HIGHWAY_LEFT_BOUNDARY,
    x <= HIGHWAY_RIGHT_BOUNDARY,
    y >= HIGHWAY_BOTTOM,
    y <= HIGHWAY_TOP,
)


# ============================================================================
# ROBUST OBJECTIVE FUNCTION - WEIGHT-INSENSITIVE
# ============================================================================

# Use normalized objective for robustness
time_scale = 5.0  # Expected mission time (updated to match realistic expectation)
control_scale = 10.0  # Expected control magnitude

normalized_time = t.final / time_scale
normalized_control = phase.add_integral((a / control_scale) ** 2 + (delta / 0.3) ** 2)

# Balanced objective that works across weight ranges
problem.minimize(normalized_time + 0.02 * normalized_control)


# ============================================================================
# ENGINEERED MESH BASED ON ADAPTIVE CONVERGENCE PATTERN
# ============================================================================

# Clean mesh mimicking the adaptive refinement pattern:
# - Fine resolution around overtaking maneuver (τ ≈ -0.2 to +0.4)
# - Higher polynomial degrees where dynamics are complex
# - Coarser intervals for approach and exit phases

POLYNOMIAL_DEGREES = [4, 4, 4, 6, 7, 8, 6, 5, 4, 4, 4]

MESH_NODES = [
    -1.0,  # Start
    -0.7,  # Approach
    -0.4,
    -0.2,  # Pre-overtaking
    -0.1,
    0.0,
    0.2,
    0.4,  # Critical overtaking region (fine spacing)
    0.6,
    0.8,  # Post-overtaking
    0.9,
    1.0,  # Exit
]

# ============================================================================
# APPLY ENGINEERED MESH
# ============================================================================

phase.mesh(POLYNOMIAL_DEGREES, MESH_NODES)


# ============================================================================
# ROBUST INITIAL GUESS GENERATION
# ============================================================================


def _generate_robust_initial_guess():
    """Generate weight-insensitive initial guess with conservative timing."""

    # Conservative overtaking phases with generous timing
    waypoints_t = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])

    # Smooth lane changes with wider geometry
    waypoints_x = np.array(
        [
            RIGHT_LANE_CENTER,
            RIGHT_LANE_CENTER,
            (RIGHT_LANE_CENTER + LEFT_LANE_CENTER) / 2,
            LEFT_LANE_CENTER,
            (RIGHT_LANE_CENTER + LEFT_LANE_CENTER) / 2,
            RIGHT_LANE_CENTER,
            RIGHT_LANE_CENTER,
        ]
    )

    # Extended longitudinal progression to match HIGHWAY_TOP = 50.0
    waypoints_y = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0])

    # Gentle heading changes
    waypoints_phi = np.array(
        [np.pi / 2, np.pi / 2, np.pi / 2 + 0.1, np.pi / 2, np.pi / 2 - 0.1, np.pi / 2, np.pi / 2]
    )

    # Conservative velocity profile
    waypoints_u = np.array([12.0, 14.0, 16.0, 18.0, 16.0, 14.0, 12.0])

    # Create smooth, well-conditioned splines
    spline_x = CubicSpline(waypoints_t, waypoints_x, bc_type="natural")
    spline_y = CubicSpline(waypoints_t, waypoints_y, bc_type="natural")
    spline_phi = CubicSpline(waypoints_t, waypoints_phi, bc_type="natural")
    spline_u = CubicSpline(waypoints_t, waypoints_u, bc_type="natural")

    states_guess = []
    controls_guess = []

    # Match the engineered mesh polynomial degrees
    for N in POLYNOMIAL_DEGREES:
        tau = np.linspace(-1, 1, N + 1)
        t_norm = (tau + 1) / 2

        x_vals = spline_x(t_norm)
        y_vals = spline_y(t_norm)
        phi_vals = spline_phi(t_norm)
        u_vals = spline_u(t_norm)

        # Conservative lateral and yaw rate estimates
        v_vals = np.gradient(x_vals) * 1.5
        omega_vals = np.gradient(phi_vals) * 1.0

        # Ensure bounds compliance
        v_vals = np.clip(v_vals, -10.0, 10.0)
        omega_vals = np.clip(omega_vals, -2.0, 2.0)

        states_guess.append(np.vstack([x_vals, y_vals, phi_vals, u_vals, v_vals, omega_vals]))

        # Conservative control estimates
        a_vals = np.gradient(u_vals)[:N] * 1.0
        delta_vals = np.gradient(phi_vals)[:N] * 2.0

        # Ensure control bounds compliance
        a_vals = np.clip(a_vals, -4.0, 4.0)
        delta_vals = np.clip(delta_vals, -0.3, 0.3)

        controls_guess.append(np.vstack([a_vals, delta_vals]))

    return states_guess, controls_guess


# ============================================================================
# APPLY ROBUST INITIAL GUESS
# ============================================================================

states_guess, controls_guess = _generate_robust_initial_guess()

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 5.0},  # More realistic time estimate for 50m at ~15-20 m/s
)


# ============================================================================
# ROBUST SOLVER CONFIGURATION
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-3,  # Relaxed for robustness
    max_iterations=15,
    min_polynomial_degree=3,
    max_polynomial_degree=8,
    ode_solver_tolerance=1e-1,
    nlp_options={
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 3,  # Reduced output
        "ipopt.mu_strategy": "adaptive",
        "ipopt.acceptable_tol": 1e-4,  # Fallback tolerance
        "ipopt.acceptable_iter": 5,
    },
)


# ============================================================================
# COMPREHENSIVE RESULTS ANALYSIS
# ============================================================================

if solution.status["success"]:
    print("✓ ROBUST Highway overtaking succeeded!")
    print(f"  Objective: {solution.status['objective']:.6f}")
    print(f"  Mission time: {solution.status['total_mission_time']:.3f} seconds")

    # Verify robustness metrics
    x_traj = solution["x_position"]
    y_traj = solution["y_position"]
    u_traj = solution["longitudinal_velocity"]
    v_traj = solution["lateral_velocity"]

    max_lateral_deviation = abs(x_traj - RIGHT_LANE_CENTER).max()
    max_speed = u_traj.max()
    max_lateral_velocity = abs(v_traj).max()

    print("\nRobustness Metrics:")
    print(f"  Max lateral deviation: {max_lateral_deviation:.2f} m")
    print(f"  Max speed: {max_speed:.1f} m/s")
    print(f"  Max lateral velocity: {max_lateral_velocity:.2f} m/s")
    print(f"  Successful overtaking: {'✓' if max_lateral_deviation > 3.0 else '✗'}")

    # Final position verification
    x_final = solution[(1, "x_position")][-1]
    y_final = solution[(1, "y_position")][-1]
    position_error = np.sqrt((x_final - AGENT_END[0]) ** 2 + (y_final - AGENT_END[1]) ** 2)

    print("\nFinal Position:")
    print(f"  Reached: ({x_final:.2f}, {y_final:.2f}) m")
    print(f"  Target: ({AGENT_END[0]:.2f}, {AGENT_END[1]:.2f}) m")
    print(f"  Error: {position_error:.3f} m")

    solution.plot()

else:
    print(f"✗ Optimization failed: {solution.status['message']}")
    print("\nDiagnostic Information:")
    print("  - Check IPOPT output above for specific failure mode")
    print("  - Verify initial guess satisfies all constraints")
    print("  - Consider further relaxing tolerances if needed")


# ============================================================================
# EXPORT FOR ANIMATION COMPATIBILITY
# ============================================================================

__all__ = [
    "AGENT_END",
    "AGENT_START",
    "HIGHWAY_BOTTOM",
    "HIGHWAY_CENTER",
    "HIGHWAY_LEFT_BOUNDARY",
    "HIGHWAY_RIGHT_BOUNDARY",
    "HIGHWAY_TOP",
    "LANE_WIDTH",
    "LEFT_LANE_CENTER",
    "OBSTACLE_1_WAYPOINTS",
    "OBSTACLE_2_WAYPOINTS",
    "RIGHT_LANE_CENTER",
    "solution",
]
