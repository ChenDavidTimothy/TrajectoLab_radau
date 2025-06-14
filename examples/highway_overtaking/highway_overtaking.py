import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

import maptor as mtor


# ============================================================================
# HIGHWAY GEOMETRY CONSTANTS FOR ANIMATION COMPATIBILITY
# ============================================================================

HIGHWAY_LEFT_BOUNDARY = 0.0
HIGHWAY_RIGHT_BOUNDARY = 18.0
HIGHWAY_BOTTOM = -5.0
HIGHWAY_TOP = 40.0

# 4-lane highway configuration
LANE_WIDTH = 4.5  # Wider lanes for truck/bus compatibility
HIGHWAY_CENTER = (HIGHWAY_LEFT_BOUNDARY + HIGHWAY_RIGHT_BOUNDARY) / 2  # x = 9.0
RIGHT_LANE_CENTER = HIGHWAY_CENTER + LANE_WIDTH / 2  # x = 11.25
LEFT_LANE_CENTER = HIGHWAY_CENTER - LANE_WIDTH / 2  # x = 6.75


# ============================================================================
# IMPROVED HIGHWAY OVERTAKING SCENARIO - WIDER AND MORE FORGIVING
# ============================================================================

# Agent trajectory: Bottom to top, more room for maneuvering
AGENT_START = (RIGHT_LANE_CENTER, 0.0)  # Start in right lane center
AGENT_END = (RIGHT_LANE_CENTER, 35.0)  # End in right lane center

# Obstacle 1: Same lane vehicle (slower, predictable)
OBSTACLE_1_WAYPOINTS = np.array(
    [
        [RIGHT_LANE_CENTER, 15.0, 0.0],  # Start in same lane as agent
        [RIGHT_LANE_CENTER, 30.0, 50.0],  # Slower vehicle - takes 20s to travel 15m
    ]
)

# Obstacle 2: Oncoming vehicle (creates overtaking window)
OBSTACLE_2_WAYPOINTS = np.array(
    [
        [LEFT_LANE_CENTER, 35.0, 0.0],  # Start in left lane
        [LEFT_LANE_CENTER, 0.0, 50.0],  # Faster oncoming - clears area by t=15s
    ]
)

# Create interpolants for both obstacles
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
# VEHICLE PHYSICAL PARAMETERS (PRESERVED FROM ORIGINAL)
# ============================================================================

m = 1412.0
I_z = 1536.7
l_f = 1.06
l_r = 1.85
k_f = 128916.0
k_r = 85944.0

vehicle_radius = 1.5
obstacle_radius = 2.5
min_separation = vehicle_radius + obstacle_radius
u_min = 0.05


# ============================================================================
# PROBLEM SETUP
# ============================================================================

problem = mtor.Problem("Improved Highway Overtaking")
phase = problem.set_phase(1)


# ============================================================================
# STATE VARIABLES
# ============================================================================

t = phase.time(initial=0.0)

x = phase.state("x_position", initial=AGENT_START[0], final=AGENT_END[0])
y = phase.state("y_position", initial=AGENT_START[1], final=AGENT_END[1])
phi = phase.state("heading", initial=np.pi / 2, final=np.pi / 2)  # Pointing upward
u = phase.state(
    "longitudinal_velocity", initial=15.0, boundary=(u_min, 35.0)
)  # Higher initial speed
v = phase.state("lateral_velocity", initial=0.0, boundary=(-20.0, 20.0))  # More lateral capability
omega = phase.state("yaw_rate", initial=0.0, boundary=(-4.0, 4.0))  # More agile turning


# ============================================================================
# CONTROL VARIABLES
# ============================================================================

a = phase.control("acceleration", boundary=(-10.0, 10.0))  # More aggressive acceleration
delta = phase.control("steering_angle", boundary=(-0.6, 0.6))  # More steering authority


# ============================================================================
# DYNAMIC BICYCLE MODEL (PRESERVED COMPUTATIONAL LOGIC)
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
# COLLISION AVOIDANCE CONSTRAINTS (PRESERVED LOGIC)
# ============================================================================

obs_x_1, obs_y_1 = obstacle_1_position(t)
distance_squared_1 = (x - obs_x_1) ** 2 + (y - obs_y_1) ** 2

obs_x_2, obs_y_2 = obstacle_2_position(t)
distance_squared_2 = (x - obs_x_2) ** 2 + (y - obs_y_2) ** 2

phase.path_constraints(
    distance_squared_1 >= min_separation**2, distance_squared_2 >= min_separation**2
)


# ============================================================================
# WIDER HIGHWAY BOUNDARIES - MORE FORGIVING GEOMETRY
# ============================================================================

phase.path_constraints(
    x >= HIGHWAY_LEFT_BOUNDARY,
    x <= HIGHWAY_RIGHT_BOUNDARY,
    y >= HIGHWAY_BOTTOM,
    y <= HIGHWAY_TOP,
)


# ============================================================================
# OBJECTIVE FUNCTION (PRESERVED)
# ============================================================================

control_effort = phase.add_integral(a**2 + delta**2)
problem.minimize(t.final + 0.02 * control_effort)  # Slightly more weight on control smoothness


# ============================================================================
# MESH CONFIGURATION
# ============================================================================

phase.mesh([8, 8, 8, 8], [-1.0, -0.5, 0.0, 0.5, 1.0])


# ============================================================================
# PHYSICS-BASED INITIAL GUESS GENERATION
# ============================================================================


def _generate_realistic_overtaking_guess():
    """Generate physics-based overtaking trajectory with proper lane change dynamics."""

    # Define realistic overtaking phases with proper timing
    # Phase 1: Approach (t=0-0.2) - accelerate in right lane
    # Phase 2: Lane change left (t=0.2-0.4) - move to left lane
    # Phase 3: Overtake (t=0.4-0.7) - pass obstacle in left lane
    # Phase 4: Lane change right (t=0.7-0.9) - return to right lane
    # Phase 5: Settle (t=0.9-1.0) - stabilize in right lane

    # Key positions (use defined lane centers for consistency)
    right_lane = RIGHT_LANE_CENTER  # Right lane center
    left_lane = LEFT_LANE_CENTER  # Left passing lane center

    # Time-based waypoints for smooth overtaking
    waypoints_t = np.array([0.0, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 1.0])
    waypoints_x = np.array(
        [
            right_lane,
            right_lane,
            (right_lane + left_lane) / 2,
            left_lane,
            left_lane,
            (right_lane + left_lane) / 2,
            right_lane,
            right_lane,
        ]
    )
    waypoints_y = np.array([0.0, 5.0, 10.0, 15.0, 25.0, 30.0, 33.0, 35.0])

    # Heading changes for lane changes
    waypoints_phi = np.array(
        [
            np.pi / 2,  # Straight up
            np.pi / 2,  # Straight up
            np.pi / 2 + 0.15,  # Slight left turn
            np.pi / 2,  # Straight up
            np.pi / 2,  # Straight up
            np.pi / 2 - 0.15,  # Slight right turn
            np.pi / 2,  # Straight up
            np.pi / 2,  # Straight up
        ]
    )

    # Velocity profile - accelerate during overtaking
    waypoints_u = np.array([15.0, 18.0, 22.0, 25.0, 25.0, 22.0, 18.0, 15.0])

    # Create smooth splines
    spline_x = CubicSpline(waypoints_t, waypoints_x, bc_type="clamped")
    spline_y = CubicSpline(waypoints_t, waypoints_y, bc_type="clamped")
    spline_phi = CubicSpline(waypoints_t, waypoints_phi, bc_type="clamped")
    spline_u = CubicSpline(waypoints_t, waypoints_u, bc_type="clamped")

    states_guess = []
    controls_guess = []

    for N in [8, 8, 8, 8]:
        tau = np.linspace(-1, 1, N + 1)
        t_norm = (tau + 1) / 2

        # Evaluate splines
        x_vals = spline_x(t_norm)
        y_vals = spline_y(t_norm)
        phi_vals = spline_phi(t_norm)
        u_vals = spline_u(t_norm)

        # Compute lateral velocity from lateral position changes
        v_vals = np.gradient(x_vals) * 3.0  # Scale factor for lateral dynamics

        # Compute yaw rate from heading changes
        omega_vals = np.gradient(phi_vals) * 2.0  # Scale factor for rotational dynamics

        states_guess.append(np.vstack([x_vals, y_vals, phi_vals, u_vals, v_vals, omega_vals]))

        # Generate controls from state derivatives (N points for controls)
        a_vals = np.gradient(u_vals)[:N] * 2.0  # Take first N points
        delta_vals = np.gradient(phi_vals)[:N] * 3.0  # Take first N points

        # Clip to bounds
        a_vals = np.clip(a_vals, -8.0, 8.0)
        delta_vals = np.clip(delta_vals, -0.5, 0.5)

        controls_guess.append(np.vstack([a_vals, delta_vals]))

    return states_guess, controls_guess


# ============================================================================
# APPLY IMPROVED INITIAL GUESS
# ============================================================================

states_guess, controls_guess = _generate_realistic_overtaking_guess()

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 18.0},  # More realistic time for longer road
)


# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-1,
    max_iterations=10,  # Start with fewer iterations for initial success
    min_polynomial_degree=3,
    max_polynomial_degree=8,  # Lower max degree for faster convergence
    ode_solver_tolerance=1e-1,
    nlp_options={
        "ipopt.max_iter": 1500,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-3,  # More relaxed constraint tolerance
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 5,  # Moderate output for debugging
    },
)


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

if solution.status["success"]:
    print("✓ IMPROVED Highway overtaking succeeded!")
    print(f"  Objective: {solution.status['objective']:.3f}")
    print(f"  Mission time: {solution.status['total_mission_time']:.3f} seconds")

    x_final = solution[(1, "x_position")][-1]
    y_final = solution[(1, "y_position")][-1]

    print("Final position verification:")
    print(f"  Reached: ({x_final:.2f}, {y_final:.2f}) m")
    print(f"  Target: ({AGENT_END[0]:.2f}, {AGENT_END[1]:.2f}) m")
    print(
        f"  Error: {np.sqrt((x_final - AGENT_END[0]) ** 2 + (y_final - AGENT_END[1]) ** 2):.3f} m"
    )

    # Analyze overtaking behavior
    x_traj = solution["x_position"]
    max_lateral_deviation = abs(x_traj - RIGHT_LANE_CENTER).max()
    print(f"  Max lateral deviation: {max_lateral_deviation:.2f} m")
    print(f"  Overtaking successful: {'✓' if max_lateral_deviation > 2.0 else '✗'}")

    solution.plot()

else:
    print(f"✗ Optimization failed: {solution.status['message']}")
    print("Debugging suggestions:")
    print("  - Check initial guess quality")
    print("  - Verify constraint feasibility")
    print("  - Consider relaxing tolerances further")


# ============================================================================
# EXPORT CONSTANTS FOR ANIMATION COMPATIBILITY
# ============================================================================

# These constants ensure the animation module can import the correct geometry
# Animation file should update plot limits to:
#   ax_main.set_xlim(HIGHWAY_LEFT_BOUNDARY-1, HIGHWAY_RIGHT_BOUNDARY+1)  # -1 to 19
#   ax_main.set_ylim(HIGHWAY_BOTTOM-2, HIGHWAY_TOP+2)                    # -7 to 42
# And use HIGHWAY_CENTER instead of hardcoded center_line = 8.5

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
