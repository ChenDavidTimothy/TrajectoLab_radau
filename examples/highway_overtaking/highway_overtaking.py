import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

import maptor as mtor


# ============================================================================
# HIGHWAY OVERTAKING SCENARIO DEFINITION
# ============================================================================

# Agent trajectory: Bottom to top in right lane
AGENT_START = (10.0, 0.0)
AGENT_END = (10.0, 20.0)

# Obstacle 1: Same lane vehicle moving upward (slower)
OBSTACLE_1_WAYPOINTS = np.array(
    [
        [10.0, 8.0, 0.0],  # Start behind agent
        [10.0, 18.0, 15.0],  # Slower vehicle - takes 15s
    ]
)

# Obstacle 2: Oncoming vehicle in left lane (creates overtaking window)
OBSTACLE_2_WAYPOINTS = np.array(
    [
        [7.0, 20.0, 0.0],  # Start further ahead
        [7.0, 5.0, 10.0],  # Faster oncoming - clears zone by t=10s
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
# VEHICLE PHYSICAL PARAMETERS (FROM GE ET AL., 2021)
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

problem = mtor.Problem("Highway Overtaking")
phase = problem.set_phase(1)


# ============================================================================
# STATE VARIABLES
# ============================================================================

t = phase.time(initial=0.0)

x = phase.state("x_position", initial=AGENT_START[0], final=AGENT_END[0])
y = phase.state("y_position", initial=AGENT_START[1], final=AGENT_END[1])
phi = phase.state("heading", initial=np.pi / 2, final=np.pi / 2)  # Pointing upward
u = phase.state("longitudinal_velocity", initial=12.0, boundary=(u_min, 30.0))
v = phase.state("lateral_velocity", initial=0.0, boundary=(-15.0, 15.0))
omega = phase.state("yaw_rate", initial=0.0, boundary=(-3.0, 3.0))


# ============================================================================
# CONTROL VARIABLES
# ============================================================================

a = phase.control("acceleration", boundary=(-8.0, 8.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))


# ============================================================================
# DYNAMIC BICYCLE MODEL
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
# COLLISION AVOIDANCE CONSTRAINTS
# ============================================================================

obs_x_1, obs_y_1 = obstacle_1_position(t)
distance_squared_1 = (x - obs_x_1) ** 2 + (y - obs_y_1) ** 2

obs_x_2, obs_y_2 = obstacle_2_position(t)
distance_squared_2 = (x - obs_x_2) ** 2 + (y - obs_y_2) ** 2

phase.path_constraints(
    distance_squared_1 >= min_separation**2, distance_squared_2 >= min_separation**2
)


# ============================================================================
# HIGHWAY BOUNDARIES
# ============================================================================

phase.path_constraints(
    x >= 2.0,  # Left boundary
    x <= 15.0,  # Right boundary
    y >= -10.0,  # Bottom boundary
    y <= 25.0,  # Top boundary
)
# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

control_effort = phase.add_integral(a**2 + delta**2)
problem.minimize(t.final + 0.05 * control_effort)


# ============================================================================
# MESH CONFIGURATION
# ============================================================================

phase.mesh([6, 6, 6, 6], [-1.0, -0.5, 0.0, 0.5, 1.0])


# ============================================================================
# INITIAL GUESS GENERATION
# ============================================================================


def _generate_overtaking_guess():
    """Generate collision-aware overtaking trajectory."""
    # Timing-aware waypoints based on obstacle movements
    # t=0: agent at (10,0), obs1 at (10,8), obs2 at (7,20)
    # t=5: agent overtakes when obs2 has moved down and obs1 is slower
    # t=10: agent completes overtaking, obs2 has passed critical zone

    waypoints_x = np.array([10.0, 10.0, 8.5, 7.0, 8.5, 10.0])  # Smooth lane change
    waypoints_y = np.array([0.0, 6.0, 10.0, 13.0, 16.0, 20.0])  # Progressive upward
    waypoints_t = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0])  # Non-uniform timing
    waypoints_phi = np.array(
        [np.pi / 2, np.pi / 2, np.pi / 2 + 0.2, np.pi / 2, np.pi / 2 - 0.2, np.pi / 2]
    )

    # Create smooth splines
    spline_x = CubicSpline(waypoints_t, waypoints_x)
    spline_y = CubicSpline(waypoints_t, waypoints_y)
    spline_phi = CubicSpline(waypoints_t, waypoints_phi)

    states_guess = []
    controls_guess = []

    for N in [6, 6, 6, 6]:
        tau = np.linspace(-1, 1, N + 1)
        t_norm = (tau + 1) / 2

        x_vals = spline_x(t_norm)
        y_vals = spline_y(t_norm)
        phi_vals = spline_phi(t_norm)
        u_vals = np.full(N + 1, 15.0)  # Higher initial speed
        v_vals = np.gradient(x_vals) * 2.0  # Lateral velocity from x movement
        omega_vals = np.gradient(phi_vals) * 0.5  # Angular velocity from heading change

        states_guess.append(np.vstack([x_vals, y_vals, phi_vals, u_vals, v_vals, omega_vals]))

        # Control guess with smooth acceleration and steering
        a_vals = np.full(N, 1.0)  # Slight acceleration
        delta_vals = np.gradient(phi_vals[:-1]) * 2.0  # Steering from heading change
        controls_guess.append(np.vstack([a_vals, delta_vals]))

    return states_guess, controls_guess


states_guess, controls_guess = _generate_overtaking_guess()

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 12.0},  # Longer time to allow overtaking
)


# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================

solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-1,
    max_iterations=5,
    min_polynomial_degree=3,
    max_polynomial_degree=10,
    ode_solver_tolerance=1e-1,
    nlp_options={
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1e-8,
        "ipopt.constr_viol_tol": 1e-2,
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 5,
    },
)


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

if solution.status["success"]:
    print(f"Highway overtaking time: {solution.status['objective']:.3f} seconds")
    print(f"Mission duration: {solution.status['total_mission_time']:.3f} seconds")

    x_final = solution[(1, "x_position")][-1]
    y_final = solution[(1, "y_position")][-1]

    print("Final position verification:")
    print(f"  Reached: ({x_final:.2f}, {y_final:.2f}) m")
    print(f"  Target: ({AGENT_END[0]:.2f}, {AGENT_END[1]:.2f}) m")
    print(
        f"  Error: {np.sqrt((x_final - AGENT_END[0]) ** 2 + (y_final - AGENT_END[1]) ** 2):.3f} m"
    )

    solution.plot()

else:
    print(f"Optimization failed: {solution.status['message']}")
