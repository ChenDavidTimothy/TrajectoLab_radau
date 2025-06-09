import casadi as ca
import numpy as np

import maptor as mtor


def obstacle_position(current_time):
    # Define waypoints: [x, y, time]
    waypoints = np.array(
        [
            [10.0, 15.0, 0.0],  # Start: upper left
            [15.0, 10.0, 3.0],  # Move to center, crossing vehicle path
            [12.0, 8.0, 5.0],  # Slow down near path
            [8.0, 6.0, 8.0],  # Block lower route
            [5.0, 12.0, 10.0],  # Move up to block alternative
            [25.0, 25.0, 15.0],  # Exit area quickly
        ]
    )

    # Clamp time to waypoint bounds
    t_min, t_max = waypoints[0, 2], waypoints[-1, 2]
    t_clamped = ca.fmax(t_min, ca.fmin(t_max, current_time))

    # Initialize with final waypoint (fallback)
    obs_x = waypoints[-1, 0]
    obs_y = waypoints[-1, 1]

    # Multi-segment interpolation with conditional logic
    for i in range(len(waypoints) - 1):
        t1, t2 = waypoints[i, 2], waypoints[i + 1, 2]
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        in_segment = ca.logic_and(t_clamped >= t1, t_clamped < t2)
        alpha = (t_clamped - t1) / (t2 - t1)
        x_interp = (1 - alpha) * wp1[0] + alpha * wp2[0]
        y_interp = (1 - alpha) * wp1[1] + alpha * wp2[1]
        obs_x = ca.if_else(in_segment, x_interp, obs_x)
        obs_y = ca.if_else(in_segment, y_interp, obs_y)

    return obs_x, obs_y


# Problem setup
problem = mtor.Problem("Dynamic Obstacle Avoidance")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=20.0)
y = phase.state("y_position", initial=0.0, final=20.0)
theta = phase.state("heading", initial=0.0, final=np.pi / 2)
v = phase.state("velocity", initial=1.0, boundary=(0.5, 8.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))
a = phase.control("acceleration", boundary=(-3.0, 3.0))

# Dynamics
L = 2.5  # Wheelbase (m)
phase.dynamics(
    {
        x: v * ca.cos(theta),
        y: v * ca.sin(theta),
        theta: v * ca.tan(delta) / L,
        v: a,
    }
)

# Path constraints
vehicle_radius = 1.5  # Vehicle safety radius (m)
obstacle_radius = 2.5  # Obstacle radius (m)
obs_x, obs_y = obstacle_position(t)
distance_squared = (x - obs_x) ** 2 + (y - obs_y) ** 2
min_separation = vehicle_radius + obstacle_radius

phase.path_constraints(distance_squared >= min_separation**2)
phase.path_constraints(x >= -5.0, x <= 25.0, y >= -5.0, y <= 25.0)

# Objective
problem.minimize(t.final)

# Mesh and guess
phase.mesh([6, 6, 6], [-1.0, -1 / 3, 1 / 3, 1.0])

states_guess = []
controls_guess = []
for N in [6, 6, 6]:
    tau = np.linspace(0, 1, N + 1)
    states_guess.append(
        np.array(
            [
                20.0 * tau,  # Linear x trajectory
                20.0 * tau,  # Linear y trajectory
                np.ones(N + 1) * 0.785,  # Constant heading ~45°
                np.ones(N + 1) * 3.0,  # Constant velocity
            ]
        )
    )
    controls_guess.append(np.array([np.zeros(N), np.zeros(N)]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: 12.0},
)

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-4,
    max_iterations=20,
    min_polynomial_degree=3,
    max_polynomial_degree=9,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000},
)

# Results
if __name__ == "__main__":
    if solution.status["success"]:
        print(f"Optimal time: {solution.status['objective']:.6f}")
        print("Reference: Problem-dependent (dynamic obstacle avoidance)")

        # Final state verification
        x_final = solution[(1, "x_position")][-1]
        y_final = solution[(1, "y_position")][-1]

        print("Final position:")
        print(f"  x: {x_final:.6f} (target: 20.0)")
        print(f"  y: {y_final:.6f} (target: 20.0)")

        solution.plot()
    else:
        print(f"Failed: {solution.status['message']}")
