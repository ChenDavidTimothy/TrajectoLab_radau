import casadi as ca
import numpy as np

import maptor as mtor


# Obstacle trajectory waypoints - single source of truth for animation
OBSTACLE_WAYPOINTS = np.array(
    [
        [20.0, 20.0, 0.0],
        [18.0, 18.0, 3.0],
        [15.0, 15.0, 5.0],
        [0.0, 0.0, 10.0],
    ]
)

# Create module-level interpolants for efficiency and reliability
_times = OBSTACLE_WAYPOINTS[:, 2]
_x_coords = OBSTACLE_WAYPOINTS[:, 0]
_y_coords = OBSTACLE_WAYPOINTS[:, 1]

_x_interpolant = ca.interpolant("obs_x_interp", "linear", [_times], _x_coords)
_y_interpolant = ca.interpolant("obs_y_interp", "linear", [_times], _y_coords)


def obstacle_position(current_time):
    t_clamped = ca.fmax(_times[0], ca.fmin(_times[-1], current_time))
    return _x_interpolant(t_clamped), _y_interpolant(t_clamped)


# Problem setup
problem = mtor.Problem("Dynamic Obstacle Avoidance")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=20.0)
y = phase.state("y_position", initial=0.0, final=20.0)
theta = phase.state("heading", initial=np.pi / 4.0, final=np.pi / 4.0)
v = phase.state("velocity", initial=1.0, boundary=(0.5, 8.0))
delta = phase.control("steering_angle", boundary=(-0.5, 0.5))
a = phase.control("acceleration", boundary=(-3.0, 3.0))

# Dynamics - bicycle model
L = 2.5  # Wheelbase (m)
phase.dynamics(
    {
        x: v * ca.cos(theta),
        y: v * ca.sin(theta),
        theta: v * ca.tan(delta) / L,
        v: a,
    }
)

# Path constraints - collision avoidance
vehicle_radius = 1.5  # Vehicle safety radius (m)
obstacle_radius = 2.5  # Obstacle radius (m)
obs_x, obs_y = obstacle_position(t)
distance_squared = (x - obs_x) ** 2 + (y - obs_y) ** 2
min_separation = vehicle_radius + obstacle_radius

phase.path_constraints(distance_squared >= min_separation**2)
phase.path_constraints(x >= -5.0, x <= 25.0, y >= -5.0, y <= 25.0)

# Objective - minimize time
problem.minimize(t.final)

# Mesh and initial guess
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])


# Solve with adaptive mesh refinement
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=5,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000},
)

# Results and analysis
if __name__ == "__main__":
    if solution.status["success"]:
        print(f"Adaptive objective: {solution.status['objective']:.6f}")
        solution.plot()

        # Final state verification
        x_final = solution[(1, "x_position")][-1]
        y_final = solution[(1, "y_position")][-1]

        print("Final position:")
        print(f"  x: {x_final:.6f} (target: 20.0)")
        print(f"  y: {y_final:.6f} (target: 20.0)")

    else:
        print(f"Failed: {solution.status['message']}")
