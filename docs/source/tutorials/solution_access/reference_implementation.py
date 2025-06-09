import matplotlib.pyplot as plt
import numpy as np

import maptor as mtor


# Problem setup
problem = mtor.Problem("Two-Phase Schwartz Problem")

# Phase 1
phase1 = problem.set_phase(1)
phase1.time(initial=0.0, final=1.0)
x0_1 = phase1.state("x0", initial=1.0)
x1_1 = phase1.state("x1", initial=1.0, boundary=(-0.8, None))
u1 = phase1.control("u", boundary=(-1.0, 1.0))

phase1.dynamics(
    {
        x0_1: x1_1,
        x1_1: u1 - 0.1 * (1 + 2 * x0_1**2) * x1_1,
    }
)

elliptical_constraint = 1 - 9 * (x0_1 - 1) ** 2 - ((x1_1 - 0.4) / 0.3) ** 2
phase1.path_constraints(elliptical_constraint <= 0)
phase1.mesh([6, 6], [-1.0, 0.0, 1.0])

# Phase 2
phase2 = problem.set_phase(2)
phase2.time(initial=1.0, final=2.9)
x0_2 = phase2.state("x0", initial=x0_1.final)
x1_2 = phase2.state("x1", initial=x1_1.final)
u2 = phase2.control("u")

phase2.dynamics(
    {
        x0_2: x1_2,
        x1_2: u2 - 0.1 * (1 + 2 * x0_2**2) * x1_2,
    }
)
phase2.mesh([8, 8], [-1.0, 0.0, 1.0])

objective_expr = 5 * (x0_2.final**2 + x1_2.final**2)
problem.minimize(objective_expr)

# Initial guess
states_p1 = []
controls_p1 = []
states_p2 = []
controls_p2 = []

for N in [6, 6]:
    tau_states = np.linspace(-1, 1, N + 1)
    t_norm_states = (tau_states + 1) / 2
    x0_vals = 1.0 + 0.2 * t_norm_states
    x1_vals = 1.0 - 0.3 * t_norm_states
    states_p1.append(np.array([x0_vals, x1_vals]))

    t_norm_controls = np.linspace(0, 1, N)
    u_vals = 0.3 * np.sin(np.pi * t_norm_controls)
    controls_p1.append(np.array([u_vals]))

for N in [8, 8]:
    tau_states = np.linspace(-1, 1, N + 1)
    t_norm_states = (tau_states + 1) / 2
    x0_end_p1 = 1.2
    x1_end_p1 = 0.7
    x0_vals = x0_end_p1 * (1 - 0.8 * t_norm_states)
    x1_vals = x1_end_p1 * (1 - 0.9 * t_norm_states)
    states_p2.append(np.array([x0_vals, x1_vals]))

    t_norm_controls = np.linspace(0, 1, N)
    u_vals = -1.0 + 0.5 * t_norm_controls
    controls_p2.append(np.array([u_vals]))

problem.guess(
    phase_states={1: states_p1, 2: states_p2},
    phase_controls={1: controls_p1, 2: controls_p2},
    phase_initial_times={1: 0.0, 2: 1.0},
    phase_terminal_times={1: 1.0, 2: 2.9},
)

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-4,
    max_iterations=25,
    min_polynomial_degree=3,
    max_polynomial_degree=12,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 3000, "ipopt.tol": 1e-8},
)

# ============================================================================
# SOLUTION ACCESS REFERENCE
# ============================================================================

# Basic status validation
success = solution.status["success"]
objective = solution.status["objective"]
total_time = solution.status["total_mission_time"]
message = solution.status["message"]

print(f"Success: {success}")
print(f"Objective: {objective:.12e}")
print(f"Mission time: {total_time:.6f}")

# Complete mission trajectories
x0_trajectory = solution["x0"]
x1_trajectory = solution["x1"]
u_trajectory = solution["u"]
time_states = solution["time_states"]
time_controls = solution["time_controls"]

print("\nComplete trajectories:")
print(f"x0: {len(x0_trajectory)} points, values - {x0_trajectory}")
print(f"x1: {len(x1_trajectory)} points, values - {x1_trajectory}")
print(f"u: {len(u_trajectory)} points, values - {u_trajectory}")

# Mission final values
final_x0 = x0_trajectory[-1]
final_x1 = x1_trajectory[-1]
final_time = time_states[-1]

print("\nMission final state:")
print(f"x0_final: {final_x0:.12e}")
print(f"x1_final: {final_x1:.12e}")
print(f"time_final: {final_time:.6f}")

# Phase-specific data access
phase_ids = list(solution.phases.keys())
print("\nPhase information:")
print(f"Number of phases: {len(phase_ids)}")

for phase_id in phase_ids:
    phase_data = solution.phases[phase_id]

    # Phase timing
    initial_time = phase_data["times"]["initial"]
    final_time = phase_data["times"]["final"]
    duration = phase_data["times"]["duration"]

    # Phase variables
    state_names = phase_data["variables"]["state_names"]
    control_names = phase_data["variables"]["control_names"]
    num_states = phase_data["variables"]["num_states"]
    num_controls = phase_data["variables"]["num_controls"]

    # Phase mesh
    num_intervals = phase_data["mesh"]["num_intervals"]
    polynomial_degrees = phase_data["mesh"]["polynomial_degrees"]

    print(f"\nPhase {phase_id}:")
    print(f"  Time: [{initial_time:.6f}, {final_time:.6f}], duration: {duration:.6f}")
    print(f"  Variables: {num_states} states, {num_controls} controls")
    print(f"  States: {state_names}")
    print(f"  Controls: {control_names}")
    print(f"  Mesh: {num_intervals} intervals, degrees: {polynomial_degrees}")

# Phase-specific trajectory access
for phase_id in phase_ids:
    x0_phase = solution[(phase_id, "x0")]
    x1_phase = solution[(phase_id, "x1")]
    u_phase = solution[(phase_id, "u")]
    t_states_phase = solution[(phase_id, "time_states")]
    t_controls_phase = solution[(phase_id, "time_controls")]

    print(f"\nPhase {phase_id} trajectories:")
    print(f"  x0: {len(x0_phase)} points, [{x0_phase[0]:.6f} to {x0_phase[-1]:.6f}]")
    print(f"  x1: {len(x1_phase)} points, [{x1_phase[0]:.6f} to {x1_phase[-1]:.6f}]")
    print(f"  u: {len(u_phase)} points, [{u_phase[0]:.6f} to {u_phase[-1]:.6f}]")
    print(f"  State times: [{t_states_phase[0]:.6f} to {t_states_phase[-1]:.6f}]")
    print(f"  Control times: [{t_controls_phase[0]:.6f} to {t_controls_phase[-1]:.6f}]")

# Variable existence checking
available_vars = []
test_vars = ["x0", "x1", "u", "time_states", "time_controls"]
for var in test_vars:
    if var in solution:
        available_vars.append(var)

print(f"\nAvailable variables: {available_vars}")

# Static parameters
if solution.parameters is not None:
    param_count = solution.parameters["count"]
    param_values = solution.parameters["values"]
    param_names = solution.parameters["names"]

    print("\nStatic parameters:")
    print(f"  Count: {param_count}")

    if param_names is not None:
        for name, value in zip(param_names, param_values, strict=False):
            print(f"  {name}: {value:.12e}")
    else:
        for i, value in enumerate(param_values):
            print(f"  param_{i}: {value:.12e}")
else:
    print("\nNo static parameters")

# Adaptive algorithm results
if solution.adaptive is not None:
    converged = solution.adaptive["converged"]
    iterations = solution.adaptive["iterations"]
    tolerance = solution.adaptive["target_tolerance"]
    phase_converged = solution.adaptive["phase_converged"]
    final_errors = solution.adaptive["final_errors"]

    print("\nAdaptive refinement:")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Tolerance: {tolerance:.3e}")

    print("  Phase convergence:")
    for phase_id, phase_conv in phase_converged.items():
        print(f"    Phase {phase_id}: {phase_conv}")

    print("  Final errors:")
    for phase_id, errors in final_errors.items():
        if errors:
            max_error = max(errors)
            mean_error = np.mean(errors)
            print(f"    Phase {phase_id}: max={max_error:.3e}, mean={mean_error:.3e}")
else:
    print("\nFixed mesh solution")

# Mesh details
print("\nMesh configuration:")
total_intervals = 0
for phase_id in phase_ids:
    mesh_data = solution.phases[phase_id]["mesh"]
    nodes = mesh_data["mesh_nodes"]
    degrees = mesh_data["polynomial_degrees"]
    intervals = mesh_data["num_intervals"]
    total_intervals += intervals

    print(f"  Phase {phase_id}: {intervals} intervals, degrees {degrees}")
    if len(nodes) <= 10:
        nodes_str = "[" + ", ".join(f"{n:.4f}" for n in nodes) + "]"
        print(f"    Nodes: {nodes_str}")

print(f"  Total intervals: {total_intervals}")

# Integral values
print("\nIntegral values:")
for phase_id in phase_ids:
    integrals = solution.phases[phase_id]["integrals"]
    if integrals is not None:
        if isinstance(integrals, int | float):
            print(f"  Phase {phase_id}: {integrals:.12e}")
        else:
            for i, val in enumerate(integrals):
                print(f"  Phase {phase_id}[{i}]: {val:.12e}")
    else:
        print(f"  Phase {phase_id}: None")

# Data extraction for analysis
print("\nData extraction examples:")

# Time-series data
mission_data = {
    "time_states": time_states,
    "time_controls": time_controls,
    "x0": x0_trajectory,
    "x1": x1_trajectory,
    "u": u_trajectory,
}

print("  Complete mission data arrays created")
print(f"  State points: {len(time_states)}")
print(f"  Control points: {len(time_controls)}")

# Specific time intervals
t_start = time_states[0]
t_mid = time_states[len(time_states) // 2]
t_end = time_states[-1]

print(f"  Time span: {t_start:.6f} to {t_end:.6f}")
print(f"  Midpoint time: {t_mid:.6f}")

# State space bounds
x0_min, x0_max = x0_trajectory.min(), x0_trajectory.max()
x1_min, x1_max = x1_trajectory.min(), x1_trajectory.max()
u_min, u_max = u_trajectory.min(), u_trajectory.max()

print(f"  State bounds: x0=[{x0_min:.6f}, {x0_max:.6f}], x1=[{x1_min:.6f}, {x1_max:.6f}]")
print(f"  Control bounds: u=[{u_min:.6f}, {u_max:.6f}]")

# Raw solver access
print("\nRaw solver data:")
print(f"  CasADi solution available: {solution.raw_solution is not None}")
print(f"  Opti object available: {solution.opti is not None}")

# Basic plotting data preparation

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(time_states, x0_trajectory)
plt.xlabel("Time")
plt.ylabel("x0")
plt.title("State x0")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(time_states, x1_trajectory)
plt.xlabel("Time")
plt.ylabel("x1")
plt.title("State x1")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(time_controls, u_trajectory)
plt.xlabel("Time")
plt.ylabel("u")
plt.title("Control u")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x0_trajectory, x1_trajectory)
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Phase Portrait")
plt.grid(True)

plt.subplot(2, 3, 5)
phase_colors = ["blue", "red"]
for i, phase_id in enumerate(phase_ids):
    x0_phase = solution[(phase_id, "x0")]
    x1_phase = solution[(phase_id, "x1")]
    plt.plot(x0_phase, x1_phase, color=phase_colors[i], label=f"Phase {phase_id}")
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Phase-Colored Trajectory")
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
for i, phase_id in enumerate(phase_ids):
    t_phase = solution[(phase_id, "time_controls")]
    u_phase = solution[(phase_id, "u")]
    plt.plot(t_phase, u_phase, color=phase_colors[i], label=f"Phase {phase_id}")
plt.xlabel("Time")
plt.ylabel("u")
plt.title("Phase-Colored Control")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nSolution access reference complete.")
