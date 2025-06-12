import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import maptor as mtor


# ============================================================================
# PRIMARY EXAMPLE: Dynamic Obstacle Avoidance
# ============================================================================

# Obstacle trajectory waypoints for dynamic avoidance
OBSTACLE_WAYPOINTS = np.array(
    [
        [5.0, 5.0, 0.0],
        [12.0, 12.0, 3.0],
        [15.0, 15.0, 6.0],
        [20.0, 20.0, 12.0],
    ]
)

# Create interpolants for obstacle motion
_times = OBSTACLE_WAYPOINTS[:, 2]
_x_coords = OBSTACLE_WAYPOINTS[:, 0]
_y_coords = OBSTACLE_WAYPOINTS[:, 1]

_x_interpolant = ca.interpolant("obs_x_interp", "linear", [_times], _x_coords)
_y_interpolant = ca.interpolant("obs_y_interp", "linear", [_times], _y_coords)


def obstacle_position(current_time):
    """Get obstacle position at given time."""
    t_clamped = ca.fmax(_times[0], ca.fmin(_times[-1], current_time))
    return _x_interpolant(t_clamped), _y_interpolant(t_clamped)


# ============================================================================
# SOLVE PRIMARY EXAMPLE
# ============================================================================

print("Setting up Dynamic Obstacle Avoidance problem...")

problem = mtor.Problem("Dynamic Obstacle Avoidance with Solution Analysis")
phase = problem.set_phase(1)

# Variables with clear engineering meaning
t = phase.time(initial=0.0)
x = phase.state("x_position", initial=0.0, final=20.0)
y = phase.state("y_position", initial=0.0, final=20.0)
theta = phase.state("heading", initial=np.pi / 4.0)
v = phase.state("velocity", initial=1.0, boundary=(0.5, 20.0))
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

# Mesh and solve
phase.mesh([8, 8, 8], [-1.0, -1 / 3, 1 / 3, 1.0])

print("Solving primary obstacle avoidance problem...")
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-3,
    max_iterations=30,
    min_polynomial_degree=5,
    max_polynomial_degree=15,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000},
)


# ============================================================================
# CREATE MULTIPHASE SCENARIO FOR COMPREHENSIVE DEMONSTRATION
# ============================================================================

print("\nSetting up multiphase autonomous mission for comprehensive API demonstration...")

multiphase_problem = mtor.Problem("Multiphase Autonomous Mission")

# Phase 1: Urban navigation (0 to parking garage)
urban_phase = multiphase_problem.set_phase(1)
t1 = urban_phase.time(initial=0.0, final=8.0)
x1 = urban_phase.state("x_position", initial=0.0, boundary=(-10.0, 30.0))
y1 = urban_phase.state("y_position", initial=0.0, boundary=(-10.0, 30.0))
theta1 = urban_phase.state("heading", initial=0.0)
v1 = urban_phase.state("velocity", initial=2.0, boundary=(0.5, 15.0))  # Urban speed limit
fuel1 = urban_phase.state("fuel_mass", initial=50.0, boundary=(0.0, 50.0))
delta1 = urban_phase.control("steering_angle", boundary=(-0.5, 0.5))
a1 = urban_phase.control("acceleration", boundary=(-2.0, 2.0))  # Gentle urban driving

# Urban dynamics with fuel consumption
fuel_rate = 0.1  # kg/s base consumption
urban_phase.dynamics(
    {
        x1: v1 * ca.cos(theta1),
        y1: v1 * ca.sin(theta1),
        theta1: v1 * ca.tan(delta1) / L,
        v1: a1,
        fuel1: -fuel_rate
        * (1.0 + 0.1 * ca.fabs(a1) + 0.05 * v1),  # Consumption increases with speed/accel
    }
)

# Urban constraints
urban_phase.path_constraints(
    v1 <= 12.0,  # Urban speed limit
    ca.fabs(delta1) <= 0.3,  # Gentle steering in city
)

# Add fuel efficiency integral
fuel_efficiency = urban_phase.add_integral(fuel_rate * v1)  # Fuel per distance metric

# Phase 2: Highway segment (parking to destination)
highway_phase = multiphase_problem.set_phase(2)
t2 = highway_phase.time(initial=t1.final, final=25.0)
x2 = highway_phase.state("x_position", initial=x1.final, boundary=(-10.0, 50.0))
y2 = highway_phase.state("y_position", initial=y1.final, boundary=(-10.0, 50.0))
theta2 = highway_phase.state("heading", initial=theta1.final)
v2 = highway_phase.state("velocity", initial=v1.final, boundary=(5.0, 25.0))  # Highway speeds
fuel2 = highway_phase.state("fuel_mass", initial=fuel1.final, boundary=(0.0, 50.0))
delta2 = highway_phase.control("steering_angle", boundary=(-0.3, 0.3))
a2 = highway_phase.control("acceleration", boundary=(-4.0, 3.0))

# Highway dynamics
highway_phase.dynamics(
    {
        x2: v2 * ca.cos(theta2),
        y2: v2 * ca.sin(theta2),
        theta2: v2 * ca.tan(delta2) / L,
        v2: a2,
        fuel2: -fuel_rate
        * (0.8 + 0.05 * ca.fabs(a2) + 0.03 * v2),  # More efficient at highway speeds
    }
)

# Highway constraints
highway_phase.path_constraints(
    v2 >= 8.0,  # Minimum highway speed
    v2 <= 22.0,  # Highway speed limit
)

# Add control smoothness integral
control_effort = highway_phase.add_integral(delta2**2 + 0.1 * a2**2)

# Phase 3: Destination approach (highway to final target)
approach_phase = multiphase_problem.set_phase(3)
t3 = approach_phase.time(initial=t2.final, final=35.0)
x3 = approach_phase.state("x_position", initial=x2.final, final=40.0, boundary=(-10.0, 50.0))
y3 = approach_phase.state("y_position", initial=y2.final, final=35.0, boundary=(-10.0, 50.0))
theta3 = approach_phase.state("heading", initial=theta2.final, final=0.0)  # Park facing forward
v3 = approach_phase.state(
    "velocity", initial=v2.final, final=0.0, boundary=(0.0, 10.0)
)  # Come to stop
fuel3 = approach_phase.state("fuel_mass", initial=fuel2.final, boundary=(0.0, 50.0))
delta3 = approach_phase.control("steering_angle", boundary=(-0.4, 0.4))
a3 = approach_phase.control("acceleration", boundary=(-3.0, 1.0))

# Approach dynamics
approach_phase.dynamics(
    {
        x3: v3 * ca.cos(theta3),
        y3: v3 * ca.sin(theta3),
        theta3: v3 * ca.tan(delta3) / L,
        v3: a3,
        fuel3: -fuel_rate * (1.2 + 0.15 * ca.fabs(a3) + 0.08 * v3),  # Less efficient in stop-and-go
    }
)

# Precision parking constraints
approach_phase.path_constraints(
    v3 <= 8.0,  # Slow approach
)

# Add comfort integral (minimize jerk)
comfort_metric = approach_phase.add_integral(ca.fabs(a3) + 2.0 * ca.fabs(delta3))

# Static parameters for vehicle design optimization
vehicle_mass = multiphase_problem.parameter("vehicle_mass", boundary=(1200.0, 2000.0))
engine_power = multiphase_problem.parameter("engine_power", boundary=(100.0, 300.0))
drag_coefficient = multiphase_problem.parameter("drag_coeff", boundary=(0.25, 0.45))

# Multi-objective: minimize time, fuel consumption, and control effort
total_fuel_used = fuel1.initial - fuel3.final
total_time = t3.final
total_control_effort = fuel_efficiency + control_effort + comfort_metric

multiphase_problem.minimize(
    0.1 * total_time  # Time penalty
    + 2.0 * total_fuel_used  # Fuel cost
    + 0.01 * total_control_effort  # Comfort/smoothness
)

# Set up mesh for each phase
urban_phase.mesh([6, 6], [-1.0, 0.0, 1.0])
highway_phase.mesh([8, 8], [-1.0, 0.0, 1.0])
approach_phase.mesh([6, 6], [-1.0, 0.0, 1.0])

# Provide initial guess
multiphase_problem.guess(
    phase_terminal_times={1: 8.0, 2: 25.0, 3: 35.0},
    static_parameters=np.array([1500.0, 200.0, 0.35]),  # mass, power, drag
)

print("Solving multiphase mission...")
multiphase_solution = mtor.solve_adaptive(
    multiphase_problem,
    error_tolerance=1e-4,
    max_iterations=20,
    nlp_options={"ipopt.print_level": 0, "ipopt.max_iter": 1500},
)


# ============================================================================
# COMPREHENSIVE SOLUTION ACCESS REFERENCE IMPLEMENTATION
# ============================================================================


def demonstrate_solution_access_api():
    """
    Comprehensive demonstration of every MAPTOR solution access capability.
    Uses both single-phase and multiphase examples to show complete API.
    """

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MAPTOR SOLUTION ACCESS REFERENCE")
    print("=" * 80)

    # ========================================================================
    # BASIC STATUS VALIDATION
    # ========================================================================

    print("\n1. ESSENTIAL SOLUTION VALIDATION")
    print("-" * 40)

    # Single-phase example
    success = solution.status["success"]
    objective = solution.status["objective"]
    total_time = solution.status["total_mission_time"]
    message = solution.status["message"]

    print("Single-phase obstacle avoidance:")
    print(f"  Success: {success}")
    print(f"  Optimal time: {objective:.6f} seconds")
    print(f"  Mission time: {total_time:.6f} seconds")
    print(f"  Solver message: {message}")

    # Multiphase example
    mp_success = multiphase_solution.status["success"]
    mp_objective = multiphase_solution.status["objective"]
    mp_total_time = multiphase_solution.status["total_mission_time"]
    mp_message = multiphase_solution.status["message"]

    print("\nMultiphase autonomous mission:")
    print(f"  Success: {mp_success}")
    print(f"  Objective: {mp_objective:.6f}")
    print(f"  Total mission time: {mp_total_time:.6f} seconds")
    print(f"  Solver message: {mp_message}")

    if not success or not mp_success:
        print("⚠ One or more solutions failed - proceeding with available data")

    # ========================================================================
    # MISSION-WIDE TRAJECTORY ACCESS
    # ========================================================================

    print("\n2. MISSION-WIDE TRAJECTORY ACCESS")
    print("-" * 40)

    if success:
        # Complete mission trajectories (single-phase)
        x_trajectory = solution["x_position"]
        y_trajectory = solution["y_position"]
        heading_trajectory = solution["heading"]
        velocity_trajectory = solution["velocity"]
        steering_trajectory = solution["steering_angle"]
        accel_trajectory = solution["acceleration"]
        time_states = solution["time_states"]
        time_controls = solution["time_controls"]

        print("Single-phase trajectories:")
        print(
            f"  x_position: {len(x_trajectory)} points, range [{x_trajectory.min():.2f}, {x_trajectory.max():.2f}] m"
        )
        print(
            f"  y_position: {len(y_trajectory)} points, range [{y_trajectory.min():.2f}, {y_trajectory.max():.2f}] m"
        )
        print(
            f"  velocity: {len(velocity_trajectory)} points, range [{velocity_trajectory.min():.2f}, {velocity_trajectory.max():.2f}] m/s"
        )
        print(
            f"  steering_angle: {len(steering_trajectory)} points, range [{steering_trajectory.min():.3f}, {steering_trajectory.max():.3f}] rad"
        )
        print(
            f"  acceleration: {len(accel_trajectory)} points, range [{accel_trajectory.min():.2f}, {accel_trajectory.max():.2f}] m/s²"
        )

    if mp_success:
        # Complete mission trajectories (multiphase - auto-concatenated)
        mp_x_trajectory = multiphase_solution["x_position"]
        mp_y_trajectory = multiphase_solution["y_position"]
        mp_velocity_trajectory = multiphase_solution["velocity"]
        mp_fuel_trajectory = multiphase_solution["fuel_mass"]
        mp_time_states = multiphase_solution["time_states"]
        mp_time_controls = multiphase_solution["time_controls"]

        print("\nMultiphase trajectories (auto-concatenated):")
        print(f"  Complete mission x_position: {len(mp_x_trajectory)} points")
        print(f"  Complete mission y_position: {len(mp_y_trajectory)} points")
        print(f"  Complete mission velocity: {len(mp_velocity_trajectory)} points")
        print(f"  Complete mission fuel_mass: {len(mp_fuel_trajectory)} points")
        print(f"  Mission span: [{mp_time_states[0]:.2f}, {mp_time_states[-1]:.2f}] seconds")

        # Mission final values
        final_x = mp_x_trajectory[-1]
        final_y = mp_y_trajectory[-1]
        final_fuel = mp_fuel_trajectory[-1]
        initial_fuel = mp_fuel_trajectory[0]
        fuel_consumed = initial_fuel - final_fuel

        print("\nMission final state:")
        print(f"  Final position: ({final_x:.2f}, {final_y:.2f}) m")
        print("  Target position: (40.0, 35.0) m")
        print(f"  Position error: {np.sqrt((final_x - 40.0) ** 2 + (final_y - 35.0) ** 2):.3f} m")
        print(
            f"  Fuel consumed: {fuel_consumed:.2f} kg ({fuel_consumed / initial_fuel * 100:.1f}% of tank)"
        )

    # ========================================================================
    # PHASE-SPECIFIC DATA ACCESS
    # ========================================================================

    print("\n3. PHASE-SPECIFIC DATA ACCESS")
    print("-" * 35)

    if success:
        # Single-phase access (tuple notation still works)
        x_phase1 = solution[(1, "x_position")]
        y_phase1 = solution[(1, "y_position")]
        v_phase1 = solution[(1, "velocity")]
        t_states_phase1 = solution[(1, "time_states")]
        t_controls_phase1 = solution[(1, "time_controls")]

        print("Single-phase data (using tuple access):")
        print(f"  Phase 1 x_position: {len(x_phase1)} points")
        print(f"  Phase 1 time span: [{t_states_phase1[0]:.3f}, {t_states_phase1[-1]:.3f}] s")
        print(f"  Phase 1 avg velocity: {np.mean(v_phase1):.2f} m/s")

    if mp_success:
        # Multiphase-specific access
        phase_ids = list(multiphase_solution.phases.keys())
        print(f"\nMultiphase data (phases {phase_ids}):")

        for phase_id in phase_ids:
            x_phase = multiphase_solution[(phase_id, "x_position")]
            y_phase = multiphase_solution[(phase_id, "y_position")]
            v_phase = multiphase_solution[(phase_id, "velocity")]
            fuel_phase = multiphase_solution[(phase_id, "fuel_mass")]
            t_states_phase = multiphase_solution[(phase_id, "time_states")]
            t_controls_phase = multiphase_solution[(phase_id, "time_controls")]

            phase_names = {1: "Urban Navigation", 2: "Highway Segment", 3: "Destination Approach"}

            print(f"  Phase {phase_id} ({phase_names.get(phase_id, 'Unknown')}):")
            print(f"    Duration: {t_states_phase[-1] - t_states_phase[0]:.2f} seconds")
            print(f"    Start: ({x_phase[0]:.1f}, {y_phase[0]:.1f}) m")
            print(f"    End: ({x_phase[-1]:.1f}, {y_phase[-1]:.1f}) m")
            print(f"    Speed range: [{v_phase.min():.1f}, {v_phase.max():.1f}] m/s")
            print(f"    Fuel: {fuel_phase[0]:.1f} → {fuel_phase[-1]:.1f} kg")
            print(f"    Data points: {len(x_phase)} states, {len(t_controls_phase)} controls")

    # ========================================================================
    # PHASE INFORMATION ANALYSIS
    # ========================================================================

    print("\n4. PHASE INFORMATION ANALYSIS")
    print("-" * 35)

    # Single-phase analysis
    if success:
        print("Single-phase structure:")
        for phase_id, phase_data in solution.phases.items():
            times = phase_data["times"]
            variables = phase_data["variables"]
            mesh = phase_data["mesh"]

            print(f"  Phase {phase_id}:")
            print(
                f"    Time: [{times['initial']:.3f}, {times['final']:.3f}], duration: {times['duration']:.3f} s"
            )
            print(
                f"    Variables: {variables['num_states']} states, {variables['num_controls']} controls"
            )
            print(f"    State names: {variables['state_names']}")
            print(f"    Control names: {variables['control_names']}")
            print(
                f"    Mesh: {mesh['num_intervals']} intervals, degrees: {mesh['polynomial_degrees']}"
            )

    # Multiphase analysis
    if mp_success:
        print("\nMultiphase structure:")
        for phase_id, phase_data in multiphase_solution.phases.items():
            times = phase_data["times"]
            variables = phase_data["variables"]
            mesh = phase_data["mesh"]

            phase_descriptions = {
                1: "Urban Navigation",
                2: "Highway Segment",
                3: "Destination Approach",
            }

            print(f"  Phase {phase_id} ({phase_descriptions.get(phase_id, 'Unknown')}):")
            print(
                f"    Time: [{times['initial']:.3f}, {times['final']:.3f}], duration: {times['duration']:.3f} s"
            )
            print(
                f"    Variables: {variables['num_states']} states, {variables['num_controls']} controls"
            )
            print(f"    State names: {variables['state_names']}")
            print(f"    Control names: {variables['control_names']}")
            print(
                f"    Mesh: {mesh['num_intervals']} intervals, degrees: {mesh['polynomial_degrees']}"
            )

    # ========================================================================
    # VARIABLE EXISTENCE CHECKING
    # ========================================================================

    print("\n5. VARIABLE EXISTENCE CHECKING")
    print("-" * 35)

    # Test variable existence for single-phase
    if success:
        test_vars_single = [
            "x_position",
            "y_position",
            "heading",
            "velocity",
            "steering_angle",
            "acceleration",
            "time_states",
            "time_controls",
        ]

        available_single = [var for var in test_vars_single if var in solution]
        missing_single = [var for var in test_vars_single if var not in solution]

        print("Single-phase variable availability:")
        print(f"  Available ({len(available_single)}): {available_single}")
        if missing_single:
            print(f"  Missing ({len(missing_single)}): {missing_single}")
        else:
            print("  ✓ All expected variables present")

    # Test variable existence for multiphase
    if mp_success:
        test_vars_multi = [
            "x_position",
            "y_position",
            "velocity",
            "fuel_mass",
            "steering_angle",
            "acceleration",
            "time_states",
            "time_controls",
        ]

        available_multi = [var for var in test_vars_multi if var in multiphase_solution]
        missing_multi = [var for var in test_vars_multi if var not in multiphase_solution]

        print("\nMultiphase variable availability:")
        print(f"  Available ({len(available_multi)}): {available_multi}")
        if missing_multi:
            print(f"  Missing ({len(missing_multi)}): {missing_multi}")
        else:
            print("  ✓ All expected variables present")

        # Phase-specific existence checking
        print("\nPhase-specific variable checking:")
        for phase_id in multiphase_solution.phases.keys():
            if (phase_id, "fuel_mass") in multiphase_solution:
                fuel_data = multiphase_solution[(phase_id, "fuel_mass")]
                print(
                    f"  Phase {phase_id}: fuel_mass available, range [{fuel_data.min():.1f}, {fuel_data.max():.1f}] kg"
                )

            if (phase_id, "steering_angle") in multiphase_solution:
                steering_data = multiphase_solution[(phase_id, "steering_angle")]
                max_steering = abs(steering_data).max()
                print(
                    f"  Phase {phase_id}: max steering angle {max_steering:.3f} rad ({max_steering * 180 / np.pi:.1f}°)"
                )

    # ========================================================================
    # STATIC PARAMETER ACCESS
    # ========================================================================

    print("\n6. STATIC PARAMETER ACCESS")
    print("-" * 30)

    # Single-phase (no parameters)
    if success:
        print("Single-phase parameters:")
        if solution.parameters is not None:
            params = solution.parameters
            print(f"  Count: {params['count']}")
            for i, value in enumerate(params["values"]):
                print(f"  param_{i}: {value:.6f}")
        else:
            print("  No static parameters in single-phase problem")

    # Multiphase (has parameters)
    if mp_success:
        print("\nMultiphase parameters:")
        if multiphase_solution.parameters is not None:
            params = multiphase_solution.parameters
            param_values = params["values"]
            param_names = params["names"]

            print(f"  Parameter count: {params['count']}")

            if param_names is not None:
                # Named parameters
                for name, value in zip(param_names, param_values, strict=False):
                    if name == "vehicle_mass":
                        print(f"  Optimal vehicle mass: {value:.1f} kg")
                    elif name == "engine_power":
                        print(f"  Optimal engine power: {value:.1f} kW")
                    elif name == "drag_coeff":
                        print(f"  Optimal drag coefficient: {value:.3f}")
                    else:
                        print(f"  {name}: {value:.6f}")
            else:
                # Unnamed parameters with interpretation
                param_descriptions = ["vehicle_mass (kg)", "engine_power (kW)", "drag_coefficient"]
                for i, value in enumerate(param_values):
                    desc = param_descriptions[i] if i < len(param_descriptions) else f"param_{i}"
                    print(f"  {desc}: {value:.6f}")

            # Parameter optimization analysis
            print("  Parameter optimization results:")
            if len(param_values) >= 3:
                mass, power, drag = param_values[0], param_values[1], param_values[2]
                power_to_weight = power / mass
                print(f"    Power-to-weight ratio: {power_to_weight:.3f} kW/kg")
                print(
                    f"    Drag efficiency: {'Good' if drag < 0.32 else 'Moderate' if drag < 0.38 else 'Poor'}"
                )
        else:
            print("  No static parameters found")

    # ========================================================================
    # ADAPTIVE ALGORITHM ANALYSIS
    # ========================================================================

    print("\n7. ADAPTIVE ALGORITHM ANALYSIS")
    print("-" * 35)

    # Single-phase adaptive analysis
    if success and solution.adaptive is not None:
        adaptive_info = solution.adaptive
        print("Single-phase adaptive refinement:")
        print(f"  Converged: {adaptive_info['converged']}")
        print(f"  Iterations: {adaptive_info['iterations']}")
        print(f"  Target tolerance: {adaptive_info['target_tolerance']:.2e}")

        print("  Phase convergence:")
        for phase_id, converged in adaptive_info["phase_converged"].items():
            status_symbol = "✓" if converged else "✗"
            print(f"    Phase {phase_id}: {status_symbol}")

        print("  Final error estimates:")
        for phase_id, errors in adaptive_info["final_errors"].items():
            if errors and len(errors) > 0:
                max_error = max(errors)
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                print(
                    f"    Phase {phase_id}: max={max_error:.2e}, mean={mean_error:.2e}, std={std_error:.2e}"
                )
                print(f"      Error distribution: [{min(errors):.2e}, {max(errors):.2e}]")
    elif success:
        print("Single-phase: Fixed mesh solution - no adaptive data")

    # Multiphase adaptive analysis
    if mp_success and multiphase_solution.adaptive is not None:
        mp_adaptive = multiphase_solution.adaptive
        print("\nMultiphase adaptive refinement:")
        print(f"  Converged: {mp_adaptive['converged']}")
        print(f"  Iterations: {mp_adaptive['iterations']}")
        print(f"  Target tolerance: {mp_adaptive['target_tolerance']:.2e}")

        print("  Per-phase convergence analysis:")
        for phase_id, converged in mp_adaptive["phase_converged"].items():
            status_symbol = "✓" if converged else "✗"
            phase_names = {1: "Urban", 2: "Highway", 3: "Approach"}
            phase_name = phase_names.get(phase_id, f"Phase{phase_id}")
            print(f"    {phase_name} (Phase {phase_id}): {status_symbol}")

        print("  Error analysis by phase:")
        for phase_id, errors in mp_adaptive["final_errors"].items():
            if errors and len(errors) > 0:
                max_error = max(errors)
                mean_error = np.mean(errors)
                phase_names = {1: "Urban", 2: "Highway", 3: "Approach"}
                phase_name = phase_names.get(phase_id, f"Phase{phase_id}")
                print(f"    {phase_name}: max={max_error:.2e}, mean={mean_error:.2e}")

                # Error quality assessment
                if max_error < 1e-4:
                    print("      ✓ Excellent accuracy")
                elif max_error < 1e-3:
                    print("      → Good accuracy")
                else:
                    print("      ⚠ Moderate accuracy")
    elif mp_success:
        print("Multiphase: Fixed mesh solution - no adaptive data")

    # ========================================================================
    # INTEGRAL VALUES ANALYSIS
    # ========================================================================

    print("\n8. INTEGRAL VALUES ANALYSIS")
    print("-" * 30)

    # Single-phase integrals
    if success:
        print("Single-phase integrals:")
        for phase_id in solution.phases.keys():
            integrals = solution.phases[phase_id]["integrals"]
            if integrals is not None:
                if isinstance(integrals, (int, float)):
                    print(f"  Phase {phase_id}: {integrals:.6f}")
                else:
                    print(f"  Phase {phase_id}: {len(integrals)} integral values")
                    for i, val in enumerate(integrals):
                        print(f"    integral[{i}]: {val:.6f}")
            else:
                print(f"  Phase {phase_id}: No integral values")

    # Multiphase integrals
    if mp_success:
        print("\nMultiphase integrals:")
        integral_interpretations = {
            1: ["fuel_efficiency"],
            2: ["control_effort"],
            3: ["comfort_metric"],
        }

        for phase_id in multiphase_solution.phases.keys():
            integrals = multiphase_solution.phases[phase_id]["integrals"]
            phase_names = {1: "Urban", 2: "Highway", 3: "Approach"}
            phase_name = phase_names.get(phase_id, f"Phase{phase_id}")

            if integrals is not None:
                if isinstance(integrals, (int, float)):
                    # Single integral
                    integral_name = integral_interpretations.get(phase_id, ["unknown"])[0]
                    print(f"  {phase_name} (Phase {phase_id}) {integral_name}: {integrals:.6f}")

                    # Interpret value
                    if integral_name == "fuel_efficiency":
                        print(f"    → Fuel consumption rate: {integrals:.3f} kg·m/s")
                    elif integral_name == "control_effort":
                        print(f"    → Control smoothness: {integrals:.3f} (lower is smoother)")
                    elif integral_name == "comfort_metric":
                        print(
                            f"    → Passenger comfort: {integrals:.3f} (lower is more comfortable)"
                        )
                else:
                    # Multiple integrals
                    interpretations = integral_interpretations.get(
                        phase_id, [f"integral_{i}" for i in range(len(integrals))]
                    )
                    print(f"  {phase_name} (Phase {phase_id}): {len(integrals)} integrals")
                    for i, val in enumerate(integrals):
                        name = interpretations[i] if i < len(interpretations) else f"integral_{i}"
                        print(f"    {name}: {val:.6f}")
            else:
                print(f"  {phase_name} (Phase {phase_id}): No integrals")

        # Mission-wide integral analysis
        print("\n  Mission-wide integral summary:")
        total_fuel_efficiency = 0.0
        total_control_effort = 0.0
        total_comfort = 0.0

        for phase_id in [1, 2, 3]:
            integrals = multiphase_solution.phases[phase_id]["integrals"]
            if integrals is not None and isinstance(integrals, (int, float)):
                if phase_id == 1:
                    total_fuel_efficiency += integrals
                elif phase_id == 2:
                    total_control_effort += integrals
                elif phase_id == 3:
                    total_comfort += integrals

        print(f"    Total fuel efficiency metric: {total_fuel_efficiency:.3f}")
        print(f"    Total control effort: {total_control_effort:.3f}")
        print(f"    Total comfort metric: {total_comfort:.3f}")

    # ========================================================================
    # ADVANCED DATA EXTRACTION AND ANALYSIS
    # ========================================================================

    print("\n9. ADVANCED DATA EXTRACTION AND ANALYSIS")
    print("-" * 45)

    if success:
        print("Single-phase advanced analysis:")

        # Extract all data for analysis
        x_traj = solution["x_position"]
        y_traj = solution["y_position"]
        v_traj = solution["velocity"]
        heading_traj = solution["heading"]
        steering_traj = solution["steering_angle"]
        accel_traj = solution["acceleration"]
        time_traj = solution["time_states"]

        # Performance metrics
        mission_time = time_traj[-1] - time_traj[0]
        path_segments = np.sqrt(np.diff(x_traj) ** 2 + np.diff(y_traj) ** 2)
        total_distance = np.sum(path_segments)
        straight_distance = np.sqrt((x_traj[-1] - x_traj[0]) ** 2 + (y_traj[-1] - y_traj[0]) ** 2)
        path_efficiency = straight_distance / total_distance

        print(f"  Mission duration: {mission_time:.3f} seconds")
        print(f"  Path length: {total_distance:.2f} m")
        print(f"  Straight-line distance: {straight_distance:.2f} m")
        print(f"  Path efficiency: {path_efficiency:.3f}")

        # Speed analysis
        avg_speed = np.mean(v_traj)
        max_speed = np.max(v_traj)
        min_speed = np.min(v_traj)
        speed_variance = np.var(v_traj)

        print("  Speed statistics:")
        print(f"    Average: {avg_speed:.2f} m/s")
        print(f"    Range: [{min_speed:.2f}, {max_speed:.2f}] m/s")
        print(f"    Variance: {speed_variance:.4f} (consistency measure)")

        # Control analysis
        max_steering = np.max(np.abs(steering_traj))
        avg_steering_effort = np.mean(np.abs(steering_traj))
        max_accel = np.max(accel_traj)
        max_decel = np.min(accel_traj)

        print("  Control characteristics:")
        print(f"    Max steering: ±{max_steering:.3f} rad (±{max_steering * 180 / np.pi:.1f}°)")
        print(f"    Avg steering effort: {avg_steering_effort:.3f} rad")
        print(f"    Acceleration range: [{max_decel:.2f}, {max_accel:.2f}] m/s²")

        # Safety analysis
        obstacle_x, obstacle_y, obstacle_radius = 12.0, 12.0, 2.5  # Approximate obstacle
        min_clearance = float("inf")
        for i in range(len(x_traj)):
            distance = np.sqrt((x_traj[i] - obstacle_x) ** 2 + (y_traj[i] - obstacle_y) ** 2)
            clearance = distance - obstacle_radius
            min_clearance = min(min_clearance, clearance)

        print("  Safety analysis:")
        print(f"    Minimum obstacle clearance: {min_clearance:.2f} m")
        print(f"    Safety status: {'✓ SAFE' if min_clearance > 0 else '⚠ COLLISION RISK'}")

    if mp_success:
        print("\nMultiphase advanced analysis:")

        # Mission-wide analysis
        mp_x_traj = multiphase_solution["x_position"]
        mp_y_traj = multiphase_solution["y_position"]
        mp_v_traj = multiphase_solution["velocity"]
        mp_fuel_traj = multiphase_solution["fuel_mass"]
        mp_time_traj = multiphase_solution["time_states"]

        # Overall mission metrics
        total_mission_time = mp_time_traj[-1] - mp_time_traj[0]
        total_fuel_consumed = mp_fuel_traj[0] - mp_fuel_traj[-1]
        fuel_efficiency_mpkm = total_fuel_consumed / (
            np.sum(np.sqrt(np.diff(mp_x_traj) ** 2 + np.diff(mp_y_traj) ** 2)) / 1000
        )

        print("  Mission overview:")
        print(
            f"    Total duration: {total_mission_time:.2f} seconds ({total_mission_time / 60:.1f} minutes)"
        )
        print(f"    Fuel consumed: {total_fuel_consumed:.2f} kg")
        print(f"    Fuel efficiency: {fuel_efficiency_mpkm:.3f} kg/km")
        print(f"    Overall avg speed: {np.mean(mp_v_traj):.2f} m/s")

        # Phase-by-phase comparison
        print("  Phase-by-phase analysis:")
        for phase_id in [1, 2, 3]:
            x_phase = multiphase_solution[(phase_id, "x_position")]
            v_phase = multiphase_solution[(phase_id, "velocity")]
            fuel_phase = multiphase_solution[(phase_id, "fuel_mass")]
            t_phase = multiphase_solution[(phase_id, "time_states")]

            phase_duration = t_phase[-1] - t_phase[0]
            phase_distance = np.sum(
                np.sqrt(
                    np.diff(x_phase) ** 2
                    + np.diff(multiphase_solution[(phase_id, "y_position")]) ** 2
                )
            )
            phase_fuel_used = fuel_phase[0] - fuel_phase[-1]
            phase_avg_speed = np.mean(v_phase)

            phase_names = {1: "Urban", 2: "Highway", 3: "Approach"}
            print(f"    {phase_names[phase_id]} (Phase {phase_id}):")
            print(f"      Duration: {phase_duration:.1f}s, Distance: {phase_distance:.1f}m")
            print(f"      Fuel used: {phase_fuel_used:.2f}kg, Avg speed: {phase_avg_speed:.1f}m/s")
            print(f"      Efficiency: {phase_fuel_used / (phase_distance / 1000):.3f} kg/km")

    # ========================================================================
    # MESH CONFIGURATION ANALYSIS
    # ========================================================================

    print("\n10. MESH CONFIGURATION ANALYSIS")
    print("-" * 35)

    # Single-phase mesh analysis
    if success:
        print("Single-phase mesh details:")
        total_intervals = 0
        total_state_points = 0
        total_control_points = 0

        for phase_id in solution.phases.keys():
            mesh_data = solution.phases[phase_id]["mesh"]
            nodes = mesh_data["mesh_nodes"]
            degrees = mesh_data["polynomial_degrees"]
            intervals = mesh_data["num_intervals"]

            total_intervals += intervals
            state_points = sum(deg + 1 for deg in degrees)
            control_points = sum(degrees)
            total_state_points += state_points
            total_control_points += control_points

            print(f"  Phase {phase_id}:")
            print(f"    Intervals: {intervals}, Degrees: {degrees}")
            print(f"    State points: {state_points}, Control points: {control_points}")

            if len(nodes) <= 8:
                nodes_str = "[" + ", ".join(f"{n:.4f}" for n in nodes) + "]"
                print(f"    Mesh nodes: {nodes_str}")

            # Mesh quality analysis
            if len(nodes) > 1:
                node_spacings = np.diff(nodes)
                min_spacing = np.min(node_spacings)
                max_spacing = np.max(node_spacings)
                spacing_ratio = max_spacing / min_spacing if min_spacing > 0 else float("inf")

                print("    Mesh quality:")
                print(f"      Spacing range: [{min_spacing:.4f}, {max_spacing:.4f}]")
                print(f"      Uniformity ratio: {spacing_ratio:.2f}")
                if spacing_ratio < 3.0:
                    print("      ✓ Uniform mesh")
                elif spacing_ratio < 10.0:
                    print("      → Moderately refined")
                else:
                    print("      ⚠ Highly non-uniform")

        print(
            f"  Computational cost: {total_intervals} intervals, {total_control_points} collocation points"
        )

    # Multiphase mesh analysis
    if mp_success:
        print("\nMultiphase mesh details:")
        mp_total_intervals = 0
        mp_total_points = 0

        for phase_id in multiphase_solution.phases.keys():
            mesh_data = multiphase_solution.phases[phase_id]["mesh"]
            degrees = mesh_data["polynomial_degrees"]
            intervals = mesh_data["num_intervals"]

            mp_total_intervals += intervals
            control_points = sum(degrees)
            mp_total_points += control_points

            phase_names = {1: "Urban", 2: "Highway", 3: "Approach"}
            print(f"  {phase_names.get(phase_id, f'Phase{phase_id}')} (Phase {phase_id}):")
            print(f"    Intervals: {intervals}, Degrees: {degrees}")
            print(f"    Collocation points: {control_points}")

        print("  Total computational cost:")
        print(f"    Combined intervals: {mp_total_intervals}")
        print(f"    Combined collocation points: {mp_total_points}")

        # Efficiency metrics
        if multiphase_solution.adaptive:
            iterations = multiphase_solution.adaptive["iterations"]
            efficiency = mp_total_points / iterations if iterations > 0 else 0
            print(f"    Adaptive efficiency: {efficiency:.1f} points/iteration")

    # ========================================================================
    # SOLUTION DATA PREPARATION FOR CUSTOM ANALYSIS
    # ========================================================================

    print("\n11. DATA PREPARATION FOR CUSTOM ANALYSIS")
    print("-" * 45)

    if success:
        # Prepare single-phase data for external analysis
        single_phase_data = {
            "mission_metadata": {
                "success": success,
                "objective": objective,
                "mission_time": total_time,
                "solver_message": message,
            },
            "trajectories": {
                "time_states": solution["time_states"],
                "time_controls": solution["time_controls"],
                "position": {"x": solution["x_position"], "y": solution["y_position"]},
                "kinematics": {"heading": solution["heading"], "velocity": solution["velocity"]},
                "controls": {
                    "steering_angle": solution["steering_angle"],
                    "acceleration": solution["acceleration"],
                },
            },
            "analysis_results": {
                "path_length": total_distance if "total_distance" in locals() else 0,
                "path_efficiency": path_efficiency if "path_efficiency" in locals() else 0,
                "avg_speed": avg_speed if "avg_speed" in locals() else 0,
                "max_speed": max_speed if "max_speed" in locals() else 0,
                "safety_clearance": min_clearance if "min_clearance" in locals() else float("inf"),
            },
        }

        print("Single-phase data structure prepared:")
        print(f"  Trajectory points: {len(single_phase_data['trajectories']['time_states'])}")
        print(f"  Data categories: {list(single_phase_data.keys())}")
        print(f"  Analysis metrics: {list(single_phase_data['analysis_results'].keys())}")

    if mp_success:
        # Prepare multiphase data for external analysis
        multiphase_data = {
            "mission_metadata": {
                "success": mp_success,
                "objective": mp_objective,
                "total_time": mp_total_time,
                "phases": len(multiphase_solution.phases),
            },
            "complete_trajectories": {
                "time_states": multiphase_solution["time_states"],
                "time_controls": multiphase_solution["time_controls"],
                "position": {
                    "x": multiphase_solution["x_position"],
                    "y": multiphase_solution["y_position"],
                },
                "kinematics": {"velocity": multiphase_solution["velocity"]},
                "vehicle_state": {"fuel_mass": multiphase_solution["fuel_mass"]},
            },
            "phase_specific_data": {},
            "optimization_results": {},
            "performance_analysis": {},
        }

        # Add phase-specific data
        for phase_id in [1, 2, 3]:
            multiphase_data["phase_specific_data"][phase_id] = {
                "position": {
                    "x": multiphase_solution[(phase_id, "x_position")],
                    "y": multiphase_solution[(phase_id, "y_position")],
                },
                "velocity": multiphase_solution[(phase_id, "velocity")],
                "fuel_mass": multiphase_solution[(phase_id, "fuel_mass")],
                "controls": {
                    "steering_angle": multiphase_solution[(phase_id, "steering_angle")],
                    "acceleration": multiphase_solution[(phase_id, "acceleration")],
                },
                "timing": {
                    "time_states": multiphase_solution[(phase_id, "time_states")],
                    "time_controls": multiphase_solution[(phase_id, "time_controls")],
                },
            }

        # Add optimization results
        if multiphase_solution.parameters:
            multiphase_data["optimization_results"]["static_parameters"] = {
                "values": multiphase_solution.parameters["values"],
                "names": multiphase_solution.parameters["names"],
                "count": multiphase_solution.parameters["count"],
            }

        # Add performance analysis
        multiphase_data["performance_analysis"] = {
            "fuel_consumed": total_fuel_consumed if "total_fuel_consumed" in locals() else 0,
            "fuel_efficiency_kg_per_km": fuel_efficiency_mpkm
            if "fuel_efficiency_mpkm" in locals()
            else 0,
            "mission_duration_minutes": total_mission_time / 60
            if "total_mission_time" in locals()
            else 0,
            "average_speed_ms": np.mean(multiphase_solution["velocity"]) if mp_success else 0,
        }

        print("\nMultiphase data structure prepared:")
        print(
            f"  Complete trajectory points: {len(multiphase_data['complete_trajectories']['time_states'])}"
        )
        print(f"  Phase-specific datasets: {len(multiphase_data['phase_specific_data'])}")
        print(
            f"  Optimization parameters: {multiphase_data['optimization_results'].get('static_parameters', {}).get('count', 0)}"
        )
        print(f"  Performance metrics: {len(multiphase_data['performance_analysis'])}")

    # ========================================================================
    # RAW SOLVER ACCESS
    # ========================================================================

    print("\n12. RAW SOLVER ACCESS")
    print("-" * 25)

    if success:
        print("Single-phase raw solver data:")
        print(f"  CasADi solution available: {solution.raw_solution is not None}")
        print(f"  Opti object available: {solution.opti is not None}")
        if solution.raw_solution is not None:
            print(f"  Raw solution type: {type(solution.raw_solution)}")

    if mp_success:
        print("\nMultiphase raw solver data:")
        print(f"  CasADi solution available: {multiphase_solution.raw_solution is not None}")
        print(f"  Opti object available: {multiphase_solution.opti is not None}")
        if multiphase_solution.raw_solution is not None:
            print(f"  Raw solution type: {type(multiphase_solution.raw_solution)}")

    print("\n" + "=" * 80)
    print("SOLUTION ACCESS REFERENCE COMPLETE")
    print("=" * 80)

    # Return structured data for further use
    return {
        "single_phase": single_phase_data if success else None,
        "multiphase": multiphase_data if mp_success else None,
    }


def create_comprehensive_visualizations():
    """Create advanced visualizations demonstrating solution plotting capabilities."""

    print("\n" + "=" * 60)
    print("COMPREHENSIVE VISUALIZATION EXAMPLES")
    print("=" * 60)

    if not solution.status["success"]:
        print("Single-phase solution failed - skipping single-phase plots")
    else:
        # Built-in plotting methods
        print("\n1. Built-in plotting methods:")

        # Basic plot
        print("   Creating basic solution plot...")
        solution.plot()

        print("   Creating custom variable plot...")
        solution.plot(None, "x_position", "y_position", "velocity")

        print("   Creating large format plot...")
        solution.plot(None, figsize=(16, 10), show_phase_boundaries=False)

    if not multiphase_solution.status["success"]:
        print("Multiphase solution failed - skipping multiphase plots")
    else:
        print("\n2. Multiphase plotting examples:")

        # Complete multiphase plot
        print("   Creating complete multiphase plot...")
        multiphase_solution.plot(show_phase_boundaries=True)

        # Phase-specific plots
        print("   Creating phase-specific plots...")
        for phase_id in [1, 2, 3]:
            phase_names = {1: "Urban Navigation", 2: "Highway Segment", 3: "Destination Approach"}
            print(f"   Plotting {phase_names[phase_id]} (Phase {phase_id})...")
            multiphase_solution.plot(phase_id=phase_id)

        # Custom multiphase analysis plot
        print("   Creating comprehensive multiphase analysis...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Multiphase Autonomous Mission - Complete Analysis", fontsize=16)

        # Mission trajectory with phase coloring
        ax = axes[0, 0]
        phase_colors = ["blue", "green", "red"]
        for i, phase_id in enumerate([1, 2, 3]):
            x_phase = multiphase_solution[(phase_id, "x_position")]
            y_phase = multiphase_solution[(phase_id, "y_position")]
            phase_names = {1: "Urban", 2: "Highway", 3: "Approach"}
            ax.plot(
                x_phase,
                y_phase,
                color=phase_colors[i],
                linewidth=3,
                label=f"{phase_names[phase_id]} (Phase {phase_id})",
            )

        ax.scatter([0], [0], color="black", s=100, marker="s", label="Start", zorder=5)
        ax.scatter([40], [35], color="gold", s=150, marker="*", label="Destination", zorder=5)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Complete Mission Trajectory")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        # Velocity profile across phases
        ax = axes[0, 1]
        complete_time = multiphase_solution["time_states"]
        complete_velocity = multiphase_solution["velocity"]
        ax.plot(complete_time, complete_velocity, "b-", linewidth=2)

        # Add phase boundaries
        for phase_id in [1, 2]:
            boundary_time = multiphase_solution.phases[phase_id]["times"]["final"]
            ax.axvline(boundary_time, color="red", linestyle="--", alpha=0.7)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Velocity Profile with Phase Boundaries")
        ax.grid(True)

        # Fuel consumption
        ax = axes[0, 2]
        complete_fuel = multiphase_solution["fuel_mass"]
        fuel_consumed = complete_fuel[0] - complete_fuel
        ax.plot(complete_time, fuel_consumed, "g-", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Fuel Consumed (kg)")
        ax.set_title("Cumulative Fuel Consumption")
        ax.grid(True)

        # Control effort by phase
        ax = axes[1, 0]
        for i, phase_id in enumerate([1, 2, 3]):
            steering = multiphase_solution[(phase_id, "steering_angle")]
            accel = multiphase_solution[(phase_id, "acceleration")]
            time_controls = multiphase_solution[(phase_id, "time_controls")]

            control_effort = np.sqrt(steering**2 + (accel / 5) ** 2)  # Normalized
            ax.plot(
                time_controls,
                control_effort,
                color=phase_colors[i],
                linewidth=2,
                label=f"Phase {phase_id}",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Control Effort")
        ax.set_title("Control Effort by Phase")
        ax.legend()
        ax.grid(True)

        # Phase performance comparison
        ax = axes[1, 1]
        phase_ids = [1, 2, 3]
        phase_names = ["Urban", "Highway", "Approach"]
        phase_durations = []
        phase_distances = []
        phase_fuel_rates = []

        for phase_id in phase_ids:
            t_phase = multiphase_solution[(phase_id, "time_states")]
            x_phase = multiphase_solution[(phase_id, "x_position")]
            y_phase = multiphase_solution[(phase_id, "y_position")]
            fuel_phase = multiphase_solution[(phase_id, "fuel_mass")]

            duration = t_phase[-1] - t_phase[0]
            distance = np.sum(np.sqrt(np.diff(x_phase) ** 2 + np.diff(y_phase) ** 2))
            fuel_used = fuel_phase[0] - fuel_phase[-1]
            fuel_rate = fuel_used / duration

            phase_durations.append(duration)
            phase_distances.append(distance)
            phase_fuel_rates.append(fuel_rate)

        x_pos = np.arange(len(phase_names))
        width = 0.25

        ax.bar(x_pos - width, phase_durations, width, label="Duration (s)", alpha=0.8)
        ax.bar(x_pos, [d / 10 for d in phase_distances], width, label="Distance (10m)", alpha=0.8)
        ax.bar(
            x_pos + width,
            [r * 10 for r in phase_fuel_rates],
            width,
            label="Fuel Rate (10×kg/s)",
            alpha=0.8,
        )

        ax.set_xlabel("Mission Phase")
        ax.set_ylabel("Normalized Values")
        ax.set_title("Phase Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phase_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mission efficiency metrics
        ax = axes[1, 2]
        metrics = [
            "Time\nEfficiency",
            "Fuel\nEfficiency",
            "Path\nEfficiency",
            "Control\nSmoothness",
        ]

        # Calculate normalized efficiency scores (0-1, higher is better)
        total_time = complete_time[-1]
        total_fuel = complete_fuel[0] - complete_fuel[-1]
        total_distance = np.sum(
            np.sqrt(
                np.diff(multiphase_solution["x_position"]) ** 2
                + np.diff(multiphase_solution["y_position"]) ** 2
            )
        )
        straight_distance = np.sqrt((40 - 0) ** 2 + (35 - 0) ** 2)

        time_efficiency = min(1.0, 30.0 / total_time)  # Normalize to 30s baseline
        fuel_efficiency = min(1.0, 5.0 / total_fuel)  # Normalize to 5kg baseline
        path_efficiency = straight_distance / total_distance

        # Control smoothness (lower variance is better)
        all_steering = multiphase_solution["steering_angle"]
        all_accel = multiphase_solution["acceleration"]
        steering_smoothness = max(0, 1.0 - np.var(all_steering) * 10)
        accel_smoothness = max(0, 1.0 - np.var(all_accel) * 2)
        control_smoothness = (steering_smoothness + accel_smoothness) / 2

        scores = [time_efficiency, fuel_efficiency, path_efficiency, control_smoothness]
        colors = ["green" if s > 0.7 else "orange" if s > 0.5 else "red" for s in scores]

        bars = ax.bar(metrics, scores, color=colors, alpha=0.7)
        ax.set_ylabel("Efficiency Score (0-1)")
        ax.set_title("Mission Efficiency Metrics")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add score labels on bars
        for bar, score in zip(bars, scores, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

        print("   Comprehensive multiphase analysis plot created")

    print("\n3. Summary and recommendations completed")
    return True


# ============================================================================
# EXECUTE COMPREHENSIVE DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("MAPTOR Solution Access API - Comprehensive Reference Implementation")
    print("=" * 80)

    # Run the comprehensive solution access demonstration
    extracted_data = demonstrate_solution_access_api()

    # Create advanced visualizations
    create_comprehensive_visualizations()

    # Display summary
    print("\n" + "=" * 80)
    print("REFERENCE IMPLEMENTATION SUMMARY")
    print("=" * 80)

    if extracted_data["single_phase"]:
        sp_data = extracted_data["single_phase"]
        print("Single-phase obstacle avoidance:")
        print(f"  ✓ Mission time: {sp_data['mission_metadata']['objective']:.3f} seconds")
        print(f"  ✓ Path efficiency: {sp_data['analysis_results']['path_efficiency']:.3f}")
        print(f"  ✓ Safety clearance: {sp_data['analysis_results']['safety_clearance']:.2f} m")

    if extracted_data["multiphase"]:
        mp_data = extracted_data["multiphase"]
        print("Multiphase autonomous mission:")
        print(
            f"  ✓ Mission duration: {mp_data['performance_analysis']['mission_duration_minutes']:.1f} minutes"
        )
        print(f"  ✓ Fuel consumed: {mp_data['performance_analysis']['fuel_consumed']:.2f} kg")
        print(
            f"  ✓ Fuel efficiency: {mp_data['performance_analysis']['fuel_efficiency_kg_per_km']:.3f} kg/km"
        )
        print(
            f"  ✓ Parameters optimized: {mp_data['optimization_results'].get('static_parameters', {}).get('count', 0)}"
        )

    print("\n✓ All MAPTOR solution access capabilities demonstrated successfully")
    print("✓ Reference implementation complete with practical examples")
    print("✓ Both single-phase and multiphase scenarios covered")
    print("✓ Advanced analysis patterns and visualizations provided")

    print("\nThis reference demonstrates:")
    print("  • Status validation and error handling")
    print("  • Mission-wide and phase-specific trajectory access")
    print("  • Variable existence checking and safe access patterns")
    print("  • Static parameter and integral values extraction")
    print("  • Adaptive algorithm analysis and diagnostics")
    print("  • Mesh configuration and computational cost analysis")
    print("  • Advanced performance metrics and safety analysis")
    print("  • Custom data preparation for external analysis")
    print("  • Comprehensive visualization techniques")
    print("  • Robust error handling and validation patterns")
