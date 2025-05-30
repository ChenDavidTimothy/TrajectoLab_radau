# examples/rocket_ascent_coast_multiphase.py
"""
TrajectoLab Example: True Multiphase Rocket Problem
Demonstrates cross-phase constraints and unified NLP optimization.

Phase 1: Powered ascent with thrust control
Phase 2: Coasting phase with no thrust
Cross-phase constraints ensure state continuity
Objective: Minimize fuel consumption while maximizing final altitude
"""

import numpy as np

import trajectolab as tl


# Physical constants
g = 9.81  # Gravity (m/s^2)
Isp = 300  # Specific impulse (s)
m0 = 1000  # Initial mass (kg)
max_thrust = 15000  # Maximum thrust (N)

# Create multiphase rocket problem
problem = tl.Problem("Multiphase Rocket Ascent and Coast")

# Static parameter: total mission time constraint
T_mission = problem.parameter("total_mission_time", boundary=(10.0, 100.0))

# Phase 1: Powered Ascent (0 to T1)
with problem.phase(1) as ascent:
    print("Defining Phase 1: Powered Ascent")

    # Time for ascent phase (free final time)
    t1 = ascent.time(initial=0.0, final=(5.0, 50.0))

    # States: altitude, velocity, mass
    h1 = ascent.state("altitude", initial=0.0, boundary=(0.0, None))  # Start at ground
    v1 = ascent.state("velocity", initial=0.0, boundary=(-50.0, 200.0))  # Start at rest
    m1 = ascent.state("mass", initial=m0, boundary=(100.0, m0))  # Mass decreases

    # Control: thrust (bounded)
    T1 = ascent.control("thrust", boundary=(0.0, max_thrust))

    # Dynamics for powered ascent
    ascent.dynamics(
        {
            h1: v1,  # altitude rate = velocity
            v1: T1 / m1 - g,  # acceleration = thrust/mass - gravity
            m1: -T1 / (Isp * g),  # mass rate = -thrust/(Isp * g)
        }
    )

    # Fuel consumption integral for phase 1
    fuel_rate1 = T1 / (Isp * g)
    fuel_used1 = ascent.add_integral(fuel_rate1)

    # Mesh for ascent phase
    ascent.set_mesh([8, 8], [-1.0, 0.0, 1.0])

# Phase 2: Coasting (T1 to T2)
with problem.phase(2) as coast:
    print("Defining Phase 2: Coasting Flight")

    # Time for coast phase (T1 to some final time)
    t2 = coast.time(initial=t1.final, final=t1.final + 20.0)

    # States: altitude, velocity, mass (mass constant during coast)
    h2 = coast.state("altitude", initial=h1.final, boundary=(0.0, None))
    v2 = coast.state("velocity", initial=v1.final, boundary=(-100.0, 200.0))
    m2 = coast.state("mass", initial=m1.final, boundary=(100.0, m0))

    # No control during coasting (or zero thrust)
    T2 = coast.control("thrust", boundary=(0.0, 0.0))  # No thrust available

    # Dynamics for coasting (ballistic flight)
    coast.dynamics(
        {
            h2: v2,  # altitude rate = velocity
            v2: -g,  # acceleration = -gravity (no thrust)
            m2: 0,  # mass constant (no fuel burn)
        }
    )

    # No fuel consumption during coast
    fuel_used2 = coast.add_integral(0.0)  # Zero fuel consumption

    # Mesh for coast phase
    coast.set_mesh([6, 6], [-1.0, 0.0, 1.0])

# Cross-phase constraints (Event constraints linking phases)
print("Defining Cross-Phase Constraints")

# State continuity constraints (automatic through initial conditions above)
# These are already enforced by setting initial conditions of phase 2 to final of phase 1


# Additional mission constraints
problem.subject_to(t2.final <= T_mission)  # Total mission time limit
problem.subject_to(h2.final >= 1000.0)  # Minimum final altitude requirement
problem.subject_to(v2.final >= -10.0)  # Don't crash (reasonable final velocity)

# Multi-objective: minimize fuel consumption, maximize final altitude
total_fuel_used = fuel_used1 + fuel_used2
final_altitude = h2.final

# Weighted objective: minimize fuel, maximize altitude
alpha = 0.1  # Weight for altitude term
problem.minimize(total_fuel_used - alpha * final_altitude)

# Set initial guess for multiphase problem
print("Setting Initial Guess")

# Phase 1 initial guess
ascent_states_guess = []
ascent_controls_guess = []
for N in [8, 8]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2  # Normalize to [0, 1]

    # Reasonable ascent trajectory guess
    h_vals = 500 * t_norm**2  # Quadratic altitude growth
    v_vals = 50 * t_norm  # Linear velocity growth
    m_vals = m0 - 200 * t_norm  # Linear mass decrease

    ascent_states_guess.append(np.vstack([h_vals, v_vals, m_vals]))
    ascent_controls_guess.append(np.full((1, N), 8000))  # Moderate thrust

# Phase 2 initial guess
coast_states_guess = []
coast_controls_guess = []
for N in [6, 6]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Ballistic trajectory guess
    h_vals = 500 + 25 * t_norm - 4.9 * t_norm**2  # Parabolic trajectory
    v_vals = 25 - 9.81 * t_norm  # Linear velocity decrease
    m_vals = np.full(N + 1, 800)  # Constant mass

    coast_states_guess.append(np.vstack([h_vals, v_vals, m_vals]))
    coast_controls_guess.append(np.zeros((1, N)))  # No thrust


# Solve the multiphase problem
print("Solving Multiphase Rocket Problem...")
print("=" * 50)

solution = tl.solve_fixed_mesh(
    problem,
    nlp_options={
        "ipopt.print_level": 3,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
    },
)

# Results and Analysis
if solution.success:
    print("\n" + "=" * 50)
    print("MULTIPHASE ROCKET MISSION SUCCESS!")
    print("=" * 50)

    # Mission summary
    print(f"Objective Value: {solution.objective:.4f}")
    print(f"Total Mission Time: {solution.get_total_mission_time():.2f} s")

    # Phase 1 results
    t1_duration = solution.get_phase_duration(1)
    print(f"\nPhase 1 (Ascent): {t1_duration:.2f} s")
    print(f"  Final altitude: {solution[(1, 'altitude')][-1]:.1f} m")
    print(f"  Final velocity: {solution[(1, 'velocity')][-1]:.1f} m/s")
    print(f"  Final mass: {solution[(1, 'mass')][-1]:.1f} kg")
    print(f"  Fuel consumed: {solution.phase_integrals[1]:.1f} kg")

    # Phase 2 results
    t2_duration = solution.get_phase_duration(2)
    print(f"\nPhase 2 (Coast): {t2_duration:.2f} s")
    print(f"  Final altitude: {solution[(2, 'altitude')][-1]:.1f} m")
    print(f"  Final velocity: {solution[(2, 'velocity')][-1]:.1f} m/s")
    print(f"  Mass (constant): {solution[(2, 'mass')][-1]:.1f} kg")

    # Mission analysis
    total_fuel = solution.phase_integrals[1] + solution.phase_integrals[2]
    fuel_fraction = total_fuel / m0 * 100
    print("\nMission Analysis:")
    print(f"  Total fuel used: {total_fuel:.1f} kg ({fuel_fraction:.1f}% of initial mass)")
    print(
        f"  Maximum altitude: {max(np.max(solution[(1, 'altitude')]), np.max(solution[(2, 'altitude')])):.1f} m"
    )
    print(f"  Static parameter (T_mission): {solution.static_parameters[0]:.2f} s")

    # Verify state continuity at phase boundary
    h1_final = solution[(1, "altitude")][-1]
    h2_initial = solution[(2, "altitude")][0]
    v1_final = solution[(1, "velocity")][-1]
    v2_initial = solution[(2, "velocity")][0]
    m1_final = solution[(1, "mass")][-1]
    m2_initial = solution[(2, "mass")][0]

    print("\nState Continuity Verification:")
    print(f"  Altitude: {h1_final:.6f} → {h2_initial:.6f} (diff: {abs(h1_final - h2_initial):.2e})")
    print(f"  Velocity: {v1_final:.6f} → {v2_initial:.6f} (diff: {abs(v1_final - v2_initial):.2e})")
    print(f"  Mass: {m1_final:.6f} → {m2_initial:.6f} (diff: {abs(m1_final - m2_initial):.2e})")

    # Plot multiphase solution
    print("\nPlotting multiphase trajectory...")
    solution.plot(show_phase_boundaries=True)  # Show all phases with boundaries

    # Also plot individual phases
    print("Plotting individual phases...")
    solution.plot(phase_id=1)  # Ascent phase only
    solution.plot(phase_id=2)  # Coast phase only

    print("\n" + "=" * 60)
    print("SUCCESS: True multiphase optimization with cross-phase coupling!")
    print("- Unified NLP solved both phases simultaneously")
    print("- Cross-phase constraints maintained state continuity")
    print("- Static parameters controlled global mission constraints")
    print("- Mixed objectives optimized across multiple phases")
    print("=" * 60)

else:
    print(f"\nMultiphase solution failed: {solution.message}")
    print("This might indicate:")
    print("- Infeasible cross-phase constraints")
    print("- Poor initial guess for multiphase problem")
    print("- Solver parameter tuning needed")

    # Print detailed problem info for debugging
    print("\nProblem Structure:")
    print(f"  Phases: {solution.get_phase_ids()}")
    for phase_id in problem.get_phase_ids():
        num_states, num_controls = problem.get_phase_variable_counts(phase_id)
        print(f"  Phase {phase_id}: {num_states} states, {num_controls} controls")
