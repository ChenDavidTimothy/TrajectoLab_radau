import maptor as mtor


# Engine sizing optimization with mass penalty
problem = mtor.Problem("Engine Sizing Optimization")
phase = problem.set_phase(1)

# Design parameter: maximum engine thrust capability
max_thrust = problem.parameter("max_thrust", boundary=(1000, 5000))

# Physical parameters
base_mass = 100.0  # kg (vehicle dry mass)
engine_mass_factor = 0.05  # kg per Newton (engine specific mass)
gravity = 9.81  # m/s²

# Mission variables
t = phase.time(initial=0.0)
altitude = phase.state("altitude", initial=0.0, final=1000.0)
velocity = phase.state("velocity", initial=0.0, final=0.0)
thrust = phase.control("thrust", boundary=(0, None))

# Engine cannot exceed design capability
phase.path_constraints(thrust <= max_thrust)

# Total vehicle mass increases with engine size
total_mass = base_mass + max_thrust * engine_mass_factor

# Vertical flight dynamics with gravity
phase.dynamics({altitude: velocity, velocity: thrust / total_mass - gravity})

# Objective: minimize mission time + engine mass penalty
engine_mass_cost = max_thrust * engine_mass_factor * 0.1  # Cost per kg of engine
problem.minimize(t.final + engine_mass_cost)

# Mesh configuration
phase.mesh([6], [-1.0, 1.0])


phase.guess(terminal_time=50.0)

# Solve with adaptive mesh refinement
solution = mtor.solve_adaptive(problem)

# Results
if solution.status["success"]:
    optimal_thrust = solution.parameters["values"][0]
    engine_mass = optimal_thrust * engine_mass_factor
    total_vehicle_mass = base_mass + engine_mass
    mission_time = solution.status["objective"] - engine_mass * 0.1

    print("Optimal Engine Design:")
    print(f"  Max thrust capability: {optimal_thrust:.0f} N")
    print(f"  Engine mass: {engine_mass:.1f} kg")
    print(f"  Total vehicle mass: {total_vehicle_mass:.1f} kg")
    print(f"  Mission time: {mission_time:.1f} seconds")
    print(f"  Thrust-to-weight ratio: {optimal_thrust / (total_vehicle_mass * gravity):.2f}")

    solution.plot()
else:
    print(f"Optimization failed: {solution.status['message']}")
