import numpy as np

import trajectolab as tl


# Create multi-phase problem
mp_problem = tl.MultiPhaseProblem("Spacecraft Mission")
# Add phases
ascent = mp_problem.add_phase("Ascent")
coast = mp_problem.add_phase("Coast")
descent = mp_problem.add_phase("Descent")
# Configure ascent phase
t1 = ascent.time(initial=0.0)
h1 = ascent.state("altitude", initial=0.0)
v1 = ascent.state("velocity", initial=0.0)
u1 = ascent.control("thrust", boundary=(0.0, 1.0))
ascent.dynamics({h1: v1, v1: u1})
ascent.set_mesh([10], np.array([-1.0, 1.0]))
# Configure coast phase
t2 = coast.time()
h2 = coast.state("altitude")
v2 = coast.state("velocity")
coast.dynamics({h2: v2, v2: 0})  # Ballistic coast
coast.set_mesh([5], np.array([-1.0, 1.0]))
# Configure descent phase
t3 = descent.time()
h3 = descent.state("altitude", final=0.0)
v3 = descent.state("velocity", final=0.0)
u3 = descent.control("thrust", boundary=(-1.0, 0.0))
descent.dynamics({h3: v3, v3: u3})
descent.set_mesh([15], np.array([-1.0, 1.0]))
# Link phases with continuity constraints
mp_problem.link_phases(h1.final == h2.initial)  # Altitude continuity
mp_problem.link_phases(v1.final == v2.initial)  # Velocity continuity
mp_problem.link_phases(t1.final == t2.initial)  # Time continuity
mp_problem.link_phases(h2.final == h3.initial)
mp_problem.link_phases(v2.final == v3.initial)
mp_problem.link_phases(t2.final == t3.initial)
# Add global parameters
gravity = mp_problem.add_global_parameter("gravity", 9.81)
mass = mp_problem.add_global_parameter("vehicle_mass", 1000.0)
# Set global objective (minimize total mission time)
total_time = t1.final - t1.initial + t2.final - t2.initial + t3.final - t3.initial
mp_problem.set_global_objective(total_time)
# Solve multi-phase problem
solution = tl.solve_multi_phase_fixed_mesh(mp_problem)
if solution.success:
    print(f"Optimal mission time: {solution.objective:.3f} seconds")
    print(f"Global parameters: {solution.global_parameters}")
    solution.plot_phases()
    solution.print_solution_summary()
    # Access individual phases
    ascent_solution = solution.get_phase_solution(0)
    print(f"Ascent duration: {ascent_solution.final_time:.3f} seconds")
