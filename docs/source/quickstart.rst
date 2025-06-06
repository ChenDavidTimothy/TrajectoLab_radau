
5-Minute Quickstart
===================

This guide gets you solving optimal control problems in 5 minutes.

Basic Problem Structure
-----------------------

Every MAPTOR problem follows this pattern:

1. **Create Problem**
2. **Define Variables** (states, controls, time)
3. **Set Dynamics**
4. **Define Objective**
5. **Configure Mesh**
6. **Solve**

Example: Minimum Time Problem
-----------------------------

.. code-block:: python

    import maptor as tl
    import numpy as np

    # 1. Create problem
    problem = tl.Problem("Minimum Time")

    # 2. Define variables
    t = problem.time(initial=0.0)                          # Free final time
    x = problem.state("position", initial=0.0, final=1.0)  # Position: 0 → 1
    v = problem.state("velocity", initial=0.0)             # Velocity: start at rest
    u = problem.control("force", boundary=(-2.0, 2.0))     # Bounded control

    # 3. Set dynamics
    problem.dynamics({
        x: v,        # dx/dt = v
        v: u         # dv/dt = u
    })

    # 4. Define objective
    problem.minimize(t.final)  # Minimize final time

    # 5. Configure mesh and solve
    problem.mesh([10], np.array([-1.0, 1.0]))
    solution = tl.solve_fixed_mesh(problem)

    # 6. Results
    if solution.status["success"]:
        print(f"Minimum time: {solution.final_time:.3f} seconds")
        solution.plot()

Key Patterns
------------

**Constraint Specification:**

.. code-block:: python

    # Equality constraints
    x = problem.state("x", initial=5.0)           # x(0) = 5
    x = problem.state("x", final=10.0)            # x(tf) = 10

    # Inequality constraints
    x = problem.state("x", boundary=(-1.0, 1.0))  # -1 ≤ x(t) ≤ 1
    u = problem.control("u", boundary=(0.0, None)) # u ≥ 0

**Solver Selection:**

.. code-block:: python

    # Fixed mesh - fast
    solution = tl.solve_fixed_mesh(problem)

    # Adaptive mesh - high accuracy
    solution = tl.solve_adaptive(problem, error_tolerance=1e-8)

**Working with Solutions:**

.. code-block:: python

    if solution.status["success"]:
        # Get trajectory data
        time, position = solution.get_trajectory("position")
        time, velocity = solution.get_trajectory("velocity")

        # Plot results
        solution.plot()

        # Access final values
        print(f"Final time: {solution.final_time}")
        print(f"Objective: {solution.status['objective']}")

Next Steps
----------

* Explore the examples gallery
* Check the API reference
* Try adaptive mesh refinement for high-accuracy solutions
