Working with Solution Data
==========================

This tutorial demonstrates comprehensive solution data extraction using MAPTOR's ``Solution`` class.
We'll work through a complete two-phase Schwartz optimal control problem to show every available
method for accessing and analyzing optimization results.

Overview of Solution Access
----------------------------

MAPTOR's ``Solution`` class provides a unified interface for accessing optimization results
across single-phase and multiphase problems. The key access patterns are:

* **Dictionary-style access**: ``solution["variable"]`` for mission-wide trajectories
* **Tuple access**: ``solution[(phase_id, "variable")]`` for phase-specific data
* **Property access**: ``solution.status``, ``solution.phases``, ``solution.adaptive``
* **Metadata bundles**: Comprehensive timing, mesh, and convergence information

Example Problem: Two-Phase Schwartz
------------------------------------

We'll use a two-phase Schwartz problem with elliptical path constraints and phase transitions.
This showcases multiphase solution access patterns and advanced constraint handling.

.. code-block:: python

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

    phase1.dynamics({
        x0_1: x1_1,
        x1_1: u1 - 0.1 * (1 + 2 * x0_1**2) * x1_1,
    })

    # Path constraint: feasible region outside ellipse
    elliptical_constraint = 1 - 9 * (x0_1 - 1) ** 2 - ((x1_1 - 0.4) / 0.3) ** 2
    phase1.path_constraints(elliptical_constraint <= 0)
    phase1.mesh([6, 6], [-1.0, 0.0, 1.0])

    # Phase 2
    phase2 = problem.set_phase(2)
    phase2.time(initial=1.0, final=2.9)
    x0_2 = phase2.state("x0", initial=x0_1.final)
    x1_2 = phase2.state("x1", initial=x1_1.final)
    u2 = phase2.control("u")

    phase2.dynamics({
        x0_2: x1_2,
        x1_2: u2 - 0.1 * (1 + 2 * x0_2**2) * x1_2,
    })
    phase2.mesh([8, 8], [-1.0, 0.0, 1.0])

    # Objective
    objective_expr = 5 * (x0_2.final**2 + x1_2.final**2)
    problem.minimize(objective_expr)

    # Solve with adaptive mesh refinement
    solution = mtor.solve_adaptive(
        problem,
        error_tolerance=1e-4,
        max_iterations=25,
        nlp_options={"ipopt.print_level": 0}
    )

Essential Solution Validation
-----------------------------

**Always validate solution status before accessing data:**

.. code-block:: python

    # Basic status validation
    success = solution.status["success"]
    objective = solution.status["objective"]
    total_time = solution.status["total_mission_time"]
    message = solution.status["message"]

    print(f"Success: {success}")
    print(f"Objective: {objective:.12e}")
    print(f"Mission time: {total_time:.6f}")

    if not success:
        print(f"Optimization failed: {message}")
        return  # Stop processing if failed

The ``status`` property is a dictionary containing:

* ``success`` (bool): Whether optimization succeeded
* ``objective`` (float): Final objective function value
* ``total_mission_time`` (float): Total time across all phases
* ``message`` (str): Detailed solver status message

Mission-Wide Trajectory Access
-------------------------------

Use string keys to automatically combine trajectory data from all phases:

.. code-block:: python

    # Complete mission trajectories (auto-concatenated)
    x0_trajectory = solution["x0"]           # All phases combined
    x1_trajectory = solution["x1"]           # All phases combined
    u_trajectory = solution["u"]             # All phases combined
    time_states = solution["time_states"]    # Complete time array
    time_controls = solution["time_controls"] # Control time points

    print(f"Complete mission:")
    print(f"x0: {len(x0_trajectory)} points, values {x0_trajectory}")
    print(f"x1: {len(x1_trajectory)} points, values {x1_trajectory}")
    print(f"u: {len(u_trajectory)} points, values {u_trajectory}")

String key access automatically:

* Concatenates data from all phases containing the variable
* Maintains temporal order (phases combined by ID)
* Preserves ``np.float64`` precision for numerical safety
* Provides seamless mission-wide trajectories for analysis

Phase-Specific Data Access
---------------------------

Use tuple keys for granular control over individual phase data:

.. code-block:: python

    # Phase-specific trajectory access
    phase_ids = [1, 2]  # Available phases

    for phase_id in phase_ids:
        x0_phase = solution[(phase_id, "x0")]
        x1_phase = solution[(phase_id, "x1")]
        u_phase = solution[(phase_id, "u")]
        t_states_phase = solution[(phase_id, "time_states")]
        t_controls_phase = solution[(phase_id, "time_controls")]

        print(f"Phase {phase_id} trajectories:")
        print(f"  x0: {len(x0_phase)} points, [{x0_phase[0]:.6f} to {x0_phase[-1]:.6f}]")
        print(f"  x1: {len(x1_phase)} points, [{x1_phase[0]:.6f} to {x1_phase[-1]:.6f}]")
        print(f"  u: {len(u_phase)} points, [{u_phase[0]:.6f} to {u_phase[-1]:.6f}]")

Tuple access pattern ``(phase_id, variable_name)`` provides:

* Complete control over which phase data to access
* Essential for analyzing phase-specific characteristics
* Required for phase boundary analysis and transitions
* Enables different processing for different mission segments

Phase Information Analysis
--------------------------

The ``phases`` property provides comprehensive metadata for each phase:

.. code-block:: python

    # Examine phase structure and timing
    phase_ids = list(solution.phases.keys())
    print(f"Number of phases: {len(phase_ids)}")

    for phase_id in phase_ids:
        phase_data = solution.phases[phase_id]

        # Phase timing information
        initial_time = phase_data["times"]["initial"]
        final_time = phase_data["times"]["final"]
        duration = phase_data["times"]["duration"]

        # Phase variables
        state_names = phase_data["variables"]["state_names"]
        control_names = phase_data["variables"]["control_names"]
        num_states = phase_data["variables"]["num_states"]
        num_controls = phase_data["variables"]["num_controls"]

        # Mesh configuration
        num_intervals = phase_data["mesh"]["num_intervals"]
        polynomial_degrees = phase_data["mesh"]["polynomial_degrees"]

        print(f"Phase {phase_id}:")
        print(f"  Time: [{initial_time:.6f}, {final_time:.6f}], duration: {duration:.6f}")
        print(f"  Variables: {num_states} states, {num_controls} controls")
        print(f"  States: {state_names}")
        print(f"  Controls: {control_names}")
        print(f"  Mesh: {num_intervals} intervals, degrees: {polynomial_degrees}")

Each phase bundle contains:

* **Timing**: Initial time, final time, and duration
* **Variables**: State and control names with counts
* **Mesh**: Intervals, polynomial degrees, and node locations
* **Time Arrays**: State and control time coordinate arrays

Variable Existence Checking
----------------------------

Safely validate variable availability before accessing:

.. code-block:: python

    # Check variable existence to prevent KeyError
    available_vars = []
    test_vars = ["x0", "x1", "u", "time_states", "time_controls"]

    for var in test_vars:
        if var in solution:
            available_vars.append(var)

    print(f"Available variables: {available_vars}")

    # Safe access pattern
    if "altitude" in solution:
        altitude_data = solution["altitude"]
        print(f"Altitude range: {altitude_data.min():.1f} to {altitude_data.max():.1f}")
    else:
        print("Altitude not available in this solution")

    # Phase-specific checking
    if (1, "thrust") in solution:
        thrust_data = solution[(1, "thrust")]
        max_thrust = thrust_data.max()
        print(f"Maximum thrust in phase 1: {max_thrust:.2f}")

The ``in`` operator works with both string and tuple keys, enabling robust solution processing
workflows that gracefully handle missing variables.

Adaptive Algorithm Analysis
---------------------------

When using ``solve_adaptive``, examine convergence and refinement performance:

.. code-block:: python

    # Adaptive algorithm results (only available for adaptive solutions)
    if solution.adaptive is not None:
        converged = solution.adaptive["converged"]
        iterations = solution.adaptive["iterations"]
        tolerance = solution.adaptive["target_tolerance"]
        phase_converged = solution.adaptive["phase_converged"]
        final_errors = solution.adaptive["final_errors"]

        print("Adaptive refinement:")
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
        print("Fixed mesh solution - no adaptive data available")

The ``adaptive`` property provides:

* **Convergence status**: Whether target tolerance was achieved
* **Iteration count**: Number of refinement cycles performed
* **Phase-specific convergence**: Per-phase convergence status
* **Error estimates**: Final error estimates for each phase and interval

Static Parameter Access
-----------------------

Extract optimized static parameters (if defined in the problem):

.. code-block:: python

    # Static parameters (constant throughout mission but optimized)
    if solution.parameters is not None:
        param_count = solution.parameters["count"]
        param_values = solution.parameters["values"]
        param_names = solution.parameters["names"]

        print("Static parameters:")
        print(f"  Count: {param_count}")

        if param_names is not None:
            for name, value in zip(param_names, param_values, strict=False):
                print(f"  {name}: {value:.12e}")
        else:
            for i, value in enumerate(param_values):
                print(f"  param_{i}: {value:.12e}")
    else:
        print("No static parameters in this problem")

Static parameters are optimization variables that remain constant throughout the mission
but are determined by the solver (e.g., optimal vehicle mass, design parameters).

Advanced Data Extraction
-------------------------

Perform comprehensive mission analysis and bounds calculation:

.. code-block:: python

    # Mission final values (complete mission, not just first phase)
    final_x0 = solution["x0"][-1]
    final_x1 = solution["x1"][-1]
    final_time = solution["time_states"][-1]

    print("Mission final state:")
    print(f"x0_final: {final_x0:.12e}")
    print(f"x1_final: {final_x1:.12e}")
    print(f"time_final: {final_time:.6f}")

    # Mission data analysis
    x0_trajectory = solution["x0"]
    x1_trajectory = solution["x1"]
    u_trajectory = solution["u"]
    time_states = solution["time_states"]

    # Time span analysis
    t_start = time_states[0]
    t_end = time_states[-1]
    t_mid = time_states[len(time_states) // 2]

    print(f"Time span: {t_start:.6f} to {t_end:.6f}")
    print(f"Midpoint time: {t_mid:.6f}")

    # State space bounds
    x0_min, x0_max = x0_trajectory.min(), x0_trajectory.max()
    x1_min, x1_max = x1_trajectory.min(), x1_trajectory.max()
    u_min, u_max = u_trajectory.min(), u_trajectory.max()

    print(f"State bounds: x0=[{x0_min:.6f}, {x0_max:.6f}], x1=[{x1_min:.6f}, {x1_max:.6f}]")
    print(f"Control bounds: u=[{u_min:.6f}, {u_max:.6f}]")

This demonstrates extracting key mission statistics for performance analysis and validation.

Mesh Configuration Analysis
----------------------------

Examine detailed mesh refinement and computational efficiency:

.. code-block:: python

    # Detailed mesh analysis
    print("Mesh configuration:")
    total_intervals = 0

    for phase_id in solution.phases.keys():
        mesh_data = solution.phases[phase_id]["mesh"]
        nodes = mesh_data["mesh_nodes"]
        degrees = mesh_data["polynomial_degrees"]
        intervals = mesh_data["num_intervals"]
        total_intervals += intervals

        print(f"  Phase {phase_id}: {intervals} intervals, degrees {degrees}")

        # Show node locations for small meshes
        if len(nodes) <= 10:
            nodes_str = "[" + ", ".join(f"{n:.4f}" for n in nodes) + "]"
            print(f"    Nodes: {nodes_str}")

    print(f"  Total intervals: {total_intervals}")

Mesh details are essential for:

* Understanding computational cost and accuracy trade-offs
* Validating adaptive refinement performance
* Analyzing where the algorithm concentrated computational effort

Custom Visualization Preparation
---------------------------------

Prepare data for advanced plotting beyond the built-in ``solution.plot()``:

.. code-block:: python

    # Prepare data for custom analysis and plotting
    mission_data = {
        "time_states": solution["time_states"],
        "time_controls": solution["time_controls"],
        "x0": solution["x0"],
        "x1": solution["x1"],
        "u": solution["u"],
    }

    print("Data extraction complete:")
    print(f"  State points: {len(mission_data['time_states'])}")
    print(f"  Control points: {len(mission_data['time_controls'])}")

    # Phase-colored trajectory plotting
    plt.figure(figsize=(15, 10))

    # Mission-wide trajectory
    plt.subplot(2, 3, 1)
    plt.plot(mission_data["time_states"], mission_data["x0"])
    plt.xlabel("Time")
    plt.ylabel("x0")
    plt.title("State x0")
    plt.grid(True)

    # Phase-specific coloring
    plt.subplot(2, 3, 2)
    phase_colors = ["blue", "red"]
    for i, phase_id in enumerate([1, 2]):
        x0_phase = solution[(phase_id, "x0")]
        x1_phase = solution[(phase_id, "x1")]
        plt.plot(x0_phase, x1_phase, color=phase_colors[i], label=f"Phase {phase_id}")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Phase-Colored Trajectory")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

Built-in Solution Methods
-------------------------

MAPTOR provides convenient built-in methods for common tasks:

.. code-block:: python

    # Built-in comprehensive plotting
    solution.plot(show_phase_boundaries=True)

    # Comprehensive solution summary
    solution.summary(comprehensive=True)

    # Quick validation summary
    solution.summary(comprehensive=False)

Error Handling Best Practices
------------------------------

Implement robust solution processing with proper error handling:

.. code-block:: python

    def process_solution_safely(solution):
        """Demonstrate robust solution processing."""

        # Always check success first
        if not solution.status["success"]:
            print(f"Solution failed: {solution.status['message']}")
            return None

        # Validate required variables exist
        required_vars = ["x0", "x1", "u"]
        missing_vars = [var for var in required_vars if var not in solution]

        if missing_vars:
            print(f"Missing required variables: {missing_vars}")
            return None

        # Safe data extraction
        try:
            mission_data = {
                "objective": solution.status["objective"],
                "total_time": solution.status["total_mission_time"],
                "trajectories": {var: solution[var] for var in required_vars},
                "final_state": {var: solution[var][-1] for var in ["x0", "x1"]},
            }

            print("✓ Solution processed successfully")
            return mission_data

        except Exception as e:
            print(f"Error processing solution: {e}")
            return None

    # Use the robust processor
    processed_data = process_solution_safely(solution)
    if processed_data:
        print(f"Final objective: {processed_data['objective']:.6e}")

Best Practices Summary
----------------------

1. **Always validate success** before accessing solution data
2. **Use string keys** for mission-wide trajectory analysis
3. **Use tuple keys** for phase-specific investigations
4. **Check variable existence** with ``in`` operator for robust code
5. **Examine adaptive results** to understand algorithm performance
6. **Bundle metadata access** using ``.phases`` and ``.adaptive`` properties
7. **Preserve data precision** - all arrays maintain ``np.float64`` precision
8. **Handle errors gracefully** with proper validation and exception handling
9. **Use built-in methods** (``plot()``, ``summary()``) for common tasks
10. **Prepare custom analysis** by extracting data into structured dictionaries

Common Pitfalls to Avoid
-------------------------

* **Don't access data from failed solutions** - always check ``status["success"]`` first
* **Don't assume variables exist** - use ``in`` operator before accessing
* **Don't mix up string vs tuple access** - understand when to use each pattern
* **Don't ignore adaptive information** - it provides crucial algorithm insights
* **Don't hardcode phase IDs** - use ``solution.phases.keys()`` for robustness

Next Steps
----------

* Explore the built-in ``solution.plot()`` method for quick visualization
* Review the ``solution.summary()`` method for comprehensive diagnostics
* Practice with different optimal control problems from the examples gallery
* Reference the complete API documentation for advanced usage patterns



Running the Complete Example
----------------------------

The complete, runnable implementation is available at:
``docs/source/tutorials/solution_access/reference_implementation.py``

Run it from the project root:

.. code-block:: bash

    python docs/source/tutorials/solution_access/reference_implementation.py
