"""
TrajectoLab Implementation: Example 10.47 gdrd07 SINGULAR ARC PROBLEM

Three-phase rocket trajectory optimization:
- Phase 1: Maximum Thrust
- Phase 2: Singular Arc
- Phase 3: No Thrust

Objective: Maximize final altitude
"""

import casadi as ca
import numpy as np

import trajectolab as tl


def solve_gdrd07_singular_arc_problem():
    """
    Implement the gdrd07 Singular Arc Problem using TrajectoLab's multiphase framework.

    Returns:
        Problem: Configured TrajectoLab problem ready for solving
    """

    # Physical constants from the problem specification
    T_m = 193.044  # Maximum thrust
    g = 32.174  # Gravitational acceleration
    sigma = 5.49153484923381010e-5  # Atmospheric density parameter
    c = 1580.9425279876559  # Specific impulse parameter
    h0 = 23800.0  # Atmospheric scale height

    problem = tl.Problem("gdrd07 Singular Arc Problem")

    # ========================================================================
    # Phase 1: Maximum Thrust
    # ========================================================================
    with problem.phase(1) as phase1:
        # Time variable: t ∈ [0, t_F^(1)]
        t1 = phase1.time(initial=0.0)

        # State variables with initial conditions
        h1 = phase1.state("altitude", initial=0.0)  # h = 0
        v1 = phase1.state("velocity", initial=0.0)  # v = 0
        m1 = phase1.state("mass", initial=3.0)  # m = 3

        # Dynamics equations (10.356-10.358)
        # ḣ = v
        # v̇ = (1/m)[T_m - σv² exp[-h/h₀]] - g
        # ṁ = -T_m/c
        drag_force = sigma * v1**2 * ca.exp(-h1 / h0)

        phase1.dynamics({h1: v1, v1: (T_m - drag_force) / m1 - g, m1: -T_m / c})

        # Set mesh for Phase 1: 3 intervals need 4 mesh points
        phase1.set_mesh([10, 8, 10], [-1.0, -0.3, 0.3, 1.0])

    # ========================================================================
    # Phase 2: Singular Arc
    # ========================================================================
    with problem.phase(2) as phase2:
        # Time variable: continuous from Phase 1
        t2 = phase2.time(initial=t1.final)

        # State variables with continuity from Phase 1
        h2 = phase2.state("altitude", initial=h1.final)
        v2 = phase2.state("velocity", initial=v1.final)
        m2 = phase2.state("mass", initial=m1.final, final=1.0)  # Final mass constraint

        # Compute complex thrust T_s for singular arc (equation 10.369)
        drag_term = sigma * v2**2 * ca.exp(-h2 / h0)
        mg_term = m2 * g

        # T_s = σv² exp[-h/h₀] + mg + correction_term
        # correction_term = (mg)/(1 + 4(c/v) + 2(c²/v²)) × [c²/(h₀g)(1 + v/c) - 1 - 2c/v]
        denominator = 1 + 4 * (c / v2) + 2 * (c / v2) ** 2
        bracket_term = ((c**2) / (h0 * g) * (1 + v2 / c)) - 1 - 2 * c / v2
        correction_term = mg_term / denominator * bracket_term

        T_s = drag_term + mg_term + correction_term

        # Dynamics equations (10.366-10.368)
        phase2.dynamics({h2: v2, v2: ((T_s - drag_term) / m2) - g, m2: -T_s / c})

        # Boundary condition: 0 = mg - (1 + v/c)σv² exp[-h/h₀]
        boundary_constraint = mg_term - (1 + (v2 / c)) * drag_term
        phase2.subject_to(boundary_constraint == 0.0)
        phase2.subject_to(t2.final - t2.initial >= 1.0)

        # Set mesh for Phase 2: 3 intervals need 4 mesh points
        phase2.set_mesh([8, 6, 8], [-1.0, -0.3, 0.3, 1.0])

    # ========================================================================
    # Phase 3: No Thrust
    # ========================================================================
    with problem.phase(3) as phase3:
        # Time variable: continuous from Phase 2
        t3 = phase3.time(initial=t2.final)

        # State variables with continuity from Phase 2
        h3 = phase3.state("altitude", initial=h2.final)
        v3 = phase3.state("velocity", initial=v2.final, final=0.0)  # Final velocity = 0
        # Note: No mass variable in Phase 3 (no propellant consumption)

        # Dynamics equations (10.370-10.371)
        # ḣ = v
        # v̇ = -σv² exp[-h/h₀] - g
        drag_force_p3 = sigma * v3**2 * ca.exp(-h3 / h0)

        phase3.dynamics({h3: v3, v3: -drag_force_p3 - g})

        # Set mesh for Phase 3: 2 intervals need 3 mesh points
        phase3.set_mesh([8, 8], [-1.0, 0.0, 1.0])

        phase3.subject_to(t3.final - t3.initial >= 1.0)

    # ========================================================================
    # Objective Function
    # ========================================================================

    # Maximize final altitude J = h_F (minimize negative altitude)
    problem.minimize(-h3.final)

    return problem


def create_initial_guess():
    """
    Create reasonable initial guess based on expected solution values.

    Expected solution:
    - J* = 18550.872
    - t_F^(1) = 13.751270
    - t_F^(2) = 21.987363
    - t_F^(3) = 42.887912

    TrajectoLab Initial Guess Format:
    - For mesh intervals [N1, N2, ..., Nk] with num_states state variables:
    - Each state array: shape (num_states, Ni + 1) for interval i
    - Each control array: shape (num_controls, Ni) for interval i
    """

    # ========================================================================
    # Phase 1: mesh [10, 8, 10], states [h, v, m] (3 states), no controls
    # ========================================================================

    states_p1 = []
    controls_p1 = []

    # Create smooth trajectory guess for Phase 1
    t1_duration = 13.75

    # Interval 1: 10 collocation points -> (3, 11) states, (0, 10) controls
    tau1 = np.linspace(-1, 1, 11)
    t_phys1 = (tau1 + 1) / 2 * t1_duration  # Map to [0, t1_duration]
    h1_vals = 0.5 * t_phys1**2  # Rough parabolic altitude
    v1_vals = t_phys1 * 60  # Rough linear velocity increase
    m1_vals = 3.0 - 0.3 * t_phys1 / t1_duration  # Linear mass decrease
    states_p1.append(np.array([h1_vals, v1_vals, m1_vals]))
    controls_p1.append(np.zeros((0, 10)))

    # Interval 2: 8 collocation points -> (3, 9) states, (0, 8) controls
    tau2 = np.linspace(-1, 1, 9)
    t_phys2 = (tau2 + 1) / 2 * t1_duration
    h2_vals = 0.5 * t_phys2**2 + 1000
    v2_vals = t_phys2 * 80
    m2_vals = 3.0 - 0.5 * t_phys2 / t1_duration
    states_p1.append(np.array([h2_vals, v2_vals, m2_vals]))
    controls_p1.append(np.zeros((0, 8)))

    # Interval 3: 10 collocation points -> (3, 11) states, (0, 10) controls
    tau3 = np.linspace(-1, 1, 11)
    t_phys3 = (tau3 + 1) / 2 * t1_duration
    h3_vals = 0.5 * t_phys3**2 + 2000
    v3_vals = t_phys3 * 100
    m3_vals = 3.0 - 0.7 * t_phys3 / t1_duration
    states_p1.append(np.array([h3_vals, v3_vals, m3_vals]))
    controls_p1.append(np.zeros((0, 10)))

    # ========================================================================
    # Phase 2: mesh [8, 6, 8], states [h, v, m] (3 states), no controls
    # ========================================================================

    states_p2 = []
    controls_p2 = []

    t2_duration = 21.99 - 13.75

    # Interval 1: 8 collocation points -> (3, 9) states, (0, 8) controls
    tau1 = np.linspace(-1, 1, 9)
    t_phys1 = (tau1 + 1) / 2 * t2_duration
    h1_vals = 5000 + t_phys1 * 800  # Continue altitude increase
    v1_vals = 800 - t_phys1 * 20  # Slight velocity decrease
    m1_vals = 2.3 - 0.4 * t_phys1 / t2_duration  # Continue mass decrease
    states_p2.append(np.array([h1_vals, v1_vals, m1_vals]))
    controls_p2.append(np.zeros((0, 8)))

    # Interval 2: 6 collocation points -> (3, 7) states, (0, 6) controls
    tau2 = np.linspace(-1, 1, 7)
    t_phys2 = (tau2 + 1) / 2 * t2_duration
    h2_vals = 8000 + t_phys2 * 600
    v2_vals = 750 - t_phys2 * 30
    m2_vals = 1.9 - 0.3 * t_phys2 / t2_duration
    states_p2.append(np.array([h2_vals, v2_vals, m2_vals]))
    controls_p2.append(np.zeros((0, 6)))

    # Interval 3: 8 collocation points -> (3, 9) states, (0, 8) controls
    tau3 = np.linspace(-1, 1, 9)
    t_phys3 = (tau3 + 1) / 2 * t2_duration
    h3_vals = 11000 + t_phys3 * 400
    v3_vals = 700 - t_phys3 * 50
    m3_vals = 1.5 - 0.5 * t_phys3 / t2_duration  # Approach final mass = 1
    states_p2.append(np.array([h3_vals, v3_vals, m3_vals]))
    controls_p2.append(np.zeros((0, 8)))

    # ========================================================================
    # Phase 3: mesh [8, 8], states [h, v] (2 states), no controls
    # ========================================================================

    states_p3 = []
    controls_p3 = []

    t3_duration = 42.89 - 21.99

    # Interval 1: 8 collocation points -> (2, 9) states, (0, 8) controls
    tau1 = np.linspace(-1, 1, 9)
    t_phys1 = (tau1 + 1) / 2 * t3_duration
    h1_vals = 13000 + t_phys1 * 250  # Continue altitude increase to apogee
    v1_vals = 600 - t_phys1 * 30  # Velocity decreases to zero
    states_p3.append(np.array([h1_vals, v1_vals]))
    controls_p3.append(np.zeros((0, 8)))

    # Interval 2: 8 collocation points -> (2, 9) states, (0, 8) controls
    tau2 = np.linspace(-1, 1, 9)
    t_phys2 = (tau2 + 1) / 2 * t3_duration
    h2_vals = 16000 + t_phys2 * 100  # Approach final altitude
    v2_vals = 300 - t_phys2 * 15  # Continue velocity decrease
    states_p3.append(np.array([h2_vals, v2_vals]))
    controls_p3.append(np.zeros((0, 8)))

    return {
        "phase_states": {1: states_p1, 2: states_p2, 3: states_p3},
        "phase_controls": {1: controls_p1, 2: controls_p2, 3: controls_p3},
        "phase_initial_times": {1: 0.0, 2: 13.75, 3: 21.99},
        "phase_terminal_times": {1: 13.75, 2: 21.99, 3: 42.89},
    }


def main():
    """Main execution function."""

    print("=" * 70)
    print("TrajectoLab Implementation: gdrd07 Singular Arc Problem")
    print("=" * 70)

    # Create and configure the problem
    problem = solve_gdrd07_singular_arc_problem()

    # Set initial guess to help convergence
    initial_guess = create_initial_guess()
    problem.set_initial_guess(**initial_guess)

    print("\nSolving three-phase singular arc problem...")
    print("Phase 1: Maximum Thrust")
    print("Phase 2: Singular Arc")
    print("Phase 3: No Thrust")
    print("Objective: Maximize final altitude")

    # Solve with fixed mesh
    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 5,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-8,
            "ipopt.constr_viol_tol": 1e-8,
            "ipopt.mu_strategy": "adaptive",
        },
    )

    # Display results
    if solution.success:
        print("\n" + "=" * 50)
        print("SOLUTION SUCCESSFUL!")
        print("=" * 50)

        final_altitude = -solution.objective
        print(f"Final Altitude (Objective): {final_altitude:.3f}")
        print(f"Phase 1 Final Time: {solution.get_phase_final_time(1):.6f}")
        print(f"Phase 2 Final Time: {solution.get_phase_final_time(2):.6f}")
        print(f"Phase 3 Final Time: {solution.get_phase_final_time(3):.6f}")

        print("\nPhase durations:")
        print(f"Phase 1: {solution.get_phase_duration(1):.6f}")
        print(f"Phase 2: {solution.get_phase_duration(2):.6f}")
        print(f"Phase 3: {solution.get_phase_duration(3):.6f}")

        print("\nExpected benchmark values:")
        print("J* = 18550.872")
        print("t_F^(1) = 13.751270")
        print("t_F^(2) = 21.987363")
        print("t_F^(3) = 42.887912")

        print("\nComputed vs Expected:")
        print(f"Altitude: {final_altitude:.3f} vs 18550.872")
        print(f"t1_final: {solution.get_phase_final_time(1):.6f} vs 13.751270")
        print(f"t2_final: {solution.get_phase_final_time(2):.6f} vs 21.987363")
        print(f"t3_final: {solution.get_phase_final_time(3):.6f} vs 42.887912")

        # Plot the solution
        print("\nGenerating solution plots...")
        solution.plot(show_phase_boundaries=True)

    else:
        print("\n" + "=" * 50)
        print("SOLUTION FAILED!")
        print("=" * 50)
        print(f"Message: {solution.message}")


if __name__ == "__main__":
    main()
