"""
TrajectoLab Implementation: Asymmetric Rigid Body Reorientation (Single Phase)

Example 10.15 asyq01: MINIMUM TIME [111], [29, Sect. 6.8]

This is a spacecraft attitude control problem that demonstrates optimal reorientation
of an asymmetric rigid body from one orientation to another in minimum time using
control torques.

Problem Details:
- 6 state variables: quaternion components (q1,q2,q3) + angular velocities (ω1,ω2,ω3)
- 1 algebraic variable: q4 (from quaternion normalization constraint)
- 3 control variables: torques (u1,u2,u3) with bounds [-50, 50]
- Objective: Minimize final time tF
- Expected solution: J* = 28.6304077

Physical System:
- Quaternion kinematics for attitude representation
- Euler's equations for rigid body rotation with asymmetric inertia
- Quaternion normalization constraint: ||q|| = 1
"""

import casadi as ca
import numpy as np

import trajectolab as tl


def solve_asymmetric_rigid_body_single_phase():
    """
    Implement the single-phase asymmetric rigid body reorientation problem.

    This formulation allows the control torques to be optimized freely within bounds,
    leading to a minimum-time attitude control problem.

    Returns:
        Problem: Configured TrajectoLab problem ready for solving
    """

    # Physical constants (typical spacecraft moments of inertia)
    Ix = 10.0  # kg⋅m² - moment of inertia about x-axis
    Iy = 20.0  # kg⋅m² - moment of inertia about y-axis
    Iz = 30.0  # kg⋅m² - moment of inertia about z-axis

    # Target orientation angle
    phi_deg = 150.0  # degrees
    phi_rad = phi_deg * np.pi / 180.0

    # Create problem
    problem = tl.Problem("Asymmetric Rigid Body Reorientation - Single Phase")

    # ========================================================================
    # Phase Definition
    # ========================================================================
    phase = problem.set_phase(1)

    # Time variable (free final time - this is what we're minimizing)
    t = phase.time(initial=0.0, final=(0.01, 50.0))

    # ========================================================================
    # State Variables (6 total)
    # ========================================================================

    # Quaternion components (3 of 4 - q4 is algebraic)
    q1 = phase.state("q1", initial=0.0, final=ca.sin(phi_rad / 2), boundary=(-1.1, 1.1))
    q2 = phase.state("q2", initial=0.0, final=0.0, boundary=(-1.1, 1.1))
    q3 = phase.state("q3", initial=0.0, final=0.0, boundary=(-1.1, 1.1))

    # Angular velocities (rad/s)
    omega1 = phase.state("omega1", initial=0.0, final=0.0)
    omega2 = phase.state("omega2", initial=0.0, final=0.0)
    omega3 = phase.state("omega3", initial=0.0, final=0.0)

    # ========================================================================
    # Control Variables (3 total)
    # ========================================================================

    # Control torques (N⋅m) - these are optimized freely
    u1 = phase.control("torque_x", boundary=(-50.0, 50.0))
    u2 = phase.control("torque_y", boundary=(-50.0, 50.0))
    u3 = phase.control("torque_z", boundary=(-50.0, 50.0))

    # ========================================================================
    # Algebraic Variable and Constraint
    # ========================================================================

    # Fourth quaternion component (determined by normalization)
    # For numerical stability, we solve for q4 explicitly from the constraint
    # ||q||² = q1² + q2² + q3² + q4² = 1
    # Therefore: q4 = ±sqrt(1 - q1² - q2² - q3²)
    # We'll use the positive root and add a path constraint
    q4 = ca.sqrt(1 - q1**2 - q2**2 - q3**2)

    # Path constraint to ensure quaternion normalization
    # This constraint ensures q1² + q2² + q3² ≤ 1
    phase.subject_to(q1**2 + q2**2 + q3**2 <= 1.0)

    # ========================================================================
    # System Dynamics
    # ========================================================================

    # Quaternion kinematics (equations 10.121-10.123)
    q1_dot = 0.5 * (omega1 * q4 - omega2 * q3 + omega3 * q2)
    q2_dot = 0.5 * (omega1 * q3 + omega2 * q4 - omega3 * q1)
    q3_dot = 0.5 * (-omega1 * q2 + omega2 * q1 + omega3 * q4)

    # Euler's equations for rigid body rotation (equations 10.124-10.126)
    omega1_dot = u1 / Ix - ((Iz - Iy) / Ix) * omega2 * omega3
    omega2_dot = u2 / Iy - ((Ix - Iz) / Iy) * omega1 * omega3
    omega3_dot = u3 / Iz - ((Iy - Ix) / Iz) * omega1 * omega2

    # Set dynamics
    phase.dynamics(
        {
            q1: q1_dot,
            q2: q2_dot,
            q3: q3_dot,
            omega1: omega1_dot,
            omega2: omega2_dot,
            omega3: omega3_dot,
        }
    )

    # ========================================================================
    # Objective Function
    # ========================================================================

    # Minimize final time (minimum-time problem)
    problem.minimize(t.final)

    # ========================================================================
    # Mesh Configuration
    # ========================================================================

    # Use adaptive mesh with reasonable initial configuration
    phase.mesh([8, 8, 8], [-1.0, -0.3, 0.3, 1.0])

    return problem


def create_single_phase_initial_guess():
    """
    Create reasonable initial guess for the single-phase problem.

    The initial guess should provide a smooth trajectory from rest at the
    initial orientation to rest at the target orientation.
    """

    # Target values
    phi_rad = 150.0 * np.pi / 180.0
    q1_final = np.sin(phi_rad / 2)

    # Estimate reasonable final time (should be close to optimal)
    estimated_final_time = 30.0

    # Generate initial guess for each mesh interval
    states_guess = []
    controls_guess = []

    for N in [8, 8, 8]:  # Match mesh configuration
        # Normalized time for this interval
        tau = np.linspace(-1, 1, N + 1)
        t_norm = (tau + 1) / 2  # Map to [0, 1]

        # Smooth quaternion trajectory (linear interpolation)
        q1_vals = q1_final * t_norm  # q1: 0 → q1_final
        q2_vals = np.zeros(N + 1)  # q2: 0 → 0
        q3_vals = np.zeros(N + 1)  # q3: 0 → 0

        # Angular velocity trajectory (smooth acceleration/deceleration)
        # Use sinusoidal profile for smooth motion
        omega_profile = np.sin(np.pi * t_norm)
        omega1_vals = 2.0 * omega_profile  # Peak angular velocity ~2 rad/s
        omega2_vals = 1.0 * omega_profile  # Smaller motion in y
        omega3_vals = 0.5 * omega_profile  # Smallest motion in z

        # At final time, angular velocities should be zero
        omega1_vals[-1] = 0.0
        omega2_vals[-1] = 0.0
        omega3_vals[-1] = 0.0

        states_guess.append(
            np.array([q1_vals, q2_vals, q3_vals, omega1_vals, omega2_vals, omega3_vals])
        )

        # Control guess: moderate torques that create the desired motion
        # Use profile that accelerates then decelerates
        control_profile = np.cos(np.pi * np.linspace(0, 1, N))
        u1_vals = 10.0 * control_profile  # Primary control about x
        u2_vals = 5.0 * control_profile  # Secondary control about y
        u3_vals = 2.0 * control_profile  # Tertiary control about z

        controls_guess.append(np.array([u1_vals, u2_vals, u3_vals]))

    return {
        "phase_states": {1: states_guess},
        "phase_controls": {1: controls_guess},
        "phase_terminal_times": {1: estimated_final_time},
    }


def main():
    """Main execution function for single-phase reorientation problem."""

    print("=" * 80)
    print("TrajectoLab Implementation: Asymmetric Rigid Body Reorientation")
    print("Single-Phase Problem (asyq01) - Free Control Optimization")
    print("=" * 80)

    # Create and configure the problem
    problem = solve_asymmetric_rigid_body_single_phase()

    # Set initial guess to aid convergence
    initial_guess = create_single_phase_initial_guess()
    problem.guess(**initial_guess)

    print("Problem Classification: Minimum-time optimal control")
    print("Optimization variables: Control torques u1, u2, u3 and final time")
    print("Expected benchmark solution: J* = 28.6304077")

    print("\nProblem Specification:")
    print("States: q1, q2, q3 (quaternion), ω1, ω2, ω3 (angular velocities)")
    print("Controls: u1, u2, u3 (torques) ∈ [-50, 50] N⋅m")
    print("Algebraic: q4 from quaternion normalization ||q|| = 1")
    print("Dynamics: Quaternion kinematics + Euler's equations")
    print("Initial: All states at rest")
    print("Final: q1 = sin(75°), q2 = q3 = 0, all ω = 0")
    print("Objective: Minimize final time tF")

    print("\nSolving single-phase reorientation problem...")

    # Solve with adaptive mesh refinement
    solution = tl.solve_adaptive(
        problem,
        error_tolerance=1e-6,
        max_iterations=20,
        min_polynomial_degree=3,
        max_polynomial_degree=12,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-8,
            "ipopt.constr_viol_tol": 1e-8,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.linear_solver": "mumps",
        },
    )

    # Display results
    if solution.success:
        print("\n" + "=" * 60)
        print("SINGLE-PHASE SOLUTION SUCCESSFUL!")
        print("=" * 60)

        final_time = solution.objective
        print(f"Optimal Final Time: {final_time:.7f}")
        print("Benchmark Solution: 28.6304077")
        error = abs(final_time - 28.6304077) / 28.6304077 * 100
        print(f"Relative Error: {error:.3f}%")

        # Extract final state values
        try:
            q1_final = solution[(1, "q1")][-1]
            q2_final = solution[(1, "q2")][-1]
            q3_final = solution[(1, "q3")][-1]
            omega1_final = solution[(1, "omega1")][-1]
            omega2_final = solution[(1, "omega2")][-1]
            omega3_final = solution[(1, "omega3")][-1]

            # Compute q4 from normalization
            q4_final = np.sqrt(1 - q1_final**2 - q2_final**2 - q3_final**2)

            print("\nFinal state values:")
            print(f"q1(tF) = {q1_final:.6f} (target: {np.sin(150 * np.pi / 180 / 2):.6f})")
            print(f"q2(tF) = {q2_final:.6f} (target: 0.0)")
            print(f"q3(tF) = {q3_final:.6f} (target: 0.0)")
            print(f"q4(tF) = {q4_final:.6f}")
            print(f"||q||  = {np.sqrt(q1_final**2 + q2_final**2 + q3_final**2 + q4_final**2):.6f}")
            print(f"ω1(tF) = {omega1_final:.6f} (target: 0.0)")
            print(f"ω2(tF) = {omega2_final:.6f} (target: 0.0)")
            print(f"ω3(tF) = {omega3_final:.6f} (target: 0.0)")

            # Check control bounds
            u1_data = solution[(1, "torque_x")]
            u2_data = solution[(1, "torque_y")]
            u3_data = solution[(1, "torque_z")]

            print("\nControl torque ranges:")
            print(f"u1 ∈ [{np.min(u1_data):.1f}, {np.max(u1_data):.1f}] N⋅m")
            print(f"u2 ∈ [{np.min(u2_data):.1f}, {np.max(u2_data):.1f}] N⋅m")
            print(f"u3 ∈ [{np.min(u3_data):.1f}, {np.max(u3_data):.1f}] N⋅m")

        except KeyError as e:
            print(f"Could not extract all final state values: {e}")

        # Plot the solution
        print("\nGenerating solution plots...")
        solution.plot()

        return solution

    else:
        print("\n" + "=" * 60)
        print("SINGLE-PHASE SOLUTION FAILED!")
        print("=" * 60)
        print(f"Message: {solution.message}")
        return None


if __name__ == "__main__":
    main()
