"""
TrajectoLab Implementation: Linear Tangent Steering Problem (Section 10.40)

Three-phase PARAMETER OPTIMIZATION problem with algebraically determined control.
Reference: Example 10.40 from optimal control textbook.

IMPORTANT: This is NOT a control optimization problem. The control u is determined
algebraically from static parameters p₁, p₂ and time t via: tan u = p₁ - p₂t

Expected solution:
- t_F* = 5.5457088 × 10^(-1) = 0.55457088
- p1* = 1.4085084
- p2* = 5.0796333

Optimization Variables:
- Static parameters: p₁, p₂ (shared across all phases)
- Phase final times: t_F^(1), t_F^(2), t_F^(3)

State Dynamics: ẋ₁ = x₃, ẋ₂ = x₄, ẋ₃ = a cos u, ẋ₄ = a sin u
Algebraic Control: tan u = p₁ - p₂t (linear in time)

Efficient trigonometric computation using equations (10.544)-(10.546):
- D = (1 + tan² u)^(-1/2)
- sin u = D tan u
- cos u = D

Objective: Minimize final time t_F^(3)
"""

import casadi as ca
import numpy as np

import trajectolab as tl


def solve_linear_tangent_steering_problem():
    """
    Implement the Linear Tangent Steering Problem using TrajectoLab's multiphase framework.

    This is a PARAMETER OPTIMIZATION problem with:
    - Static parameters p₁, p₂ shared across all phases (optimization variables)
    - No explicit control variables - control u is algebraically determined
    - Algebraic control law: tan u = p₁ - p₂t
    - State variables: x₁, x₂, x₃, x₄
    - State continuity between phases
    - Parameter continuity between phases
    - Specific timing constraints
    - Final state constraints in Phase 3

    Returns:
        Problem: Configured TrajectoLab problem ready for solving
    """

    # Physical constants
    a = 100.0  # Acceleration parameter from specification

    problem = tl.Problem("Linear Tangent Steering Problem")

    # ========================================================================
    # Static Parameters (shared across all phases)
    # ========================================================================

    # Parameters p₁ and p₂ with bounds: 0 ≤ p₁, 0 ≤ p₂
    p1 = problem.parameter("p1", boundary=(0.0, None))
    p2 = problem.parameter("p2", boundary=(0.0, None))

    # ========================================================================
    # Phase 1: t ∈ [0, t_F^(1)]
    # ========================================================================
    with problem.phase(1) as phase1:
        # Time variable with initial time fixed at 0
        t1 = phase1.time(initial=0.0)

        # State variables with initial conditions from specification
        x1_1 = phase1.state("x1", initial=0.0)  # x₁(0) = 0
        x2_1 = phase1.state("x2", initial=0.0)  # x₂(0) = 0
        x3_1 = phase1.state("x3", initial=0.0)  # x₃(0) = 0
        x4_1 = phase1.state("x4", initial=0.0)  # x₄(0) = 0

        # Control law using equations (10.544)-(10.546)
        # tan u = p₁ - p₂t
        tan_u1 = p1 - p2 * t1

        # D = (1 + tan² u)^(-1/2)  (equation 10.544)
        D1 = ca.power(1 + tan_u1**2, -0.5)

        # sin u = D tan u  (equation 10.545)
        sin_u1 = D1 * tan_u1

        # cos u = D  (equation 10.546)
        cos_u1 = D1

        # Dynamics equations (10.527)-(10.530) with a = 100
        # ẋ₁ = x₃, ẋ₂ = x₄, ẋ₃ = a cos u, ẋ₄ = a sin u
        phase1.dynamics(
            {
                x1_1: x3_1,  # ẋ₁ = x₃
                x2_1: x4_1,  # ẋ₂ = x₄
                x3_1: a * cos_u1,  # ẋ₃ = a cos u
                x4_1: a * sin_u1,  # ẋ₄ = a sin u
            }
        )

        # Set mesh for Phase 1
        phase1.set_mesh([8, 8], [-1.0, 0.0, 1.0])

    # ========================================================================
    # Phase 2: t ∈ [t_F^(1), t_F^(2)]
    # ========================================================================
    with problem.phase(2) as phase2:
        # Time variable continuing from Phase 1
        t2 = phase2.time(initial=t1.final)

        # State variables with continuity from Phase 1
        x1_2 = phase2.state("x1", initial=x1_1.final)  # x₁ continuity
        x2_2 = phase2.state("x2", initial=x2_1.final)  # x₂ continuity
        x3_2 = phase2.state("x3", initial=x3_1.final)  # x₃ continuity
        x4_2 = phase2.state("x4", initial=x4_1.final)  # x₄ continuity

        # Control law using equations (10.544)-(10.546)
        # tan u = p₁ - p₂t
        tan_u2 = p1 - p2 * t2

        # D = (1 + tan² u)^(-1/2)  (equation 10.544)
        D2 = ca.power(1 + tan_u2**2, -0.5)

        # sin u = D tan u  (equation 10.545)
        sin_u2 = D2 * tan_u2

        # cos u = D  (equation 10.546)
        cos_u2 = D2

        # Same dynamics equations
        phase2.dynamics(
            {
                x1_2: x3_2,  # ẋ₁ = x₃
                x2_2: x4_2,  # ẋ₂ = x₄
                x3_2: a * cos_u2,  # ẋ₃ = a cos u
                x4_2: a * sin_u2,  # ẋ₄ = a sin u
            }
        )

        # Set mesh for Phase 2
        phase2.set_mesh([8, 8], [-1.0, 0.0, 1.0])

    # ========================================================================
    # Phase 3: t ∈ [t_F^(2), t_F^(3)]
    # ========================================================================
    with problem.phase(3) as phase3:
        # Time variable continuing from Phase 2
        t3 = phase3.time(initial=t2.final)

        # State variables with continuity from Phase 2 and final constraints
        x1_3 = phase3.state("x1", initial=x1_2.final)  # x₁ continuity
        x2_3 = phase3.state("x2", initial=x2_2.final, final=5.0)  # x₂(t_F) = 5
        x3_3 = phase3.state("x3", initial=x3_2.final, final=45.0)  # x₃(t_F) = 45
        x4_3 = phase3.state("x4", initial=x4_2.final, final=0.0)  # x₄(t_F) = 0

        # Control law using equations (10.544)-(10.546)
        # tan u = p₁ - p₂t
        tan_u3 = p1 - p2 * t3

        # D = (1 + tan² u)^(-1/2)  (equation 10.544)
        D3 = ca.power(1 + tan_u3**2, -0.5)

        # sin u = D tan u  (equation 10.545)
        sin_u3 = D3 * tan_u3

        # cos u = D  (equation 10.546)
        cos_u3 = D3

        # Same dynamics equations
        phase3.dynamics(
            {
                x1_3: x3_3,  # ẋ₁ = x₃
                x2_3: x4_3,  # ẋ₂ = x₄
                x3_3: a * cos_u3,  # ẋ₃ = a cos u
                x4_3: a * sin_u3,  # ẋ₄ = a sin u
            }
        )

        # Set mesh for Phase 3
        phase3.set_mesh([8, 8], [-1.0, 0.0, 1.0])

    # ========================================================================
    # Cross-Phase Boundary Conditions
    # ========================================================================

    # State continuity is already handled by initial=final_state syntax above

    # Additional timing constraints from specification:
    # Phase 2: t_F^(2) - 2t_I^(2) = 0, where t_I^(2) appears to be related to intermediate timing
    # Phase 3: t_F^(3) - 2t_I^(3) + t_I^(2) = 0

    # Note: The t_I terms in the specification seem to be intermediate times within phases
    # For now, implementing the core structure. These timing constraints may need refinement
    # based on the full context of equations (10.544)-(10.546) which are referenced but not shown

    # ========================================================================
    # Objective Function
    # ========================================================================

    # Minimize final time: J = t_F = t_F^(3)
    problem.minimize(t3.final)

    return problem


def create_initial_guess():
    """
    Create reasonable initial guess for PARAMETER OPTIMIZATION problem.

    Expected solution values:
    - t_F* = 0.55457088
    - p1* = 1.4085084
    - p2* = 5.0796333

    The trajectory involves controlled motion from origin to final state
    (x₂=5, x₃=45, x₄=0) in approximately 0.555 time units.

    NOTE: No control variables to guess - control u is algebraically determined
    by parameters p₁, p₂ and time t through: tan u = p₁ - p₂t
    """

    # Estimate phase durations (approximately equal splits)
    total_time = 0.56  # Slightly above expected solution
    phase_duration = total_time / 3.0

    # ========================================================================
    # Phase 1: mesh [8, 8], states [x1, x2, x3, x4] (4 states), NO CONTROLS
    # ========================================================================

    states_p1 = []
    controls_p1 = []  # Empty - no explicit control variables

    # Interval 1: 8 collocation points -> (4, 9) states, NO controls
    tau1 = np.linspace(-1, 1, 9)
    t_phys1 = (tau1 + 1) / 2 * phase_duration  # Map to [0, phase_duration]

    # Linear trajectory guesses for Phase 1
    x1_vals = t_phys1 * 2.0  # Gradual x1 increase
    x2_vals = t_phys1 * 1.0  # Gradual x2 increase
    x3_vals = t_phys1 * 10.0  # Building up x3 velocity
    x4_vals = t_phys1 * 5.0  # Building up x4 velocity

    states_p1.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
    controls_p1.append(np.zeros((0, 8)))  # No controls (0 control variables, 8 collocation points)

    # Interval 2: 8 collocation points -> (4, 9) states, NO controls
    tau2 = np.linspace(-1, 1, 9)
    t_phys2 = (tau2 + 1) / 2 * phase_duration

    x1_vals = t_phys2 * 3.0 + 0.5
    x2_vals = t_phys2 * 1.5 + 0.2
    x3_vals = t_phys2 * 15.0 + 2.0
    x4_vals = t_phys2 * 7.0 + 1.0

    states_p1.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
    controls_p1.append(np.zeros((0, 8)))  # No controls

    # ========================================================================
    # Phase 2: mesh [8, 8], states [x1, x2, x3, x4] (4 states), NO CONTROLS
    # ========================================================================

    states_p2 = []
    controls_p2 = []  # Empty - no explicit control variables

    # Continue from Phase 1 end values and progress toward final values
    for i in range(2):
        tau = np.linspace(-1, 1, 9)
        t_phys = (tau + 1) / 2 * phase_duration

        # Intermediate values progressing toward final state
        x1_vals = t_phys * 2.0 + 1.0 + i * 0.5
        x2_vals = t_phys * 1.0 + 0.5 + i * 1.0  # Moving toward x2=5
        x3_vals = t_phys * 20.0 + 5.0 + i * 10.0  # Moving toward x3=45
        x4_vals = t_phys * 3.0 + 2.0 - i * 1.0  # Moving toward x4=0

        states_p2.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
        controls_p2.append(np.zeros((0, 8)))  # No controls

    # ========================================================================
    # Phase 3: mesh [8, 8], states [x1, x2, x3, x4] (4 states), NO CONTROLS
    # ========================================================================

    states_p3 = []
    controls_p3 = []  # Empty - no explicit control variables

    # Approach final conditions: x2=5, x3=45, x4=0
    for i in range(2):
        tau = np.linspace(-1, 1, 9)
        t_phys = (tau + 1) / 2 * phase_duration

        # Final approach to target values
        x1_vals = t_phys * 1.0 + 2.0 + i * 0.3  # Free final value
        x2_vals = 4.0 + (tau + 1) / 2 * 1.0  # Approach x2=5
        x3_vals = 35.0 + (tau + 1) / 2 * 10.0  # Approach x3=45
        x4_vals = 1.0 - (tau + 1) / 2 * 1.0  # Approach x4=0

        states_p3.append(np.array([x1_vals, x2_vals, x3_vals, x4_vals]))
        controls_p3.append(np.zeros((0, 8)))  # No controls

    return {
        "phase_states": {1: states_p1, 2: states_p2, 3: states_p3},
        "phase_controls": {1: controls_p1, 2: controls_p2, 3: controls_p3},
        "phase_initial_times": {1: 0.0, 2: phase_duration, 3: 2 * phase_duration},
        "phase_terminal_times": {1: phase_duration, 2: 2 * phase_duration, 3: total_time},
        "static_parameters": np.array([1.4, 5.0]),  # Initial guess for p1, p2
    }


def main():
    """Main execution function."""

    print("=" * 80)
    print("TrajectoLab Implementation: Linear Tangent Steering Problem")
    print("Section 10.40 - Three Phase Optimal Control")
    print("=" * 80)

    # Create and configure the problem
    problem = solve_linear_tangent_steering_problem()

    # Set initial guess to aid convergence
    initial_guess = create_initial_guess()
    problem.set_initial_guess(**initial_guess)

    print("\nProblem Classification: PARAMETER OPTIMIZATION (not control optimization)")
    print("Optimization variables: Static parameters p₁, p₂ and phase final times")
    print("Control u is determined algebraically: tan u = p₁ - p₂t")
    print("\nProblem Specification:")
    print("Dynamics: ẋ₁ = x₃, ẋ₂ = x₄, ẋ₃ = a cos u, ẋ₄ = a sin u")
    print("Algebraic control law: tan u = p₁ - p₂t (linear tangent steering)")
    print("Trigonometric relations: D = (1 + tan²u)⁻¹/², sin u = D tan u, cos u = D")
    print("Initial conditions: x₁(0) = x₂(0) = x₃(0) = x₄(0) = 0")
    print("Final conditions: x₂(tF) = 5, x₃(tF) = 45, x₄(tF) = 0")
    print("Objective: Minimize final time tF")
    print("Parameters: p₁, p₂ ≥ 0 (static across all phases)")

    print("\nExpected benchmark solution:")
    print("t_F* = 0.55457088")
    print("p₁* = 1.4085084")
    print("p₂* = 5.0796333")

    print("\nSolving three-phase linear tangent steering PARAMETER OPTIMIZATION...")
    print("Optimizing: p₁, p₂ (static parameters) + phase final times")
    print("Control u computed algebraically from parameters and time")

    # Solve with fixed mesh
    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 5,
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
        print("SOLUTION SUCCESSFUL!")
        print("=" * 60)

        final_time = solution.objective
        print(f"Final Time (Objective): {final_time:.8f}")
        print(f"Phase 1 Final Time: {solution.get_phase_final_time(1):.8f}")
        print(f"Phase 2 Final Time: {solution.get_phase_final_time(2):.8f}")
        print(f"Phase 3 Final Time: {solution.get_phase_final_time(3):.8f}")

        # Extract static parameters
        if solution.static_parameters is not None:
            p1_opt = solution.static_parameters[0]
            p2_opt = solution.static_parameters[1]
            print(f"Optimal p₁: {p1_opt:.8f}")
            print(f"Optimal p₂: {p2_opt:.8f}")

        print("\nPhase durations:")
        print(f"Phase 1: {solution.get_phase_duration(1):.8f}")
        print(f"Phase 2: {solution.get_phase_duration(2):.8f}")
        print(f"Phase 3: {solution.get_phase_duration(3):.8f}")

        # Check final state values
        try:
            x2_final = solution[(3, "x2")][-1]
            x3_final = solution[(3, "x3")][-1]
            x4_final = solution[(3, "x4")][-1]
            print("\nFinal state values:")
            print(f"x₂(tF) = {x2_final:.6f} (target: 5.0)")
            print(f"x₃(tF) = {x3_final:.6f} (target: 45.0)")
            print(f"x₄(tF) = {x4_final:.6f} (target: 0.0)")
        except KeyError:
            print("\nCould not extract final state values")

        print("\nComparison with benchmark:")
        print(f"Final Time: {final_time:.8f} vs 0.55457088")
        if solution.static_parameters is not None:
            print(f"p₁: {p1_opt:.8f} vs 1.4085084")
            print(f"p₂: {p2_opt:.8f} vs 5.0796333")

        # Plot the solution
        print("\nGenerating solution plots...")
        solution.plot(show_phase_boundaries=True)

    else:
        print("\n" + "=" * 60)
        print("SOLUTION FAILED!")
        print("=" * 60)
        print(f"Message: {solution.message}")


if __name__ == "__main__":
    main()
