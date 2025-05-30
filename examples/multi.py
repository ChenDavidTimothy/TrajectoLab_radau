# examples/kinetic_batch_reactor.py
"""
TrajectoLab Example: Kinetic Batch Reactor - Three Phase Chemical Process Control
Based on Example 10.80 lwbr01 from optimal control literature.

This is a complex chemical reactor problem with:
- 3 phases: Transient Stage 1, Transient Stage 2, Steady State
- 6 differential state variables (y1-y6)
- 5 control variables (u1-u5) with algebraic constraints
- Temperature-dependent chemical kinetics
- Objective: Minimize final time + penalty on final parameter

Expected optimal: J* = 3.16466910; t_F^(3) = 1.7468208
"""

import casadi as ca

import trajectolab as tl


# Chemical kinetics model constants
k1_hat = 1.3708e12
beta1 = 9.2984e3
K1 = 2.575e-16

k_minus1_hat = 1.6215e20
beta_minus1 = 1.3108e4
K2 = 4.876e-14

k2_hat = 5.2282e12
beta2 = 9.5999e3
K3 = 1.7884e-16

print("Kinetic Batch Reactor - Chemical Process Control")
print("=" * 60)
print("Model Constants:")
print(f"  k̂₁ = {k1_hat:.4e}, β₁ = {beta1:.4e}, K₁ = {K1:.4e}")
print(f"  k̂₋₁ = {k_minus1_hat:.4e}, β₋₁ = {beta_minus1:.4e}, K₂ = {K2:.4e}")
print(f"  k̂₂ = {k2_hat:.4e}, β₂ = {beta2:.4e}, K₃ = {K3:.4e}")

# Create multiphase chemical reactor problem
problem = tl.Problem("Kinetic Batch Reactor")

# Static parameter for optimization
p = problem.parameter("p", boundary=(0.0, 0.0262))

# Phase 1: Transient Stage 1 (0 to 0.01)
with problem.phase(1) as phase1:
    print("\nDefining Phase 1: Transient Stage 1")

    # Time for phase 1 (fixed duration)
    t1 = phase1.time(initial=0.0, final=0.01)

    # Differential state variables
    y1_1 = phase1.state("y1", initial=1.5776, boundary=(None, 2))
    y2_1 = phase1.state("y2", initial=8.32, boundary=(5, 10))
    y3_1 = phase1.state("y3", initial=0.0, boundary=(None, 2))
    y4_1 = phase1.state("y4", initial=0.0, boundary=(None, 2))
    y5_1 = phase1.state("y5", initial=0.0, boundary=(None, 2))
    y6_1 = phase1.state("y6", final=p, boundary=(None, 0.1))

    # Control variables (algebraic variables in original formulation)
    u1_1 = phase1.control("u1", boundary=(0, 15))
    u2_1 = phase1.control("u2", boundary=(0, 0.02))
    u3_1 = phase1.control("u3", boundary=(0, 5e-5))
    u4_1 = phase1.control("u4", boundary=(0, 5e-5))
    u5_1 = phase1.control("u5", boundary=(293.15, 393.15))

    # Temperature-dependent rate constants
    k1_1 = k1_hat * ca.exp(-beta1 / u5_1)
    k_minus1_1 = k_minus1_hat * ca.exp(-beta_minus1 / u5_1)
    k2_1 = k2_hat * ca.exp(-beta2 / u5_1)
    k3_1 = k1_1
    k_minus3_1 = k_minus1_1 / 2

    # Differential equations (10.584)-(10.589)
    phase1.dynamics(
        {
            y1_1: -k2_1 * y2_1 * u2_1,
            y2_1: -k1_1 * y2_1 * y6_1 + k_minus1_1 * u4_1 - k2_1 * y2_1 * u2_1,
            y3_1: k2_1 * y2_1 * u2_1 + k3_1 * y4_1 * y6_1 - k_minus3_1 * u3_1,
            y4_1: -k3_1 * y4_1 * y6_1 + k_minus3_1 * u3_1,
            y5_1: k1_1 * y2_1 * y6_1 - k_minus1_1 * u4_1,
            y6_1: -k1_1 * y2_1 * y6_1 + k_minus1_1 * u4_1 - k3_1 * y4_1 * y6_1 + k_minus3_1 * u3_1,
        }
    )

    # Algebraic constraints as path constraints (10.590)-(10.594)
    phase1.subject_to(p - y6_1 + 10 ** (-u1_1) - u2_1 - u3_1 - u4_1 == 0)
    phase1.subject_to(u2_1 - K2 * y1_1 / (K2 + 10 ** (-u1_1)) == 0)
    phase1.subject_to(u3_1 - K3 * y3_1 / (K3 + 10 ** (-u1_1)) == 0)
    phase1.subject_to(u4_1 - K1 * y5_1 / (K1 + 10 ** (-u1_1)) == 0)
    # Inequality constraint (10.594)
    phase1.subject_to(y4_1 <= 2 * t1**2)

    # Mesh for phase 1
    phase1.set_mesh([8, 8], [-1.0, 0.0, 1.0])

# Phase 2: Transient Stage 2 (0.01 to t_F^(2))
with problem.phase(2) as phase2:
    print("Defining Phase 2: Transient Stage 2")

    # Time for phase 2 (free final time)
    t2 = phase2.time(initial=t1.final, final=(0.02, None))

    # State continuity from phase 1
    y1_2 = phase2.state("y1", initial=y1_1.final, boundary=(None, 2))
    y2_2 = phase2.state("y2", initial=y2_1.final, boundary=(5, 10))
    y3_2 = phase2.state("y3", initial=y3_1.final, boundary=(None, 2))
    y4_2 = phase2.state("y4", initial=y4_1.final, boundary=(None, 2))
    y5_2 = phase2.state("y5", initial=y5_1.final, boundary=(None, 2))
    y6_2 = phase2.state("y6", initial=y6_1.final, boundary=(None, 0.1))

    # Control variables
    u1_2 = phase2.control("u1", boundary=(0, 15))
    u2_2 = phase2.control("u2", boundary=(0, 0.02))
    u3_2 = phase2.control("u3", boundary=(0, 5e-5))
    u4_2 = phase2.control("u4", boundary=(0, 5e-5))
    u5_2 = phase2.control("u5", boundary=(293.15, 393.15))

    # Temperature-dependent rate constants
    k1_2 = k1_hat * ca.exp(-beta1 / u5_2)
    k_minus1_2 = k_minus1_hat * ca.exp(-beta_minus1 / u5_2)
    k2_2 = k2_hat * ca.exp(-beta2 / u5_2)
    k3_2 = k1_2
    k_minus3_2 = k_minus1_2 / 2

    # Same differential equations
    phase2.dynamics(
        {
            y1_2: -k2_2 * y2_2 * u2_2,
            y2_2: -k1_2 * y2_2 * y6_2 + k_minus1_2 * u4_2 - k2_2 * y2_2 * u2_2,
            y3_2: k2_2 * y2_2 * u2_2 + k3_2 * y4_2 * y6_2 - k_minus3_2 * u3_2,
            y4_2: -k3_2 * y4_2 * y6_2 + k_minus3_2 * u3_2,
            y5_2: k1_2 * y2_2 * y6_2 - k_minus1_2 * u4_2,
            y6_2: -k1_2 * y2_2 * y6_2 + k_minus1_2 * u4_2 - k3_2 * y4_2 * y6_2 + k_minus3_2 * u3_2,
        }
    )

    # Same algebraic constraints
    phase2.subject_to(p - y6_2 + 10 ** (-u1_2) - u2_2 - u3_2 - u4_2 == 0)
    phase2.subject_to(u2_2 - K2 * y1_2 / (K2 + 10 ** (-u1_2)) == 0)
    phase2.subject_to(u3_2 - K3 * y3_2 / (K3 + 10 ** (-u1_2)) == 0)
    phase2.subject_to(u4_2 - K1 * y5_2 / (K1 + 10 ** (-u1_2)) == 0)
    phase2.subject_to(y4_2 <= 2 * t2**2)

    # Mesh for phase 2
    phase2.set_mesh([10, 10], [-1.0, 0.0, 1.0])

# Phase 3: Steady State (t_F^(2) to t_F^(3))
with problem.phase(3) as phase3:
    print("Defining Phase 3: Steady State")

    # Time for phase 3 (free final time with minimum bound)
    t3 = phase3.time(initial=t2.final, final=(1.5, None))

    # State continuity from phase 2
    y1_3 = phase3.state("y1", initial=y1_2.final, boundary=(None, 2))
    y2_3 = phase3.state("y2", initial=y2_2.final, boundary=(5, 10))
    y3_3 = phase3.state("y3", initial=y3_2.final, boundary=(None, 2))
    y4_3 = phase3.state("y4", initial=y4_2.final, boundary=(None, 2))
    y5_3 = phase3.state("y5", initial=y5_2.final, boundary=(None, 2))
    y6_3 = phase3.state("y6", initial=y6_2.final, boundary=(None, 0.1))

    # Control variables
    u1_3 = phase3.control("u1", boundary=(0, 15))
    u2_3 = phase3.control("u2", boundary=(0, 0.02))
    u3_3 = phase3.control("u3", boundary=(0, 5e-5))
    u4_3 = phase3.control("u4", boundary=(0, 5e-5))
    u5_3 = phase3.control("u5", boundary=(293.15, 393.15))

    # Temperature-dependent rate constants
    k1_3 = k1_hat * ca.exp(-beta1 / u5_3)
    k_minus1_3 = k_minus1_hat * ca.exp(-beta_minus1 / u5_3)
    k2_3 = k2_hat * ca.exp(-beta2 / u5_3)
    k3_3 = k1_3
    k_minus3_3 = k_minus1_3 / 2

    # Same differential equations (but no inequality constraint in steady state)
    phase3.dynamics(
        {
            y1_3: -k2_3 * y2_3 * u2_3,
            y2_3: -k1_3 * y2_3 * y6_3 + k_minus1_3 * u4_3 - k2_3 * y2_3 * u2_3,
            y3_3: k2_3 * y2_3 * u2_3 + k3_3 * y4_3 * y6_3 - k_minus3_3 * u3_3,
            y4_3: -k3_3 * y4_3 * y6_3 + k_minus3_3 * u3_3,
            y5_3: k1_3 * y2_3 * y6_3 - k_minus1_3 * u4_3,
            y6_3: -k1_3 * y2_3 * y6_3 + k_minus1_3 * u4_3 - k3_3 * y4_3 * y6_3 + k_minus3_3 * u3_3,
        }
    )

    # Algebraic constraints (equations 10.590-10.593, no 10.594 in steady state)
    phase3.subject_to(p - y6_3 + 10 ** (-u1_3) - u2_3 - u3_3 - u4_3 == 0)
    phase3.subject_to(u2_3 - K2 * y1_3 / (K2 + 10 ** (-u1_3)) == 0)
    phase3.subject_to(u3_3 - K3 * y3_3 / (K3 + 10 ** (-u1_3)) == 0)
    phase3.subject_to(u4_3 - K1 * y5_3 / (K1 + 10 ** (-u1_3)) == 0)

    # Mesh for phase 3
    phase3.set_mesh([12, 12], [-1.0, 0.0, 1.0])

# Cross-phase constraints (state continuity is handled automatically through initial conditions)
print("Cross-phase constraints: State continuity enforced through initial conditions")

# Objective: Minimize J = t_F^(3) + 100*p^(3)
problem.minimize(t3.final + 100 * p)

print("Objective: Minimize final time + 100*parameter penalty")
print("Expected optimal: J* = 3.16466910; t_F^(3) = 1.7468208")

# Solve the three-phase kinetic batch reactor problem
print("\nSolving Three-Phase Kinetic Batch Reactor Problem...")
print("=" * 60)

solution = tl.solve_fixed_mesh(
    problem,
    nlp_options={
        "ipopt.print_level": 3,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-8,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
    },
)

# Results and Analysis
if solution.success:
    print("\n" + "=" * 60)
    print("KINETIC BATCH REACTOR OPTIMIZATION SUCCESS!")
    print("=" * 60)

    print(f"Objective Value: {solution.objective:.8f}")
    print("Expected Optimal: 3.16466910")
    print(f"Error: {abs(solution.objective - 3.16466910):.2e}")

    # Phase results
    print(f"\nPhase 1 Duration: {solution.get_phase_duration(1):.6f} (fixed at 0.01)")
    print(f"Phase 2 Duration: {solution.get_phase_duration(2):.6f}")
    print(f"Phase 3 Duration: {solution.get_phase_duration(3):.6f}")
    print(f"Total Process Time: {solution.get_total_mission_time():.6f}")
    print("Expected Final Time: 1.7468208")

    # Parameter value
    print(f"\nOptimal Parameter p: {solution.static_parameters[0]:.8f}")

    # Final states for each phase
    print("\nFinal State Values:")
    for phase_id in [1, 2, 3]:
        print(f"  Phase {phase_id}:")
        for i in range(1, 7):
            state_name = f"y{i}"
            if (phase_id, state_name) in solution:
                final_val = solution[(phase_id, state_name)][-1]
                print(f"    {state_name}: {final_val:.6f}")

    # Verify state continuity
    print("\nState Continuity Verification:")
    for i in range(1, 7):
        state_name = f"y{i}"
        y1_final = solution[(1, state_name)][-1]
        y2_initial = solution[(2, state_name)][0]
        y2_final = solution[(2, state_name)][-1]
        y3_initial = solution[(3, state_name)][0]

        print(
            f"  {state_name}: P1→P2 diff: {abs(y1_final - y2_initial):.2e}, P2→P3 diff: {abs(y2_final - y3_initial):.2e}"
        )

    # Plot the solution
    print("\nPlotting kinetic batch reactor trajectories...")
    solution.plot(show_phase_boundaries=True, figsize=(15, 12))

    # Plot individual phases for detailed analysis
    for phase_id in [1, 2, 3]:
        solution.plot(phase_id=phase_id, figsize=(12, 10))

    print("\n" + "=" * 70)
    print("- Complex chemical kinetics with temperature-dependent rates")
    print("- Three-phase process: Transient 1 → Transient 2 → Steady State")
    print("- Algebraic constraints handled as path constraints")
    print("=" * 70)

else:
    print(f"\nOptimization failed: {solution.message}")
    print("\nProblem Structure:")
    print(f"  Phases: {solution.get_phase_ids()}")
    for phase_id in problem.get_phase_ids():
        num_states, num_controls = problem.get_phase_variable_counts(phase_id)
        print(f"  Phase {phase_id}: {num_states} states, {num_controls} controls")

    print("\nThis is a challenging problem with:")
    print("  - Complex chemical kinetics")
    print("  - Temperature-dependent rate constants")
    print("  - Algebraic constraints")
    print("  - Multiple time scales across phases")
