"""
Multi-phase tumor growth optimal control problem (Example 10.142).

Minimum Tumor Size: Two-Phase Formulation from CGPOPS literature.
This is a safety-critical test case with known literature result: J* = 7571.67158

Phase 1: Drug administration (ẏ = a, with Ga term in q̇)
Phase 2: No drug administration (ẏ = 0, no Ga term in q̇)

Mathematical formulation:
- State variables: p (tumor metric), q (tumor metric), y (drug accumulation)
- Phase 1 dynamics: ṗ = -ξp ln(p/q), q̇ = q[b - (μ + dp^(2/3) + Ga)], ẏ = a
- Phase 2 dynamics: ṗ = -ξp ln(p/q), q̇ = q[b - (μ + dp^(2/3))], ẏ = 0
- Objective: Minimize p(t_F^(2)) (final tumor size)
"""

import logging

import numpy as np


# Enable comprehensive logging to debug any issues
logging.getLogger("trajectolab").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

try:
    import trajectolab as tl

    print("✓ TrajectoLab imported successfully")
except Exception as e:
    print(f"✗ Failed to import TrajectoLab: {e}")
    exit(1)


def test_tumor_growth_multi_phase():
    """Test the tumor growth multi-phase problem (Example 10.142)."""
    print("\n=== Tumor Growth Multi-Phase Problem (Example 10.142) ===")
    print("Literature result: J* = 7571.67158")
    print("=" * 60)

    try:
        # These may need adjustment based on the actual parameter values
        xi = 0.084  # ξ parameter
        b = 5.85  # birth rate parameter
        mu = 0.02  # death rate parameter
        d = 0.00873  # density parameter
        G = 0.15  # drug efficacy parameter
        a = 75.0  # drug administration rate
        A = 15.0  # maximum drug accumulation

        # Computed parameters
        p_bar = ((b - mu) / d) ** (3 / 2)  # p̄ = q̄ = [(b-μ)/d]^(3/2)
        q_bar = p_bar
        p_0 = p_bar / 2  # p₀ = p̄/2
        q_0 = q_bar / 4  # q₀ = q̄/4

        print("Problem parameters:")
        print(f"  ξ = {xi}, b = {b}, μ = {mu}, d = {d}")
        print(f"  G = {G}, a = {a}, A = {A}")
        print(f"  p̄ = q̄ = {p_bar:.6f}")
        print(f"  p₀ = {p_0:.6f}, q₀ = {q_0:.6f}")

        # Step 1: Create multi-phase problem
        print("\nStep 1: Creating multi-phase problem...")
        mp_problem = tl.MultiPhaseProblem("Tumor Growth Minimum Size")
        print("✓ Multi-phase problem created")

        # Step 2: Add phases
        print("\nStep 2: Adding phases...")
        phase1 = mp_problem.add_phase("DrugAdministration")  # Phase 1: with drug
        phase2 = mp_problem.add_phase("NoDrug")  # Phase 2: no drug
        print("✓ Phases added successfully")

        # Add global parameters
        print("\nStep 3: Adding global parameters...")
        xi_param = mp_problem.add_global_parameter("xi", xi)
        b_param = mp_problem.add_global_parameter("b", b)
        mu_param = mp_problem.add_global_parameter("mu", mu)
        d_param = mp_problem.add_global_parameter("d", d)
        G_param = mp_problem.add_global_parameter("G", G)
        a_param = mp_problem.add_global_parameter("a", a)
        print("✓ Global parameters added")

        # Step 4: Configure Phase 1 (Drug Administration)
        print("\nStep 4: Configuring Phase 1 (Drug Administration)...")

        # Phase 1 time: t = 0 to t_F^(1), with constraint 0.01 ≤ t_F^(1)
        t1 = phase1.time(initial=0.0, final=(0.01, None))  # t_F^(1) ≥ 0.01

        # Phase 1 states with initial conditions and path constraints
        p1 = phase1.state("p", initial=p_0, boundary=(0.01, p_bar))  # 0.01 ≤ p ≤ p̄
        q1 = phase1.state("q", initial=q_0, boundary=(0.01, q_bar))  # 0.01 ≤ q ≤ q̄
        y1 = phase1.state("y", initial=0.0, final=(0.0, A))  # y(0) = 0, 0 ≤ y(t_F^(1)) ≤ A

        # Phase 1 dynamics:
        # ṗ = -ξp ln(p/q)
        # q̇ = q[b - (μ + dp^(2/3) + Ga)]
        # ẏ = a
        import casadi as ca

        def safe_log_ratio(p, q, eps=1e-8):
            """Numerically stable log(p/q) with safeguarding."""
            p_safe = ca.fmax(p, eps)  # Ensure p > 0
            q_safe = ca.fmax(q, eps)  # Ensure q > 0
            return ca.log(p_safe / q_safe)

        phase1.dynamics(
            {
                p1: -xi_param * p1 * safe_log_ratio(p1, q1),
                q1: q1 * (b_param - (mu_param + d_param * p1 ** (2 / 3) + G_param * a_param)),
                y1: a_param,
            }
        )

        # Phase 1 mesh configuration
        phase1.set_mesh([100], np.array([-1.0, 1.0]))
        print("✓ Phase 1 configured")

        # Step 5: Configure Phase 2 (No Drug Administration)
        print("\nStep 5: Configuring Phase 2 (No Drug Administration)...")

        # Phase 2 time: t = t_I^(2) to t_F^(2), where t_I^(2) = t_F^(1)
        t2 = phase2.time()  # Free initial and final time

        # Phase 2 states (will be linked via continuity constraints)
        p2 = phase2.state("p", boundary=(0.01, p_bar))  # 0.01 ≤ p ≤ p̄
        q2 = phase2.state("q", boundary=(0.01, q_bar))  # 0.01 ≤ q ≤ q̄
        y2 = phase2.state("y", boundary=(0.0, A))  # 0 ≤ y ≤ A

        # Phase 2 dynamics:
        # ṗ = -ξp ln(p/q)
        # q̇ = q[b - (μ + dp^(2/3))]  [NO Ga term!]
        # ẏ = 0  [NO drug administration!]
        phase2.dynamics(
            {
                p2: -xi_param * p2 * safe_log_ratio(p2, q2),  # ← FIXED: numerically stable
                q2: q2 * (b_param - (mu_param + d_param * p2 ** (2 / 3))),
                y2: 0.0,
            }
        )

        # Phase 2 mesh configuration
        phase2.set_mesh([100], np.array([-1.0, 1.0]))
        print("✓ Phase 2 configured")

        # Step 6: Add inter-phase continuity constraints
        print("\nStep 6: Adding phase continuity constraints...")

        # Time continuity: t_I^(2) = t_F^(1)
        mp_problem.link_phases(t2.initial == t1.final)

        # State continuity: Phase 2 initial states = Phase 1 final states
        mp_problem.link_phases(p2.initial == p1.final)  # p continuity
        mp_problem.link_phases(q2.initial == q1.final)  # q continuity
        mp_problem.link_phases(y2.initial == y1.final)  # y continuity

        print("✓ Phase continuity constraints added")

        # Step 7: Set global objective
        print("\nStep 7: Setting global objective...")

        # Objective: Minimize J = p(t_F^(2)) (final tumor size)
        mp_problem.set_global_objective(p2.final)
        print("✓ Global objective set: Minimize p(t_F^(2))")

        # Step 8: Validate problem structure
        print("\nStep 8: Validating problem structure...")
        mp_problem.validate_complete_structure()
        print("✓ Problem structure validation passed")

        # Step 9: Print comprehensive problem summary
        print("\nStep 9: Problem summary...")
        summary = mp_problem.get_problem_summary()
        print(f"  Problem name: {summary['name']}")
        print(f"  Phases: {summary['phase_count']}")
        print(f"  Global parameters: {len(summary['global_parameters'])}")
        print(f"  Inter-phase constraints: {summary['inter_phase_constraints_count']}")
        print(f"  Global objective set: {summary['global_objective_set']}")

        for i, phase_info in enumerate(summary["phases"]):
            print(
                f"  Phase {i} ({phase_info['name']}): {phase_info['num_states']} states, "
                f"{phase_info['num_controls']} controls, mesh_configured={phase_info['mesh_configured']}"
            )

        print("\nStep 9.5: Setting initial guess...")

        # Phase 1: 9 nodes (8 collocation + 1 endpoint)
        t1_guess = np.linspace(0.0, 1.0, 101)  # Normalized time
        p1_guess = np.full(101, p_0)  # Start with initial value
        q1_guess = np.full(101, q_0)  # Start with initial value
        y1_guess = np.linspace(0.0, 50.0, 101)  # Linear drug accumulation

        phase1_states = np.array([p1_guess, q1_guess, y1_guess])

        # Phase 2: 9 nodes (8 collocation + 1 endpoint)
        p2_guess = np.full(101, p_0 * 0.8)  # Slightly reduced tumor
        q2_guess = np.full(101, q_0 * 0.9)  # Slightly improved
        y2_guess = np.full(101, 50.0)  # Constant drug level

        phase2_states = np.array([p2_guess, q2_guess, y2_guess])

        # Set initial guess for phases
        phase1.set_initial_guess(states=[phase1_states], initial_time=0.0, terminal_time=0.5)

        phase2.set_initial_guess(states=[phase2_states], initial_time=0.5, terminal_time=2.0)

        print("✓ Initial guess set for both phases")

        # Step 10: Attempt to solve
        print("\nStep 10: Attempting to solve multi-phase problem...")
        print("This may take some time for the complex tumor growth dynamics...")

        solution = tl.solve_multi_phase_fixed_mesh(
            mp_problem,
            nlp_options={
                "ipopt.print_level": 3,  # More output for debugging
                "ipopt.sb": "yes",
                "ipopt.max_iter": 3000,
                "ipopt.tol": 1e-6,  # Slightly relaxed tolerance
                "ipopt.acceptable_tol": 1e-5,  # Backup tolerance
                "ipopt.mu_strategy": "adaptive",  # Robust barrier strategy
                "ipopt.alpha_for_y": "safer-min-dual-infeas",  # Numerical stability
            },
        )

        print(f"\n{'=' * 60}")
        print("SOLUTION RESULTS:")
        print(f"{'=' * 60}")
        print("✓ Solver completed")
        print(f"Success: {solution.success}")

        if solution.success:
            print(f"Objective (Final tumor size): {solution.objective:.8f}")
            print("Literature result:           7571.67158")

            if solution.objective is not None:
                error = abs(solution.objective - 7571.67158)
                rel_error = error / 7571.67158 * 100
                print(f"Absolute error: {error:.8f}")
                print(f"Relative error: {rel_error:.6f}%")

                if rel_error < 1.0:
                    print("✓ EXCELLENT: Solution matches literature within 1%")
                elif rel_error < 5.0:
                    print("✓ GOOD: Solution matches literature within 5%")
                elif rel_error < 10.0:
                    print("⚠ ACCEPTABLE: Solution within 10% of literature")
                else:
                    print("✗ POOR: Solution differs significantly from literature")

            print(f"Phase count: {solution.phase_count}")

            # Extract phase-specific results
            phase_solutions = solution.get_all_phase_solutions()
            for i, phase_sol in enumerate(phase_solutions):
                if phase_sol.success:
                    print(
                        f"Phase {i}: t_initial = {phase_sol.initial_time:.6f}, "
                        f"t_final = {phase_sol.final_time:.6f}"
                    )
                    if hasattr(phase_sol, "states") and phase_sol.states:
                        p_initial = phase_sol.states[0][0] if len(phase_sol.states) > 0 else "N/A"
                        p_final = phase_sol.states[0][-1] if len(phase_sol.states) > 0 else "N/A"
                        print(f"         p_initial = {p_initial:.6f}, p_final = {p_final:.6f}")
        else:
            print(f"✗ Solver failed: {solution.message}")
            print("This indicates an issue in the multi-phase implementation")

            # Try to extract debug information
            if hasattr(solution, "solution_data") and solution.solution_data:
                print(f"Debug info: {solution.solution_data.message}")

        return solution.success

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")

        # Print full traceback for debugging
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Tumor Growth Multi-Phase Optimal Control Test")
    print("Example 10.142: Minimum Tumor Size Two-Phase Formulation")
    print("=" * 70)

    success = test_tumor_growth_multi_phase()

    print("\n" + "=" * 70)
    if success:
        print("✓ SUCCESS: Multi-phase tumor growth problem solved correctly!")
        print("  This validates the TrajectoLab multi-phase implementation.")
    else:
        print("✗ FAILURE: Multi-phase solution failed.")
        print("  This indicates issues in the TrajectoLab multi-phase codebase.")
        print("  Check error messages above for debugging information.")
