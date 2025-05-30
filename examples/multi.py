# examples/multi_debug.py
"""
TrajectoLab Debugging: Hypersensitive Problem Single vs Multi-Phase
Focuses on inspecting the two-phase solution data.
"""

import matplotlib.pyplot as plt

import trajectolab as tl


def solve_two_phase_hypersensitive_debug():
    """Solve two-phase hypersensitive problem with detailed debugging."""
    print("\n" + "=" * 60)
    print("🔄 Solving Two-Phase Hypersensitive (DEBUG MODE)...")
    print("=" * 60)

    problem_phase = tl.Problem("Two-Phase Hypersensitive Debug")

    # Phase 1: [0, 20]
    with problem_phase.phase(1) as phase1:
        t1 = phase1.time(initial=0.0, final=20.0)  # Fixed intermediate time
        x1 = phase1.state("x", initial=1.5)
        u1 = phase1.control("u")
        phase1.dynamics({x1: -(x1**3) + u1})
        integral1 = phase1.add_integral(0.5 * (x1**2 + u1**2))
        phase1.set_mesh([10, 6, 10], [-1.0, -1 / 3, 1 / 3, 1.0])

    # Phase 2: [20, 40]
    with problem_phase.phase(2) as phase2:
        t2 = phase2.time(initial=t1.final, final=40.0)
        x2 = phase2.state("x", initial=x1.final, final=1.0)
        u2 = phase2.control("u")
        phase2.dynamics({x2: -(x2**3) + u2})
        integral2 = phase2.add_integral(0.5 * (x2**2 + u2**2))
        phase2.set_mesh([10, 6, 10], [-1.0, -1 / 3, 1 / 3, 1.0])

    problem_phase.minimize(integral1 + integral2)

    # Simplified initial guess for debugging
    # Approximate x values: 0-20s (1.5 -> 1.25), 20-40s (1.25 -> 1.0)

    print("Solving with IPOPT print_level = 3...")
    solution = tl.solve_fixed_mesh(
        problem_phase,
        nlp_options={
            "ipopt.print_level": 3,  # Increased verbosity
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-8,
            "ipopt.constr_viol_tol": 1e-8,  # Added for robustness
        },
    )

    # --- DETAILED SOLUTION INSPECTION ---
    print("\n" + "-" * 20 + " TWO-PHASE SOLUTION INSPECTION " + "-" * 20)
    if solution is None:
        print("SOLUTION OBJECT IS NONE!")
        return None

    print(f"Overall Success: {solution.success}")
    print(f"Overall Objective: {solution.objective}")
    print(f"Message: {solution.message}")

    phase_ids = solution.get_phase_ids()
    print(f"Phase IDs in Solution: {phase_ids}")

    for pid in phase_ids:
        print(f"\n--- Inspecting Phase {pid} ---")
        try:
            initial_time = solution.get_phase_initial_time(pid)
            final_time = solution.get_phase_final_time(pid)
            duration = solution.get_phase_duration(pid)
            integral_val = solution.phase_integrals.get(pid, "N/A")

            print(f"  Phase {pid} Initial Time: {initial_time}")
            print(f"  Phase {pid} Final Time: {final_time}")
            print(f"  Phase {pid} Duration: {duration}")
            print(f"  Phase {pid} Integral Value: {integral_val}")

            print(f"  Phase {pid} state names: {solution._phase_state_names.get(pid)}")
            print(f"  Phase {pid} control names: {solution._phase_control_names.get(pid)}")

            # Check for presence of data keys
            print(
                f"  Keys in phase_time_states for phase {pid}: {pid in solution.phase_time_states}"
            )
            print(f"  Keys in phase_states for phase {pid}: {pid in solution.phase_states}")
            print(
                f"  Keys in phase_time_controls for phase {pid}: {pid in solution.phase_time_controls}"
            )
            print(f"  Keys in phase_controls for phase {pid}: {pid in solution.phase_controls}")

            if pid in solution.phase_time_states:
                time_states_data = solution[(pid, "time_states")]
                print(f"  Phase {pid} time_states data (first 5): {time_states_data[:5]}")
                print(f"  Phase {pid} time_states length: {len(time_states_data)}")
            else:
                print(f"  Phase {pid} time_states: NO DATA")

            if pid in solution.phase_states and solution._phase_state_names.get(pid):
                state_name = solution._phase_state_names[pid][0]  # Assuming 'x' is the first state
                state_data = solution[(pid, state_name)]
                print(f"  Phase {pid} state '{state_name}' data (first 5): {state_data[:5]}")
                print(f"  Phase {pid} state '{state_name}' length: {len(state_data)}")
            else:
                print(f"  Phase {pid} state 'x': NO DATA or no state names")

            if pid in solution.phase_time_controls:
                time_controls_data = solution[(pid, "time_controls")]
                print(f"  Phase {pid} time_controls data (first 5): {time_controls_data[:5]}")
                print(f"  Phase {pid} time_controls length: {len(time_controls_data)}")
            else:
                print(f"  Phase {pid} time_controls: NO DATA")

            if pid in solution.phase_controls and solution._phase_control_names.get(pid):
                control_name = solution._phase_control_names[pid][
                    0
                ]  # Assuming 'u' is the first control
                control_data = solution[(pid, control_name)]
                print(f"  Phase {pid} control '{control_name}' data (first 5): {control_data[:5]}")
                print(f"  Phase {pid} control '{control_name}' length: {len(control_data)}")
            else:
                print(f"  Phase {pid} control 'u': NO DATA or no control names")

        except Exception as e:
            print(f"  Error inspecting phase {pid}: {e}")

    print("-" * 60)
    return solution


if __name__ == "__main__":
    # Solve Two-Phase with Debugging
    solution_2p = solve_two_phase_hypersensitive_debug()
    if solution_2p and solution_2p.success:
        print("\n✅ TWO-PHASE (DEBUG) SUCCESS")
        print(f"   Objective:   {solution_2p.objective:.8f}")
        try:
            print("Plotting Two-Phase Solution...")
            solution_2p.plot(show_phase_boundaries=True)  # Plot all phases by default
            plt.suptitle("Two-Phase Hypersensitive Solution (Debug)")
            # plt.show() # Ensure plot is shown
        except Exception as e:
            print(f"Error plotting two-phase: {e}")
    elif solution_2p:
        print(f"\n❌ TWO-PHASE (DEBUG) FAILED: {solution_2p.message}")
    else:
        print("\n❌ TWO-PHASE (DEBUG) FAILED: Solution object is None.")
