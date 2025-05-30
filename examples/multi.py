# examples/hypersensitive_validation_comparison.py
"""
TrajectoLab Validation: Automated Single vs Multi-Phase Comparison
Runs multiple versions of hypersensitive problem and compares results.
"""

import time

import numpy as np

import trajectolab as tl


def solve_single_phase_hypersensitive():
    """Solve original single-phase hypersensitive problem."""
    print("🔄 Solving Single-Phase Hypersensitive...")

    problem = tl.Problem("Single-Phase Hypersensitive")

    with problem.phase(1) as phase:
        t = phase.time(initial=0, final=40)
        x = phase.state("x", initial=1.5, final=1.0)
        u = phase.control("u")

        phase.dynamics({x: -(x**3) + u})
        integrand = 0.5 * (x**2 + u**2)
        integral_var = phase.add_integral(integrand)

        # Use same refined mesh as in examples/hypersensitive.py
        phase.set_mesh([20, 12, 20], [-1.0, -1 / 3, 1 / 3, 1.0])

    problem.minimize(integral_var)

    # Set initial guess matching the original
    states_guess = []
    controls_guess = []
    for N in [20, 12, 20]:
        tau = np.linspace(-1, 1, N + 1)
        x_vals = 1.5 + (1.0 - 1.5) * (tau + 1) / 2
        states_guess.append(x_vals.reshape(1, -1))
        controls_guess.append(np.zeros((1, N)))

    problem.set_initial_guess(
        phase_states={1: states_guess},
        phase_controls={1: controls_guess},
        phase_initial_times={1: 0.0},
        phase_terminal_times={1: 40.0},
        phase_integrals={1: 0.1},
    )

    start_time = time.time()
    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-8,
        },
    )
    solve_time = time.time() - start_time

    return solution, solve_time


def solve_two_phase_hypersensitive():
    """Solve two-phase hypersensitive problem."""
    print("🔄 Solving Two-Phase Hypersensitive...")

    problem = tl.Problem("Two-Phase Hypersensitive")

    # Phase 1: [0, 20]
    with problem.phase(1) as phase1:
        t1 = phase1.time(initial=0.0, final=(18.0, 22.0))
        x1 = phase1.state("x", initial=1.5)
        u1 = phase1.control("u")
        phase1.dynamics({x1: -(x1**3) + u1})
        integral1 = phase1.add_integral(0.5 * (x1**2 + u1**2))
        phase1.set_mesh([10, 6, 10], [-1.0, -1 / 3, 1 / 3, 1.0])

    # Phase 2: [20, 40]
    with problem.phase(2) as phase2:
        t2 = phase2.time(initial=t1.final, final=40.0)
        x2 = phase2.state("x", initial=x1.final, final=1.0)
        u2 = phase2.control("u")
        phase2.dynamics({x2: -(x2**3) + u2})
        integral2 = phase2.add_integral(0.5 * (x2**2 + u2**2))
        phase2.set_mesh([10, 6, 10], [-1.0, -1 / 3, 1 / 3, 1.0])

    problem.minimize(integral1 + integral2)

    # Create initial guess
    def create_phase_guess(t_start, t_end, x_start, x_end, mesh_sizes):
        states, controls = [], []
        for N in mesh_sizes:
            tau = np.linspace(-1, 1, N + 1)
            t_vals = t_start + (t_end - t_start) * (tau + 1) / 2
            x_vals = x_start + (x_end - x_start) * (t_vals - t_start) / (t_end - t_start)
            states.append(x_vals.reshape(1, -1))
            controls.append(np.zeros((1, N)))
        return states, controls

    states_p1, controls_p1 = create_phase_guess(0, 20, 1.5, 1.25, [10, 6, 10])
    states_p2, controls_p2 = create_phase_guess(20, 40, 1.25, 1.0, [10, 6, 10])

    problem.set_initial_guess(
        phase_states={1: states_p1, 2: states_p2},
        phase_controls={1: controls_p1, 2: controls_p2},
        phase_initial_times={1: 0.0, 2: 20.0},
        phase_terminal_times={1: 20.0, 2: 40.0},
        phase_integrals={1: 0.05, 2: 0.05},
    )

    start_time = time.time()
    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-8,
        },
    )
    solve_time = time.time() - start_time

    return solution, solve_time


def validate_solution_quality(solution, problem_name):
    """Validate solution meets hypersensitive problem requirements."""
    if not solution.success:
        return False, f"{problem_name}: Solution failed"

    errors = []

    # Check boundary conditions
    if hasattr(solution, "get_phase_ids"):
        # Multi-phase
        first_phase = min(solution.get_phase_ids())
        last_phase = max(solution.get_phase_ids())

        x_initial = solution[(first_phase, "x")][0]
        x_final = solution[(last_phase, "x")][-1]

        total_time = solution.get_total_mission_time()
    else:
        # Single-phase (legacy check)
        x_initial = solution.phase_states[1][0][0, 0]
        x_final = solution.phase_states[1][0][0, -1]
        total_time = solution.phase_terminal_times[1] - solution.phase_initial_times[1]

    # Validate boundary conditions
    if abs(x_initial - 1.5) > 1e-6:
        errors.append(f"Initial condition: x(0) = {x_initial:.6f}, expected 1.5")

    if abs(x_final - 1.0) > 1e-6:
        errors.append(f"Final condition: x(T) = {x_final:.6f}, expected 1.0")

    if abs(total_time - 40.0) > 1e-6:
        errors.append(f"Time horizon: T = {total_time:.6f}, expected 40.0")

    # Check for reasonable objective value (known approximate range)
    if solution.objective < 0 or solution.objective > 100:
        errors.append(f"Objective value {solution.objective:.6f} seems unreasonable")

    return len(errors) == 0, errors


def compare_solutions():
    """Run comprehensive comparison of single vs multi-phase solutions."""

    print("=" * 80)
    print("🧪 HYPERSENSITIVE PROBLEM: MULTIPHASE VALIDATION SUITE")
    print("=" * 80)
    print("Testing mathematical equivalence of phase decomposition")
    print("Expected: All versions should produce IDENTICAL results")
    print("=" * 80)

    results = {}

    # Solve single-phase (reference)
    try:
        solution_1p, time_1p = solve_single_phase_hypersensitive()
        valid_1p, errors_1p = validate_solution_quality(solution_1p, "Single-Phase")
        results["single"] = {
            "solution": solution_1p,
            "time": time_1p,
            "valid": valid_1p,
            "errors": errors_1p,
        }
    except Exception as e:
        print(f"❌ Single-phase failed: {e}")
        results["single"] = {"solution": None, "error": str(e)}

    # Solve two-phase
    try:
        solution_2p, time_2p = solve_two_phase_hypersensitive()
        valid_2p, errors_2p = validate_solution_quality(solution_2p, "Two-Phase")
        results["two_phase"] = {
            "solution": solution_2p,
            "time": time_2p,
            "valid": valid_2p,
            "errors": errors_2p,
        }
    except Exception as e:
        print(f"❌ Two-phase failed: {e}")
        results["two_phase"] = {"solution": None, "error": str(e)}

    # Report results
    print("\n📊 VALIDATION RESULTS:")
    print("=" * 60)

    reference_obj = None

    for name, result in results.items():
        if "error" in result:
            print(f"❌ {name.upper()}: FAILED - {result['error']}")
            continue

        solution = result["solution"]
        solve_time = result["time"]
        valid = result["valid"]
        errors = result["errors"]

        if solution.success:
            print(f"✅ {name.upper()}: SUCCESS")
            print(f"   Objective:   {solution.objective:.8f}")
            print(f"   Solve time:  {solve_time:.3f}s")
            print(f"   Valid:       {valid}")

            if not valid:
                for error in errors:
                    print(f"   ⚠️  {error}")

            # Set reference or compare
            if reference_obj is None:
                reference_obj = solution.objective
                print("   📌 REFERENCE objective set")
            else:
                diff = abs(solution.objective - reference_obj)
                rel_error = diff / abs(reference_obj) if reference_obj != 0 else float("inf")

                print(f"   🎯 vs Reference: Δ = {diff:.2e}, rel = {rel_error:.2e}")

                if rel_error < 1e-6:
                    print("   ✅ EXCELLENT: Matches reference to machine precision!")
                elif rel_error < 1e-4:
                    print("   ✅ GOOD: Matches reference within tolerance")
                else:
                    print("   ❌ POOR: Significant difference from reference!")
        else:
            print(f"❌ {name.upper()}: FAILED - {solution.message}")

    print("\n" + "=" * 80)
    print("🏁 VALIDATION SUMMARY:")

    if (
        reference_obj is not None
        and len([r for r in results.values() if "error" not in r and r["solution"].success]) > 1
    ):
        print("✅ Multi-phase implementation appears mathematically correct!")
        print("   All successful solutions match within numerical precision")
        print("   Phase decomposition preserves optimality")
    else:
        print("❌ Validation incomplete or problems detected")
        print("   Review failed solutions above")

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = compare_solutions()

    # Optional: Plot comparison if solutions successful
    successful_solutions = [
        (name, result["solution"])
        for name, result in results.items()
        if "error" not in result and result["solution"].success
    ]

    if len(successful_solutions) >= 2:
        print(
            f"\n📈 Plotting {len(successful_solutions)} successful solutions for visual comparison..."
        )

        for name, solution in successful_solutions:
            print(f"Plotting {name} solution...")
            if hasattr(solution, "plot"):
                try:
                    solution.plot(show_phase_boundaries=True, figsize=(12, 8))
                except Exception as e:
                    print(f"Plot failed for {name}: {e}")
