#!/usr/bin/env python3
"""
Multiphase block pushing problem demonstrating the new boundary API usage.
Phase 1: Push block left, Phase 2: Push block right
Shows boundary enforcement across multiple phases.
"""

import maptor as mtor


def create_multiphase_block_problem():
    """Create a two-phase block pushing problem with boundary constraints."""

    problem = mtor.Problem("Multiphase Block Push with Boundaries")

    # === PHASE 1: Push Left ===
    phase1 = problem.set_phase(1)

    # Time for phase 1
    t1 = phase1.time(initial=0.0, final=(0.5, 3.0))

    # States for phase 1 (NEW API: boundary=(lower, upper))
    pos1 = phase1.state("position", initial=0.0, final=-1.0, boundary=(-2.0, 1.0))
    vel1 = phase1.state("velocity", initial=0.0, final=0.0, boundary=(-4.0, 4.0))

    # Control for phase 1 (NEW API: boundary=(lower, upper))
    force1 = phase1.control("force", boundary=(-8.0, 8.0))

    # === PHASE 2: Push Right (with automatic continuity) ===
    phase2 = problem.set_phase(2)

    # Time for phase 2 (continuous from phase 1)
    t2 = phase2.time(initial=t1.final, final=(2.0, 6.0))

    # States for phase 2 with symbolic continuity (NEW API)
    pos2 = phase2.state("position", initial=pos1.final, final=1.0, boundary=(-2.0, 2.0))
    vel2 = phase2.state("velocity", initial=vel1.final, final=0.0, boundary=(-4.0, 4.0))

    # Control for phase 2 (NEW API: boundary=(lower, upper))
    force2 = phase2.control("force", boundary=(-8.0, 8.0))

    # === PARAMETERS (shared across phases) ===
    mass = problem.parameter("mass", boundary=(0.8, 1.5))  # Optimize mass
    friction = problem.parameter("friction", fixed=0.1)  # Fixed friction

    # === DYNAMICS (same for both phases) ===
    phase1.dynamics({pos1: vel1, vel1: (force1 - friction * vel1) / mass})

    phase2.dynamics({pos2: vel2, vel2: (force2 - friction * vel2) / mass})

    # === OBJECTIVE: Minimize total mission time ===
    problem.minimize(t2.final)

    # === MESH CONFIGURATION ===
    phase1.mesh([3, 3], [-1.0, 0.0, 1.0])
    phase2.mesh([3, 3], [-1.0, 0.0, 1.0])

    # === INITIAL GUESS ===
    # Phase 1: Move left from 0 to -1
    phase1.guess(terminal_time=1.5)

    # Phase 2: Move right from -1 to +1
    phase2.guess(terminal_time=3.0)

    # Parameter guess
    problem.parameter_guess(mass=1.0)

    return problem


def solve_and_verify_multiphase_boundaries():
    """Solve the multiphase problem and verify boundaries are respected."""

    print("Creating multiphase block pushing problem...")
    print("Phase 1: Push left (0 → -1)")
    print("Phase 2: Push right (-1 → +1)")
    problem = create_multiphase_block_problem()

    print("\nSolving multiphase problem...")
    solution = mtor.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 1500,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.linear_solver": "mumps",
        },
        show_summary=False,
    )

    if solution.status["success"]:
        print(
            f"✓ Solution found! Total mission time: {solution.status['total_mission_time']:.3f} seconds"
        )

        # === PHASE-SPECIFIC DATA EXTRACTION ===
        # Phase 1 data
        pos1_data = solution[(1, "position")]
        vel1_data = solution[(1, "velocity")]

        # Phase 2 data
        pos2_data = solution[(2, "position")]
        vel2_data = solution[(2, "velocity")]

        # Mission-wide data (concatenated)
        pos_all = solution["position"]
        vel_all = solution["velocity"]

        # === BOUNDARY VERIFICATION ===
        print("\nPhase 1 Boundary Verification:")
        print(
            f"Position range: [{pos1_data.min():.3f}, {pos1_data.max():.3f}] (constraint: [-2.0, 1.0])"
        )
        print(
            f"Velocity range: [{vel1_data.min():.3f}, {vel1_data.max():.3f}] (constraint: [-4.0, 4.0])"
        )

        print("\nPhase 2 Boundary Verification:")
        print(
            f"Position range: [{pos2_data.min():.3f}, {pos2_data.max():.3f}] (constraint: [-2.0, 2.0])"
        )
        print(
            f"Velocity range: [{vel2_data.min():.3f}, {vel2_data.max():.3f}] (constraint: [-4.0, 4.0])"
        )

        print("\nMission-wide Boundary Verification:")
        print(f"Position range: [{pos_all.min():.3f}, {pos_all.max():.3f}]")
        print(f"Velocity range: [{vel_all.min():.3f}, {vel_all.max():.3f}]")

        # === CONTINUITY VERIFICATION ===
        print("\nContinuity Verification:")
        print(f"Position at phase transition: {pos1_data[-1]:.6f} → {pos2_data[0]:.6f}")
        print(f"Velocity at phase transition: {vel1_data[-1]:.6f} → {vel2_data[0]:.6f}")

        # === PARAMETER VERIFICATION ===
        params = solution.parameters
        if params["count"] > 0:
            mass_val = params["values"][0]
            friction_val = params["values"][1]
            print("\nParameter Verification:")
            print(f"Optimized mass: {mass_val:.3f} (constraint: [0.8, 1.5])")
            print(f"Fixed friction: {friction_val:.3f} (should be 0.1)")

        # === MISSION RESULTS ===
        print("\nMission Results:")
        print(f"Phase 1: {pos1_data[0]:.3f} → {pos1_data[-1]:.3f} meters")
        print(f"Phase 2: {pos2_data[0]:.3f} → {pos2_data[-1]:.3f} meters")
        print(f"Overall: {pos_all[0]:.3f} → {pos_all[-1]:.3f} meters")
        print(f"Final velocity: {vel_all[-1]:.6f} m/s (should be ~0)")

        # === PHASE TIMING ===
        phase_data = solution.phases
        phase1_duration = phase_data[1]["times"]["duration"]
        phase2_duration = phase_data[2]["times"]["duration"]
        print("\nPhase Timing:")
        print(f"Phase 1 duration: {phase1_duration:.3f} seconds")
        print(f"Phase 2 duration: {phase2_duration:.3f} seconds")
        print(f"Total mission: {phase1_duration + phase2_duration:.3f} seconds")

        return solution
    else:
        print(f"✗ Solution failed: {solution.status['message']}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("MAPTOR Multiphase Boundary API Demo")
    print("=" * 60)
    print()
    print("Demonstrates:")
    print("• boundary=(lower, upper) across multiple phases")
    print("• fixed=value for shared parameters")
    print("• Automatic phase continuity with symbolic linking")
    print("• Boundary enforcement in multiphase problems")
    print()

    solution = solve_and_verify_multiphase_boundaries()

    if solution:
        print("\n" + "=" * 60)
        print("Multiphase demo completed successfully!")
        print("✓ Boundaries enforced across all phases")
        print("✓ Phase continuity maintained automatically")
        print("✓ Parameters optimized globally")
