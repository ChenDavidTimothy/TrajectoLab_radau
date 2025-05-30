"""
Multi-phase debug test script.

This script tests the multi-phase implementation step by step to identify
exactly where failures occur.
"""

import logging

import numpy as np


# Enable logging to see detailed output
logging.getLogger("trajectolab").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

try:
    import trajectolab as tl

    print("✓ TrajectoLab imported successfully")
except Exception as e:
    print(f"✗ Failed to import TrajectoLab: {e}")
    exit(1)


def test_multi_phase_step_by_step():
    print("\n=== Multi-Phase Problem Creation Test ===")

    try:
        # Step 1: Create multi-phase problem
        print("Step 1: Creating multi-phase problem...")
        mp_problem = tl.MultiPhaseProblem("Debug Test Mission")
        print("✓ Multi-phase problem created")

        # Step 2: Add phases
        print("Step 2: Adding phases...")
        ascent = mp_problem.add_phase("Ascent")
        coast = mp_problem.add_phase("Coast")
        print("✓ Phases added successfully")

        # Step 3: Configure ascent phase
        print("Step 3: Configuring ascent phase...")
        t1 = ascent.time(initial=0.0)
        h1 = ascent.state("altitude", initial=0.0)
        v1 = ascent.state("velocity", initial=0.0)
        u1 = ascent.control("thrust", boundary=(0.0, 1.0))
        ascent.dynamics({h1: v1, v1: u1})
        ascent.set_mesh([5], np.array([-1.0, 1.0]))  # Smaller mesh for debugging
        print("✓ Ascent phase configured")

        # Step 4: Configure coast phase
        print("Step 4: Configuring coast phase...")
        t2 = coast.time()
        h2 = coast.state("altitude")
        v2 = coast.state("velocity")
        coast.dynamics({h2: v2, v2: 0})  # Ballistic coast
        coast.set_mesh([3], np.array([-1.0, 1.0]))  # Smaller mesh for debugging
        print("✓ Coast phase configured")

        # Step 5: Add continuity constraints
        print("Step 5: Adding phase linking constraints...")
        mp_problem.link_phases(h1.final == h2.initial)  # Altitude continuity
        mp_problem.link_phases(v1.final == v2.initial)  # Velocity continuity
        mp_problem.link_phases(t1.final == t2.initial)  # Time continuity
        print("✓ Phase linking constraints added")

        # Step 6: Set global objective
        print("Step 6: Setting global objective...")
        total_time = (t1.final - t1.initial) + (t2.final - t2.initial)
        mp_problem.set_global_objective(total_time)
        print("✓ Global objective set")

        # Step 7: Validate problem structure
        print("Step 7: Validating problem structure...")
        mp_problem.validate_complete_structure()
        print("✓ Problem structure validation passed")

        # Step 8: Print problem summary
        print("Step 8: Problem summary...")
        summary = mp_problem.get_problem_summary()
        print(f"  Phases: {summary['phase_count']}")
        print(f"  Global parameters: {len(summary['global_parameters'])}")
        print(f"  Inter-phase constraints: {summary['inter_phase_constraints_count']}")
        print(f"  Global objective set: {summary['global_objective_set']}")

        # Step 9: Test solver interface methods
        print("Step 9: Testing solver interface methods...")
        try:
            obj_func = mp_problem.get_global_objective_function()
            print("✓ Global objective function obtained")
        except Exception as e:
            print(f"✗ Global objective function failed: {e}")
            return False

        try:
            constraint_func = mp_problem.get_inter_phase_constraints_function()
            if constraint_func is not None:
                print("✓ Inter-phase constraints function obtained")
            else:
                print("✓ No inter-phase constraints (expected for this simple case)")
        except Exception as e:
            print(f"✗ Inter-phase constraints function failed: {e}")
            return False

        # Step 10: Attempt solve
        print("Step 10: Attempting to solve...")
        solution = tl.solve_multi_phase_fixed_mesh(mp_problem)

        print(f"✓ Solver completed, success: {solution.success}")
        if not solution.success:
            print(f"  Failure message: {solution.message}")
        else:
            print(f"  Objective: {solution.objective}")
            print(f"  Phase count: {solution.phase_count}")

        return solution.success

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")

        # Print full traceback for debugging
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Multi-Phase TrajectoLab Debug Test")
    print("=" * 40)

    success = test_multi_phase_step_by_step()

    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Multi-phase implementation is working.")
    else:
        print("✗ Tests failed. Check the error messages above.")
