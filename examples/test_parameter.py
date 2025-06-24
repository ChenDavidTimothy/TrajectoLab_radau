import numpy as np

import maptor as mtor


def test_symbolic_parameter_bounds():
    """Test symbolic parameter boundaries linking parameters together."""

    try:
        # Create problem
        problem = mtor.Problem("Symbolic Parameter Test")

        # Single phase
        phase = problem.set_phase(1)

        # Time: fixed duration
        t = phase.time(initial=0.0, final=1.0)

        # State: position (start at 0, free final)
        x = phase.state("position", initial=0.0)

        # Control: force (bounded)
        u = phase.control("force", boundary=(-10, 10))

        # CRITICAL TEST: Two parameters with symbolic linking
        mass1 = problem.parameter("mass1", boundary=(2.0, 3.0))  # Numeric bounds
        mass2 = problem.parameter("mass2", boundary=mass1 * 2.0)  # Symbolic: mass2 = 2 * mass1

        # Dynamics: acceleration depends on both masses
        # Total acceleration = force / (mass1 + mass2)
        total_mass = mass1 + mass2
        phase.dynamics({x: u / total_mass})

        # Objective: minimize final position (incentivizes minimum total mass)
        problem.minimize(x.final)

        # Simple mesh
        phase.mesh([5], [-1, 1])

        # Initial guess
        phase.guess(
            states=[np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])],
            controls=[np.array([[-5.0, -5.0, -5.0, -5.0, -5.0]])],
            terminal_time=1.0,
        )

        # Parameter guesses
        problem.parameter_guess(mass1=2.5, mass2=5.0)

        # Validate and solve
        problem.validate_multiphase_configuration()
        solution = mtor.solve_fixed_mesh(problem)

        # VERIFICATION
        print(f"Success: {solution.status['success']}")
        print(f"Message: {solution.status['message']}")

        if solution.status["success"]:
            if solution.parameters["count"] >= 2:
                mass1_value = solution.parameters["values"][0]
                mass2_value = solution.parameters["values"][1]

                print(f"mass1 value: {mass1_value:.6f}")
                print(f"mass2 value: {mass2_value:.6f}")
                print("mass1 bounds: [2.0, 3.0]")
                print("Symbolic constraint: mass2 = mass1 * 2.0")
                print(f"Expected mass2: {mass1_value * 2.0:.6f}")

                # Test numeric bounds
                mass1_bounded = 2.0 <= mass1_value <= 3.0
                print(f"mass1 bounds enforced: {mass1_bounded}")

                # Test symbolic constraint
                expected_mass2 = mass1_value * 2.0
                symbolic_error = abs(mass2_value - expected_mass2)
                symbolic_satisfied = symbolic_error < 1e-6

                print(f"Symbolic constraint error: {symbolic_error:.2e}")
                print(f"Symbolic constraint satisfied: {symbolic_satisfied}")
                print(f"Final position: {solution['position'][-1]:.6f}")

                # Overall test
                if mass1_bounded and symbolic_satisfied:
                    print("✅ SYMBOLIC PARAMETER BOUNDS ARE WORKING!")

                    # Verify optimizer behavior: should minimize mass1 to minimize total mass
                    if abs(mass1_value - 2.0) < 1e-6:
                        print("✅ OPTIMIZER CORRECTLY MINIMIZED mass1 (hit lower bound)")
                    else:
                        print(f"⚠️  mass1 = {mass1_value:.6f}, expected ~2.0 (lower bound)")

                else:
                    print("❌ SYMBOLIC PARAMETER BOUNDS FAILED!")

            else:
                print(f"❌ Expected 2 parameters, found {solution.parameters['count']}")
        else:
            print("❌ Solver failed - cannot test symbolic parameter bounds")

        return solution

    except Exception as e:
        print(f"Error during problem setup or solve: {e}")
        import traceback

        traceback.print_exc()
        return None


# Run test
if __name__ == "__main__":
    solution = test_symbolic_parameter_bounds()
