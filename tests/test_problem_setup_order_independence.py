import numpy as np
import pytest

from maptor import ConfigurationError, DataIntegrityError, Problem, solve_fixed_mesh
from maptor.problem.core_problem import Phase
from maptor.tl_types import FloatArray


class TestProblemSetupOrderIndependence:
    def create_standard_problem(self) -> tuple[Problem, Phase]:
        problem = Problem("Order Test Problem")

        # Create phase first
        phase = problem.set_phase(1)

        # Define variables on phase
        _t = phase.time(initial=0.0, final=1.0)
        x = phase.state("x", initial=0.0, final=1.0)
        u = phase.control("u", boundary=(-2.0, 2.0))

        # Define dynamics on phase
        phase.dynamics({x: u})

        # Define integral on phase
        integral_var = phase.add_integral(u**2)
        problem.minimize(integral_var)

        return problem, phase

    def create_initial_guess(self) -> tuple[list[FloatArray], list[FloatArray]]:
        # For mesh [3], we need states shape (1, 4) and controls shape (1, 3)
        states = [np.array([[0.0, 0.3, 0.6, 1.0]])]  # Linear interpolation
        controls = [np.array([[1.0, 1.0, 1.0]])]  # Constant control
        return states, controls

    def _solve_with_mesh_first_order(self) -> tuple[float, float]:
        problem, phase = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        # Order: mesh first, then guess, then solve
        # Set mesh on phase
        phase.mesh([3], [-1.0, 1.0])

        # Set initial guess using multiphase format
        problem.guess(phase_states={1: states}, phase_controls={1: controls})

        solution = solve_fixed_mesh(problem)
        assert solution.status["success"], f"Mesh→Guess→Solve failed: {solution.status['message']}"

        # Access results by phase
        return solution.status["objective"], solution[(1, "x")][-1]

    def _solve_with_guess_first_order(self) -> tuple[float, float]:
        problem, phase = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        # Order: guess first, then mesh, then solve
        # Set initial guess using multiphase format
        problem.guess(phase_states={1: states}, phase_controls={1: controls})

        # Set mesh on phase
        phase.mesh([3], [-1.0, 1.0])

        solution = solve_fixed_mesh(problem)
        assert solution.status["success"], f"Guess→Mesh→Solve failed: {solution.status['message']}"

        x_trajectory = solution[(1, "x")]
        return solution.status["objective"], float(x_trajectory[-1])

    def test_mesh_first_then_guess_then_solve(self):
        # This is a focused integration test that verifies the mesh-first workflow
        objective, final_state = self._solve_with_mesh_first_order()

        # Verify solution makes sense (basic sanity checks)
        assert objective > 0, "Objective should be positive for this control problem"
        assert abs(final_state - 1.0) < 1e-6, "Final state should match boundary condition"

    def test_guess_first_then_mesh_then_solve(self):
        # This is a focused integration test that verifies the guess-first workflow
        objective, final_state = self._solve_with_guess_first_order()

        # Verify solution makes sense (basic sanity checks)
        assert objective > 0, "Objective should be positive for this control problem"
        assert abs(final_state - 1.0) < 1e-6, "Final state should match boundary condition"

    def test_order_independence_gives_same_results(self):
        # Get results from both orders
        ref_obj, ref_state = self._solve_with_mesh_first_order()
        alt_obj, alt_state = self._solve_with_guess_first_order()

        # Results must be identical regardless of order
        assert abs(ref_obj - alt_obj) < 1e-10, (
            f"Different orders give different objectives: {ref_obj} vs {alt_obj}"
        )
        assert abs(ref_state - alt_state) < 1e-10, (
            f"Different orders give different final states: {ref_state} vs {alt_state}"
        )

    def test_modify_existing_problem_scenarios(self):
        # Scenario 1: mesh → solve → mesh → solve (mesh refinement)
        problem1, phase1 = self.create_standard_problem()
        states1, controls1 = self.create_initial_guess()

        # Initial solve
        phase1.mesh([3], [-1.0, 1.0])
        problem1.guess(phase_states={1: states1}, phase_controls={1: controls1})
        solution1a = solve_fixed_mesh(problem1)
        assert solution1a.status["success"], "Initial solve failed"

        # Refine mesh and solve again
        states1_refined = [np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])]  # For N=4
        controls1_refined = [np.array([[1.0, 1.0, 1.0, 1.0]])]

        phase1.mesh([4], [-1.0, 1.0])  # Finer mesh
        problem1.guess(phase_states={1: states1_refined}, phase_controls={1: controls1_refined})
        solution1b = solve_fixed_mesh(problem1)
        assert solution1b.status["success"], "Refined solve failed"

        # Refined solution should be at least as good (lower objective)
        assert solution1b.status["objective"] <= solution1a.status["objective"] + 1e-6, (
            "Mesh refinement should not worsen the solution significantly"
        )

    def test_partial_modification_scenarios(self):
        problem, phase = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        # Initial setup and solve
        phase.mesh([3], [-1.0, 1.0])
        problem.guess(phase_states={1: states}, phase_controls={1: controls})
        solution1 = solve_fixed_mesh(problem)
        assert solution1.status["success"], "Initial solve failed"

        # Scenario: Change only mesh, keep same initial guess structure
        phase.mesh([5], [-1.0, 1.0])  # Different polynomial degree
        # Need new initial guess for new mesh size
        states_new = [np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])]  # 6 state points for N=5
        controls_new = [np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])]  # 5 control points for N=5

        problem.guess(phase_states={1: states_new}, phase_controls={1: controls_new})
        solution2 = solve_fixed_mesh(problem)
        assert solution2.status["success"], "Modified mesh solve failed"

        # Scenario: Change mesh without changing initial guess (should validate at solve time)
        problem3, phase3 = self.create_standard_problem()

        phase3.mesh([2], [-1.0, 1.0])
        problem3.guess(phase_states={1: states}, phase_controls={1: controls})  # Wrong size for N=2

        # This should fail gracefully with informative error
        with pytest.raises(DataIntegrityError) as exc_info:
            solve_fixed_mesh(problem3)

        error_msg = str(exc_info.value).lower()
        assert "shape" in error_msg or "size" in error_msg or "dimension" in error_msg, (
            f"Should get informative error about size mismatch, got: {exc_info.value}"
        )

    def test_incomplete_problem_specifications(self):
        # Test 1: Mesh set but no initial guess
        problem1, phase1 = self.create_standard_problem()

        phase1.mesh([3], [-1.0, 1.0])
        # No initial guess set - should work (solver uses defaults)

        solution1 = solve_fixed_mesh(problem1)
        assert solution1.status["success"], "No initial guess should work with defaults"

        # Test 2: Initial guess set but no mesh (should fail at solve time)
        problem2, _phase2 = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        problem2.guess(phase_states={1: states}, phase_controls={1: controls})
        # No mesh set

        with pytest.raises(ConfigurationError, match="mesh") as exc_info:
            solve_fixed_mesh(problem2)

        error_msg = str(exc_info.value).lower()
        assert "mesh" in error_msg, f"Should mention mesh requirement, got: {exc_info.value}"

    def test_multi_interval_order_independence(self):
        def create_multi_interval_problem():
            problem = Problem("Multi-Interval Test")

            # Create phase
            phase = problem.set_phase(1)

            # Define variables on phase
            _t = phase.time(initial=0.0, final=2.0)
            x = phase.state("x", initial=0.0, final=2.0)
            u = phase.control("u", boundary=(-1.0, 1.0))

            # Define dynamics and objective on phase
            phase.dynamics({x: u})
            integral_var = phase.add_integral(u**2)
            problem.minimize(integral_var)

            return problem, phase

        def create_multi_interval_guess():
            # For mesh [2, 3] we need 2 intervals
            states = [
                np.array([[0.0, 0.5, 1.0]]),  # Interval 1: N=2, so 3 state points
                np.array([[1.0, 1.5, 1.75, 2.0]]),  # Interval 2: N=3, so 4 state points
            ]
            controls = [
                np.array([[0.5, 0.5]]),  # Interval 1: N=2, so 2 control points
                np.array([[0.5, 0.5, 0.5]]),  # Interval 2: N=3, so 3 control points
            ]
            return states, controls

        # Test mesh-first order
        problem1, phase1 = create_multi_interval_problem()
        states, controls = create_multi_interval_guess()

        phase1.mesh([2, 3], [-1.0, 0.0, 1.0])
        problem1.guess(phase_states={1: states}, phase_controls={1: controls})
        solution1 = solve_fixed_mesh(problem1)
        assert solution1.status["success"], "Multi-interval mesh-first failed"

        # Test guess-first order
        problem2, phase2 = create_multi_interval_problem()
        states2, controls2 = create_multi_interval_guess()

        problem2.guess(phase_states={1: states2}, phase_controls={1: controls2})
        phase2.mesh([2, 3], [-1.0, 0.0, 1.0])
        solution2 = solve_fixed_mesh(problem2)
        assert solution2.status["success"], "Multi-interval guess-first failed"

        # Results should be identical
        assert abs(solution1.status["objective"] - solution2.status["objective"]) < 1e-12, (
            "Multi-interval order dependence detected"
        )

    def test_error_propagation_and_recovery(self):
        problem, phase = self.create_standard_problem()

        # Set mesh first
        phase.mesh([3], [-1.0, 1.0])

        # Try to set wrong initial guess
        wrong_states = [np.array([[0.0, 1.0]])]  # Wrong size: should be (1, 4)
        wrong_controls = [np.array([[1.0]])]  # Wrong size: should be (1, 3)

        # This should be caught at solve time, not at guess time
        problem.guess(phase_states={1: wrong_states}, phase_controls={1: wrong_controls})

        # Solve should fail with clear error message
        with pytest.raises(DataIntegrityError) as exc_info:
            solve_fixed_mesh(problem)

        error_msg = str(exc_info.value).lower()
        assert "shape" in error_msg or "dimension" in error_msg, (
            f"Error message should be informative: {exc_info.value}"
        )

        # System should recover - set correct initial guess
        correct_states, correct_controls = self.create_initial_guess()
        problem.guess(phase_states={1: correct_states}, phase_controls={1: correct_controls})

        solution = solve_fixed_mesh(problem)
        assert solution.status["success"], "System should recover after fixing initial guess"
