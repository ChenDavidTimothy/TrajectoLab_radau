# test_problem_setup_order_independence.py
"""
Safety-critical tests for problem setup order independence and state management.
Tests various scenarios of mesh/initial guess setting order.

FIXED: Removed return values from test methods to eliminate pytest warnings.
Aligned with Grug's Pragmatic Testing Philosophy:
- Integration tests that verify critical system behavior
- Simple, focused tests that catch real problems
- No excessive mocking or over-engineering
"""

import numpy as np
import pytest

from trajectolab import Problem, solve_fixed_mesh
from trajectolab.tl_types import FloatArray


class TestProblemSetupOrderIndependence:
    """Test that problem setup works correctly regardless of order of operations."""

    def create_standard_problem(self) -> Problem:
        """Create a standard test problem for order testing."""
        problem = Problem("Order Test Problem")

        t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=1.0)
        u = problem.control("u", boundary=(-2.0, 2.0))

        problem.dynamics({x: u})

        integral_var = problem.add_integral(u**2)
        problem.minimize(integral_var)

        return problem

    def create_initial_guess(self) -> tuple[list[FloatArray], list[FloatArray]]:
        """Create standard initial guess arrays."""
        # For mesh [3], we need states shape (1, 4) and controls shape (1, 3)
        states = [np.array([[0.0, 0.3, 0.6, 1.0]])]  # Linear interpolation
        controls = [np.array([[1.0, 1.0, 1.0]])]  # Constant control
        return states, controls

    def _solve_with_mesh_first_order(self) -> tuple[float, float]:
        """Helper method: set_mesh() → set_initial_guess() → solve()"""
        problem = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        # Order: mesh first, then guess, then solve
        problem.set_mesh([3], [-1.0, 1.0])
        problem.set_initial_guess(states=states, controls=controls)

        solution = solve_fixed_mesh(problem)
        assert solution.success, f"Mesh→Guess→Solve failed: {solution.message}"

        return solution.objective, solution.states[0][-1]

    def _solve_with_guess_first_order(self) -> tuple[float, float]:
        """Helper method: set_initial_guess() → set_mesh() → solve()"""
        problem = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        # Order: guess first, then mesh, then solve
        problem.set_initial_guess(states=states, controls=controls)
        problem.set_mesh([3], [-1.0, 1.0])

        solution = solve_fixed_mesh(problem)
        assert solution.success, f"Guess→Mesh→Solve failed: {solution.message}"

        return solution.objective, solution.states[0][-1]

    def test_mesh_first_then_guess_then_solve(self):
        """Test: set_mesh() → set_initial_guess() → solve() - Integration Test"""
        # This is a focused integration test that verifies the mesh-first workflow
        objective, final_state = self._solve_with_mesh_first_order()

        # Verify solution makes sense (basic sanity checks)
        assert objective > 0, "Objective should be positive for this control problem"
        assert abs(final_state - 1.0) < 1e-6, "Final state should match boundary condition"

    def test_guess_first_then_mesh_then_solve(self):
        """Test: set_initial_guess() → set_mesh() → solve() - Integration Test"""
        # This is a focused integration test that verifies the guess-first workflow
        objective, final_state = self._solve_with_guess_first_order()

        # Verify solution makes sense (basic sanity checks)
        assert objective > 0, "Objective should be positive for this control problem"
        assert abs(final_state - 1.0) < 1e-6, "Final state should match boundary condition"

    def test_order_independence_gives_same_results(self):
        """
        CRITICAL INTEGRATION TEST: Different orders must give identical results.

        This is the most important test - it verifies that the core system behavior
        (order independence) works correctly. This is exactly the kind of integration
        test that Grug's philosophy emphasizes as most valuable.
        """
        # Get results from both orders
        ref_obj, ref_state = self._solve_with_mesh_first_order()
        alt_obj, alt_state = self._solve_with_guess_first_order()

        # Results must be identical regardless of order - this is CRITICAL
        assert abs(ref_obj - alt_obj) < 1e-10, (
            f"CRITICAL: Different orders give different objectives: {ref_obj} vs {alt_obj}"
        )
        assert abs(ref_state - alt_state) < 1e-10, (
            f"CRITICAL: Different orders give different final states: {ref_state} vs {alt_state}"
        )

    def test_modify_existing_problem_scenarios(self):
        """
        Integration test for problem modification workflows (critical for NASA operations).

        This tests real-world usage patterns where engineers iteratively refine problems.
        """
        # Scenario 1: set_mesh → solve → set_mesh → solve (mesh refinement)
        problem1 = self.create_standard_problem()
        states1, controls1 = self.create_initial_guess()

        # Initial solve
        problem1.set_mesh([3], [-1.0, 1.0])
        problem1.set_initial_guess(states=states1, controls=controls1)
        solution1a = solve_fixed_mesh(problem1)
        assert solution1a.success, "Initial solve failed"

        # Refine mesh and solve again
        states1_refined = [np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])]  # For N=4
        controls1_refined = [np.array([[1.0, 1.0, 1.0, 1.0]])]

        problem1.set_mesh([4], [-1.0, 1.0])  # Finer mesh
        problem1.set_initial_guess(states=states1_refined, controls=controls1_refined)
        solution1b = solve_fixed_mesh(problem1)
        assert solution1b.success, "Refined solve failed"

        # Refined solution should be at least as good (lower objective)
        assert solution1b.objective <= solution1a.objective + 1e-6, (
            "Mesh refinement should not worsen the solution significantly"
        )

    def test_partial_modification_scenarios(self):
        """Test partial modifications without complete re-specification."""

        problem = self.create_standard_problem()
        states, controls = self.create_initial_guess()

        # Initial setup and solve
        problem.set_mesh([3], [-1.0, 1.0])
        problem.set_initial_guess(states=states, controls=controls)
        solution1 = solve_fixed_mesh(problem)
        assert solution1.success, "Initial solve failed"

        # Scenario: Change only mesh, keep same initial guess structure
        problem.set_mesh([5], [-1.0, 1.0])  # Different polynomial degree
        # Need new initial guess for new mesh size
        states_new = [np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])]  # 6 state points for N=5
        controls_new = [np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])]  # 5 control points for N=5

        problem.set_initial_guess(states=states_new, controls=controls_new)
        solution2 = solve_fixed_mesh(problem)
        assert solution2.success, "Modified mesh solve failed"

        # Scenario: Change mesh without changing initial guess (should validate at solve time)
        problem3 = self.create_standard_problem()
        problem3.set_mesh([2], [-1.0, 1.0])
        problem3.set_initial_guess(states=states, controls=controls)  # Wrong size for N=2

        # This should fail gracefully with informative error
        with pytest.raises((ValueError, AssertionError)) as exc_info:
            solve_fixed_mesh(problem3)

        error_msg = str(exc_info.value).lower()
        assert "shape" in error_msg or "size" in error_msg or "dimension" in error_msg, (
            f"Should get informative error about size mismatch, got: {exc_info.value}"
        )

    def test_incomplete_problem_specifications(self):
        """
        Test handling of incomplete problem specifications.

        This is a critical safety test - the system must handle incomplete
        configurations gracefully and provide clear error messages.
        """

        # Test 1: Mesh set but no initial guess
        problem1 = self.create_standard_problem()
        problem1.set_mesh([3], [-1.0, 1.0])
        # No initial guess set - should work (solver uses defaults)

        solution1 = solve_fixed_mesh(problem1)
        assert solution1.success, "No initial guess should work with defaults"

        # Test 2: Initial guess set but no mesh (should fail at solve time)
        problem2 = self.create_standard_problem()
        states, controls = self.create_initial_guess()
        problem2.set_initial_guess(states=states, controls=controls)
        # No mesh set

        with pytest.raises(ValueError) as exc_info:
            solve_fixed_mesh(problem2)

        error_msg = str(exc_info.value).lower()
        assert "mesh" in error_msg, f"Should mention mesh requirement, got: {exc_info.value}"

    def test_requirements_and_summary_consistency(self):
        """
        Test that requirements and summary reporting is consistent with actual setup.

        This is important for user experience - the system should provide accurate
        feedback about what's needed.
        """

        problem = self.create_standard_problem()

        # Initially, requirements should indicate mesh needed
        reqs1 = problem.get_initial_guess_requirements()
        assert len(reqs1.states_shapes) == 0, "Should indicate mesh needed"

        # After mesh set, should show specific requirements
        problem.set_mesh([3], [-1.0, 1.0])
        reqs2 = problem.get_initial_guess_requirements()
        assert len(reqs2.states_shapes) == 1, "Should show 1 interval"
        assert reqs2.states_shapes[0] == (1, 4), (
            "Should require (1, 4) state array"
        )  # 1 state, N+1 points
        assert reqs2.controls_shapes[0] == (1, 3), (
            "Should require (1, 3) control array"
        )  # 1 control, N points

        # Summary should reflect current state
        summary1 = problem.get_solver_input_summary()
        assert "no_initial_guess" in summary1.initial_guess_source

        # After setting initial guess
        states, controls = self.create_initial_guess()
        problem.set_initial_guess(states=states, controls=controls)

        summary2 = problem.get_solver_input_summary()
        assert "partial_user_provided" in summary2.initial_guess_source
        assert summary2.states_guess_shapes == [(1, 4)]
        assert summary2.controls_guess_shapes == [(1, 3)]

        # After setting initial guess
        states, controls = self.create_initial_guess()
        problem.set_initial_guess(states=states, controls=controls, integrals=0.1)

        summary3 = problem.get_solver_input_summary()
        assert "complete_user_provided" in summary3.initial_guess_source
        assert summary3.states_guess_shapes == [(1, 4)]
        assert summary3.controls_guess_shapes == [(1, 3)]

    def test_multi_interval_order_independence(self):
        """
        Test order independence with multi-interval meshes.

        This is a more complex integration test that verifies the system works
        correctly with realistic multi-interval problems.
        """

        def create_multi_interval_problem():
            problem = Problem("Multi-Interval Test")
            t = problem.time(initial=0.0, final=2.0)
            x = problem.state("x", initial=0.0, final=2.0)
            u = problem.control("u", boundary=(-1.0, 1.0))
            problem.dynamics({x: u})
            problem.minimize(problem.add_integral(u**2))
            return problem

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
        problem1 = create_multi_interval_problem()
        states, controls = create_multi_interval_guess()

        problem1.set_mesh([2, 3], [-1.0, 0.0, 1.0])
        problem1.set_initial_guess(states=states, controls=controls)
        solution1 = solve_fixed_mesh(problem1)
        assert solution1.success, "Multi-interval mesh-first failed"

        # Test guess-first order
        problem2 = create_multi_interval_problem()
        states2, controls2 = create_multi_interval_guess()

        problem2.set_initial_guess(states=states2, controls=controls2)
        problem2.set_mesh([2, 3], [-1.0, 0.0, 1.0])
        solution2 = solve_fixed_mesh(problem2)
        assert solution2.success, "Multi-interval guess-first failed"

        # Results should be identical
        assert abs(solution1.objective - solution2.objective) < 1e-12, (
            "Multi-interval order dependence detected"
        )

    def test_error_propagation_and_recovery(self):
        """
        Test that errors are properly caught and system can recover.

        This is a critical safety test - the system must handle errors gracefully
        and allow recovery without requiring complete restart.
        """

        problem = self.create_standard_problem()

        # Set mesh first
        problem.set_mesh([3], [-1.0, 1.0])

        # Try to set wrong initial guess
        wrong_states = [np.array([[0.0, 1.0]])]  # Wrong size: should be (1, 4)
        wrong_controls = [np.array([[1.0]])]  # Wrong size: should be (1, 3)

        # This should be caught at solve time, not at set_initial_guess time
        problem.set_initial_guess(states=wrong_states, controls=wrong_controls)

        # Solve should fail with clear error message
        with pytest.raises((ValueError, AssertionError)) as exc_info:
            solve_fixed_mesh(problem)

        error_msg = str(exc_info.value).lower()
        assert "shape" in error_msg or "dimension" in error_msg, (
            f"Error message should be informative: {exc_info.value}"
        )

        # System should recover - set correct initial guess
        correct_states, correct_controls = self.create_initial_guess()
        problem.set_initial_guess(states=correct_states, controls=correct_controls)

        solution = solve_fixed_mesh(problem)
        assert solution.success, "System should recover after fixing initial guess"
