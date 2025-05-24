# test_core_solver_integration.py
"""
Safety-critical integration tests for core solver with known analytical solutions.
"""

import casadi as ca
import numpy as np

from trajectolab import Problem, solve_fixed_mesh


class TestCoreSolverIntegration:
    """Integration tests for core solver against known analytical solutions."""

    def test_simple_integrator_problem(self):
        """Test against simple integrator: dx/dt = u, minimize ∫u² dt."""
        # Analytical solution: u(t) = -2t + 2, x(t) = -t² + 2t + x0

        # Create problem
        problem = Problem("Simple Integrator Test")

        # Define variables
        t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0)  # Free final state
        u = problem.control("u", boundary=(-10.0, 10.0))

        # Dynamics: dx/dt = u
        problem.dynamics({x: u})

        # Objective: minimize ∫u² dt
        integral_var = problem.add_integral(u**2)
        problem.minimize(integral_var)

        # Set mesh
        problem.set_mesh([4], [-1.0, 1.0])

        # Solve
        solution = solve_fixed_mesh(problem)

        # Verify solution success
        assert solution.success, f"Solver failed: {solution.message}"

        # Check analytical solution at final time
        # u(1) = -2*1 + 2 = 0, x(1) = -1 + 2 + 0 = 1
        final_state = solution.states[0][-1]
        final_control = solution.controls[0][-1]

        assert abs(final_state - 1.0) < 1e-6, f"Final state wrong: {final_state}, expected 1.0"
        assert abs(final_control - 0.0) < 1e-6, (
            f"Final control wrong: {final_control}, expected 0.0"
        )

        # Check objective value: ∫₀¹ (-2t + 2)² dt = ∫₀¹ (4t² - 8t + 4) dt = 4/3
        expected_objective = 4.0 / 3.0
        assert abs(solution.objective - expected_objective) < 1e-5, (
            f"Objective wrong: {solution.objective}, expected {expected_objective}"
        )

    def test_brachistochrone_problem(self):
        """Test against brachistochrone problem with known properties."""
        # Classic optimal control problem - should have cycloid solution

        problem = Problem("Brachistochrone")

        # Variables
        t = problem.time(initial=0.0, final=(0.1, 10.0))  # Free final time
        x = problem.state("x", initial=0.0, final=1.0)  # Horizontal position
        y = problem.state("y", initial=0.0, final=-1.0)  # Vertical position
        v = problem.state("v", initial=0.0)  # Speed
        theta = problem.control("theta", boundary=(-np.pi / 2, np.pi / 2))  # Angle

        # Dynamics (assuming g=1 for simplicity)
        problem.dynamics(
            {
                x: v * ca.cos(theta),
                y: v * ca.sin(theta),
                v: ca.sin(theta),  # dv/dt = g*sin(theta) with g=1
            }
        )

        # Minimize time
        problem.minimize(t.final)

        # Set mesh
        problem.set_mesh([6, 6], [-1.0, 0.0, 1.0])

        # Solve
        solution = solve_fixed_mesh(problem)

        # Verify solution
        assert solution.success, f"Brachistochrone solver failed: {solution.message}"

        # Check boundary conditions
        assert abs(solution.states[0][0] - 0.0) < 1e-8, "Initial x wrong"
        assert abs(solution.states[1][0] - 0.0) < 1e-8, "Initial y wrong"
        assert abs(solution.states[0][-1] - 1.0) < 1e-6, "Final x wrong"
        assert abs(solution.states[1][-1] - (-1.0)) < 1e-6, "Final y wrong"

        # Speed should be increasing (energy conservation)
        speeds = solution.states[2]
        assert np.all(np.diff(speeds) >= -1e-6), "Speed not monotonically increasing"

        # Final speed should satisfy energy conservation: v² = 2gh = 2*1*1 = 2
        final_speed = speeds[-1]
        expected_final_speed = np.sqrt(2.0)
        assert abs(final_speed - expected_final_speed) < 1e-2, (
            f"Final speed wrong: {final_speed}, expected {expected_final_speed}"
        )

    def test_constraint_satisfaction_verification(self):
        """Test that all constraints are properly satisfied in solution."""

        problem = Problem("Constraint Satisfaction Test")

        # Variables with various constraint types
        t = problem.time(initial=0.0, final=2.0)
        x = problem.state("x", initial=1.0, final=3.0, boundary=(0.0, 5.0))
        u = problem.control("u", boundary=(-1.0, 1.0))

        # Dynamics
        problem.dynamics({x: u})

        # Path constraint: x + u <= 4
        problem.subject_to(x + u <= 4.0)

        # Minimize control effort
        integral_var = problem.add_integral(u**2)
        problem.minimize(integral_var)

        # Set mesh
        problem.set_mesh([5], [-1.0, 1.0])

        # Solve
        solution = solve_fixed_mesh(problem)
        assert solution.success, f"Constraint test failed: {solution.message}"

        # Verify all constraints
        # 1. Time bounds
        assert abs(solution.initial_time - 0.0) < 1e-10, "Initial time constraint violated"
        assert abs(solution.final_time - 2.0) < 1e-10, "Final time constraint violated"

        # 2. State bounds
        x_traj = solution.states[0]
        assert abs(x_traj[0] - 1.0) < 1e-8, "Initial state constraint violated"
        assert abs(x_traj[-1] - 3.0) < 1e-8, "Final state constraint violated"
        assert np.all(x_traj >= -1e-6), "State lower bound violated"
        assert np.all(x_traj <= 5.0 + 1e-6), "State upper bound violated"

        # 3. Control bounds
        u_traj = solution.controls[0]
        assert np.all(u_traj >= -1.0 - 1e-6), "Control lower bound violated"
        assert np.all(u_traj <= 1.0 + 1e-6), "Control upper bound violated"

        # 4. Path constraint: x + u <= 4
        for i in range(len(solution.time_states)):
            x_val = solution.states[0][i]
            # Find corresponding control (controls have different time points)
            t_val = solution.time_states[i]
            u_val = np.interp(t_val, solution.time_controls, solution.controls[0])

            constraint_value = x_val + u_val
            assert constraint_value <= 4.0 + 1e-4, (
                f"Path constraint violated at t={t_val}: x+u={constraint_value} > 4.0"
            )

    def test_solver_robustness_edge_cases(self):
        """Test solver robustness with edge cases that could cause mission failures."""

        # Test Case 1: Very small time interval
        problem1 = Problem("Small Time Interval")
        t1 = problem1.time(initial=0.0, final=1e-6)
        x1 = problem1.state("x", initial=0.0, final=1e-6)
        u1 = problem1.control("u")
        problem1.dynamics({x1: u1})
        problem1.minimize(u1**2)
        problem1.set_mesh([3], [-1.0, 1.0])

        solution1 = solve_fixed_mesh(problem1)
        assert solution1.success, "Small time interval case failed"

        # Test Case 2: Large dynamic range
        problem2 = Problem("Large Dynamic Range")
        t2 = problem2.time(initial=0.0, final=1.0)
        x2 = problem2.state("x", initial=1e-8, final=1e8)
        u2 = problem2.control("u")
        problem2.dynamics({x2: u2})
        problem2.minimize(u2**2)
        problem2.set_mesh([4], [-1.0, 1.0])

        solution2 = solve_fixed_mesh(problem2)
        # Should either succeed or fail gracefully (not crash)
        if not solution2.success:
            assert (
                "numerical" in solution2.message.lower()
                or "infeasible" in solution2.message.lower()
            )

        # Test Case 3: Tight constraints
        problem3 = Problem("Tight Constraints")
        t3 = problem3.time(initial=0.0, final=1.0)
        x3 = problem3.state("x", initial=0.0, boundary=(0.0, 0.001))  # Very tight bounds
        u3 = problem3.control("u", boundary=(-0.001, 0.001))
        problem3.dynamics({x3: u3})
        problem3.minimize(x3**2 + u3**2)
        problem3.set_mesh([3], [-1.0, 1.0])

        solution3 = solve_fixed_mesh(problem3)
        # Should handle tight constraints gracefully
        if solution3.success:
            # Verify constraints are satisfied
            assert np.all(solution3.states[0] >= -1e-8), "Tight constraint violated"
            assert np.all(solution3.states[0] <= 0.001 + 1e-8), "Tight constraint violated"

    def test_solver_failure_modes(self):
        """Test that solver fails gracefully for infeasible problems."""

        # Infeasible problem: contradictory constraints
        problem = Problem("Infeasible Problem")
        t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=10.0)  # Need to go from 0 to 10
        u = problem.control("u", boundary=(0.0, 1.0))  # But max control is 1

        problem.dynamics({x: u})  # dx/dt = u, so max x(1) = 1*1 = 1 < 10
        problem.minimize(u**2)
        problem.set_mesh([4], [-1.0, 1.0])

        solution = solve_fixed_mesh(problem)

        # Should fail gracefully, not crash
        assert not solution.success, "Infeasible problem should fail"
        assert solution.message is not None, "Should provide failure message"
        assert len(solution.message) > 0, "Failure message should be informative"

        # Solution object should be safe to access
        assert solution.objective is None or np.isnan(solution.objective), (
            "Objective should be None/NaN"
        )
        assert solution.initial_time is None or np.isfinite(solution.initial_time), (
            "Time should be None or finite"
        )
