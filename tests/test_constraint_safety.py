# test_constraint_safety.py
"""
Safety-critical tests for constraint application and satisfaction.
Ensures constraints are properly enforced to prevent unsafe trajectories.
"""

import numpy as np

from trajectolab import Problem, solve_fixed_mesh


class TestConstraintSafety:
    """Test constraint application and satisfaction for mission safety."""

    def test_state_boundary_constraints_enforcement(self):
        """Test that state boundary constraints are strictly enforced."""

        problem = Problem("State Boundary Test")

        t = problem.time(initial=0.0, final=1.0)
        # Critical: altitude must stay above 1000m for safety
        altitude = problem.state("alt", initial=2000.0, boundary=(1000.0, 5000.0))
        # Velocity
        velocity = problem.state("vel", initial=0.0, boundary=(-50.0, 50.0))
        # Control: thrust
        thrust = problem.control("thrust", boundary=(-10.0, 10.0))

        # Simple dynamics
        problem.dynamics(
            {
                altitude: velocity,
                velocity: thrust - 9.81,  # Gravity
            }
        )

        # Minimize fuel (thrust squared)
        problem.minimize(problem.add_integral(thrust**2))

        problem.set_mesh([5], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        assert solution.success, f"Boundary constraint test failed: {solution.message}"

        # Critical safety check: altitude never below 1000
        altitudes = solution.states[0]  # altitude is first state
        min_altitude = np.min(altitudes)
        assert min_altitude >= 1000.0 - 1e-4, (
            f"SAFETY VIOLATION: Altitude dropped to {min_altitude}, below safe limit 1000"
        )

        # Check altitude never above 5000
        max_altitude = np.max(altitudes)
        assert max_altitude <= 5000.0 + 1e-4, (
            f"Altitude exceeded upper limit: {max_altitude} > 5000"
        )

        # Check velocity bounds
        velocities = solution.states[1]
        assert np.all(velocities >= -50.0 - 1e-4), "Velocity below lower bound"
        assert np.all(velocities <= 50.0 + 1e-4), "Velocity above upper bound"

    def test_control_boundary_constraints_enforcement(self):
        """Test that control boundary constraints prevent actuator saturation."""

        problem = Problem("Control Boundary Test")

        t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=5.0)
        # Critical: actuator limited to ±2.0 units
        u = problem.control("u", boundary=(-2.0, 2.0))

        problem.dynamics({x: u})
        problem.minimize(problem.add_integral(u**2))

        problem.set_mesh([4], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        assert solution.success, f"Control boundary test failed: {solution.message}"

        # Critical safety check: control never exceeds actuator limits
        controls = solution.controls[0]
        min_control = np.min(controls)
        max_control = np.max(controls)

        assert min_control >= -2.0 - 1e-4, (
            f"SAFETY VIOLATION: Control below actuator limit: {min_control} < -2.0"
        )
        assert max_control <= 2.0 + 1e-4, (
            f"SAFETY VIOLATION: Control above actuator limit: {max_control} > 2.0"
        )

    def test_path_constraints_safety_enforcement(self):
        """Test that path constraints prevent unsafe conditions throughout trajectory."""

        problem = Problem("Path Safety Constraints")

        t = problem.time(initial=0.0, final=2.0)
        position = problem.state("pos", initial=0.0)
        velocity = problem.state("vel", initial=1.0)
        thrust = problem.control("thrust", boundary=(-5.0, 5.0))

        problem.dynamics({position: velocity, velocity: thrust})

        # Critical path constraint: maintain safe separation
        # pos + vel <= 10 (combined position and velocity must not exceed limit)
        problem.subject_to(position + velocity <= 10.0)

        # Additional safety constraint: velocity must not be too negative
        problem.subject_to(velocity >= -2.0)

        problem.minimize(problem.add_integral(thrust**2))

        problem.set_mesh([6], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        assert solution.success, f"Path constraint test failed: {solution.message}"

        # Verify path constraints at all solution points
        positions = solution.states[0]
        velocities = solution.states[1]

        # Check safety constraint: pos + vel <= 10
        for i, (pos, vel) in enumerate(zip(positions, velocities, strict=False)):
            safety_margin = pos + vel
            assert safety_margin <= 10.0 + 1e-4, (
                f"SAFETY VIOLATION at point {i}: pos + vel = {safety_margin} > 10.0"
            )

        # Check velocity lower bound
        min_velocity = np.min(velocities)
        assert min_velocity >= -2.0 - 1e-4, (
            f"SAFETY VIOLATION: Velocity too negative: {min_velocity} < -2.0"
        )

    def test_event_constraints_boundary_conditions(self):
        """Test that event constraints properly enforce boundary conditions."""

        problem = Problem("Event Constraints Test")

        t = problem.time(initial=0.0, final=1.0)
        # Critical: must start at specific position and end at target
        x = problem.state("x", initial=1.0, final=10.0)  # Event constraints
        y = problem.state("y", initial=2.0, final=5.0)  # Event constraints
        u = problem.control("u", boundary=(-5.0, 5.0))
        v = problem.control("v", boundary=(-5.0, 5.0))

        problem.dynamics({x: u, y: v})

        problem.minimize(problem.add_integral(u**2 + v**2))

        problem.set_mesh([4], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        assert solution.success, f"Event constraint test failed: {solution.message}"

        # Critical: verify exact boundary conditions
        x_traj = solution.states[0]
        y_traj = solution.states[1]

        # Initial conditions
        assert abs(x_traj[0] - 1.0) < 1e-8, (
            f"CRITICAL: Initial x condition violated: {x_traj[0]} ≠ 1.0"
        )
        assert abs(y_traj[0] - 2.0) < 1e-8, (
            f"CRITICAL: Initial y condition violated: {y_traj[0]} ≠ 2.0"
        )

        # Final conditions
        assert abs(x_traj[-1] - 10.0) < 1e-6, (
            f"CRITICAL: Final x target missed: {x_traj[-1]} ≠ 10.0"
        )
        assert abs(y_traj[-1] - 5.0) < 1e-6, f"CRITICAL: Final y target missed: {y_traj[-1]} ≠ 5.0"

    def test_constraint_violation_detection(self):
        """Test that constraint violations are properly detected and reported."""

        # Create an infeasible problem with contradictory constraints
        problem = Problem("Constraint Violation Test")

        t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=0.0, final=10.0)  # Need to reach 10
        u = problem.control("u", boundary=(0.0, 1.0))  # Max control is 1

        problem.dynamics({x: u})  # dx/dt = u, so max possible x(1) = 1

        # Add impossible path constraint
        problem.subject_to(x <= 5.0)  # Can't exceed 5, but final condition is 10

        problem.minimize(problem.add_integral(u**2))

        problem.set_mesh([4], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        # Should detect infeasibility
        assert not solution.success, "Infeasible problem should be detected"

        # Should provide informative error message
        assert "infeasible" in solution.message.lower() or "constraint" in solution.message.lower()

    def test_numerical_constraint_stability(self):
        """Test constraint handling numerical stability."""

        problem = Problem("Numerical Stability Test")

        t = problem.time(initial=0.0, final=1.0)
        x = problem.state("x", initial=1e-8, final=1e8)  # Large dynamic range
        u = problem.control("u", boundary=(-1e6, 1e6))  # Large control bounds

        problem.dynamics({x: u})

        # Constraint with potential numerical issues
        problem.subject_to(x**2 <= 1e16)  # x^2 constraint

        problem.minimize(problem.add_integral(u**2))

        problem.set_mesh([3], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        if solution.success:
            # If solved successfully, verify no NaN/Inf in solution
            x_vals = solution.states[0]
            u_vals = solution.controls[0]

            assert np.all(np.isfinite(x_vals)), "State contains NaN/Inf"
            assert np.all(np.isfinite(u_vals)), "Control contains NaN/Inf"

            # Verify constraint satisfaction
            assert np.all(x_vals**2 <= 1e16 + 1e10), "x^2 constraint violated"
        else:
            # If failed, should fail gracefully without crash
            assert solution.message is not None, "Should provide failure message"

    def test_constraint_priority_and_feasibility(self):
        """Test constraint priority when multiple constraints compete."""

        problem = Problem("Constraint Priority Test")

        t = problem.time(initial=0.0, final=1.0)
        # Safety-critical state with tight bounds
        safety_margin = problem.state("safety", initial=5.0, boundary=(3.0, 7.0))
        performance = problem.state("perf", initial=0.0, final=10.0)

        control = problem.control("ctrl", boundary=(-2.0, 2.0))

        problem.dynamics(
            {
                safety_margin: -0.5 + 0.1 * control,  # Safety margin decreases over time
                performance: control,
            }
        )

        # Competing constraints
        problem.subject_to(safety_margin >= 4.0)  # Safety constraint (more important)
        problem.subject_to(performance >= 8.0)  # Performance constraint

        problem.minimize(problem.add_integral(control**2))

        problem.set_mesh([5], [-1.0, 1.0])
        solution = solve_fixed_mesh(problem)

        if solution.success:
            safety_vals = solution.states[0]
            perf_vals = solution.states[1]

            # Safety constraint should be satisfied (higher priority)
            min_safety = np.min(safety_vals)
            assert min_safety >= 4.0 - 1e-4, (
                f"CRITICAL: Safety constraint violated: min safety = {min_safety}"
            )

            # Performance may or may not be achieved (lower priority)
            print(f"Final performance: {perf_vals[-1]}")

        # System should handle competing constraints gracefully
        assert solution.success or "infeasible" in solution.message.lower()

    def test_time_constraint_enforcement(self):
        """Test that time bounds are properly enforced."""

        # Test fixed time constraints
        problem1 = Problem("Fixed Time Test")
        t1 = problem1.time(initial=2.0, final=5.0)  # Fixed times
        x1 = problem1.state("x", initial=0.0, final=1.0)
        u1 = problem1.control("u")
        problem1.dynamics({x1: u1})
        problem1.minimize(problem1.add_integral(u1**2))

        problem1.set_mesh([3], [-1.0, 1.0])
        solution1 = solve_fixed_mesh(problem1)

        assert solution1.success, "Fixed time problem failed"
        assert abs(solution1.initial_time - 2.0) < 1e-10, "Initial time constraint violated"
        assert abs(solution1.final_time - 5.0) < 1e-10, "Final time constraint violated"

        # Test time bound constraints
        problem2 = Problem("Time Bounds Test")
        t2 = problem2.time(initial=(1.0, 3.0), final=(4.0, 8.0))  # Time ranges
        x2 = problem2.state("x", initial=0.0, final=1.0)
        u2 = problem2.control("u")
        problem2.dynamics({x2: u2})
        problem2.minimize(
            t2.final - t2.initial + problem2.add_integral(u2**2)
        )  # Minimize time + control

        problem2.set_mesh([3], [-1.0, 1.0])
        solution2 = solve_fixed_mesh(problem2)

        assert solution2.success, "Time bounds problem failed"

        # Verify time bounds
        assert 1.0 - 1e-6 <= solution2.initial_time <= 3.0 + 1e-6, (
            f"Initial time out of bounds: {solution2.initial_time} ∉ [1.0, 3.0]"
        )
        assert 4.0 - 1e-6 <= solution2.final_time <= 8.0 + 1e-6, (
            f"Final time out of bounds: {solution2.final_time} ∉ [4.0, 8.0]"
        )
        assert solution2.final_time > solution2.initial_time + 1e-6, (
            "Final time must be greater than initial time"
        )
