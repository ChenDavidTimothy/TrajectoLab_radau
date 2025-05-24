"""
RPM Benchmark Test for TrajectoLab - Following Grug's Pragmatic Testing Philosophy

This integration test validates the entire TrajectoLab pipeline against a known analytical
solution from the literature. It follows Grug's philosophy by:
1. Testing the real system components working together (no excessive mocking)
2. Focusing on integration testing as the "sweet spot"
3. Providing regression testing against mathematical ground truth
4. Being practical and valuable for safety-critical applications

Test Problem from Literature:
    min J = -y(tf) s.t. { ẏ = -y + yu - u²
                        { y(0) = 1
    where tf = 5

Analytical Solution:
    y*(t) = 4/(1 + 3*exp(t))
    u*(t) = y*(t)/2
    λy*(t) = -exp(2*ln(1 + 3*exp(t)) - t)/(exp(-5) + 6 + 9*exp(5))
"""

import numpy as np
import pytest

import trajectolab as tl


class AnalyticalSolution:
    """Analytical solution for the benchmark optimal control problem."""

    def __init__(self, tf: float = 5.0):
        self.tf = tf

    def state(self, t: np.ndarray) -> np.ndarray:
        """Analytical state solution y*(t) = 4/(1 + 3*exp(t))"""
        return 4.0 / (1.0 + 3.0 * np.exp(t))

    def control(self, t: np.ndarray) -> np.ndarray:
        """Analytical control solution u*(t) = y*(t)/2"""
        return self.state(t) / 2.0

    def costate(self, t: np.ndarray) -> np.ndarray:
        """Analytical costate solution λy*(t)"""
        numerator = -np.exp(2 * np.log(1 + 3 * np.exp(t)) - t)
        denominator = np.exp(-5) + 6 + 9 * np.exp(5)
        return numerator / denominator

    def objective_value(self) -> float:
        """Analytical objective value J* = -y*(tf)"""
        return -self.state(np.array([self.tf]))[0]


class BenchmarkProblemBuilder:
    """Builder for the benchmark optimal control problem."""

    @staticmethod
    def create_problem() -> tl.Problem:
        """Create the benchmark problem using TrajectoLab API."""
        problem = tl.Problem("RPM Benchmark - Literature Problem")

        # Time: fixed initial, fixed final
        t = problem.time(initial=0.0, final=5.0)

        # States: y with initial condition y(0) = 1
        y = problem.state("y", initial=1.0)

        # Controls: u (unconstrained for this problem)
        u = problem.control("u")

        # Dynamics: ẏ = -y + yu - u²
        problem.dynamics({y: -y + y * u - u**2})

        # Objective: minimize J = -y(tf)
        problem.minimize(-y)  # Note: we minimize -y, which means maximize y

        return problem


class ErrorAnalyzer:
    """Analyzes numerical errors against analytical solution."""

    @staticmethod
    def compute_l_infinity_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Compute L∞ error: max|numerical - analytical|"""
        return np.max(np.abs(numerical - analytical))

    @staticmethod
    def compute_relative_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Compute relative L∞ error: max|numerical - analytical|/max|analytical|"""
        abs_error = np.max(np.abs(numerical - analytical))
        max_analytical = np.max(np.abs(analytical))
        return abs_error / max_analytical if max_analytical > 1e-12 else abs_error

    @staticmethod
    def compute_log10_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Compute log10 of relative error (as shown in paper plots)"""
        rel_error = ErrorAnalyzer.compute_relative_error(numerical, analytical)
        return np.log10(rel_error) if rel_error > 1e-16 else -16.0


@pytest.fixture
def analytical_solution():
    """Fixture providing the analytical solution."""
    return AnalyticalSolution()


@pytest.fixture
def benchmark_problem():
    """Fixture providing the benchmark problem."""
    return BenchmarkProblemBuilder.create_problem()


class TestRPMBenchmark:
    """
    RPM Benchmark Test Suite

    Following Grug's Philosophy:
    - Integration tests that verify the entire pipeline
    - Tests against known mathematical ground truth
    - Practical error tolerances for safety-critical applications
    - Clear failure diagnostics for debugging
    """

    def test_problem_setup_correctness(self, benchmark_problem):
        """
        Test that the problem is correctly defined.

        This is a quick sanity check before expensive solving.
        Follows Grug's principle of simple, testable components.
        """
        # Verify problem has correct structure
        num_states, num_controls = benchmark_problem.get_variable_counts()
        assert num_states == 1, f"Expected 1 state, got {num_states}"
        assert num_controls == 1, f"Expected 1 control, got {num_controls}"

        # Verify time bounds are set correctly
        t0_bounds, tf_bounds = benchmark_problem._t0_bounds, benchmark_problem._tf_bounds
        assert np.isclose(t0_bounds[0], 0.0)
        assert np.isclose(t0_bounds[1], 0.0)
        assert np.isclose(tf_bounds[0], 5.0)
        assert np.isclose(tf_bounds[1], 5.0)

        # Verify dynamics and objective are defined
        assert benchmark_problem._dynamics_expressions is not None
        assert benchmark_problem._objective_expression is not None

    @pytest.mark.parametrize("polynomial_degree", [10, 15, 20, 25, 30])
    def test_rpm_accuracy_vs_polynomial_degree(
        self, benchmark_problem, analytical_solution, polynomial_degree
    ):
        """
        Test RPM accuracy for different polynomial degrees.

        This is the core integration test - validates the entire TrajectoLab
        pipeline against known analytical solution. Follows Grug's preference
        for tests that verify real system behavior.
        """
        # Set up mesh (single interval with specified polynomial degree)
        mesh_points = np.array([-1.0, 1.0])
        benchmark_problem.set_mesh([polynomial_degree], mesh_points)

        # Solve the problem
        solution = tl.solve_fixed_mesh(
            benchmark_problem,
            nlp_options={
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
                "ipopt.tol": 1e-12,  # High precision for accuracy testing
                "ipopt.max_iter": 1000,  # Allow enough iterations for convergence
            },
        )

        # Verify solution succeeded
        assert solution.success, f"Solver failed: {solution.message}"

        # FIXED: Extract numerical solution with correct time points for each variable
        t_states, y_num = solution.get_trajectory("y")  # State time points
        t_controls, u_num = solution.get_trajectory("u")  # Control time points

        # FIXED: Compute analytical solution at the correct time points for each variable
        y_analytical = analytical_solution.state(t_states)  # Use state time points
        u_analytical = analytical_solution.control(t_controls)  # Use control time points

        # Compute errors
        state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical)
        control_error = ErrorAnalyzer.compute_relative_error(u_num, u_analytical)

        if polynomial_degree == 10:
            expected_state_accuracy = 6e-5
            expected_control_accuracy = 6e-5
        if polynomial_degree == 15:
            expected_state_accuracy = 1e-8
            expected_control_accuracy = 1e-8

        if polynomial_degree >= 20:
            expected_state_accuracy = 1e-11
            expected_control_accuracy = 1e-11

        if polynomial_degree >= 25:
            expected_state_accuracy = 6e-15
            expected_control_accuracy = 6e-15

        if polynomial_degree >= 30:
            expected_state_accuracy = 6e-16
            expected_control_accuracy = 6e-16

        # Assert accuracy requirements
        assert state_error < expected_state_accuracy, (
            f"State error {state_error:.2e} exceeds tolerance {expected_state_accuracy:.2e} "
            f"for polynomial degree {polynomial_degree}"
        )

        assert control_error < expected_control_accuracy, (
            f"Control error {control_error:.2e} exceeds tolerance {expected_control_accuracy:.2e} "
            f"for polynomial degree {polynomial_degree}"
        )

        # Test objective value accuracy
        obj_analytical = analytical_solution.objective_value()
        obj_error = abs(solution.objective - obj_analytical) / abs(obj_analytical)
        assert obj_error < 1e-8, (
            f"Objective error {obj_error:.2e} too large. "
            f"Numerical: {solution.objective:.8f}, Analytical: {obj_analytical:.8f}"
        )

    def test_rpm_convergence_study(self, benchmark_problem, analytical_solution):
        """
        Test that RPM shows spectral convergence (exponential error reduction).

        This validates the fundamental mathematical property of pseudospectral methods.
        Critical for safety-critical applications where convergence guarantees matter.
        """
        polynomial_degrees = [10, 15, 20, 25, 30]
        state_errors = []

        for degree in polynomial_degrees:
            # Reset problem for each degree
            problem = BenchmarkProblemBuilder.create_problem()
            problem.set_mesh([degree], np.array([-1.0, 1.0]))

            solution = tl.solve_fixed_mesh(
                problem,
                nlp_options={
                    "ipopt.print_level": 0,
                    "ipopt.sb": "yes",
                    "print_time": 0,
                    "ipopt.tol": 1e-12,
                },
            )

            assert solution.success, f"Solver failed for degree {degree}: {solution.message}"

            # FIXED: Use correct time points for state trajectory
            t_states, y_num = solution.get_trajectory("y")
            y_analytical = analytical_solution.state(t_states)
            error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical)
            state_errors.append(error)

        # Verify spectral convergence: each doubling of degree should dramatically reduce error
        for i in range(1, len(state_errors)):
            convergence_ratio = state_errors[i - 1] / state_errors[i]
            assert convergence_ratio > 5.0, (
                f"Insufficient convergence between degrees {polynomial_degrees[i - 1]} and {polynomial_degrees[i]}. "
                f"Error ratio: {convergence_ratio:.2f}, errors: {state_errors[i - 1]:.2e} -> {state_errors[i]:.2e}"
            )

    def test_multi_interval_accuracy(self, analytical_solution):
        """
        Test RPM accuracy with multiple mesh intervals.

        Validates that mesh refinement works correctly - important for
        problems with varying solution characteristics.
        """
        problem = BenchmarkProblemBuilder.create_problem()

        # Use 3 intervals with moderate polynomial degrees
        mesh_points = np.array([-1.0, -0.2, 0.4, 1.0])
        polynomial_degrees = [6, 6, 6]
        problem.set_mesh(polynomial_degrees, mesh_points)

        solution = tl.solve_fixed_mesh(
            problem,
            nlp_options={
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
            },
        )

        assert solution.success, f"Multi-interval solve failed: {solution.message}"

        # FIXED: Use correct time points for state trajectory
        t_states, y_num = solution.get_trajectory("y")
        y_analytical = analytical_solution.state(t_states)

        state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical)
        assert state_error < 3e-6, f"Multi-interval state error {state_error:.2e} too large"

    def test_adaptive_solver_accuracy(self, analytical_solution):
        """
        Test adaptive mesh refinement against analytical solution.

        This tests the most sophisticated part of the system - adaptive
        refinement should automatically achieve high accuracy.
        """
        problem = BenchmarkProblemBuilder.create_problem()

        # Start with coarse mesh
        problem.set_mesh([4, 4], np.array([-1.0, 0.0, 1.0]))

        solution = tl.solve_adaptive(
            problem,
            error_tolerance=1e-8,
            max_iterations=10,
            min_polynomial_degree=4,
            max_polynomial_degree=12,
            nlp_options={
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
            },
        )

        assert solution.success, f"Adaptive solve failed: {solution.message}"

        # FIXED: Use correct time points for state trajectory
        t_states, y_num = solution.get_trajectory("y")
        y_analytical = analytical_solution.state(t_states)

        state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical)
        assert state_error < 1e-7, (
            f"Adaptive solver failed to achieve target accuracy. "
            f"Error: {state_error:.2e}, Target: 1e-7"
        )

    def test_regression_against_reference_values(self, benchmark_problem, analytical_solution):
        """
        Regression test with fixed reference values.

        Following Grug's philosophy: when we know what the right answer should be,
        we test against it to prevent regressions. This test will catch any
        changes that affect numerical accuracy.
        """
        # Use specific mesh configuration
        benchmark_problem.set_mesh([8], np.array([-1.0, 1.0]))

        solution = tl.solve_fixed_mesh(
            benchmark_problem,
            nlp_options={
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
                "ipopt.tol": 1e-12,
            },
        )

        assert solution.success, f"Regression test solve failed: {solution.message}"

        # Reference values (these should be updated if algorithm changes intentionally)
        REFERENCE_OBJECTIVE = -0.8  # Approximately -y(5) for analytical solution
        REFERENCE_FINAL_STATE = 0.8  # Approximately y(5) for analytical solution

        # Test objective value
        obj_error = abs(solution.objective - REFERENCE_OBJECTIVE)
        assert obj_error < 1e-6, (
            f"Objective regression detected. "
            f"Current: {solution.objective:.8f}, Reference: {REFERENCE_OBJECTIVE:.8f}, "
            f"Error: {obj_error:.2e}"
        )

        # FIXED: Use correct time points for state trajectory
        t_states, y_num = solution.get_trajectory("y")
        final_state_error = abs(y_num[-1] - REFERENCE_FINAL_STATE)
        assert final_state_error < 1e-6, (
            f"Final state regression detected. "
            f"Current: {y_num[-1]:.8f}, Reference: {REFERENCE_FINAL_STATE:.8f}, "
            f"Error: {final_state_error:.2e}"
        )


def test_benchmark_suite_diagnostic():
    """
    Diagnostic test that runs the full benchmark and reports detailed results.

    This isn't a pass/fail test, but generates detailed diagnostic information
    similar to what's shown in the paper. Useful for performance analysis
    and debugging.
    """
    print("\n" + "=" * 60)
    print("RPM BENCHMARK DIAGNOSTIC REPORT")
    print("=" * 60)

    analytical = AnalyticalSolution()
    degrees = [4, 6, 8, 10, 12]

    print(f"{'Degree':>6} {'State Error':>12} {'Control Error':>14} {'Obj Error':>12}")
    print("-" * 50)

    for degree in degrees:
        try:
            problem = BenchmarkProblemBuilder.create_problem()
            problem.set_mesh([degree], np.array([-1.0, 1.0]))

            solution = tl.solve_fixed_mesh(
                problem, nlp_options={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}
            )

            if solution.success:
                # FIXED: Use correct time points for each variable type
                t_states, y_num = solution.get_trajectory("y")
                t_controls, u_num = solution.get_trajectory("u")

                y_analytical = analytical.state(t_states)
                u_analytical = analytical.control(t_controls)
                obj_analytical = analytical.objective_value()

                state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical)
                control_error = ErrorAnalyzer.compute_relative_error(u_num, u_analytical)
                obj_error = abs(solution.objective - obj_analytical) / abs(obj_analytical)

                print(f"{degree:6d} {state_error:12.2e} {control_error:14.2e} {obj_error:12.2e}")
            else:
                print(f"{degree:6d} {'FAILED':>12} {'FAILED':>14} {'FAILED':>12}")

        except Exception as e:
            print(f"{degree:6d} {'ERROR':>12} {'ERROR':>14} {'ERROR':>12} - {e!s}")

    print("=" * 60)


if __name__ == "__main__":
    # Run diagnostic when executed directly
    test_benchmark_suite_diagnostic()
