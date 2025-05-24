"""
Comprehensive diagnostic to verify TrajectoLab solver accuracy.
Run this to confirm your solver is working correctly.
"""

import numpy as np

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
        problem.time(initial=0.0, final=5.0)

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


def run_comprehensive_diagnostic():
    """
    Run comprehensive diagnostic tests to verify solver accuracy.
    This will help confirm the solver is working correctly.
    """
    print("=" * 80)
    print("TRAJECTOLAB SOLVER DIAGNOSTIC")
    print("=" * 80)

    analytical = AnalyticalSolution()

    # Test 1: Verify analytical solution calculations
    print("\n1. ANALYTICAL SOLUTION VERIFICATION")
    print("-" * 40)

    t_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_analytical = analytical.state(t_test)
    u_analytical = analytical.control(t_test)
    obj_analytical = analytical.objective_value()

    print(f"Time points: {t_test}")
    print(f"Analytical states: {y_analytical}")
    print(f"Analytical controls: {u_analytical}")
    print(f"Analytical objective: {obj_analytical:.8f}")
    print(f"Final state y*(5): {y_analytical[-1]:.8f}")

    # Test 2: Single degree accuracy test
    print("\n2. SINGLE POLYNOMIAL DEGREE TEST")
    print("-" * 40)

    problem = BenchmarkProblemBuilder.create_problem()
    problem.set_mesh([15], np.array([-1.0, 1.0]))

    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.tol": 1e-12,
        },
    )

    if solution.success:
        t_states, y_num = solution.get_trajectory("y")
        t_controls, u_num = solution.get_trajectory("u")

        # Compute errors at the numerical solution points
        y_analytical_at_solver_points = analytical.state(t_states)
        u_analytical_at_solver_points = analytical.control(t_controls)

        state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical_at_solver_points)
        control_error = ErrorAnalyzer.compute_relative_error(u_num, u_analytical_at_solver_points)
        obj_error = abs(solution.objective - obj_analytical) / abs(obj_analytical)

        print(f"Solver objective: {solution.objective:.8f}")
        print(f"Analytical objective: {obj_analytical:.8f}")
        print(f"Objective relative error: {obj_error:.2e}")
        print(f"State relative error: {state_error:.2e}")
        print(f"Control relative error: {control_error:.2e}")
        print(f"Final numerical state: {y_num[-1]:.8f}")
        print(f"Final analytical state: {y_analytical[-1]:.8f}")

        # Accuracy assessment
        if obj_error < 1e-8 and state_error < 1e-8 and control_error < 1e-8:
            print("✅ EXCELLENT: Solver achieving high accuracy")
        elif obj_error < 1e-6 and state_error < 1e-6 and control_error < 1e-6:
            print("✅ GOOD: Solver achieving acceptable accuracy")
        else:
            print("❌ WARNING: Solver accuracy may be insufficient")
    else:
        print(f"❌ FAILED: Solver failed - {solution.message}")

    # Test 3: Convergence study
    print("\n3. CONVERGENCE STUDY")
    print("-" * 40)

    degrees = [10, 15, 20, 25, 30]
    print(f"{'Degree':<8}{'Obj Error':<12}{'State Error':<14}{'Control Error':<14}")
    print("-" * 48)

    for degree in degrees:
        try:
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

            if solution.success:
                t_states, y_num = solution.get_trajectory("y")
                t_controls, u_num = solution.get_trajectory("u")

                y_analytical_points = analytical.state(t_states)
                u_analytical_points = analytical.control(t_controls)

                state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical_points)
                control_error = ErrorAnalyzer.compute_relative_error(u_num, u_analytical_points)
                obj_error = abs(solution.objective - obj_analytical) / abs(obj_analytical)

                print(f"{degree:<8}{obj_error:<12.2e}{state_error:<14.2e}{control_error:<14.2e}")
            else:
                print(f"{degree:<8}{'FAILED':<12}{'FAILED':<14}{'FAILED':<14}")

        except Exception:
            print(f"{degree:<8}{'ERROR':<12}{'ERROR':<14}{'ERROR':<14}")

    # Test 4: Multi-interval test
    print("\n4. MULTI-INTERVAL TEST")
    print("-" * 40)

    problem = BenchmarkProblemBuilder.create_problem()
    problem.set_mesh([8, 8, 8], np.array([-1.0, -0.3, 0.3, 1.0]))

    solution = tl.solve_fixed_mesh(
        problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        },
    )

    if solution.success:
        t_states, y_num = solution.get_trajectory("y")
        y_analytical_points = analytical.state(t_states)

        state_error = ErrorAnalyzer.compute_relative_error(y_num, y_analytical_points)
        obj_error = abs(solution.objective - obj_analytical) / abs(obj_analytical)

        print(f"Multi-interval objective error: {obj_error:.2e}")
        print(f"Multi-interval state error: {state_error:.2e}")

        if obj_error < 1e-5 and state_error < 1e-5:
            print("✅ Multi-interval test PASSED")
        else:
            print("❌ Multi-interval test shows higher errors")
    else:
        print(f"❌ Multi-interval test FAILED: {solution.message}")

    # Test 5: Correct reference values for regression tests
    print("\n5. CORRECT REFERENCE VALUES")
    print("-" * 40)
    print("For your regression tests, use these CORRECT reference values:")
    print(f"REFERENCE_OBJECTIVE = {obj_analytical:.8f}")
    print(f"REFERENCE_FINAL_STATE = {y_analytical[-1]:.8f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_diagnostic()
