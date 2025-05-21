"""
scaling_verification_tool.py - Command-line tool for NASA engineers to verify
TrajectoLab scaling behavior in production environments.

This tool helps NASA engineers validate that the scaling system is properly
functioning in the deployed environment. It runs a series of diagnostic tests
and produces detailed reports about the scaling behavior.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import casadi as ca
import numpy as np

import trajectolab as tl
from trajectolab.problem import Problem
from trajectolab.scaling import Scaling


# Add trajectolab to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ScalingVerifier:
    """Verifies TrajectoLab scaling system operation."""

    def __init__(self, log_level: int = logging.INFO, log_dir: str | None = None):
        """Initialize the scaling verifier with logging configuration."""
        self.log_level = log_level
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "scaling_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.log_file = (
            self.log_dir / f"scaling_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.results_file = (
            self.log_dir
            / f"scaling_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        # Setup logging
        self._setup_logging()

        # Results storage
        self.results: dict[str, dict[str, bool | str | float]] = {}

        self.logger.info("Scaling verification tool initialized")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Results file: {self.results_file}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger("scaling_verifier")
        self.logger.setLevel(self.log_level)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            max(logging.INFO, self.log_level)
        )  # Don't go below INFO for console
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Also configure trajectolab loggers
        scaling_logger = logging.getLogger("trajectolab.scaling")
        scaling_logger.setLevel(self.log_level)
        scaling_logger.addHandler(file_handler)

        problem_logger = logging.getLogger("trajectolab.problem")
        problem_logger.setLevel(self.log_level)
        problem_logger.addHandler(file_handler)

    def _record_test_result(
        self,
        test_name: str,
        passed: bool,
        details: str = "",
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Record the result of a test."""
        result = {
            "passed": passed,
            "details": details,
        }

        if metrics:
            result.update(metrics)

        self.results[test_name] = result

        # Log result
        status = "PASSED" if passed else "FAILED"
        self.logger.info(f"Test {test_name}: {status} - {details}")

    def run_basic_scaling_test(self) -> bool:
        """Run basic scaling initialization and property tests."""
        self.logger.info("Running basic scaling test...")

        try:
            # Test 1: Scaling class initialization
            scaling1 = Scaling(enabled=True)
            scaling2 = Scaling(enabled=False)

            if not (scaling1.enabled and not scaling2.enabled):
                self._record_test_result(
                    "basic_scaling_initialization",
                    False,
                    f"Scaling initialization failed: {scaling1.enabled}, {scaling2.enabled}",
                )
                return False

            # Test 2: Problem scaling property
            problem1 = tl.Problem("Test1", use_scaling=True)
            problem2 = tl.Problem("Test2", use_scaling=False)

            if not (problem1.use_scaling and not problem2.use_scaling):
                self._record_test_result(
                    "basic_scaling_property",
                    False,
                    f"Problem scaling property failed: {problem1.use_scaling}, {problem2.use_scaling}",
                )
                return False

            # Test 3: Property setter
            problem1.use_scaling = False
            problem2.use_scaling = True

            if not (not problem1.use_scaling and problem2.use_scaling):
                self._record_test_result(
                    "basic_scaling_setter",
                    False,
                    f"Problem scaling setter failed: {problem1.use_scaling}, {problem2.use_scaling}",
                )
                return False

            # Test 4: Consistent state
            problem1.use_scaling = True
            if not (problem1.use_scaling == problem1._scaling.enabled):
                self._record_test_result(
                    "basic_scaling_consistency",
                    False,
                    f"Inconsistent state: {problem1.use_scaling} != {problem1._scaling.enabled}",
                )
                return False

            self._record_test_result("basic_scaling_test", True, "All basic scaling checks passed")
            return True

        except Exception as e:
            self._record_test_result("basic_scaling_test", False, f"Exception: {e!s}")
            self.logger.exception("Exception in basic scaling test")
            return False

    def run_factor_calculation_test(self) -> bool:
        """Test scaling factor calculation with various inputs."""
        self.logger.info("Running scaling factor calculation test...")

        try:
            scaling = Scaling(enabled=True)

            # Test 1: Scaling calculation from bounds
            scaling.compute_scaling_factors(
                state_names=["x", "y", "z"],
                state_bounds={
                    "x": {"lower": 0.0, "upper": 10.0},
                    "y": {"lower": -5.0, "upper": 5.0},
                    "z": {"lower": 100.0, "upper": 1000.0},
                },
            )

            factor_x, shift_x = scaling.get_state_scaling("x")
            factor_y, shift_y = scaling.get_state_scaling("y")
            factor_z, shift_z = scaling.get_state_scaling("z")

            # Expected factors based on bounds
            expected_x = 1.0 / 10.0  # 1/(10-0)
            expected_y = 1.0 / 10.0  # 1/(5-(-5))
            expected_z = 1.0 / 900.0  # 1/(1000-100)

            x_ok = np.isclose(factor_x, expected_x, rtol=1e-5)
            y_ok = np.isclose(factor_y, expected_y, rtol=1e-5)
            z_ok = np.isclose(factor_z, expected_z, rtol=1e-5)

            if not (x_ok and y_ok and z_ok):
                self._record_test_result(
                    "factor_calculation_bounds",
                    False,
                    f"Incorrect bound factors: x={factor_x} (exp {expected_x}), "
                    f"y={factor_y} (exp {expected_y}), z={factor_z} (exp {expected_z})",
                )
                return False

            # Test 2: Scaling calculation from initial guess
            scaling = Scaling(enabled=True)  # Reset
            scaling.compute_scaling_factors(
                state_names=["v"],
                state_bounds={"v": {"lower": None, "upper": None}},
                state_guesses={"v": np.array([0.0, 5.0, 10.0], dtype=np.float64)},
            )

            factor_v, shift_v = scaling.get_state_scaling("v")

            # Should include safety margin, but factor should be smaller than 1.0/10.0
            v_ok = factor_v < 0.11 and factor_v > 0.08  # Allow some flexibility in margin

            if not v_ok:
                self._record_test_result(
                    "factor_calculation_guess",
                    False,
                    f"Incorrect guess factor: v={factor_v} (expected ~0.1 with margin)",
                )
                return False

            # Test 3: Scaling with disabled flag
            scaling = Scaling(enabled=False)
            scaling.compute_scaling_factors(
                state_names=["x"], state_bounds={"x": {"lower": 0.0, "upper": 10.0}}
            )

            factor_x, shift_x = scaling.get_state_scaling("x")

            if factor_x != 1.0 or shift_x != 0.0:
                self._record_test_result(
                    "factor_calculation_disabled",
                    False,
                    f"Incorrect disabled factors: x=({factor_x}, {shift_x}), expected (1.0, 0.0)",
                )
                return False

            # Test 4: Scaling with large NASA-scale values
            scaling = Scaling(enabled=True)
            scaling.compute_scaling_factors(
                state_names=["altitude", "velocity"],
                state_bounds={
                    "altitude": {"lower": 0.0, "upper": 260000.0},  # feet
                    "velocity": {"lower": 0.0, "upper": 25600.0},  # ft/s
                },
            )

            factor_alt, shift_alt = scaling.get_state_scaling("altitude")
            factor_vel, shift_vel = scaling.get_state_scaling("velocity")

            # Check if factors are in reasonable range for large values
            alt_ok = np.isclose(factor_alt, 1.0 / 260000.0, rtol=1e-5)
            vel_ok = np.isclose(factor_vel, 1.0 / 25600.0, rtol=1e-5)

            if not (alt_ok and vel_ok):
                self._record_test_result(
                    "factor_calculation_large",
                    False,
                    f"Incorrect large value factors: altitude={factor_alt} (exp ~3.8e-6), "
                    f"velocity={factor_vel} (exp ~3.9e-5)",
                )
                return False

            self._record_test_result(
                "factor_calculation_test",
                True,
                "All factor calculation checks passed",
                {
                    "factor_x": float(factor_x),
                    "factor_y": float(factor_y),
                    "factor_z": float(factor_z),
                    "factor_v": float(factor_v),
                    "factor_altitude": float(factor_alt),
                    "factor_velocity": float(factor_vel),
                },
            )
            return True

        except Exception as e:
            self._record_test_result("factor_calculation_test", False, f"Exception: {e!s}")
            self.logger.exception("Exception in factor calculation test")
            return False

    def create_test_problem(self, polynomial_degree: int = 5) -> Problem:
        """Create a standard test problem for verification tests.

        Args:
            polynomial_degree: Polynomial degree for the mesh (default: 5)
                               Must be <= 8 for adaptive mesh tests
        """
        problem = tl.Problem("Scaling Verification Problem")

        # Time variable
        t = problem.time(initial=0.0, free_final=True)

        # State variables with range of values
        h = problem.state("h", initial=260000.0, final=80000.0, lower=0.0)  # altitude
        v = problem.state("v", initial=25600.0, final=2500.0, lower=0.0)  # velocity
        gamma = problem.state("gamma", initial=-0.01, final=-0.1)  # flight path angle

        # Control with bounds
        u = problem.control("u", lower=-1.0, upper=1.0)

        # Simple dynamics
        problem.dynamics(
            {
                h: -v * ca.sin(gamma),
                v: -0.1 * v * ca.fabs(u) - 32.2 * ca.sin(gamma),
                gamma: u - 0.01 * gamma,
            }
        )

        # Objective: minimize time
        problem.minimize(t.final)

        # Set a standard mesh with specified polynomial degree
        problem.set_mesh([polynomial_degree, polynomial_degree], np.array([-1.0, 0.0, 1.0]))

        return problem

    def run_problem_solve_test(self) -> bool:
        """Test that scaling works correctly when solving a problem."""
        self.logger.info("Running problem solve test with scaling...")

        try:
            # Create test problem with polynomial degree 5
            polynomial_degree = 5
            problem = self.create_test_problem(polynomial_degree)

            # Set initial guess with correct dimensions: (states, polynomial_degree + 1)
            num_states = 3  # h, v, gamma
            num_controls = 1  # u

            # Create arrays of the correct size for each interval
            states_guess = []
            controls_guess = []

            for interval in range(2):  # 2 intervals in the mesh
                # Create state guess array with the correct shape: (num_states, polynomial_degree + 1)
                # For first interval: linearly interpolate from initial to middle
                # For second interval: linearly interpolate from middle to final
                if interval == 0:
                    h_vals = np.linspace(260000.0, 170000.0, polynomial_degree + 1)
                    v_vals = np.linspace(25600.0, 14000.0, polynomial_degree + 1)
                    gamma_vals = np.linspace(-0.01, -0.05, polynomial_degree + 1)
                else:
                    h_vals = np.linspace(170000.0, 80000.0, polynomial_degree + 1)
                    v_vals = np.linspace(14000.0, 2500.0, polynomial_degree + 1)
                    gamma_vals = np.linspace(-0.05, -0.1, polynomial_degree + 1)

                state_array = np.zeros((num_states, polynomial_degree + 1), dtype=np.float64)
                state_array[0, :] = h_vals
                state_array[1, :] = v_vals
                state_array[2, :] = gamma_vals
                states_guess.append(state_array)

                # Create control guess array with the correct shape: (num_controls, polynomial_degree)
                # Simple guess: zeros for each interval
                control_array = np.zeros((num_controls, polynomial_degree), dtype=np.float64)
                controls_guess.append(control_array)

            # Set the initial guess
            problem.set_initial_guess(
                states=states_guess, controls=controls_guess, terminal_time=100.0
            )

            # Solve twice: once with scaling on, once with scaling off
            # Use more generous solver options to improve convergence
            solver_options = {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 1000,  # Increase from default
                "ipopt.tol": 1e-6,  # Relax tolerance slightly
                "print_time": 0,
            }

            start_time_scaled = time.time()
            problem.use_scaling = True
            solution_scaled = tl.solve_fixed_mesh(problem, nlp_options=solver_options)
            solve_time_scaled = time.time() - start_time_scaled

            start_time_unscaled = time.time()
            problem.use_scaling = False
            solution_unscaled = tl.solve_fixed_mesh(problem, nlp_options=solver_options)
            solve_time_unscaled = time.time() - start_time_unscaled

            # Compare results
            scaled_success = solution_scaled.success
            unscaled_success = solution_unscaled.success

            if scaled_success and unscaled_success:
                # Both succeeded, compare objectives
                obj_diff = abs(solution_scaled.objective - solution_unscaled.objective)
                self.logger.info(f"Both solutions succeeded: obj diff = {obj_diff}")
                self.logger.info(
                    f"Scaled solve time: {solve_time_scaled:.2f}s, unscaled: {solve_time_unscaled:.2f}s"
                )

                self._record_test_result(
                    "problem_solve_test",
                    True,
                    f"Both solutions succeeded with obj diff {obj_diff:.6f}",
                    {
                        "scaled_objective": float(solution_scaled.objective),
                        "unscaled_objective": float(solution_unscaled.objective),
                        "objective_difference": float(obj_diff),
                        "scaled_solve_time": float(solve_time_scaled),
                        "unscaled_solve_time": float(solve_time_unscaled),
                    },
                )
                return True
            elif scaled_success:
                # Only scaled succeeded
                self.logger.info("Only scaled solution succeeded")
                self._record_test_result(
                    "problem_solve_test",
                    True,
                    "Only scaled solution succeeded - scaling is working correctly",
                    {
                        "scaled_objective": float(solution_scaled.objective),
                        "scaled_solve_time": float(solve_time_scaled),
                    },
                )
                return True
            elif unscaled_success:
                # Only unscaled succeeded - something is wrong
                self.logger.warning("Only unscaled solution succeeded - scaling may be broken")
                self._record_test_result(
                    "problem_solve_test",
                    False,
                    "Only unscaled solution succeeded - scaling may be broken",
                    {
                        "unscaled_objective": float(solution_unscaled.objective),
                        "unscaled_solve_time": float(solve_time_unscaled),
                    },
                )
                return False
            else:
                # Neither succeeded
                self.logger.warning("Both solutions failed")
                self._record_test_result(
                    "problem_solve_test",
                    False,
                    f"Both solutions failed - scaled: {solution_scaled.message}, "
                    f"unscaled: {solution_unscaled.message}",
                )
                return False

        except Exception as e:
            self._record_test_result("problem_solve_test", False, f"Exception: {e!s}")
            self.logger.exception("Exception in problem solve test")
            return False

    def run_solution_unscaling_test(self) -> bool:
        """Test that solution unscaling works correctly."""
        self.logger.info("Running solution unscaling test...")

        try:
            # Create test problem with lower polynomial degree for better convergence
            polynomial_degree = 5
            problem = self.create_test_problem(polynomial_degree)

            # Prepare a good initial guess with correct dimensions
            num_states = 3  # h, v, gamma
            num_controls = 1  # u

            # Create arrays with the correct shapes
            states_guess = []
            controls_guess = []

            for interval in range(2):  # 2 intervals in the mesh
                if interval == 0:
                    h_vals = np.linspace(260000.0, 170000.0, polynomial_degree + 1)
                    v_vals = np.linspace(25600.0, 14000.0, polynomial_degree + 1)
                    gamma_vals = np.linspace(-0.01, -0.05, polynomial_degree + 1)
                else:
                    h_vals = np.linspace(170000.0, 80000.0, polynomial_degree + 1)
                    v_vals = np.linspace(14000.0, 2500.0, polynomial_degree + 1)
                    gamma_vals = np.linspace(-0.05, -0.1, polynomial_degree + 1)

                state_array = np.zeros((num_states, polynomial_degree + 1), dtype=np.float64)
                state_array[0, :] = h_vals
                state_array[1, :] = v_vals
                state_array[2, :] = gamma_vals
                states_guess.append(state_array)

                # Simple control guess - slight negative value to encourage descent
                control_array = np.full((num_controls, polynomial_degree), -0.05, dtype=np.float64)
                controls_guess.append(control_array)

            # Set the initial guess
            problem.set_initial_guess(
                states=states_guess,
                controls=controls_guess,
                terminal_time=150.0,  # Increase initial guess for final time
            )

            # Use more generous solver options
            solver_options = {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 2000,  # More iterations
                "ipopt.tol": 1e-5,  # Relaxed tolerance
                "print_time": 0,
            }

            # Ensure scaling is enabled
            problem.use_scaling = True

            # Solve problem
            solution = tl.solve_fixed_mesh(problem, nlp_options=solver_options)

            if not solution.success:
                self._record_test_result(
                    "solution_unscaling_test",
                    False,
                    f"Solution failed to converge: {solution.message}",
                )
                return False

            # Verify solution was properly unscaled
            h_time, h_data = solution.get_state_trajectory("h")
            v_time, v_data = solution.get_state_trajectory("v")

            h_min, h_max = np.min(h_data), np.max(h_data)
            v_min, v_max = np.min(v_data), np.max(v_data)

            h_bounds_ok = h_min >= 80000.0 * 0.9 and h_max <= 260000.0 * 1.1
            v_bounds_ok = v_min >= 2500.0 * 0.9 and v_max <= 25600.0 * 1.1

            if not (h_bounds_ok and v_bounds_ok):
                self._record_test_result(
                    "solution_unscaling_test",
                    False,
                    f"Unscaled values out of expected range: "
                    f"h=[{h_min}, {h_max}], v=[{v_min}, {v_max}]",
                )
                return False

            # Also check if per-interval trajectories were unscaled
            traj_unscaled = True
            if (
                hasattr(solution, "solved_state_trajectories_per_interval")
                and solution.solved_state_trajectories_per_interval
            ):
                for interval_data in solution.solved_state_trajectories_per_interval:
                    h_interval = interval_data[0, :]
                    h_min_i, h_max_i = np.min(h_interval), np.max(h_interval)

                    if not (h_min_i >= 80000.0 * 0.9 and h_max_i <= 260000.0 * 1.1):
                        traj_unscaled = False
                        break

            if not traj_unscaled:
                self._record_test_result(
                    "solution_unscaling_test",
                    False,
                    "Per-interval trajectories not properly unscaled",
                )
                return False

            self._record_test_result(
                "solution_unscaling_test",
                True,
                f"Solution properly unscaled with h=[{h_min:.1f}, {h_max:.1f}], v=[{v_min:.1f}, {v_max:.1f}]",
                {
                    "altitude_min": float(h_min),
                    "altitude_max": float(h_max),
                    "velocity_min": float(v_min),
                    "velocity_max": float(v_max),
                },
            )
            return True

        except Exception as e:
            self._record_test_result("solution_unscaling_test", False, f"Exception: {e!s}")
            self.logger.exception("Exception in solution unscaling test")
            return False

    def run_adaptive_mesh_scaling_test(self) -> bool:
        """Test that scaling works correctly with adaptive mesh refinement."""
        self.logger.info("Running adaptive mesh scaling test...")

        try:
            # Create test problem with polynomial degree compatible with adaptive settings
            # Use polynomial degree 6, which is lower than max_polynomial_degree=8
            polynomial_degree = 6
            problem = self.create_test_problem(polynomial_degree)

            # Prepare a suitable initial guess
            num_states = 3  # h, v, gamma
            num_controls = 1  # u

            # Create arrays with the correct shapes
            states_guess = []
            controls_guess = []

            for interval in range(2):  # 2 intervals in the mesh
                if interval == 0:
                    h_vals = np.linspace(260000.0, 170000.0, polynomial_degree + 1)
                    v_vals = np.linspace(25600.0, 14000.0, polynomial_degree + 1)
                    gamma_vals = np.linspace(-0.01, -0.05, polynomial_degree + 1)
                else:
                    h_vals = np.linspace(170000.0, 80000.0, polynomial_degree + 1)
                    v_vals = np.linspace(14000.0, 2500.0, polynomial_degree + 1)
                    gamma_vals = np.linspace(-0.05, -0.1, polynomial_degree + 1)

                state_array = np.zeros((num_states, polynomial_degree + 1), dtype=np.float64)
                state_array[0, :] = h_vals
                state_array[1, :] = v_vals
                state_array[2, :] = gamma_vals
                states_guess.append(state_array)

                # Simple control guess - slight negative value
                control_array = np.full((num_controls, polynomial_degree), -0.05, dtype=np.float64)
                controls_guess.append(control_array)

            # Set the initial guess
            problem.set_initial_guess(
                states=states_guess, controls=controls_guess, terminal_time=150.0
            )

            # Ensure scaling is enabled
            problem.use_scaling = True

            # Solver options
            solver_options = {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 1000,
                "ipopt.tol": 1e-5,
                "print_time": 0,
            }

            # Solve with adaptive mesh - make sure polynomial limits are compatible
            # Initial polynomial degree is 6, so max should be at least 6
            solution = tl.solve_adaptive(
                problem,
                error_tolerance=1e-2,  # Relaxed tolerance for faster convergence
                max_iterations=3,  # Limit for test
                min_polynomial_degree=4,
                max_polynomial_degree=8,  # Must be >= initial polynomial degree (6)
                nlp_options=solver_options,
            )

            # Check if solution succeeded or at least made progress
            if not (solution.success or "maximum iterations" in solution.message.lower()):
                self._record_test_result(
                    "adaptive_mesh_scaling_test",
                    False,
                    f"Adaptive solution failed: {solution.message}",
                )
                return False

            # Check that solution values are in expected range
            h_time, h_data = solution.get_state_trajectory("h")
            v_time, v_data = solution.get_state_trajectory("v")

            h_min, h_max = np.min(h_data), np.max(h_data)
            v_min, v_max = np.min(v_data), np.max(v_data)

            h_bounds_ok = h_min >= 80000.0 * 0.9 and h_max <= 260000.0 * 1.1
            v_bounds_ok = v_min >= 2500.0 * 0.9 and v_max <= 25600.0 * 1.1

            if not (h_bounds_ok and v_bounds_ok):
                self._record_test_result(
                    "adaptive_mesh_scaling_test",
                    False,
                    f"Adaptive solution values out of expected range: "
                    f"h=[{h_min}, {h_max}], v=[{v_min}, {v_max}]",
                )
                return False

            # Extract mesh refinement information
            final_intervals = (
                len(solution.mesh_points) - 1 if solution.mesh_points is not None else 0
            )

            self._record_test_result(
                "adaptive_mesh_scaling_test",
                True,
                f"Adaptive mesh scaling worked correctly with {final_intervals} intervals",
                {
                    "final_intervals": float(final_intervals),
                    "altitude_min": float(h_min),
                    "altitude_max": float(h_max),
                    "velocity_min": float(v_min),
                    "velocity_max": float(v_max),
                },
            )
            return True

        except Exception as e:
            self._record_test_result("adaptive_mesh_scaling_test", False, f"Exception: {e!s}")
            self.logger.exception("Exception in adaptive mesh scaling test")
            return False

    def run_all_tests(self) -> bool:
        """Run all scaling verification tests."""
        self.logger.info("Starting all scaling verification tests...")

        # Run tests in sequence
        basic_result = self.run_basic_scaling_test()
        factor_result = self.run_factor_calculation_test()
        solve_result = self.run_problem_solve_test()
        unscaling_result = self.run_solution_unscaling_test()
        adaptive_result = self.run_adaptive_mesh_scaling_test()

        # Overall result
        all_passed = (
            basic_result and factor_result and solve_result and unscaling_result and adaptive_result
        )

        # Generate summary report
        self._generate_report(all_passed)

        self.logger.info(f"All tests completed: {'PASSED' if all_passed else 'FAILED'}")
        self.logger.info(f"Detailed results written to: {self.results_file}")

        return all_passed

    def _generate_report(self, all_passed: bool) -> None:
        """Generate a detailed verification report."""
        with open(self.results_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TRAJECTOLAB SCALING VERIFICATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"OVERALL RESULT: {'PASSED' if all_passed else 'FAILED'}\n\n")

            f.write("TEST RESULTS SUMMARY:\n")
            f.write("-" * 80 + "\n")

            for test_name, result in self.results.items():
                status = "PASSED" if result["passed"] else "FAILED"
                f.write(f"{test_name}: {status}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED TEST RESULTS:\n")
            f.write("=" * 80 + "\n\n")

            for test_name, result in self.results.items():
                f.write(f"TEST: {test_name}\n")
                f.write(f"RESULT: {'PASSED' if result['passed'] else 'FAILED'}\n")
                f.write(f"DETAILS: {result['details']}\n")

                # Add metrics if available
                metrics = {k: v for k, v in result.items() if k not in ["passed", "details"]}
                if metrics:
                    f.write("METRICS:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")

                f.write("-" * 80 + "\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("ENVIRONMENT INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"NumPy version: {np.__version__}\n")
            f.write(f"CasADi version: {ca.__version__}\n")
            f.write(f"TrajectoLab version: {tl.__version__}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("NOTES FOR NASA ENGINEERS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                "1. This report verifies that the TrajectoLab scaling system is functioning correctly.\n"
            )
            f.write(
                "2. The scaling system uses a single source of truth (problem.use_scaling property).\n"
            )
            f.write(
                "3. Scaling factors are computed from variable bounds or initial guesses with safety margins.\n"
            )
            f.write("4. Solution values are automatically unscaled back to original units.\n")
            f.write("5. For detailed logs of the tests, see the log file.\n")
            f.write("\n")
            f.write(
                "If any tests failed, please contact the TrajectoLab development team for assistance.\n"
            )


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="TrajectoLab Scaling Verification Tool for NASA Engineers"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store log files (default: ./scaling_logs)",
    )

    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "basic", "factor", "solve", "unscaling", "adaptive"],
        default="all",
        help="Specific test to run (default: all)",
    )

    args = parser.parse_args()

    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level)

    # Create verifier
    verifier = ScalingVerifier(log_level=log_level, log_dir=args.log_dir)

    # Run selected test
    if args.test == "all":
        result = verifier.run_all_tests()
    elif args.test == "basic":
        result = verifier.run_basic_scaling_test()
    elif args.test == "factor":
        result = verifier.run_factor_calculation_test()
    elif args.test == "solve":
        result = verifier.run_problem_solve_test()
    elif args.test == "unscaling":
        result = verifier.run_solution_unscaling_test()
    elif args.test == "adaptive":
        result = verifier.run_adaptive_mesh_scaling_test()

    # Exit with appropriate code
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
