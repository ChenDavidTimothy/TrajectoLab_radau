from io import StringIO
from unittest.mock import patch

import numpy as np

import maptor as mtor


class TestBenchmarkAPIExhaustive:
    """Exhaustive test suite for adaptive benchmark API capabilities."""

    def test_algorithm_status_complete(self):
        problem = self._create_simple_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=6,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        assert solution.status["success"], "Problem should converge"
        adaptive = solution.adaptive
        assert adaptive is not None, "Adaptive data must be available"

        # Test all documented algorithm status fields
        assert isinstance(adaptive["converged"], bool), "converged must be bool"
        assert isinstance(adaptive["iterations"], int), "iterations must be int"
        assert adaptive["iterations"] >= 0, "iterations must be non-negative"
        assert isinstance(adaptive["target_tolerance"], float), "target_tolerance must be float"
        assert adaptive["target_tolerance"] > 0, "target_tolerance must be positive"

        # Test phase convergence status
        phase_converged = adaptive["phase_converged"]
        assert isinstance(phase_converged, dict), "phase_converged must be dict"
        for phase_id, converged in phase_converged.items():
            assert isinstance(phase_id, int), f"Phase ID {phase_id} must be int"
            assert isinstance(converged, bool), f"Phase {phase_id} convergence must be bool"

        # Test final errors structure
        final_errors = adaptive["final_errors"]
        assert isinstance(final_errors, dict), "final_errors must be dict"
        for phase_id, errors in final_errors.items():
            assert isinstance(errors, list), f"Phase {phase_id} errors must be list"
            assert all(isinstance(e, float) for e in errors), (
                f"Phase {phase_id} errors must be floats"
            )

    def test_benchmark_arrays_complete_structure(self):
        problem = self._create_simple_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-5,
            max_iterations=5,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        benchmark = solution.adaptive["benchmark"]

        # Test all 6 documented benchmark arrays exist
        required_arrays = {
            "mesh_iteration",
            "estimated_error",
            "collocation_points",
            "mesh_intervals",
            "polynomial_degrees",
            "refinement_strategy",
        }
        assert set(benchmark.keys()) == required_arrays, (
            f"Missing arrays: {required_arrays - set(benchmark.keys())}"
        )

        # Verify consistent lengths
        num_iterations = len(benchmark["mesh_iteration"])
        assert num_iterations > 0, "Must have at least one iteration"

        for array_name in required_arrays:
            assert len(benchmark[array_name]) == num_iterations, f"{array_name} length mismatch"

        # Test specific array types and contents
        self._validate_mesh_iteration_array(benchmark["mesh_iteration"])
        self._validate_estimated_error_array(benchmark["estimated_error"])
        self._validate_collocation_points_array(benchmark["collocation_points"])
        self._validate_mesh_intervals_array(benchmark["mesh_intervals"])
        self._validate_polynomial_degrees_array(benchmark["polynomial_degrees"])
        self._validate_refinement_strategy_array(benchmark["refinement_strategy"])

    def test_phase_specific_benchmark_complete(self):
        """Verify phase-specific benchmark data has complete structure."""
        problem = self._create_multiphase_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=4,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        phase_benchmarks = solution.adaptive["phase_benchmarks"]
        phase_ids = list(solution.phases.keys())

        # Verify all phases have benchmark data
        assert set(phase_benchmarks.keys()) == set(phase_ids), "Phase benchmark coverage mismatch"

        # Test each phase has complete structure
        for phase_id in phase_ids:
            phase_data = phase_benchmarks[phase_id]
            required_arrays = {
                "mesh_iteration",
                "estimated_error",
                "collocation_points",
                "mesh_intervals",
                "polynomial_degrees",
                "refinement_strategy",
            }
            assert set(phase_data.keys()) == required_arrays, f"Phase {phase_id} missing arrays"

            # Verify consistent lengths within phase
            num_iterations = len(phase_data["mesh_iteration"])
            for array_name in required_arrays:
                assert len(phase_data[array_name]) == num_iterations, (
                    f"Phase {phase_id} {array_name} length mismatch"
                )

            # Test phase-specific array validity
            self._validate_phase_benchmark_arrays(phase_data, phase_id)

    def test_iteration_history_access(self):
        problem = self._create_simple_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=4,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        history = solution.adaptive["iteration_history"]
        assert isinstance(history, dict), "iteration_history must be dict"

        iterations = sorted(history.keys())
        assert iterations == list(range(len(iterations))), "Iteration keys must be sequential"

        # Test each iteration has complete data structure
        for iteration in iterations:
            data = history[iteration]
            required_fields = {
                "iteration",
                "phase_error_estimates",
                "phase_collocation_points",
                "phase_mesh_intervals",
                "phase_polynomial_degrees",
                "phase_mesh_nodes",
                "refinement_strategy",
                "total_collocation_points",
                "max_error_all_phases",
                "convergence_status",
            }
            assert set(data.keys()) == required_fields, f"Iteration {iteration} missing fields"

            # Validate field types and content
            assert data["iteration"] == iteration, "Iteration number mismatch"
            assert isinstance(data["total_collocation_points"], int), (
                "total_collocation_points must be int"
            )
            assert data["total_collocation_points"] > 0, "total_collocation_points must be positive"
            assert isinstance(data["max_error_all_phases"], float), (
                "max_error_all_phases must be float"
            )

    def test_benchmark_data_progression_logic(self):
        problem = self._create_simple_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=8,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        benchmark = solution.adaptive["benchmark"]

        # Test iteration progression
        iterations = benchmark["mesh_iteration"]
        assert iterations == list(range(len(iterations))), (
            "Iterations must be sequential starting from 0"
        )

        # Test error progression (first iteration should be NaN, then decreasing trend)
        errors = benchmark["estimated_error"]

        finite_errors = [e for e in errors[1:] if not (np.isnan(e) or np.isinf(e))]
        if len(finite_errors) >= 2:
            # General decreasing trend (allowing for some fluctuation)
            early_avg = np.mean(finite_errors[: len(finite_errors) // 2])
            late_avg = np.mean(finite_errors[len(finite_errors) // 2 :])
            assert late_avg <= early_avg, "Error should generally decrease over iterations"

        # Test collocation points progression (should generally increase)
        points = benchmark["collocation_points"]
        assert all(p > 0 for p in points), "All collocation point counts must be positive"
        assert points[-1] >= points[0], "Final mesh should have at least as many points as initial"

        # Test mesh intervals progression
        intervals = benchmark["mesh_intervals"]
        assert all(i > 0 for i in intervals), "All interval counts must be positive"

    def test_built_in_methods_functionality(self):
        problem = self._create_simple_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=3,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        # Test print_benchmark_summary doesn't crash
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            solution.print_benchmark_summary()

        output = captured_output.getvalue()
        assert "ADAPTIVE MESH REFINEMENT BENCHMARK" in output, (
            "Summary should contain expected header"
        )
        assert "Status:" in output, "Summary should contain status"
        assert "Iterations:" in output, "Summary should contain iteration count"

        # Test plot_refinement_history doesn't crash (mock matplotlib)
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            mock_fig = patch("matplotlib.figure.Figure").start()
            mock_ax = patch("matplotlib.axes.Axes").start()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Should not raise exception
            solution.plot_refinement_history(phase_id=1)

            # Verify matplotlib was called
            mock_subplots.assert_called_once()

    def test_edge_cases_and_failures(self):
        # Test fixed mesh solution (no adaptive data)
        problem = self._create_simple_problem()

        fixed_solution = mtor.solve_fixed_mesh(
            problem,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        assert fixed_solution.adaptive is None, "Fixed mesh should have no adaptive data"

        # Test early convergence (minimal iterations)
        convergent_problem = self._create_trivial_problem()

        solution = mtor.solve_adaptive(
            convergent_problem,
            error_tolerance=1e-2,  # Very loose tolerance
            max_iterations=2,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        if solution.status["success"]:
            benchmark = solution.adaptive["benchmark"]
            # Should still have valid structure even with minimal iterations
            assert len(benchmark["mesh_iteration"]) >= 1, "Should have at least initial iteration"
            assert all(
                len(benchmark[key]) == len(benchmark["mesh_iteration"]) for key in benchmark.keys()
            ), "Consistent array lengths even with early convergence"

    def test_multiphase_benchmark_consistency(self):
        problem = self._create_multiphase_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=5,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        mission_benchmark = solution.adaptive["benchmark"]
        phase_benchmarks = solution.adaptive["phase_benchmarks"]

        # Test mission-wide totals match sum of phases
        for i in range(len(mission_benchmark["mesh_iteration"])):
            # Total collocation points should equal sum across phases
            mission_points = mission_benchmark["collocation_points"][i]
            phase_points_sum = sum(
                phase_data["collocation_points"][i] for phase_data in phase_benchmarks.values()
            )
            assert mission_points == phase_points_sum, (
                f"Iteration {i}: mission points != sum of phase points"
            )

            # Total intervals should equal sum across phases
            mission_intervals = mission_benchmark["mesh_intervals"][i]
            phase_intervals_sum = sum(
                phase_data["mesh_intervals"][i] for phase_data in phase_benchmarks.values()
            )
            assert mission_intervals == phase_intervals_sum, (
                f"Iteration {i}: mission intervals != sum of phase intervals"
            )

    def test_data_export_patterns(self):
        problem = self._create_simple_problem()

        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=4,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        benchmark = solution.adaptive["benchmark"]

        # Test CSV-ready extraction
        csv_rows = []
        for i in range(len(benchmark["mesh_iteration"])):
            iteration = benchmark["mesh_iteration"][i]
            error = benchmark["estimated_error"][i]
            points = benchmark["collocation_points"][i]
            intervals = benchmark["mesh_intervals"][i]

            # Should be able to format as CSV
            csv_row = f"{iteration},{error},{points},{intervals}"
            csv_rows.append(csv_row)

        assert len(csv_rows) > 0, "Should generate CSV rows"
        assert all("," in row for row in csv_rows), "All rows should be comma-separated"

        # Test numpy array conversion
        points_array = np.array(benchmark["collocation_points"])
        assert points_array.dtype in [np.int32, np.int64], "Points should convert to integer array"

        error_array = np.array(benchmark["estimated_error"])
        assert error_array.dtype == np.float64, "Errors should convert to float array"

    def _validate_mesh_iteration_array(self, iterations):
        assert isinstance(iterations, list), "mesh_iteration must be list"
        assert all(isinstance(i, int) for i in iterations), "All iterations must be int"
        assert iterations == list(range(len(iterations))), "Iterations must be sequential from 0"

    def _validate_estimated_error_array(self, errors):
        assert isinstance(errors, list), "estimated_error must be list"
        assert all(isinstance(e, float) for e in errors), "All errors must be float"

    def _validate_collocation_points_array(self, points):
        assert isinstance(points, list), "collocation_points must be list"
        assert all(isinstance(p, int) for p in points), "All points must be int"
        assert all(p > 0 for p in points), "All points must be positive"

    def _validate_mesh_intervals_array(self, intervals):
        assert isinstance(intervals, list), "mesh_intervals must be list"
        assert all(isinstance(i, int) for i in intervals), "All intervals must be int"
        assert all(i > 0 for i in intervals), "All intervals must be positive"

    def _validate_polynomial_degrees_array(self, degrees):
        assert isinstance(degrees, list), "polynomial_degrees must be list"
        for deg_list in degrees:
            assert isinstance(deg_list, list), "Each polynomial_degrees entry must be list"
            assert all(isinstance(d, int) for d in deg_list), "All degrees must be int"
            assert all(d > 0 for d in deg_list), "All degrees must be positive"

    def _validate_refinement_strategy_array(self, strategies):
        assert isinstance(strategies, list), "refinement_strategy must be list"
        for strategy_dict in strategies:
            assert isinstance(strategy_dict, dict), "Each strategy entry must be dict"
            for interval_idx, strategy in strategy_dict.items():
                assert isinstance(interval_idx, int), "Interval indices must be int"
                assert strategy in ["p", "h"], f"Strategy must be 'p' or 'h', got {strategy}"

    def _validate_phase_benchmark_arrays(self, phase_data, phase_id):
        # Same validation as mission-wide but for phase data
        iterations = phase_data["mesh_iteration"]
        assert iterations == list(range(len(iterations))), (
            f"Phase {phase_id} iterations not sequential"
        )

        points = phase_data["collocation_points"]
        assert all(p >= 0 for p in points), f"Phase {phase_id} negative collocation points"

        intervals = phase_data["mesh_intervals"]
        assert all(i >= 0 for i in intervals), f"Phase {phase_id} negative intervals"

    def _create_simple_problem(self):
        problem = mtor.Problem("Benchmark Test Problem")
        phase = problem.set_phase(1)

        t = phase.time(initial=0.0, final=1.0)
        x = phase.state("x", initial=0.0, final=1.0)
        u = phase.control("u", boundary=(-2.0, 2.0))

        phase.dynamics({x: u})

        control_effort = phase.add_integral(u**2)
        problem.minimize(control_effort)

        phase.mesh([3, 3], [-1.0, 0.0, 1.0])
        return problem

    def _create_multiphase_problem(self):
        problem = mtor.Problem("Multiphase Benchmark Test")

        # Phase 1
        phase1 = problem.set_phase(1)
        t1 = phase1.time(initial=0.0, final=0.5)
        x1 = phase1.state("x", initial=0.0)
        u1 = phase1.control("u", boundary=(-1.0, 1.0))
        phase1.dynamics({x1: u1})
        phase1.mesh([3], [-1.0, 1.0])

        # Phase 2
        phase2 = problem.set_phase(2)
        t2 = phase2.time(initial=t1.final, final=1.0)
        x2 = phase2.state("x", initial=x1.final, final=1.0)
        u2 = phase2.control("u", boundary=(-1.0, 1.0))
        phase2.dynamics({x2: u2})
        phase2.mesh([3], [-1.0, 1.0])

        effort1 = phase1.add_integral(u1**2)
        effort2 = phase2.add_integral(u2**2)
        problem.minimize(effort1 + effort2)

        return problem

    def _create_trivial_problem(self):
        problem = mtor.Problem("Trivial Test")
        phase = problem.set_phase(1)

        t = phase.time(initial=0.0, final=0.1)
        x = phase.state("x", initial=0.0, final=0.1)
        u = phase.control("u", boundary=(-0.1, 0.1))

        phase.dynamics({x: u})
        problem.minimize(u**2)

        phase.mesh([2], [-1.0, 1.0])
        return problem
