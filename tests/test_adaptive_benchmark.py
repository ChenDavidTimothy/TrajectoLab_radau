from io import StringIO
from unittest.mock import patch

import numpy as np

import maptor as mtor


class TestBenchmarkAPITargeted:
    def test_algorithm_status_access(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=4,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        assert solution.status["success"], "Problem should converge"
        adaptive = solution.adaptive
        assert adaptive is not None, "Adaptive data must be available"

        # Test all algorithm status fields
        assert isinstance(adaptive["converged"], bool)
        assert isinstance(adaptive["iterations"], int)
        assert adaptive["iterations"] >= 0
        assert isinstance(adaptive["target_tolerance"], float)
        assert adaptive["target_tolerance"] > 0

        # Test phase convergence status
        assert isinstance(adaptive["phase_converged"], dict)
        for phase_id, converged in adaptive["phase_converged"].items():
            assert isinstance(phase_id, int)
            assert isinstance(converged, bool)

        # Test final errors structure
        assert isinstance(adaptive["final_errors"], dict)
        for phase_id, errors in adaptive["final_errors"].items():
            assert isinstance(errors, list)
            assert all(isinstance(e, float) for e in errors)

    def test_mission_benchmark_arrays_structure(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-5,
            max_iterations=5,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        benchmark = solution.adaptive["benchmark"]

        # Test all 6 required arrays exist
        required_arrays = {
            "mesh_iteration",
            "estimated_error",
            "collocation_points",
            "mesh_intervals",
            "polynomial_degrees",
            "refinement_strategy",
        }
        assert set(benchmark.keys()) == required_arrays

        # Test consistent lengths
        num_iterations = len(benchmark["mesh_iteration"])
        assert num_iterations > 0
        for array_name in required_arrays:
            assert len(benchmark[array_name]) == num_iterations

        # Test specific array types and content
        assert benchmark["mesh_iteration"] == list(range(num_iterations))
        assert all(isinstance(e, float) for e in benchmark["estimated_error"])
        assert all(isinstance(p, int) and p > 0 for p in benchmark["collocation_points"])
        assert all(isinstance(i, int) and i > 0 for i in benchmark["mesh_intervals"])
        assert all(isinstance(deg_list, list) for deg_list in benchmark["polynomial_degrees"])
        assert all(
            isinstance(strategy_dict, dict) for strategy_dict in benchmark["refinement_strategy"]
        )

    def test_phase_benchmark_arrays_structure(self):
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
        assert set(phase_benchmarks.keys()) == set(phase_ids)

        # Test each phase has same structure as mission-wide
        required_arrays = {
            "mesh_iteration",
            "estimated_error",
            "collocation_points",
            "mesh_intervals",
            "polynomial_degrees",
            "refinement_strategy",
        }

        for phase_id in phase_ids:
            phase_data = phase_benchmarks[phase_id]
            assert set(phase_data.keys()) == required_arrays

            # Verify consistent lengths within phase
            num_iterations = len(phase_data["mesh_iteration"])
            for array_name in required_arrays:
                assert len(phase_data[array_name]) == num_iterations

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
        assert isinstance(history, dict)

        iterations = sorted(history.keys())
        assert iterations == list(range(len(iterations)))

        # Test each iteration has complete IterationData structure
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

        for iteration in iterations:
            data = history[iteration]
            assert set(data.keys()) == required_fields
            assert data["iteration"] == iteration
            assert isinstance(data["total_collocation_points"], int)
            assert data["total_collocation_points"] > 0
            assert isinstance(data["max_error_all_phases"], float)

    def test_mission_vs_phase_consistency(self):
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

        # Test mission-wide totals equal sum of phases
        for i in range(len(mission_benchmark["mesh_iteration"])):
            # Total collocation points
            mission_points = mission_benchmark["collocation_points"][i]
            phase_points_sum = sum(
                phase_data["collocation_points"][i] for phase_data in phase_benchmarks.values()
            )
            assert mission_points == phase_points_sum

            # Total intervals
            mission_intervals = mission_benchmark["mesh_intervals"][i]
            phase_intervals_sum = sum(
                phase_data["mesh_intervals"][i] for phase_data in phase_benchmarks.values()
            )
            assert mission_intervals == phase_intervals_sum

    def test_algorithmic_progression_logic(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-6,
            max_iterations=8,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        benchmark = solution.adaptive["benchmark"]

        # Test iteration sequence
        iterations = benchmark["mesh_iteration"]
        assert iterations == list(range(len(iterations)))

        # Test collocation points progression (should generally increase)
        points = benchmark["collocation_points"]
        assert all(p > 0 for p in points)
        assert points[-1] >= points[0]

        # Test mesh intervals progression
        intervals = benchmark["mesh_intervals"]
        assert all(i > 0 for i in intervals)

        # Test error progression (finite errors should generally decrease)
        errors = benchmark["estimated_error"]
        finite_errors = [e for e in errors[1:] if not (np.isnan(e) or np.isinf(e))]
        if len(finite_errors) >= 2:
            early_avg = np.mean(finite_errors[: len(finite_errors) // 2])
            late_avg = np.mean(finite_errors[len(finite_errors) // 2 :])
            assert late_avg <= early_avg

    def test_refinement_strategy_tracking(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-5,
            max_iterations=6,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        if not solution.status["success"]:
            return

        # Test benchmark array refinement strategies
        benchmark = solution.adaptive["benchmark"]
        strategies = benchmark["refinement_strategy"]

        for strategy_dict in strategies:
            assert isinstance(strategy_dict, dict)
            for interval_idx, strategy in strategy_dict.items():
                assert isinstance(interval_idx, int)
                assert strategy in ["p", "h"]

        # Test iteration history refinement strategy consistency
        history = solution.adaptive["iteration_history"]
        for iteration_data in history.values():
            for phase_strategies in iteration_data["refinement_strategy"].values():
                for strategy in phase_strategies.values():
                    assert strategy in ["p", "h"]

    def test_export_data_access_patterns(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=4,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        benchmark = solution.adaptive["benchmark"]

        # Test CSV export capability
        csv_rows = []
        for i in range(len(benchmark["mesh_iteration"])):
            iteration = benchmark["mesh_iteration"][i]
            error = benchmark["estimated_error"][i]
            points = benchmark["collocation_points"][i]
            intervals = benchmark["mesh_intervals"][i]
            error_str = "NaN" if np.isnan(error) else f"{error:.6e}"
            csv_row = f"{iteration},{error_str},{points},{intervals}"
            csv_rows.append(csv_row)

        assert len(csv_rows) > 0
        assert all("," in row for row in csv_rows)

        # Test numpy array conversion
        points_array = np.array(benchmark["collocation_points"])
        assert points_array.dtype in [np.int32, np.int64]

        error_array = np.array(benchmark["estimated_error"])
        assert error_array.dtype == np.float64

    def test_built_in_analysis_methods(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=3,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        # Test print_benchmark_summary
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            solution.print_benchmark_summary()

        output = captured_output.getvalue()
        assert "ADAPTIVE MESH REFINEMENT BENCHMARK" in output
        assert "Status:" in output
        assert "Iterations:" in output

        # Test plot_refinement_history
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            mock_fig = patch("matplotlib.figure.Figure").start()
            mock_ax = patch("matplotlib.axes.Axes").start()
            mock_subplots.return_value = (mock_fig, mock_ax)

            solution.plot_refinement_history(phase_id=1)
            mock_subplots.assert_called_once()

    def test_single_source_data_integrity(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=3,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        # Verify iteration_history is source of truth
        history = solution.adaptive["iteration_history"]
        benchmark = solution.adaptive["benchmark"]

        # Test that benchmark data matches iteration_history data
        assert len(benchmark["mesh_iteration"]) == len(history)

        for i, iteration in enumerate(sorted(history.keys())):
            data = history[iteration]
            assert benchmark["mesh_iteration"][i] == iteration
            assert benchmark["collocation_points"][i] == data["total_collocation_points"]
            assert benchmark["estimated_error"][i] == data["max_error_all_phases"]

    def test_edge_cases_and_fixed_mesh(self):
        # Test fixed mesh solution (no adaptive data)
        problem = self._create_simple_problem()
        fixed_solution = mtor.solve_fixed_mesh(
            problem,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )
        assert fixed_solution.adaptive is None

        # Test early convergence with loose tolerance
        simple_problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            simple_problem,
            error_tolerance=1e-2,  # Very loose tolerance
            max_iterations=2,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        if solution.status["success"] and solution.adaptive:
            benchmark = solution.adaptive["benchmark"]
            assert len(benchmark["mesh_iteration"]) >= 1
            # Verify all arrays have consistent lengths
            base_length = len(benchmark["mesh_iteration"])
            for key in benchmark.keys():
                assert len(benchmark[key]) == base_length

    def test_gamma_factors_access(self):
        problem = self._create_simple_problem()
        solution = mtor.solve_adaptive(
            problem,
            error_tolerance=1e-4,
            max_iterations=3,
            nlp_options={"ipopt.print_level": 0},
            show_summary=False,
        )

        gamma_factors = solution.adaptive["gamma_factors"]
        assert isinstance(gamma_factors, dict)

        for phase_id, factors in gamma_factors.items():
            assert isinstance(phase_id, int)
            # factors can be FloatArray or None
            if factors is not None:
                assert isinstance(factors, np.ndarray)
                assert factors.dtype == np.float64

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
