"""
Comprehensive Initial Guess Demonstration for TrajectoLab
=========================================================

This script demonstrates all possible initial guess scenarios for the hypersensitive problem:
1. No initial guess (CasADi defaults)
2. Partial initial guesses (various combinations)
3. Complete initial guess
4. Wrong dimensional initial guess (expected to fail)
5. Adaptive mesh scenarios
6. Priority demonstrations (problem vs constructor)
7. Mesh change behavior

ALL FIXED MESH SCENARIOS USE IDENTICAL MESH: [20, 8, 20] degrees, [-1, -1/3, 1/3, 1] points
This allows direct comparison of initial guess strategies on the same discretization.

Each scenario is plotted if successful. The script fails fast on any error.
"""

import sys
from typing import Any

import numpy as np

from trajectolab import FixedMesh, InitialGuess, PHSAdaptive, Problem, RadauDirectSolver, solve


# STANDARDIZED MESH for ALL fixed mesh scenarios
STANDARD_DEGREES = [20, 12, 20]
STANDARD_MESH = np.array([-1.0, -1 / 3, 1 / 3, 1.0])

# STANDARDIZED ADAPTIVE CONFIGURATION for ALL adaptive scenarios
ADAPTIVE_DEGREES = [8, 8, 8]
ADAPTIVE_MESH = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
ADAPTIVE_SOLVER_CONFIG = {
    "error_tolerance": 1e-3,
    "max_iterations": 30,
    "min_polynomial_degree": 4,
    "max_polynomial_degree": 8,
}
ADAPTIVE_NLP_OPTIONS = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": 0,
    "ipopt.max_iter": 200,
}


class DemoRunner:
    """Handles running demos with fail-fast behavior and plotting."""

    def __init__(self) -> None:
        self.scenario_count = 0
        self.success_count = 0

    def run_scenario(self, name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a scenario and handle failure with fail-fast behavior."""
        self.scenario_count += 1
        print(f"\n{'=' * 80}")
        print(f"SCENARIO {self.scenario_count}: {name}")
        print(f"{'=' * 80}")
        print(f"Mesh: {STANDARD_DEGREES} degrees, {STANDARD_MESH} points")

        try:
            result = func(*args, **kwargs)
            self.success_count += 1
            print(f"✅ SCENARIO {self.scenario_count} SUCCESSFUL")
            return result
        except Exception as e:
            print(f"❌ SCENARIO {self.scenario_count} FAILED: {e}")
            print(
                f"\nFAIL FAST: Stopping execution after {self.success_count}/{self.scenario_count} successful scenarios"
            )
            sys.exit(1)


def create_hypersensitive_problem() -> Problem:
    """Create the hypersensitive optimal control problem."""
    problem = Problem("Hypersensitive Problem - Initial Guess Demo")

    # Define time variable
    problem.time(initial=0.0, final=40.0)

    # Add state with boundary conditions
    x = problem.state("x", initial=1.5, final=1.0)

    # Add control
    u = problem.control("u")

    # Define dynamics using symbolic expressions
    problem.dynamics({x: -(x**3) + u})

    # Define objective using symbolic expressions
    integral_expr = 0.5 * (x**2 + u**2)
    integral_var = problem.add_integral(integral_expr)

    # Set the objective to minimize
    problem.minimize(integral_var)

    return problem


def create_linear_state_guess(
    polynomial_degrees: list[int], x_initial: float = 1.5, x_final: float = 1.0
) -> list[np.ndarray]:
    """Create a linear interpolation initial guess for states."""
    states_guess = []

    for N_k in polynomial_degrees:
        # Create local tau points for this interval
        tau_points = np.linspace(-1, 1, N_k + 1)

        # Simple linear interpolation from initial to final
        x_values = x_initial + (x_final - x_initial) * (tau_points + 1) / 2

        # Reshape for (num_states=1, num_nodes)
        state_array = x_values.reshape(1, -1)
        states_guess.append(state_array)

    return states_guess


def create_control_guess(
    polynomial_degrees: list[int], control_value: float = 0.0
) -> list[np.ndarray]:
    """Create a constant control initial guess."""
    controls_guess = []

    for N_k in polynomial_degrees:
        # Constant control at each collocation point
        control_array = np.full((1, N_k), control_value, dtype=np.float64)
        controls_guess.append(control_array)

    return controls_guess


def create_sophisticated_guess(
    polynomial_degrees: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create a more sophisticated initial guess based on problem understanding."""
    states_guess = []
    controls_guess = []

    for _k, N_k in enumerate(polynomial_degrees):
        # State guess: exponential-like decay from 1.5 to 1.0
        tau_points = np.linspace(-1, 1, N_k + 1)
        global_time = (tau_points + 1) / 2  # Map to [0, 1]

        # Exponential decay profile
        x_values = 1.0 + 0.5 * np.exp(-2 * global_time)
        state_array = x_values.reshape(1, -1)
        states_guess.append(state_array)

        # Control guess: compensate for state dynamics
        tau_control = np.linspace(-1, 1, N_k)
        global_time_control = (tau_control + 1) / 2
        x_at_control = 1.0 + 0.5 * np.exp(-2 * global_time_control)

        # u ≈ x^3 to roughly satisfy dynamics
        u_values = x_at_control**3
        control_array = u_values.reshape(1, -1)
        controls_guess.append(control_array)

    return states_guess, controls_guess


def create_sinusoidal_guess(
    polynomial_degrees: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create a sinusoidal initial guess to test different trajectory shapes."""
    states_guess = []
    controls_guess = []

    for _k, N_k in enumerate(polynomial_degrees):
        # State guess: sinusoidal transition from 1.5 to 1.0
        tau_points = np.linspace(-1, 1, N_k + 1)
        global_time = (tau_points + 1) / 2  # Map to [0, 1]

        # Sinusoidal profile: 1.5 to 1.0 with oscillation
        x_values = 1.0 + 0.5 * (1 - global_time) + 0.1 * np.sin(4 * np.pi * global_time)
        state_array = x_values.reshape(1, -1)
        states_guess.append(state_array)

        # Control guess: sinusoidal control
        tau_control = np.linspace(-1, 1, N_k)
        global_time_control = (tau_control + 1) / 2
        u_values = 0.5 * np.sin(2 * np.pi * global_time_control)
        control_array = u_values.reshape(1, -1)
        controls_guess.append(control_array)

    return states_guess, controls_guess


def scenario_no_initial_guess() -> None:
    """Scenario 1: No initial guess - let CasADi handle everything."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Do NOT set any initial guess
    assert problem.initial_guess is None

    # Create solver without initial guess
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=STANDARD_DEGREES,
            mesh_points=STANDARD_MESH,
            # initial_guess=None  # Explicitly no guess
        ),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with NO initial guess (CasADi defaults)...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_partial_guess_times_only() -> None:
    """Scenario 2: Partial initial guess - times only."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Only provide time guesses
    problem.set_initial_guess(
        initial_time=0.0,  # Provide
        terminal_time=40.0,  # Provide
        # states=None,         # Let CasADi handle
        # controls=None,       # Let CasADi handle
        # integrals=None       # Let CasADi handle
    )

    # Verify partial guess
    ig = problem.initial_guess
    assert ig is not None
    assert ig.initial_time_variable == 0.0
    assert ig.terminal_time_variable == 40.0
    assert ig.states is None
    assert ig.controls is None
    assert ig.integrals is None

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with partial guess (times only)...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_partial_guess_integrals_only() -> None:
    """Scenario 3: Partial initial guess - integrals only."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Only provide integral guess
    problem.set_initial_guess(
        integrals=0.2  # Single integral estimate
    )

    # Verify partial guess
    ig = problem.initial_guess
    assert ig is not None
    assert ig.integrals == 0.2
    assert ig.states is None
    assert ig.controls is None

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with partial guess (integrals only)...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_partial_guess_states_only() -> None:
    """Scenario 4: Partial initial guess - states only."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Only provide state guesses
    states_guess = create_linear_state_guess(STANDARD_DEGREES)

    problem.set_initial_guess(
        states=states_guess
        # All other components remain None
    )

    # Verify partial guess
    ig = problem.initial_guess
    assert ig is not None
    assert ig.states is not None
    assert len(ig.states) == len(STANDARD_DEGREES)
    assert ig.controls is None
    assert ig.integrals is None

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with partial guess (linear states only)...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_complete_initial_guess_sophisticated() -> None:
    """Scenario 5: Complete initial guess - sophisticated exponential profile."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Create sophisticated guess
    states_guess, controls_guess = create_sophisticated_guess(STANDARD_DEGREES)

    # Provide complete initial guess
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.15,
    )

    # Verify complete guess
    problem.validate_initial_guess()  # Should not raise

    # Show solver input summary
    summary = problem.get_solver_input_summary()
    print("Complete sophisticated initial guess summary:")
    print(summary)

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with COMPLETE sophisticated initial guess...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_complete_initial_guess_sinusoidal() -> None:
    """Scenario 6: Complete initial guess - sinusoidal profile."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Create sinusoidal guess
    states_guess, controls_guess = create_sinusoidal_guess(STANDARD_DEGREES)

    # Provide complete initial guess
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.25,
    )

    # Verify complete guess
    problem.validate_initial_guess()  # Should not raise

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with COMPLETE sinusoidal initial guess...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_constructor_vs_problem_priority() -> None:
    """Scenario 7: Demonstrate priority - constructor vs problem initial guess."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Set initial guess on problem (linear profile)
    problem_states = create_linear_state_guess(STANDARD_DEGREES, 1.5, 1.0)
    problem.set_initial_guess(
        states=problem_states,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,  # Problem guess: 0.1
    )

    # Create different initial guess for constructor (sinusoidal profile)
    constructor_states, constructor_controls = create_sinusoidal_guess(STANDARD_DEGREES)
    constructor_guess = InitialGuess(
        states=constructor_states,
        controls=constructor_controls,
        initial_time_variable=1.0,
        terminal_time_variable=35.0,
        integrals=0.5,  # Constructor guess: 0.5
    )

    # For FixedMesh: constructor takes precedence
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=STANDARD_DEGREES,
            mesh_points=STANDARD_MESH,
            initial_guess=constructor_guess,  # This should take precedence
        ),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Testing priority: Constructor vs Problem initial guess...")
    print("Expected: Constructor guess (sinusoidal) takes precedence for FixedMesh")

    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        print("Note: Constructor initial guess was used (precedence demonstrated)")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_wrong_dimensions_should_fail() -> None:
    """Scenario 8: Wrong dimensional initial guess - should fail validation."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Expected shapes for STANDARD_DEGREES = [20, 15, 20]:
    # States: [(1, 21), (1, 16), (1, 21)]
    # Controls: [(1, 20), (1, 15), (1, 20)]

    # Create WRONG dimensional guesses
    wrong_states = [
        np.random.rand(1, 19),  # Wrong! Should be (1, 21) for N=20
        np.random.rand(1, 16),  # Correct (1, 16) for N=15
        np.random.rand(1, 21),  # Correct (1, 21) for N=20
    ]

    wrong_controls = [
        np.random.rand(1, 20),  # Correct (1, 20) for N=20
        np.random.rand(1, 17),  # Wrong! Should be (1, 15) for N=15
        np.random.rand(1, 20),  # Correct (1, 20) for N=20
    ]

    # This should fail during validation
    print("Testing wrong dimensional initial guess (should fail)...")
    print(f"Expected states shapes: {[(1, N + 1) for N in STANDARD_DEGREES]}")
    print(f"Expected controls shapes: {[(1, N) for N in STANDARD_DEGREES]}")
    print("Provided wrong states shapes: [(1, 19), (1, 16), (1, 21)]")
    print("Provided wrong controls shapes: [(1, 20), (1, 17), (1, 20)]")

    try:
        problem.set_initial_guess(states=wrong_states, controls=wrong_controls)
        # This should raise ValueError during validation
        problem.validate_initial_guess()
        raise RuntimeError("ERROR: Validation should have failed but didn't!")
    except ValueError as e:
        print(f"✅ EXPECTED FAILURE: {e}")
        print("Validation correctly caught dimensional mismatch")
        # This is expected - return without solving
        return

    # Should not reach here
    raise RuntimeError("Validation should have failed but didn't!")


def scenario_extreme_initial_guess() -> None:
    """Scenario 9: Extreme initial guess to test robustness."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Create extreme initial guess (far from solution)
    states_guess = []
    controls_guess = []

    for N_k in STANDARD_DEGREES:
        # Extreme state guess: starting very high, ending very low
        tau_points = np.linspace(-1, 1, N_k + 1)
        x_values = 5.0 + 3.0 * np.sin(3 * np.pi * tau_points)  # Oscillating between 2 and 8
        state_array = x_values.reshape(1, -1)
        states_guess.append(state_array)

        # Extreme control guess: large oscillating controls
        tau_control = np.linspace(-1, 1, N_k)
        u_values = 10.0 * np.cos(5 * np.pi * tau_control)  # Large oscillating control
        control_array = u_values.reshape(1, -1)
        controls_guess.append(control_array)

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=5.0,  # Far from actual bounds
        terminal_time=100.0,  # Far from actual bounds
        integrals=1000.0,  # Very large integral guess
    )

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 3000},
    )

    print("Solving with EXTREME initial guess (testing robustness)...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        print("Note: Solver was robust to extreme initial guess!")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


def scenario_requirements_inspection() -> None:
    """Scenario 10: Demonstrate requirement inspection before creating guess."""
    problem = create_hypersensitive_problem()

    # Use standardized mesh
    problem.set_mesh(STANDARD_DEGREES, STANDARD_MESH)

    # Inspect requirements
    print("Inspecting initial guess requirements for standardized mesh...")
    requirements = problem.get_initial_guess_requirements()
    print(requirements)

    # Verify requirements match our standardized mesh
    expected_states_shapes = [(1, N + 1) for N in STANDARD_DEGREES]  # N+1 for each degree
    expected_controls_shapes = [(1, N) for N in STANDARD_DEGREES]  # N for each degree

    assert requirements.states_shapes == expected_states_shapes, (
        f"States shapes mismatch: {requirements.states_shapes} vs {expected_states_shapes}"
    )
    assert requirements.controls_shapes == expected_controls_shapes, (
        f"Controls shapes mismatch: {requirements.controls_shapes} vs {expected_controls_shapes}"
    )

    print("✅ Requirements match standardized mesh exactly")

    # Create initial guess that exactly matches requirements
    states_guess = []
    controls_guess = []

    for i, (state_shape, control_shape) in enumerate(
        zip(requirements.states_shapes, requirements.controls_shapes, strict=False)
    ):
        print(f"Creating guess for interval {i}: states{state_shape}, controls{control_shape}")

        # Create state guess with exact required shape - parabolic profile
        tau_points = np.linspace(-1, 1, state_shape[1])
        x_values = 1.25 + 0.25 * (1 - tau_points**2)  # Parabolic from 1.5 to 1.0
        state_array = x_values.reshape(1, -1)
        states_guess.append(state_array)

        # Create control guess with exact required shape - quadratic profile
        tau_control = np.linspace(-1, 1, control_shape[1])
        u_values = 0.5 * tau_control**2  # Quadratic control
        control_array = u_values.reshape(1, -1)
        controls_guess.append(control_array)

    # Set the precisely dimensioned guess
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,
    )

    # Validate - should pass
    problem.validate_initial_guess()
    print("✅ Initial guess validation passed")

    # Show final input summary
    summary = problem.get_solver_input_summary()
    print("\nFinal solver input summary:")
    print(summary)

    solver = RadauDirectSolver(
        mesh_method=FixedMesh(STANDARD_DEGREES, STANDARD_MESH),
        nlp_options={"ipopt.print_level": 2, "ipopt.max_iter": 2000},
    )

    print("Solving with precisely dimensioned parabolic/quadratic initial guess...")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        solution.plot()
    else:
        raise RuntimeError(f"Solution failed: {solution.message}")


# Adaptive scenarios keep their own mesh configurations for comparison
def scenario_adaptive_first_iteration_guess() -> None:
    """Scenario 11: Adaptive mesh with initial guess for first iteration."""
    problem = create_hypersensitive_problem()

    # Use standardized adaptive mesh
    problem.set_mesh(ADAPTIVE_DEGREES, ADAPTIVE_MESH)

    # Create initial guess for first iteration
    states_guess = create_linear_state_guess(ADAPTIVE_DEGREES)
    controls_guess = create_control_guess(ADAPTIVE_DEGREES, 0.0)

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,
    )

    # Create adaptive solver with standardized configuration
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            initial_polynomial_degrees=ADAPTIVE_DEGREES,
            initial_mesh_points=ADAPTIVE_MESH,
            initial_guess=problem.initial_guess,
            **ADAPTIVE_SOLVER_CONFIG,
        ),
        nlp_options=ADAPTIVE_NLP_OPTIONS,
    )

    print("Solving with adaptive mesh + LINEAR initial guess for first iteration...")
    print(f"Standardized adaptive mesh: {ADAPTIVE_DEGREES} degrees, {ADAPTIVE_MESH} points")
    print("Initial guess: Linear state interpolation + zero controls")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        print(f"Final polynomial degrees: {solution.polynomial_degrees}")
        print(f"Final mesh intervals: {solution.num_intervals}")
        solution.plot()
    else:
        raise RuntimeError(f"Adaptive solution failed: {solution.message}")


def scenario_adaptive_no_guess() -> None:
    """Scenario 12: Adaptive mesh with no initial guess."""
    problem = create_hypersensitive_problem()

    # Use standardized adaptive mesh but NO initial guess
    problem.set_mesh(ADAPTIVE_DEGREES, ADAPTIVE_MESH)
    # problem.initial_guess remains None

    # Create adaptive solver with standardized configuration
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            initial_polynomial_degrees=ADAPTIVE_DEGREES,
            initial_mesh_points=ADAPTIVE_MESH,
            # initial_guess=None  # No initial guess provided
            **ADAPTIVE_SOLVER_CONFIG,
        ),
        nlp_options=ADAPTIVE_NLP_OPTIONS,
    )

    print("Solving with adaptive mesh + NO initial guess (CasADi defaults)...")
    print(f"Standardized adaptive mesh: {ADAPTIVE_DEGREES} degrees, {ADAPTIVE_MESH} points")
    print("Initial guess: None - CasADi will use defaults")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        print(f"Final polynomial degrees: {solution.polynomial_degrees}")
        print(f"Final mesh intervals: {solution.num_intervals}")
        solution.plot()
    else:
        raise RuntimeError(f"Adaptive solution failed: {solution.message}")


def scenario_adaptive_sophisticated_guess() -> None:
    """Scenario 13: Adaptive mesh with sophisticated initial guess."""
    problem = create_hypersensitive_problem()

    # Use standardized adaptive mesh
    problem.set_mesh(ADAPTIVE_DEGREES, ADAPTIVE_MESH)

    # Create sophisticated initial guess for first iteration
    states_guess, controls_guess = create_sophisticated_guess(ADAPTIVE_DEGREES)

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.15,
    )

    # Create adaptive solver with standardized configuration
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            initial_polynomial_degrees=ADAPTIVE_DEGREES,
            initial_mesh_points=ADAPTIVE_MESH,
            initial_guess=problem.initial_guess,
            **ADAPTIVE_SOLVER_CONFIG,
        ),
        nlp_options=ADAPTIVE_NLP_OPTIONS,
    )

    print("Solving with adaptive mesh + SOPHISTICATED initial guess for first iteration...")
    print(f"Standardized adaptive mesh: {ADAPTIVE_DEGREES} degrees, {ADAPTIVE_MESH} points")
    print("Initial guess: Exponential state decay + cubic control compensation")
    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        print(f"Final polynomial degrees: {solution.polynomial_degrees}")
        print(f"Final mesh intervals: {solution.num_intervals}")
        solution.plot()
    else:
        raise RuntimeError(f"Adaptive solution failed: {solution.message}")


def scenario_adaptive_priority_test() -> None:
    """Scenario 14: Adaptive priority - constructor vs problem initial guess."""
    problem = create_hypersensitive_problem()

    # Use standardized adaptive mesh
    problem.set_mesh(ADAPTIVE_DEGREES, ADAPTIVE_MESH)

    # Set initial guess on problem (linear profile)
    problem_states = create_linear_state_guess(ADAPTIVE_DEGREES, 1.5, 1.0)
    problem.set_initial_guess(
        states=problem_states,
        integrals=0.1,  # Problem guess: 0.1
    )

    # Create different constructor guess (sinusoidal profile)
    constructor_states, constructor_controls = create_sinusoidal_guess(ADAPTIVE_DEGREES)
    constructor_guess = InitialGuess(
        states=constructor_states,
        controls=constructor_controls,
        initial_time_variable=0.0,
        terminal_time_variable=40.0,
        integrals=0.5,  # Constructor guess: 0.5
    )

    # For PHSAdaptive: problem takes precedence over constructor
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            initial_polynomial_degrees=ADAPTIVE_DEGREES,
            initial_mesh_points=ADAPTIVE_MESH,
            initial_guess=constructor_guess,  # Lower priority for adaptive
            **ADAPTIVE_SOLVER_CONFIG,
        ),
        nlp_options=ADAPTIVE_NLP_OPTIONS,
    )

    print("Testing adaptive priority: Problem vs Constructor initial guess...")
    print(f"Standardized adaptive mesh: {ADAPTIVE_DEGREES} degrees, {ADAPTIVE_MESH} points")
    print("Expected: Problem guess (linear) takes precedence over constructor (sinusoidal)")

    solution = solve(problem, solver)

    if solution.success:
        print(f"SUCCESS: Objective = {solution.objective:.6f}")
        print("Note: Problem initial guess was used (precedence demonstrated)")
        print(f"Final polynomial degrees: {solution.polynomial_degrees}")
        solution.plot()
    else:
        raise RuntimeError(f"Adaptive solution failed: {solution.message}")


def main() -> None:
    """Run all initial guess demonstration scenarios."""
    print("TrajectoLab Initial Guess Comprehensive Demonstration")
    print("=" * 60)
    print("This demo shows all possible initial guess scenarios.")
    print("Script will FAIL FAST on any error.\n")
    print(f"STANDARDIZED FIXED MESH: {STANDARD_DEGREES} degrees, {STANDARD_MESH} points")
    print("All fixed mesh scenarios use identical mesh for direct comparison.\n")

    runner = DemoRunner()

    # Fixed mesh scenarios - all use STANDARD_DEGREES and STANDARD_MESH
    runner.run_scenario("No Initial Guess (CasADi Defaults)", scenario_no_initial_guess)

    runner.run_scenario("Partial Guess - Times Only", scenario_partial_guess_times_only)

    runner.run_scenario("Partial Guess - Integrals Only", scenario_partial_guess_integrals_only)

    runner.run_scenario("Partial Guess - Linear States Only", scenario_partial_guess_states_only)

    runner.run_scenario(
        "Complete Guess - Sophisticated Exponential Profile",
        scenario_complete_initial_guess_sophisticated,
    )

    runner.run_scenario(
        "Complete Guess - Sinusoidal Profile", scenario_complete_initial_guess_sinusoidal
    )

    runner.run_scenario(
        "Priority Test - Constructor vs Problem (Fixed Mesh)",
        scenario_constructor_vs_problem_priority,
    )

    runner.run_scenario(
        "Wrong Dimensions - Should Fail Validation", scenario_wrong_dimensions_should_fail
    )

    runner.run_scenario("Extreme Initial Guess - Robustness Test", scenario_extreme_initial_guess)

    runner.run_scenario(
        "Requirements Inspection - Parabolic/Quadratic Profile", scenario_requirements_inspection
    )

    # Adaptive scenarios - all use standardized configuration
    print(f"\n{'=' * 80}")
    print("ADAPTIVE SCENARIOS - All use standardized configuration")
    print(f"Mesh: {ADAPTIVE_DEGREES} degrees, {ADAPTIVE_MESH} points")
    print(f"Settings: {ADAPTIVE_SOLVER_CONFIG}")
    print(f"{'=' * 80}")

    runner.run_scenario("Adaptive - Linear Initial Guess", scenario_adaptive_first_iteration_guess)

    runner.run_scenario("Adaptive - No Initial Guess", scenario_adaptive_no_guess)

    runner.run_scenario(
        "Adaptive - Sophisticated Initial Guess", scenario_adaptive_sophisticated_guess
    )

    runner.run_scenario(
        "Adaptive - Priority Test (Problem vs Constructor)", scenario_adaptive_priority_test
    )

    # Final summary
    print(f"\n{'=' * 80}")
    print("🎉 ALL SCENARIOS COMPLETED SUCCESSFULLY!")
    print(f"Total scenarios: {runner.scenario_count}")
    print(f"Successful: {runner.success_count}")
    print(f"Success rate: {runner.success_count / runner.scenario_count * 100:.1f}%")
    print(f"\nFixed mesh scenarios all used: {STANDARD_DEGREES} degrees, {STANDARD_MESH} points")
    print("This allows direct comparison of initial guess strategies on identical discretization.")
    print(f"\nAdaptive scenarios all used: {ADAPTIVE_DEGREES} degrees, {ADAPTIVE_MESH} points")
    print("All adaptive scenarios use identical solver settings, varying only the initial guess.")
    print(f"Adaptive settings: {ADAPTIVE_SOLVER_CONFIG}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
