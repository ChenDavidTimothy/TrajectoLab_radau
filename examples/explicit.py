# Import InitialGuess for constructor example
from trajectolab import InitialGuess

import numpy as np

from trajectolab import FixedMesh, PHSAdaptive, Problem, RadauDirectSolver, solve, InitialGuess


def create_hypersensitive_problem() -> Problem:
    """Create the hypersensitive optimal control problem."""
    problem = Problem("Hypersensitive Problem")

    # Define time variable
    t = problem.time(initial=0.0, final=40.0)

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


def create_explicit_initial_guess(
    polynomial_degrees: list[int], mesh_points: np.ndarray, num_states: int, num_controls: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Create explicit initial guess arrays for all intervals.
    NASA engineers must specify every value.
    """
    states_guess = []
    controls_guess = []

    for k, N_k in enumerate(polynomial_degrees):
        # State trajectory for interval k: (num_states, N_k + 1)
        # Example: linear interpolation from 1.5 to 1.0
        state_nodes = np.linspace(1.5, 1.0, N_k + 1)
        state_array = np.array([state_nodes])  # Shape: (1, N_k + 1)
        states_guess.append(state_array)

        # Control trajectory for interval k: (num_controls, N_k)
        # Example: constant control
        control_array = np.zeros((num_controls, N_k))
        controls_guess.append(control_array)

    return states_guess, controls_guess


def solve_with_fixed_mesh():
    """Solve with fixed mesh - complete explicit control."""
    print("\n=== Fixed Mesh Solver (with initial guess via problem) ===")

    # Create problem
    problem = create_hypersensitive_problem()

    # Explicitly define mesh - NO DEFAULTS
    polynomial_degrees = [20, 8, 20]
    mesh_points = np.array([-1.0, -1/3, 1/3, 1.0])


    # Set mesh explicitly
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Initial guess via problem.set_initial_guess() - OPTIONAL
    states_guess, controls_guess = create_explicit_initial_guess(
        polynomial_degrees, mesh_points, num_states=1, num_controls=1
    )

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,  # Initial guess for the integral
    )

    # Verify what will be sent to solver
    summary = problem.get_solver_input_summary()
    print("Solver input summary:")
    print(summary)

    # Create solver with explicit mesh method (no initial guess needed here)
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=polynomial_degrees,
            mesh_points=mesh_points,
            # initial_guess=None,  # Not needed - using problem.set_initial_guess()
        ),
        nlp_options={"ipopt.print_level": 2},
    )

    # Solve
    solution = solve(problem, solver)

    if solution.success:
        print(f"Fixed mesh success! Objective: {solution.objective:.6f}")
        solution.plot()
    else:
        print(f"Fixed mesh failed: {solution.message}")


def solve_with_fixed_mesh_via_constructor():
    """Solve with fixed mesh - initial guess via FixedMesh constructor."""
    print("\n=== Fixed Mesh Solver (with initial guess via constructor) ===")

    # Create problem
    problem = create_hypersensitive_problem()

    # Explicitly define mesh
    polynomial_degrees = [20, 8, 20]
    mesh_points = np.array([-1.0, -1/3, 1/3, 1.0])

    # Create initial guess for the FixedMesh constructor
    states_guess, controls_guess = create_explicit_initial_guess(
        polynomial_degrees, mesh_points, num_states=1, num_controls=1
    )

    initial_guess = InitialGuess(
        initial_time_variable=0.0,
        terminal_time_variable=40.0,
        states=states_guess,
        controls=controls_guess,
        integrals=0.1,
    )

    # Set mesh explicitly
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Don't set initial guess on problem - will use constructor's guess

    # Create solver with initial guess in constructor
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=polynomial_degrees,
            mesh_points=mesh_points,
            initial_guess=initial_guess,  # Via constructor
        ),
        nlp_options={"ipopt.print_level": 2},
    )

    # Solve
    solution = solve(problem, solver)

    if solution.success:
        print(f"Fixed mesh (constructor guess) success! Objective: {solution.objective:.6f}")
    else:
        print(f"Fixed mesh (constructor guess) failed: {solution.message}")


def solve_with_fixed_mesh_no_guess():
    """Solve with fixed mesh - no initial guess (CasADi defaults)."""
    print("\n=== Fixed Mesh Solver (No Initial Guess) ===")

    # Create problem
    problem = create_hypersensitive_problem()

    # Explicitly define mesh - NO DEFAULTS
    polynomial_degrees = [20, 8, 20]
    mesh_points = np.array([-1.0, -1/3, 1/3, 1.0])

    # Set mesh explicitly
    problem.set_mesh(polynomial_degrees, mesh_points)

    # NO initial guess - CasADi will handle it
    # problem.set_initial_guess(...)  # Not called - completely optional

    # Verify what will be sent to solver
    summary = problem.get_solver_input_summary()
    print("Solver input summary (no guess provided):")
    print(summary)

    # Create solver
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=polynomial_degrees,
            mesh_points=mesh_points,
        ),
        nlp_options={"ipopt.print_level": 2},
    )

    # Solve
    solution = solve(problem, solver)

    if solution.success:
        print(f"Fixed mesh (no guess) success! Objective: {solution.objective:.6f}")
    else:
        print(f"Fixed mesh (no guess) failed: {solution.message}")


def solve_with_partial_guess():
    """Solve with partial initial guess - only some variables specified."""
    print("\n=== Fixed Mesh Solver (Partial Initial Guess) ===")

    # Create problem
    problem = create_hypersensitive_problem()

    # Explicitly define mesh
    polynomial_degrees = [20, 8, 20]
    mesh_points = np.array([-1.0, -1/3, 1/3, 1.0])

    # Set mesh explicitly
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Partial initial guess - only provide time and integral guesses
    problem.set_initial_guess(
        # states=None,          # Not provided - CasADi will use defaults
        # controls=None,        # Not provided - CasADi will use defaults
        initial_time=0.0,       # Provided
        terminal_time=40.0,     # Provided
        integrals=0.5,          # Provided
    )

    # Verify what will be sent to solver
    summary = problem.get_solver_input_summary()
    print("Solver input summary (partial guess):")
    print(summary)

    # Create solver
    solver = RadauDirectSolver(
        mesh_method=FixedMesh(
            polynomial_degrees=polynomial_degrees,
            mesh_points=mesh_points,
        ),
        nlp_options={"ipopt.print_level": 2},
    )

    # Solve
    solution = solve(problem, solver)

    if solution.success:
        print(f"Partial guess success! Objective: {solution.objective:.6f}")
    else:
        print(f"Partial guess failed: {solution.message}")


def solve_with_adaptive_mesh():
    """Solve with adaptive mesh - explicit first iteration, auto-propagation after."""
    print("\n=== Adaptive Mesh Solver ===")

    # Create problem
    problem = create_hypersensitive_problem()

    # Explicitly define INITIAL mesh for adaptive algorithm
    initial_polynomial_degrees = [4, 4, 4]
    initial_mesh_points = np.array([-1.0, -0.5, 0.0, 1.0])

    # Set initial mesh explicitly
    problem.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # Initial guess is OPTIONAL for adaptive too
    # Example: provide partial initial guess for first iteration
    states_guess, controls_guess = create_explicit_initial_guess(
        initial_polynomial_degrees, initial_mesh_points, num_states=1, num_controls=1
    )

    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
        integrals=0.1,  # Provide integral guess
    )

    # Verify what will be sent to solver for first iteration
    summary = problem.get_solver_input_summary()
    print("First iteration solver input:")
    print(summary)

    # Create adaptive solver
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            # Required parameters - no defaults
            initial_polynomial_degrees=initial_polynomial_degrees,
            initial_mesh_points=initial_mesh_points,
            # Optional initial guess - if not provided, CasADi uses defaults for first iteration
            initial_guess=problem.initial_guess,
            # Optional parameters with explicit values
            error_tolerance=1e-3,
            max_iterations=10,
            min_polynomial_degree=4,
            max_polynomial_degree=16,
        ),
        nlp_options={"ipopt.print_level": 2},
    )

    # Solve - adaptive algorithm will handle subsequent iterations automatically
    solution = solve(problem, solver)

    if solution.success:
        print(f"Adaptive success! Objective: {solution.objective:.6f}")
        print(f"Final mesh: {solution.polynomial_degrees}")
        print(f"Final mesh points: {np.array2string(solution.mesh_points, precision=3)}")
        solution.plot()
    else:
        print(f"Adaptive failed: {solution.message}")


def solve_with_adaptive_mesh_no_guess():
    """Solve with adaptive mesh - no initial guess for first iteration."""
    print("\n=== Adaptive Mesh Solver (No Initial Guess) ===")

    # Create problem
    problem = create_hypersensitive_problem()

    # Explicitly define INITIAL mesh for adaptive algorithm
    initial_polynomial_degrees = [4, 4, 4]
    initial_mesh_points = np.array([-1.0, -0.5, 0.0, 1.0])

    # Set initial mesh explicitly
    problem.set_mesh(initial_polynomial_degrees, initial_mesh_points)

    # NO initial guess for first iteration - let CasADi handle it

    # Create adaptive solver without initial guess
    solver = RadauDirectSolver(
        mesh_method=PHSAdaptive(
            initial_polynomial_degrees=initial_polynomial_degrees,
            initial_mesh_points=initial_mesh_points,
            # initial_guess=None,  # Not provided - optional
            error_tolerance=1e-3,
            max_iterations=10,
        ),
        nlp_options={"ipopt.print_level": 2},
    )

    # Solve
    solution = solve(problem, solver)

    if solution.success:
        print(f"Adaptive (no guess) success! Objective: {solution.objective:.6f}")
        print(f"Final mesh: {solution.polynomial_degrees}")
    else:
        print(f"Adaptive (no guess) failed: {solution.message}")


def demonstrate_error_handling():
    """Demonstrate clear error messages for invalid configurations."""
    print("\n=== Error Handling Demonstration ===")

    problem = create_hypersensitive_problem()

    # Example 1: Try to solve without setting mesh
    try:
        solver = RadauDirectSolver(
            mesh_method=FixedMesh(
                polynomial_degrees=[5],
                mesh_points=[-1.0, 1.0],
            )
        )
        solution = solve(problem, solver)
    except ValueError as e:
        print(f"Expected error (no initial guess): {e}")

    # Example 2: Try to create solver without mesh method
    try:
        solver = RadauDirectSolver()  # No mesh_method provided
    except ValueError as e:
        print(f"Expected error (no mesh method): {e}")

    # Example 3: Try to solve without providing solver
    try:
        solution = solve(problem)  # No solver provided
    except ValueError as e:
        print(f"Expected error (no solver): {e}")

    # Example 4: Try adaptive without required initial mesh (corrected)
    try:
        solver = RadauDirectSolver(
            mesh_method=PHSAdaptive(
                # Missing required parameters
                # initial_polynomial_degrees=[4, 4],
                # initial_mesh_points=[-1.0, 0.0, 1.0],
            )
        )
    except TypeError as e:
        print(f"Expected error (adaptive missing required mesh config): {e}")
    except ValueError as e:
        print(f"Expected error (adaptive missing required mesh config): {e}")


def demonstrate_requirements_inspection():
    """Demonstrate how to inspect requirements before solving."""
    print("\n=== Requirements Inspection ===")

    problem = create_hypersensitive_problem()

    # Set mesh first
    polynomial_degrees = [5, 7]
    mesh_points = np.array([-1.0, 0.0, 1.0])
    problem.set_mesh(polynomial_degrees, mesh_points)

    # Inspect requirements
    requirements = problem.get_initial_guess_requirements()
    print("Initial guess requirements:")
    print(requirements)

    # Create initial guess that matches requirements exactly
    states_guess, controls_guess = create_explicit_initial_guess(
        polynomial_degrees, mesh_points, num_states=1, num_controls=1
    )

    # Set initial guess
    problem.set_initial_guess(
        states=states_guess,
        controls=controls_guess,
        initial_time=0.0,
        terminal_time=40.0,
    )

    # Validate that everything is correct
    try:
        problem.validate_initial_guess()
        print("✓ Initial guess validation passed")
    except ValueError as e:
        print(f"✗ Initial guess validation failed: {e}")

    # Show final solver input summary
    summary = problem.get_solver_input_summary()
    print("\nFinal solver input summary:")
    print(summary)


if __name__ == "__main__":
    print("NASA-Appropriate TrajectoLab Example")
    print("====================================")

    # Demonstrate explicit control
    solve_with_fixed_mesh()
    solve_with_fixed_mesh_via_constructor()
    solve_with_fixed_mesh_no_guess()
    solve_with_partial_guess()
    solve_with_adaptive_mesh()
    solve_with_adaptive_mesh_no_guess()

    # Show error handling
    demonstrate_error_handling()

    # Show requirements inspection
    demonstrate_requirements_inspection()

    print("\nExample completed successfully!")
