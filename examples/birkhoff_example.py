"""
Example usage of MAPTOR Birkhoff solver.

This demonstrates how to use the new solve_birkhoff_mesh function
with user-defined grid points following the Birkhoff interpolation theory.
"""

import casadi as ca
import numpy as np

import maptor as mtor


def example_simple_birkhoff_solve():
    """Example: Simple integrator problem using Birkhoff method."""

    # Create problem
    problem = mtor.Problem("Birkhoff Simple Integrator")
    phase = problem.set_phase(1)

    # Define variables
    t = phase.time(initial=0.0, final=2.0)
    x = phase.state("position", initial=0.0, final=4.0)
    v = phase.state("velocity", initial=0.0)
    u = phase.control("force", boundary=(-1.0, 1.0))

    # Dynamics: simple double integrator
    phase.dynamics({x: v, v: u})

    # Objective: minimize control effort
    control_effort = phase.add_integral(u**2)
    problem.minimize(control_effort)

    # Define grid points for Birkhoff method
    # Using 5 equally spaced points on [-1, 1]
    grid_points = {1: (-1.0, -0.5, 0.0, 0.5, 1.0)}

    # Solve using Birkhoff method
    solution = mtor.solve_birkhoff_mesh(
        problem, grid_points_per_phase=grid_points, show_summary=True
    )

    return solution


def example_birkhoff_with_lgl_grid():
    """Example: Using Legendre-Gauss-Lobatto grid points."""

    problem = mtor.Problem("Birkhoff with LGL Grid")
    phase = problem.set_phase(1)

    # Brachistochrone-like problem
    t = phase.time(initial=0.0)
    x = phase.state("x", initial=0.0, final=1.0)
    y = phase.state("y", initial=1.0, final=0.0)
    v = phase.state("v", initial=0.0, boundary=(0.0, None))
    theta = phase.control("theta")

    # Dynamics
    phase.dynamics({x: v * ca.sin(theta), y: v * ca.cos(theta), v: 9.81 * ca.cos(theta)})

    # Minimize time
    problem.minimize(t.final)

    # Generate LGL-like grid (for demonstration - in practice use scipy)
    # This is a simplified approximation
    n_points = 7
    grid_interior = np.cos(np.pi * np.arange(1, n_points - 1) / (n_points - 1))
    lgl_grid = tuple(np.concatenate([[-1.0], grid_interior, [1.0]]))

    grid_points = {1: lgl_grid}

    solution = mtor.solve_birkhoff_mesh(
        problem, grid_points_per_phase=grid_points, nlp_options={"ipopt.tol": 1e-8}
    )

    return solution


def example_multiphase_birkhoff():
    """Example: Multiphase problem with different grids per phase."""

    problem = mtor.Problem("Multiphase Birkhoff")

    # Phase 1: Acceleration
    boost = problem.set_phase(1)
    t1 = boost.time(initial=0.0, final=1.0)
    h1 = boost.state("altitude", initial=0.0)
    v1 = boost.state("velocity", initial=0.0)
    u1 = boost.control("thrust", boundary=(0.0, 2.0))

    boost.dynamics({h1: v1, v1: u1 - 1.0})  # gravity = 1.0

    # Phase 2: Coast with continuity
    coast = problem.set_phase(2)
    t2 = coast.time(initial=t1.final, final=3.0)
    h2 = coast.state("altitude", initial=h1.final)
    v2 = coast.state("velocity", initial=v1.final)

    coast.dynamics({h2: v2, v2: -1.0})  # no thrust, just gravity

    # Maximize final altitude
    problem.minimize(-h2.final)

    # Different grid densities for each phase
    grid_points = {
        1: (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0),  # 6 points for boost phase
        2: (-1.0, -0.33, 0.33, 1.0),  # 4 points for coast phase
    }

    solution = mtor.solve_birkhoff_mesh(problem, grid_points_per_phase=grid_points)

    return solution


if __name__ == "__main__":
    print("MAPTOR Birkhoff Solver Examples")
    print("=" * 40)

    # Run examples
    print("\n1. Simple integrator with Birkhoff method:")
    sol1 = example_simple_birkhoff_solve()
    if sol1.status["success"]:
        print(f"   ✓ Solved successfully! Objective: {sol1.status['objective']:.6f}")

    print("\n2. Using LGL-like grid:")
    sol2 = example_birkhoff_with_lgl_grid()
    if sol2.status["success"]:
        print(f"   ✓ Solved successfully! Final time: {sol2.status['objective']:.6f}")

    print("\n3. Multiphase with different grids:")
    sol3 = example_multiphase_birkhoff()
    if sol3.status["success"]:
        print(f"   ✓ Solved successfully! Final altitude: {-sol3.status['objective']:.6f}")
