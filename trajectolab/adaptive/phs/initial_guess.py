"""
Initial guess handling for the PHS adaptive algorithm.
NASA-appropriate: explicit control, no hidden assumptions.
"""

import numpy as np

from trajectolab.tl_types import FloatArray, InitialGuess, OptimalControlSolution, ProblemProtocol


__all__ = [
    "propagate_solution_to_new_mesh",
]


def propagate_solution_to_new_mesh(
    prev_solution: OptimalControlSolution,
    problem: ProblemProtocol,
    target_polynomial_degrees: list[int],
    target_mesh_points: FloatArray,
) -> InitialGuess:
    """
    Propagate solution from previous iteration to a new mesh.

    This function creates an initial guess for the current iteration by:
    1. Always propagating: time variables (t0, tf) and integral values
    2. Conditionally propagating trajectories:
       - If mesh structure is identical: propagate state/control trajectories exactly
       - If mesh structure changed: use default trajectories but keep times/integrals

    Args:
        prev_solution: Solution from previous adaptive iteration
        problem: Current problem definition
        target_polynomial_degrees: Polynomial degrees for current iteration
        target_mesh_points: Mesh points for current iteration

    Returns:
        InitialGuess for current iteration

    Raises:
        ValueError: If previous solution is invalid
    """
    if not prev_solution.success:
        raise ValueError("Cannot propagate from unsuccessful previous solution")

    # Always propagate time variables and integrals
    t0_guess = prev_solution.initial_time_variable
    tf_guess = prev_solution.terminal_time_variable
    integrals_guess = prev_solution.integrals

    # Check if mesh structure is identical
    prev_degrees = prev_solution.num_collocation_nodes_list_at_solve_time
    prev_mesh = prev_solution.global_mesh_nodes_at_solve_time

    identical_mesh = False
    if prev_degrees is not None and prev_mesh is not None:
        identical_mesh = (
            len(target_polynomial_degrees) == len(prev_degrees)
            and all(
                target_deg == prev_deg
                for target_deg, prev_deg in zip(
                    target_polynomial_degrees, prev_degrees, strict=False
                )
            )
            and np.allclose(target_mesh_points, prev_mesh, atol=1e-12)
        )

    if identical_mesh:
        # Mesh structure is identical - propagate trajectories exactly
        print("    Mesh structure identical. Propagating trajectories from previous solution.")

        prev_states = prev_solution.solved_state_trajectories_per_interval
        prev_controls = prev_solution.solved_control_trajectories_per_interval

        if prev_states is None or prev_controls is None:
            raise ValueError("Previous solution missing trajectory data for propagation")

        # Use trajectories directly (they already match the target mesh)
        states_guess = [np.array(state_traj, dtype=np.float64) for state_traj in prev_states]
        controls_guess = [
            np.array(control_traj, dtype=np.float64) for control_traj in prev_controls
        ]

    else:
        # Mesh structure changed - cannot propagate trajectories safely
        print(
            "    Mesh structure changed. Using default trajectories (times/integrals propagated)."
        )

        # Create default trajectory arrays for new mesh
        num_states = len(problem._states)
        num_controls = len(problem._controls)

        states_guess = []
        controls_guess = []

        for _k, N_k in enumerate(target_polynomial_degrees):
            # Default state trajectory: zeros
            state_traj = np.zeros((num_states, N_k + 1), dtype=np.float64)
            states_guess.append(state_traj)

            # Default control trajectory: zeros
            control_traj = np.zeros((num_controls, N_k), dtype=np.float64)
            controls_guess.append(control_traj)

    return InitialGuess(
        initial_time_variable=t0_guess,
        terminal_time_variable=tf_guess,
        states=states_guess,
        controls=controls_guess,
        integrals=integrals_guess,
    )
