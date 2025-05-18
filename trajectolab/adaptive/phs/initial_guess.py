"""
Initial guess generation for the PHS adaptive algorithm.
"""

import numpy as np

from trajectolab.direct_solver import InitialGuess, OptimalControlSolution
from trajectolab.tl_types import (
    FloatArray,
    FloatMatrix,
    ProblemProtocol,
)


__all__ = [
    "generate_robust_default_initial_guess",
    "propagate_guess_from_previous",
]


def generate_robust_default_initial_guess(
    problem: ProblemProtocol,
    collocation_nodes_list: list[int],
    initial_time_guess: float | None = None,
    terminal_time_guess: float | None = None,
    integral_values_guess: float | FloatArray | None = None,
) -> InitialGuess:
    """Generates a robust default initial guess with correct dimensions."""
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Get default values
    default_state = 0.0
    default_control = 0.0
    default_integral = 0.0

    # Get default values from problem.default_initial_guess_values if available
    default_guess_values = getattr(problem, "default_initial_guess_values", None)
    if default_guess_values:
        default_state = getattr(default_guess_values, "state", 0.0)
        default_control = getattr(default_guess_values, "control", 0.0)
        default_integral = getattr(default_guess_values, "integral", 0.0)

    # Initialize state and control trajectories
    states: list[FloatMatrix] = []
    controls: list[FloatMatrix] = []

    for _idx, Nk in enumerate(collocation_nodes_list):
        # State trajectory for this interval
        state_traj = np.full((num_states, Nk + 1), default_state, dtype=np.float64)
        states.append(state_traj)

        # Control trajectory for this interval
        if num_controls > 0:
            control_traj = np.full((num_controls, Nk), default_control, dtype=np.float64)
        else:
            control_traj = np.empty((0, Nk), dtype=np.float64)
        controls.append(control_traj)

    # Time variable guesses
    if initial_time_guess is None and problem.initial_guess:
        initial_time_guess = problem.initial_guess.initial_time_variable

    if terminal_time_guess is None and problem.initial_guess:
        terminal_time_guess = problem.initial_guess.terminal_time_variable

    # Integral guesses
    final_integral_guess = None
    if num_integrals > 0:
        if integral_values_guess is not None:
            final_integral_guess = integral_values_guess
        else:
            if problem.initial_guess and problem.initial_guess.integrals is not None:
                raw_guess = problem.initial_guess.integrals
            else:
                raw_guess = (
                    [default_integral] * num_integrals if num_integrals > 1 else default_integral
                )

            if num_integrals == 1:
                final_integral_guess = (
                    float(raw_guess)
                    if not isinstance(raw_guess, list | np.ndarray)
                    else float(raw_guess[0])
                )
            elif isinstance(raw_guess, list | np.ndarray) and len(raw_guess) == num_integrals:
                final_integral_guess = np.array(raw_guess, dtype=np.float64)
            else:
                final_integral_guess = np.full(num_integrals, default_integral, dtype=np.float64)

    return InitialGuess(
        initial_time_variable=initial_time_guess,
        terminal_time_variable=terminal_time_guess,
        states=states,
        controls=controls,
        integrals=final_integral_guess,
    )


def propagate_guess_from_previous(
    prev_solution: "OptimalControlSolution",
    problem: ProblemProtocol,
    target_nodes_list: list[int],
    target_mesh: FloatArray,
) -> InitialGuess:
    """Creates initial guess for current NLP, propagating from previous solution."""
    t0_prop = prev_solution.initial_time_variable
    tf_prop = prev_solution.terminal_time_variable
    integrals_prop = prev_solution.integrals

    # Generate default guess with propagated time and integral values
    guess = generate_robust_default_initial_guess(
        problem,
        target_nodes_list,
        initial_time_guess=t0_prop,
        terminal_time_guess=tf_prop,
        integral_values_guess=integrals_prop,
    )

    # Check if mesh structure is identical to previous solution
    prev_nodes = prev_solution.num_collocation_nodes_list_at_solve_time
    prev_mesh = prev_solution.global_mesh_nodes_at_solve_time

    can_propagate_trajectories = False
    if prev_nodes is not None and prev_mesh is not None:
        if np.array_equal(target_nodes_list, prev_nodes) and np.allclose(target_mesh, prev_mesh):
            can_propagate_trajectories = True

    if can_propagate_trajectories:
        print(
            "  Mesh structure identical to previous. Propagating state/control trajectories directly."
        )
        prev_states = prev_solution.solved_state_trajectories_per_interval
        prev_controls = prev_solution.solved_control_trajectories_per_interval

        # Propagate state trajectories if available
        if prev_states and len(prev_states) == len(target_nodes_list):
            guess.states = prev_states
        else:
            print("    Warning: Previous states mismatch or missing. Using default states.")

        # Propagate control trajectories if available
        if prev_controls and len(prev_controls) == len(target_nodes_list):
            guess.controls = prev_controls
        else:
            print("    Warning: Previous controls mismatch or missing. Using default controls.")
    else:
        print(
            "  Mesh structure changed. Using robust default for state/control trajectories (times/integrals propagated)."
        )

    return guess
