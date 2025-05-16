"""Initialization utilities for adaptive mesh refinement."""

import logging
from typing import List, Optional, Sequence, Union

import numpy as np

from trajectolab.trajectolab_types import (
    InitialGuess,
    NumControls,
    NumIntegrals,
    NumStates,
    OptimalControlProblem,
    OptimalControlSolution,
    _FloatArray,
    _Vector,
)

logger = logging.getLogger(__name__)


def generate_robust_default_initial_guess(
    problem: OptimalControlProblem,
    collocation_nodes_list: List[int],
    initial_time_guess: Optional[float] = None,
    terminal_time_guess: Optional[float] = None,
    integral_values_guess: Optional[Union[_Vector, float]] = None,
) -> InitialGuess:
    """Generate a robust default initial guess.

    Args:
        problem: Optimal control problem definition
        collocation_nodes_list: List of collocation nodes per interval
        initial_time_guess: Optional initial time guess
        terminal_time_guess: Optional terminal time guess
        integral_values_guess: Optional integral values guess

    Returns:
        Initial guess object
    """
    num_states: NumStates = problem.num_states
    num_controls: NumControls = problem.num_controls
    num_integrals: NumIntegrals = problem.num_integrals

    default_state_val = getattr(problem.default_initial_guess_values, "state", 0.0)
    default_control_val = getattr(problem.default_initial_guess_values, "control", 0.0)

    states_guess_list: List[_FloatArray] = []
    controls_guess_list: List[_FloatArray] = []

    for Nk_val in collocation_nodes_list:
        # State guess: (num_states, Nk_val + 1)
        state_traj_guess = np.full((num_states, Nk_val + 1), default_state_val, dtype=np.float64)
        states_guess_list.append(state_traj_guess)

        # Control guess: (num_controls, Nk_val)
        if num_controls > 0:
            control_traj_guess = np.full(
                (num_controls, Nk_val), default_control_val, dtype=np.float64
            )
        else:  # No controls
            control_traj_guess = np.empty((0, Nk_val), dtype=np.float64)
        controls_guess_list.append(control_traj_guess)

    # Time guesses
    final_t0_guess: Optional[float] = initial_time_guess
    if final_t0_guess is None and problem.initial_guess:
        final_t0_guess = problem.initial_guess.initial_time_variable

    final_tf_guess: Optional[float] = terminal_time_guess
    if final_tf_guess is None and problem.initial_guess:
        final_tf_guess = problem.initial_guess.terminal_time_variable

    # Integral guesses
    final_integrals_guess: Optional[Union[float, Sequence[float]]] = None
    if num_integrals > 0:
        if integral_values_guess is not None:
            if isinstance(integral_values_guess, np.ndarray):
                final_integrals_guess = list(integral_values_guess.astype(np.float64))
            else:  # float
                final_integrals_guess = float(integral_values_guess)
        elif problem.initial_guess and problem.initial_guess.integrals is not None:
            raw_problem_integrals = problem.initial_guess.integrals
            if isinstance(raw_problem_integrals, np.ndarray):
                final_integrals_guess = [float(x) for x in raw_problem_integrals.astype(np.float64)]
            elif isinstance(raw_problem_integrals, (list, tuple)):
                final_integrals_guess = [float(x) for x in raw_problem_integrals]
            elif isinstance(raw_problem_integrals, float):
                final_integrals_guess = raw_problem_integrals
        else:  # Default value for integrals
            default_integral_val = getattr(problem.default_initial_guess_values, "integral", 0.0)
            if num_integrals == 1:
                final_integrals_guess = default_integral_val
            else:
                final_integrals_guess = [default_integral_val] * num_integrals

    return InitialGuess(
        initial_time_variable=final_t0_guess,
        terminal_time_variable=final_tf_guess,
        states=states_guess_list,
        controls=controls_guess_list,
        integrals=final_integrals_guess,
    )


def propagate_guess_from_previous(
    prev_solution: OptimalControlSolution,
    problem_for_new_nlp: OptimalControlProblem,
    target_collocation_nodes_list: List[int],
    target_mesh_global_tau: _FloatArray,
) -> InitialGuess:
    """Propagate initial guess from a previous solution.

    Args:
        prev_solution: Previous optimization solution
        problem_for_new_nlp: New optimal control problem definition
        target_collocation_nodes_list: Target list of collocation nodes per interval
        target_mesh_global_tau: Target global mesh points

    Returns:
        Initial guess object propagated from previous solution
    """
    t0_prop: Optional[float] = prev_solution.initial_time_variable
    tf_prop: Optional[float] = prev_solution.terminal_time_variable

    integrals_prop_for_guess: Optional[Union[_Vector, float]] = None
    if prev_solution.integrals is not None:
        if isinstance(prev_solution.integrals, np.ndarray):
            integrals_prop_for_guess = prev_solution.integrals.astype(np.float64)
        else:  # Should be float
            integrals_prop_for_guess = float(prev_solution.integrals)

    # Start with a robust default guess, then overwrite parts if possible
    current_guess = generate_robust_default_initial_guess(
        problem_for_new_nlp,
        target_collocation_nodes_list,
        initial_time_guess=t0_prop,
        terminal_time_guess=tf_prop,
        integral_values_guess=integrals_prop_for_guess,
    )

    prev_nodes_list = prev_solution.num_collocation_nodes_list_at_solve_time
    prev_mesh_list_type = prev_solution.global_mesh_nodes_at_solve_time

    can_propagate_trajectories = False
    if prev_nodes_list is not None and prev_mesh_list_type is not None:
        prev_mesh_np = np.array(prev_mesh_list_type, dtype=np.float64)
        # Check if mesh structure is identical
        if target_collocation_nodes_list == prev_nodes_list and np.allclose(
            target_mesh_global_tau, prev_mesh_np, atol=1e-9, rtol=1e-9
        ):
            can_propagate_trajectories = True

    if can_propagate_trajectories:
        logger.info(
            "  Mesh structure identical to previous. Propagating state/control trajectories."
        )
        if prev_solution.states and len(prev_solution.states) == len(target_collocation_nodes_list):
            current_guess.states = [s.astype(np.float64) for s in prev_solution.states]
        else:
            logger.warning(
                "    Warning: Previous states mismatch or missing. Using default states."
            )

        if prev_solution.controls and len(prev_solution.controls) == len(
            target_collocation_nodes_list
        ):
            current_guess.controls = [c.astype(np.float64) for c in prev_solution.controls]
        else:
            logger.warning("    Warning: Previous controls mismatch or missing. Using default.")
    else:
        logger.info("  Mesh structure changed. Using robust default for state/control.")

    return current_guess
