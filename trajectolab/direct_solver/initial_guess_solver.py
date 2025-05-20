"""
Initial guess application functions for the direct solver.
"""

from typing import cast

from ..input_validation import validate_and_set_integral_guess
from ..tl_types import CasadiOpti, FloatMatrix, ProblemProtocol
from .types_solver import VariableReferences


def apply_initial_guess(
    opti: CasadiOpti,
    variables: VariableReferences,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> None:
    """
    Apply initial guess to optimization variables.

    Args:
        opti: CasADi optimization object
        variables: Container with all variable references
        problem: Problem definition
        num_mesh_intervals: Number of mesh intervals

    Note:
        If no initial guess is provided, CasADi will use its default values.
    """
    if problem.initial_guess is None:
        return

    ig = problem.initial_guess
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Apply time variable guesses
    _apply_time_guesses(opti, variables, ig)

    # Apply state guesses
    _apply_state_guesses(opti, variables, ig, num_states, num_mesh_intervals, problem)

    # Apply control guesses
    _apply_control_guesses(opti, variables, ig, num_controls, num_mesh_intervals, problem)

    # Apply integral guesses
    _apply_integral_guesses(opti, variables, ig, num_integrals)


def _apply_time_guesses(opti: CasadiOpti, variables: VariableReferences, ig) -> None:
    """Apply initial guess for time variables."""
    if ig.initial_time_variable is not None:
        opti.set_initial(variables.initial_time, ig.initial_time_variable)
    if ig.terminal_time_variable is not None:
        opti.set_initial(variables.terminal_time, ig.terminal_time_variable)


def _apply_state_guesses(
    opti: CasadiOpti,
    variables: VariableReferences,
    ig,
    num_states: int,
    num_mesh_intervals: int,
    problem: ProblemProtocol,
) -> None:
    """Apply initial guess for state variables."""
    if ig.states is None:
        return

    if len(ig.states) != num_mesh_intervals:
        raise ValueError(
            f"States guess must have {num_mesh_intervals} arrays, got {len(ig.states)}"
        )

    # Apply global mesh node states
    for k in range(num_mesh_intervals):
        state_guess_k = cast(FloatMatrix, ig.states[k])
        expected_shape = (num_states, problem.collocation_points_per_interval[k] + 1)
        if state_guess_k.shape != expected_shape:
            raise ValueError(
                f"State guess for interval {k} has shape {state_guess_k.shape}, "
                f"expected {expected_shape}"
            )

        # Set initial node guess (only once)
        if k == 0:
            opti.set_initial(variables.state_at_mesh_nodes[0], state_guess_k[:, 0])
        # Set terminal node guess
        opti.set_initial(variables.state_at_mesh_nodes[k + 1], state_guess_k[:, -1])

    # Apply interior state node guesses
    for k in range(num_mesh_intervals):
        interior_var = variables.interior_variables[k]
        if interior_var is not None:
            state_guess_k = cast(FloatMatrix, ig.states[k])
            num_interior_nodes = interior_var.shape[1]

            if state_guess_k.shape[1] >= num_interior_nodes + 2:
                interior_guess = state_guess_k[:, 1 : 1 + num_interior_nodes]
                opti.set_initial(interior_var, interior_guess)
            else:
                raise ValueError(
                    f"State guess for interval {k} has {state_guess_k.shape[1]} nodes, "
                    f"but needs at least {num_interior_nodes + 2} for interior nodes"
                )


def _apply_control_guesses(
    opti: CasadiOpti,
    variables: VariableReferences,
    ig,
    num_controls: int,
    num_mesh_intervals: int,
    problem: ProblemProtocol,
) -> None:
    """Apply initial guess for control variables."""
    if ig.controls is None:
        return

    if len(ig.controls) != num_mesh_intervals:
        raise ValueError(
            f"Controls guess must have {num_mesh_intervals} arrays, got {len(ig.controls)}"
        )

    for k in range(num_mesh_intervals):
        control_guess_k = cast(FloatMatrix, ig.controls[k])
        expected_shape = (num_controls, problem.collocation_points_per_interval[k])

        if control_guess_k.shape != expected_shape:
            raise ValueError(
                f"Control guess for interval {k} has shape {control_guess_k.shape}, "
                f"expected {expected_shape}"
            )

        opti.set_initial(variables.control_variables[k], control_guess_k)


def _apply_integral_guesses(
    opti: CasadiOpti, variables: VariableReferences, ig, num_integrals: int
) -> None:
    """Apply initial guess for integral variables."""
    if ig.integrals is not None and num_integrals > 0 and variables.integral_variables is not None:
        validate_and_set_integral_guess(
            opti, variables.integral_variables, ig.integrals, num_integrals
        )


def apply_scaled_initial_guess(
    opti: CasadiOpti,
    variables: VariableReferences,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
    scaling_manager,  # Type hint left off for compatibility
) -> None:
    """
    Apply initial guess with scaling.

    When variables in the NLP are in the scaled space, we need to scale
    the user's unscaled initial guess.

    Args:
        opti: CasADi optimization object
        variables: Variable references
        problem: Problem definition
        num_mesh_intervals: Number of mesh intervals
        scaling_manager: Scaling manager for variable scaling
    """
    if problem.initial_guess is None:
        return

    ig = problem.initial_guess
    num_states = len(problem._states)
    num_controls = len(problem._controls)
    num_integrals = problem._num_integrals

    # Apply time variable guesses (unscaled)
    if ig.initial_time_variable is not None:
        opti.set_initial(variables.initial_time, ig.initial_time_variable)
    if ig.terminal_time_variable is not None:
        opti.set_initial(variables.terminal_time, ig.terminal_time_variable)

    # Apply state guesses with scaling
    if ig.states is not None and len(ig.states) == num_mesh_intervals:
        # Apply mesh node guesses
        for k in range(num_mesh_intervals + 1):
            if k < num_mesh_intervals:
                # Initial node of interval k
                unscaled_init_state = ig.states[k][:, 0]
                scaled_init_state = scaling_manager.scale_vector(unscaled_init_state, is_state=True)
                opti.set_initial(variables.state_at_mesh_nodes[k], scaled_init_state)
            elif k > 0 and k - 1 < len(ig.states):
                # Final node of last interval
                unscaled_final_state = ig.states[k - 1][:, -1]
                scaled_final_state = scaling_manager.scale_vector(
                    unscaled_final_state, is_state=True
                )
                opti.set_initial(variables.state_at_mesh_nodes[k], scaled_final_state)

        # Apply interior state node guesses
        for k in range(num_mesh_intervals):
            interior_var = variables.interior_variables[k]
            if interior_var is not None and k < len(ig.states):
                state_guess_k = ig.states[k]
                num_interior_nodes = interior_var.shape[1]

                if state_guess_k.shape[1] >= num_interior_nodes + 2:
                    unscaled_interior = state_guess_k[:, 1 : 1 + num_interior_nodes]
                    scaled_interior = scaling_manager.scale_state(unscaled_interior)
                    opti.set_initial(interior_var, scaled_interior)

    # Apply control guesses with scaling
    if ig.controls is not None and len(ig.controls) == num_mesh_intervals:
        for k in range(num_mesh_intervals):
            if k < len(ig.controls):
                unscaled_control = ig.controls[k]
                scaled_control = scaling_manager.scale_control(unscaled_control)
                opti.set_initial(variables.control_variables[k], scaled_control)

    # Apply integral guesses (unscaled)
    if ig.integrals is not None and num_integrals > 0 and variables.integral_variables is not None:
        validate_and_set_integral_guess(
            opti, variables.integral_variables, ig.integrals, num_integrals
        )
