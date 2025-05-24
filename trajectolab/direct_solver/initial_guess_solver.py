"""
Initial guess application functions for the direct solver - USES CENTRALIZED VALIDATION.
All redundant validations removed since validate_initial_guess_structure() provides comprehensive validation.
"""

from typing import cast

from ..input_validation import set_integral_guess_values, validate_integral_values
from ..tl_types import CasadiOpti, FloatArray, ProblemProtocol
from .types_solver import VariableReferences


def apply_initial_guess(
    opti: CasadiOpti,
    variables: VariableReferences,
    problem: ProblemProtocol,
    num_mesh_intervals: int,
) -> None:
    """
    Apply initial guess to optimization variables using unified storage.

    NOTE: Assumes initial guess has been validated by validate_initial_guess_structure()
    at the entry point (validate_problem_ready_for_solving).
    """
    if problem.initial_guess is None:
        return

    ig = problem.initial_guess

    # Get variable counts (already validated)
    num_states, num_controls = problem.get_variable_counts()
    num_integrals = problem._num_integrals

    # Apply initial guess components - no need for redundant validation
    _apply_time_guesses(opti, variables, ig)
    _apply_state_guesses(opti, variables, ig, num_states, num_mesh_intervals, problem)
    _apply_control_guesses(opti, variables, ig, num_controls, num_mesh_intervals, problem)
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
    """
    Apply initial guess for state variables.

    NOTE: All validation removed since validate_initial_guess_structure() already validates:
    - Array types and dtypes
    - Array shapes and dimensions
    - Count consistency with mesh intervals
    """
    if ig.states is None:
        return

    # Apply global mesh node states
    for k in range(num_mesh_intervals):
        state_guess_k = cast(FloatArray, ig.states[k])

        # Set initial node guess (only once)
        if k == 0:
            opti.set_initial(variables.state_at_mesh_nodes[0], state_guess_k[:, 0])
        # Set terminal node guess
        opti.set_initial(variables.state_at_mesh_nodes[k + 1], state_guess_k[:, -1])

    # Apply interior state node guesses
    for k in range(num_mesh_intervals):
        interior_var = variables.interior_variables[k]
        if interior_var is not None:
            state_guess_k = cast(FloatArray, ig.states[k])
            num_interior_nodes = interior_var.shape[1]

            # Extract interior guess (shape validation already done)
            interior_guess = state_guess_k[:, 1 : 1 + num_interior_nodes]
            opti.set_initial(interior_var, interior_guess)


def _apply_control_guesses(
    opti: CasadiOpti,
    variables: VariableReferences,
    ig,
    num_controls: int,
    num_mesh_intervals: int,
    problem: ProblemProtocol,
) -> None:
    """
    Apply initial guess for control variables.

    NOTE: All validation removed since validate_initial_guess_structure() already validates:
    - Array types and dtypes
    - Array shapes and dimensions
    - Count consistency with mesh intervals
    """
    if ig.controls is None:
        return

    for k in range(num_mesh_intervals):
        control_guess_k = cast(FloatArray, ig.controls[k])
        opti.set_initial(variables.control_variables[k], control_guess_k)


def _apply_integral_guesses(
    opti: CasadiOpti, variables: VariableReferences, ig, num_integrals: int
) -> None:
    """
    Apply initial guess for integral variables.

    Uses centralized validation from input_validation.py
    """
    if ig.integrals is not None and num_integrals > 0 and variables.integral_variables is not None:
        # Use centralized validation function
        validate_integral_values(ig.integrals, num_integrals)
        set_integral_guess_values(opti, variables.integral_variables, ig.integrals, num_integrals)
