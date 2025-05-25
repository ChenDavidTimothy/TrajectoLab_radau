"""
Centralized input validation for all user configuration parameters.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any, TypeVar

import casadi as ca
import numpy as np

from .exceptions import ConfigurationError, DataIntegrityError
from .tl_types import CasadiMX, CasadiOpti, FloatArray, InitialGuess, ProblemProtocol
from .utils.constants import MESH_TOLERANCE, MINIMUM_TIME_INTERVAL, ZERO_TOLERANCE


logger = logging.getLogger(__name__)
T = TypeVar("T")


# ============================================================================
# CONSOLIDATED CONFIGURATION VALIDATION FUNCTIONS
# ============================================================================


def validate_problem_ready_for_solving(problem: ProblemProtocol) -> None:
    """
    COMPREHENSIVE validation that problem is properly configured for solving.
    This should be called at ALL solver entry points.
    """
    # Validate basic problem structure
    validate_problem_basic_structure(problem)

    # Validate mesh configuration
    if not hasattr(problem, "_mesh_configured") or not problem._mesh_configured:
        raise ConfigurationError(
            "Problem mesh must be configured before solving",
            "Call problem.set_mesh(polynomial_degrees, mesh_points) first",
        )

    # Validate mesh details
    validate_mesh_configuration(
        problem.collocation_points_per_interval,
        problem.global_normalized_mesh_nodes,
        len(problem.collocation_points_per_interval),
    )

    # Validate problem completeness
    validate_problem_completeness(problem)

    # Validate initial guess if present
    if problem.initial_guess is not None:
        num_states, num_controls = problem.get_variable_counts()
        validate_initial_guess_structure(
            problem.initial_guess,
            num_states,
            num_controls,
            problem._num_integrals,
            problem.collocation_points_per_interval,
        )


def validate_problem_basic_structure(problem: ProblemProtocol) -> None:
    """Validate basic problem structure and dimensions."""
    if not hasattr(problem, "get_variable_counts"):
        raise ConfigurationError(
            "Problem object missing required methods", "Internal problem structure error"
        )

    num_states, num_controls = problem.get_variable_counts()
    num_integrals = getattr(problem, "_num_integrals", 0)

    validate_problem_dimensions(num_states, num_controls, num_integrals)


def validate_problem_completeness(problem: ProblemProtocol) -> None:
    """Validate that problem has all required components defined."""
    # Must have dynamics
    if not hasattr(problem, "_dynamics_expressions") or not problem._dynamics_expressions:
        raise ConfigurationError(
            "Problem dynamics must be defined before solving",
            "Call problem.dynamics({state: expression, ...}) first",
        )

    # Must have objective
    if not hasattr(problem, "_objective_expression") or problem._objective_expression is None:
        raise ConfigurationError(
            "Problem objective must be defined before solving",
            "Call problem.minimize(expression) first",
        )

    # Must have at least one variable
    num_states, num_controls = problem.get_variable_counts()
    if num_states == 0 and num_controls == 0:
        raise ConfigurationError(
            "Problem must have at least one state or control variable",
            "Define variables using problem.state() or problem.control()",
        )


def validate_polynomial_degree(degree: int, context: str = "polynomial degree") -> None:
    """Validate polynomial degree specification - CENTRALIZED."""
    if not isinstance(degree, int):
        raise ConfigurationError(
            f"{context.capitalize()} must be integer, got {type(degree)}",
            "Polynomial degree specification error",
        )

    if degree < 1:
        raise ConfigurationError(
            f"{context.capitalize()} must be >= 1, got {degree}",
            "Invalid polynomial degree for collocation",
        )


def validate_polynomial_degrees_list(
    degrees: list[int], context: str = "polynomial degrees"
) -> None:
    """Validate list of polynomial degrees - CENTRALIZED."""
    if not isinstance(degrees, list):
        raise ConfigurationError(
            f"{context.capitalize()} must be list, got {type(degrees)}",
            "Polynomial degrees specification error",
        )

    if not degrees:
        raise ConfigurationError(
            f"{context.capitalize()} list cannot be empty", "Polynomial degrees specification error"
        )

    for i, degree in enumerate(degrees):
        validate_polynomial_degree(degree, f"{context} for interval {i}")


def validate_constraint_input_format(constraint_input: Any, context: str) -> None:
    """Validate constraint input format - CENTRALIZED."""
    if constraint_input is None:
        return  # None is valid (no constraint)

    if isinstance(constraint_input, int | float):
        # Validate numeric constraint
        if math.isnan(constraint_input) or math.isinf(constraint_input):
            raise ConfigurationError(
                f"Constraint value cannot be NaN or infinite, got {constraint_input}",
                f"Constraint value error in {context}",
            )
    elif isinstance(constraint_input, tuple):
        # Validate tuple constraint
        if len(constraint_input) != 2:
            raise ConfigurationError(
                f"Constraint tuple must have exactly 2 elements, got {len(constraint_input)}",
                f"Constraint specification error in {context}",
            )

        for i, val in enumerate(constraint_input):
            if val is not None:
                if not isinstance(val, int | float):
                    raise ConfigurationError(
                        f"Constraint bound {i} must be numeric or None, got {type(val)}",
                        f"Constraint specification error in {context}",
                    )

                if math.isnan(val) or math.isinf(val):
                    raise ConfigurationError(
                        f"Constraint bound {i} cannot be NaN or infinite, got {val}",
                        f"Constraint value error in {context}",
                    )

        # Check bounds ordering
        lower_val, upper_val = constraint_input
        if lower_val is not None and upper_val is not None and lower_val > upper_val:
            raise ConfigurationError(
                f"Lower bound ({lower_val}) cannot be greater than upper bound ({upper_val})",
                f"Constraint bounds ordering error in {context}",
            )
    else:
        raise ConfigurationError(
            f"Invalid constraint input type: {type(constraint_input)}. Expected float, int, tuple, or None",
            f"Constraint specification error in {context}",
        )


def validate_variable_name(name: str, var_type: str) -> None:
    """Validate variable name - CENTRALIZED."""
    if not isinstance(name, str):
        raise ConfigurationError(
            f"{var_type.capitalize()} name must be string, got {type(name)}",
            "Variable naming error",
        )

    if not name.strip():
        raise ConfigurationError(
            f"{var_type.capitalize()} name cannot be empty", "Variable naming error"
        )


def validate_mesh_interval_count(num_intervals: int, context: str = "mesh intervals") -> None:
    """Validate mesh interval count - CENTRALIZED."""
    if not isinstance(num_intervals, int):
        raise ConfigurationError(
            f"Number of {context} must be integer, got {type(num_intervals)}",
            "Mesh configuration error",
        )

    if num_intervals <= 0:
        raise ConfigurationError(
            f"Number of {context} must be positive, got {num_intervals}", "Mesh configuration error"
        )


def validate_casadi_optimization_object(opti: CasadiOpti, context: str = "solver setup") -> None:
    """Validate CasADi optimization object - CENTRALIZED."""
    if opti is None:
        raise ConfigurationError(
            "CasADi optimization object cannot be None", f"TrajectoLab {context} error"
        )


def validate_adaptive_solver_parameters(
    error_tolerance: float,
    max_iterations: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
) -> None:
    """Validate adaptive solver parameters - CENTRALIZED."""
    if error_tolerance <= 0:
        raise ConfigurationError(
            f"Error tolerance must be positive, got {error_tolerance}",
            "Provide a positive error tolerance value",
        )

    if max_iterations <= 0:
        raise ConfigurationError(
            f"Max iterations must be positive, got {max_iterations}",
            "Provide a positive max iterations value",
        )

    if min_polynomial_degree < 1:
        raise ConfigurationError(
            f"Min polynomial degree must be >= 1, got {min_polynomial_degree}",
            "Invalid polynomial degree range",
        )

    if max_polynomial_degree < min_polynomial_degree:
        raise ConfigurationError(
            f"Max polynomial degree ({max_polynomial_degree}) must be >= min degree ({min_polynomial_degree})",
            "Invalid polynomial degree range",
        )


# ============================================================================
# EXISTING VALIDATION FUNCTIONS (keep these)
# ============================================================================


def validate_dynamics_output(
    output: list[CasadiMX] | CasadiMX | Sequence[CasadiMX], num_states: int
) -> CasadiMX:
    """
    Validates and converts dynamics function output to the expected CasadiMX format.
    NOTE: This stays here as it's about data integrity, not user configuration.
    """
    # Guard clause: Check for None output (TrajectoLab logic error)
    if output is None:
        raise DataIntegrityError(
            "Dynamics function returned None",
            "This indicates a TrajectoLab internal error in dynamics function construction",
        )

    if isinstance(output, list):
        result = ca.vertcat(*output) if output else ca.MX(num_states, 1)
        result = ca.MX(result) if isinstance(result, ca.DM) else result
    elif isinstance(output, ca.MX):
        result = output
        if output.shape[1] == 1:
            result = output
        elif output.shape[0] == 1 and num_states > 1:
            result = output.T
        elif num_states == 1:
            result = output
    elif isinstance(output, ca.DM):
        result = ca.MX(output)
        if result.shape[1] == 1:
            pass
        elif result.shape[0] == 1 and num_states > 1:
            result = result.T
    elif isinstance(output, Sequence):
        result = validate_dynamics_output(list(output), num_states)
    else:
        raise DataIntegrityError(
            f"Dynamics function output has unsupported type: {type(output)}",
            "TrajectoLab failed to process dynamics function return value",
        )

    # Critical shape validation - TrajectoLab's responsibility
    if result.shape[0] != num_states:
        raise DataIntegrityError(
            f"Dynamics output has {result.shape[0]} rows, expected {num_states}",
            "Shape mismatch in TrajectoLab dynamics processing",
        )

    return result


def validate_initial_guess_structure(
    initial_guess: InitialGuess,
    num_states: int,
    num_controls: int,
    num_integrals: int,
    polynomial_degrees: list[int],
) -> None:
    """COMPREHENSIVE initial guess validation - CENTRALIZED."""
    # Guard clause: Essential configuration must be present
    if not polynomial_degrees:
        raise ConfigurationError(
            "Cannot validate initial guess: polynomial degrees not configured",
            "Call problem.set_mesh() before validation",
        )

    num_intervals = len(polynomial_degrees)

    # Validate time values
    validate_time_values(initial_guess.initial_time_variable, initial_guess.terminal_time_variable)

    # Validate integrals
    validate_integral_values(initial_guess.integrals, num_integrals)

    # Critical state trajectory validation
    if initial_guess.states is not None:
        expected_state_shapes = [(num_states, N + 1) for N in polynomial_degrees]
        validate_trajectory_arrays(initial_guess.states, expected_state_shapes, "state")

        if len(initial_guess.states) != num_intervals:
            raise DataIntegrityError(
                f"State trajectories count ({len(initial_guess.states)}) doesn't match intervals ({num_intervals})",
                "Initial guess structure inconsistency",
            )

    # Critical control trajectory validation
    if initial_guess.controls is not None:
        expected_control_shapes = [(num_controls, N) for N in polynomial_degrees]
        validate_trajectory_arrays(initial_guess.controls, expected_control_shapes, "control")

        if len(initial_guess.controls) != num_intervals:
            raise DataIntegrityError(
                f"Control trajectories count ({len(initial_guess.controls)}) doesn't match intervals ({num_intervals})",
                "Initial guess structure inconsistency",
            )


def validate_trajectory_arrays(
    trajectories: list[FloatArray],
    expected_shapes: list[tuple[int, int]],
    trajectory_type: str,
) -> None:
    """Validate trajectory arrays against expected shapes with fail-fast."""
    if len(trajectories) != len(expected_shapes):
        raise DataIntegrityError(
            f"{trajectory_type.capitalize()} trajectory count mismatch: got {len(trajectories)}, expected {len(expected_shapes)}",
            f"TrajectoLab {trajectory_type} data preparation error",
        )

    for k, (traj, expected_shape) in enumerate(zip(trajectories, expected_shapes, strict=False)):
        # Critical shape validation
        if traj.shape != expected_shape:
            raise DataIntegrityError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} has shape {traj.shape}, expected {expected_shape}",
                "Shape mismatch in TrajectoLab trajectory data",
            )

        # Critical NaN/Inf validation - TrajectoLab's responsibility to catch this
        if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
            raise DataIntegrityError(
                f"{trajectory_type.capitalize()} trajectory for interval {k} contains NaN or Inf values",
                "Invalid numerical data in TrajectoLab trajectory",
            )


def validate_mesh_configuration(
    polynomial_degrees: list[int],
    mesh_points: FloatArray,
    num_mesh_intervals: int,
) -> None:
    """COMPREHENSIVE mesh configuration validation - CENTRALIZED."""
    # Validate polynomial degrees
    validate_polynomial_degrees_list(polynomial_degrees, "polynomial degrees")

    # Validate mesh intervals
    validate_mesh_interval_count(num_mesh_intervals, "mesh intervals")

    if len(polynomial_degrees) != num_mesh_intervals:
        raise ConfigurationError(
            f"Number of polynomial degrees ({len(polynomial_degrees)}) must equal intervals ({num_mesh_intervals})",
            "Mesh configuration mismatch",
        )

    if len(mesh_points) != num_mesh_intervals + 1:
        raise ConfigurationError(
            f"Number of mesh points ({len(mesh_points)}) must be exactly one more than intervals ({num_mesh_intervals})",
            "Mesh points configuration error",
        )

    # Critical boundary validation
    if not np.isclose(mesh_points[0], -1.0, atol=ZERO_TOLERANCE):
        raise ConfigurationError(
            f"First mesh point must be -1.0, got {mesh_points[0]}", "Invalid mesh boundary"
        )

    if not np.isclose(mesh_points[-1], 1.0, atol=ZERO_TOLERANCE):
        raise ConfigurationError(
            f"Last mesh point must be 1.0, got {mesh_points[-1]}", "Invalid mesh boundary"
        )

    # Critical spacing validation
    mesh_diffs = np.diff(mesh_points)
    if not np.all(mesh_diffs > MESH_TOLERANCE):
        raise ConfigurationError(
            f"Mesh points must be strictly increasing with minimum spacing of {MESH_TOLERANCE}, found minimum difference of {np.min(mesh_diffs)}",
            "Invalid mesh spacing",
        )


def validate_time_bounds(
    t0_bounds: tuple[float, float],
    tf_bounds: tuple[float, float],
) -> None:
    """Validate time bound constraints with fail-fast."""
    # Guard clause: Check bound ordering
    if t0_bounds[0] > t0_bounds[1]:
        raise ConfigurationError(
            f"Initial time bounds are invalid: {t0_bounds}", "Lower bound cannot exceed upper bound"
        )
    if tf_bounds[0] > tf_bounds[1]:
        raise ConfigurationError(
            f"Final time bounds are invalid: {tf_bounds}", "Lower bound cannot exceed upper bound"
        )

    # Check if valid time duration is possible
    max_possible_duration = tf_bounds[1] - t0_bounds[0]
    if max_possible_duration < MINIMUM_TIME_INTERVAL:
        raise ConfigurationError(
            f"Maximum possible time duration ({max_possible_duration}) is below minimum ({MINIMUM_TIME_INTERVAL})",
            f"Time bounds: t0 ∈ {t0_bounds}, tf ∈ {tf_bounds}",
        )


def validate_problem_dimensions(num_states: int, num_controls: int, num_integrals: int) -> None:
    """Validate problem dimension parameters."""
    if num_states < 0:
        raise ConfigurationError(
            f"Number of states must be non-negative, got {num_states}", "Invalid problem dimensions"
        )

    if num_controls < 0:
        raise ConfigurationError(
            f"Number of controls must be non-negative, got {num_controls}",
            "Invalid problem dimensions",
        )

    if num_integrals < 0:
        raise ConfigurationError(
            f"Number of integrals must be non-negative, got {num_integrals}",
            "Invalid problem dimensions",
        )

    # Warning for edge case
    if num_states == 0:
        import warnings

        warnings.warn(
            "Problem has no state variables. This may not be a meaningful optimal control problem.",
            stacklevel=2,
        )


# ============================================================================
# HELPER FUNCTIONS (these handle data integrity, not user configuration)
# ============================================================================


def _validate_numeric_value(value: float | int, name: str) -> None:
    """Validate a single numeric value."""
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be scalar, got {type(value)}")
    if np.isnan(value) or np.isinf(value):
        raise DataIntegrityError(f"{name} has invalid value: {value}", "Numerical data corruption")


def _validate_array_values(array: FloatArray, name: str) -> None:
    """Validate array for NaN and infinite values."""
    if np.any(np.isnan(array)):
        raise DataIntegrityError(f"{name} contains NaN values", "Numerical data corruption")
    if np.any(np.isinf(array)):
        raise DataIntegrityError(f"{name} contains infinite values", "Numerical data corruption")
    if array.dtype != np.float64:
        raise ValueError(f"{name} has dtype {array.dtype}, expected float64")


def validate_integral_values(integrals: float | FloatArray | None, num_integrals: int) -> None:
    """Validate integral values - UNIFIED validation."""
    if integrals is None:
        return

    if num_integrals == 1:
        if not isinstance(integrals, int | float):
            raise ValueError(f"For single integral, expected scalar, got {type(integrals)}")
        _validate_numeric_value(integrals, "Integral")
    elif num_integrals > 1:
        if isinstance(integrals, int | float):
            raise ValueError(
                f"For {num_integrals} integrals, guess must be array-like, got scalar {integrals}"
            )

        integrals_array = np.array(integrals, dtype=np.float64)
        if integrals_array.size != num_integrals:
            raise ValueError(
                f"Integral guess must have exactly {num_integrals} elements, got {integrals_array.size}"
            )
        _validate_array_values(integrals_array, "Integrals")


def validate_time_values(initial_time: float | None, terminal_time: float | None) -> None:
    """Validate actual time values (not bounds)."""
    if initial_time is not None:
        _validate_numeric_value(initial_time, "Initial time")

    if terminal_time is not None:
        _validate_numeric_value(terminal_time, "Terminal time")

    # Check time ordering if both are present
    if initial_time is not None and terminal_time is not None:
        if terminal_time <= initial_time:
            raise ValueError(
                f"Terminal time ({terminal_time}) must be greater than initial time ({initial_time})"
            )


def validate_interval_length(
    interval_start: float, interval_end: float, interval_index: int
) -> None:
    """Validate that an interval has sufficient length."""
    interval_length = interval_end - interval_start
    if interval_length <= MESH_TOLERANCE:
        raise ConfigurationError(
            f"Mesh interval {interval_index} has insufficient length: {interval_length}",
            f"Minimum length required: {MESH_TOLERANCE}",
        )


def set_integral_guess_values(
    opti: CasadiOpti,
    integral_vars: CasadiMX,
    guess: float | FloatArray | None,
    num_integrals: int,
) -> None:
    """Set initial guess values for integrals in CasADi optimization object."""
    if guess is None:
        return

    if num_integrals == 1:
        if isinstance(guess, int | float):
            opti.set_initial(integral_vars, float(guess))
        else:
            raise ValueError(f"Expected scalar for single integral, got {type(guess)}")
    elif num_integrals > 1:
        guess_array = np.array(guess, dtype=np.float64)
        opti.set_initial(integral_vars, guess_array.flatten())
