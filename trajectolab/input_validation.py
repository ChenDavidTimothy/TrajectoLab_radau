# trajectolab/input_validation.py
"""
Centralized input validation for all multiphase user configuration parameters.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any, TypeVar, cast

import casadi as ca
import numpy as np

from .exceptions import ConfigurationError, DataIntegrityError
from .tl_types import FloatArray, MultiPhaseInitialGuess, PhaseID, ProblemProtocol
from .utils.constants import MESH_TOLERANCE, ZERO_TOLERANCE


logger = logging.getLogger(__name__)
T = TypeVar("T")


def validate_multiphase_problem_ready_for_solving(problem: ProblemProtocol) -> None:
    """
    COMPREHENSIVE validation that multiphase problem is properly configured for solving.
    This should be called at ALL solver entry points.
    """
    # Validate basic multiphase problem structure
    validate_multiphase_problem_basic_structure(problem)

    # Validate each phase
    phase_ids = problem.get_phase_ids()
    if not phase_ids:
        raise ConfigurationError(
            "Problem must have at least one phase defined",
            "Use problem.phase(phase_id) to define phases",
        )

    for phase_id in phase_ids:
        validate_phase_configuration(problem, phase_id)

    # Validate multiphase problem completeness
    validate_multiphase_problem_completeness(problem)

    # Validate multiphase initial guess if present
    if problem.initial_guess is not None:
        validate_multiphase_initial_guess_structure(problem.initial_guess, problem)


def validate_multiphase_problem_basic_structure(problem: ProblemProtocol) -> None:
    """Validate basic multiphase problem structure and dimensions."""
    if not hasattr(problem, "get_phase_ids"):
        raise ConfigurationError(
            "Problem object missing required multiphase methods",
            "Internal multiphase problem structure error",
        )

    if not hasattr(problem, "_phases") or not problem._phases:
        raise ConfigurationError(
            "Problem must have at least one phase defined",
            "Use problem.phase(phase_id) to define phases",
        )

    total_states, total_controls, num_static_params = problem.get_total_variable_counts()
    validate_problem_dimensions(
        total_states, total_controls, num_static_params, "multiphase problem"
    )


def validate_phase_configuration(problem: ProblemProtocol, phase_id: PhaseID) -> None:
    """Validate configuration for a specific phase."""
    # Check phase exists
    if phase_id not in problem._phases:
        raise ConfigurationError(
            f"Phase {phase_id} not found in problem",
            "Phase configuration error",
        )

    phase_def = problem._phases[phase_id]

    # Check mesh configuration
    if not phase_def.mesh_configured:
        raise ConfigurationError(
            f"Phase {phase_id} mesh must be configured before solving",
            f"Call phase.set_mesh(polynomial_degrees, mesh_points) for phase {phase_id}",
        )

    # Validate mesh details for this phase
    validate_mesh_configuration(
        phase_def.collocation_points_per_interval,
        phase_def.global_normalized_mesh_nodes,
        len(phase_def.collocation_points_per_interval),
    )

    # Check dynamics
    if not phase_def.dynamics_expressions:
        raise ConfigurationError(
            f"Phase {phase_id} must have dynamics defined before solving",
            f"Call phase.dynamics({{state: expression, ...}}) for phase {phase_id}",
        )

    # Check variables
    num_states, num_controls = phase_def.get_variable_counts()
    if num_states == 0:
        raise ConfigurationError(
            f"Phase {phase_id} must have at least one state variable",
            f"Define variables using phase.state() for phase {phase_id}",
        )


def validate_multiphase_problem_completeness(problem: ProblemProtocol) -> None:
    """Validate that multiphase problem has all required components defined."""
    # Must have objective
    if (
        not hasattr(problem, "_multiphase_state")
        or problem._multiphase_state.objective_expression is None
    ):
        raise ConfigurationError(
            "Multiphase problem must have objective function defined before solving",
            "Call problem.minimize(expression) to define multiphase objective",
        )

    logger.debug("Multiphase configuration validated: %d phases", len(problem.get_phase_ids()))


def validate_multiphase_initial_guess_structure(
    initial_guess: MultiPhaseInitialGuess,
    problem: ProblemProtocol,
) -> None:
    """COMPREHENSIVE multiphase initial guess validation - CENTRALIZED."""
    phase_ids = problem.get_phase_ids()

    # Validate phase states
    if initial_guess.phase_states is not None:
        for phase_id, states_list in initial_guess.phase_states.items():
            if phase_id not in phase_ids:
                raise ConfigurationError(
                    f"Initial guess provided for undefined phase {phase_id}",
                    "Phase initial guess error",
                )

            phase_def = problem._phases[phase_id]
            if not phase_def.mesh_configured:
                raise ConfigurationError(
                    f"Cannot validate initial guess for phase {phase_id}: mesh not configured",
                    "Configure phase mesh before setting initial guess",
                )

            num_states, _ = phase_def.get_variable_counts()
            num_intervals = len(phase_def.collocation_points_per_interval)

            expected_shapes = [
                (num_states, N + 1) for N in phase_def.collocation_points_per_interval
            ]
            validate_interval_trajectory_consistency(
                states_list, num_intervals, expected_shapes, f"phase {phase_id} state"
            )

    # Validate phase controls
    if initial_guess.phase_controls is not None:
        for phase_id, controls_list in initial_guess.phase_controls.items():
            if phase_id not in phase_ids:
                raise ConfigurationError(
                    f"Initial guess provided for undefined phase {phase_id}",
                    "Phase initial guess error",
                )

            phase_def = problem._phases[phase_id]
            if not phase_def.mesh_configured:
                raise ConfigurationError(
                    f"Cannot validate initial guess for phase {phase_id}: mesh not configured",
                    "Configure phase mesh before setting initial guess",
                )

            _, num_controls = phase_def.get_variable_counts()
            num_intervals = len(phase_def.collocation_points_per_interval)

            expected_shapes = [(num_controls, N) for N in phase_def.collocation_points_per_interval]
            validate_interval_trajectory_consistency(
                controls_list, num_intervals, expected_shapes, f"phase {phase_id} control"
            )

    # Validate phase integrals
    if initial_guess.phase_integrals is not None:
        for phase_id, integrals in initial_guess.phase_integrals.items():
            if phase_id not in phase_ids:
                raise ConfigurationError(
                    f"Initial guess provided for undefined phase {phase_id}",
                    "Phase initial guess error",
                )

            phase_def = problem._phases[phase_id]
            validate_integral_values(integrals, phase_def.num_integrals)

    # Validate static parameters
    if initial_guess.static_parameters is not None:
        _, _, num_static_params = problem.get_total_variable_counts()
        if num_static_params == 0:
            raise ValueError(
                "Static parameters guess provided but problem has no static parameters"
            )

        params_array = np.array(initial_guess.static_parameters)
        if params_array.size != num_static_params:
            raise ValueError(
                f"Static parameters guess must have {num_static_params} elements, "
                f"got {params_array.size}"
            )

    # Validate time values
    if initial_guess.phase_initial_times is not None:
        for phase_id, t0 in initial_guess.phase_initial_times.items():
            validate_array_numerical_integrity(
                np.array([t0]), f"Phase {phase_id} initial time", "time validation"
            )

    if initial_guess.phase_terminal_times is not None:
        for phase_id, tf in initial_guess.phase_terminal_times.items():
            validate_array_numerical_integrity(
                np.array([tf]), f"Phase {phase_id} terminal time", "time validation"
            )

        # Check time ordering within phases
        if initial_guess.phase_initial_times is not None:
            for phase_id in initial_guess.phase_terminal_times:
                if phase_id in initial_guess.phase_initial_times:
                    t0 = initial_guess.phase_initial_times[phase_id]
                    tf = initial_guess.phase_terminal_times[phase_id]
                    if tf <= t0:
                        raise DataIntegrityError(
                            f"Phase {phase_id} terminal time ({tf}) must be greater than initial time ({t0})"
                        )


# Re-export existing validation functions with updated signatures where needed
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


def validate_casadi_optimization_object(opti: ca.Opti, context: str = "solver setup") -> None:
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


def validate_dynamics_output(
    output: list[ca.MX] | ca.MX | Sequence[ca.MX], num_states: int
) -> ca.MX:
    """
    Validates and converts dynamics function output to the expected ca.MX format.
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


def validate_problem_dimensions(
    num_states: int, num_controls: int, num_integrals: int, context: str = "problem dimensions"
) -> None:
    """Validate problem dimension parameters with context support."""
    if num_states < 0:
        raise ConfigurationError(
            f"Number of states must be non-negative, got {num_states}",
            f"Invalid problem dimensions in {context}",
        )

    if num_controls < 0:
        raise ConfigurationError(
            f"Number of controls must be non-negative, got {num_controls}",
            f"Invalid problem dimensions in {context}",
        )

    if num_integrals < 0:
        raise ConfigurationError(
            f"Number of integrals must be non-negative, got {num_integrals}",
            f"Invalid problem dimensions in {context}",
        )

    # Warning for edge case
    if num_states == 0:
        import warnings

        warnings.warn(
            "Problem has no state variables. This may not be a meaningful optimal control problem.",
            stacklevel=2,
        )


def validate_array_numerical_integrity(
    array: FloatArray, name: str, context: str = "data validation"
) -> None:
    """Centralized NaN/Inf validation with context - CONSOLIDATED."""
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        raise DataIntegrityError(
            f"{name} contains NaN or Inf values", f"Numerical corruption in {context}"
        )


def validate_array_shape_and_integrity(
    array: FloatArray, expected_shape: tuple[int, ...], name: str, context: str = "data validation"
) -> None:
    """Centralized shape and numerical validation - CONSOLIDATED."""
    if array.shape != expected_shape:
        raise DataIntegrityError(
            f"{name} has shape {array.shape}, expected {expected_shape}",
            f"Shape mismatch in {context}",
        )

    validate_array_numerical_integrity(array, name, context)


def validate_casadi_result_integrity(
    result: Any,
    expected_shape: tuple[int, ...] | None = None,
    name: str = "CasADi result",
    context: str = "CasADi evaluation",
) -> FloatArray:
    """Centralized CasADi result validation and conversion - CONSOLIDATED."""
    try:
        if isinstance(result, ca.DM | ca.MX):
            result_np = np.array(result.full(), dtype=np.float64)
        elif isinstance(result, list | tuple):
            if not result:
                raise DataIntegrityError(f"Empty {name}", f"{context} returned empty result")
            # Handle array of CasADi objects
            result_np = np.array([float(ca.evalf(item)) for item in result], dtype=np.float64)
        else:
            result_np = np.array(result, dtype=np.float64)

        # Ensure it's a proper FloatArray
        result_np = np.asarray(result_np, dtype=np.float64)

        # Validate numerical integrity
        validate_array_numerical_integrity(result_np, name, context)

        # Validate shape if provided
        if expected_shape is not None:
            validate_array_shape_and_integrity(result_np, expected_shape, name, context)

        return cast(FloatArray, result_np)

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Failed to validate and convert {name}: {e}", f"{context} validation error"
        ) from e


def validate_interval_trajectory_consistency(
    trajectories: list[FloatArray],
    expected_intervals: int,
    expected_shapes: list[tuple[int, int]],
    trajectory_type: str,
    context: str = "trajectory validation",
) -> None:
    """Centralized trajectory consistency validation - CONSOLIDATED."""
    if len(trajectories) != expected_intervals:
        raise DataIntegrityError(
            f"{trajectory_type.capitalize()} trajectory count ({len(trajectories)}) doesn't match expected intervals ({expected_intervals})",
            f"Trajectory structure inconsistency in {context}",
        )

    for k, (traj, expected_shape) in enumerate(zip(trajectories, expected_shapes, strict=False)):
        validate_array_shape_and_integrity(
            traj,
            expected_shape,
            f"{trajectory_type} trajectory for interval {k}",
            f"{context} - interval {k}",
        )


def validate_integral_values(integrals: float | FloatArray | None, num_integrals: int) -> None:
    """Validate integral values - UNIFIED validation."""
    if integrals is None:
        return

    if num_integrals == 1:
        if not isinstance(integrals, int | float):
            raise DataIntegrityError(f"For single integral, expected scalar, got {type(integrals)}")
        validate_array_numerical_integrity(np.array([integrals]), "Integral", "integral validation")
    elif num_integrals > 1:
        if isinstance(integrals, int | float):
            raise DataIntegrityError(
                f"For {num_integrals} integrals, guess must be array-like, got scalar {integrals}"
            )

        integrals_array = np.array(integrals, dtype=np.float64)
        if integrals_array.size != num_integrals:
            raise DataIntegrityError(
                f"Integral guess must have exactly {num_integrals} elements, got {integrals_array.size}"
            )
        validate_array_numerical_integrity(integrals_array, "Integrals")


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
    opti: ca.Opti,
    integral_vars: ca.MX,
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
            raise ConfigurationError(f"Expected scalar for single integral, got {type(guess)}")
    elif num_integrals > 1:
        guess_array = np.array(guess, dtype=np.float64)
        opti.set_initial(integral_vars, guess_array.flatten())
