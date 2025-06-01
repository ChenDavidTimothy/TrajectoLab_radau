import logging
import math
from collections.abc import Sequence
from typing import Any, TypeVar

import casadi as ca
import numpy as np

from .exceptions import ConfigurationError, DataIntegrityError
from .tl_types import FloatArray, MultiPhaseInitialGuess, PhaseID, ProblemProtocol
from .utils.constants import MESH_TOLERANCE, ZERO_TOLERANCE


logger = logging.getLogger(__name__)
T = TypeVar("T")


# ============================================================================
# CORE VALIDATION PRIMITIVES - Used everywhere, defined once
# ============================================================================


def validate_not_none(value: Any, name: str, context: str = "validation") -> None:
    """Single source for None validation."""
    if value is None:
        raise DataIntegrityError(f"{name} cannot be None", context)


def validate_positive_integer(value: Any, name: str, min_value: int = 1) -> None:
    """Single source for positive integer validation."""
    if not isinstance(value, int):
        raise ConfigurationError(f"{name} must be integer, got {type(value)}")
    if value < min_value:
        raise ConfigurationError(f"{name} must be >= {min_value}, got {value}")


def validate_positive_number(value: Any, name: str) -> None:
    """Single source for positive number validation."""
    if not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be numeric, got {type(value)}")
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")
    if math.isnan(value) or math.isinf(value):
        raise ConfigurationError(f"{name} cannot be NaN or infinite, got {value}")


def validate_string_not_empty(value: Any, name: str) -> None:
    """Single source for non-empty string validation."""
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be string, got {type(value)}")
    if not value.strip():
        raise ConfigurationError(f"{name} cannot be empty")


def validate_array_numerical_integrity(
    array: FloatArray, name: str, context: str = "validation"
) -> None:
    """Single source for NaN/Inf validation."""
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        raise DataIntegrityError(
            f"{name} contains NaN or Inf values", f"Numerical corruption in {context}"
        )


def validate_array_shape(
    array: FloatArray, expected_shape: tuple[int, ...], name: str, context: str = "validation"
) -> None:
    """Single source for shape validation."""
    if array.shape != expected_shape:
        raise DataIntegrityError(
            f"{name} has shape {array.shape}, expected {expected_shape}",
            f"Shape mismatch in {context}",
        )


def validate_casadi_object(obj: Any, name: str, context: str = "validation") -> None:
    """Single source for CasADi object validation."""
    if obj is None:
        raise DataIntegrityError(f"{name} cannot be None", f"CasADi object error in {context}")


# ============================================================================
# CONSTRAINT VALIDATION - Enhanced for symbolic constraints
# ============================================================================


def validate_constraint_input_format(constraint_input: Any, context: str) -> None:
    """SINGLE SOURCE for all constraint validation - supports symbolic expressions."""
    if constraint_input is None or isinstance(constraint_input, ca.MX):
        return  # None and CasADi expressions are valid

    if isinstance(constraint_input, int | float):
        if math.isnan(constraint_input) or math.isinf(constraint_input):
            raise ConfigurationError(
                f"Constraint cannot be NaN/infinite: {constraint_input}", context
            )
    elif isinstance(constraint_input, tuple):
        if len(constraint_input) != 2:
            raise ConfigurationError(
                f"Constraint tuple must have 2 elements, got {len(constraint_input)}", context
            )

        lower, upper = constraint_input
        for i, val in enumerate([lower, upper]):
            if val is not None:
                if not isinstance(val, int | float):
                    raise ConfigurationError(
                        f"Constraint bound {i} must be numeric/None, got {type(val)}", context
                    )
                if math.isnan(val) or math.isinf(val):
                    raise ConfigurationError(
                        f"Constraint bound {i} cannot be NaN/infinite: {val}", context
                    )

        if lower is not None and upper is not None and lower > upper:
            raise ConfigurationError(f"Lower bound ({lower}) > upper bound ({upper})", context)
    else:
        raise ConfigurationError(f"Invalid constraint type: {type(constraint_input)}", context)


# ============================================================================
# MESH VALIDATION - All mesh validation consolidated
# ============================================================================


def validate_polynomial_degree(degree: int, context: str = "polynomial degree") -> None:
    """SINGLE SOURCE for polynomial degree validation."""
    validate_positive_integer(degree, context, min_value=1)


def validate_mesh_configuration(
    polynomial_degrees: list[int], mesh_points: FloatArray, num_intervals: int
) -> None:
    """SINGLE SOURCE for complete mesh validation."""
    # Validate polynomial degrees
    if not isinstance(polynomial_degrees, list) or not polynomial_degrees:
        raise ConfigurationError("Polynomial degrees must be non-empty list")

    for i, degree in enumerate(polynomial_degrees):
        validate_polynomial_degree(degree, f"polynomial degree for interval {i}")

    # Validate intervals
    validate_positive_integer(num_intervals, "number of mesh intervals")

    if len(polynomial_degrees) != num_intervals:
        raise ConfigurationError(
            f"Polynomial degrees count ({len(polynomial_degrees)}) != intervals ({num_intervals})"
        )

    if len(mesh_points) != num_intervals + 1:
        raise ConfigurationError(
            f"Mesh points count ({len(mesh_points)}) != intervals+1 ({num_intervals + 1})"
        )

    # Boundary validation
    if not np.isclose(mesh_points[0], -1.0, atol=ZERO_TOLERANCE):
        raise ConfigurationError(f"First mesh point must be -1.0, got {mesh_points[0]}")
    if not np.isclose(mesh_points[-1], 1.0, atol=ZERO_TOLERANCE):
        raise ConfigurationError(f"Last mesh point must be 1.0, got {mesh_points[-1]}")

    # Spacing validation
    mesh_diffs = np.diff(mesh_points)
    if not np.all(mesh_diffs > MESH_TOLERANCE):
        raise ConfigurationError(
            f"Mesh points must be strictly increasing with min spacing {MESH_TOLERANCE}"
        )


def validate_interval_length(start: float, end: float, interval_index: int) -> None:
    """SINGLE SOURCE for interval length validation."""
    if end - start <= MESH_TOLERANCE:
        raise ConfigurationError(
            f"Interval {interval_index} length ({end - start}) < minimum ({MESH_TOLERANCE})"
        )


# ============================================================================
# PROBLEM VALIDATION - Complete problem structure validation
# ============================================================================


def validate_problem_dimensions(
    num_states: int, num_controls: int, num_static_params: int, context: str = "problem"
) -> None:
    """SINGLE SOURCE for problem dimension validation."""
    for count, name in [
        (num_states, "states"),
        (num_controls, "controls"),
        (num_static_params, "static parameters"),
    ]:
        if not isinstance(count, int) or count < 0:
            raise ConfigurationError(
                f"Number of {name} must be non-negative integer, got {count}", context
            )


def validate_phase_configuration(problem: ProblemProtocol, phase_id: PhaseID) -> None:
    """SINGLE SOURCE for phase configuration validation."""
    if phase_id not in problem._phases:
        raise ConfigurationError(f"Phase {phase_id} not found in problem")

    phase_def = problem._phases[phase_id]

    # Mesh validation
    if not phase_def.mesh_configured:
        raise ConfigurationError(
            f"Phase {phase_id} mesh must be configured - call phase.set_mesh()"
        )

    validate_mesh_configuration(
        phase_def.collocation_points_per_interval,
        phase_def.global_normalized_mesh_nodes,
        len(phase_def.collocation_points_per_interval),
    )

    # Dynamics validation
    if not phase_def.dynamics_expressions:
        raise ConfigurationError(
            f"Phase {phase_id} dynamics must be defined - call phase.dynamics()"
        )

    # Variable validation
    num_states, num_controls = phase_def.get_variable_counts()
    if num_states == 0:
        raise ConfigurationError(f"Phase {phase_id} must have at least one state variable")


def validate_multiphase_problem_ready_for_solving(problem: ProblemProtocol) -> None:
    """SINGLE SOURCE - MASTER validation for solve readiness."""
    # Basic structure
    if not hasattr(problem, "_phases") or not problem._phases:
        raise ConfigurationError("Problem must have at least one phase - use problem.phase()")

    # Validate dimensions
    total_states, total_controls, num_static_params = problem.get_total_variable_counts()
    validate_problem_dimensions(
        total_states, total_controls, num_static_params, "multiphase problem"
    )

    # Validate each phase
    for phase_id in problem.get_phase_ids():
        validate_phase_configuration(problem, phase_id)

    # Objective validation
    if (
        not hasattr(problem, "_multiphase_state")
        or problem._multiphase_state.objective_expression is None
    ):
        raise ConfigurationError("Problem must have objective - call problem.minimize()")

    # Process symbolic constraints and validate configuration
    problem.validate_multiphase_configuration()

    # Validate initial guess if present
    if problem.initial_guess is not None:
        validate_multiphase_initial_guess_structure(problem.initial_guess, problem)


# ============================================================================
# INITIAL GUESS VALIDATION - Complete initial guess validation
# ============================================================================


def validate_trajectory_consistency(
    trajectories: list[FloatArray],
    expected_intervals: int,
    expected_shapes: list[tuple[int, int]],
    trajectory_type: str,
) -> None:
    """SINGLE SOURCE for trajectory consistency validation."""
    if len(trajectories) != expected_intervals:
        raise DataIntegrityError(
            f"{trajectory_type} count ({len(trajectories)}) != expected intervals ({expected_intervals})"
        )

    for k, (traj, expected_shape) in enumerate(zip(trajectories, expected_shapes, strict=False)):
        validate_array_shape(traj, expected_shape, f"{trajectory_type} interval {k}")
        validate_array_numerical_integrity(traj, f"{trajectory_type} interval {k}")


def validate_integral_values(integrals: float | FloatArray | None, num_integrals: int) -> None:
    """SINGLE SOURCE for integral values validation."""
    if integrals is None:
        return

    if num_integrals == 1:
        if not isinstance(integrals, int | float):
            raise DataIntegrityError(f"Single integral must be scalar, got {type(integrals)}")
        validate_array_numerical_integrity(np.array([integrals]), "integral value")
    elif num_integrals > 1:
        if isinstance(integrals, int | float):
            raise DataIntegrityError(f"Multiple integrals ({num_integrals}) need array, got scalar")

        integrals_array = np.array(integrals, dtype=np.float64)
        if integrals_array.size != num_integrals:
            raise DataIntegrityError(
                f"Integral count mismatch: expected {num_integrals}, got {integrals_array.size}"
            )
        validate_array_numerical_integrity(integrals_array, "integral values")


def validate_multiphase_initial_guess_structure(
    initial_guess: MultiPhaseInitialGuess, problem: ProblemProtocol
) -> None:
    """SINGLE SOURCE for complete initial guess validation."""
    phase_ids = problem.get_phase_ids()

    # Validate phase states
    if initial_guess.phase_states is not None:
        for phase_id, states_list in initial_guess.phase_states.items():
            if phase_id not in phase_ids:
                raise ConfigurationError(f"Initial guess for undefined phase {phase_id}")

            phase_def = problem._phases[phase_id]
            if not phase_def.mesh_configured:
                raise ConfigurationError(
                    f"Phase {phase_id} mesh not configured for initial guess validation"
                )

            num_states, _ = phase_def.get_variable_counts()
            expected_shapes = [
                (num_states, N + 1) for N in phase_def.collocation_points_per_interval
            ]
            validate_trajectory_consistency(
                states_list, len(expected_shapes), expected_shapes, f"phase {phase_id} states"
            )

    # Validate phase controls
    if initial_guess.phase_controls is not None:
        for phase_id, controls_list in initial_guess.phase_controls.items():
            if phase_id not in phase_ids:
                raise ConfigurationError(f"Initial guess for undefined phase {phase_id}")

            phase_def = problem._phases[phase_id]
            _, num_controls = phase_def.get_variable_counts()
            expected_shapes = [(num_controls, N) for N in phase_def.collocation_points_per_interval]
            validate_trajectory_consistency(
                controls_list, len(expected_shapes), expected_shapes, f"phase {phase_id} controls"
            )

    # Validate phase integrals
    if initial_guess.phase_integrals is not None:
        for phase_id, integrals in initial_guess.phase_integrals.items():
            if phase_id not in phase_ids:
                raise ConfigurationError(f"Initial guess for undefined phase {phase_id}")
            validate_integral_values(integrals, problem._phases[phase_id].num_integrals)

    # Validate static parameters
    if initial_guess.static_parameters is not None:
        _, _, num_static_params = problem.get_total_variable_counts()
        if num_static_params == 0:
            raise ConfigurationError(
                "Static parameter guess provided but problem has no static parameters"
            )

        params_array = np.array(initial_guess.static_parameters)
        if params_array.size != num_static_params:
            raise ConfigurationError(
                f"Static parameter count mismatch: expected {num_static_params}, got {params_array.size}"
            )
        validate_array_numerical_integrity(params_array, "static parameters")

    # Validate time values
    for time_dict, time_type in [
        (initial_guess.phase_initial_times, "initial"),
        (initial_guess.phase_terminal_times, "terminal"),
    ]:
        if time_dict is not None:
            for phase_id, time_val in time_dict.items():
                validate_array_numerical_integrity(
                    np.array([time_val]), f"phase {phase_id} {time_type} time"
                )

    # Validate time ordering
    if (
        initial_guess.phase_initial_times is not None
        and initial_guess.phase_terminal_times is not None
    ):
        for phase_id in initial_guess.phase_terminal_times:
            if phase_id in initial_guess.phase_initial_times:
                t0, tf = (
                    initial_guess.phase_initial_times[phase_id],
                    initial_guess.phase_terminal_times[phase_id],
                )
                if tf <= t0:
                    raise DataIntegrityError(
                        f"Phase {phase_id} terminal time ({tf}) <= initial time ({t0})"
                    )


# ============================================================================
# ADAPTIVE SOLVER VALIDATION
# ============================================================================


def validate_adaptive_solver_parameters(
    error_tolerance: float,
    max_iterations: int,
    min_polynomial_degree: int,
    max_polynomial_degree: int,
) -> None:
    """SINGLE SOURCE for adaptive solver parameter validation."""
    validate_positive_number(error_tolerance, "error tolerance")
    validate_positive_integer(max_iterations, "max iterations")
    validate_positive_integer(min_polynomial_degree, "min polynomial degree")
    validate_positive_integer(max_polynomial_degree, "max polynomial degree")

    if max_polynomial_degree < min_polynomial_degree:
        raise ConfigurationError(
            f"Max degree ({max_polynomial_degree}) < min degree ({min_polynomial_degree})"
        )


# ============================================================================
# DYNAMICS OUTPUT VALIDATION - OPTIMIZED for direct vector interface
# ============================================================================


def validate_dynamics_output(output: Any, num_states: int) -> ca.MX:
    """
    OPTIMIZED: Dynamics validation for both legacy list format and new direct vector format.

    Handles backward compatibility while optimizing for the new direct ca.MX interface.
    """
    if output is None:
        raise DataIntegrityError("Dynamics function returned None", "Dynamics evaluation error")

    # OPTIMIZED PATH: Direct ca.MX vector (new interface)
    if isinstance(output, ca.MX):
        if output.shape[0] == num_states and output.shape[1] == 1:
            return output  # Perfect match - no conversion needed
        elif output.shape[0] == 1 and output.shape[1] == num_states:
            return output.T  # Simple transpose
        elif output.shape[0] == num_states:
            return output  # Allow row vector for single-state case
        else:
            raise DataIntegrityError(
                f"Dynamics ca.MX shape mismatch: got {output.shape}, expected ({num_states}, 1)"
            )

    # LEGACY COMPATIBILITY: List format (slower but maintained for compatibility)
    if isinstance(output, list):
        if len(output) != num_states:
            raise DataIntegrityError(
                f"Dynamics list length mismatch: got {len(output)}, expected {num_states}"
            )
        result = ca.vertcat(*output) if output else ca.MX(num_states, 1)
        return result

    # Handle CasADi DM
    if isinstance(output, ca.DM):
        result = ca.MX(output)
        if result.shape[0] == 1 and num_states > 1:
            result = result.T
        return result

    # Handle sequences (convert to list path)
    if isinstance(output, Sequence):
        return validate_dynamics_output(list(output), num_states)

    # Unsupported type
    raise DataIntegrityError(
        f"Unsupported dynamics output type: {type(output)}", "Dynamics type error"
    )


# ============================================================================
# UTILITY FUNCTIONS FOR CASADI INTEGRATION
# ============================================================================


def set_integral_guess_values(
    opti: ca.Opti, integral_vars: ca.MX, guess: float | FloatArray | None, num_integrals: int
) -> None:
    """SINGLE SOURCE for setting integral guess values in CasADi."""
    if guess is None:
        return

    if num_integrals == 1:
        if isinstance(guess, int | float):
            opti.set_initial(integral_vars, float(guess))
        else:
            raise ConfigurationError(f"Single integral needs scalar guess, got {type(guess)}")
    elif num_integrals > 1:
        guess_array = np.array(guess, dtype=np.float64)
        opti.set_initial(integral_vars, guess_array.flatten())
