import logging
from collections.abc import Callable
from dataclasses import dataclass

import casadi as ca

from ..birkhoff import BirkhoffBasisComponents, _compute_birkhoff_basis_components
from ..exceptions import DataIntegrityError, SolutionExtractionError
from ..mtor_types import (
    Constraint,
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)
from ..problem.state import PhaseDefinition
from ..solution_extraction import _extract_and_format_multiphase_solution
from .constraints_birkhoff_solver import (
    _apply_birkhoff_boundary_constraint,
    _apply_birkhoff_collocation_constraints,
    _apply_birkhoff_multiphase_cross_phase_event_constraints,
    _apply_birkhoff_path_constraints,
)
from .initial_guess_birkhoff_solver import _apply_birkhoff_multiphase_initial_guess
from .integrals_birkhoff_solver import (
    _apply_birkhoff_phase_integral_constraints,
    _setup_birkhoff_phase_integrals,
)
from .types_birkhoff_solver import (
    _BirkhoffMultiPhaseVariable,
    _BirkhoffPhaseVariable,
)
from .variables_birkhoff_solver import (
    _setup_birkhoff_multiphase_optimization_variables,
)


logger = logging.getLogger(__name__)


@dataclass
class _BirkhoffPhaseFunctions:
    dynamics_function: Callable[..., ca.MX]
    path_constraints_function: Callable[..., list[Constraint]] | None
    integral_integrand_function: Callable[..., ca.MX] | None


@dataclass
class _BirkhoffPhaseContext:
    phase_id: PhaseID
    num_states: int
    grid_points: tuple[float, ...]
    basis_components: BirkhoffBasisComponents

    initial_time_var: ca.MX
    terminal_time_var: ca.MX

    static_parameters_vec: ca.MX | None
    static_parameter_symbols: list[ca.MX] | None

    num_integrals: int
    accumulated_integral_expressions: list[ca.MX]


@dataclass
class _BirkhoffSolverConfiguration:
    opti: ca.Opti
    variables: _BirkhoffMultiPhaseVariable
    phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]]
    problem: ProblemProtocol
    grid_points_per_phase: dict[PhaseID, tuple[float, ...]]


def _extract_birkhoff_phase_endpoint_data(
    variables: _BirkhoffMultiPhaseVariable, problem: ProblemProtocol
) -> dict[PhaseID, dict[str, ca.MX]]:
    phase_endpoint_data = {}

    for phase_id, phase_vars in variables.phase_variables.items():
        if len(phase_vars.state_at_mesh_nodes) < 2:
            raise DataIntegrityError(
                f"Phase {phase_id} must have at least 2 mesh nodes for Birkhoff method",
                "Birkhoff mesh configuration error",
            )

        initial_state = phase_vars.state_at_mesh_nodes[0]
        terminal_state = phase_vars.state_at_mesh_nodes[-1]

        phase_endpoint_data[phase_id] = {
            "t0": phase_vars.initial_time,
            "tf": phase_vars.terminal_time,
            "x0": initial_state,
            "xf": terminal_state,
            "q": phase_vars.integral_variables,
        }

    return phase_endpoint_data


def _setup_birkhoff_phase_variables(
    config: _BirkhoffSolverConfiguration,
    phase_vars: _BirkhoffPhaseVariable,
    context: _BirkhoffPhaseContext,
) -> ca.MX:
    # For Birkhoff, we create a matrix of states at all grid points
    num_grid_points = len(context.grid_points)

    if len(phase_vars.state_at_mesh_nodes) < 2:
        raise DataIntegrityError(
            f"Phase {context.phase_id} requires at least 2 mesh nodes", "Birkhoff phase setup error"
        )

    # Create state matrix with states at all grid points
    state_list = []
    for j in range(num_grid_points):
        # For simplicity, we'll interpolate between initial and final states
        # In a more sophisticated implementation, this could use the mesh structure
        alpha = (context.grid_points[j] + 1.0) / 2.0  # Map [-1,1] to [0,1]
        initial_state = phase_vars.state_at_mesh_nodes[0]
        final_state = phase_vars.state_at_mesh_nodes[-1]
        interpolated_state = (1.0 - alpha) * initial_state + alpha * final_state
        state_list.append(interpolated_state)

    state_matrix = ca.horzcat(*state_list)
    phase_vars.state_matrices.append(state_matrix)

    return state_matrix


def _apply_birkhoff_phase_constraints(
    config: _BirkhoffSolverConfiguration,
    context: _BirkhoffPhaseContext,
    state_at_grid_points: ca.MX,
    phase_vars: _BirkhoffPhaseVariable,
    functions: _BirkhoffPhaseFunctions,
) -> None:
    # Apply Birkhoff collocation constraints
    _apply_birkhoff_collocation_constraints(
        config.opti,
        context.phase_id,
        state_at_grid_points,
        phase_vars.virtual_variables,
        phase_vars.control_variables,
        context.basis_components,
        phase_vars.state_at_mesh_nodes[0],  # initial state x^a
        functions.dynamics_function,
        config.problem,
        context.static_parameters_vec,
    )

    # Apply boundary constraint: x^b = x^a + w^B^T * V
    _apply_birkhoff_boundary_constraint(
        config.opti,
        context.phase_id,
        phase_vars.state_at_mesh_nodes[0],  # x^a
        phase_vars.state_at_mesh_nodes[-1],  # x^b
        phase_vars.virtual_variables,
        context.basis_components,
    )

    # Apply path constraints if they exist
    if functions.path_constraints_function is not None:
        _apply_birkhoff_path_constraints(
            config.opti,
            context.phase_id,
            state_at_grid_points,
            phase_vars.control_variables,
            context.basis_components,
            functions.path_constraints_function,
            config.problem,
            context.static_parameters_vec,
            context.static_parameter_symbols,
            context.initial_time_var,
            context.terminal_time_var,
        )


def _setup_birkhoff_phase_integrals_processing(
    config: _BirkhoffSolverConfiguration,
    context: _BirkhoffPhaseContext,
    state_at_grid_points: ca.MX,
    phase_vars: _BirkhoffPhaseVariable,
    functions: _BirkhoffPhaseFunctions,
) -> None:
    if context.num_integrals == 0 or functions.integral_integrand_function is None:
        return

    _setup_birkhoff_phase_integrals(
        config.opti,
        context.phase_id,
        state_at_grid_points,
        phase_vars.control_variables,
        context.basis_components,
        context.initial_time_var,
        context.terminal_time_var,
        functions.integral_integrand_function,
        context.num_integrals,
        context.accumulated_integral_expressions,
        context.static_parameters_vec,
    )


def _process_birkhoff_phase(
    config: _BirkhoffSolverConfiguration,
    phase_vars: _BirkhoffPhaseVariable,
    context: _BirkhoffPhaseContext,
    functions: _BirkhoffPhaseFunctions,
) -> None:
    try:
        state_at_grid_points = _setup_birkhoff_phase_variables(config, phase_vars, context)

        _apply_birkhoff_phase_constraints(
            config, context, state_at_grid_points, phase_vars, functions
        )

        _setup_birkhoff_phase_integrals_processing(
            config, context, state_at_grid_points, phase_vars, functions
        )

        if context.num_integrals > 0 and phase_vars.integral_variables is not None:
            _apply_birkhoff_phase_integral_constraints(
                config.opti,
                phase_vars.integral_variables,
                context.accumulated_integral_expressions,
                context.num_integrals,
                context.phase_id,
            )

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Failed to process Birkhoff phase {context.phase_id}: {e}",
            "MAPTOR Birkhoff phase setup error",
        ) from e


def _create_birkhoff_phase_context(
    phase_id: PhaseID,
    phase_def: PhaseDefinition,
    config: _BirkhoffSolverConfiguration,
    accumulated_integral_expressions: list[ca.MX],
) -> _BirkhoffPhaseContext:
    num_states, _ = config.problem._get_phase_variable_counts(phase_id)

    if phase_id not in config.grid_points_per_phase:
        raise DataIntegrityError(
            f"No grid points provided for phase {phase_id}", "Birkhoff grid configuration error"
        )

    grid_points = config.grid_points_per_phase[phase_id]

    # Transform grid points to [-1, 1] if needed
    grid_points_normalized = tuple(grid_points)
    basis_components = _compute_birkhoff_basis_components(grid_points_normalized, -1.0, 1.0)

    static_parameter_symbols = None
    if config.variables.static_parameters is not None:
        static_parameter_symbols = config.problem._static_parameters.get_ordered_parameter_symbols()

    endpoint_data = config.phase_endpoint_data[phase_id]

    return _BirkhoffPhaseContext(
        phase_id=phase_id,
        num_states=num_states,
        grid_points=grid_points,
        basis_components=basis_components,
        initial_time_var=endpoint_data["t0"],
        terminal_time_var=endpoint_data["tf"],
        static_parameters_vec=config.variables.static_parameters,
        static_parameter_symbols=static_parameter_symbols,
        num_integrals=phase_def.num_integrals,
        accumulated_integral_expressions=accumulated_integral_expressions,
    )


def _extract_birkhoff_phase_functions(
    config: _BirkhoffSolverConfiguration, phase_id: PhaseID
) -> _BirkhoffPhaseFunctions:
    return _BirkhoffPhaseFunctions(
        dynamics_function=config.problem._get_phase_dynamics_function(phase_id),
        path_constraints_function=config.problem._get_phase_path_constraints_function(phase_id),
        integral_integrand_function=config.problem._get_phase_integrand_function(phase_id),
    )


def _process_all_birkhoff_phases(config: _BirkhoffSolverConfiguration) -> None:
    for phase_id in config.problem._get_phase_ids():
        if phase_id not in config.variables.phase_variables:
            continue

        phase_vars = config.variables.phase_variables[phase_id]
        phase_def = config.problem._phases[phase_id]

        accumulated_integral_expressions = (
            [ca.MX(0) for _ in range(phase_def.num_integrals)]
            if phase_def.num_integrals > 0
            else []
        )

        context = _create_birkhoff_phase_context(
            phase_id, phase_def, config, accumulated_integral_expressions
        )
        functions = _extract_birkhoff_phase_functions(config, phase_id)

        _process_birkhoff_phase(config, phase_vars, context, functions)


def _setup_birkhoff_objective_and_constraints(config: _BirkhoffSolverConfiguration) -> None:
    objective_function = config.problem._get_objective_function()

    try:
        objective_value = objective_function(
            config.phase_endpoint_data,
            config.variables.static_parameters,
        )
        config.opti.minimize(objective_value)

    except Exception as e:
        raise DataIntegrityError(
            f"Failed to set up Birkhoff multiphase objective function: {e}",
            "Birkhoff multiphase objective function evaluation error",
        ) from e

    _apply_birkhoff_multiphase_cross_phase_event_constraints(
        config.opti,
        config.phase_endpoint_data,
        config.variables.static_parameters,
        config.problem,
    )


def _configure_birkhoff_solver(config: _BirkhoffSolverConfiguration) -> None:
    solver_options_to_use = config.problem.solver_options or {}

    try:
        config.opti.solver("ipopt", solver_options_to_use)
    except Exception as e:
        raise DataIntegrityError(
            f"Failed to configure Birkhoff solver: {e}", "Invalid solver options"
        ) from e

    config.opti.multiphase_variables_reference = config.variables

    objective_function = config.problem._get_objective_function()
    objective_expression = objective_function(
        config.phase_endpoint_data, config.variables.static_parameters
    )
    config.opti.multiphase_objective_expression_reference = objective_expression


def _create_birkhoff_solver_configuration(
    problem: ProblemProtocol, grid_points_per_phase: dict[PhaseID, tuple[float, ...]]
) -> _BirkhoffSolverConfiguration:
    try:
        opti = ca.Opti()
        variables = _setup_birkhoff_multiphase_optimization_variables(
            opti, problem, grid_points_per_phase
        )
        phase_endpoint_data = _extract_birkhoff_phase_endpoint_data(variables, problem)

        return _BirkhoffSolverConfiguration(
            opti=opti,
            variables=variables,
            phase_endpoint_data=phase_endpoint_data,
            problem=problem,
            grid_points_per_phase=grid_points_per_phase,
        )

    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Failed to create Birkhoff solver configuration: {e}",
            "MAPTOR Birkhoff solver setup error",
        ) from e


def _execute_birkhoff_solve(config: _BirkhoffSolverConfiguration) -> OptimalControlSolution:
    try:
        solver_solution = config.opti.solve()
        logger.debug("Birkhoff multiphase NLP solver completed successfully")

        try:
            solution_obj = _extract_and_format_multiphase_solution(
                solver_solution, config.opti, config.problem
            )
            logger.debug("Birkhoff multiphase solution extraction completed")
            return solution_obj

        except Exception as e:
            logger.error("Birkhoff multiphase solution extraction failed: %s", str(e))
            raise SolutionExtractionError(
                f"Failed to extract Birkhoff multiphase solution: {e}",
                "MAPTOR Birkhoff multiphase solution processing error",
            ) from e

    except RuntimeError as e:
        logger.warning("Birkhoff multiphase NLP solver failed: %s", str(e))
        return _handle_birkhoff_solver_failure(config, e)


def _handle_birkhoff_solver_failure(
    config: _BirkhoffSolverConfiguration, error: RuntimeError
) -> OptimalControlSolution:
    try:
        solution_obj = _extract_and_format_multiphase_solution(None, config.opti, config.problem)
    except Exception as extract_error:
        logger.error(
            "Birkhoff multiphase solution extraction failed after solver failure: %s",
            str(extract_error),
        )
        raise SolutionExtractionError(
            f"Failed to extract Birkhoff multiphase solution after solver failure: {extract_error}",
            "Birkhoff multiphase solution extraction error",
        ) from extract_error

    solution_obj.success = False
    solution_obj.message = f"Birkhoff multiphase solver runtime error: {error}"

    _extract_birkhoff_debug_values(config, solution_obj)
    return solution_obj


def _extract_birkhoff_debug_values(
    config: _BirkhoffSolverConfiguration, solution_obj: OptimalControlSolution
) -> None:
    try:
        if hasattr(config.opti, "debug") and config.opti.debug is not None:
            for phase_id, phase_vars in config.variables.phase_variables.items():
                try:
                    solution_obj.phase_initial_times[phase_id] = float(
                        config.opti.debug.value(phase_vars.initial_time)
                    )
                    solution_obj.phase_terminal_times[phase_id] = float(
                        config.opti.debug.value(phase_vars.terminal_time)
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not extract debug values for Birkhoff phase {phase_id}: {e}"
                    )
            logger.debug("Retrieved debug values from failed Birkhoff multiphase solve")
    except Exception as e:
        logger.debug(f"Could not extract debug values from failed Birkhoff multiphase solve: {e}")


def _solve_multiphase_birkhoff_collocation(
    problem: ProblemProtocol, grid_points_per_phase: dict[PhaseID, tuple[float, ...]]
) -> OptimalControlSolution:
    logger.debug("Starting multiphase Birkhoff collocation solver")

    phase_ids = problem._get_phase_ids()
    total_states, total_controls, num_static_params = problem._get_total_variable_counts()

    logger.debug(
        "Birkhoff multiphase problem structure: phases=%d, total_states=%d, total_controls=%d, static_params=%d",
        len(phase_ids),
        total_states,
        total_controls,
        num_static_params,
    )

    config = _create_birkhoff_solver_configuration(problem, grid_points_per_phase)

    logger.debug("Processing %d Birkhoff phases", len(phase_ids))
    _process_all_birkhoff_phases(config)

    logger.debug("Setting up Birkhoff multiphase objective and cross-phase constraints")
    _setup_birkhoff_objective_and_constraints(config)

    logger.debug("Applying Birkhoff multiphase initial guess")
    _apply_birkhoff_multiphase_initial_guess(
        config.opti, config.variables, problem, config.grid_points_per_phase
    )

    logger.debug("Configuring Birkhoff NLP solver")
    _configure_birkhoff_solver(config)

    logger.debug("Executing Birkhoff multiphase NLP solve")
    return _execute_birkhoff_solve(config)
