import logging

import casadi as ca

from ..exceptions import DataIntegrityError, SolutionExtractionError
from ..radau import RadauBasisComponents, compute_radau_collocation_components
from ..solution_extraction import extract_and_format_multiphase_solution
from ..tl_types import (
    OptimalControlSolution,
    PhaseID,
    ProblemProtocol,
)
from .constraints_solver import (
    apply_multiphase_cross_phase_event_constraints,
    apply_phase_collocation_constraints,
    apply_phase_path_constraints,
)
from .initial_guess_solver import apply_multiphase_initial_guess
from .integrals_solver import apply_phase_integral_constraints, setup_phase_integrals
from .types_solver import MultiPhaseVariableReferences
from .variables_solver import (
    setup_multiphase_optimization_variables,
    setup_phase_interval_state_variables,
)


logger = logging.getLogger(__name__)


def _extract_phase_endpoint_data(
    variables: MultiPhaseVariableReferences, problem: ProblemProtocol
) -> dict[PhaseID, dict[str, ca.MX]]:
    """Extract phase endpoint data once for reuse across all solver components."""
    phase_endpoint_data = {}

    for phase_id, phase_vars in variables.phase_variables.items():
        # Use problem data directly instead of duplicated metadata
        num_mesh_intervals = len(problem._phases[phase_id].collocation_points_per_interval)

        initial_state = phase_vars.state_at_mesh_nodes[0]
        terminal_state = phase_vars.state_at_mesh_nodes[num_mesh_intervals]

        phase_endpoint_data[phase_id] = {
            "t0": phase_vars.initial_time,
            "tf": phase_vars.terminal_time,
            "x0": initial_state,
            "xf": terminal_state,
            "q": phase_vars.integral_variables,
        }

    return phase_endpoint_data


def solve_multiphase_radau_collocation(problem: ProblemProtocol) -> OptimalControlSolution:
    """
    Solve a multiphase optimal control problem using Radau pseudospectral collocation.
    """
    logger.debug("Starting multiphase Radau collocation solver")

    opti: ca.Opti = ca.Opti()
    phase_ids = problem.get_phase_ids()
    total_states, total_controls, num_static_params = problem.get_total_variable_counts()

    logger.debug(
        "Multiphase problem structure: phases=%d, total_states=%d, total_controls=%d, static_params=%d",
        len(phase_ids),
        total_states,
        total_controls,
        num_static_params,
    )

    try:
        # Set up multiphase optimization variables
        logger.debug("Setting up multiphase optimization variables")
        variables = setup_multiphase_optimization_variables(opti, problem)

        # Extract phase endpoint data once for reuse
        logger.debug("Extracting unified phase endpoint data")
        phase_endpoint_data = _extract_phase_endpoint_data(variables, problem)

        # Process all phases with unified data
        logger.debug("Processing %d phases", len(phase_ids))
        _process_all_phases_unified(opti, variables, problem, phase_endpoint_data)

        # Set up multiphase objective and constraints using shared endpoint data
        logger.debug("Setting up multiphase objective and cross-phase constraints")
        _setup_objective_and_constraints_unified(opti, variables, problem, phase_endpoint_data)

        # Apply multiphase initial guess
        logger.debug("Applying multiphase initial guess")
        apply_multiphase_initial_guess(opti, variables, problem)

        # Configure solver with shared endpoint data
        logger.debug("Configuring NLP solver")
        _configure_solver_unified(opti, variables, problem, phase_endpoint_data)

    except Exception as e:
        logger.error("Failed to set up multiphase optimization problem: %s", str(e))
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(
            f"Failed to set up multiphase optimization problem: {e}",
            "TrajectoLab multiphase problem construction error",
        ) from e

    # Execute solve
    logger.debug("Executing multiphase NLP solve")
    solution_obj = _execute_multiphase_solve(opti, problem)

    return solution_obj


def _process_all_phases_unified(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: ProblemProtocol,
    phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]],
) -> None:
    """Process all phases with unified endpoint data to eliminate redundant extraction."""
    for phase_id in problem.get_phase_ids():
        if phase_id not in variables.phase_variables:
            continue

        phase_vars = variables.phase_variables[phase_id]
        endpoint_data = phase_endpoint_data[phase_id]

        _process_single_phase_unified(
            opti,
            phase_vars,
            problem,
            phase_id,
            variables.static_parameters,
            endpoint_data,
        )


def _process_single_phase_unified(
    opti: ca.Opti,
    phase_vars,
    problem: ProblemProtocol,
    phase_id: PhaseID,
    static_parameters_vec: ca.MX | None = None,
    endpoint_data: dict[str, ca.MX] | None = None,
) -> None:
    """Process a single phase with unified endpoint data."""
    # Get phase information directly from problem (no metadata duplication)
    num_states, num_controls = problem.get_phase_variable_counts(phase_id)
    phase_def = problem._phases[phase_id]
    num_mesh_intervals = len(phase_def.collocation_points_per_interval)
    num_integrals = phase_def.num_integrals

    # Get essential functions
    dynamics_function = problem.get_phase_dynamics_function(phase_id)
    path_constraints_function = problem.get_phase_path_constraints_function(phase_id)
    integral_integrand_function = problem.get_phase_integrand_function(phase_id)

    # Use problem data directly (no duplication in metadata)
    global_mesh_nodes = phase_def.global_normalized_mesh_nodes

    # Get static parameter symbols
    static_parameter_symbols = None
    if static_parameters_vec is not None:
        static_parameter_symbols = problem._static_parameters.get_ordered_parameter_symbols()

    # Data integrity validation
    if len(global_mesh_nodes) != num_mesh_intervals + 1:
        raise DataIntegrityError(
            f"Phase {phase_id} mesh nodes count ({len(global_mesh_nodes)}) doesn't match expected ({num_mesh_intervals + 1})",
            "Phase mesh configuration inconsistency",
        )

    # Initialize accumulated integrals
    accumulated_integral_expressions: list[ca.MX] = (
        [ca.MX(0) for _ in range(num_integrals)] if num_integrals > 0 else []
    )

    # Extract time variables from unified endpoint data
    initial_time_var = endpoint_data["t0"] if endpoint_data else phase_vars.initial_time
    terminal_time_var = endpoint_data["tf"] if endpoint_data else phase_vars.terminal_time

    for mesh_interval_index in range(num_mesh_intervals):
        num_colloc_nodes = phase_def.collocation_points_per_interval[mesh_interval_index]

        try:
            # Set up state variables for this interval
            state_at_nodes, interior_nodes_var = setup_phase_interval_state_variables(
                opti,
                phase_id,
                mesh_interval_index,
                num_states,
                num_colloc_nodes,
                phase_vars.state_at_mesh_nodes,
            )

            # Store state variables and interior nodes
            phase_vars.state_matrices.append(state_at_nodes)
            phase_vars.interior_variables.append(interior_nodes_var)

            # Get Radau collocation components
            basis_components: RadauBasisComponents = compute_radau_collocation_components(
                num_colloc_nodes
            )

            # Apply collocation constraints
            apply_phase_collocation_constraints(
                opti,
                phase_id,
                mesh_interval_index,
                state_at_nodes,
                phase_vars.control_variables[mesh_interval_index],
                basis_components,
                global_mesh_nodes,
                initial_time_var,
                terminal_time_var,
                dynamics_function,
                problem,
                static_parameters_vec,
            )

            # Apply path constraints if they exist
            if path_constraints_function is not None:
                apply_phase_path_constraints(
                    opti,
                    phase_id,
                    mesh_interval_index,
                    state_at_nodes,
                    phase_vars.control_variables[mesh_interval_index],
                    basis_components,
                    global_mesh_nodes,
                    initial_time_var,
                    terminal_time_var,
                    path_constraints_function,
                    problem,
                    static_parameters_vec,
                    static_parameter_symbols,
                )

            # Set up integrals if they exist
            if num_integrals > 0 and integral_integrand_function is not None:
                setup_phase_integrals(
                    opti,
                    phase_id,
                    mesh_interval_index,
                    state_at_nodes,
                    phase_vars.control_variables[mesh_interval_index],
                    basis_components,
                    global_mesh_nodes,
                    initial_time_var,
                    terminal_time_var,
                    integral_integrand_function,
                    num_integrals,
                    accumulated_integral_expressions,
                    static_parameters_vec,
                )

        except Exception as e:
            if isinstance(e, DataIntegrityError):
                raise
            raise DataIntegrityError(
                f"Failed to process phase {phase_id} interval {mesh_interval_index}: {e}",
                "TrajectoLab phase interval setup error",
            ) from e

    # Apply integral constraints for this phase
    if num_integrals > 0 and phase_vars.integral_variables is not None:
        apply_phase_integral_constraints(
            opti,
            phase_vars.integral_variables,
            accumulated_integral_expressions,
            num_integrals,
            phase_id,
        )


def _setup_objective_and_constraints_unified(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: ProblemProtocol,
    phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]],
) -> None:
    """Set up objective and constraints using pre-extracted endpoint data."""
    # Get multiphase objective function
    objective_function = problem.get_objective_function()

    try:
        # Set up multiphase objective using shared endpoint data
        objective_value: ca.MX = objective_function(
            phase_endpoint_data,
            variables.static_parameters,
        )
        opti.minimize(objective_value)

    except Exception as e:
        raise DataIntegrityError(
            f"Failed to set up multiphase objective function: {e}",
            "Multiphase objective function evaluation error",
        ) from e

    # Apply cross-phase event constraints using shared endpoint data
    apply_multiphase_cross_phase_event_constraints(
        opti,
        phase_endpoint_data,
        variables.static_parameters,
        problem,
    )


def _configure_solver_unified(
    opti: ca.Opti,
    variables: MultiPhaseVariableReferences,
    problem: ProblemProtocol,
    phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]],
) -> None:
    """Configure solver options and store references using unified endpoint data."""
    # Set solver options
    solver_options_to_use: dict[str, object] = problem.solver_options or {}

    try:
        opti.solver("ipopt", solver_options_to_use)
    except Exception as e:
        raise DataIntegrityError(
            f"Failed to configure solver: {e}", "Invalid solver options"
        ) from e

    # Store references for solution extraction
    opti.multiphase_variables_reference = variables

    # Store objective expression using shared endpoint data
    objective_function = problem.get_objective_function()
    objective_expression = objective_function(phase_endpoint_data, variables.static_parameters)
    opti.multiphase_objective_expression_reference = objective_expression


def _execute_multiphase_solve(opti: ca.Opti, problem: ProblemProtocol) -> OptimalControlSolution:
    """Execute the multiphase solve and handle results."""
    try:
        # Attempt solve
        solver_solution: ca.OptiSol = opti.solve()

        # Log successful solve
        logger.debug("Multiphase NLP solver completed successfully")

        # Extract solution
        try:
            solution_obj = extract_and_format_multiphase_solution(solver_solution, opti, problem)
            logger.debug("Multiphase solution extraction completed")
        except Exception as e:
            logger.error("Multiphase solution extraction failed: %s", str(e))
            raise SolutionExtractionError(
                f"Failed to extract multiphase solution: {e}",
                "TrajectoLab multiphase solution processing error",
            ) from e

    except RuntimeError as e:
        # Log solver failure
        logger.warning("Multiphase NLP solver failed: %s", str(e))

        # Create failed solution object
        try:
            solution_obj = extract_and_format_multiphase_solution(None, opti, problem)
        except Exception as extract_error:
            logger.error(
                "Critical: Multiphase solution extraction failed after solver failure: %s",
                str(extract_error),
            )
            raise SolutionExtractionError(
                f"Failed to extract multiphase solution after solver failure: {extract_error}",
                "Critical multiphase solution extraction error",
            ) from extract_error

        solution_obj.success = False
        solution_obj.message = f"Multiphase solver runtime error: {e}"

        try:
            if hasattr(opti, "debug") and opti.debug is not None:
                variables = opti.multiphase_variables_reference
                for phase_id, phase_vars in variables.phase_variables.items():
                    try:
                        solution_obj.phase_initial_times[phase_id] = float(
                            opti.debug.value(phase_vars.initial_time)
                        )
                        solution_obj.phase_terminal_times[phase_id] = float(
                            opti.debug.value(phase_vars.terminal_time)
                        )
                    except Exception as e:
                        logger.debug(f"Could not extract debug values for phase {phase_id}: {e}")
                logger.debug("Retrieved debug values from failed multiphase solve")
        except Exception as e:
            logger.debug(f"Could not extract debug values from failed multiphase solve: {e}")

    return solution_obj
