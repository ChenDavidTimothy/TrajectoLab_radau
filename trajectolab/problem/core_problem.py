import logging
from collections.abc import Sequence
from typing import Any

import casadi as ca

from ..input_validation import (
    validate_constraint_input_format,
    validate_string_not_empty,
)
from ..tl_types import FloatArray, NumericArrayLike, PhaseID
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .constraints_problem import (
    get_cross_phase_event_constraints_function,
    get_phase_path_constraints_function,
)
from .state import ConstraintInput, MultiPhaseVariableState
from .variables_problem import StateVariableImpl, TimeVariableImpl


logger = logging.getLogger(__name__)


class Phase:
    """Direct phase object for defining phase-specific variables and constraints."""

    def __init__(self, problem: "Problem", phase_id: PhaseID) -> None:
        self.problem = problem
        self.phase_id = phase_id
        self._phase_def = self.problem._multiphase_state.set_phase(self.phase_id)

    def time(
        self, initial: ConstraintInput = 0.0, final: ConstraintInput = None
    ) -> TimeVariableImpl:
        """Define time variable for this phase.

        Args:
            initial: Initial time constraint (default: 0.0)
            final: Final time constraint (default: free)

        Returns:
            TimeVariableImpl: Time variable with initial/final properties

        Examples:
            >>> # Fixed initial time, free final time
            >>> t = phase.time(initial=0.0)
            >>>
            >>> # Both times constrained
            >>> t = phase.time(initial=0.0, final=10.0)
            >>>
            >>> # Time bounds
            >>> t = phase.time(initial=(0.0, 5.0), final=(10.0, 20.0))
        """
        return variables_problem.create_phase_time_variable(self._phase_def, initial, final)

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> StateVariableImpl:
        """Define state variable for this phase.

        Args:
            name: Unique name for the state variable
            initial: Initial state constraint
            final: Final state constraint
            boundary: Constraint applied to both initial and final values

        Returns:
            StateVariableImpl: State variable with initial/final properties

        Examples:
            >>> # Altitude with initial condition
            >>> h = phase.state("altitude", initial=0.0)
            >>>
            >>> # Velocity with bounds at both ends
            >>> v = phase.state("velocity", initial=(0, 100), final=(50, 200))
            >>>
            >>> # Mass with boundary constraint (same at start and end)
            >>> m = phase.state("mass", boundary=1000.0)
        """
        return variables_problem._create_phase_state_variable(
            self._phase_def, name, initial, final, boundary
        )

    def control(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        """Define control variable for this phase.

        Args:
            name: Unique name for the control variable
            boundary: Constraint applied throughout the phase

        Returns:
            ca.MX: Control variable symbol

        Examples:
            >>> # Thrust with limits
            >>> thrust = phase.control("thrust", boundary=(0, 100))
            >>>
            >>> # Steering angle with symmetric bounds
            >>> delta = phase.control("steering", boundary=(-30, 30))
        """
        return variables_problem.create_phase_control_variable(self._phase_def, name, boundary)

    def dynamics(
        self,
        dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int | StateVariableImpl],
    ) -> None:
        """Define dynamics for this phase.

        Specifies the differential equations that govern state evolution.
        Each state variable must have its derivative defined.

        Args:
            dynamics_dict: Mapping from state variables to their derivatives

        Examples:
            >>> # Simple integrator dynamics
            >>> phase.dynamics({
            ...     position: velocity,
            ...     velocity: acceleration
            ... })
            >>>
            >>> # Rocket dynamics with gravity
            >>> phase.dynamics({
            ...     altitude: velocity,
            ...     velocity: thrust/mass - 9.81,
            ...     mass: -fuel_flow_rate
            ... })
        """
        variables_problem._set_phase_dynamics(self._phase_def, dynamics_dict)
        logger.info(
            "Dynamics defined for phase %d with %d state variables",
            self.phase_id,
            len(dynamics_dict),
        )

    def add_integral(self, integrand_expr: ca.MX | float | int) -> ca.MX:
        """Add integral expression for this phase.

        Integrals are computed automatically during trajectory optimization
        and can represent quantities like total fuel consumption or flight time.

        Args:
            integrand_expr: Expression to integrate over the phase

        Returns:
            ca.MX: Symbol representing the integral value

        Examples:
            >>> # Total fuel consumed
            >>> fuel_used = phase.add_integral(fuel_flow_rate)
            >>>
            >>> # Quadratic cost on control effort
            >>> control_cost = phase.add_integral(thrust**2)
        """
        return variables_problem._set_phase_integral(self._phase_def, integrand_expr)

    def path_constraints(self, *constraint_expressions: ca.MX | float | int) -> None:
        """Add path constraints for this phase.

        Path constraints are enforced at every collocation point throughout the phase,
        ensuring the trajectory satisfies operational limits and safety requirements.

        Args:
            *constraint_expressions: Constraint expressions using comparison operators
                                   (>=, <=, ==) or symbolic expressions

        Examples:
            >>> # Altitude and velocity limits for safety
            >>> phase.path_constraints(
            ...     altitude >= 1000,
            ...     velocity <= max_velocity,
            ...     thrust >= 0
            ... )
            >>>
            >>> # Combined performance envelope
            >>> phase.path_constraints(
            ...     altitude + velocity <= combined_limit,
            ...     dynamic_pressure <= max_q
            ... )
        """
        if not constraint_expressions:
            raise ValueError(
                f"Phase {self.phase_id} path_constraints() requires at least one constraint expression"
            )

        for expr in constraint_expressions:
            constraints_problem.add_path_constraint(self._phase_def, expr)

        logger.debug(
            "Added %d path constraint(s) to phase %d", len(constraint_expressions), self.phase_id
        )

    def event_constraints(self, *constraint_expressions: ca.MX | float | int) -> None:
        """Add event constraints for this phase.

        Event constraints are enforced only at phase boundaries, enabling
        precise specification of initial/final conditions and continuity requirements.

        Args:
            *constraint_expressions: Constraint expressions involving .initial or .final
                                   properties of state variables

        Examples:
            >>> # Launch and orbit insertion conditions
            >>> phase.event_constraints(
            ...     altitude.initial == 0,
            ...     velocity.initial == 0,
            ...     altitude.final >= target_altitude,
            ...     velocity.final >= orbital_speed
            ... )
            >>>
            >>> # Energy conservation
            >>> phase.event_constraints(
            ...     total_energy.final >= minimum_orbital_energy
            ... )
        """
        if not constraint_expressions:
            raise ValueError(
                f"Phase {self.phase_id} event_constraints() requires at least one constraint expression"
            )

        for expr in constraint_expressions:
            constraints_problem.add_event_constraint(self.problem._multiphase_state, expr)

        logger.debug(
            "Added %d event constraint(s) to phase %d", len(constraint_expressions), self.phase_id
        )

    def mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """Configure mesh for this phase.

        Defines the discretization strategy for numerical integration.
        Higher polynomial degrees provide better accuracy but increase computational cost.

        Args:
            polynomial_degrees: Polynomial degree for each mesh interval
            mesh_points: Normalized mesh points in [-1, 1]

        Examples:
            >>> # Uniform mesh with degree 3 polynomials
            >>> phase.mesh([3, 3, 3], [-1, -0.5, 0, 0.5, 1])
            >>>
            >>> # Adaptive mesh with higher resolution near boundaries
            >>> phase.mesh([5, 3, 5], [-1, -0.8, 0.8, 1])
        """
        logger.info(
            "Setting mesh for phase %d: %d intervals", self.phase_id, len(polynomial_degrees)
        )
        mesh._configure_phase_mesh(self._phase_def, polynomial_degrees, mesh_points)


class Problem:
    """Main class for defining multiphase optimal control problems.

    Provides a builder pattern interface for constructing complex trajectory
    optimization problems with multiple phases, constraints, and objectives.
    """

    def __init__(self, name: str = "Multiphase Problem") -> None:
        """Initialize a new multiphase problem instance.

        Args:
            name: Descriptive name for the problem (used in logging)
        """
        validate_string_not_empty(name, "Problem name")
        self.name = name
        logger.debug("Created multiphase problem: '%s'", name)

        self._multiphase_state = MultiPhaseVariableState()
        self._initial_guess_container = [None]
        self.solver_options: dict[str, Any] = {}

    def set_phase(self, phase_id: PhaseID) -> Phase:
        """Add a new phase and return the phase object for direct manipulation.

        Phases enable modeling of distinct flight regimes, each with their own
        dynamics, constraints, and mesh discretization.

        Args:
            phase_id: Unique integer identifier for the phase

        Returns:
            Phase: Phase object for defining variables and constraints

        Examples:
            >>> # Multi-stage rocket with separate phases
            >>> boost_phase = problem.set_phase(0)
            >>> coast_phase = problem.set_phase(1)
            >>> landing_phase = problem.set_phase(2)
        """
        if phase_id in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} already exists")

        logger.debug("Adding phase %d to problem '%s'", phase_id, self.name)
        return Phase(self, phase_id)

    def parameter(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        """Define a static parameter that spans across all phases.

        Static parameters represent design variables or constants that remain
        fixed throughout all phases but can be optimized.

        Args:
            name: Unique name for the parameter
            boundary: Constraint applied to the parameter value

        Returns:
            ca.MX: Parameter symbol

        Examples:
            >>> # Vehicle mass (optimization variable)
            >>> vehicle_mass = problem.parameter("mass", boundary=(100, 1000))
            >>>
            >>> # Engine efficiency (fixed parameter)
            >>> efficiency = problem.parameter("eta", boundary=0.85)
        """
        validate_string_not_empty(name, "Parameter name")
        validate_constraint_input_format(boundary, f"parameter '{name}' boundary")

        param_var = variables_problem._create_static_parameter(
            self._multiphase_state.static_parameters, name, boundary
        )
        logger.debug("Static parameter created: name='%s'", name)
        return param_var

    def minimize(self, objective_expr: ca.MX | float | int) -> None:
        """Define the multiphase objective function to minimize.

        The objective drives the optimization toward the desired solution.
        Common objectives include fuel consumption, flight time, or tracking error.

        Args:
            objective_expr: Expression to minimize (can reference phase integrals,
                          endpoint states, or static parameters)

        Examples:
            >>> # Minimize total fuel consumption across all phases
            >>> problem.minimize(fuel_phase1 + fuel_phase2 + fuel_phase3)
            >>>
            >>> # Minimize final time (time-optimal control)
            >>> problem.minimize(final_time)
            >>>
            >>> # Weighted combination of fuel and time
            >>> problem.minimize(0.8 * total_fuel + 0.2 * flight_time)
        """
        variables_problem._set_multiphase_objective(self._multiphase_state, objective_expr)
        logger.info("Multiphase objective function defined")

    def guess(
        self,
        phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_initial_times: dict[PhaseID, float] | None = None,
        phase_terminal_times: dict[PhaseID, float] | None = None,
        phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
        static_parameters: FloatArray | None = None,
    ) -> None:
        """Set initial guess for the multiphase optimization variables.

        Good initial guesses significantly improve convergence speed and success rate.
        Guesses should be physically reasonable and satisfy basic constraints.

        Args:
            phase_states: State trajectories for each phase and mesh interval
            phase_controls: Control trajectories for each phase and mesh interval
            phase_initial_times: Initial time for each phase
            phase_terminal_times: Final time for each phase
            phase_integrals: Integral values for each phase
            static_parameters: Values for static parameters

        Examples:
            >>> # Linear interpolation guess for single phase
            >>> problem.guess(
            ...     phase_states={0: [np.linspace([0, 0], [100, 50], 11).T]},
            ...     phase_controls={0: [np.ones((1, 10)) * 10]},
            ...     phase_initial_times={0: 0.0},
            ...     phase_terminal_times={0: 10.0}
            ... )
        """
        components = []
        if phase_states is not None:
            components.append(f"states({len(phase_states)} phases)")
        if phase_controls is not None:
            components.append(f"controls({len(phase_controls)} phases)")
        if static_parameters is not None:
            components.append(f"parameters({len(static_parameters)})")

        logger.info("Setting multiphase initial guess: %s", ", ".join(components))

        initial_guess_problem._set_multiphase_initial_guess(
            self._initial_guess_container,
            self._multiphase_state,
            phase_states=phase_states,
            phase_controls=phase_controls,
            phase_initial_times=phase_initial_times,
            phase_terminal_times=phase_terminal_times,
            phase_integrals=phase_integrals,
            static_parameters=static_parameters,
        )

    # ========================================================================
    # PROTOCOL INTERFACE - Required by ProblemProtocol
    # ========================================================================

    @property
    def _phases(self) -> dict[PhaseID, Any]:
        return self._multiphase_state.phases

    @property
    def _static_parameters(self) -> Any:
        return self._multiphase_state.static_parameters

    @property
    def _cross_phase_constraints(self) -> list[ca.MX]:
        return self._multiphase_state.cross_phase_constraints

    @property
    def _num_phases(self) -> int:
        return len(self._multiphase_state.phases)

    @property
    def initial_guess(self):
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        self._initial_guess_container[0] = value

    def _get_phase_ids(self) -> list[PhaseID]:
        return self._multiphase_state._get_phase_ids()

    def get_phase_variable_counts(self, phase_id: PhaseID) -> tuple[int, int]:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].get_variable_counts()

    def get_total_variable_counts(self) -> tuple[int, int, int]:
        return self._multiphase_state.get_total_variable_counts()

    def get_phase_ordered_state_names(self, phase_id: PhaseID) -> list[str]:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].state_names.copy()

    def get_phase_ordered_control_names(self, phase_id: PhaseID) -> list[str]:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].control_names.copy()

    def _get_phase_dynamics_function(self, phase_id: PhaseID) -> Any:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")

        static_parameter_symbols = (
            self._multiphase_state.static_parameters.get_ordered_parameter_symbols()
        )
        return solver_interface._get_phase_dynamics_function(
            self._multiphase_state.phases[phase_id], static_parameter_symbols
        )

    def _get_objective_function(self) -> Any:
        return solver_interface._get_multiphase_objective_function(self._multiphase_state)

    def _get_phase_integrand_function(self, phase_id: PhaseID) -> Any:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")

        static_parameter_symbols = (
            self._multiphase_state.static_parameters.get_ordered_parameter_symbols()
        )
        return solver_interface._get_phase_integrand_function(
            self._multiphase_state.phases[phase_id], static_parameter_symbols
        )

    def get_phase_path_constraints_function(self, phase_id: PhaseID) -> Any:
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return get_phase_path_constraints_function(self._multiphase_state.phases[phase_id])

    def get_cross_phase_event_constraints_function(self) -> Any:
        return get_cross_phase_event_constraints_function(self._multiphase_state)

    def validate_multiphase_configuration(self) -> None:
        logger.debug("Processing symbolic boundary constraints for automatic cross-phase linking")

        # Convert symbolic boundary constraints to cross-phase constraints for solver compatibility
        for phase_id, phase_def in self._multiphase_state.phases.items():
            # Process time constraints
            if phase_def.t0_constraint.is_symbolic():
                if (
                    phase_def.sym_time_initial is None
                    or phase_def.t0_constraint.symbolic_expression is None
                ):
                    raise ValueError(
                        f"Phase {phase_id} has an undefined symbolic time initial expression."
                    )
                constraint_expr = (
                    phase_def.sym_time_initial - phase_def.t0_constraint.symbolic_expression
                )
                self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                logger.debug(f"Added automatic time initial constraint for phase {phase_id}")

            if phase_def.tf_constraint.is_symbolic():
                if (
                    phase_def.sym_time_final is None
                    or phase_def.tf_constraint.symbolic_expression is None
                ):
                    raise ValueError(
                        f"Phase {phase_id} has an undefined symbolic time final expression."
                    )
                constraint_expr = (
                    phase_def.sym_time_final - phase_def.tf_constraint.symbolic_expression
                )
                self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                logger.debug(f"Added automatic time final constraint for phase {phase_id}")

            # Process state symbolic boundary constraints
            for var_name, constraint_type, symbolic_expr in phase_def.symbolic_boundary_constraints:
                state_index = phase_def.state_name_to_index[var_name]

                if constraint_type == "initial":
                    state_initial_sym = phase_def.get_ordered_state_initial_symbols()[state_index]
                    constraint_expr = state_initial_sym - symbolic_expr
                    self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                    logger.debug(
                        f"Added automatic initial constraint for phase {phase_id} state '{var_name}'"
                    )

                elif constraint_type == "final":
                    state_final_sym = phase_def.get_ordered_state_final_symbols()[state_index]
                    constraint_expr = state_final_sym - symbolic_expr
                    self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                    logger.debug(
                        f"Added automatic final constraint for phase {phase_id} state '{var_name}'"
                    )

                elif constraint_type == "boundary":
                    state_initial_sym = phase_def.get_ordered_state_initial_symbols()[state_index]
                    state_final_sym = phase_def.get_ordered_state_final_symbols()[state_index]

                    initial_constraint = state_initial_sym - symbolic_expr
                    final_constraint = state_final_sym - symbolic_expr

                    self._multiphase_state.cross_phase_constraints.extend(
                        [initial_constraint, final_constraint]
                    )
                    logger.debug(
                        f"Added automatic boundary constraints for phase {phase_id} state '{var_name}'"
                    )

        if not self._multiphase_state.phases:
            raise ValueError("Problem must have at least one phase defined")

        for phase_id, phase_def in self._multiphase_state.phases.items():
            if not phase_def.dynamics_expressions:
                raise ValueError(f"Phase {phase_id} must have dynamics defined")
            if not phase_def.mesh_configured:
                raise ValueError(f"Phase {phase_id} must have mesh configured")

        if self._multiphase_state.objective_expression is None:
            raise ValueError("Problem must have objective function defined")

        logger.debug(
            "Multiphase configuration validated: %d phases, %d cross-phase constraints",
            len(self._multiphase_state.phases),
            len(self._multiphase_state.cross_phase_constraints),
        )
