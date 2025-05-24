# trajectolab/problem/core_problem.py
"""
Core problem definition with production logging.
"""

import logging
from collections.abc import Sequence
from typing import Any

from ..tl_types import FloatArray, NumericArrayLike, SymExpr, SymType
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .state import ConstraintInput, ConstraintState, MeshState, VariableState


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


class Problem:
    """Main class for defining optimal control problems."""

    def __init__(self, name: str = "Unnamed Problem") -> None:
        """Initialize a new problem instance."""
        self.name = name

        # Log problem creation (DEBUG - developer info)
        logger.debug("Created problem: '%s'", name)

        # State containers
        self._variable_state = VariableState()
        self._constraint_state = ConstraintState()
        self._mesh_state = MeshState()
        self._initial_guess_container = [None]
        self.solver_options: dict[str, Any] = {}

    # ========================================================================
    # UNIFIED PROPERTIES - Direct access to optimized storage
    # ========================================================================

    @property
    def _parameters(self) -> dict[str, Any]:
        return self._variable_state.parameters

    @property
    def _sym_time(self) -> SymType | None:
        return self._variable_state.sym_time

    @property
    def _sym_time_initial(self) -> SymType | None:
        return self._variable_state.sym_time_initial

    @property
    def _sym_time_final(self) -> SymType | None:
        return self._variable_state.sym_time_final

    @property
    def _t0_bounds(self) -> tuple[float, float]:
        return self._variable_state.t0_bounds

    @property
    def _tf_bounds(self) -> tuple[float, float]:
        return self._variable_state.tf_bounds

    @property
    def _dynamics_expressions(self) -> dict[SymType, SymExpr]:
        return self._variable_state.dynamics_expressions

    @property
    def _objective_expression(self) -> SymExpr | None:
        return self._variable_state.objective_expression

    @property
    def _constraints(self) -> list[SymExpr]:
        return self._constraint_state.constraints

    @property
    def _integral_expressions(self) -> list[SymExpr]:
        return self._variable_state.integral_expressions

    @property
    def _integral_symbols(self) -> list[SymType]:
        return self._variable_state.integral_symbols

    @property
    def _num_integrals(self) -> int:
        return self._variable_state.num_integrals

    @property
    def collocation_points_per_interval(self) -> list[int]:
        return self._mesh_state.collocation_points_per_interval

    @property
    def global_normalized_mesh_nodes(self) -> FloatArray | None:
        return self._mesh_state.global_normalized_mesh_nodes

    @property
    def _mesh_configured(self) -> bool:
        return self._mesh_state.configured

    @property
    def initial_guess(self):
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        self._initial_guess_container[0] = value

    # ========================================================================
    # PROTOCOL INTERFACE METHODS - Required by ProblemProtocol
    # ========================================================================

    def get_variable_counts(self) -> tuple[int, int]:
        """Return (num_states, num_controls)."""
        return self._variable_state.get_variable_counts()

    def get_ordered_state_symbols(self) -> list[SymType]:
        """Get state symbols in order."""
        return self._variable_state.get_ordered_state_symbols()

    def get_ordered_control_symbols(self) -> list[SymType]:
        """Get control symbols in order."""
        return self._variable_state.get_ordered_control_symbols()

    def get_ordered_state_names(self) -> list[str]:
        """Get state names in order."""
        return self._variable_state.get_ordered_state_names()

    def get_ordered_control_names(self) -> list[str]:
        """Get control names in order."""
        return self._variable_state.get_ordered_control_names()

    def get_state_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get state bounds in order (compatibility method)."""
        # Convert boundary constraints to bounds for compatibility
        boundary_constraints = self._variable_state.get_state_boundary_constraints()
        # Fix: Explicitly type the bounds list
        bounds: list[tuple[float | None, float | None]] = []
        for constraint in boundary_constraints:
            if constraint is None or not constraint.has_constraint():
                bounds.append((None, None))
            elif constraint.equals is not None:
                bounds.append((constraint.equals, constraint.equals))
            else:
                bounds.append((constraint.lower, constraint.upper))
        return bounds

    def get_control_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get control bounds in order (compatibility method)."""
        # Convert boundary constraints to bounds for compatibility
        boundary_constraints = self._variable_state.get_control_boundary_constraints()
        # Fix: Explicitly type the bounds list
        bounds: list[tuple[float | None, float | None]] = []
        for constraint in boundary_constraints:
            if constraint is None or not constraint.has_constraint():
                bounds.append((None, None))
            elif constraint.equals is not None:
                bounds.append((constraint.equals, constraint.equals))
            else:
                bounds.append((constraint.lower, constraint.upper))
        return bounds

    # ========================================================================
    # UNIFIED CONSTRAINT API METHODS - New API
    # ========================================================================

    def time(
        self,
        initial: ConstraintInput = 0.0,
        final: ConstraintInput = None,
    ) -> variables_problem.TimeVariableImpl:
        """Define time variable with unified constraint specification."""
        time_var = variables_problem.create_time_variable(self._variable_state, initial, final)

        # Log time variable creation (DEBUG - developer info)
        logger.debug("Time variable created: initial=%s, final=%s", initial, final)

        return time_var

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> SymType:
        """Define a state variable with unified constraint specification."""
        state_var = variables_problem.create_state_variable(
            self._variable_state, name, initial, final, boundary
        )

        # Log state variable creation (DEBUG - developer info)
        logger.debug(
            "State variable created: name='%s', initial=%s, final=%s, boundary=%s",
            name,
            initial,
            final,
            boundary,
        )

        return state_var

    def control(
        self,
        name: str,
        boundary: ConstraintInput = None,
    ) -> SymType:
        """Define a control variable with path constraints."""
        control_var = variables_problem.create_control_variable(
            self._variable_state, name, boundary
        )

        # Log control variable creation (DEBUG - developer info)
        logger.debug("Control variable created: name='%s', boundary=%s", name, boundary)

        return control_var

    def parameter(self, name: str, value: Any) -> SymType:
        """Define a parameter variable."""
        param_var = variables_problem.create_parameter_variable(self._variable_state, name, value)

        # Log parameter creation (DEBUG)
        logger.debug("Parameter created: name='%s', value=%s", name, value)

        return param_var

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
        """Define system dynamics."""
        variables_problem.set_dynamics(self._variable_state, dynamics_dict)

        # Log dynamics definition (INFO - user cares about major setup)
        logger.info("Dynamics defined for %d state variables", len(dynamics_dict))

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        """Add an integral expression."""
        integral_var = variables_problem.add_integral(self._variable_state, integrand_expr)

        # Log integral addition (DEBUG)
        logger.debug("Integral added: total_integrals=%d", self._variable_state.num_integrals)

        return integral_var

    def minimize(self, objective_expr: SymExpr) -> None:
        """Define the objective function to minimize."""
        variables_problem.set_objective(self._variable_state, objective_expr)

        # Log objective definition (INFO - user cares about major setup)
        logger.info("Objective function defined")

    def subject_to(self, constraint_expr: SymExpr) -> None:
        """Add a constraint to the problem."""
        constraints_problem.add_constraint(self._constraint_state, constraint_expr)

        # Log constraint addition (DEBUG)
        logger.debug(
            "Constraint added: total_constraints=%d", len(self._constraint_state.constraints)
        )

    def set_mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """Configure mesh structure for the problem."""

        # Log mesh configuration start (INFO - user cares about major setup)
        logger.info(
            "Setting mesh: %d intervals, degrees=%s", len(polynomial_degrees), polynomial_degrees
        )

        mesh.configure_mesh(self._mesh_state, polynomial_degrees, mesh_points)

        # Log successful mesh configuration (INFO)
        logger.info("Mesh configured successfully: %d intervals", len(polynomial_degrees))

    def set_initial_guess(
        self,
        states: Sequence[FloatArray] | None = None,
        controls: Sequence[FloatArray] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """Set initial guess for the problem."""

        # Log initial guess setup (INFO - user cares about major setup)
        components = []
        if states is not None:
            components.append(f"states({len(states)} intervals)")
        if controls is not None:
            components.append(f"controls({len(controls)} intervals)")
        if initial_time is not None:
            components.append("initial_time")
        if terminal_time is not None:
            components.append("terminal_time")
        if integrals is not None:
            components.append("integrals")

        logger.info("Setting initial guess: %s", ", ".join(components))

        initial_guess_problem.set_initial_guess(
            self._initial_guess_container,
            self._mesh_state,
            self._variable_state,
            states=states,
            controls=controls,
            initial_time=initial_time,
            terminal_time=terminal_time,
            integrals=integrals,
        )

        # Log validation status (DEBUG)
        if self.can_validate_initial_guess():
            try:
                self.validate_initial_guess()
                logger.debug("Initial guess validated successfully")
            except Exception as e:
                logger.debug("Initial guess validation deferred: %s", str(e))
        else:
            logger.debug("Initial guess validation deferred until mesh is configured")

    def can_validate_initial_guess(self) -> bool:
        """Check if we have enough information to validate the initial guess."""
        return initial_guess_problem.can_validate_initial_guess(
            self._mesh_state, self._variable_state
        )

    def get_initial_guess_requirements(self):
        """
        Get initial guess requirements.

        FIXED: Now handles case where mesh isn't configured yet.
        """
        requirements = initial_guess_problem.get_initial_guess_requirements(
            self._mesh_state, self._variable_state
        )

        if not self._mesh_configured:
            print("Note: Mesh must be configured to get specific shape requirements")

        return requirements

    def validate_initial_guess(self) -> None:
        """
        Validate the current initial guess.

        FIXED: Provides clear error message if mesh not configured.
        This is called automatically when the solver runs.
        """
        initial_guess_problem.validate_initial_guess(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    def get_solver_input_summary(self):
        """Get solver input summary."""
        return initial_guess_problem.get_solver_input_summary(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    # ========================================================================
    # SOLVER INTERFACE METHODS
    # ========================================================================

    def get_dynamics_function(self):
        """Get dynamics function for solver."""
        return solver_interface.get_dynamics_function(self._variable_state)

    def get_objective_function(self):
        """Get objective function for solver."""
        return solver_interface.get_objective_function(self._variable_state)

    def get_integrand_function(self):
        """Get integrand function for solver."""
        return solver_interface.get_integrand_function(self._variable_state)

    def get_path_constraints_function(self):
        """Get path constraints function for solver."""
        return solver_interface.get_path_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )

    def get_event_constraints_function(self):
        """Get event constraints function for solver."""
        return solver_interface.get_event_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )
