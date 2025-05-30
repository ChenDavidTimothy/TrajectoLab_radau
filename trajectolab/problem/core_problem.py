"""
Core problem definition for optimal control problems.

This module provides the main Problem class that users interact with to define
optimal control problems using TrajectoLab's unified constraint API.
"""

import logging
from collections.abc import Sequence
from typing import Any

import casadi as ca

from ..tl_types import FloatArray, NumericArrayLike
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .state import ConstraintInput, ConstraintState, MeshState, VariableState
from .variables_problem import StateVariableImpl


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


class Problem:
    """
    Main class for defining optimal control problems.

    The Problem class provides a unified interface for defining optimal control problems
    using symbolic variables, constraints, and dynamics. It supports the complete
    workflow from problem definition to solution.

    Args:
        name: A descriptive name for the problem (used in logging and output)

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> # Create problem
        >>> problem = tl.Problem("Minimum Time")
        >>>
        >>> # Define variables
        >>> t = problem.time(initial=0.0)
        >>> x = problem.state("position", initial=0.0, final=1.0)
        >>> u = problem.control("thrust", boundary=(-1.0, 1.0))
        >>>
        >>> # Set dynamics and objective
        >>> problem.dynamics({x: u})
        >>> problem.minimize(t.final)
        >>>
        >>> # Configure and solve
        >>> problem.set_mesh([10], np.array([-1.0, 1.0]))
        >>> solution = tl.solve_fixed_mesh(problem)
    """

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
    def _sym_time(self) -> ca.MX | None:
        """Internal time symbol."""
        return self._variable_state.sym_time

    @property
    def _sym_time_initial(self) -> ca.MX | None:
        """Internal initial time symbol."""
        return self._variable_state.sym_time_initial

    @property
    def _sym_time_final(self) -> ca.MX | None:
        """Internal final time symbol."""
        return self._variable_state.sym_time_final

    @property
    def _t0_bounds(self) -> tuple[float, float]:
        """Internal initial time bounds."""
        return self._variable_state.t0_bounds

    @property
    def _tf_bounds(self) -> tuple[float, float]:
        """Internal final time bounds."""
        return self._variable_state.tf_bounds

    @property
    def _dynamics_expressions(self) -> dict[ca.MX, ca.MX]:
        """Internal dynamics expressions storage."""
        return self._variable_state.dynamics_expressions

    @property
    def _objective_expression(self) -> ca.MX | None:
        """Internal objective expression storage."""
        return self._variable_state.objective_expression

    @property
    def _constraints(self) -> list[ca.MX]:
        """Internal constraint expressions storage."""
        return self._constraint_state.constraints

    @property
    def _integral_expressions(self) -> list[ca.MX]:
        """Internal integral expressions storage."""
        return self._variable_state.integral_expressions

    @property
    def _integral_symbols(self) -> list[ca.MX]:
        """Internal integral symbols storage."""
        return self._variable_state.integral_symbols

    @property
    def _num_integrals(self) -> int:
        """Internal integral count."""
        return self._variable_state.num_integrals

    @property
    def collocation_points_per_interval(self) -> list[int]:
        """Collocation points configuration for each mesh interval."""
        return self._mesh_state.collocation_points_per_interval

    @property
    def global_normalized_mesh_nodes(self) -> FloatArray | None:
        """Global normalized mesh node positions."""
        return self._mesh_state.global_normalized_mesh_nodes

    @property
    def _mesh_configured(self) -> bool:
        """Whether the mesh has been configured."""
        return self._mesh_state.configured

    @property
    def initial_guess(self):
        """Current initial guess for the problem."""
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        """Set the initial guess for the problem."""
        self._initial_guess_container[0] = value

    # ========================================================================
    # PROTOCOL INTERFACE METHODS - Required by ProblemProtocol
    # ========================================================================

    def get_variable_counts(self) -> tuple[int, int]:
        """
        Get the number of state and control variables.

        Returns:
            Tuple of (num_states, num_controls)
        """
        return self._variable_state.get_variable_counts()

    def get_ordered_state_symbols(self) -> list[ca.MX]:
        """Get state variable symbols in definition order."""
        return self._variable_state.get_ordered_state_symbols()

    def get_ordered_control_symbols(self) -> list[ca.MX]:
        """Get control variable symbols in definition order."""
        return self._variable_state.get_ordered_control_symbols()

    def get_ordered_state_names(self) -> list[str]:
        """Get state variable names in definition order."""
        return self._variable_state.get_ordered_state_names()

    def get_ordered_control_names(self) -> list[str]:
        """Get control variable names in definition order."""
        return self._variable_state.get_ordered_control_names()

    def get_state_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get state variable bounds for compatibility."""
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
        """Get control variable bounds for compatibility."""
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
    # UNIFIED CONSTRAINT API METHODS - Public Interface
    # ========================================================================

    def time(
        self,
        initial: ConstraintInput = 0.0,
        final: ConstraintInput = None,
    ) -> variables_problem.TimeVariableImpl:
        """
        Define the time variable with constraint specification.

        Args:
            initial: Constraint on initial time. Can be:
                - float/int: Fixed initial time (default: 0.0)
                - tuple(lower, upper): Range constraint for t0
                - None: Treated as fixed at 0.0
            final: Constraint on final time. Can be:
                - float/int: Fixed final time
                - tuple(lower, upper): Range constraint for tf
                - None: Fully free (subject to tf > t0 + epsilon)

        Returns:
            TimeVariableImpl object with .initial and .final properties

        Example:
            >>> t = problem.time(initial=0.0)              # t0 = 0, tf free
            >>> t = problem.time(initial=0.0, final=10.0)  # Both fixed
            >>> t = problem.time(final=(5.0, 15.0))        # tf ∈ [5, 15]
        """
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
    ) -> variables_problem.StateVariableImpl:
        """
        Define a state variable with constraint specification and initial/final properties.

        Args:
            name: Variable name (must be unique)
            initial: Initial condition constraint (event constraint at t0):
                - float/int: Fixed value at t0
                - tuple(lower, upper): Range constraint at t0
                - None: No initial constraint
            final: Final condition constraint (event constraint at tf):
                - float/int: Fixed value at tf
                - tuple(lower, upper): Range constraint at tf
                - None: No final constraint
            boundary: Path constraint (applies throughout trajectory):
                - float/int: Fixed value for entire trajectory
                - tuple(lower, upper): Range constraint for entire trajectory
                - None: No path constraint

        Returns:
            StateVariableImpl object with .initial and .final properties

        Example:
            >>> x = problem.state("position", initial=0.0, final=1.0)
            >>> v = problem.state("velocity", boundary=(-10.0, 10.0))
            >>> y = problem.state("height", initial=(0.0, 5.0))
            >>>
            >>> # Use in dynamics (current state)
            >>> problem.dynamics({x: v, v: u})
            >>>
            >>> # Use in objective (endpoint values)
            >>> problem.minimize(x.final**2 + 0.1 * x.initial)
            >>>
            >>> # Use in constraints (endpoint values)
            >>> problem.subject_to(x.final >= x.initial + 5.0)
        """
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
    ) -> ca.MX:
        """
        Define a control variable with path constraints.

        Args:
            name: Variable name (must be unique)
            boundary: Path constraint (applies throughout trajectory):
                - float/int: Fixed value for entire trajectory
                - tuple(lower, upper): Range constraint for entire trajectory
                - None: No path constraint (unbounded - not recommended)

        Returns:
            CasADi symbolic variable for use in dynamics and constraints

        Example:
            >>> u = problem.control("thrust", boundary=(-1.0, 1.0))
            >>> throttle = problem.control("throttle", boundary=(0.0, 1.0))
            >>> steering = problem.control("steering")  # Unbounded
        """
        control_var = variables_problem.create_control_variable(
            self._variable_state, name, boundary
        )

        # Log control variable creation (DEBUG - developer info)
        logger.debug("Control variable created: name='%s', boundary=%s", name, boundary)

        return control_var

    def dynamics(
        self,
        dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int | StateVariableImpl],
    ) -> None:
        """
        Define the system dynamics as differential equations.

        Specifies the time derivatives of all state variables as functions of
        states, controls, time, and parameters.

        Args:
            dynamics_dict: Dictionary mapping each state variable to its time derivative.
                Keys can be state variables created with problem.state() or their underlying symbols.
                Values are symbolic expressions using states, controls, time, and parameters.

        Example:
            >>> x = problem.state("position")
            >>> v = problem.state("velocity")
            >>> u = problem.control("acceleration")
            >>> problem.dynamics({
            ...     x: v,           # dx/dt = v
            ...     v: u            # dv/dt = u
            ... })
        """
        converted_dynamics = {
            key: (val._symbolic_var if isinstance(val, StateVariableImpl) else val)
            for key, val in dynamics_dict.items()
        }
        variables_problem.set_dynamics(self._variable_state, converted_dynamics)

        # Log dynamics definition (INFO - user cares about major setup)
        logger.info("Dynamics defined for %d state variables", len(dynamics_dict))

    def add_integral(self, integrand_expr: ca.MX | float | int) -> ca.MX:
        """
        Add an integral expression to be computed during solution.

        Integrals are computed over the time interval and can be used in the
        objective function or constraints. Common uses include energy consumption,
        path length, or accumulated cost.

        Args:
            integrand_expr: Symbolic expression to integrate over time.
                Can depend on states, controls, time, and parameters.

        Returns:
            Symbolic variable representing the integral value

        Example:
            >>> u = problem.control("thrust")
            >>> energy = problem.add_integral(u**2)  # Energy consumption
            >>> problem.minimize(energy)             # Minimize energy
        """
        integral_var = variables_problem.add_integral(self._variable_state, integrand_expr)

        # Log integral addition (DEBUG)
        logger.debug("Integral added: total_integrals=%d", self._variable_state.num_integrals)

        return integral_var

    def minimize(self, objective_expr: ca.MX | float | int) -> None:
        """
        Define the objective function to minimize.

        Args:
            objective_expr: Symbolic expression to minimize.
                Can depend on initial/final states, initial/final time,
                integrals, and parameters. Cannot depend on path variables.

        Example:
            >>> t = problem.time()
            >>> problem.minimize(t.final)           # Minimum time
            >>>
            >>> energy = problem.add_integral(u**2)
            >>> problem.minimize(energy)            # Minimum energy
            >>>
            >>> x = problem.state("position")
            >>> problem.minimize(x.final**2)        # Minimize final position squared
        """
        variables_problem.set_objective(self._variable_state, objective_expr)

        # Log objective definition (INFO - user cares about major setup)
        logger.info("Objective function defined")

    def subject_to(self, constraint_expr: ca.MX | float | int) -> None:
        """
        Add a constraint to the problem.

        Constraints can be equality or inequality constraints that must be satisfied
        during optimization. They can depend on states, controls, time, integrals,
        and parameters.

        Args:
            constraint_expr: Symbolic constraint expression.
                Use ==, <=, >= operators to create constraint expressions.

        Example:
            >>> x = problem.state("position")
            >>> v = problem.state("velocity")
            >>> problem.subject_to(x + v <= 10.0)     # Path constraint
            >>> problem.subject_to(x.final == 5.0)    # Final condition
        """
        constraints_problem.add_constraint(self._constraint_state, constraint_expr)

        # Log constraint addition (DEBUG)
        logger.debug(
            "Constraint added: total_constraints=%d", len(self._constraint_state.constraints)
        )

    def set_mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """
        Configure the pseudospectral mesh for discretization.

        The mesh defines how the continuous optimal control problem is discretized
        for numerical solution. Higher polynomial degrees and more intervals
        generally provide higher accuracy but require more computation.

        Args:
            polynomial_degrees: List of polynomial degrees for each mesh interval.
                Each degree must be >= 1. Common values are 3-10.
            mesh_points: Mesh node locations in normalized domain [-1, 1].
                Must start at -1.0 and end at 1.0, with strictly increasing values.
                Length must be len(polynomial_degrees) + 1.

        Example:
            >>> # Single interval with degree 8
            >>> problem.set_mesh([8], [-1.0, 1.0])
            >>>
            >>> # Three intervals with different degrees
            >>> problem.set_mesh([5, 8, 5], [-1.0, -0.2, 0.3, 1.0])
            >>>
            >>> # Uniform spacing with numpy
            >>> import numpy as np
            >>> problem.set_mesh([6, 6, 6], np.linspace(-1, 1, 4))
        """

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
        """
        Set initial guess for the optimization variables.

        Providing a good initial guess can significantly improve solver performance
        and convergence. If no guess is provided, the solver will use default values.

        Args:
            states: List of state trajectory arrays, one per mesh interval.
                Each array shape: (num_states, num_collocation_nodes + 1)
            controls: List of control trajectory arrays, one per mesh interval.
                Each array shape: (num_controls, num_collocation_nodes)
            initial_time: Initial time guess (if time is free)
            terminal_time: Terminal time guess (if time is free)
            integrals: Integral values guess (scalar for single integral,
                array for multiple integrals)

        Note:
            Array shapes depend on the mesh configuration. Call get_initial_guess_requirements()
            to see the exact shapes needed.

        Example:
            >>> import numpy as np
            >>> # For single interval with 5 collocation nodes
            >>> state_guess = np.zeros((2, 6))    # 2 states, 6 nodes (5+1)
            >>> control_guess = np.ones((1, 5))   # 1 control, 5 nodes
            >>> problem.set_initial_guess([state_guess], [control_guess],
            ...                          initial_time=0.0, terminal_time=10.0)
        """

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
        """
        Check if the current initial guess can be validated.

        Returns:
            True if mesh is configured and initial guess can be validated
        """
        return initial_guess_problem.can_validate_initial_guess(
            self._mesh_state, self._variable_state
        )

    def get_initial_guess_requirements(self):
        """
        Get the required shapes and types for initial guess arrays.

        Returns:
            InitialGuessRequirements object describing the required array shapes
            and whether time/integral guesses are needed.

        Note:
            The mesh must be configured before calling this method to get
            specific shape requirements.

        Example:
            >>> problem.set_mesh([5, 8], [-1.0, 0.0, 1.0])
            >>> req = problem.get_initial_guess_requirements()
            >>> print(req)
            Initial Guess Requirements:
              States: 2 intervals
                Interval 0: array of shape (2, 6)
                Interval 1: array of shape (2, 9)
              Controls: 2 intervals
                Interval 0: array of shape (1, 5)
                Interval 1: array of shape (1, 8)
        """
        requirements = initial_guess_problem.get_initial_guess_requirements(
            self._mesh_state, self._variable_state
        )

        if not self._mesh_configured:
            print("Note: Mesh must be configured to get specific shape requirements")

        return requirements

    def validate_initial_guess(self) -> None:
        """
        Validate the current initial guess against the mesh configuration.

        Raises:
            ValueError: If initial guess is invalid or mesh is not configured

        Note:
            This is called automatically when solving, but can be called manually
            to check initial guess validity.
        """
        initial_guess_problem.validate_initial_guess(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    def get_solver_input_summary(self):
        """
        Get a summary of the current solver input configuration.

        Returns:
            SolverInputSummary object describing the current problem configuration
            including mesh, initial guess status, and variable dimensions.
        """
        return initial_guess_problem.get_solver_input_summary(
            self._initial_guess_container[0], self._mesh_state, self._variable_state
        )

    # ========================================================================
    # SOLVER INTERFACE METHODS - Internal
    # ========================================================================

    def get_dynamics_function(self):
        """Get dynamics function for internal solver interface."""
        return solver_interface.get_dynamics_function(self._variable_state)

    def get_objective_function(self):
        """Get objective function for internal solver interface."""
        return solver_interface.get_objective_function(self._variable_state)

    def get_integrand_function(self):
        """Get integrand function for internal solver interface."""
        return solver_interface.get_integrand_function(self._variable_state)

    def get_path_constraints_function(self):
        """Get path constraints function for internal solver interface."""
        return solver_interface.get_path_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )

    def get_event_constraints_function(self):
        """Get event constraints function for internal solver interface."""
        return solver_interface.get_event_constraints_function_for_problem(
            self._constraint_state, self._variable_state
        )
