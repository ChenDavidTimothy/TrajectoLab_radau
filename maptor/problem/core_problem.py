import logging
from collections.abc import Mapping, Sequence
from typing import Any

import casadi as ca

from maptor.exceptions import ConfigurationError

from ..input_validation import (
    _validate_complete_dynamics,
    _validate_constraint_input_format,
    _validate_string_not_empty,
)
from ..tl_types import FloatArray, NumericArrayLike, PhaseID
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .constraints_problem import (
    _get_cross_phase_event_constraints_function,
    _get_phase_path_constraints_function,
)
from .state import ConstraintInput, MultiPhaseVariableState
from .variables_problem import StateVariableImpl, TimeVariableImpl


logger = logging.getLogger(__name__)


def _validate_phase_exists(phases: dict[PhaseID, Any], phase_id: PhaseID) -> None:
    if phase_id not in phases:
        raise ValueError(f"Phase {phase_id} does not exist")


def _validate_constraint_inputs(name: str, boundary: ConstraintInput, context: str) -> None:
    _validate_string_not_empty(name, f"{context} name")
    _validate_constraint_input_format(boundary, f"{context} '{name}' boundary")


def _log_constraint_addition(
    constraint_count: int, phase_id: PhaseID, constraint_type: str
) -> None:
    logger.debug(
        "Added %d %s constraint(s) to phase %d", constraint_count, constraint_type, phase_id
    )


def _validate_constraint_expressions_not_empty(
    constraint_expressions: tuple, phase_id: PhaseID, constraint_type: str
) -> None:
    if not constraint_expressions:
        raise ValueError(
            f"Phase {phase_id} {constraint_type}_constraints() requires at least one constraint expression"
        )


def _process_symbolic_time_constraints(
    phase_def: Any, phase_id: PhaseID, cross_phase_constraints: list[ca.MX]
) -> None:
    if phase_def.t0_constraint.is_symbolic():
        if (
            phase_def.sym_time_initial is None
            or phase_def.t0_constraint.symbolic_expression is None
        ):
            raise ValueError(f"Phase {phase_id} has an undefined symbolic time initial expression.")
        constraint_expr = phase_def.sym_time_initial - phase_def.t0_constraint.symbolic_expression
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic time initial constraint for phase {phase_id}")

    if phase_def.tf_constraint.is_symbolic():
        if phase_def.sym_time_final is None or phase_def.tf_constraint.symbolic_expression is None:
            raise ValueError(f"Phase {phase_id} has an undefined symbolic time final expression.")
        constraint_expr = phase_def.sym_time_final - phase_def.tf_constraint.symbolic_expression
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic time final constraint for phase {phase_id}")


def _process_single_symbolic_boundary_constraint(
    var_name: str,
    constraint_type: str,
    symbolic_expr: ca.MX,
    phase_def: Any,
    phase_id: PhaseID,
    cross_phase_constraints: list[ca.MX],
) -> None:
    state_index = phase_def.state_name_to_index[var_name]

    if constraint_type == "initial":
        state_initial_sym = phase_def._get_ordered_state_initial_symbols()[state_index]
        constraint_expr = state_initial_sym - symbolic_expr
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic initial constraint for phase {phase_id} state '{var_name}'")

    elif constraint_type == "final":
        state_final_sym = phase_def._get_ordered_state_final_symbols()[state_index]
        constraint_expr = state_final_sym - symbolic_expr
        cross_phase_constraints.append(constraint_expr)
        logger.debug(f"Added automatic final constraint for phase {phase_id} state '{var_name}'")

    elif constraint_type == "boundary":
        state_initial_sym = phase_def._get_ordered_state_initial_symbols()[state_index]
        state_final_sym = phase_def._get_ordered_state_final_symbols()[state_index]

        initial_constraint = state_initial_sym - symbolic_expr
        final_constraint = state_final_sym - symbolic_expr

        cross_phase_constraints.extend([initial_constraint, final_constraint])
        logger.debug(
            f"Added automatic boundary constraints for phase {phase_id} state '{var_name}'"
        )


def _process_symbolic_state_constraints(
    phase_def: Any, phase_id: PhaseID, cross_phase_constraints: list[ca.MX]
) -> None:
    for var_name, constraint_type, symbolic_expr in phase_def.symbolic_boundary_constraints:
        _process_single_symbolic_boundary_constraint(
            var_name, constraint_type, symbolic_expr, phase_def, phase_id, cross_phase_constraints
        )


def _validate_phase_requirements(phases: dict[PhaseID, Any]) -> None:
    if not phases:
        raise ConfigurationError("Problem must have at least one phase defined")

    for phase_id, phase_def in phases.items():
        if not phase_def.dynamics_expressions:
            raise ConfigurationError(f"Phase {phase_id} must have dynamics defined")
        _validate_complete_dynamics(phase_def, phase_id)
        if not phase_def.mesh_configured:
            raise ConfigurationError(
                f"Phase {phase_id} must have mesh configured before validation. "
                f"Call phase.mesh(polynomial_degrees, mesh_points) first."
            )


def _process_symbolic_constraints_for_all_phases(
    phases: dict[PhaseID, Any], cross_phase_constraints: list[ca.MX]
) -> None:
    logger.debug("Processing symbolic boundary constraints for automatic cross-phase linking")

    for phase_id, phase_def in phases.items():
        _process_symbolic_time_constraints(phase_def, phase_id, cross_phase_constraints)
        _process_symbolic_state_constraints(phase_def, phase_id, cross_phase_constraints)


def _build_all_phase_functions(multiphase_state: MultiPhaseVariableState) -> None:
    static_parameter_symbols = multiphase_state.static_parameters.get_ordered_parameter_symbols()

    logger.debug("Building functions for all phases")
    for phase_id, phase_def in multiphase_state.phases.items():
        if phase_def._functions_built:
            continue

        logger.debug(f"Building functions for phase {phase_id}")

        phase_def._dynamics_function, phase_def._numerical_dynamics_function = (
            solver_interface._build_unified_phase_dynamics_functions(
                phase_def, static_parameter_symbols
            )
        )

        phase_def._integrand_function = solver_interface._build_phase_integrand_function(
            phase_def, static_parameter_symbols
        )

        phase_def._path_constraints_function = _get_phase_path_constraints_function(phase_def)

        phase_def._functions_built = True

    if not multiphase_state._functions_built:
        logger.debug("Building multiphase objective function")
        multiphase_state._objective_function = (
            solver_interface._build_multiphase_objective_function(multiphase_state)
        )

        multiphase_state._cross_phase_constraints_function = (
            _get_cross_phase_event_constraints_function(multiphase_state)
        )

        multiphase_state._functions_built = True


class Phase:
    """
    Single phase definition for multiphase optimal control problems.

    A Phase represents one segment of a multiphase trajectory with its own time
    domain, state variables, control inputs, dynamics, and constraints. Phases
    can be linked together through symbolic constraints to create complex
    multiphase missions.

    The Phase class provides a fluent interface for defining:

    - Time variables with boundary conditions
    - State variables with initial, final, and path constraints
    - Control variables with bounds
    - System dynamics as differential equations
    - Integral cost terms and constraints
    - Path constraints applied throughout the phase
    - Event constraints at phase boundaries
    - Mesh discretization for numerical solution

    Note:
        Phase objects are created through Problem.set_phase() and should not
        be instantiated directly.

    Examples:
        Basic single-phase problem setup:

        >>> problem = mtor.Problem("Rocket Ascent")
        >>> phase = problem.set_phase(1)
        >>>
        >>> # Define time and variables
        >>> t = phase.time(initial=0, final=10)
        >>> h = phase.state("altitude", initial=0, final=1000)
        >>> v = phase.state("velocity", initial=0)
        >>> T = phase.control("thrust", boundary=(0, 2000))
        >>>
        >>> # Set dynamics
        >>> phase.dynamics({h: v, v: T/1000 - 9.81})
        >>>
        >>> # Add constraints and mesh
        >>> phase.path_constraints(h >= 0, T <= 1500)
        >>> phase.mesh([5, 5], [-1, 0, 1])

        Multiphase trajectory with automatic linking:

        >>> # Phase 1: Boost
        >>> p1 = problem.set_phase(1)
        >>> t1 = p1.time(initial=0, final=120)
        >>> h1 = p1.state("altitude", initial=0)
        >>> v1 = p1.state("velocity", initial=0)
        >>> # ... dynamics and constraints
        >>>
        >>> # Phase 2: Coast (automatically linked)
        >>> p2 = problem.set_phase(2)
        >>> t2 = p2.time(initial=t1.final, final=300)  # Continuous time
        >>> h2 = p2.state("altitude", initial=h1.final)  # Continuous altitude
        >>> v2 = p2.state("velocity", initial=v1.final)  # Continuous velocity
        >>> # ... dynamics for coast phase
    """

    def __init__(self, problem: "Problem", phase_id: PhaseID) -> None:
        self.problem = problem
        self.phase_id = phase_id
        self._phase_def = self.problem._multiphase_state.set_phase(self.phase_id)

    def time(
        self, initial: ConstraintInput = 0.0, final: ConstraintInput = None
    ) -> TimeVariableImpl:
        """
        Define the time variable for this phase with boundary conditions.

        Creates the time coordinate for this phase, allowing specification of
        initial and final time constraints. Supports both fixed times and
        symbolic linking between phases for multiphase problems.

        Args:
            initial: Initial time constraint. Can be:

                - float: Fixed initial time (e.g., 0.0)
                - (lower, upper): Bounded initial time range
                - ca.MX: Symbolic expression for phase linking
                - None: Unconstrained initial time

            final: Final time constraint. Can be:

                - float: Fixed final time
                - (lower, upper): Bounded final time range
                - ca.MX: Symbolic expression for phase linking
                - None: Free final time (optimization variable)

        Returns:
            TimeVariableImpl: Time variable object with .initial and .final properties
            for use in expressions and constraints

        Examples:
            Fixed time phase (0 to 10 seconds):

            >>> t = phase.time(initial=0.0, final=10.0)

            Free final time (minimum time problem):

            >>> t = phase.time(initial=0.0)  # final=None means optimize final time
            >>> problem.minimize(t.final)   # Minimize final time

            Bounded final time:

            >>> t = phase.time(initial=0.0, final=(8.0, 12.0))  # 8 ≤ tf ≤ 12

            Multiphase with automatic time continuity:

            >>> # Phase 1
            >>> t1 = phase1.time(initial=0.0, final=100.0)
            >>>
            >>> # Phase 2 continues from Phase 1
            >>> t2 = phase2.time(initial=t1.final, final=200.0)

            Free initial and final times:

            >>> t = phase.time()  # Both times are optimization variables
        """
        return variables_problem.create_phase_time_variable(self._phase_def, initial, final)

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> StateVariableImpl:
        """
        Define a state variable for this phase with boundary and path constraints.

        Creates a state variable representing a component of the system state vector.
        State variables are governed by differential equations defined through the
        dynamics() method and can have constraints on their initial values, final
        values, and bounds throughout the trajectory.

        Args:
            name: Unique name for the state variable within this phase
            initial: Initial state constraint. Can be:

                - float: Fixed initial value (e.g., position=0.0)
                - (lower, upper): Bounded initial value range
                - ca.MX: Symbolic expression for multiphase continuity
                - None: Unconstrained initial state (optimization variable)

            final: Final state constraint. Can be:

                - float: Fixed final value (e.g., altitude=1000.0)
                - (lower, upper): Bounded final value range
                - ca.MX: Symbolic expression for multiphase continuity
                - None: Unconstrained final state

            boundary: Path constraint applied throughout the trajectory. Can be:

                - float: State must equal this value at all times
                - (lower, upper): State bounds (e.g., altitude >= 0)
                - None: No path bounds on this state

        Returns:
            StateVariableImpl: State variable object with .initial and .final
            properties for use in dynamics, constraints, and objective functions

        Examples:
            Position with fixed initial and final conditions:

            >>> pos = phase.state("position", initial=0.0, final=100.0)

            Velocity starting from rest with no final constraint:

            >>> vel = phase.state("velocity", initial=0.0)

            Altitude with physical bounds (must stay above ground):

            >>> alt = phase.state("altitude", initial=0, boundary=(0, None))

            Mass with bounds and consumption:

            >>> mass = phase.state("mass", initial=1000, boundary=(100, 1000))

            Multiphase state continuity:

            >>> # Phase 1
            >>> h1 = phase1.state("altitude", initial=0)
            >>> v1 = phase1.state("velocity", initial=0)
            >>>
            >>> # Phase 2 with continuous states
            >>> h2 = phase2.state("altitude", initial=h1.final)  # h continuous
            >>> v2 = phase2.state("velocity", initial=v1.final)  # v continuous

            Target final state with tolerance:

            >>> pos = phase.state("position", final=(99.0, 101.0))  # Target ≈ 100
        """
        return variables_problem._create_phase_state_variable(
            self._phase_def, name, initial, final, boundary
        )

    def control(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        """
        Define a control variable for this phase with bounds.

        Creates a control input variable representing actuator commands, forces,
        or other control inputs that can be varied to optimize the trajectory.
        Control variables appear in the dynamics equations and can be bounded
        to represent physical actuator limitations.

        Args:
            name: Unique name for the control variable within this phase
            boundary: Control bounds constraint. Can be:

                - float: Control must equal this constant value
                - (lower, upper): Control bounds (e.g., thrust limits)
                - (None, upper): Upper bound only
                - (lower, None): Lower bound only
                - None: Unconstrained control

        Returns:
            ca.MX: CasADi symbolic variable representing the control input
            for use in dynamics equations and cost functions

        Examples:
            Bounded thrust control:

            >>> thrust = phase.control("thrust", boundary=(0, 2000))  # 0 ≤ T ≤ 2000 N

            Steering angle with symmetric bounds:

            >>> steer = phase.control("steering", boundary=(-30, 30))  # ±30 degrees

            Throttle as fraction:

            >>> throttle = phase.control("throttle", boundary=(0, 1))  # 0 to 100%

            Unconstrained force:

            >>> force = phase.control("force")  # No bounds

            One-sided constraint:

            >>> power = phase.control("power", boundary=(0, None))  # Power ≥ 0

            Use in dynamics:

            >>> thrust = phase.control("thrust", boundary=(0, 1000))
            >>> mass = phase.state("mass", initial=1000)
            >>> acceleration = thrust / mass  # Control appears in dynamics
        """
        return variables_problem.create_phase_control_variable(self._phase_def, name, boundary)

    def dynamics(
        self,
        dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int | StateVariableImpl],
    ) -> None:
        """
        Define the differential equations governing this phase's dynamics.

        Specifies the system of ordinary differential equations (ODEs) that
        describe how the state variables evolve over time. The dynamics must
        be provided for all state variables in the phase.

        Args:
            dynamics_dict: Dictionary mapping state variables to their time
                derivatives. Keys are state variables (from state() method),
                values are expressions for dx/dt in terms of states, controls,
                and parameters.

        Examples:
            Simple point mass dynamics:

            >>> pos = phase.state("position", initial=0)
            >>> vel = phase.state("velocity", initial=0)
            >>> thrust = phase.control("thrust", boundary=(0, 100))
            >>>
            >>> phase.dynamics({
            ...     pos: vel,                    # dx/dt = v
            ...     vel: thrust - 0.1 * vel     # dv/dt = T - drag
            ... })

            Rocket dynamics with mass consumption:

            >>> h = phase.state("altitude", initial=0)
            >>> v = phase.state("velocity", initial=0)
            >>> m = phase.state("mass", initial=1000)
            >>> T = phase.control("thrust", boundary=(0, 2000))
            >>>
            >>> phase.dynamics({
            ...     h: v,                        # altitude rate
            ...     v: T/m - 9.81,              # acceleration
            ...     m: -T * 0.001               # mass consumption
            ... })

            Nonlinear satellite dynamics:

            >>> import casadi as ca
            >>> r = phase.state("radius", initial=7000e3)
            >>> theta = phase.state("angle", initial=0)
            >>> vr = phase.state("radial_velocity", initial=0)
            >>> vt = phase.state("tangential_velocity", initial=7500)
            >>> ur = phase.control("radial_thrust")
            >>> ut = phase.control("tangential_thrust")
            >>>
            >>> mu = 3.986e14  # Earth gravitational parameter
            >>> phase.dynamics({
            ...     r: vr,
            ...     theta: vt / r,
            ...     vr: vt**2/r - mu/r**2 + ur,
            ...     vt: -vr*vt/r + ut
            ... })

            Using parameters in dynamics:

            >>> mass_param = problem.parameter("vehicle_mass", boundary=(100, 1000))
            >>> drag_coeff = problem.parameter("drag_coefficient", boundary=(0.1, 0.5))
            >>>
            >>> phase.dynamics({
            ...     pos: vel,
            ...     vel: thrust/mass_param - drag_coeff * vel**2
            ... })
        """
        self._phase_def._functions_built = False
        variables_problem._set_phase_dynamics(self._phase_def, dynamics_dict)
        logger.info(
            "Dynamics defined for phase %d with %d state variables",
            self.phase_id,
            len(dynamics_dict),
        )

    def add_integral(self, integrand_expr: ca.MX | float | int) -> ca.MX:
        """
        Add an integral term to be computed over this phase.

        Creates an integral variable that accumulates the specified integrand
        expression over the phase duration. Useful for cost functions,
        constraint integrals, and quantities that accumulate over time.

        Args:
            integrand_expr: Expression to integrate over the phase. Can include
                states, controls, time, and parameters.

        Returns:
            ca.MX: Symbolic variable representing the integral value, which can
            be used in objective functions or constraints

        Examples:
            Quadratic cost integral:

            >>> pos = phase.state("position")
            >>> vel = phase.state("velocity")
            >>> thrust = phase.control("thrust")
            >>>
            >>> # Minimize energy and tracking error
            >>> cost = phase.add_integral(pos**2 + vel**2 + 0.1*thrust**2)
            >>> problem.minimize(cost)

            Fuel consumption:

            >>> thrust = phase.control("thrust", boundary=(0, 1000))
            >>> fuel_used = phase.add_integral(thrust * 0.001)  # kg/s consumption
            >>>
            >>> # Constraint on total fuel
            >>> phase.event_constraints(fuel_used <= 50)  # Max 50 kg fuel

            Distance traveled:

            >>> velocity = phase.state("velocity")
            >>> distance = phase.add_integral(velocity)  # ∫v dt = distance

            Heat load accumulation:

            >>> import casadi as ca
            >>> velocity = phase.state("velocity")
            >>> altitude = phase.state("altitude")
            >>> heat_rate = 0.001 * velocity**3 * ca.exp(-altitude/7000)
            >>> total_heat = phase.add_integral(heat_rate)
            >>>
            >>> # Heat constraint
            >>> phase.event_constraints(total_heat <= 1000)

            Multiple integrals:

            >>> # Different cost components
            >>> fuel_cost = phase.add_integral(thrust * fuel_price)
            >>> time_cost = phase.add_integral(1.0)  # Time integral
            >>> comfort_cost = phase.add_integral(acceleration**2)
            >>>
            >>> # Weighted objective
            >>> problem.minimize(fuel_cost + 10*time_cost + comfort_cost)
        """
        self._phase_def._functions_built = False
        return variables_problem._set_phase_integral(self._phase_def, integrand_expr)

    def path_constraints(self, *constraint_expressions: ca.MX | float | int) -> None:
        r"""
        Add path constraints enforced at every point along the trajectory.

        Path constraints are enforced at all collocation points throughout the
        phase, ensuring the specified conditions hold continuously along the
        trajectory. Use for bounds, inequality constraints, and safety limits.

        Args:
            \*constraint_expressions: Variable number of constraint expressions.
                Each expression should evaluate to zero for equality constraints
                or be written as inequality expressions (<=, >=, <, >, ==).

        Examples:
            State bounds and safety constraints:

            >>> altitude = phase.state("altitude")
            >>> velocity = phase.state("velocity")
            >>> thrust = phase.control("thrust")
            >>>
            >>> phase.path_constraints(
            ...     altitude >= 0,           # Stay above ground
            ...     velocity <= 200,         # Speed limit
            ...     thrust >= 0,             # Physical thrust limit
            ...     altitude <= 50000        # Maximum altitude
            ... )

            Nonlinear path constraints:

            >>> import casadi as ca
            >>> x = phase.state("x_position")
            >>> y = phase.state("y_position")
            >>>
            >>> # Avoid circular obstacle at (10, 10) with radius 5
            >>> obstacle_constraint = (x-10)**2 + (y-10)**2 >= 25
            >>> phase.path_constraints(obstacle_constraint)

            Dynamic pressure limits:

            >>> velocity = phase.state("velocity")
            >>> altitude = phase.state("altitude")
            >>>
            >>> # Atmospheric density model
            >>> rho = 1.225 * ca.exp(-altitude/8400)  # kg/m³
            >>> dynamic_pressure = 0.5 * rho * velocity**2
            >>>
            >>> phase.path_constraints(dynamic_pressure <= 50000)  # Pa limit

            Control rate limits:

            >>> # Note: Control rates need to be defined as additional states
            >>> thrust = phase.control("thrust")
            >>> thrust_rate = phase.state("thrust_rate")
            >>>
            >>> phase.dynamics({thrust_rate: 0})  # Defined elsewhere
            >>> phase.path_constraints(
            ...     thrust_rate >= -100,      # Thrust rate bounds
            ...     thrust_rate <= 100
            ... )

            Complex geometric constraints:

            >>> x = phase.state("x")
            >>> y = phase.state("y")
            >>> z = phase.state("z")
            >>>
            >>> # Stay within cylindrical flight corridor
            >>> lateral_distance = ca.sqrt(x**2 + y**2)
            >>> phase.path_constraints(
            ...     lateral_distance <= 1000,  # 1km radius
            ...     z >= 100,                  # Minimum altitude
            ...     z <= 10000                 # Maximum altitude
            ... )
        """
        _validate_constraint_expressions_not_empty(constraint_expressions, self.phase_id, "path")

        self._phase_def._functions_built = False
        for expr in constraint_expressions:
            constraints_problem._add_path_constraint(self._phase_def, expr)

        _log_constraint_addition(len(constraint_expressions), self.phase_id, "path")

    def event_constraints(self, *constraint_expressions: ca.MX | float | int) -> None:
        r"""
        Add event constraints enforced at phase boundaries or between phases.

        Event constraints are enforced at discrete points (phase start/end) rather
        than continuously along the trajectory. Use for boundary conditions,
        discontinuous jumps, and constraints linking multiple phases.

        Args:
            \*constraint_expressions: Variable number of constraint expressions
                involving boundary values, final states, or cross-phase continuity.

        Examples:
            Boundary condition constraints:

            >>> altitude = phase.state("altitude", initial=0)
            >>> velocity = phase.state("velocity", initial=0)
            >>>
            >>> # Target final conditions
            >>> phase.event_constraints(
            ...     altitude.final >= 1000,    # Reach minimum altitude
            ...     velocity.final <= 10       # Final velocity limit
            ... )

            Multiphase continuity (automatic alternative to symbolic):

            >>> # Manual continuity constraints (symbolic is preferred)
            >>> h1 = phase1.state("altitude")
            >>> v1 = phase1.state("velocity")
            >>> h2 = phase2.state("altitude")
            >>> v2 = phase2.state("velocity")
            >>>
            >>> phase2.event_constraints(
            ...     h2.initial == h1.final,    # Altitude continuity
            ...     v2.initial == v1.final     # Velocity continuity
            ... )

            Mission-specific boundary constraints:

            >>> # Orbit insertion requirements
            >>> radius = phase.state("radius")
            >>> speed = phase.state("speed")
            >>> flight_angle = phase.state("flight_path_angle")
            >>>
            >>> target_radius = 7000e3  # 7000 km
            >>> orbital_speed = ca.sqrt(3.986e14 / target_radius)
            >>>
            >>> phase.event_constraints(
            ...     radius.final >= target_radius,       # Reach orbit altitude
            ...     speed.final >= orbital_speed * 0.95,  # Near orbital velocity
            ...     ca.fabs(flight_angle.final) <= 0.1   # Nearly horizontal
            ... )

            Resource constraints:

            >>> fuel_used = phase.add_integral(thrust_magnitude * 0.001)
            >>> phase.event_constraints(fuel_used <= 100)  # Max 100kg fuel

            Equality constraints for exact targeting:

            >>> pos_x = phase.state("x_position")
            >>> pos_y = phase.state("y_position")
            >>>
            >>> # Exact final position
            >>> phase.event_constraints(
            ...     pos_x.final == 1000.0,     # Exactly at x=1000
            ...     pos_y.final == 500.0       # Exactly at y=500
            ... )
        """
        _validate_constraint_expressions_not_empty(constraint_expressions, self.phase_id, "event")

        for expr in constraint_expressions:
            constraints_problem._add_event_constraint(self.problem._multiphase_state, expr)

        _log_constraint_addition(len(constraint_expressions), self.phase_id, "event")

    def mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """
        Configure the pseudospectral mesh for numerical solution of this phase.

        Defines the mesh discretization used by the Radau pseudospectral method.
        The mesh consists of intervals with specified polynomial degrees and
        node distributions that determine solution accuracy and computational cost.

        Args:
            polynomial_degrees: List of polynomial degrees for each mesh interval.
                Higher degrees provide better accuracy but increase computational cost.
                Typical values: 3-8 for most problems, up to 15 for smooth solutions.
            mesh_points: Array of normalized mesh points in [-1, 1] defining
                interval boundaries. Must have length = len(polynomial_degrees) + 1.
                Points are automatically scaled to the actual phase time domain.

        Examples:
            Simple uniform mesh:

            >>> # 3 intervals, each with 4th-order polynomials
            >>> phase.mesh([4, 4, 4], [-1, -0.3, 0.3, 1])

            Non-uniform mesh for rapid initial transients:

            >>> # Fine mesh early, coarser later
            >>> phase.mesh([6, 4, 3], [-1, -0.8, 0, 1])

            High-accuracy smooth solution:

            >>> # High-order polynomials for very smooth dynamics
            >>> phase.mesh([10, 10], [-1, 0, 1])

            Adaptive starting point:

            >>> # Start with simple mesh for adaptive refinement
            >>> phase.mesh([3, 3, 3], [-1, -1/3, 1/3, 1])

            Complex trajectory with multiple regions:

            >>> # Different polynomial orders for different flight phases
            >>> phase.mesh(
            ...     [6, 4, 4, 6],           # Higher order for boost/entry
            ...     [-1, -0.5, 0, 0.5, 1]  # 4 intervals
            ... )

            Minimum mesh for testing:

            >>> # Simplest possible mesh
            >>> phase.mesh([3], [-1, 1])

        Note:
            - Total collocation points = sum(polynomial_degrees)
            - Computational cost scales roughly as O(N³) where N = total points
            - For adaptive solving, start with moderate polynomial degrees (3-6)
            - Non-uniform mesh points can focus resolution where needed
            - Mesh points are normalized; actual timing comes from time constraints
        """
        logger.info(
            "Setting mesh for phase %d: %d intervals", self.phase_id, len(polynomial_degrees)
        )
        mesh._configure_phase_mesh(self._phase_def, polynomial_degrees, mesh_points)


class Problem:
    """
    Multiphase optimal control problem definition and configuration interface.

    The Problem class is the main entry point for defining optimal control problems
    in MAPTOR. It supports both single-phase and multiphase trajectory optimization
    with automatic phase linking, static parameter optimization, and comprehensive
    constraint specification.

    Key capabilities:

    - **Multiphase trajectory definition** with automatic continuity
    - **Static parameter optimization** (design variables)
    - **Flexible objective functions** (minimize/maximize any expression)
    - **Initial guess specification** for improved convergence
    - **Solver configuration** and validation
    - **Cross-phase constraints** and event handling

    The Problem follows a builder pattern where you incrementally define phases,
    variables, dynamics, constraints, and objectives before solving.

    Examples:
        Single-phase minimum time problem:

        >>> import maptor as mtor
        >>>
        >>> problem = mtor.Problem("Minimum Time")
        >>> phase = problem.set_phase(1)
        >>>
        >>> # Variables and dynamics
        >>> t = phase.time(initial=0.0)
        >>> x = phase.state("position", initial=0, final=1)
        >>> v = phase.state("velocity", initial=0)
        >>> u = phase.control("force", boundary=(-1, 1))
        >>>
        >>> phase.dynamics({x: v, v: u})
        >>> problem.minimize(t.final)
        >>>
        >>> # Solve
        >>> phase.mesh([5, 5], [-1, 0, 1])
        >>> solution = mtor.solve_fixed_mesh(problem)

        Multiphase rocket trajectory:

        >>> problem = mtor.Problem("Rocket Launch")
        >>>
        >>> # Boost phase
        >>> boost = problem.set_phase(1)
        >>> t1 = boost.time(initial=0, final=120)
        >>> h1 = boost.state("altitude", initial=0)
        >>> v1 = boost.state("velocity", initial=0)
        >>> m1 = boost.state("mass", initial=1000)
        >>> T1 = boost.control("thrust", boundary=(0, 2000))
        >>>
        >>> boost.dynamics({
        ...     h1: v1,
        ...     v1: T1/m1 - 9.81,
        ...     m1: -T1 * 0.001
        ... })
        >>>
        >>> # Coast phase with automatic continuity
        >>> coast = problem.set_phase(2)
        >>> t2 = coast.time(initial=t1.final, final=300)
        >>> h2 = coast.state("altitude", initial=h1.final)
        >>> v2 = coast.state("velocity", initial=v1.final)
        >>> m2 = coast.state("mass", initial=m1.final)
        >>>
        >>> coast.dynamics({h2: v2, v2: -9.81, m2: 0})
        >>>
        >>> # Objective and solve
        >>> problem.minimize(-h2.final)  # Maximize final altitude
        >>> # ... mesh configuration and solve

        Problem with static parameters:

        >>> problem = mtor.Problem("Design Optimization")
        >>>
        >>> # Design parameters to optimize
        >>> engine_mass = problem.parameter("engine_mass", boundary=(50, 200))
        >>> thrust_level = problem.parameter("max_thrust", boundary=(1000, 5000))
        >>>
        >>> # Use parameters in dynamics
        >>> total_mass = vehicle_mass + engine_mass
        >>> phase.dynamics({v: thrust_level/total_mass - 9.81})
        >>>
        >>> # Multi-objective: maximize performance, minimize mass
        >>> performance = altitude.final
        >>> mass_penalty = engine_mass * 10
        >>> problem.minimize(mass_penalty - performance)
    """

    def __init__(self, name: str = "Multiphase Problem") -> None:
        """
        Initialize a new optimal control problem.

        Args:
            name: Descriptive name for the problem (used in logging and output)

        Examples:
            >>> problem = mtor.Problem("Spacecraft Trajectory")
            >>> problem = mtor.Problem("Robot Path Planning")
            >>> problem = mtor.Problem()  # Uses default name
        """
        _validate_string_not_empty(name, "Problem name")
        self.name = name
        logger.debug("Created multiphase problem: '%s'", name)

        self._multiphase_state = MultiPhaseVariableState()
        self._initial_guess_container = [None]
        self.solver_options: dict[str, Any] = {}

    def set_phase(self, phase_id: PhaseID) -> Phase:
        """
        Create and configure a new phase in the multiphase problem.

        Each phase represents a distinct segment of the trajectory with its own
        time domain, dynamics, and constraints. Phases can be linked through
        symbolic boundary constraints for trajectory continuity.

        Args:
            phase_id: Unique integer identifier for this phase. Phases are solved
                in order of their IDs, so use sequential numbering (1, 2, 3...).

        Returns:
            Phase: Phase object for defining variables, dynamics, and constraints

        Raises:
            ValueError: If phase_id already exists

        Examples:
            Single phase problem:

            >>> problem = mtor.Problem("Single Phase")
            >>> phase = problem.set_phase(1)
            >>> # Define phase variables, dynamics, constraints...

            Sequential multiphase problem:

            >>> problem = mtor.Problem("Three Phase Mission")
            >>>
            >>> # Launch phase
            >>> launch = problem.set_phase(1)
            >>> # ... configure launch phase
            >>>
            >>> # Coast phase
            >>> coast = problem.set_phase(2)
            >>> # ... configure coast phase
            >>>
            >>> # Landing phase
            >>> landing = problem.set_phase(3)
            >>> # ... configure landing phase

            Phase naming convention:

            >>> # Use descriptive variable names for clarity
            >>> ascent_phase = problem.set_phase(1)
            >>> orbit_phase = problem.set_phase(2)
            >>> descent_phase = problem.set_phase(3)
        """
        if phase_id in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} already exists")

        logger.debug("Adding phase %d to problem '%s'", phase_id, self.name)
        return Phase(self, phase_id)

    def parameter(self, name: str, boundary: ConstraintInput = None) -> ca.MX:
        """
        Define a static parameter for design optimization.

        Static parameters are optimization variables that remain constant throughout
        the entire mission but can be varied by the optimizer to improve performance.
        Useful for design variables, physical constants, or configuration parameters.

        Args:
            name: Unique name for the parameter
            boundary: Parameter bounds constraint. Can be:

                - float: Fixed parameter value
                - (lower, upper): Parameter bounds
                - (None, upper): Upper bound only
                - (lower, None): Lower bound only
                - None: Unconstrained parameter

        Returns:
            ca.MX: CasADi symbolic variable representing the parameter for use
            in dynamics, constraints, and objective functions across all phases

        Examples:
            Vehicle design parameters:

            >>> # Mass and thrust parameters to optimize
            >>> dry_mass = problem.parameter("dry_mass", boundary=(100, 500))
            >>> max_thrust = problem.parameter("max_thrust", boundary=(1000, 5000))
            >>>
            >>> # Use in dynamics across multiple phases
            >>> total_mass = dry_mass + fuel_mass
            >>> acceleration = thrust_control * max_thrust / total_mass

            Physical constants that might be uncertain:

            >>> # Atmospheric density variation
            >>> density_factor = problem.parameter("density_factor", boundary=(0.8, 1.2))
            >>> drag_coeff = problem.parameter("drag_coefficient", boundary=(0.1, 0.5))
            >>>
            >>> # Use in atmospheric flight dynamics
            >>> drag_force = 0.5 * density_factor * 1.225 * velocity**2 * drag_coeff

            Mission configuration parameters:

            >>> # Orbit insertion parameters
            >>> target_altitude = problem.parameter("target_altitude", boundary=(200e3, 800e3))
            >>> inclination = problem.parameter("inclination", boundary=(0, 90))
            >>>
            >>> # Use in final constraints
            >>> phase.event_constraints(
            ...     altitude.final >= target_altitude,
            ...     orbit_inclination.final == inclination
            ... )

            Economic optimization parameters:

            >>> # Cost factors for economic optimization
            >>> fuel_cost = problem.parameter("fuel_price", boundary=(1, 10))  # $/kg
            >>> time_cost = problem.parameter("time_value", boundary=(100, 1000))  # $/hour
            >>>
            >>> # Use in economic objective
            >>> fuel_expense = fuel_used * fuel_cost
            >>> time_expense = mission_time * time_cost / 3600
            >>> problem.minimize(fuel_expense + time_expense)
        """
        _validate_constraint_inputs(name, boundary, "Parameter")

        param_var = variables_problem._create_static_parameter(
            self._multiphase_state.static_parameters, name, boundary
        )
        logger.debug("Static parameter created: name='%s'", name)
        return param_var

    def minimize(self, objective_expr: ca.MX | float | int) -> None:
        """
        Set the objective function to minimize.

        Defines the scalar cost function that the optimizer will minimize. Can include
        final state values, integral terms, parameters, and complex expressions
        combining multiple phases and objectives.

        Args:
            objective_expr: Scalar expression to minimize. Can involve:

                - Final state values (e.g., time.final, altitude.final)
                - Integral terms from add_integral()
                - Static parameters
                - Mathematical combinations of the above

        Examples:
            Minimum time problems:

            >>> t = phase.time(initial=0)
            >>> problem.minimize(t.final)  # Minimize final time

            Energy minimization:

            >>> # Minimize control energy
            >>> energy = phase.add_integral(thrust**2)
            >>> problem.minimize(energy)

            Maximum final altitude:

            >>> altitude = phase.state("altitude")
            >>> problem.minimize(-altitude.final)  # Negative for maximization

            Multiphase mission optimization:

            >>> # Launch to orbit with minimum fuel
            >>> fuel_p1 = phase1.add_integral(thrust1 * 0.001)  # Phase 1 fuel
            >>> fuel_p2 = phase2.add_integral(thrust2 * 0.001)  # Phase 2 fuel
            >>> total_fuel = fuel_p1 + fuel_p2
            >>> problem.minimize(total_fuel)

            Multi-objective with weights:

            >>> # Balance fuel consumption and mission time
            >>> fuel_cost = phase.add_integral(thrust * fuel_price)
            >>> time_cost = mission_time.final * time_value
            >>> comfort_cost = phase.add_integral(acceleration**2)
            >>>
            >>> total_cost = fuel_cost + 0.1*time_cost + 0.01*comfort_cost
            >>> problem.minimize(total_cost)

            Design optimization:

            >>> # Minimize mass while meeting performance requirements
            >>> vehicle_mass = problem.parameter("mass", boundary=(100, 1000))
            >>> final_velocity = velocity.final
            >>>
            >>> # Penalize low performance and high mass
            >>> performance_penalty = ca.fmax(0, 1000 - final_velocity)**2
            >>> mass_penalty = vehicle_mass * 0.1
            >>> problem.minimize(performance_penalty + mass_penalty)

            Terminal constraint optimization:

            >>> # Soft constraints through penalty terms
            >>> target_altitude = 1000
            >>> altitude_error = (altitude.final - target_altitude)**2
            >>> velocity_error = velocity.final**2  # Zero final velocity
            >>>
            >>> problem.minimize(altitude_error + velocity_error)
        """
        self._multiphase_state._functions_built = False
        variables_problem._set_multiphase_objective(self._multiphase_state, objective_expr)
        logger.info("Multiphase objective function defined")

    def guess(
        self,
        phase_states: Mapping[PhaseID, Sequence[NumericArrayLike]] | None = None,
        phase_controls: Mapping[PhaseID, Sequence[NumericArrayLike]] | None = None,
        phase_initial_times: Mapping[PhaseID, float] | None = None,
        phase_terminal_times: Mapping[PhaseID, float] | None = None,
        phase_integrals: Mapping[PhaseID, float | NumericArrayLike] | None = None,
        static_parameters: FloatArray | None = None,
    ) -> None:
        """
        Provide initial guess for improved optimization convergence.

        Supplies the nonlinear programming (NLP) solver with starting values for
        optimization variables. Good initial guesses significantly improve convergence
        speed and success rate, especially for complex nonlinear problems.

        Args:
            phase_states: Dictionary mapping phase IDs to state trajectory guesses.
                Each phase entry is a list of state arrays (one per mesh interval).
                Each array has shape (num_states, num_collocation_points).

            phase_controls: Dictionary mapping phase IDs to control trajectory guesses.
                Each phase entry is a list of control arrays (one per mesh interval).
                Each array has shape (num_controls, num_mesh_points).

            phase_initial_times: Dictionary mapping phase IDs to initial time guesses.

            phase_terminal_times: Dictionary mapping phase IDs to final time guesses.

            phase_integrals: Dictionary mapping phase IDs to integral value guesses.

            static_parameters: Array of static parameter guesses.

        Examples:
            Simple linear interpolation guess:

            >>> import numpy as np
            >>>
            >>> # Configure mesh first
            >>> phase.mesh([5, 5], [-1, 0, 1])  # 2 intervals, 5 points each
            >>>
            >>> # Generate state guess for each interval
            >>> states_guess = []
            >>> controls_guess = []
            >>>
            >>> for N in [5, 5]:  # For each mesh interval
            ...     # Time points within interval
            ...     t = np.linspace(0, 1, N+1)
            ...
            ...     # Linear interpolation between boundary conditions
            ...     pos_vals = 0.0 + (100.0 - 0.0) * t  # 0 to 100
            ...     vel_vals = 0.0 + (10.0 - 0.0) * t   # 0 to 10
            ...     states_guess.append(np.array([pos_vals, vel_vals]))
            ...
            ...     # Constant control guess
            ...     thrust_vals = np.ones(N) * 5.0  # Constant thrust
            ...     controls_guess.append(np.array([thrust_vals]))
            >>>
            >>> problem.guess(
            ...     phase_states={1: states_guess},
            ...     phase_controls={1: controls_guess},
            ...     phase_terminal_times={1: 10.0}
            ... )

            Multiphase rocket trajectory guess:

            >>> # Phase 1: Boost phase guess
            >>> boost_states = []
            >>> boost_controls = []
            >>> for N in [6, 6]:  # Mesh intervals
            ...     t_norm = np.linspace(0, 1, N+1)
            ...     # Quadratic altitude profile
            ...     h_vals = 0.5 * 100 * t_norm**2  # Accelerating ascent
            ...     v_vals = 100 * t_norm           # Linear velocity increase
            ...     m_vals = 1000 - 50 * t_norm     # Mass consumption
            ...     boost_states.append(np.array([h_vals, v_vals, m_vals]))
            ...
            ...     # High initial thrust, tapering off
            ...     T_vals = 2000 * (1 - 0.5 * np.linspace(0, 1, N))
            ...     boost_controls.append(np.array([T_vals]))
            >>>
            >>> # Phase 2: Coast phase guess
            >>> coast_states = []
            >>> coast_controls = []
            >>> for N in [4, 4]:
            ...     t_norm = np.linspace(0, 1, N+1)
            ...     # Ballistic trajectory
            ...     h_start = 2500  # End of boost
            ...     v_start = 120   # End of boost velocity
            ...     h_vals = h_start + v_start*t_norm - 0.5*9.81*t_norm**2
            ...     v_vals = v_start - 9.81*t_norm
            ...     m_vals = np.ones(N+1) * 950  # Constant mass
            ...     coast_states.append(np.array([h_vals, v_vals, m_vals]))
            ...
            ...     # Zero thrust during coast
            ...     T_vals = np.zeros(N)
            ...     coast_controls.append(np.array([T_vals]))
            >>>
            >>> problem.guess(
            ...     phase_states={1: boost_states, 2: coast_states},
            ...     phase_controls={1: boost_controls, 2: coast_controls},
            ...     phase_initial_times={1: 0.0, 2: 120.0},
            ...     phase_terminal_times={1: 120.0, 2: 300.0},
            ...     phase_integrals={1: 50.0, 2: 0.0}  # Fuel consumption guess
            ... )

            Parameter optimization guess:

            >>> # Include static parameter guesses
            >>> problem.guess(
            ...     phase_states={1: states_guess},
            ...     phase_controls={1: controls_guess},
            ...     static_parameters=np.array([500.0, 1500.0])  # [mass, thrust]
            ... )

            Analytical solution as guess:

            >>> # Use known analytical solution for similar problem
            >>> def analytical_trajectory(t):
            ...     # Simplified analytical solution
            ...     pos = 0.5 * thrust_nominal * t**2 / mass_nominal
            ...     vel = thrust_nominal * t / mass_nominal
            ...     return pos, vel
            >>>
            >>> # Generate guess from analytical solution
            >>> # ... create state/control arrays from analytical_trajectory

        Note:
            - Array dimensions must match mesh configuration exactly
            - Poor initial guesses can cause convergence failure
            - Linear interpolation between boundary conditions often works well
            - Consider using solutions from similar problems as starting points
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

    @property
    def _phases(self) -> dict[PhaseID, Any]:
        return self._multiphase_state.phases

    @property
    def _static_parameters(self) -> Any:
        return self._multiphase_state.static_parameters

    @property
    def initial_guess(self):
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        self._initial_guess_container[0] = value

    def _get_phase_ids(self) -> list[PhaseID]:
        return self._multiphase_state._get_phase_ids()

    def _get_phase_variable_counts(self, phase_id: PhaseID) -> tuple[int, int]:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return self._multiphase_state.phases[phase_id].get_variable_counts()

    def _get_total_variable_counts(self) -> tuple[int, int, int]:
        return self._multiphase_state._get_total_variable_counts()

    def _get_phase_ordered_state_names(self, phase_id: PhaseID) -> list[str]:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return self._multiphase_state.phases[phase_id].state_names.copy()

    def _get_phase_ordered_control_names(self, phase_id: PhaseID) -> list[str]:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        return self._multiphase_state.phases[phase_id].control_names.copy()

    def _get_phase_dynamics_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        phase_def = self._multiphase_state.phases[phase_id]

        if phase_def._dynamics_function is None:
            raise ValueError(
                f"Phase {phase_id} dynamics function not built - call validate_multiphase_configuration() first"
            )

        return phase_def._dynamics_function

    def _get_phase_numerical_dynamics_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        phase_def = self._multiphase_state.phases[phase_id]

        if phase_def._numerical_dynamics_function is None:
            raise ValueError(
                f"Phase {phase_id} numerical dynamics function not built - call validate_multiphase_configuration() first"
            )

        return phase_def._numerical_dynamics_function

    def _get_objective_function(self) -> Any:
        if self._multiphase_state._objective_function is None:
            raise ValueError(
                "Multiphase objective function not built - call validate_multiphase_configuration() first"
            )

        return self._multiphase_state._objective_function

    def _get_phase_integrand_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        phase_def = self._multiphase_state.phases[phase_id]
        return phase_def._integrand_function

    def _get_phase_path_constraints_function(self, phase_id: PhaseID) -> Any:
        _validate_phase_exists(self._multiphase_state.phases, phase_id)
        phase_def = self._multiphase_state.phases[phase_id]
        return phase_def._path_constraints_function

    def _get_cross_phase_event_constraints_function(self) -> Any:
        return self._multiphase_state._cross_phase_constraints_function

    def validate_multiphase_configuration(self) -> None:
        _process_symbolic_constraints_for_all_phases(
            self._multiphase_state.phases, self._multiphase_state.cross_phase_constraints
        )

        _validate_phase_requirements(self._multiphase_state.phases)

        if self._multiphase_state.objective_expression is None:
            raise ValueError("Problem must have objective function defined")

        _build_all_phase_functions(self._multiphase_state)

        logger.debug(
            "Multiphase configuration validated: %d phases, %d cross-phase constraints",
            len(self._multiphase_state.phases),
            len(self._multiphase_state.cross_phase_constraints),
        )
