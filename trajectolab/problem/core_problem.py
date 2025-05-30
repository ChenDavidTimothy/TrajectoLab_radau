# trajectolab/problem/core_problem.py
"""
Core multiphase problem definition for optimal control problems.

This module provides the main Problem class that users interact with to define
multiphase optimal control problems using TrajectoLab's unified constraint API.
"""

import logging
from collections.abc import Sequence
from typing import Any

import casadi as ca

from ..tl_types import FloatArray, NumericArrayLike, PhaseID
from . import constraints_problem, initial_guess_problem, mesh, solver_interface, variables_problem
from .constraints_problem import (
    get_cross_phase_event_constraints_function,
    get_phase_path_constraints_function,
)
from .state import ConstraintInput, MultiPhaseVariableState, PhaseDefinition
from .variables_problem import StateVariableImpl, TimeVariableImpl


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


class PhaseContext:
    """Context manager for phase-specific variable definition."""

    def __init__(self, problem: "Problem", phase_id: PhaseID) -> None:
        self.problem = problem
        self.phase_id = phase_id
        self._phase_def: PhaseDefinition | None = None

    def __enter__(self) -> "PhaseContext":
        # Set current phase context
        self.problem._current_phase_id = self.phase_id
        if self.phase_id not in self.problem._multiphase_state.phases:
            self._phase_def = self.problem._multiphase_state.add_phase(self.phase_id)
        else:
            self._phase_def = self.problem._multiphase_state.phases[self.phase_id]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Clear current phase context
        self.problem._current_phase_id = None

    def time(
        self,
        initial: ConstraintInput = 0.0,
        final: ConstraintInput = None,
    ) -> TimeVariableImpl:
        """Define time variable for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")
        return variables_problem.create_phase_time_variable(self._phase_def, initial, final)

    def state(
        self,
        name: str,
        initial: ConstraintInput = None,
        final: ConstraintInput = None,
        boundary: ConstraintInput = None,
    ) -> StateVariableImpl:
        """Define state variable for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")
        return variables_problem.create_phase_state_variable(
            self._phase_def, name, initial, final, boundary
        )

    def control(
        self,
        name: str,
        boundary: ConstraintInput = None,
    ) -> ca.MX:
        """Define control variable for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")
        return variables_problem.create_phase_control_variable(self._phase_def, name, boundary)

    def dynamics(
        self,
        dynamics_dict: dict[ca.MX | StateVariableImpl, ca.MX | float | int | StateVariableImpl],
    ) -> None:
        """Define dynamics for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")
        variables_problem.set_phase_dynamics(self._phase_def, dynamics_dict)

        # Log dynamics definition
        logger.info(
            "Dynamics defined for phase %d with %d state variables",
            self.phase_id,
            len(dynamics_dict),
        )

    def add_integral(self, integrand_expr: ca.MX | float | int) -> ca.MX:
        """Add integral expression for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")
        return variables_problem.add_phase_integral(self._phase_def, integrand_expr)

    def subject_to(self, constraint_expr: ca.MX | float | int) -> None:
        """Add path constraint for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")
        constraints_problem.add_phase_path_constraint(self._phase_def, constraint_expr)

        logger.debug("Path constraint added to phase %d", self.phase_id)

    def set_mesh(self, polynomial_degrees: list[int], mesh_points: NumericArrayLike) -> None:
        """Configure mesh for this phase."""
        if self._phase_def is None:
            raise RuntimeError("Phase context not properly initialized")

        logger.info(
            "Setting mesh for phase %d: %d intervals", self.phase_id, len(polynomial_degrees)
        )
        mesh.configure_phase_mesh(self._phase_def, polynomial_degrees, mesh_points)
        logger.info(
            "Mesh configured for phase %d: %d intervals", self.phase_id, len(polynomial_degrees)
        )


class Problem:
    """
    Main class for defining multiphase optimal control problems.

    The Problem class provides a unified interface for defining multiphase optimal control
    problems using symbolic variables, constraints, and dynamics following the CGPOPS structure.

    Args:
        name: A descriptive name for the problem (used in logging and output)

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> # Create multiphase problem
        >>> problem = tl.Problem("Multiphase Mission")
        >>>
        >>> # Phase 1: Ascent
        >>> with problem.phase(1) as ascent:
        >>>     t1 = ascent.time(initial=0.0, final=(100, 200))
        >>>     h1 = ascent.state("altitude", initial=0.0)
        >>>     v1 = ascent.state("velocity", initial=0.0)
        >>>     u1 = ascent.control("thrust", boundary=(0, 1))
        >>>     ascent.dynamics({h1: v1, v1: u1})
        >>>     ascent.set_mesh([8, 8], [-1.0, 0.0, 1.0])
        >>>
        >>> # Phase 2: Coast
        >>> with problem.phase(2) as coast:
        >>>     t2 = coast.time(initial=t1.final)
        >>>     h2 = coast.state("altitude", initial=h1.final)
        >>>     v2 = coast.state("velocity", initial=v1.final)
        >>>     coast.dynamics({h2: v2, v2: 0})
        >>>     coast.set_mesh([5], [-1.0, 1.0])
        >>>
        >>> # Cross-phase constraints
        >>> problem.subject_to(h1.final == h2.initial)
        >>> problem.subject_to(v1.final == v2.initial)
        >>>
        >>> # Global objective
        >>> fuel = ascent.add_integral(u1**2)
        >>> problem.minimize(fuel + t2.final)
    """

    def __init__(self, name: str = "Multiphase Problem") -> None:
        """Initialize a new multiphase problem instance."""
        self.name = name

        # Log problem creation
        logger.debug("Created multiphase problem: '%s'", name)

        # Multiphase state management
        self._multiphase_state = MultiPhaseVariableState()
        self._current_phase_id: PhaseID | None = None
        self._initial_guess_container = [None]  # MultiPhaseInitialGuess container
        self.solver_options: dict[str, Any] = {}

    # ========================================================================
    # MULTIPHASE API METHODS - Public Interface
    # ========================================================================

    def phase(self, phase_id: PhaseID) -> PhaseContext:
        """
        Create a phase context for defining phase-specific variables and constraints.

        Args:
            phase_id: Unique identifier for the phase

        Returns:
            PhaseContext object for phase-specific operations

        Example:
            >>> with problem.phase(1) as ascent:
            >>>     x = ascent.state("position", initial=0.0)
            >>>     u = ascent.control("thrust", boundary=(0, 1))
            >>>     ascent.dynamics({x: u})
        """
        return PhaseContext(self, phase_id)

    def parameter(
        self,
        name: str,
        boundary: ConstraintInput = None,
    ) -> ca.MX:
        """
        Define a static parameter that spans across all phases.

        Args:
            name: Parameter name (must be unique)
            boundary: Constraint specification for the parameter

        Returns:
            CasADi symbolic variable for use across all phases

        Example:
            >>> mass = problem.parameter("mass", boundary=(100, 1000))
        """
        param_var = variables_problem.create_static_parameter(
            self._multiphase_state.static_parameters, name, boundary
        )

        logger.debug("Static parameter created: name='%s', boundary=%s", name, boundary)
        return param_var

    def minimize(self, objective_expr: ca.MX | float | int) -> None:
        """
        Define the multiphase objective function to minimize.

        Args:
            objective_expr: Symbolic expression to minimize.
                Can depend on initial/final states from any phase, times,
                integrals, and static parameters.

        Example:
            >>> # Single phase objective
            >>> problem.minimize(t1.final)
            >>>
            >>> # Multi-phase objective with integrals
            >>> fuel1 = phase1.add_integral(u1**2)
            >>> fuel2 = phase2.add_integral(u2**2)
            >>> problem.minimize(fuel1 + fuel2 + t2.final)
        """
        variables_problem.set_multiphase_objective(self._multiphase_state, objective_expr)

        logger.info("Multiphase objective function defined")

    def subject_to(self, constraint_expr: ca.MX | float | int) -> None:
        """
        Add a cross-phase constraint to the problem.

        Cross-phase constraints link variables from different phases and are applied
        as event constraints in the unified NLP formulation.

        Args:
            constraint_expr: Symbolic constraint expression.
                Can reference variables from multiple phases using ==, <=, >= operators.

        Example:
            >>> # State continuity between phases
            >>> problem.subject_to(x1.final == x2.initial)
            >>> problem.subject_to(v1.final == v2.initial)
            >>>
            >>> # Time continuity
            >>> problem.subject_to(t1.final == t2.initial)
            >>>
            >>> # Complex cross-phase constraints
            >>> problem.subject_to(x1.final + x2.final <= 100)
        """
        constraints_problem.add_cross_phase_constraint(self._multiphase_state, constraint_expr)

        logger.debug(
            "Cross-phase constraint added: total=%d",
            len(self._multiphase_state.cross_phase_constraints),
        )

    def set_initial_guess(
        self,
        phase_states: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_controls: dict[PhaseID, Sequence[FloatArray]] | None = None,
        phase_initial_times: dict[PhaseID, float] | None = None,
        phase_terminal_times: dict[PhaseID, float] | None = None,
        phase_integrals: dict[PhaseID, float | FloatArray] | None = None,
        static_parameters: FloatArray | None = None,
    ) -> None:
        """
        Set initial guess for the multiphase optimization variables.

        Args:
            phase_states: Dictionary mapping phase_id to list of state trajectory arrays
            phase_controls: Dictionary mapping phase_id to list of control trajectory arrays
            phase_initial_times: Dictionary mapping phase_id to initial time guess
            phase_terminal_times: Dictionary mapping phase_id to terminal time guess
            phase_integrals: Dictionary mapping phase_id to integral values guess
            static_parameters: Array of static parameter guesses

        Example:
            >>> # Set guess for multiple phases
            >>> problem.set_initial_guess(
            ...     phase_states={1: [state_guess_p1], 2: [state_guess_p2]},
            ...     phase_controls={1: [control_guess_p1], 2: [control_guess_p2]},
            ...     phase_terminal_times={1: 100.0, 2: 200.0},
            ...     static_parameters=np.array([500.0])  # mass parameter
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

        initial_guess_problem.set_multiphase_initial_guess(
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
    # PROTOCOL INTERFACE METHODS - Required by ProblemProtocol
    # ========================================================================

    @property
    def _phases(self) -> dict[PhaseID, Any]:
        """Internal phases access for protocol."""
        return self._multiphase_state.phases

    @property
    def _static_parameters(self) -> Any:
        """Internal static parameters access for protocol."""
        return self._multiphase_state.static_parameters

    @property
    def _cross_phase_constraints(self) -> list[ca.MX]:
        """Internal cross-phase constraints access for protocol."""
        return self._multiphase_state.cross_phase_constraints

    @property
    def _num_phases(self) -> int:
        """Internal number of phases for protocol."""
        return len(self._multiphase_state.phases)

    @property
    def initial_guess(self):
        """Current initial guess for the problem."""
        return self._initial_guess_container[0]

    @initial_guess.setter
    def initial_guess(self, value) -> None:
        """Set the initial guess for the problem."""
        self._initial_guess_container[0] = value

    def get_phase_ids(self) -> list[PhaseID]:
        """Return ordered list of phase IDs."""
        return self._multiphase_state.get_phase_ids()

    def get_phase_variable_counts(self, phase_id: PhaseID) -> tuple[int, int]:
        """Return (num_states, num_controls) for given phase."""
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].get_variable_counts()

    def get_total_variable_counts(self) -> tuple[int, int, int]:
        """Return (total_states, total_controls, num_static_params)."""
        return self._multiphase_state.get_total_variable_counts()

    def get_phase_ordered_state_names(self, phase_id: PhaseID) -> list[str]:
        """Get state names for given phase in order."""
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].state_names.copy()

    def get_phase_ordered_control_names(self, phase_id: PhaseID) -> list[str]:
        """Get control names for given phase in order."""
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return self._multiphase_state.phases[phase_id].control_names.copy()

    def get_phase_dynamics_function(self, phase_id: PhaseID) -> Any:
        """Get dynamics function for given phase."""
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return solver_interface.get_phase_dynamics_function(self._multiphase_state.phases[phase_id])

    def get_objective_function(self) -> Any:
        """Get multiphase objective function."""
        return solver_interface.get_multiphase_objective_function(self._multiphase_state)

    def get_phase_integrand_function(self, phase_id: PhaseID) -> Any:
        """Get integrand function for given phase."""
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return solver_interface.get_phase_integrand_function(
            self._multiphase_state.phases[phase_id]
        )

    def get_phase_path_constraints_function(self, phase_id: PhaseID) -> Any:
        """Get path constraints function for given phase."""
        if phase_id not in self._multiphase_state.phases:
            raise ValueError(f"Phase {phase_id} does not exist")
        return get_phase_path_constraints_function(self._multiphase_state.phases[phase_id])

    def get_cross_phase_event_constraints_function(self) -> Any:
        """Get cross-phase event constraints function."""
        return get_cross_phase_event_constraints_function(self._multiphase_state)

    def validate_multiphase_configuration(self) -> None:
        """Validate the multiphase problem configuration with automatic symbolic processing."""
        print("DEBUG: validate_multiphase_configuration() called")

        # First, process symbolic boundary constraints
        print("DEBUG: About to call _process_symbolic_boundary_constraints()")
        self._process_symbolic_boundary_constraints()
        print("DEBUG: Finished _process_symbolic_boundary_constraints()")

        # Then do existing validation
        # Validate that we have at least one phase
        if not self._multiphase_state.phases:
            raise ValueError("Problem must have at least one phase defined")

        # Validate each phase
        for phase_id, phase_def in self._multiphase_state.phases.items():
            if not phase_def.dynamics_expressions:
                raise ValueError(f"Phase {phase_id} must have dynamics defined")

            if not phase_def.mesh_configured:
                raise ValueError(f"Phase {phase_id} must have mesh configured")

        # Validate objective
        if self._multiphase_state.objective_expression is None:
            raise ValueError("Problem must have objective function defined")

        logger.debug(
            "Multiphase configuration validated: %d phases, %d cross-phase constraints",
            len(self._multiphase_state.phases),
            len(self._multiphase_state.cross_phase_constraints),
        )

    def _process_symbolic_boundary_constraints(self) -> None:
        """
        Process symbolic boundary constraints and convert them to cross-phase event constraints.
        """
        logger.debug("Processing symbolic boundary constraints for automatic cross-phase linking")

        for phase_id, phase_def in self._multiphase_state.phases.items():
            # ADD THIS DEBUG BLOCK
            print(f"\n=== DEBUG: Processing Phase {phase_id} ===")
            print(f"t0_constraint.is_symbolic(): {phase_def.t0_constraint.is_symbolic()}")
            print(f"tf_constraint.is_symbolic(): {phase_def.tf_constraint.is_symbolic()}")
            if phase_def.t0_constraint.is_symbolic():
                print(f"t0 symbolic expression: {phase_def.t0_constraint.symbolic_expression}")
            if phase_def.tf_constraint.is_symbolic():
                print(f"tf symbolic expression: {phase_def.tf_constraint.symbolic_expression}")

            # Process time constraints
            if phase_def.t0_constraint.is_symbolic():
                constraint_expr = (
                    phase_def.sym_time_initial - phase_def.t0_constraint.symbolic_expression
                )
                self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                print(f"Added t0 cross-phase constraint: {constraint_expr}")
                logger.debug(f"Added automatic time initial constraint for phase {phase_id}")

            if phase_def.tf_constraint.is_symbolic():
                constraint_expr = (
                    phase_def.sym_time_final - phase_def.tf_constraint.symbolic_expression
                )
                self._multiphase_state.cross_phase_constraints.append(constraint_expr)
                print(f"Added tf cross-phase constraint: {constraint_expr}")
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
                    # For boundary constraints, apply to both initial and final
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
