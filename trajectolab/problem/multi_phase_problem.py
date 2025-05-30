"""
Multi-phase optimal control problem definition.

This module provides the MultiPhaseProblem class for defining optimal control problems
with multiple phases, faithfully implementing the general multiple-phase optimal control
problem structure from CGPOPS Section 2.

The implementation supports:
- Multiple independent phases with different dynamics and constraints
- Inter-phase event constraints linking phases through endpoint vectors
- Global static parameters shared across all phases
- Global objective function over all phase endpoints
- Unified constraint API consistent with TrajectoLab patterns
"""

import logging
from collections.abc import Callable
from typing import Any

import casadi as ca

from ..exceptions import ConfigurationError
from ..input_validation import validate_variable_name
from ..tl_types import (
    Constraint,
    NumericArrayLike,
)
from .core_problem import Problem


# Library logger - no handler configuration
logger = logging.getLogger(__name__)


class MultiPhaseProblem:
    """
    Multi-phase optimal control problem definition.

    Implements the general multiple-phase optimal control problem from CGPOPS Section 2,
    where each phase p ∈ {1, ..., P} is defined on interval t ∈ [t₀^(p), t_f^(p)] with
    independent dynamics, constraints, and variables, linked through event constraints
    and a global objective function.

    The mathematical structure follows CGPOPS exactly:
    - Each phase has state y^(p)(t), control u^(p)(t), integrals q^(p)
    - Global static parameters s shared across all phases
    - Inter-phase event constraints: b_min ≤ b(E^(1), ..., E^(P), s) ≤ b_max
    - Global objective: J = φ(E^(1), ..., E^(P), s)

    Where E^(p) = [y^(p)(t₀^(p)), t₀^(p), y^(p)(t_f^(p)), t_f^(p), q^(p)] are endpoint vectors.

    Args:
        name: Descriptive name for the multi-phase problem

    Example:
        >>> import trajectolab as tl
        >>> import numpy as np
        >>>
        >>> # Create multi-phase problem
        >>> mp_problem = tl.MultiPhaseProblem("Spacecraft Mission")
        >>>
        >>> # Add phases
        >>> launch_phase = mp_problem.add_phase("Launch")
        >>> coast_phase = mp_problem.add_phase("Coast")
        >>> landing_phase = mp_problem.add_phase("Landing")
        >>>
        >>> # Define each phase independently
        >>> # Launch phase
        >>> t1 = launch_phase.time(initial=0.0)
        >>> x1 = launch_phase.state("position", initial=0.0)
        >>> u1 = launch_phase.control("thrust", boundary=(0.0, 1.0))
        >>> launch_phase.dynamics({x1: u1})
        >>> launch_phase.minimize(t1.final)
        >>>
        >>> # Coast phase
        >>> t2 = coast_phase.time()
        >>> x2 = coast_phase.state("position")
        >>> coast_phase.dynamics({x2: 0})  # Coast dynamics
        >>>
        >>> # Landing phase
        >>> t3 = landing_phase.time()
        >>> x3 = landing_phase.state("position", final=100.0)
        >>> u3 = landing_phase.control("thrust", boundary=(-1.0, 0.0))
        >>> landing_phase.dynamics({x3: u3})
        >>>
        >>> # Add global parameters
        >>> gravity = mp_problem.add_global_parameter("gravity", 9.81)
        >>> mass = mp_problem.add_global_parameter("vehicle_mass", 1000.0)
        >>>
        >>> # Link phases with event constraints
        >>> mp_problem.link_phases(x1.final == x2.initial)  # Position continuity
        >>> mp_problem.link_phases(t1.final == t2.initial)  # Time continuity
        >>> mp_problem.link_phases(x2.final == x3.initial)  # Position continuity
        >>> mp_problem.link_phases(t2.final == t3.initial)  # Time continuity
        >>>
        >>> # Set global objective (minimize total mission time)
        >>> mp_problem.set_global_objective(t1.final + (t2.final - t2.initial) + (t3.final - t3.initial))
        >>>
        >>> # Configure meshes for each phase
        >>> launch_phase.set_mesh([10], np.array([-1.0, 1.0]))
        >>> coast_phase.set_mesh([5], np.array([-1.0, 1.0]))
        >>> landing_phase.set_mesh([15], np.array([-1.0, 1.0]))
        >>>
        >>> # Solve multi-phase problem
        >>> solution = tl.solve_multi_phase_fixed_mesh(mp_problem)
    """

    def __init__(self, name: str = "Multi-Phase Problem") -> None:
        """Initialize multi-phase problem."""
        self.name = name

        # Log multi-phase problem creation (DEBUG - developer info)
        logger.debug("Created multi-phase problem: '%s'", name)

        # Phase management - faithful to CGPOPS structure
        self.phases: list[Problem] = []
        self._phase_names: list[str] = []
        self._phase_name_to_index: dict[str, int] = {}

        # Global static parameters s - shared across all phases (CGPOPS Equation 6)
        self.global_parameters: dict[str, float] = {}
        self._global_parameter_symbols: dict[str, ca.MX] = {}

        # Inter-phase event constraints - CGPOPS Equation (3)
        self.inter_phase_constraints: list[ca.MX] = []

        # Global objective function - CGPOPS Equation (1)
        self.global_objective_expression: ca.MX | None = None

        # Solver configuration
        self.solver_options: dict[str, Any] = {}

        # Multi-phase validation state
        self._phases_configured: bool = False
        self._global_objective_set: bool = False

    def add_phase(self, phase_name: str) -> Problem:
        """
        Add a new phase to the multi-phase problem.

        Creates and returns a new Problem instance representing phase p.
        Each phase is independent and can have different dynamics, constraints,
        and mesh configurations as per CGPOPS Section 2.

        Args:
            phase_name: Unique name for the phase (must be unique across all phases)

        Returns:
            Problem instance for defining the phase dynamics, constraints, and objective

        Raises:
            ConfigurationError: If phase name is not unique or invalid

        Example:
            >>> mp_problem = MultiPhaseProblem("Mission")
            >>> phase1 = mp_problem.add_phase("Ascent")
            >>> phase2 = mp_problem.add_phase("Orbit")
            >>> phase3 = mp_problem.add_phase("Descent")
        """
        # Centralized validation
        validate_variable_name(phase_name, "phase")

        # Check for duplicate phase names
        if phase_name in self._phase_name_to_index:
            raise ConfigurationError(
                f"Phase '{phase_name}' already exists. Phase names must be unique.",
                "Multi-phase problem configuration error",
            )

        # Create new phase as independent Problem instance
        try:
            phase = Problem(f"{self.name} - {phase_name}")
            phase_index = len(self.phases)

            # Register phase
            self.phases.append(phase)
            self._phase_names.append(phase_name)
            self._phase_name_to_index[phase_name] = phase_index

            # Log phase creation (INFO - user cares about major setup)
            logger.info(
                "Added phase '%s' (index %d) to multi-phase problem '%s'",
                phase_name,
                phase_index,
                self.name,
            )

            return phase

        except Exception as e:
            # Clean up on failure
            if phase_name in self._phase_name_to_index:
                del self._phase_name_to_index[phase_name]
            if len(self._phase_names) > 0 and self._phase_names[-1] == phase_name:
                self._phase_names.pop()
            if len(self.phases) > 0:
                self.phases.pop()

            raise ConfigurationError(
                f"Failed to create phase '{phase_name}': {e}", "Multi-phase problem setup error"
            ) from e

    def get_phase(self, phase_name: str) -> Problem:
        """
        Get phase by name with validation.

        Args:
            phase_name: Name of the phase to retrieve

        Returns:
            Problem instance for the specified phase

        Raises:
            ConfigurationError: If phase name does not exist
        """
        if phase_name not in self._phase_name_to_index:
            raise ConfigurationError(
                f"Phase '{phase_name}' not found. Available phases: {list(self._phase_name_to_index.keys())}",
                "Multi-phase problem phase access error",
            )

        phase_index = self._phase_name_to_index[phase_name]
        return self.phases[phase_index]

    def get_phase_by_index(self, phase_index: int) -> Problem:
        """
        Get phase by index with bounds checking.

        Args:
            phase_index: Zero-based index of the phase

        Returns:
            Problem instance for the specified phase

        Raises:
            ConfigurationError: If phase index is out of bounds
        """
        if not (0 <= phase_index < len(self.phases)):
            raise ConfigurationError(
                f"Phase index {phase_index} out of range [0, {len(self.phases)})",
                "Multi-phase problem phase access error",
            )

        return self.phases[phase_index]

    def add_global_parameter(self, name: str, value: float) -> ca.MX:
        """
        Add global static parameter shared across all phases.

        Global parameters s are shared across all phases and can be used in
        dynamics, constraints, and objective functions of any phase, as per
        CGPOPS Equation (6): s_min ≤ s ≤ s_max.

        Args:
            name: Parameter name (must be unique across all global parameters)
            value: Parameter value (numeric)

        Returns:
            CasADi symbolic variable for use in expressions across all phases

        Raises:
            ConfigurationError: If parameter name is not unique or value is invalid

        Example:
            >>> gravity = mp_problem.add_global_parameter("gravity", 9.81)
            >>> mass = mp_problem.add_global_parameter("vehicle_mass", 1000.0)
            >>> # Use in any phase dynamics
            >>> phase1.dynamics({x: u - gravity/mass})
        """
        # Centralized validation
        validate_variable_name(name, "global parameter")

        # Validate parameter value
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(
                f"Global parameter '{name}' value must be numeric, got {type(value)}: {value}",
                "Multi-phase problem parameter error",
            ) from e

        # Check for duplicate parameter names
        if name in self.global_parameters:
            raise ConfigurationError(
                f"Global parameter '{name}' already exists. Parameter names must be unique.",
                "Multi-phase problem configuration error",
            )

        # Create symbolic variable
        try:
            param_symbol = ca.MX.sym(name, 1)  # type: ignore[arg-type]

            # Store parameter
            self.global_parameters[name] = value
            self._global_parameter_symbols[name] = param_symbol

            # Log parameter creation (DEBUG - developer info)
            logger.debug("Added global parameter: name='%s', value=%s", name, value)

            return param_symbol

        except Exception as e:
            # Clean up on failure
            self.global_parameters.pop(name, None)
            self._global_parameter_symbols.pop(name, None)

            raise ConfigurationError(
                f"Failed to create global parameter '{name}': {e}",
                "Multi-phase problem parameter creation error",
            ) from e

    def link_phases(self, constraint_expr: ca.MX | float | int) -> None:
        """
        Add inter-phase event constraint linking phases.

        Inter-phase event constraints link information at the start and/or terminus
        of any phases, implementing CGPOPS Equation (3):
        b_min ≤ b(E^(1), ..., E^(P), s) ≤ b_max

        These constraints can relate:
        - State continuity between phases: x_final^(p) == x_initial^(p+1)
        - Time continuity: t_f^(p) == t_0^(p+1)
        - Jump conditions: x_initial^(p+1) == x_final^(p) + Δx
        - Complex multi-phase relationships involving integrals and parameters

        Args:
            constraint_expr: Symbolic constraint expression using phase endpoints.
                Can use phase.state("name").initial, phase.state("name").final,
                phase.time().initial, phase.time().final, integrals, and global parameters.

        Raises:
            ConfigurationError: If constraint expression is invalid

        Example:
            >>> # State continuity between phases
            >>> mp_problem.link_phases(phase1_x.final == phase2_x.initial)
            >>>
            >>> # Time continuity
            >>> mp_problem.link_phases(phase1_t.final == phase2_t.initial)
            >>>
            >>> # Jump condition with global parameter
            >>> delta_v = mp_problem.add_global_parameter("delta_v", 100.0)
            >>> mp_problem.link_phases(phase2_v.initial == phase1_v.final + delta_v)
        """
        # Convert to ca.MX if needed
        try:
            if isinstance(constraint_expr, ca.MX):
                constraint_mx = constraint_expr
            else:
                constraint_mx = ca.MX(constraint_expr)
        except Exception as e:
            raise ConfigurationError(
                f"Invalid inter-phase constraint expression: {constraint_expr}. "
                f"Expression must be symbolic using phase endpoints. Error: {e}",
                "Multi-phase problem constraint error",
            ) from e

        # Store inter-phase constraint
        self.inter_phase_constraints.append(constraint_mx)

        # Log constraint addition (DEBUG - developer info)
        logger.debug(
            "Added inter-phase constraint: total_constraints=%d", len(self.inter_phase_constraints)
        )

    def set_global_objective(self, objective_expr: ca.MX | float | int) -> None:
        """
        Set global objective function over all phase endpoints.

        Sets the global objective function that depends on endpoint vectors from
        all phases, implementing CGPOPS Equation (1) and (17):
        J = φ(E^(1), ..., E^(P), s)

        The objective can depend on:
        - Initial/final states from any phase: phase.state("name").initial/final
        - Initial/final times from any phase: phase.time().initial/final
        - Integrals from any phase: phase.add_integral(expression)
        - Global parameters: mp_problem.add_global_parameter("name", value)

        Args:
            objective_expr: Symbolic objective expression to minimize.
                Must use phase endpoint information (initial/final values).

        Raises:
            ConfigurationError: If objective expression is invalid

        Example:
            >>> # Minimize total mission time
            >>> total_time = (phase1_t.final - phase1_t.initial +
            ...               phase2_t.final - phase2_t.initial)
            >>> mp_problem.set_global_objective(total_time)
            >>>
            >>> # Minimize final position error with fuel penalty
            >>> fuel_penalty = phase1.add_integral(u1**2) + phase2.add_integral(u2**2)
            >>> position_error = (phase2_x.final - target)**2
            >>> mp_problem.set_global_objective(position_error + 0.1 * fuel_penalty)
        """
        # Convert to ca.MX with error handling
        try:
            if isinstance(objective_expr, ca.MX):
                objective_mx = objective_expr
            else:
                objective_mx = ca.MX(objective_expr)
        except Exception as e:
            if callable(objective_expr):
                raise ConfigurationError(
                    f"Global objective appears to be a function {objective_expr}. "
                    f"Did you forget to call it? Use expressions with phase endpoints.",
                    "Multi-phase problem objective error",
                ) from e
            else:
                raise ConfigurationError(
                    f"Cannot convert global objective expression of type {type(objective_expr)} to CasADi MX: {objective_expr}. "
                    f"Original error: {e}",
                    "Multi-phase problem objective error",
                ) from e

        # Store global objective
        self.global_objective_expression = objective_mx
        self._global_objective_set = True

        # Log objective definition (INFO - user cares about major setup)
        logger.info("Global objective function defined for multi-phase problem '%s'", self.name)

    def configure_all_phases(
        self, mesh_configurations: list[tuple[list[int], NumericArrayLike]]
    ) -> None:
        """
        Configure meshes for all phases simultaneously.

        Convenience method to configure meshes for all phases at once.
        Each phase can have different mesh configuration as appropriate
        for its dynamics and accuracy requirements.

        Args:
            mesh_configurations: List of (polynomial_degrees, mesh_points) for each phase.
                Length must match number of phases.

        Raises:
            ConfigurationError: If configuration count doesn't match phase count

        Example:
            >>> mp_problem.configure_all_phases([
            ...     ([10], [-1.0, 1.0]),           # Phase 0: single interval, degree 10
            ...     ([5, 8, 5], [-1.0, 0.0, 0.5, 1.0]),  # Phase 1: three intervals
            ...     ([15], [-1.0, 1.0]),           # Phase 2: single interval, degree 15
            ... ])
        """
        if len(mesh_configurations) != len(self.phases):
            raise ConfigurationError(
                f"Mesh configuration count ({len(mesh_configurations)}) "
                f"must match phase count ({len(self.phases)})",
                "Multi-phase problem mesh configuration error",
            )

        # Configure each phase
        for i, (polynomial_degrees, mesh_points) in enumerate(mesh_configurations):
            try:
                self.phases[i].set_mesh(polynomial_degrees, mesh_points)
                logger.debug("Configured mesh for phase %d (%s)", i, self._phase_names[i])
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to configure mesh for phase {i} ({self._phase_names[i]}): {e}",
                    "Multi-phase problem mesh configuration error",
                ) from e

        self._phases_configured = True
        logger.info("Configured meshes for all %d phases", len(self.phases))

    def validate_complete_structure(self) -> None:
        """
        Validate complete multi-phase problem structure.

        Performs comprehensive validation of the multi-phase problem including:
        - Phase consistency and completeness
        - Inter-phase constraint validity
        - Global objective function presence
        - Mesh configuration status
        - Global parameter consistency

        Raises:
            ConfigurationError: If multi-phase structure is invalid or incomplete
        """
        # Validate basic structure
        if not self.phases:
            raise ConfigurationError(
                "Multi-phase problem has no phases. Add phases using add_phase().",
                "Multi-phase problem validation error",
            )

        if len(self.phases) < 2:
            raise ConfigurationError(
                f"Multi-phase problem must have at least 2 phases, got {len(self.phases)}. "
                f"For single-phase problems, use Problem class directly.",
                "Multi-phase problem validation error",
            )

        # Validate global objective is set
        if self.global_objective_expression is None:
            raise ConfigurationError(
                "Multi-phase problem requires global objective function. "
                "Use set_global_objective() to define objective over phase endpoints.",
                "Multi-phase problem validation error",
            )

        # Validate each phase individually
        for i, phase in enumerate(self.phases):
            phase_name = self._phase_names[i]

            try:
                # Check phase has dynamics
                if not phase._dynamics_expressions:
                    raise ConfigurationError(
                        f"Phase '{phase_name}' has no dynamics defined. "
                        f"Use phase.dynamics() to define differential equations.",
                        "Multi-phase problem validation error",
                    )

                # Check phase mesh configuration
                if not phase._mesh_configured:
                    raise ConfigurationError(
                        f"Phase '{phase_name}' mesh not configured. "
                        f"Use phase.set_mesh() or configure_all_phases().",
                        "Multi-phase problem validation error",
                    )

                # Validate phase structure
                phase.validate_initial_guess()

            except ConfigurationError:
                # Re-raise configuration errors as-is
                raise
            except Exception as e:
                raise ConfigurationError(
                    f"Phase '{phase_name}' validation failed: {e}",
                    "Multi-phase problem validation error",
                ) from e

        # Validate inter-phase constraints if present
        if self.inter_phase_constraints:
            logger.debug("Validating %d inter-phase constraints", len(self.inter_phase_constraints))

            # Basic validation - detailed validation done at solve time
            for i, constraint in enumerate(self.inter_phase_constraints):
                if constraint is None:
                    raise ConfigurationError(
                        f"Inter-phase constraint {i} is None",
                        "Multi-phase problem validation error",
                    )

        # Log successful validation
        logger.info(
            "Multi-phase problem '%s' validation successful: %d phases, %d inter-phase constraints",
            self.name,
            len(self.phases),
            len(self.inter_phase_constraints),
        )

    def get_problem_summary(self) -> dict[str, Any]:
        """
        Get comprehensive summary of multi-phase problem structure.

        Returns:
            Dictionary containing detailed problem information for analysis and debugging
        """
        summary = {
            "name": self.name,
            "phase_count": len(self.phases),
            "phase_names": self._phase_names.copy(),
            "global_parameters": dict(self.global_parameters),
            "inter_phase_constraints_count": len(self.inter_phase_constraints),
            "global_objective_set": self._global_objective_set,
            "phases_configured": self._phases_configured,
            "phases": [],
        }

        # Add phase-specific information
        for i, phase in enumerate(self.phases):
            phase_name = self._phase_names[i]
            num_states, num_controls = phase.get_variable_counts()

            phase_info = {
                "index": i,
                "name": phase_name,
                "num_states": num_states,
                "num_controls": num_controls,
                "num_integrals": phase._num_integrals,
                "mesh_configured": phase._mesh_configured,
                "has_dynamics": bool(phase._dynamics_expressions),
                "has_objective": phase._objective_expression is not None,
                "collocation_points": phase.collocation_points_per_interval.copy()
                if phase._mesh_configured
                else None,
            }
            summary["phases"].append(phase_info)

        return summary

    # ========================================================================
    # PROTOCOL INTERFACE METHODS - Required by MultiPhaseProblemProtocol
    # ========================================================================

    def get_phase_count(self) -> int:
        """Get total number of phases P."""
        return len(self.phases)

    def get_phase_endpoint_vectors(self) -> list[ca.MX]:
        """
        Get symbolic endpoint vectors E^(p) for all phases.

        Implementation will be completed in solver interface layer.
        This method provides the protocol interface.
        """
        # This will be implemented in the solver interface layer
        # when converting the problem to solver-ready format
        endpoint_vectors = []
        for i, phase in enumerate(self.phases):
            # Placeholder - actual endpoint vector construction done in solver
            endpoint_placeholder = ca.MX.sym(f"E_{i}", 1)  # type: ignore[arg-type]
            endpoint_vectors.append(endpoint_placeholder)
        return endpoint_vectors

    def get_global_objective_function(self) -> Callable[[list, dict[str, float]], ca.MX]:
        """
        Get global objective function for solver.

        Implements CGPOPS Equation (17): J = φ(E^(1), ..., E^(P), s)
        where E^(p) are phase endpoint vectors and s are global static parameters.

        Returns:
            Function that evaluates global objective over phase endpoints
        """
        if self.global_objective_expression is None:
            raise ConfigurationError(
                "Global objective function not defined",
                "Multi-phase problem solver interface error",
            )

        def evaluate_global_objective(
            phase_endpoint_data: list[dict[str, Any]], global_params: dict[str, float]
        ) -> ca.MX:
            """
            Evaluate global objective function over phase endpoints.

            Args:
                phase_endpoint_data: List of phase endpoint data dictionaries
                global_params: Global parameter values

            Returns:
                Evaluated objective expression
            """
            try:
                # Create substitution map for phase endpoints and global parameters
                substitution_old = []
                substitution_new = []

                # Substitute phase endpoint information
                for phase_idx, endpoint_data in enumerate(phase_endpoint_data):
                    # Map phase variables to endpoint data
                    phase = self.phases[phase_idx]

                    # Get state symbols and map to initial/final values
                    state_symbols = phase.get_ordered_state_symbols()
                    state_initial_symbols = (
                        phase._variable_state.get_ordered_state_initial_symbols()
                    )
                    state_final_symbols = phase._variable_state.get_ordered_state_final_symbols()

                    # Map initial states Y_1^(p)
                    if (
                        "initial_state" in endpoint_data
                        and endpoint_data["initial_state"] is not None
                    ):
                        for i, sym in enumerate(state_initial_symbols):
                            substitution_old.append(sym)
                            if (
                                hasattr(endpoint_data["initial_state"], "shape")
                                and endpoint_data["initial_state"].shape
                            ):
                                substitution_new.append(
                                    endpoint_data["initial_state"][i]
                                    if i < endpoint_data["initial_state"].shape[0]
                                    else endpoint_data["initial_state"]
                                )
                            else:
                                substitution_new.append(endpoint_data["initial_state"])

                    # Map final states Y_{N^(p)+1}^(p)
                    if "final_state" in endpoint_data and endpoint_data["final_state"] is not None:
                        for i, sym in enumerate(state_final_symbols):
                            substitution_old.append(sym)
                            if (
                                hasattr(endpoint_data["final_state"], "shape")
                                and endpoint_data["final_state"].shape
                            ):
                                substitution_new.append(
                                    endpoint_data["final_state"][i]
                                    if i < endpoint_data["final_state"].shape[0]
                                    else endpoint_data["final_state"]
                                )
                            else:
                                substitution_new.append(endpoint_data["final_state"])

                    # Map times t_0^(p), t_f^(p)
                    if (
                        phase._variable_state.sym_time_initial is not None
                        and "initial_time" in endpoint_data
                    ):
                        substitution_old.append(phase._variable_state.sym_time_initial)
                        substitution_new.append(endpoint_data["initial_time"])

                    if (
                        phase._variable_state.sym_time_final is not None
                        and "terminal_time" in endpoint_data
                    ):
                        substitution_old.append(phase._variable_state.sym_time_final)
                        substitution_new.append(endpoint_data["terminal_time"])

                    # Map integrals Q^(p)
                    if "integrals" in endpoint_data and endpoint_data["integrals"] is not None:
                        integral_symbols = phase._variable_state.integral_symbols
                        for i, integral_sym in enumerate(integral_symbols):
                            substitution_old.append(integral_sym)
                            if (
                                hasattr(endpoint_data["integrals"], "shape")
                                and endpoint_data["integrals"].shape
                            ):
                                substitution_new.append(
                                    endpoint_data["integrals"][i]
                                    if i < endpoint_data["integrals"].shape[0]
                                    else endpoint_data["integrals"]
                                )
                            else:
                                substitution_new.append(endpoint_data["integrals"])

                # Substitute global parameters s
                for param_name, param_symbol in self._global_parameter_symbols.items():
                    if param_name in global_params:
                        substitution_old.append(param_symbol)
                        substitution_new.append(global_params[param_name])

                # Perform substitution
                if substitution_old and substitution_new:
                    objective_result = ca.substitute(
                        [self.global_objective_expression], substitution_old, substitution_new
                    )[0]
                else:
                    objective_result = self.global_objective_expression

                return objective_result

            except Exception as e:
                raise ConfigurationError(
                    f"Global objective function evaluation failed: {e}",
                    "Multi-phase problem objective evaluation error",
                ) from e

        return evaluate_global_objective

    def get_inter_phase_constraints_function(self) -> Callable[..., list[Constraint]] | None:
        """
        Get inter-phase event constraints function for solver.

        Implements CGPOPS Equation (20): b_min ≤ b(E^(1), ..., E^(P), s) ≤ b_max
        where constraints link phase endpoint vectors through event constraints.

        Returns:
            Function that evaluates inter-phase constraints, or None if no constraints
        """
        if not self.inter_phase_constraints:
            return None

        def evaluate_inter_phase_constraints(
            phase_endpoint_data: list[dict[str, Any]], global_params: dict[str, float]
        ) -> list[Constraint]:
            """
            Evaluate inter-phase event constraints.

            Args:
                phase_endpoint_data: List of phase endpoint data dictionaries
                global_params: Global parameter values

            Returns:
                List of evaluated constraint objects
            """
            try:
                constraints = []

                for constraint_expr in self.inter_phase_constraints:
                    # Create substitution map similar to objective function
                    substitution_old = []
                    substitution_new = []

                    # Substitute phase endpoint information
                    for phase_idx, endpoint_data in enumerate(phase_endpoint_data):
                        phase = self.phases[phase_idx]

                        # Get state symbols
                        state_symbols = phase.get_ordered_state_symbols()
                        state_initial_symbols = (
                            phase._variable_state.get_ordered_state_initial_symbols()
                        )
                        state_final_symbols = (
                            phase._variable_state.get_ordered_state_final_symbols()
                        )

                        # Map initial states
                        if (
                            "initial_state" in endpoint_data
                            and endpoint_data["initial_state"] is not None
                        ):
                            for i, sym in enumerate(state_initial_symbols):
                                substitution_old.append(sym)
                                if (
                                    hasattr(endpoint_data["initial_state"], "shape")
                                    and endpoint_data["initial_state"].shape
                                ):
                                    substitution_new.append(
                                        endpoint_data["initial_state"][i]
                                        if i < endpoint_data["initial_state"].shape[0]
                                        else endpoint_data["initial_state"]
                                    )
                                else:
                                    substitution_new.append(endpoint_data["initial_state"])

                        # Map final states
                        if (
                            "final_state" in endpoint_data
                            and endpoint_data["final_state"] is not None
                        ):
                            for i, sym in enumerate(state_final_symbols):
                                substitution_old.append(sym)
                                if (
                                    hasattr(endpoint_data["final_state"], "shape")
                                    and endpoint_data["final_state"].shape
                                ):
                                    substitution_new.append(
                                        endpoint_data["final_state"][i]
                                        if i < endpoint_data["final_state"].shape[0]
                                        else endpoint_data["final_state"]
                                    )
                                else:
                                    substitution_new.append(endpoint_data["final_state"])

                        # Map times
                        if (
                            phase._variable_state.sym_time_initial is not None
                            and "initial_time" in endpoint_data
                        ):
                            substitution_old.append(phase._variable_state.sym_time_initial)
                            substitution_new.append(endpoint_data["initial_time"])

                        if (
                            phase._variable_state.sym_time_final is not None
                            and "terminal_time" in endpoint_data
                        ):
                            substitution_old.append(phase._variable_state.sym_time_final)
                            substitution_new.append(endpoint_data["terminal_time"])

                        # Map integrals
                        if "integrals" in endpoint_data and endpoint_data["integrals"] is not None:
                            integral_symbols = phase._variable_state.integral_symbols
                            for i, integral_sym in enumerate(integral_symbols):
                                substitution_old.append(integral_sym)
                                if (
                                    hasattr(endpoint_data["integrals"], "shape")
                                    and endpoint_data["integrals"].shape
                                ):
                                    substitution_new.append(
                                        endpoint_data["integrals"][i]
                                        if i < endpoint_data["integrals"].shape[0]
                                        else endpoint_data["integrals"]
                                    )
                                else:
                                    substitution_new.append(endpoint_data["integrals"])

                    # Substitute global parameters
                    for param_name, param_symbol in self._global_parameter_symbols.items():
                        if param_name in global_params:
                            substitution_old.append(param_symbol)
                            substitution_new.append(global_params[param_name])

                    # Perform substitution and create constraint
                    if substitution_old and substitution_new:
                        constraint_result = ca.substitute(
                            [constraint_expr], substitution_old, substitution_new
                        )[0]
                    else:
                        constraint_result = constraint_expr

                    # Convert to unified Constraint object
                    from ..tl_types import Constraint

                    # Handle different constraint types (==, <=, >=)
                    if hasattr(constraint_result, "is_op"):
                        if constraint_result.is_op(getattr(ca, "OP_EQ", "eq")):
                            lhs = constraint_result.dep(0)
                            rhs = constraint_result.dep(1)
                            constraints.append(Constraint(val=lhs - rhs, equals=0.0))
                        elif constraint_result.is_op(getattr(ca, "OP_LE", "le")):
                            lhs = constraint_result.dep(0)
                            rhs = constraint_result.dep(1)
                            constraints.append(Constraint(val=lhs - rhs, max_val=0.0))
                        elif constraint_result.is_op(getattr(ca, "OP_GE", "ge")):
                            lhs = constraint_result.dep(0)
                            rhs = constraint_result.dep(1)
                            constraints.append(Constraint(val=lhs - rhs, min_val=0.0))
                        else:
                            # Default: treat as equality constraint
                            constraints.append(Constraint(val=constraint_result, equals=0.0))
                    else:
                        constraints.append(Constraint(val=constraint_result, equals=0.0))

                return constraints

            except Exception as e:
                raise ConfigurationError(
                    f"Inter-phase constraint evaluation failed: {e}",
                    "Multi-phase problem constraint evaluation error",
                ) from e

        return evaluate_inter_phase_constraints

    def validate_multi_phase_structure(self) -> None:
        """
        Validate multi-phase problem structure - protocol method.

        This replaces the NotImplementedError with actual validation.
        """
        # Delegate to main validation method
        self.validate_complete_structure()
