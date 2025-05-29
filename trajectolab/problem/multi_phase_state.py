"""
Multi-phase state management for variables, constraints, and mesh configuration.

This module extends TrajectoLab's unified state management system to handle multi-phase
optimal control problems, faithfully implementing the CGPOPS mathematical structure
while maintaining thread-safety and validation patterns.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import casadi as ca

from ..exceptions import ConfigurationError, DataIntegrityError
from ..input_validation import validate_variable_name
from ..tl_types import ProblemProtocol
from .state import ConstraintState, MeshState, VariableState


@dataclass
class GlobalParameterInfo:
    """Information for global static parameter shared across all phases."""

    name: str
    symbol: ca.MX
    value: float

    def __post_init__(self) -> None:
        """Validate global parameter info after initialization."""
        if self.symbol is None:
            raise DataIntegrityError(
                f"Global parameter symbol for '{self.name}' cannot be None",
                "TrajectoLab global parameter definition error",
            )

        if not isinstance(self.value, (int, float)):
            raise DataIntegrityError(
                f"Global parameter '{self.name}' value must be numeric, got {type(self.value)}",
                "TrajectoLab global parameter value error",
            )


@dataclass
class InterPhaseConstraintInfo:
    """Information for inter-phase event constraint linking phases."""

    constraint_index: int
    expression: ca.MX
    description: str = ""

    def __post_init__(self) -> None:
        """Validate inter-phase constraint info after initialization."""
        if self.expression is None:
            raise DataIntegrityError(
                f"Inter-phase constraint {self.constraint_index} expression cannot be None",
                "TrajectoLab inter-phase constraint definition error",
            )


@dataclass
class PhaseStateBundle:
    """
    Bundle of state containers for a single phase within multi-phase context.

    Maintains the three-component state management pattern from TrajectoLab
    while adding phase-specific identification and metadata.
    """

    phase_index: int
    phase_name: str
    variable_state: VariableState
    constraint_state: ConstraintState
    mesh_state: MeshState

    # Phase-specific metadata
    configured: bool = False
    has_dynamics: bool = False
    has_objective: bool = False

    def __post_init__(self) -> None:
        """Validate phase state bundle after initialization."""
        if self.phase_index < 0:
            raise DataIntegrityError(
                f"Phase index must be non-negative, got {self.phase_index}",
                "TrajectoLab phase state bundle error",
            )

        if not self.phase_name:
            raise DataIntegrityError(
                "Phase name cannot be empty", "TrajectoLab phase state bundle error"
            )

        # Validate component states are properly initialized
        if self.variable_state is None:
            raise DataIntegrityError(
                f"Phase {self.phase_index} variable state cannot be None",
                "TrajectoLab phase state bundle error",
            )

        if self.constraint_state is None:
            raise DataIntegrityError(
                f"Phase {self.phase_index} constraint state cannot be None",
                "TrajectoLab phase state bundle error",
            )

        if self.mesh_state is None:
            raise DataIntegrityError(
                f"Phase {self.phase_index} mesh state cannot be None",
                "TrajectoLab phase state bundle error",
            )

    def get_variable_counts(self) -> tuple[int, int]:
        """Get (num_states, num_controls) for this phase."""
        return self.variable_state.get_variable_counts()

    def is_fully_configured(self) -> bool:
        """Check if phase is fully configured for solving."""
        return (
            self.configured
            and self.has_dynamics
            and self.mesh_state.configured
            and bool(self.variable_state._dynamics_expressions)
        )

    def update_configuration_status(self) -> None:
        """Update configuration status based on current state."""
        self.has_dynamics = bool(self.variable_state._dynamics_expressions)
        self.has_objective = self.variable_state.objective_expression is not None
        self.configured = self.has_dynamics and self.mesh_state.configured


@dataclass
class MultiPhaseState:
    """
    Unified state management for multi-phase optimal control problems.

    Coordinates multiple phase states plus global multi-phase information,
    implementing the CGPOPS mathematical structure while maintaining
    TrajectoLab's thread-safe state management patterns.
    """

    # Phase state management - faithful to CGPOPS structure
    phase_states: list[PhaseStateBundle] = field(default_factory=list)
    phase_count: int = 0

    # Global static parameters s - shared across all phases (CGPOPS Equation 6)
    global_parameters: dict[str, GlobalParameterInfo] = field(default_factory=dict)
    global_parameter_lock: threading.Lock = field(default_factory=threading.Lock)

    # Inter-phase event constraints - CGPOPS Equation (3)
    inter_phase_constraints: list[InterPhaseConstraintInfo] = field(default_factory=list)
    inter_phase_constraint_lock: threading.Lock = field(default_factory=threading.Lock)

    # Global objective function - CGPOPS Equation (1)
    global_objective_expression: ca.MX | None = None
    global_objective_lock: threading.Lock = field(default_factory=threading.Lock)

    # Multi-phase validation state
    structure_validated: bool = False
    all_phases_configured: bool = False

    def __post_init__(self) -> None:
        """Validate multi-phase state after initialization."""
        if self.phase_count != len(self.phase_states):
            raise DataIntegrityError(
                f"Phase count mismatch: expected {self.phase_count}, "
                f"got {len(self.phase_states)} phase states",
                "TrajectoLab multi-phase state initialization error",
            )

    # ========================================================================
    # PHASE STATE MANAGEMENT - Thread-safe with validation
    # ========================================================================

    def add_phase_state(
        self,
        phase_name: str,
        variable_state: VariableState,
        constraint_state: ConstraintState,
        mesh_state: MeshState,
    ) -> int:
        """
        Add state bundle for a new phase.

        Args:
            phase_name: Unique name for the phase
            variable_state: Variable state for the phase
            constraint_state: Constraint state for the phase
            mesh_state: Mesh state for the phase

        Returns:
            Phase index for the newly added phase

        Raises:
            ConfigurationError: If phase name is not unique or states are invalid
        """
        # Centralized validation
        validate_variable_name(phase_name, "phase")

        # Check for duplicate phase names
        if any(bundle.phase_name == phase_name for bundle in self.phase_states):
            raise ConfigurationError(
                f"Phase '{phase_name}' already exists. Phase names must be unique.",
                "Multi-phase state configuration error",
            )

        # Validate state components
        if variable_state is None:
            raise ConfigurationError(
                f"Variable state for phase '{phase_name}' cannot be None",
                "Multi-phase state configuration error",
            )

        if constraint_state is None:
            raise ConfigurationError(
                f"Constraint state for phase '{phase_name}' cannot be None",
                "Multi-phase state configuration error",
            )

        if mesh_state is None:
            raise ConfigurationError(
                f"Mesh state for phase '{phase_name}' cannot be None",
                "Multi-phase state configuration error",
            )

        # Create phase state bundle
        phase_index = len(self.phase_states)

        try:
            phase_bundle = PhaseStateBundle(
                phase_index=phase_index,
                phase_name=phase_name,
                variable_state=variable_state,
                constraint_state=constraint_state,
                mesh_state=mesh_state,
            )

            # Add to phase states
            self.phase_states.append(phase_bundle)
            self.phase_count = len(self.phase_states)

            # Update configuration status
            self._update_all_phases_configured_status()

            return phase_index

        except Exception as e:
            # Clean up on failure
            if len(self.phase_states) > phase_index:
                self.phase_states.pop()
                self.phase_count = len(self.phase_states)

            raise ConfigurationError(
                f"Failed to add phase state for '{phase_name}': {e}",
                "Multi-phase state setup error",
            ) from e

    def get_phase_state(self, phase_index: int) -> PhaseStateBundle:
        """
        Get phase state bundle by index with bounds checking.

        Args:
            phase_index: Zero-based phase index

        Returns:
            Phase state bundle for the specified phase

        Raises:
            ConfigurationError: If phase index is out of bounds
        """
        if not (0 <= phase_index < len(self.phase_states)):
            raise ConfigurationError(
                f"Phase index {phase_index} out of range [0, {len(self.phase_states)})",
                "Multi-phase state phase access error",
            )

        return self.phase_states[phase_index]

    def get_phase_state_by_name(self, phase_name: str) -> PhaseStateBundle:
        """
        Get phase state bundle by name.

        Args:
            phase_name: Name of the phase

        Returns:
            Phase state bundle for the specified phase

        Raises:
            ConfigurationError: If phase name is not found
        """
        for bundle in self.phase_states:
            if bundle.phase_name == phase_name:
                return bundle

        available_names = [bundle.phase_name for bundle in self.phase_states]
        raise ConfigurationError(
            f"Phase '{phase_name}' not found. Available phases: {available_names}",
            "Multi-phase state phase access error",
        )

    def update_phase_configuration_status(self, phase_index: int) -> None:
        """Update configuration status for specific phase."""
        phase_bundle = self.get_phase_state(phase_index)
        phase_bundle.update_configuration_status()
        self._update_all_phases_configured_status()

    def _update_all_phases_configured_status(self) -> None:
        """Update overall configuration status based on all phases."""
        self.all_phases_configured = (
            len(self.phase_states) >= 2  # Multi-phase requires at least 2 phases
            and all(bundle.is_fully_configured() for bundle in self.phase_states)
            and self.global_objective_expression is not None
        )

    # ========================================================================
    # GLOBAL PARAMETER MANAGEMENT - Thread-safe with validation
    # ========================================================================

    def add_global_parameter(self, name: str, symbol: ca.MX, value: float) -> None:
        """
        Add global static parameter shared across all phases.

        Args:
            name: Parameter name (must be unique)
            symbol: CasADi symbolic variable
            value: Parameter value

        Raises:
            ConfigurationError: If parameter is invalid or name not unique
        """
        # Centralized validation
        validate_variable_name(name, "global parameter")

        # Validate parameter value
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(
                f"Global parameter '{name}' value must be numeric, got {type(value)}: {value}",
                "Multi-phase state parameter error",
            ) from e

        with self.global_parameter_lock:
            # Check for duplicate parameter names
            if name in self.global_parameters:
                raise ConfigurationError(
                    f"Global parameter '{name}' already exists. Parameter names must be unique.",
                    "Multi-phase state configuration error",
                )

            try:
                # Create parameter info
                param_info = GlobalParameterInfo(
                    name=name,
                    symbol=symbol,
                    value=value,
                )

                # Store parameter
                self.global_parameters[name] = param_info

            except Exception as e:
                # Clean up on failure
                self.global_parameters.pop(name, None)

                raise ConfigurationError(
                    f"Failed to add global parameter '{name}': {e}",
                    "Multi-phase state parameter creation error",
                ) from e

    def get_global_parameter(self, name: str) -> GlobalParameterInfo:
        """
        Get global parameter by name.

        Args:
            name: Parameter name

        Returns:
            Global parameter information

        Raises:
            ConfigurationError: If parameter name is not found
        """
        with self.global_parameter_lock:
            if name not in self.global_parameters:
                available_names = list(self.global_parameters.keys())
                raise ConfigurationError(
                    f"Global parameter '{name}' not found. Available parameters: {available_names}",
                    "Multi-phase state parameter access error",
                )

            return self.global_parameters[name]

    def get_all_global_parameters(self) -> dict[str, GlobalParameterInfo]:
        """Get all global parameters with thread-safe access."""
        with self.global_parameter_lock:
            return dict(self.global_parameters)

    def get_global_parameter_symbols(self) -> dict[str, ca.MX]:
        """Get global parameter symbols for solver interface."""
        with self.global_parameter_lock:
            return {name: info.symbol for name, info in self.global_parameters.items()}

    def get_global_parameter_values(self) -> dict[str, float]:
        """Get global parameter values for solver interface."""
        with self.global_parameter_lock:
            return {name: info.value for name, info in self.global_parameters.items()}

    # ========================================================================
    # INTER-PHASE CONSTRAINT MANAGEMENT - Thread-safe with validation
    # ========================================================================

    def add_inter_phase_constraint(self, expression: ca.MX, description: str = "") -> int:
        """
        Add inter-phase event constraint.

        Args:
            expression: CasADi constraint expression
            description: Optional description for debugging

        Returns:
            Constraint index for the newly added constraint

        Raises:
            ConfigurationError: If constraint expression is invalid
        """
        if expression is None:
            raise ConfigurationError(
                "Inter-phase constraint expression cannot be None",
                "Multi-phase state constraint error",
            )

        with self.inter_phase_constraint_lock:
            constraint_index = len(self.inter_phase_constraints)

            try:
                # Create constraint info
                constraint_info = InterPhaseConstraintInfo(
                    constraint_index=constraint_index,
                    expression=expression,
                    description=description,
                )

                # Store constraint
                self.inter_phase_constraints.append(constraint_info)

                return constraint_index

            except Exception as e:
                # Clean up on failure
                if len(self.inter_phase_constraints) > constraint_index:
                    self.inter_phase_constraints.pop()

                raise ConfigurationError(
                    f"Failed to add inter-phase constraint: {e}",
                    "Multi-phase state constraint creation error",
                ) from e

    def get_inter_phase_constraint(self, constraint_index: int) -> InterPhaseConstraintInfo:
        """
        Get inter-phase constraint by index.

        Args:
            constraint_index: Zero-based constraint index

        Returns:
            Inter-phase constraint information

        Raises:
            ConfigurationError: If constraint index is out of bounds
        """
        with self.inter_phase_constraint_lock:
            if not (0 <= constraint_index < len(self.inter_phase_constraints)):
                raise ConfigurationError(
                    f"Inter-phase constraint index {constraint_index} out of range "
                    f"[0, {len(self.inter_phase_constraints)})",
                    "Multi-phase state constraint access error",
                )

            return self.inter_phase_constraints[constraint_index]

    def get_all_inter_phase_constraints(self) -> list[InterPhaseConstraintInfo]:
        """Get all inter-phase constraints with thread-safe access."""
        with self.inter_phase_constraint_lock:
            return list(self.inter_phase_constraints)

    def get_inter_phase_constraint_expressions(self) -> list[ca.MX]:
        """Get inter-phase constraint expressions for solver interface."""
        with self.inter_phase_constraint_lock:
            return [info.expression for info in self.inter_phase_constraints]

    # ========================================================================
    # GLOBAL OBJECTIVE MANAGEMENT - Thread-safe
    # ========================================================================

    def set_global_objective(self, expression: ca.MX) -> None:
        """
        Set global objective function expression.

        Args:
            expression: CasADi objective expression

        Raises:
            ConfigurationError: If objective expression is invalid
        """
        if expression is None:
            raise ConfigurationError(
                "Global objective expression cannot be None", "Multi-phase state objective error"
            )

        with self.global_objective_lock:
            self.global_objective_expression = expression
            # Update configuration status
            self._update_all_phases_configured_status()

    def get_global_objective(self) -> ca.MX | None:
        """Get global objective expression with thread-safe access."""
        with self.global_objective_lock:
            return self.global_objective_expression

    # ========================================================================
    # COMPREHENSIVE VALIDATION - Safety-critical
    # ========================================================================

    def validate_multi_phase_structure(self) -> None:
        """
        Comprehensive validation of multi-phase state structure.

        Validates:
        - Phase count and configuration
        - Phase state consistency
        - Global parameter consistency
        - Inter-phase constraint validity
        - Global objective presence

        Raises:
            ConfigurationError: If multi-phase structure is invalid
        """
        # Validate basic structure
        if len(self.phase_states) < 2:
            raise ConfigurationError(
                f"Multi-phase problem must have at least 2 phases, got {len(self.phase_states)}. "
                f"For single-phase problems, use Problem class directly.",
                "Multi-phase state validation error",
            )

        if self.phase_count != len(self.phase_states):
            raise ConfigurationError(
                f"Phase count inconsistency: phase_count={self.phase_count}, "
                f"actual phases={len(self.phase_states)}",
                "Multi-phase state validation error",
            )

        # Validate global objective is set
        if self.global_objective_expression is None:
            raise ConfigurationError(
                "Multi-phase problem requires global objective function. "
                "Use set_global_objective() to define objective over phase endpoints.",
                "Multi-phase state validation error",
            )

        # Validate each phase state
        for i, phase_bundle in enumerate(self.phase_states):
            try:
                # Validate phase bundle integrity
                if phase_bundle.phase_index != i:
                    raise ConfigurationError(
                        f"Phase index inconsistency: bundle has index {phase_bundle.phase_index}, "
                        f"but is at position {i}",
                        "Multi-phase state validation error",
                    )

                # Validate phase is configured
                if not phase_bundle.is_fully_configured():
                    raise ConfigurationError(
                        f"Phase '{phase_bundle.phase_name}' is not fully configured. "
                        f"Status: configured={phase_bundle.configured}, "
                        f"has_dynamics={phase_bundle.has_dynamics}, "
                        f"mesh_configured={phase_bundle.mesh_state.configured}",
                        "Multi-phase state validation error",
                    )

                # Validate phase variable state
                phase_bundle.variable_state.get_variable_counts()  # Triggers validation

            except Exception as e:
                raise ConfigurationError(
                    f"Phase {i} ('{phase_bundle.phase_name}') validation failed: {e}",
                    "Multi-phase state validation error",
                ) from e

        # Validate global parameters
        with self.global_parameter_lock:
            for name, param_info in self.global_parameters.items():
                if param_info.symbol is None:
                    raise ConfigurationError(
                        f"Global parameter '{name}' has None symbol",
                        "Multi-phase state validation error",
                    )

                if not isinstance(param_info.value, (int, float)):
                    raise ConfigurationError(
                        f"Global parameter '{name}' has invalid value type: {type(param_info.value)}",
                        "Multi-phase state validation error",
                    )

        # Validate inter-phase constraints
        with self.inter_phase_constraint_lock:
            for i, constraint_info in enumerate(self.inter_phase_constraints):
                if constraint_info.expression is None:
                    raise ConfigurationError(
                        f"Inter-phase constraint {i} has None expression",
                        "Multi-phase state validation error",
                    )

                if constraint_info.constraint_index != i:
                    raise ConfigurationError(
                        f"Inter-phase constraint index inconsistency: "
                        f"constraint has index {constraint_info.constraint_index}, "
                        f"but is at position {i}",
                        "Multi-phase state validation error",
                    )

        # Mark as validated
        self.structure_validated = True

    def get_validation_summary(self) -> dict[str, Any]:
        """
        Get comprehensive validation summary.

        Returns:
            Dictionary containing validation status and detailed information
        """
        summary = {
            "structure_validated": self.structure_validated,
            "all_phases_configured": self.all_phases_configured,
            "phase_count": self.phase_count,
            "global_parameters_count": len(self.global_parameters),
            "inter_phase_constraints_count": len(self.inter_phase_constraints),
            "has_global_objective": self.global_objective_expression is not None,
            "phases": [],
            "validation_errors": [],
        }

        # Add phase-specific validation info
        for i, phase_bundle in enumerate(self.phase_states):
            try:
                num_states, num_controls = phase_bundle.get_variable_counts()
                phase_info = {
                    "index": i,
                    "name": phase_bundle.phase_name,
                    "configured": phase_bundle.configured,
                    "is_fully_configured": phase_bundle.is_fully_configured(),
                    "has_dynamics": phase_bundle.has_dynamics,
                    "has_objective": phase_bundle.has_objective,
                    "mesh_configured": phase_bundle.mesh_state.configured,
                    "num_states": num_states,
                    "num_controls": num_controls,
                    "num_integrals": phase_bundle.variable_state.num_integrals,
                }
                summary["phases"].append(phase_info)

            except Exception as e:
                summary["validation_errors"].append(f"Phase {i}: {e}")

        return summary

    # ========================================================================
    # SOLVER INTERFACE METHODS - For protocol compliance
    # ========================================================================

    def create_multi_phase_problem_protocol(self, problems: list[ProblemProtocol]) -> Any:
        """
        Create protocol-compliant multi-phase problem representation.

        This method bridges the state management layer to the protocol interface
        required by the solver, maintaining separation of concerns.

        Args:
            problems: List of Problem instances for each phase

        Returns:
            Protocol-compliant representation for solver interface
        """
        # Validate input
        if len(problems) != self.phase_count:
            raise ConfigurationError(
                f"Problem count ({len(problems)}) must match phase count ({self.phase_count})",
                "Multi-phase state protocol creation error",
            )

        # Create protocol representation
        # This will be fully implemented in the solver interface layer
        protocol_data = {
            "phases": problems,
            "global_parameters": self.get_global_parameter_values(),
            "inter_phase_constraints": self.get_inter_phase_constraint_expressions(),
            "global_objective_expression": self.get_global_objective(),
            "phase_count": self.phase_count,
            "structure_validated": self.structure_validated,
        }

        return protocol_data
