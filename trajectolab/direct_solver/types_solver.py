"""
Type definitions and data structure containers for the direct solver with multi-phase support.

This module defines internal type aliases and data structures for both single-phase and
multi-phase optimal control solvers, faithfully implementing the CGPOPS NLP structure
for multi-phase problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import casadi as ca

from ..tl_types import (
    FloatArray,
)


# --- SINGLE-PHASE INTERNAL TYPE ALIASES (Preserved) ---
_VariableBundle: TypeAlias = tuple[
    ca.MX,  # initial_time
    ca.MX,  # terminal_time
    list[ca.MX],  # state_at_mesh_nodes
    list[ca.MX],  # control_variables
    ca.MX | None,  # integral_variables
]

_IntervalBundle: TypeAlias = tuple[ca.MX, ca.MX | None]  # state_matrix, interior_nodes


# --- MULTI-PHASE INTERNAL TYPE ALIASES ---
_PhaseVariableBundle: TypeAlias = tuple[
    ca.MX,  # phase_initial_time
    ca.MX,  # phase_terminal_time
    list[ca.MX],  # phase_state_at_mesh_nodes
    list[ca.MX],  # phase_control_variables
    ca.MX | None,  # phase_integral_variables
]
"""
Single phase variable bundle within multi-phase context.
Corresponds to z^(p) components in CGPOPS Equation (31).
"""

_MultiPhaseVariableBundle: TypeAlias = tuple[
    list[_PhaseVariableBundle],  # phase_variable_bundles z^(1), ..., z^(P)
    ca.MX | None,  # global_parameters s₁, ..., sₙₛ
    list[ca.MX],  # inter_phase_constraint_expressions
]
"""
Complete multi-phase variable bundle matching CGPOPS NLP structure.
Implements z = [z^(1), ..., z^(P), s₁, ..., sₙₛ]ᵀ from Equation (31).
"""

_PhaseEndpointBundle: TypeAlias = tuple[
    ca.MX,  # initial_state_vector Y_1^(p)
    ca.MX,  # initial_time t_0^(p)
    ca.MX,  # final_state_vector Y_{N^(p)+1}^(p)
    ca.MX,  # final_time t_f^(p)
    ca.MX | None,  # integral_vector Q^(p)
]
"""
Phase endpoint vector bundle E^(p) as defined in CGPOPS Equation (15).
E^(p) = [Y_1^(p), t_0^(p), Y_{N^(p)+1}^(p), t_f^(p), Q^(p)]
"""


# --- SINGLE-PHASE DATA CONTAINERS (Preserved) ---
@dataclass
class VariableReferences:
    """Container for single-phase optimization variable references."""

    initial_time: ca.MX
    terminal_time: ca.MX
    state_at_mesh_nodes: list[ca.MX]
    control_variables: list[ca.MX]
    integral_variables: ca.MX | None
    state_matrices: list[ca.MX] = field(default_factory=list)
    interior_variables: list[ca.MX | None] = field(default_factory=list)


@dataclass
class MetadataBundle:
    """Container for single-phase solver metadata."""

    local_state_tau: list[FloatArray] = field(default_factory=list)
    local_control_tau: list[FloatArray] = field(default_factory=list)
    global_mesh_nodes: FloatArray = field(default_factory=lambda: FloatArray([]))
    objective_expression: ca.MX | None = None


# --- MULTI-PHASE DATA CONTAINERS ---
@dataclass
class PhaseVariableReferences:
    """
    Container for single phase optimization variables within multi-phase context.

    Represents z^(p) component of the multi-phase NLP decision vector from
    CGPOPS Equation (31), containing all optimization variables for phase p.
    """

    phase_index: int
    initial_time: ca.MX
    terminal_time: ca.MX
    state_at_mesh_nodes: list[ca.MX]
    control_variables: list[ca.MX]
    integral_variables: ca.MX | None
    state_matrices: list[ca.MX] = field(default_factory=list)
    interior_variables: list[ca.MX | None] = field(default_factory=list)

    # Phase-specific metadata
    num_states: int = 0
    num_controls: int = 0
    num_integrals: int = 0
    num_mesh_intervals: int = 0
    collocation_points_per_interval: list[int] = field(default_factory=list)

    def to_single_phase_reference(self) -> VariableReferences:
        """Convert to single-phase VariableReferences for compatibility."""
        return VariableReferences(
            initial_time=self.initial_time,
            terminal_time=self.terminal_time,
            state_at_mesh_nodes=self.state_at_mesh_nodes,
            control_variables=self.control_variables,
            integral_variables=self.integral_variables,
            state_matrices=self.state_matrices,
            interior_variables=self.interior_variables,
        )

    def get_endpoint_vector_symbolic(self) -> _PhaseEndpointBundle:
        """
        Get symbolic endpoint vector E^(p) for this phase.

        Returns symbolic representation of endpoint vector as defined in
        CGPOPS Equation (15): E^(p) = [Y_1^(p), t_0^(p), Y_{N^(p)+1}^(p), t_f^(p), Q^(p)]
        """
        if not self.state_at_mesh_nodes:
            raise ValueError(f"Phase {self.phase_index} has no state nodes for endpoint vector")

        initial_state = self.state_at_mesh_nodes[0]  # Y_1^(p)
        final_state = self.state_at_mesh_nodes[-1]  # Y_{N^(p)+1}^(p)

        return (
            initial_state,  # Y_1^(p)
            self.initial_time,  # t_0^(p)
            final_state,  # Y_{N^(p)+1}^(p)
            self.terminal_time,  # t_f^(p)
            self.integral_variables,  # Q^(p)
        )


@dataclass
class MultiPhaseVariableReferences:
    """
    Container for multi-phase optimization variable references.

    Implements the complete NLP decision vector structure from CGPOPS Equation (31):
    z = [z^(1), ..., z^(P), s₁, ..., sₙₛ]ᵀ

    This container maintains the hierarchical block structure essential for
    exploiting the block-diagonal sparsity pattern described in CGPOPS Section 4.2.
    """

    # Phase-specific variables z^(1), ..., z^(P)
    phase_variables: list[PhaseVariableReferences] = field(default_factory=list)

    # Global static parameters s₁, ..., sₙₛ (shared across all phases)
    global_parameters: ca.MX | None = None
    global_parameter_names: list[str] = field(default_factory=list)
    global_parameter_values: dict[str, float] = field(default_factory=dict)

    # Inter-phase constraint variables and expressions
    inter_phase_constraint_expressions: list[ca.MX] = field(default_factory=list)
    inter_phase_constraint_multipliers: ca.MX | None = None

    # Multi-phase structure metadata
    phase_count: int = 0
    total_states: int = 0
    total_controls: int = 0
    total_integrals: int = 0
    total_collocation_points: int = 0

    def __post_init__(self) -> None:
        """Validate multi-phase variable structure after initialization."""
        if self.phase_count != len(self.phase_variables):
            raise ValueError(
                f"Phase count mismatch: expected {self.phase_count}, "
                f"got {len(self.phase_variables)} phase variables"
            )

        # Update totals from phase data
        self.total_states = sum(phase.num_states for phase in self.phase_variables)
        self.total_controls = sum(phase.num_controls for phase in self.phase_variables)
        self.total_integrals = sum(phase.num_integrals for phase in self.phase_variables)
        self.total_collocation_points = sum(
            sum(phase.collocation_points_per_interval) for phase in self.phase_variables
        )

    def get_phase_variables(self, phase_index: int) -> PhaseVariableReferences:
        """Get variables for specific phase with bounds checking."""
        if not (0 <= phase_index < len(self.phase_variables)):
            raise IndexError(
                f"Phase index {phase_index} out of range [0, {len(self.phase_variables)})"
            )
        return self.phase_variables[phase_index]

    def get_all_phase_endpoint_vectors(self) -> list[_PhaseEndpointBundle]:
        """
        Get endpoint vectors E^(p) for all phases.

        Returns list of symbolic endpoint vectors as defined in CGPOPS Equation (15),
        used for constructing inter-phase event constraints and global objective function.
        """
        return [phase.get_endpoint_vector_symbolic() for phase in self.phase_variables]

    def validate_phase_consistency(self) -> None:
        """
        Validate consistency across all phases.

        Performs comprehensive validation of multi-phase variable structure,
        ensuring all phases have consistent variable organization and proper
        indexing for block-diagonal NLP construction.
        """
        # Validate phase indices are sequential
        for i, phase in enumerate(self.phase_variables):
            if phase.phase_index != i:
                raise ValueError(
                    f"Phase index inconsistency: phase at position {i} "
                    f"has index {phase.phase_index}"
                )

        # Validate all phases have required variable structures
        for i, phase in enumerate(self.phase_variables):
            if not phase.state_at_mesh_nodes:
                raise ValueError(f"Phase {i} missing state nodes")
            if not phase.control_variables:
                raise ValueError(f"Phase {i} missing control variables")
            if phase.num_mesh_intervals != len(phase.control_variables):
                raise ValueError(
                    f"Phase {i} mesh interval count mismatch: "
                    f"expected {phase.num_mesh_intervals}, got {len(phase.control_variables)}"
                )

        # Validate global parameters consistency
        if self.global_parameters is not None:
            if len(self.global_parameter_names) != self.global_parameters.size1():
                raise ValueError(
                    f"Global parameter count mismatch: "
                    f"names={len(self.global_parameter_names)}, "
                    f"variables={self.global_parameters.size1()}"
                )

    def get_phase_variable_counts(self) -> list[tuple[int, int, int]]:
        """Get (num_states, num_controls, num_integrals) for each phase."""
        return [
            (phase.num_states, phase.num_controls, phase.num_integrals)
            for phase in self.phase_variables
        ]

    def get_nlp_variable_count(self) -> int:
        """
        Get total NLP variable count.

        Computes total number of decision variables in the multi-phase NLP,
        including all phase variables and global parameters.
        """
        total_vars = 0

        # Count phase variables: states, controls, integrals, times
        for phase in self.phase_variables:
            # State variables: (num_intervals + 1) * num_states
            total_vars += (phase.num_mesh_intervals + 1) * phase.num_states

            # Control variables: sum over intervals of (colloc_points * num_controls)
            total_vars += sum(
                colloc_points * phase.num_controls
                for colloc_points in phase.collocation_points_per_interval
            )

            # Integral variables
            total_vars += phase.num_integrals

            # Time variables: t_0 and t_f
            total_vars += 2

        # Global parameter variables
        if self.global_parameters is not None:
            total_vars += self.global_parameters.size1()

        return total_vars


@dataclass
class PhaseMetadataBundle:
    """Container for single phase solver metadata within multi-phase context."""

    phase_index: int
    local_state_tau: list[FloatArray] = field(default_factory=list)
    local_control_tau: list[FloatArray] = field(default_factory=list)
    global_mesh_nodes: FloatArray = field(default_factory=lambda: FloatArray([]))
    phase_objective_expression: ca.MX | None = None
    phase_constraint_count: int = 0
    phase_defect_constraint_count: int = 0
    phase_path_constraint_count: int = 0

    def to_single_phase_metadata(self) -> MetadataBundle:
        """Convert to single-phase MetadataBundle for compatibility."""
        return MetadataBundle(
            local_state_tau=self.local_state_tau,
            local_control_tau=self.local_control_tau,
            global_mesh_nodes=self.global_mesh_nodes,
            objective_expression=self.phase_objective_expression,
        )


@dataclass
class MultiPhaseMetadataBundle:
    """
    Container for multi-phase solver metadata.

    Maintains metadata for each phase plus global multi-phase information,
    essential for solution extraction and analysis of multi-phase problems.
    """

    # Phase-specific metadata
    phase_metadata: list[PhaseMetadataBundle] = field(default_factory=list)

    # Global multi-phase metadata
    global_objective_expression: ca.MX | None = None
    inter_phase_constraint_expressions: list[ca.MX] = field(default_factory=list)
    inter_phase_constraint_count: int = 0

    # Solution extraction metadata
    phase_count: int = 0
    total_nlp_variables: int = 0
    total_nlp_constraints: int = 0
    nlp_sparsity_pattern: dict[str, Any] = field(default_factory=dict)

    # Performance metadata
    phase_setup_times: list[float] = field(default_factory=list)
    constraint_evaluation_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate multi-phase metadata structure after initialization."""
        if self.phase_count != len(self.phase_metadata):
            raise ValueError(
                f"Phase count mismatch: expected {self.phase_count}, "
                f"got {len(self.phase_metadata)} phase metadata entries"
            )

        # Update constraint counts
        self.total_nlp_constraints = (
            sum(phase.phase_constraint_count for phase in self.phase_metadata)
            + self.inter_phase_constraint_count
        )

    def get_phase_metadata(self, phase_index: int) -> PhaseMetadataBundle:
        """Get metadata for specific phase with bounds checking."""
        if not (0 <= phase_index < len(self.phase_metadata)):
            raise IndexError(
                f"Phase index {phase_index} out of range [0, {len(self.phase_metadata)})"
            )
        return self.phase_metadata[phase_index]

    def get_constraint_distribution(self) -> dict[str, int]:
        """Get distribution of constraints across phases and constraint types."""
        distribution = {
            "total_constraints": self.total_nlp_constraints,
            "inter_phase_constraints": self.inter_phase_constraint_count,
            "phase_constraints": {},
            "defect_constraints": 0,
            "path_constraints": 0,
        }

        for i, phase in enumerate(self.phase_metadata):
            distribution["phase_constraints"][f"phase_{i}"] = phase.phase_constraint_count
            distribution["defect_constraints"] += phase.phase_defect_constraint_count
            distribution["path_constraints"] += phase.phase_path_constraint_count

        return distribution

    def analyze_sparsity_structure(self) -> dict[str, Any]:
        """
        Analyze the block-diagonal sparsity structure of the multi-phase NLP.

        Returns analysis of the sparsity pattern that should match the
        block-diagonal structure described in CGPOPS Section 4.2.
        """
        analysis = {
            "total_phases": self.phase_count,
            "diagonal_blocks": [],
            "off_diagonal_coupling": {
                "inter_phase_constraints": self.inter_phase_constraint_count,
                "global_parameter_coupling": len(self.phase_metadata) > 1,
            },
            "sparsity_ratio": 0.0,
        }

        # Analyze diagonal block structure (phase-internal constraints)
        for i, phase in enumerate(self.phase_metadata):
            block_info = {
                "phase_index": i,
                "block_size": phase.phase_constraint_count,
                "defect_constraints": phase.phase_defect_constraint_count,
                "path_constraints": phase.phase_path_constraint_count,
            }
            analysis["diagonal_blocks"].append(block_info)

        # Estimate sparsity ratio (assumes proper block-diagonal structure)
        if self.total_nlp_variables > 0 and self.total_nlp_constraints > 0:
            # In block-diagonal structure, most entries should be zero
            # Only diagonal blocks and inter-phase constraints are non-zero
            diagonal_entries = sum(phase.phase_constraint_count for phase in self.phase_metadata)
            coupling_entries = self.inter_phase_constraint_count * self.phase_count

            total_possible_entries = self.total_nlp_constraints * self.total_nlp_variables
            non_zero_entries = diagonal_entries + coupling_entries

            analysis["sparsity_ratio"] = 1.0 - (non_zero_entries / max(total_possible_entries, 1))

        return analysis
