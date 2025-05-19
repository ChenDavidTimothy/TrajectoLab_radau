"""
State data classes for problem definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tl_types import FloatArray, SymExpr, SymType
from ..utils.variable_scaling import ProblemScalingInfo


@dataclass
class VariableState:
    """State for all variables and expressions."""

    # Symbolic variables
    sym_states: dict[str, SymType] = field(default_factory=dict)
    sym_controls: dict[str, SymType] = field(default_factory=dict)
    sym_parameters: dict[str, SymType] = field(default_factory=dict)
    sym_time: SymType | None = None
    sym_time_initial: SymType | None = None
    sym_time_final: SymType | None = None

    # Variable metadata
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    # Expressions
    dynamics_expressions: dict[SymType, SymExpr] = field(default_factory=dict)
    objective_expression: SymExpr | None = None

    # Integral tracking
    integral_expressions: list[SymExpr] = field(default_factory=list)
    integral_symbols: list[SymType] = field(default_factory=list)
    num_integrals: int = 0

    # Time bounds
    t0_bounds: tuple[float, float] = (0.0, 0.0)
    tf_bounds: tuple[float, float] = (1.0, 1.0)

    # Variable scaling
    scaling_info: ProblemScalingInfo | None = field(default=None)


@dataclass
class ConstraintState:
    """State for constraints."""

    constraints: list[SymExpr] = field(default_factory=list)


@dataclass
class MeshState:
    """State for mesh configuration."""

    collocation_points_per_interval: list[int] = field(default_factory=list)
    global_normalized_mesh_nodes: FloatArray | None = None
    configured: bool = False
