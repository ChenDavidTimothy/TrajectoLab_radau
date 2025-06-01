import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import casadi as ca

from ..tl_types import PhaseID


@dataclass
class ExpressionCacheKey:
    """Cache key for CasADi expressions."""

    expression_type: str
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    num_integrals: int
    expression_hash: str
    phase_id: PhaseID | None = None  # For phase-specific caching

    def __post_init__(self) -> None:
        """Create deterministic hash for cache lookup."""
        key_data = (
            self.expression_type,
            self.state_names,
            self.control_names,
            self.num_integrals,
            self.expression_hash,
            self.phase_id,
        )
        self.cache_key = hashlib.sha256(str(key_data).encode()).hexdigest()


class CasADiExpressionCache:
    """Unified cache for ALL expensive CasADi expressions with thread safety and multiphase support."""

    _instance: ClassVar["CasADiExpressionCache | None"] = None
    _cache: ClassVar[dict[str, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "CasADiExpressionCache":
        """Singleton pattern for global cache."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_dynamics_function(
        self, cache_key: ExpressionCacheKey, builder_func: Callable[[], ca.Function]
    ) -> ca.Function:
        """Get cached dynamics function or build if not cached."""
        return self._get_cached_item(f"dynamics_{cache_key.cache_key}", builder_func)

    def get_objective_function(
        self, cache_key: ExpressionCacheKey, builder_func: Callable[[], ca.Function]
    ) -> ca.Function:
        """Get cached objective function or build if not cached."""
        return self._get_cached_item(f"objective_{cache_key.cache_key}", builder_func)

    def get_integrand_functions(
        self, cache_key: ExpressionCacheKey, builder_func: Callable[[], list[ca.Function]]
    ) -> list[ca.Function]:
        """Get cached integrand functions or build if not cached."""
        result = self._get_cached_item(f"integrand_{cache_key.cache_key}", builder_func)
        return cast(list[ca.Function], result)

    def _get_cached_item(self, full_key: str, builder_func: Callable[[], Any]) -> Any:
        """Unified cache retrieval method."""
        with self._lock:
            if full_key not in self._cache:
                self._cache[full_key] = builder_func()
            return self._cache[full_key]

    def clear_cache(self) -> None:
        """Clear all cached expressions."""
        with self._lock:
            self._cache.clear()

    def get_cache_size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


# Global cache instance
_expression_cache = CasADiExpressionCache()


def create_cache_key_from_phase_state(
    phase_def: Any,  # PhaseDefinition type from unified storage
    expression_type: str,
    expression_hash: str | None = None,
) -> ExpressionCacheKey:
    """Create cache key from phase state using unified storage."""
    state_names = tuple(phase_def.state_names)
    control_names = tuple(phase_def.control_names)

    # Create expression hash if not provided
    if expression_hash is None:
        # Use variable structure as hash
        hash_data = (state_names, control_names, phase_def.num_integrals, phase_def.phase_id)
        expression_hash = hashlib.sha256(str(hash_data).encode()).hexdigest()[:16]

    return ExpressionCacheKey(
        expression_type=expression_type,
        state_names=state_names,
        control_names=control_names,
        num_integrals=phase_def.num_integrals,
        expression_hash=expression_hash,
        phase_id=phase_def.phase_id,
    )


def create_cache_key_from_multiphase_state(
    multiphase_state: Any,  # MultiPhaseVariableState type from unified storage
    expression_type: str,
    expression_hash: str | None = None,
) -> ExpressionCacheKey:
    """Create cache key from multiphase state using unified storage."""
    # Collect all state and control names across phases
    all_state_names = []
    all_control_names = []
    total_integrals = 0

    for phase_id in sorted(multiphase_state.phases.keys()):
        phase_def = multiphase_state.phases[phase_id]
        all_state_names.extend([f"p{phase_id}_{name}" for name in phase_def.state_names])
        all_control_names.extend([f"p{phase_id}_{name}" for name in phase_def.control_names])
        total_integrals += phase_def.num_integrals

    state_names = tuple(all_state_names)
    control_names = tuple(all_control_names)

    # Create expression hash if not provided
    if expression_hash is None:
        # Use multiphase structure as hash
        phase_structure = {
            phase_id: (
                len(phase_def.state_names),
                len(phase_def.control_names),
                phase_def.num_integrals,
            )
            for phase_id, phase_def in multiphase_state.phases.items()
        }
        static_params = multiphase_state.static_parameters.get_parameter_count()
        hash_data = (
            state_names,
            control_names,
            total_integrals,
            str(phase_structure),
            static_params,
        )
        expression_hash = hashlib.sha256(str(hash_data).encode()).hexdigest()[:16]

    return ExpressionCacheKey(
        expression_type=expression_type,
        state_names=state_names,
        control_names=control_names,
        num_integrals=total_integrals,
        expression_hash=expression_hash,
        phase_id=None,  # Multiphase cache keys don't have specific phase
    )


def get_global_expression_cache() -> CasADiExpressionCache:
    """Get global expression cache instance."""
    return _expression_cache
