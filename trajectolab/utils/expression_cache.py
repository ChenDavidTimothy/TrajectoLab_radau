"""
Expression caching system for CasADi functions to improve computational performance.
"""

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import casadi as ca


@dataclass
class ExpressionCacheKey:
    """Cache key for CasADi expressions."""

    expression_type: str
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    num_integrals: int
    expression_hash: str

    def __post_init__(self) -> None:
        """Create deterministic hash for cache lookup."""
        key_data = (
            self.expression_type,
            self.state_names,
            self.control_names,
            self.num_integrals,
            self.expression_hash,
        )
        self.cache_key = hashlib.sha256(str(key_data).encode()).hexdigest()


class CasADiExpressionCache:
    """Unified cache for ALL expensive CasADi expressions with thread safety."""

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
        # Fix: Cast the result to the correct type
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


def create_cache_key_from_variable_state(
    variable_state: Any,  # VariableState type from unified storage
    expression_type: str,
    expression_hash: str | None = None,
) -> ExpressionCacheKey:
    """Create cache key from variable state using unified storage."""
    state_names = tuple(variable_state.get_ordered_state_names())
    control_names = tuple(variable_state.get_ordered_control_names())

    # Create expression hash if not provided
    if expression_hash is None:
        # Use variable structure as hash
        hash_data = (state_names, control_names, variable_state.num_integrals)
        expression_hash = hashlib.sha256(str(hash_data).encode()).hexdigest()[:16]

    return ExpressionCacheKey(
        expression_type=expression_type,
        state_names=state_names,
        control_names=control_names,
        num_integrals=variable_state.num_integrals,
        expression_hash=expression_hash,
    )


def get_global_expression_cache() -> CasADiExpressionCache:
    """Get global expression cache instance."""
    return _expression_cache
