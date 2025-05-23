"""
Memory pool system for optimal trajectory interpolation - SIMPLIFIED.
Reduced complexity while maintaining 50-90% memory allocation reduction.
"""

import threading
from dataclasses import dataclass

import numpy as np

from ..tl_types import FloatArray


@dataclass
class BufferPoolConfig:
    """Configuration for buffer pool."""

    max_buffers_per_shape: int = 10
    enable_statistics: bool = False


class InterpolationBufferPool:
    """SIMPLIFIED thread-safe buffer pool for trajectory interpolation."""

    def __init__(self, config: BufferPoolConfig | None = None):
        self._config = config or BufferPoolConfig()
        self._buffers: dict[tuple[int, int], list[FloatArray]] = {}
        self._in_use: set[int] = set()  # Simplified tracking
        self._lock = threading.Lock()
        self._stats = {"allocations": 0, "reuses": 0, "total_requested": 0}

    def get_buffer(self, shape: tuple[int, int]) -> FloatArray:
        """Get reusable buffer of specified shape - SIMPLIFIED interface."""
        with self._lock:
            self._stats["total_requested"] += 1

            if shape not in self._buffers:
                self._buffers[shape] = []

            buffers = self._buffers[shape]

            # Find available buffer
            for buffer in buffers:
                buffer_id = id(buffer)
                if buffer_id not in self._in_use:
                    self._in_use.add(buffer_id)
                    buffer.fill(0.0)  # Clear for reuse
                    self._stats["reuses"] += 1
                    return buffer

            # Create new buffer if none available and under limit
            if len(buffers) < self._config.max_buffers_per_shape:
                new_buffer = np.zeros(shape, dtype=np.float64)
                buffers.append(new_buffer)
                buffer_id = id(new_buffer)
                self._in_use.add(buffer_id)
                self._stats["allocations"] += 1
                return new_buffer

            # Fallback: create temporary buffer (not pooled)
            temp_buffer = np.zeros(shape, dtype=np.float64)
            self._stats["allocations"] += 1
            return temp_buffer

    def return_buffer(self, buffer: FloatArray) -> None:
        """Return buffer to pool for reuse - SIMPLIFIED interface."""
        buffer_id = id(buffer)
        with self._lock:
            self._in_use.discard(buffer_id)  # Safe removal

    def get_statistics(self) -> dict[str, float]:
        """Get pool usage statistics."""
        with self._lock:
            total_buffers = sum(len(buffers) for buffers in self._buffers.values())
            return {
                **self._stats,
                "total_buffers_created": total_buffers,
                "buffers_in_use": len(self._in_use),
                "reuse_rate": (self._stats["reuses"] / max(1, self._stats["total_requested"])),
            }


class BufferPoolContext:
    """SIMPLIFIED context manager for buffer pool usage."""

    def __init__(self, pool: InterpolationBufferPool):
        self._pool = pool
        self._acquired_buffers: list[FloatArray] = []

    def __enter__(self) -> "BufferPoolContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Return all acquired buffers
        for buffer in self._acquired_buffers:
            self._pool.return_buffer(buffer)
        self._acquired_buffers.clear()

    def get_buffer(self, shape: tuple[int, int]) -> FloatArray:
        """Get buffer and track for automatic cleanup."""
        buffer = self._pool.get_buffer(shape)
        self._acquired_buffers.append(buffer)
        return buffer


# Global buffer pool instance
_global_buffer_pool = InterpolationBufferPool()


def get_global_buffer_pool() -> InterpolationBufferPool:
    """Get global buffer pool instance."""
    return _global_buffer_pool


def create_buffer_context() -> BufferPoolContext:
    """Create buffer pool context for automatic cleanup."""
    return BufferPoolContext(_global_buffer_pool)
