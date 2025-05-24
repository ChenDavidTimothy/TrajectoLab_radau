"""
TrajectoLab custom exceptions for fail-fast error handling.

This module defines a minimal set of TrajectoLab-specific exceptions that represent
critical failure modes unique to TrajectoLab's operational logic, configuration,
and data processing responsibilities.
"""

import logging


logger = logging.getLogger(__name__)


class TrajectoLabBaseError(Exception):
    """
    Base class for all TrajectoLab-specific errors.

    These errors represent critical failures in TrajectoLab's own logic,
    configuration, or data processing that require immediate failure.
    """

    def __init__(self, message: str, context: str | None = None) -> None:
        self.message = message
        self.context = context
        super().__init__(self._format_message())

        # Log critical TrajectoLab errors
        logger.error(f"TrajectoLab Error: {self._format_message()}")

    def _format_message(self) -> str:
        if self.context:
            return f"{self.message} (Context: {self.context})"
        return self.message


class ConfigurationError(TrajectoLabBaseError):
    """
    Raised when TrajectoLab configuration is invalid or incomplete purely based on user input.

    This indicates TrajectoLab's internal configuration state is wrong,
    such as attempting to solve without proper mesh setup, or accessing
    variables before they're defined.
    """

    pass


class DataIntegrityError(TrajectoLabBaseError):
    """
    Raised when TrajectoLab detects data corruption or inconsistency.

    This includes shape mismatches in data TrajectoLab prepares,
    unexpected NaN/Inf values in TrajectoLab's intermediate results,
    or internal state inconsistencies.
    """

    pass


class SolutionExtractionError(TrajectoLabBaseError):
    """
    Raised when TrajectoLab fails to extract or process solution data.

    This indicates TrajectoLab's solution processing logic encountered
    a critical failure, not solver convergence issues.
    """

    pass


class InterpolationError(TrajectoLabBaseError):
    """
    Raised when TrajectoLab's interpolation/propagation algorithms fail.

    This indicates critical failures in TrajectoLab's adaptive mesh
    refinement logic, not numerical convergence issues.
    """

    pass
