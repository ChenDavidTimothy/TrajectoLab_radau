# trajectolab/exceptions.py
"""
TrajectoLab custom exceptions for fail-fast error handling.
"""

import logging


# Library logger - no configuration, user controls output
logger = logging.getLogger(__name__)


class TrajectoLabBaseError(Exception):
    """
    Base class for all TrajectoLab-specific errors.
    """

    def __init__(self, message: str, context: str | None = None) -> None:
        self.message = message
        self.context = context

        # Library logs at DEBUG level - user can promote if needed
        logger.debug("TrajectoLab exception: %s", self._format_message())
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.context:
            return f"{self.message} (Context: {self.context})"
        return self.message


class ConfigurationError(TrajectoLabBaseError):
    """Invalid or incomplete TrajectoLab configuration."""

    pass


class DataIntegrityError(TrajectoLabBaseError):
    """Data corruption or inconsistency in TrajectoLab processing."""

    pass


class SolutionExtractionError(TrajectoLabBaseError):
    """Critical failure in TrajectoLab's solution processing."""

    pass


class InterpolationError(TrajectoLabBaseError):
    """Critical failure in TrajectoLab's interpolation algorithms."""

    pass
