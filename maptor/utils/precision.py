"""Mathematical precision utilities for MAPTOR.

Provides mathematically honest operations with fail-fast behavior instead of
arbitrary tolerance manipulations. All operations are scale-relative and
condition-number aware for numerical integrity.
"""

import numpy as np

from .constants import MAX_MATHEMATICAL_CONDITION, RELATIVE_PRECISION


def _is_mathematically_zero(value: float, reference_scale: float = 1.0) -> bool:
    """Check if value is mathematically zero relative to reference scale.

    Uses scale-relative precision instead of arbitrary absolute tolerances.
    Essential for problems spanning multiple orders of magnitude.

    Args:
        value: Value to test for mathematical zero
        reference_scale: Reference scale for relative comparison

    Returns:
        True if value is mathematically zero relative to scale

    Examples:
        >>> is_mathematically_zero(1e-10, 1.0)  # True - small relative to 1.0
        >>> is_mathematically_zero(1e-10, 1e-15)  # False - large relative to 1e-15
        >>> is_mathematically_zero(1e-5, 1e6)  # True - small relative to 1e6
    """
    return np.abs(value) <= np.abs(reference_scale) * RELATIVE_PRECISION


def _validate_mathematical_condition(condition_number: float, operation: str) -> None:
    """Validate mathematical conditioning with fail-fast approach.

    Explicitly checks condition numbers against mathematical limits instead
    of silently applying regularization or arbitrary fixes.

    Args:
        condition_number: Condition number to validate
        operation: Description of operation for error context

    Raises:
        DataIntegrityError: If operation is mathematically ill-conditioned

    Examples:
        >>> validate_mathematical_condition(1e5, "matrix inversion")  # OK
        >>> validate_mathematical_condition(1e15, "barycentric weights")  # Raises error
    """
    if condition_number > MAX_MATHEMATICAL_CONDITION:
        from ..exceptions import DataIntegrityError

        raise DataIntegrityError(
            f"{operation} is mathematically ill-conditioned: κ={condition_number:.2e}. "
            f"Maximum reliable condition number: {MAX_MATHEMATICAL_CONDITION:.2e}",
            "Numerical instability detected",
        )


def _safe_division(numerator: float, denominator: float, reference_scale: float = 1.0) -> float:
    """Mathematically safe division with scale-relative precision checking.

    Fails fast when division would lose mathematical meaning instead of
    applying arbitrary regularization or returning misleading results.

    Args:
        numerator: Division numerator
        denominator: Division denominator
        reference_scale: Reference scale for zero detection

    Returns:
        Result of mathematically valid division

    Raises:
        ZeroDivisionError: If denominator is computationally zero

    Examples:
        >>> safe_division(1.0, 1e-10, 1.0)  # OK - denominator not zero relative to scale
        >>> safe_division(1.0, 1e-20, 1.0)  # Raises - denominator too small
        >>> safe_division(1e-10, 1e-15, 1e-12)  # OK - both small but ratio meaningful
    """
    if _is_mathematically_zero(denominator, reference_scale):
        raise ZeroDivisionError(
            f"Division by computational zero: {denominator} (relative to scale {reference_scale})"
        )
    return numerator / denominator
