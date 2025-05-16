"""Polynomial interpolation utilities for adaptive mesh refinement."""

import logging
from typing import Any, Optional, Protocol, Union

import numpy as np

from trajectolab.radau import compute_barycentric_weights, evaluate_lagrange_polynomial_at_point
from trajectolab.trajectolab_types import (
    CasADiDM,
    _Matrix,
    _NormalizedTimePoint,
    _Vector,
)

logger = logging.getLogger(__name__)


class DummyEvaluator(Protocol):
    """Protocol for dummy evaluator function."""

    def __call__(self, tau: Union[_NormalizedTimePoint, _Vector]) -> Union[_Matrix, _Vector]: ...


def extract_and_prepare_array(
    casadi_value: Any,
    expected_rows: int,
    expected_cols: int,
) -> _Matrix:
    """Extract numerical value from CasADi/Python types and ensure correct shape.

    Args:
        casadi_value: Value to convert, can be CasADi object or NumPy array
        expected_rows: Expected number of rows
        expected_cols: Expected number of columns

    Returns:
        2D NumPy array with float64 dtype and correct shape

    Raises:
        TypeError: If casadi_value has unsupported type
        ValueError: If the shape cannot be adjusted to match expectations
    """
    import casadi as ca

    np_array_intermediate: np.ndarray

    # Convert to numpy array based on type
    if isinstance(casadi_value, (ca.MX, ca.SX)):
        if casadi_value.is_constant() and not casadi_value.is_symbolic():
            try:
                dm_val = ca.DM(casadi_value)
                np_array_intermediate = np.array(dm_val.toarray(), dtype=np.float64)
            except Exception as e:
                logger.warning(
                    f"Could not convert symbolic CasADi type {type(casadi_value)} to DM/numpy directly: {e}"
                )
                np_array_intermediate = np.array([], dtype=np.float64)
        elif casadi_value.is_symbolic():
            logger.warning("Attempting to convert symbolic CasADi type to NumPy. This may fail.")
            try:
                np_array_intermediate = np.array(CasADiDM(casadi_value), dtype=np.float64)
            except (RuntimeError, TypeError, ValueError):
                np_array_intermediate = np.array([], dtype=np.float64)
        else:
            try:
                dm_val = ca.DM(casadi_value)
                np_array_intermediate = np.array(dm_val.toarray(), dtype=np.float64)
            except Exception as e:
                logger.warning(f"Could not convert non-symbolic CasADi type to DM/numpy: {e}")
                np_array_intermediate = np.array([], dtype=np.float64)
    elif isinstance(casadi_value, ca.DM):
        np_array_intermediate = np.array(casadi_value.toarray(), dtype=np.float64)
    elif isinstance(casadi_value, np.ndarray):
        np_array_intermediate = casadi_value.astype(np.float64)
    elif isinstance(casadi_value, (list, tuple)):
        np_array_intermediate = np.array(casadi_value, dtype=np.float64)
    else:
        # Handle scalar values
        if isinstance(casadi_value, (int, float, np.number)):
            if expected_rows == 1 and expected_cols == 1:
                return np.array([[float(casadi_value)]], dtype=np.float64)
            elif expected_rows == 1:
                return np.array([float(casadi_value)], dtype=np.float64).reshape(1, -1)
            elif expected_cols == 1:
                return np.array([float(casadi_value)], dtype=np.float64).reshape(-1, 1)

        raise TypeError(f"Unsupported type for casadi_value: {type(casadi_value)}")

    # Ensure float64
    if np_array_intermediate.dtype != np.float64:
        np_array_intermediate = np_array_intermediate.astype(np.float64)

    # Handle empty cases
    if expected_rows == 0 and expected_cols == 0:
        return np.empty((0, 0), dtype=np.float64)
    if expected_rows == 0 and np_array_intermediate.size > 0:
        return np.empty(
            (0, np_array_intermediate.shape[1] if np_array_intermediate.ndim == 2 else 0),
            dtype=np.float64,
        )
    if np_array_intermediate.size == 0:
        return np.empty((expected_rows, expected_cols), dtype=np.float64)

    # Ensure 2D shape
    if np_array_intermediate.ndim == 0:  # Handle scalar value
        np_array_intermediate = np.array([[float(np_array_intermediate)]], dtype=np.float64)
    elif np_array_intermediate.ndim == 1:
        if expected_rows == 1 and (
            expected_cols == 0 or len(np_array_intermediate) == expected_cols
        ):
            np_array_intermediate = np_array_intermediate.reshape(1, -1)
        elif expected_cols == 1 and (
            expected_rows == 0 or len(np_array_intermediate) == expected_rows
        ):
            np_array_intermediate = np_array_intermediate.reshape(-1, 1)
        elif expected_rows == 1 and np_array_intermediate.size == expected_cols:
            np_array_intermediate = np_array_intermediate.reshape(1, expected_cols)
        elif expected_cols == 1 and np_array_intermediate.size == expected_rows:
            np_array_intermediate = np_array_intermediate.reshape(expected_rows, 1)
        else:
            logger.debug(
                f"Ambiguous 1D to 2D conversion. Expected ({expected_rows},{expected_cols})"
            )
            if np_array_intermediate.size == expected_cols and expected_rows == 1:
                np_array_intermediate = np_array_intermediate.reshape(1, expected_cols)
            elif np_array_intermediate.size == expected_rows and expected_cols == 1:
                np_array_intermediate = np_array_intermediate.reshape(expected_rows, 1)
            elif np_array_intermediate.size == expected_rows * expected_cols:
                np_array_intermediate = np_array_intermediate.reshape(expected_rows, expected_cols)
            else:
                np_array_intermediate = np_array_intermediate.reshape(1, -1)

    # Final shape check and adjustment
    if (
        np_array_intermediate.shape == (expected_cols, expected_rows)
        and expected_rows != expected_cols
    ):
        np_array_intermediate = np_array_intermediate.T

    if np_array_intermediate.shape != (expected_rows, expected_cols):
        if np_array_intermediate.size == expected_rows * expected_cols:
            try:
                np_array_intermediate = np_array_intermediate.reshape(expected_rows, expected_cols)
            except ValueError as e:
                logger.error(f"Final shape mismatch and reshape failed: {e}")
                raise ValueError(
                    f"Array shape mismatch: Expected ({expected_rows},{expected_cols}), "
                    f"got {np_array_intermediate.shape} after processing."
                ) from e
        else:
            logger.error("Final shape and size mismatch.")
            raise ValueError(
                f"Array shape/size mismatch: Expected ({expected_rows},{expected_cols}), "
                f"got {np_array_intermediate.shape}."
            )
    return np_array_intermediate


class PolynomialInterpolant:
    """Implements Lagrange polynomial interpolation with barycentric weights."""

    def __init__(
        self, nodes: _Vector, values: _Matrix, barycentric_weights: Optional[_Vector] = None
    ):
        """Initialize polynomial interpolant.

        Args:
            nodes: Interpolation nodes
            values: Values at nodes (each row is a variable, each column a node)
            barycentric_weights: Optional precomputed weights for efficiency

        Raises:
            ValueError: If nodes and values dimensions don't match
        """
        self.nodes_array = np.asarray(nodes, dtype=np.float64).flatten()

        # Ensure values_at_nodes is 2D and float64
        _values = np.asarray(values, dtype=np.float64)
        if _values.ndim == 0 and _values.size == 1:
            _values = _values.reshape(1, 1)
        elif _values.ndim == 1:
            if len(self.nodes_array) == _values.size:
                _values = _values.reshape(1, -1)
            else:
                _values = _values.reshape(-1, 1)

        self.values_at_nodes = _values
        self.num_vars, self.num_nodes_val = self.values_at_nodes.shape
        self.num_nodes_pts = len(self.nodes_array)

        if self.num_nodes_pts == 0 and self.num_vars > 0 and self.num_nodes_val > 0:
            raise ValueError(
                "Cannot create interpolant with non-empty values but empty nodes array."
            )
        if (
            self.num_nodes_pts > 0
            and self.num_vars > 0
            and self.num_nodes_pts != self.num_nodes_val
        ):
            raise ValueError(
                f"Mismatch in number of nodes ({self.num_nodes_pts}) and "
                f"columns in values_at_nodes ({self.num_nodes_val})"
            )

        if barycentric_weights is None:
            if self.num_nodes_pts > 0:
                self.bary_weights = compute_barycentric_weights(self.nodes_array)
            else:
                self.bary_weights = np.array([], dtype=np.float64)
        else:
            self.bary_weights = np.asarray(barycentric_weights, dtype=np.float64)

        if self.num_nodes_pts > 0 and len(self.bary_weights) != self.num_nodes_pts:
            raise ValueError("Barycentric weights length does not match nodes length")

    def __call__(
        self, points: Union[_NormalizedTimePoint, _Vector]
    ) -> Union[_Matrix, _Vector, float, np.number]:
        """Evaluate the interpolant at given points.

        Args:
            points: Point(s) where to evaluate the interpolant

        Returns:
            Interpolated values at points.
            If points is scalar and num_vars=1, returns scalar.
            If points is scalar and num_vars>1, returns vector of length num_vars.
            If points is vector, returns matrix of shape (num_vars, len(points)).
        """
        is_scalar_input_point = np.isscalar(points)
        zeta_arr: _Vector = np.atleast_1d(np.asarray(points, dtype=np.float64))

        if self.num_vars == 0 or self.num_nodes_pts == 0:
            empty_shape_matrix = (self.num_vars, len(zeta_arr))
            return np.empty(
                (self.num_vars,) if is_scalar_input_point else empty_shape_matrix,
                dtype=np.float64,
            )

        result = np.zeros((self.num_vars, len(zeta_arr)), dtype=np.float64)

        for i, zeta_val in enumerate(zeta_arr):
            scalar_zeta: float = float(zeta_val)
            L_j: _Vector = evaluate_lagrange_polynomial_at_point(
                self.nodes_array, self.bary_weights, scalar_zeta
            )
            result[:, i] = np.dot(self.values_at_nodes, L_j)

        if self.num_vars == 1:
            final_result_vector = result.flatten()
            if is_scalar_input_point and len(final_result_vector) == 1:
                return float(final_result_vector[0])  # Return scalar
            return final_result_vector

        return result[:, 0] if is_scalar_input_point else result


def get_polynomial_interpolant(
    nodes: _Vector, values: _Matrix, barycentric_weights: Optional[_Vector] = None
) -> PolynomialInterpolant:
    """Create a polynomial interpolant.

    Args:
        nodes: Interpolation nodes
        values: Values at nodes
        barycentric_weights: Optional precomputed weights for efficiency

    Returns:
        PolynomialInterpolant object
    """
    return PolynomialInterpolant(nodes, values, barycentric_weights)


def dummy_evaluator(tau: Union[_NormalizedTimePoint, _Vector]) -> Union[_Matrix, _Vector]:
    """Provides a dummy evaluator for cases where interpolation is not applicable.

    Args:
        tau: Point(s) where to evaluate

    Returns:
        Empty arrays of appropriate shape (matching PolynomialInterpolant behavior)
    """
    if np.isscalar(tau):
        return np.zeros(0, dtype=np.float64)

    tau_arr = np.atleast_1d(tau)
    return np.zeros((0, len(tau_arr)), dtype=np.float64)
