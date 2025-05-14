# radau_pseudospectral_basis.py
import numpy as np
from scipy.special import roots_jacobi # eval_jacobi removed as it was unused

# Tolerance for floating point comparisons
ZERO_TOLERANCE = 1e-12

# ==============================================================================
# LGR Nodes, Weights, and Basis/Collocation Points (GPOPS-II Style)
# ==============================================================================

class RadauBasisComponents:
    """Class representing the mathematical components required for RPM method."""
    
    def __init__(self, 
                 state_approximation_nodes=None, 
                 collocation_nodes=None, 
                 quadrature_weights=None, 
                 differentiation_matrix=None,
                 barycentric_weights_for_state_nodes=None,
                 lagrange_basis_evaluation_at_local_tau_plus_one=None):
        self.state_approximation_nodes = state_approximation_nodes
        self.collocation_nodes = collocation_nodes
        self.quadrature_weights = quadrature_weights
        self.differentiation_matrix = differentiation_matrix
        self.barycentric_weights_for_state_nodes = barycentric_weights_for_state_nodes
        self.lagrange_basis_evaluation_at_local_tau_plus_one = lagrange_basis_evaluation_at_local_tau_plus_one

class RadauNodesAndWeights:
    """Class representing LGR nodes and weights."""
    
    def __init__(self, state_approximation_nodes=None, collocation_nodes=None, quadrature_weights=None):
        self.state_approximation_nodes = state_approximation_nodes
        self.collocation_nodes = collocation_nodes
        self.quadrature_weights = quadrature_weights

def compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes):
    """
    Computes Legendre-Gauss-Radau (LGR) points and related data structures
    according to the GPOPS-II conventions.

    The interval is assumed to be tau in [-1, 1].

    GPOPS-II Convention (as interpreted):
    - num_collocation_nodes: The number of LGR collocation points (denoted Nk in gpops.txt).
    - Collocation Points (Nk total): These are tau_1, ..., tau_Nk.
        - tau_1 = -1.0
        - tau_2, ..., tau_Nk are the Nk-1 roots of P_(Nk-1)^(0,1)(tau) if Nk > 1.
          These are interior to (-1, 1).
    - Basis Points (Nk+1 total): These are the Nk collocation points plus tau_(Nk+1) = +1.0.
        So, state_approximation_nodes = [-1.0, interior_roots..., +1.0].
    - LGR Quadrature Weights (Nk total): Corresponding to the Nk collocation points.
      These are for an integral from -1 to +1.

    Args:
        num_collocation_nodes (int): The number of LGR collocation points (Nk in gpops.txt).
                                     Must be >= 1.

    Returns:
        RadauNodesAndWeights: An object containing:
            - state_approximation_nodes: (Nk+1,) numpy array of sorted basis points.
            - collocation_nodes: (Nk,) numpy array of sorted LGR collocation points.
            - quadrature_weights: (Nk,) numpy array of LGR quadrature weights
                                  corresponding to the collocation_nodes.
    """
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ValueError("Number of collocation points (num_collocation_nodes) must be an integer >= 1.")

    collocation_nodes_list = [-1.0] # Start with the fixed point
    quadrature_weights_list = [0.0] # Placeholder for weight of -1.0, will be set later

    if num_collocation_nodes == 1:
        # Only one collocation point: -1. Basis points: [-1, 1].
        # Weight for a 1-point Radau rule (just -1) covering [-1,1] is 2.
        quadrature_weights_list[0] = 2.0
        # interior_roots will be empty, state_approximation_nodes_list constructed after if/else
    else:
        # num_collocation_nodes > 1
        num_interior_roots = num_collocation_nodes - 1
        
        # interior_roots are in (-1, 1)
        # jacobi_polynomial_weights are weights for the measure (1-x)^alpha * (1+x)^beta = (1+x)
        interior_roots, jacobi_polynomial_weights, _ = roots_jacobi(num_interior_roots, 0, 1, mu=True)

        # Adjust Jacobi weights for the standard Legendre measure (dx) on (-1,1)
        denominator = 1.0 + interior_roots # interior_roots are > -1
        standard_interval_interior_node_weights = jacobi_polynomial_weights / denominator
        
        collocation_nodes_list.extend(interior_roots)
        quadrature_weights_list.extend(standard_interval_interior_node_weights)

        # Weight for the -1 point. For an N-point Left-Radau rule, w_(-1) = 2/N^2.
        # Here N = num_collocation_nodes
        quadrature_weights_list[0] = 2.0 / (num_collocation_nodes**2)

    # Construct state_approximation_nodes_list using the final collocation_nodes_list
    # Ensure collocation_nodes_list elements are unique before adding +1.0 for basis points
    # This is generally true by construction but defensive.
    # For Nk=1, collocation_nodes_list is just [-1.0]
    # For Nk>1, collocation_nodes_list is [-1.0, root1, root2, ...]
    temporary_collocation_nodes = sorted(list(set(collocation_nodes_list))) # Should not change order if already sorted as intended
    
    state_approximation_nodes_list = list(temporary_collocation_nodes) + [1.0]
    
    # Sort collocation points and corresponding weights
    # (interior_roots from roots_jacobi are sorted, -1 is prepended)
    # So, collocation_nodes_list might not be fully sorted if roots are < -1 (not possible)
    # or if sorting after extend is needed.
    # Let's ensure collocation_nodes_list is sorted before creating the final arrays.
    
    # Combine lists for sorting by points
    combined_nodes_and_weights = sorted(zip(collocation_nodes_list, quadrature_weights_list), key=lambda x: x[0])
    collocation_nodes = np.array([item[0] for item in combined_nodes_and_weights])
    quadrature_weights = np.array([item[1] for item in combined_nodes_and_weights])

    # Ensure state_approximation_nodes are sorted and unique.
    # +1.0 could theoretically be an interior root if num_collocation_nodes is very low
    # and P_(Nk-1)^(0,1) has +1 as a root, but this is not expected for Jacobi polys used.
    state_approximation_nodes = np.array(sorted(list(set(state_approximation_nodes_list))))

    # Final check for basis points length
    expected_state_approximation_nodes_length = num_collocation_nodes + 1
    if len(state_approximation_nodes) != expected_state_approximation_nodes_length:
        # Reconstruct carefully if set removed something unexpected (e.g. +1 was an interior root)
        current_state_approximation_nodes_list_for_debug = [-1.0]
        if num_collocation_nodes > 1: # interior_roots only exist if Nk > 1
             current_state_approximation_nodes_list_for_debug.extend(interior_roots) # Use original interior_roots
        current_state_approximation_nodes_list_for_debug.append(1.0)
        state_approximation_nodes = np.array(sorted(list(set(current_state_approximation_nodes_list_for_debug))))
        if len(state_approximation_nodes) != expected_state_approximation_nodes_length:
             raise ValueError(
                 f"Error in basis point construction. Expected {expected_state_approximation_nodes_length}, got {len(state_approximation_nodes)}. "
                 f"Collocation points: {collocation_nodes}, Initial basis list: {state_approximation_nodes_list}"
            )
            
    return RadauNodesAndWeights(
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights
    )

# ==============================================================================
# Barycentric Interpolation Utilities
# ==============================================================================

def _compute_barycentric_weights(nodes):
    """
    Computes barycentric weights for a given set of distinct nodes.
    w_j = 1 / product_{k!=j} (x_j - x_k)
    """
    num_nodes = len(nodes)
    barycentric_weights = np.ones(num_nodes)
    nodes_array = np.asarray(nodes) 
    for j in range(num_nodes):
        node_differences = nodes_array[j] - np.delete(nodes_array, j)
        # Ensure no zero node_differences if nodes were extremely close (should not happen for distinct LGR nodes)
        node_differences[np.abs(node_differences) < ZERO_TOLERANCE * 1e-1] = np.sign(node_differences[np.abs(node_differences) < ZERO_TOLERANCE * 1e-1]) * ZERO_TOLERANCE * 1e-1 \
                                           if np.any(node_differences[np.abs(node_differences) < ZERO_TOLERANCE * 1e-1] !=0) else ZERO_TOLERANCE*1e-1
        barycentric_weights[j] = 1.0 / np.prod(node_differences)
    return barycentric_weights

def _evaluate_lagrange_polynomial_at_point(polynomial_definition_nodes, barycentric_weights, evaluation_point_tau):
    """
    Computes values L_j(eval_pt) of Lagrange polynomials defined over nodes.
    Uses Barycentric formula (second form): L_j(x) = [w_j / (x-x_j)] / sum_k [w_k / (x-x_k)].
    Returns a vector [L_0(eval_pt), L_1(evaluation_point_tau), ..., L_{N-1}(evaluation_point_tau)].
    """
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_polynomial_values_at_evaluation_point = np.zeros(num_polynomial_definition_nodes)

    # Check if evaluation_point_tau is one of the nodes (within tolerance)
    for j in range(num_polynomial_definition_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_polynomial_values_at_evaluation_point[j] = 1.0
            return lagrange_polynomial_values_at_evaluation_point # Exact match, L_j(x_j)=1, L_k(x_j)=0 for k!=j

    barycentric_sum_denominator = 0.0
    weighted_inverse_evaluation_point_differences = np.zeros(num_polynomial_definition_nodes) # Stores w_j / (evaluation_point_tau - polynomial_definition_nodes[j])

    for j in range(num_polynomial_definition_nodes):
        evaluation_point_difference_from_node = evaluation_point_tau - polynomial_definition_nodes[j]
        if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE: 
            # Should have been caught by the exact match loop above.
            # If reached, evaluation_point_tau is extremely close but not identical within ZERO_TOLERANCE.
            # Treat as near-singular, which can lead to issues.
            # However, the barycentric formula is designed for this.
            # Using a small, non-zero evaluation_point_difference_from_node might be better than ZERO_TOLERANCE if evaluation_point_difference_from_node is truly non-zero.
            evaluation_point_difference_from_node = np.sign(evaluation_point_difference_from_node) * ZERO_TOLERANCE if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE else evaluation_point_difference_from_node # Ensure non-zero
            if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE: # Still effectively zero.
                 weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / (np.sign(polynomial_definition_nodes[j]) *1e-100 if polynomial_definition_nodes[j]!=0 else 1e-100) # Avoid NaN if bary_weight is also 0
                 # This term will likely be huge or tiny.
            else:
                 weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / evaluation_point_difference_from_node

        else:
            weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / evaluation_point_difference_from_node
        barycentric_sum_denominator += weighted_inverse_evaluation_point_differences[j]

    if abs(barycentric_sum_denominator) < ZERO_TOLERANCE:
        # This case is problematic: sum_k [w_k / (x-x_k)] is close to zero.
        # Can happen if evaluation_point_tau is far from the interval of nodes, or due to cancellation.
        # Returning zeros, but could indicate numerical instability.
        # For well-behaved distinct nodes and evaluation_point_tau within or near their range, this should be rare.
        return lagrange_polynomial_values_at_evaluation_point # All lagrange_polynomial_values_at_evaluation_point remain 0.0

    lagrange_polynomial_values_at_evaluation_point = weighted_inverse_evaluation_point_differences / barycentric_sum_denominator
    return lagrange_polynomial_values_at_evaluation_point


def _compute_lagrange_derivative_coefficients_at_point(polynomial_definition_nodes, barycentric_weights, evaluation_point_tau):
    """
    Computes coefficients (derivatives) dL_j/dtau evaluated at evaluation_point_tau.
    Returns a vector [dL_0/dtau(evaluation_point_tau), dL_1/dtau(evaluation_point_tau), ..., dL_{N-1}/dtau(evaluation_point_tau)].

    Formulas based on Berrut & Trefethen (2004), "Barycentric Lagrange Interpolation".
    - Eq. 4.2 for evaluation at a node x_k (matched_node_index != -1).
    - Eq. 5.4 (adapted) for evaluation at t != x_k (matched_node_index == -1).
    """
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_derivative_coefficients = np.zeros(num_polynomial_definition_nodes) # Stores L'_j(evaluation_point_tau) for j=0..N-1
    matched_node_index = -1

    for current_node_index_for_match_check in range(num_polynomial_definition_nodes): # Check if evaluation_point_tau is one of the polynomial_definition_nodes
        if abs(evaluation_point_tau - polynomial_definition_nodes[current_node_index_for_match_check]) < ZERO_TOLERANCE:
            matched_node_index = current_node_index_for_match_check
            break
    
    if matched_node_index != -1: # Case 1: Derivative at one of the basis polynomial_definition_nodes (evaluation_point_tau = x_k where k=matched_node_index)
        # Off-diagonal elements: L'_j(x_k) = (w_j/w_k) / (x_k - x_j) for j != k
        # Diagonal element: L'_k(x_k) = - sum_{j!=k} L'_j(x_k)
        
        sum_for_diagonal_derivative_coefficient = 0.0
        for polynomial_index in range(num_polynomial_definition_nodes):
            if polynomial_index == matched_node_index:
                continue # Skip diagonal element for now
            
            node_difference_denominator = polynomial_definition_nodes[matched_node_index] - polynomial_definition_nodes[polynomial_index]
            if abs(node_difference_denominator) < ZERO_TOLERANCE or abs(barycentric_weights[matched_node_index]) < ZERO_TOLERANCE:
                # This implies coincident polynomial_definition_nodes or zero barycentric weight for x_k,
                # which should not happen for distinct polynomial_definition_nodes and valid weights.
                lagrange_derivative_coefficients[polynomial_index] = 0.0 # Fallback
            else:
                lagrange_derivative_coefficients[polynomial_index] = (barycentric_weights[polynomial_index] / barycentric_weights[matched_node_index]) / node_difference_denominator
            sum_for_diagonal_derivative_coefficient += lagrange_derivative_coefficients[polynomial_index]
        
        lagrange_derivative_coefficients[matched_node_index] = -sum_for_diagonal_derivative_coefficient
    return lagrange_derivative_coefficients

# ==============================================================================
# Public API Function (GPOPS-II Style)
# ==============================================================================

def compute_radau_collocation_components(num_collocation_nodes):
    """
    Computes components for Radau Pseudospectral Method based on LGR quadrature
    points, following GPOPS-II conventions as interpreted from gpops.txt.

    Args:
        num_collocation_nodes (int): The number of LGR collocation points (Nk in gpops.txt).

    Returns:
        RadauBasisComponents: An object containing:
            - differentiation_matrix: (Nk x (Nk+1)) differentiation matrix. Maps state values
                          at state_approximation_nodes to derivatives at collocation_nodes.
            - state_approximation_nodes: (Nk+1,) array of LGR basis points [-1, ..., +1].
            - collocation_nodes: (Nk,) array of LGR collocation points [-1, ...).
            - quadrature_weights: (Nk,) array of LGR weights corresponding
                                    to collocation_nodes for integration on [-1,1].
            - barycentric_weights_for_state_nodes: (Nk+1,) array of barycentric weights
                                           for state_approximation_nodes.
            - lagrange_basis_evaluation_at_local_tau_plus_one: (Nk+1,) array: L_j(+1) for basis polynomials L_j
                               defined over state_approximation_nodes, evaluated at +1.
                               Expected to be [0, ..., 0, 1] due to +1 being a basis point.
    """
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ValueError("Number of collocation points Nk must be an integer >= 1.")

    legendre_gauss_radau_components = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)
    state_approximation_nodes = legendre_gauss_radau_components.state_approximation_nodes
    collocation_nodes = legendre_gauss_radau_components.collocation_nodes
    quadrature_weights = legendre_gauss_radau_components.quadrature_weights

    num_state_approximation_nodes = len(state_approximation_nodes) 
    num_actual_collocation_nodes = len(collocation_nodes)

    if num_state_approximation_nodes != num_collocation_nodes + 1:
        raise ValueError(f"Mismatch in expected number of basis points. Expected {num_collocation_nodes + 1}, Got {num_state_approximation_nodes}")
    if num_actual_collocation_nodes != num_collocation_nodes:
        raise ValueError(f"Mismatch in expected number of collocation points. Expected {num_collocation_nodes}, Got {num_actual_collocation_nodes}")

    # Calculate barycentric weights for the basis points
    barycentric_weights_for_state_nodes = _compute_barycentric_weights(state_approximation_nodes)

    # --- Calculate Differentiation Matrix D ---
    # D has shape num_collocation_nodes x num_state_approximation_nodes
    differentiation_matrix = np.zeros((num_actual_collocation_nodes, num_state_approximation_nodes))

    for collocation_node_index in range(num_actual_collocation_nodes): # Row index -> corresponds to collocation_nodes[collocation_node_index]
        evaluation_point_at_collocation_node_tau = collocation_nodes[collocation_node_index]
        # differentiation_matrix[collocation_node_index, :] = [dL_0/dtau(eval_pt), dL_1/dtau(eval_pt), ..., dL_{num_state_approximation_nodes-1}/dtau(eval_pt)]
        # where L_j are basis polynomials defined over state_approximation_nodes.
        differentiation_matrix[collocation_node_index, :] = _compute_lagrange_derivative_coefficients_at_point(state_approximation_nodes, barycentric_weights_for_state_nodes, evaluation_point_at_collocation_node_tau)

    # --- Calculate Lagrange Polynomial values at tau = +1 (non-collocated end point for dynamics) ---
    # Since +1 is one of the state_approximation_nodes (expected to be the last one if sorted),
    # L_j(+1) will be 1 if state_approximation_nodes[j] == +1, and 0 otherwise.
    lagrange_basis_evaluation_at_local_tau_plus_one = np.zeros(num_state_approximation_nodes)
    # Find index of +1.0 in state_approximation_nodes. Basis_pts should be sorted.
    # Using np.isclose for robust floating point comparison.
    indices_of_tau_plus_one_candidates = np.where(np.isclose(state_approximation_nodes, 1.0, atol=ZERO_TOLERANCE))[0]
    
    if len(indices_of_tau_plus_one_candidates) == 1:
        lagrange_basis_evaluation_at_local_tau_plus_one[indices_of_tau_plus_one_candidates[0]] = 1.0
    elif len(indices_of_tau_plus_one_candidates) > 1:
        # Should not happen if state_approximation_nodes are distinct
        print("Warning: Multiple basis points found close to +1.0. Using the last one.")
        lagrange_basis_evaluation_at_local_tau_plus_one[indices_of_tau_plus_one_candidates[-1]] = 1.0
    else:
        # This should not happen if +1 is correctly included in state_approximation_nodes and ZERO_TOLERANCE is consistent.
        print("Warning: +1.0 not found precisely in state_approximation_nodes using np.isclose. Interpolating as fallback.")
        lagrange_basis_evaluation_at_local_tau_plus_one = _evaluate_lagrange_polynomial_at_point(state_approximation_nodes, barycentric_weights_for_state_nodes, 1.0)

    return RadauBasisComponents(
        differentiation_matrix=differentiation_matrix,
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
        barycentric_weights_for_state_nodes=barycentric_weights_for_state_nodes,
        lagrange_basis_evaluation_at_local_tau_plus_one=lagrange_basis_evaluation_at_local_tau_plus_one
    )