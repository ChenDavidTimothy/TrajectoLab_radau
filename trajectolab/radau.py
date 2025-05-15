import numpy as np
from dataclasses import dataclass
from scipy.special import roots_jacobi

# Tolerance for floating point comparisons
ZERO_TOLERANCE = 1e-12

@dataclass
class RadauBasisComponents:
    state_approximation_nodes: np.ndarray = None
    collocation_nodes: np.ndarray = None
    quadrature_weights: np.ndarray = None
    differentiation_matrix: np.ndarray = None
    barycentric_weights_for_state_nodes: np.ndarray = None
    lagrange_at_tau_plus_one: np.ndarray = None

@dataclass
class RadauNodesAndWeights:
    state_approximation_nodes: np.ndarray = None
    collocation_nodes: np.ndarray = None
    quadrature_weights: np.ndarray = None

def compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes):
    if not isinstance(num_collocation_nodes, int) or num_collocation_nodes < 1:
        raise ValueError("Number of collocation points must be an integer >= 1.")

    # Initialize with the fixed left endpoint
    collocation_nodes = np.array([-1.0])
    
    if num_collocation_nodes == 1:
        # Only one collocation point case
        quadrature_weights = np.array([2.0])
    else:
        # Multi-point case: compute interior roots and weights
        num_interior_roots = num_collocation_nodes - 1
        interior_roots, jacobi_weights, _ = roots_jacobi(num_interior_roots, 0, 1, mu=True)
        
        # Adjust Jacobi weights for standard Legendre measure
        interior_weights = jacobi_weights / (1.0 + interior_roots)
        
        # Weight for the -1 point
        left_endpoint_weight = 2.0 / (num_collocation_nodes**2)
        
        # Combine nodes and weights
        collocation_nodes = np.concatenate([collocation_nodes, interior_roots])
        quadrature_weights = np.concatenate([np.array([left_endpoint_weight]), interior_weights])
    
    # Create state approximation nodes (collocation nodes + right endpoint)
    state_approximation_nodes = np.concatenate([collocation_nodes, np.array([1.0])])
    
    return RadauNodesAndWeights(
        state_approximation_nodes=state_approximation_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights
    )

def compute_barycentric_weights(nodes):
    num_nodes = len(nodes)
    barycentric_weights = np.ones(num_nodes)
    nodes_array = np.asarray(nodes) 
    
    for j in range(num_nodes):
        node_differences = nodes_array[j] - np.delete(nodes_array, j)
        # Ensure no zero node_differences if nodes were extremely close 
        node_differences[np.abs(node_differences) < ZERO_TOLERANCE * 1e-1] = np.sign(node_differences[np.abs(node_differences) < ZERO_TOLERANCE * 1e-1]) * ZERO_TOLERANCE * 1e-1 \
                                      if np.any(node_differences[np.abs(node_differences) < ZERO_TOLERANCE * 1e-1] !=0) else ZERO_TOLERANCE*1e-1
        barycentric_weights[j] = 1.0 / np.prod(node_differences)
    
    return barycentric_weights

def evaluate_lagrange_polynomial_at_point(polynomial_definition_nodes, barycentric_weights, evaluation_point_tau):
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_polynomial_values_at_evaluation_point = np.zeros(num_polynomial_definition_nodes)

    # Check if evaluation_point_tau is one of the nodes (within tolerance)
    for j in range(num_polynomial_definition_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[j]) < ZERO_TOLERANCE:
            lagrange_polynomial_values_at_evaluation_point[j] = 1.0
            return lagrange_polynomial_values_at_evaluation_point

    barycentric_sum_denominator = 0.0
    weighted_inverse_evaluation_point_differences = np.zeros(num_polynomial_definition_nodes)

    for j in range(num_polynomial_definition_nodes):
        evaluation_point_difference_from_node = evaluation_point_tau - polynomial_definition_nodes[j]
        if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE:
            evaluation_point_difference_from_node = np.sign(evaluation_point_difference_from_node) * ZERO_TOLERANCE if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE else evaluation_point_difference_from_node
            if abs(evaluation_point_difference_from_node) < ZERO_TOLERANCE:
                 weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / (np.sign(polynomial_definition_nodes[j]) *1e-100 if polynomial_definition_nodes[j]!=0 else 1e-100)
            else:
                 weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / evaluation_point_difference_from_node
        else:
            weighted_inverse_evaluation_point_differences[j] = barycentric_weights[j] / evaluation_point_difference_from_node
        barycentric_sum_denominator += weighted_inverse_evaluation_point_differences[j]

    if abs(barycentric_sum_denominator) < ZERO_TOLERANCE:
        return lagrange_polynomial_values_at_evaluation_point

    lagrange_polynomial_values_at_evaluation_point = weighted_inverse_evaluation_point_differences / barycentric_sum_denominator
    return lagrange_polynomial_values_at_evaluation_point

def compute_lagrange_derivative_coefficients_at_point(polynomial_definition_nodes, barycentric_weights, evaluation_point_tau):
    num_polynomial_definition_nodes = len(polynomial_definition_nodes)
    lagrange_derivative_coefficients = np.zeros(num_polynomial_definition_nodes)
    matched_node_index = -1

    for current_node_index_for_match_check in range(num_polynomial_definition_nodes):
        if abs(evaluation_point_tau - polynomial_definition_nodes[current_node_index_for_match_check]) < ZERO_TOLERANCE:
            matched_node_index = current_node_index_for_match_check
            break
    
    if matched_node_index != -1:
        sum_for_diagonal_derivative_coefficient = 0.0
        for polynomial_index in range(num_polynomial_definition_nodes):
            if polynomial_index == matched_node_index:
                continue
            
            node_difference_denominator = polynomial_definition_nodes[matched_node_index] - polynomial_definition_nodes[polynomial_index]
            if abs(node_difference_denominator) < ZERO_TOLERANCE or abs(barycentric_weights[matched_node_index]) < ZERO_TOLERANCE:
                lagrange_derivative_coefficients[polynomial_index] = 0.0
            else:
                lagrange_derivative_coefficients[polynomial_index] = (barycentric_weights[polynomial_index] / barycentric_weights[matched_node_index]) / node_difference_denominator
            sum_for_diagonal_derivative_coefficient += lagrange_derivative_coefficients[polynomial_index]
        
        lagrange_derivative_coefficients[matched_node_index] = -sum_for_diagonal_derivative_coefficient
    return lagrange_derivative_coefficients

def compute_radau_collocation_components(num_collocation_nodes):
    lgr_components = compute_legendre_gauss_radau_nodes_and_weights(num_collocation_nodes)
    state_nodes = lgr_components.state_approximation_nodes
    collocation_nodes = lgr_components.collocation_nodes
    quadrature_weights = lgr_components.quadrature_weights

    num_state_nodes = len(state_nodes) 
    num_actual_collocation_nodes = len(collocation_nodes)

    # Validate dimensions
    if num_state_nodes != num_collocation_nodes + 1:
        raise ValueError(f"Mismatch in expected number of basis points. Expected {num_collocation_nodes + 1}, Got {num_state_nodes}")
    if num_actual_collocation_nodes != num_collocation_nodes:
        raise ValueError(f"Mismatch in expected number of collocation points. Expected {num_collocation_nodes}, Got {num_actual_collocation_nodes}")

    # Calculate barycentric weights for the basis points
    bary_weights = compute_barycentric_weights(state_nodes)

    # Calculate Differentiation Matrix D
    diff_matrix = np.zeros((num_actual_collocation_nodes, num_state_nodes))

    for collocation_node_index in range(num_actual_collocation_nodes):
        evaluation_point_at_collocation_node_tau = collocation_nodes[collocation_node_index]
        diff_matrix[collocation_node_index, :] = compute_lagrange_derivative_coefficients_at_point(
            state_nodes, bary_weights, evaluation_point_at_collocation_node_tau
        )

    # Calculate Lagrange Polynomial values at tau = +1
    lagrange_at_tau_plus_one = np.zeros(num_state_nodes)
    
    # Find index of +1.0 in state_nodes using np.isclose for robust floating point comparison
    tau_plus_one_indices = np.where(np.isclose(state_nodes, 1.0, atol=ZERO_TOLERANCE))[0]
    
    if len(tau_plus_one_indices) == 1:
        lagrange_at_tau_plus_one[tau_plus_one_indices[0]] = 1.0
    elif len(tau_plus_one_indices) > 1:
        # Should not happen if state_nodes are distinct
        print("Warning: Multiple basis points found close to +1.0. Using the last one.")
        lagrange_at_tau_plus_one[tau_plus_one_indices[-1]] = 1.0
    else:
        # Fallback
        print("Warning: +1.0 not found precisely in state_nodes. Interpolating as fallback.")
        lagrange_at_tau_plus_one = evaluate_lagrange_polynomial_at_point(state_nodes, bary_weights, 1.0)

    return RadauBasisComponents(
        differentiation_matrix=diff_matrix,
        state_approximation_nodes=state_nodes,
        collocation_nodes=collocation_nodes,
        quadrature_weights=quadrature_weights,
        barycentric_weights_for_state_nodes=bary_weights,
        lagrange_at_tau_plus_one=lagrange_at_tau_plus_one
    ) 