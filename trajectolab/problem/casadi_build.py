# trajectolab/problem/casadi_build.py
"""
Unified CasADi function building utilities with vectorized coordinate transformations.
OPTIMIZED: Added vectorized transformations to eliminate per-point computation inefficiency.
"""

import casadi as ca

from ..problem.state import PhaseDefinition
from ..tl_types import FloatArray


def build_unified_casadi_function_inputs(
    phase_def: PhaseDefinition,
    static_parameter_symbols: list[ca.MX] | None = None,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX, list[ca.MX]]:
    """
    Build unified CasADi function inputs to eliminate redundancy across dynamics/integrand functions.

    Returns:
        tuple: (states_vec, controls_vec, time, static_params_vec, function_inputs)
    """
    state_syms = phase_def.get_ordered_state_symbols()
    control_syms = phase_def.get_ordered_control_symbols()

    states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
    controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
    time = phase_def.sym_time if phase_def.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]

    # Create static parameters with correct dimensions
    num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0
    if num_static_params > 0:
        static_params_vec = ca.MX.sym("static_params", num_static_params, 1)
    else:
        static_params_vec = ca.MX.sym("static_params", 1, 1)  # Dummy parameter

    function_inputs = [states_vec, controls_vec, time, static_params_vec]

    return states_vec, controls_vec, time, static_params_vec, function_inputs


def build_static_parameter_substitution_map(
    static_parameter_symbols: list[ca.MX] | None,
    static_params_vec: ca.MX,
) -> dict[ca.MX, ca.MX]:
    """
    Build substitution map for static parameters to eliminate redundant mapping logic.
    """
    subs_map = {}
    num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0

    if static_parameter_symbols and num_static_params > 0:
        for i, param_sym in enumerate(static_parameter_symbols):
            if num_static_params == 1:
                subs_map[param_sym] = static_params_vec
            else:
                subs_map[param_sym] = static_params_vec[i]

    return subs_map


def build_unified_multiphase_symbol_inputs(multiphase_state) -> tuple[list[ca.MX], ca.MX]:
    """
    Build unified multiphase symbol inputs to eliminate redundant symbol-to-vector conversion.

    Returns:
        tuple: (phase_inputs, static_params_vec)
    """
    phase_inputs = []

    for phase_id in sorted(multiphase_state.phases.keys()):
        phase_def = multiphase_state.phases[phase_id]

        # Time symbols
        t0 = (
            phase_def.sym_time_initial
            if phase_def.sym_time_initial is not None
            else ca.MX.sym(f"t0_p{phase_id}", 1)
        )  # type: ignore[arg-type]
        tf = (
            phase_def.sym_time_final
            if phase_def.sym_time_final is not None
            else ca.MX.sym(f"tf_p{phase_id}", 1)
        )  # type: ignore[arg-type]

        # State vectors
        state_syms = phase_def.get_ordered_state_symbols()
        x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}_p{phase_id}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]
        xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}_p{phase_id}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]

        # Integral vector
        q_vec = (
            ca.vertcat(
                *[ca.MX.sym(f"q_{i}_p{phase_id}", 1) for i in range(phase_def.num_integrals)]
            )  # type: ignore[arg-type]
            if phase_def.num_integrals > 0
            else ca.MX.sym(f"q_p{phase_id}", 1)  # type: ignore[arg-type]
        )

        phase_inputs.extend([t0, tf, x0_vec, xf_vec, q_vec])

    # Static parameters
    static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
    s_vec = (
        ca.vertcat(*[ca.MX.sym(f"s_{i}", 1) for i in range(len(static_param_syms))])  # type: ignore[arg-type]
        if static_param_syms
        else ca.MX.sym("s", 1)  # type: ignore[arg-type]
    )

    phase_inputs.append(s_vec)

    return phase_inputs, s_vec


def build_unified_symbol_substitution_map(
    multiphase_state, phase_inputs: list[ca.MX], s_vec: ca.MX
) -> dict[ca.MX, ca.MX]:
    """
    Build unified symbol substitution map to eliminate redundant mapping logic.
    """
    phase_symbols_map = {}
    input_idx = 0

    for phase_id in sorted(multiphase_state.phases.keys()):
        phase_def = multiphase_state.phases[phase_id]

        # Extract inputs for this phase
        t0 = phase_inputs[input_idx]
        tf = phase_inputs[input_idx + 1]
        x0_vec = phase_inputs[input_idx + 2]
        xf_vec = phase_inputs[input_idx + 3]
        q_vec = phase_inputs[input_idx + 4]
        input_idx += 5

        # Get symbols
        state_syms = phase_def.get_ordered_state_symbols()
        state_initial_syms = phase_def.get_ordered_state_initial_symbols()
        state_final_syms = phase_def.get_ordered_state_final_symbols()

        # Build substitution map
        phase_symbols_map[t0] = t0
        phase_symbols_map[tf] = tf

        for i, (state_sym, initial_sym, final_sym) in enumerate(
            zip(state_syms, state_initial_syms, state_final_syms, strict=True)
        ):
            # Map state symbols to final values by default
            if len(state_syms) == 1:
                phase_symbols_map[state_sym] = xf_vec
                phase_symbols_map[initial_sym] = x0_vec
                phase_symbols_map[final_sym] = xf_vec
            else:
                phase_symbols_map[state_sym] = xf_vec[i]
                phase_symbols_map[initial_sym] = x0_vec[i]
                phase_symbols_map[final_sym] = xf_vec[i]

        # Map integral symbols
        for i, integral_sym in enumerate(phase_def.integral_symbols):
            if phase_def.num_integrals == 1:
                phase_symbols_map[integral_sym] = q_vec
            else:
                phase_symbols_map[integral_sym] = q_vec[i]

    # Add static parameter substitution
    static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
    for i, param_sym in enumerate(static_param_syms):
        if len(static_param_syms) == 1:
            phase_symbols_map[param_sym] = s_vec
        else:
            phase_symbols_map[param_sym] = s_vec[i]

    return phase_symbols_map


def transform_tau_to_physical_time(
    local_tau: float,
    global_normalized_mesh_nodes,
    mesh_interval_index: int,
    initial_time_variable: ca.MX,
    terminal_time_variable: ca.MX,
) -> ca.MX:
    """
    Unified time coordinate transformation to eliminate redundant calculations.

    Converts local tau coordinate to physical time using standard pseudospectral transformation.
    """
    # Calculate global segment properties
    global_segment_length = (
        global_normalized_mesh_nodes[mesh_interval_index + 1]
        - global_normalized_mesh_nodes[mesh_interval_index]
    )

    # Transform local tau to global tau
    global_tau = (
        global_segment_length / 2 * local_tau
        + (
            global_normalized_mesh_nodes[mesh_interval_index + 1]
            + global_normalized_mesh_nodes[mesh_interval_index]
        )
        / 2
    )

    # Transform global tau to physical time
    physical_time = (terminal_time_variable - initial_time_variable) / 2 * global_tau + (
        terminal_time_variable + initial_time_variable
    ) / 2

    return physical_time


# ============================================================================
# OPTIMIZED: Vectorized Coordinate Transformation Utilities
# ============================================================================


def vectorized_tau_to_global_tau(
    local_tau_array: FloatArray,
    global_normalized_mesh_nodes: FloatArray,
    mesh_interval_index: int,
) -> FloatArray:
    """
    OPTIMIZED: Vectorized local tau to global tau transformation.

    Eliminates per-point computation by applying transformation to entire array.
    """
    segment_start = global_normalized_mesh_nodes[mesh_interval_index]
    segment_end = global_normalized_mesh_nodes[mesh_interval_index + 1]
    global_segment_length = segment_end - segment_start

    # Vectorized transformation
    global_tau_array = (
        global_segment_length / 2 * local_tau_array + (segment_end + segment_start) / 2
    )

    return global_tau_array


def vectorized_global_tau_to_physical_time(
    global_tau_array: FloatArray,
    initial_time: float,
    terminal_time: float,
) -> FloatArray:
    """
    OPTIMIZED: Vectorized global tau to physical time transformation.

    Eliminates per-point computation by applying transformation to entire array.
    """
    # Vectorized transformation
    physical_time_array = (terminal_time - initial_time) / 2 * global_tau_array + (
        terminal_time + initial_time
    ) / 2

    return physical_time_array


def vectorized_tau_to_physical_time(
    local_tau_array: FloatArray,
    global_normalized_mesh_nodes: FloatArray,
    mesh_interval_index: int,
    initial_time: float,
    terminal_time: float,
) -> FloatArray:
    """
    OPTIMIZED: Complete vectorized coordinate transformation pipeline.

    Single function call replaces multiple per-point transformations.
    Provides O(1) per-interval complexity instead of O(n) per-point.
    """
    # Step 1: Local tau to global tau (vectorized)
    global_tau_array = vectorized_tau_to_global_tau(
        local_tau_array, global_normalized_mesh_nodes, mesh_interval_index
    )

    # Step 2: Global tau to physical time (vectorized)
    physical_time_array = vectorized_global_tau_to_physical_time(
        global_tau_array, initial_time, terminal_time
    )

    return physical_time_array


def build_coordinate_transformation_matrix(
    global_normalized_mesh_nodes: FloatArray,
    mesh_interval_index: int,
    initial_time: float,
    terminal_time: float,
) -> tuple[float, float, float]:
    """
    OPTIMIZED: Pre-compute transformation parameters for efficient coordinate conversion.

    Returns transformation parameters that can be applied to any tau array for this interval.

    Returns:
        tuple: (scale_factor, offset_factor, time_scale) for transformation:
               physical_time = time_scale * (scale_factor * tau + offset_factor) + time_offset
    """
    segment_start = global_normalized_mesh_nodes[mesh_interval_index]
    segment_end = global_normalized_mesh_nodes[mesh_interval_index + 1]
    global_segment_length = segment_end - segment_start

    # Transformation parameters
    scale_factor = global_segment_length / 2
    offset_factor = (segment_end + segment_start) / 2
    time_scale = (terminal_time - initial_time) / 2
    time_offset = (terminal_time + initial_time) / 2

    return scale_factor, offset_factor, time_scale, time_offset


def apply_precomputed_transformation(
    local_tau_array: FloatArray,
    scale_factor: float,
    offset_factor: float,
    time_scale: float,
    time_offset: float,
) -> FloatArray:
    """
    OPTIMIZED: Apply pre-computed transformation parameters to tau array.

    Fastest possible coordinate transformation using pre-computed parameters.
    """
    global_tau_array = scale_factor * local_tau_array + offset_factor
    physical_time_array = time_scale * global_tau_array + time_offset

    return physical_time_array
