import casadi as ca

from ..problem.state import PhaseDefinition


def _build_unified_casadi_function_inputs(
    phase_def: PhaseDefinition,
    static_parameter_symbols: list[ca.MX] | None = None,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX, list[ca.MX]]:
    state_syms = phase_def._get_ordered_state_symbols()
    control_syms = phase_def._get_ordered_control_symbols()

    states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
    controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
    time = phase_def.sym_time if phase_def.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]

    num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0
    if num_static_params > 0:
        static_params_vec = ca.MX.sym("static_params", num_static_params, 1)  # type: ignore[arg-type]
    else:
        static_params_vec = ca.MX.sym("static_params", 1, 1)  # type: ignore[arg-type]

    function_inputs = [states_vec, controls_vec, time, static_params_vec]

    return (
        ca.MX(states_vec),
        ca.MX(controls_vec),
        ca.MX(time),
        ca.MX(static_params_vec),
        function_inputs,
    )


def _build_static_parameter_substitution_map(
    static_parameter_symbols: list[ca.MX] | None,
    static_params_vec: ca.MX,
) -> dict[ca.MX, ca.MX]:
    subs_map = {}
    num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0

    if static_parameter_symbols and num_static_params > 0:
        for i, param_sym in enumerate(static_parameter_symbols):
            if num_static_params == 1:
                subs_map[param_sym] = static_params_vec
            else:
                subs_map[param_sym] = static_params_vec[i]

    return subs_map


def _build_unified_multiphase_symbol_inputs(multiphase_state) -> tuple[list[ca.MX], ca.MX]:
    phase_inputs = []

    for phase_id in sorted(multiphase_state.phases.keys()):
        phase_def = multiphase_state.phases[phase_id]

        t0 = (
            phase_def.sym_time_initial
            if phase_def.sym_time_initial is not None
            else ca.MX.sym(f"t0_p{phase_id}", 1)  # type: ignore[arg-type]
        )
        tf = (
            phase_def.sym_time_final
            if phase_def.sym_time_final is not None
            else ca.MX.sym(f"tf_p{phase_id}", 1)  # type: ignore[arg-type]
        )

        state_syms = phase_def._get_ordered_state_symbols()
        x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}_p{phase_id}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]
        xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}_p{phase_id}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]

        q_vec = (
            ca.vertcat(
                *[ca.MX.sym(f"q_{i}_p{phase_id}", 1) for i in range(phase_def.num_integrals)]  # type: ignore[arg-type]
            )
            if phase_def.num_integrals > 0
            else ca.MX.sym(f"q_p{phase_id}", 1)  # type: ignore[arg-type]
        )

        phase_inputs.extend([t0, tf, x0_vec, xf_vec, q_vec])

    static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
    s_vec = (
        ca.vertcat(*[ca.MX.sym(f"s_{i}", 1) for i in range(len(static_param_syms))])  # type: ignore[arg-type]
        if static_param_syms
        else ca.MX.sym("s", 1)  # type: ignore[arg-type]
    )

    phase_inputs.append(s_vec)

    return phase_inputs, ca.MX(s_vec)


def _build_unified_symbol_substitution_map(
    multiphase_state, phase_inputs: list[ca.MX], s_vec: ca.MX
) -> dict[ca.MX, ca.MX]:
    phase_symbols_map = {}
    input_idx = 0

    for phase_id in sorted(multiphase_state.phases.keys()):
        phase_def = multiphase_state.phases[phase_id]

        t0 = phase_inputs[input_idx]
        tf = phase_inputs[input_idx + 1]
        x0_vec = phase_inputs[input_idx + 2]
        xf_vec = phase_inputs[input_idx + 3]
        q_vec = phase_inputs[input_idx + 4]
        input_idx += 5

        state_syms = phase_def._get_ordered_state_symbols()
        state_initial_syms = phase_def.get_ordered_state_initial_symbols()
        state_final_syms = phase_def.get_ordered_state_final_symbols()

        phase_symbols_map[t0] = t0
        phase_symbols_map[tf] = tf

        for i, (state_sym, initial_sym, final_sym) in enumerate(
            zip(state_syms, state_initial_syms, state_final_syms, strict=True)
        ):
            if len(state_syms) == 1:
                phase_symbols_map[state_sym] = xf_vec
                phase_symbols_map[initial_sym] = x0_vec
                phase_symbols_map[final_sym] = xf_vec
            else:
                phase_symbols_map[state_sym] = xf_vec[i]
                phase_symbols_map[initial_sym] = x0_vec[i]
                phase_symbols_map[final_sym] = xf_vec[i]

        for i, integral_sym in enumerate(phase_def.integral_symbols):
            if phase_def.num_integrals == 1:
                phase_symbols_map[integral_sym] = q_vec
            else:
                phase_symbols_map[integral_sym] = q_vec[i]

    static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
    for i, param_sym in enumerate(static_param_syms):
        if len(static_param_syms) == 1:
            phase_symbols_map[param_sym] = s_vec
        else:
            phase_symbols_map[param_sym] = s_vec[i]

    return phase_symbols_map
