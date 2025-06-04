from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

import casadi as ca

from ..tl_types import PhaseID
from ..utils.expression_cache import (
    _create_cache_key_from_multiphase_state,
    _create_cache_key_from_phase_state,
    _get_global_expression_cache,
)
from .casadi_build import (
    _build_static_parameter_substitution_map,
    _build_unified_casadi_function_inputs,
    _build_unified_multiphase_symbol_inputs,
    _build_unified_symbol_substitution_map,
)
from .state import MultiPhaseVariableState, PhaseDefinition


def _get_phase_dynamics_function(
    phase_def: PhaseDefinition, static_parameter_symbols: list[ca.MX] | None = None
) -> Callable[..., ca.MX]:
    """Get phase dynamics function using optimized direct vector interface."""
    dynamics_exprs = [str(expr) for expr in phase_def.dynamics_expressions.values()]
    param_info = f"_params_{len(static_parameter_symbols) if static_parameter_symbols else 0}"
    expr_hash = hashlib.sha256("".join(sorted(dynamics_exprs)).encode()).hexdigest()[:16]

    cache_key = _create_cache_key_from_phase_state(phase_def, f"dynamics{param_info}", expr_hash)

    def _build_dynamics_function() -> ca.Function:
        """Build CasADi dynamics function using unified input builder."""
        states_vec, controls_vec, time, static_params_vec, function_inputs = (
            _build_unified_casadi_function_inputs(phase_def, static_parameter_symbols)
        )

        subs_map = _build_static_parameter_substitution_map(
            static_parameter_symbols, static_params_vec
        )

        state_syms = phase_def._get_ordered_state_symbols()
        dynamics_expr = []
        for state_sym in state_syms:
            if state_sym in phase_def.dynamics_expressions:
                expr = phase_def.dynamics_expressions[state_sym]
                casadi_expr = ca.MX(expr) if not isinstance(expr, ca.MX) else expr

                if subs_map:
                    casadi_expr = ca.substitute(
                        [casadi_expr], list(subs_map.keys()), list(subs_map.values())
                    )[0]

                dynamics_expr.append(casadi_expr)
            else:
                dynamics_expr.append(ca.MX(0))

        dynamics_vec = ca.vertcat(*dynamics_expr) if dynamics_expr else ca.MX()

        return ca.Function(f"dynamics_p{phase_def.phase_id}", function_inputs, [dynamics_vec])

    dynamics_func = _get_global_expression_cache().get_dynamics_function(
        cache_key, _build_dynamics_function
    )

    def _vectorized_dynamics(
        states_vec: ca.MX,
        controls_vec: ca.MX,
        time: ca.MX,
        static_parameters_vec: ca.MX | None = None,
    ) -> ca.MX:
        """Direct vector return eliminates list conversion inefficiency."""
        num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0

        if static_parameters_vec is None or num_static_params == 0:
            static_params_input = ca.DM(max(1, num_static_params), 1)
        else:
            static_params_input = static_parameters_vec

        result = dynamics_func(states_vec, controls_vec, time, static_params_input)
        dynamics_output = result[0] if isinstance(result, list | tuple) else result

        if dynamics_output is None:
            raise ValueError(f"Phase {phase_def.phase_id} dynamics function returned None")

        return cast(ca.MX, dynamics_output)

    return _vectorized_dynamics


def _get_multiphase_objective_function(
    multiphase_state: MultiPhaseVariableState,
) -> Callable[..., ca.MX]:
    """Get multiphase objective function using unified symbol mapping."""
    if multiphase_state.objective_expression is None:
        raise ValueError("Multiphase objective expression not defined")

    obj_hash = hashlib.sha256(str(multiphase_state.objective_expression).encode()).hexdigest()[:16]
    cache_key = _create_cache_key_from_multiphase_state(multiphase_state, "objective", obj_hash)

    def _build_objective_function() -> ca.Function:
        """Build CasADi multiphase objective function using unified builders."""
        phase_inputs, s_vec = _build_unified_multiphase_symbol_inputs(multiphase_state)

        phase_symbols_map = _build_unified_symbol_substitution_map(
            multiphase_state, phase_inputs, s_vec
        )

        objective_expr = multiphase_state.objective_expression
        if phase_symbols_map:
            objective_expr = ca.substitute(
                [objective_expr], list(phase_symbols_map.keys()), list(phase_symbols_map.values())
            )[0]

        return ca.Function("multiphase_objective", phase_inputs, [objective_expr])

    obj_func = _get_global_expression_cache()._get_objective_function(
        cache_key, _build_objective_function
    )

    def _unified_multiphase_objective(
        phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]], static_parameters_vec: ca.MX | None
    ) -> ca.MX:
        """Evaluate multiphase objective with phase endpoint data."""
        inputs = []

        for phase_id in sorted(multiphase_state.phases.keys()):
            if phase_id in phase_endpoint_data:
                data = phase_endpoint_data[phase_id]
                phase_def = multiphase_state.phases[phase_id]

                q_val = (
                    data["q"]
                    if data["q"] is not None
                    else ca.DM(max(1, phase_def.num_integrals), 1)
                )

                inputs.extend([data["t0"], data["tf"], data["x0"], data["xf"], q_val])
            else:
                phase_def = multiphase_state.phases[phase_id]
                num_states = len(phase_def.state_info)
                inputs.extend(
                    [
                        ca.DM(1, 1),  # t0
                        ca.DM(1, 1),  # tf
                        ca.DM(num_states, 1),  # x0
                        ca.DM(num_states, 1),  # xf
                        ca.DM(max(1, phase_def.num_integrals), 1),  # q
                    ]
                )

        if static_parameters_vec is not None:
            inputs.append(static_parameters_vec)
        else:
            num_params = multiphase_state.static_parameters.get_parameter_count()
            inputs.append(ca.DM.zeros(max(1, num_params), 1))  # type: ignore[arg-type]

        result = obj_func(*inputs)
        obj_output = result[0] if isinstance(result, list | tuple) else result
        return cast(ca.MX, obj_output)

    return _unified_multiphase_objective


def _get_phase_integrand_function(
    phase_def: PhaseDefinition, static_parameter_symbols: list[ca.MX] | None = None
) -> Callable[..., ca.MX] | None:
    """Get phase integrand function using unified CasADi function building."""
    if not phase_def.integral_expressions:
        return None

    integrand_exprs = [str(expr) for expr in phase_def.integral_expressions]
    param_info = f"_params_{len(static_parameter_symbols) if static_parameter_symbols else 0}"
    expr_hash = hashlib.sha256("".join(integrand_exprs).encode()).hexdigest()[:16]
    cache_key = _create_cache_key_from_phase_state(phase_def, f"integrand{param_info}", expr_hash)

    def _build_integrand_functions() -> list[ca.Function]:
        """Build CasADi integrand functions using unified input builder."""
        states_vec, controls_vec, time, static_params_vec, function_inputs = (
            _build_unified_casadi_function_inputs(phase_def, static_parameter_symbols)
        )

        subs_map = _build_static_parameter_substitution_map(
            static_parameter_symbols, static_params_vec
        )

        integrand_funcs = []
        for i, expr in enumerate(phase_def.integral_expressions):
            processed_expr = expr
            if subs_map:
                processed_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]

            integrand_funcs.append(
                ca.Function(
                    f"integrand_{i}_p{phase_def.phase_id}", function_inputs, [processed_expr]
                )
            )

        return integrand_funcs

    integrand_funcs = _get_global_expression_cache().get_integrand_functions(
        cache_key, _build_integrand_functions
    )

    def _vectorized_integrand(
        states_vec: ca.MX,
        controls_vec: ca.MX,
        time: ca.MX,
        integral_idx: int,
        static_parameters_vec: ca.MX | None = None,
    ) -> ca.MX:
        if integral_idx >= len(integrand_funcs):
            return ca.MX(0.0)

        num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0

        if static_parameters_vec is None or num_static_params == 0:
            static_params_input = ca.DM(max(1, num_static_params), 1)
        else:
            static_params_input = static_parameters_vec

        result = integrand_funcs[integral_idx](states_vec, controls_vec, time, static_params_input)
        integrand_output = result[0] if isinstance(result, list | tuple) else result
        return cast(ca.MX, integrand_output)

    return _vectorized_integrand
