# trajectolab/problem/solver_interface.py
"""
Interface conversion functions between multiphase problem definition and solver requirements.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

import casadi as ca

from ..tl_types import PhaseID
from ..utils.expression_cache import (
    create_cache_key_from_multiphase_state,
    create_cache_key_from_phase_state,
    get_global_expression_cache,
)
from .state import MultiPhaseVariableState, PhaseDefinition


def get_phase_dynamics_function(
    phase_def: PhaseDefinition, static_parameter_symbols: list[ca.MX] | None = None
) -> Callable[..., list[ca.MX]]:
    """Get dynamics function for a specific phase with expression caching and static parameter support."""
    dynamics_exprs = [str(expr) for expr in phase_def.dynamics_expressions.values()]
    # CRITICAL FIX: Include static parameter info in cache key
    param_info = f"_params_{len(static_parameter_symbols) if static_parameter_symbols else 0}"
    expr_hash = hashlib.sha256("".join(sorted(dynamics_exprs)).encode()).hexdigest()[:16]

    cache_key = create_cache_key_from_phase_state(phase_def, f"dynamics{param_info}", expr_hash)

    def build_dynamics_function() -> ca.Function:
        """Build CasADi dynamics function for phase - expensive operation."""
        state_syms = phase_def.get_ordered_state_symbols()
        control_syms = phase_def.get_ordered_control_symbols()

        states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
        controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
        time = phase_def.sym_time if phase_def.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]

        # CRITICAL FIX: Create static parameters with correct dimensions
        num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0
        if num_static_params > 0:
            static_params_vec = ca.MX.sym("static_params", num_static_params, 1)
        else:
            static_params_vec = ca.MX.sym(
                "static_params", 1, 1
            )  # Dummy parameter for consistent signature

        # CRITICAL FIX: Build substitution map for static parameters
        subs_map = {}
        if static_parameter_symbols and num_static_params > 0:
            for i, param_sym in enumerate(static_parameter_symbols):
                if num_static_params == 1:
                    subs_map[param_sym] = static_params_vec
                else:
                    subs_map[param_sym] = static_params_vec[i]

        dynamics_expr = []
        for state_sym in state_syms:
            if state_sym in phase_def.dynamics_expressions:
                expr = phase_def.dynamics_expressions[state_sym]
                casadi_expr = ca.MX(expr) if not isinstance(expr, ca.MX) else expr

                # CRITICAL FIX: Apply substitution if we have static parameters
                if subs_map:
                    casadi_expr = ca.substitute(
                        [casadi_expr], list(subs_map.keys()), list(subs_map.values())
                    )[0]

                dynamics_expr.append(casadi_expr)
            else:
                dynamics_expr.append(ca.MX(0))

        dynamics_vec = ca.vertcat(*dynamics_expr) if dynamics_expr else ca.MX()

        # CRITICAL FIX: Include static parameters in function inputs
        function_inputs = [states_vec, controls_vec, time, static_params_vec]

        return ca.Function(f"dynamics_p{phase_def.phase_id}", function_inputs, [dynamics_vec])

    dynamics_func = get_global_expression_cache().get_dynamics_function(
        cache_key, build_dynamics_function
    )

    def vectorized_dynamics(
        states_vec: ca.MX,
        controls_vec: ca.MX,
        time: ca.MX,
        static_parameters_vec: ca.MX | None = None,  # CRITICAL FIX: Add parameter support
    ) -> list[ca.MX]:
        # CRITICAL FIX: Handle static parameters in function call
        num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0

        if static_parameters_vec is None or num_static_params == 0:
            # Create dummy parameters for consistent function signature
            static_params_input = ca.DM.zeros(max(1, num_static_params), 1)
        else:
            static_params_input = static_parameters_vec

        result = dynamics_func(states_vec, controls_vec, time, static_params_input)
        dynamics_output = result[0] if isinstance(result, list | tuple) else result

        if dynamics_output is None:
            raise ValueError(f"Phase {phase_def.phase_id} dynamics function returned None")

        num_states = int(dynamics_output.size1())
        result_list: list[ca.MX] = []
        for i in range(num_states):
            result_list.append(cast(ca.MX, dynamics_output[i]))

        return result_list

    return vectorized_dynamics


def get_multiphase_objective_function(
    multiphase_state: MultiPhaseVariableState,
) -> Callable[..., ca.MX]:
    """Get multiphase objective function with expression caching."""
    if multiphase_state.objective_expression is None:
        raise ValueError("Multiphase objective expression not defined")

    obj_hash = hashlib.sha256(str(multiphase_state.objective_expression).encode()).hexdigest()[:16]
    cache_key = create_cache_key_from_multiphase_state(multiphase_state, "objective", obj_hash)

    def build_objective_function() -> ca.Function:
        """Build CasADi multiphase objective function - expensive operation."""
        # Collect all phase endpoint symbols
        phase_inputs = []
        phase_symbols_map = {}

        for phase_id in sorted(multiphase_state.phases.keys()):
            phase_def = multiphase_state.phases[phase_id]

            # Time symbols
            t0 = (
                phase_def.sym_time_initial
                if phase_def.sym_time_initial is not None
                else ca.MX.sym(f"t0_p{phase_id}", 1)  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
            tf = (
                phase_def.sym_time_final
                if phase_def.sym_time_final is not None
                else ca.MX.sym(f"tf_p{phase_id}", 1)  # type: ignore[arg-type]
            )  # type: ignore[arg-type]

            # State vectors
            state_syms = phase_def.get_ordered_state_symbols()
            state_initial_syms = phase_def.get_ordered_state_initial_symbols()
            state_final_syms = phase_def.get_ordered_state_final_symbols()

            x0_vec = ca.vertcat(
                *[ca.MX.sym(f"x0_{i}_p{phase_id}", 1) for i in range(len(state_syms))]  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
            xf_vec = ca.vertcat(
                *[ca.MX.sym(f"xf_{i}_p{phase_id}", 1) for i in range(len(state_syms))]  # type: ignore[arg-type]
            )  # type: ignore[arg-type]

            # Integral vector
            q_vec = (
                ca.vertcat(
                    *[ca.MX.sym(f"q_{i}_p{phase_id}", 1) for i in range(phase_def.num_integrals)]  # type: ignore[arg-type]
                )
                if phase_def.num_integrals > 0
                else ca.MX.sym(f"q_p{phase_id}", 1)  # type: ignore[arg-type]
            )  # type: ignore[arg-type]

            phase_inputs.extend([t0, tf, x0_vec, xf_vec, q_vec])

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

        # Static parameters
        static_param_syms = multiphase_state.static_parameters.get_ordered_parameter_symbols()
        s_vec = (
            ca.vertcat(*[ca.MX.sym(f"s_{i}", 1) for i in range(len(static_param_syms))])  # type: ignore[arg-type]
            if static_param_syms
            else ca.MX.sym("s", 1)  # type: ignore[arg-type]
        )  # type: ignore[arg-type]

        # CRITICAL FIX: Add static parameter substitution
        for i, param_sym in enumerate(static_param_syms):
            if len(static_param_syms) == 1:
                phase_symbols_map[param_sym] = s_vec
            else:
                phase_symbols_map[param_sym] = s_vec[i]

        phase_inputs.append(s_vec)

        # CRITICAL FIX: Substitute in objective expression
        objective_expr = multiphase_state.objective_expression
        if phase_symbols_map:
            objective_expr = ca.substitute(
                [objective_expr], list(phase_symbols_map.keys()), list(phase_symbols_map.values())
            )[0]

        return ca.Function("multiphase_objective", phase_inputs, [objective_expr])

    obj_func = get_global_expression_cache().get_objective_function(
        cache_key, build_objective_function
    )

    def unified_multiphase_objective(
        phase_endpoint_data: dict[PhaseID, dict[str, ca.MX]],
        static_parameters_vec: ca.MX | None,
    ) -> ca.MX:
        """Evaluate multiphase objective with phase endpoint data."""
        inputs = []

        # In unified_multiphase_objective function:
        for phase_id in sorted(multiphase_state.phases.keys()):
            if phase_id in phase_endpoint_data:
                data = phase_endpoint_data[phase_id]
                phase_def = multiphase_state.phases[phase_id]  # Get phase definition

                q_val = (
                    data["q"]
                    if data["q"] is not None
                    else ca.DM.zeros(max(1, phase_def.num_integrals), 1)
                )

                inputs.extend(
                    [
                        data["t0"],
                        data["tf"],
                        data["x0"],
                        data["xf"],
                        q_val,  # Use the corrected value
                    ]
                )
            else:
                # Use default zeros for missing phases
                phase_def = multiphase_state.phases[phase_id]
                num_states = len(phase_def.state_info)
                inputs.extend(
                    [
                        ca.DM.zeros(1, 1),  # t0
                        ca.DM.zeros(1, 1),  # tf
                        ca.DM.zeros(num_states, 1),  # x0
                        ca.DM.zeros(num_states, 1),  # xf
                        ca.DM.zeros(max(1, phase_def.num_integrals), 1),  # q
                    ]
                )

        # Add static parameters
        if static_parameters_vec is not None:
            inputs.append(static_parameters_vec)
        else:
            num_params = multiphase_state.static_parameters.get_parameter_count()
            inputs.append(ca.DM.zeros(max(1, num_params), 1))  # type: ignore[arg-type]

        result = obj_func(*inputs)
        obj_output = result[0] if isinstance(result, list | tuple) else result
        return cast(ca.MX, obj_output)

    return unified_multiphase_objective


def get_phase_integrand_function(
    phase_def: PhaseDefinition, static_parameter_symbols: list[ca.MX] | None = None
) -> Callable[..., ca.MX] | None:
    """Get integrand function for a specific phase with expression caching and static parameter support."""
    if not phase_def.integral_expressions:
        return None

    integrand_exprs = [str(expr) for expr in phase_def.integral_expressions]
    param_info = f"_params_{len(static_parameter_symbols) if static_parameter_symbols else 0}"
    expr_hash = hashlib.sha256("".join(integrand_exprs)).hexdigest()[:16]
    cache_key = create_cache_key_from_phase_state(phase_def, f"integrand{param_info}", expr_hash)

    def build_integrand_functions() -> list[ca.Function]:
        """Build CasADi integrand functions for phase - expensive operation."""
        state_syms = phase_def.get_ordered_state_symbols()
        control_syms = phase_def.get_ordered_control_symbols()

        states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
        controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
        time = phase_def.sym_time if phase_def.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]

        # CRITICAL FIX: Include static parameters as inputs for integrand functions
        num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0
        if num_static_params > 0:
            static_params_vec = ca.MX.sym("static_params", num_static_params, 1)
        else:
            static_params_vec = ca.MX.sym("static_params", 1, 1)

        # CRITICAL FIX: Build substitution map for static parameters
        subs_map = {}
        if static_parameter_symbols and num_static_params > 0:
            for i, param_sym in enumerate(static_parameter_symbols):
                if num_static_params == 1:
                    subs_map[param_sym] = static_params_vec
                else:
                    subs_map[param_sym] = static_params_vec[i]

        integrand_funcs = []
        for i, expr in enumerate(phase_def.integral_expressions):
            # CRITICAL FIX: Apply substitution if we have static parameters
            processed_expr = expr
            if subs_map:
                processed_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]

            # CRITICAL FIX: Include static parameters in integrand function inputs
            function_inputs = [states_vec, controls_vec, time, static_params_vec]
            integrand_funcs.append(
                ca.Function(
                    f"integrand_{i}_p{phase_def.phase_id}", function_inputs, [processed_expr]
                )
            )

        return integrand_funcs

    integrand_funcs = get_global_expression_cache().get_integrand_functions(
        cache_key, build_integrand_functions
    )

    def vectorized_integrand(
        states_vec: ca.MX,
        controls_vec: ca.MX,
        time: ca.MX,
        integral_idx: int,
        static_parameters_vec: ca.MX | None = None,  # CRITICAL FIX: Add parameter support
    ) -> ca.MX:
        if integral_idx >= len(integrand_funcs):
            return ca.MX(0.0)

        # CRITICAL FIX: Handle static parameters in integrand function call
        num_static_params = len(static_parameter_symbols) if static_parameter_symbols else 0

        if static_parameters_vec is None or num_static_params == 0:
            # Create dummy parameters if none provided
            static_params_input = ca.DM.zeros(max(1, num_static_params), 1)
        else:
            static_params_input = static_parameters_vec

        result = integrand_funcs[integral_idx](states_vec, controls_vec, time, static_params_input)
        integrand_output = result[0] if isinstance(result, list | tuple) else result
        return cast(ca.MX, integrand_output)

    return vectorized_integrand
