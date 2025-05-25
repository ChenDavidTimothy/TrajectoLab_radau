"""
Interface conversion functions between problem definition and solver requirements.
"""

from __future__ import annotations

import hashlib
from typing import cast

import casadi as ca

from ..tl_types import (
    CasadiMX,
    DynamicsCallable,
    EventConstraintsCallable,
    IntegralIntegrandCallable,
    ObjectiveCallable,
    PathConstraintsCallable,
    ProblemParameters,
    SymExpr,
)
from ..utils.expression_cache import (
    create_cache_key_from_variable_state,
    get_global_expression_cache,
)
from .constraints_problem import get_event_constraints_function, get_path_constraints_function
from .state import ConstraintState, VariableState


def _convert_expression_symbols(expr: SymExpr, variable_state: VariableState) -> SymExpr:
    """
    Convert any StateVariableImpl or TimeVariableImpl objects in expressions to underlying symbols.

    This ensures that expressions stored in dynamics, objectives, etc. contain only pure CasADi symbols.
    """
    if not isinstance(expr, ca.MX):
        return expr

    # Get all symbols that might need conversion
    all_state_syms = variable_state.get_ordered_state_symbols()
    all_state_initial_syms = variable_state.get_ordered_state_initial_symbols()
    all_state_final_syms = variable_state.get_ordered_state_final_symbols()

    # For now, return expression as-is since the main issue is in storage, not conversion
    # The conversion should happen at the storage level in set_dynamics
    return expr


def get_dynamics_function(variable_state: VariableState) -> DynamicsCallable:
    """Get dynamics function for solver with expression caching and unified storage."""

    # Create expression hash for caching
    dynamics_exprs = [str(expr) for expr in variable_state.dynamics_expressions.values()]
    expr_hash = hashlib.sha256("".join(sorted(dynamics_exprs)).encode()).hexdigest()[:16]

    cache_key = create_cache_key_from_variable_state(variable_state, "dynamics", expr_hash)

    def build_dynamics_function() -> ca.Function:
        """Build CasADi dynamics function - expensive operation."""
        # Use unified storage - direct O(1) access
        state_syms = variable_state.get_ordered_state_symbols()
        control_syms = variable_state.get_ordered_control_symbols()

        # Create combined vectors
        states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
        controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
        time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
        param_syms = (
            ca.vertcat(*variable_state.parameters.keys())
            if variable_state.parameters
            else ca.MX.sym("p", 0)  # type: ignore[arg-type]
        )

        # Create output vector in same order as state_syms
        dynamics_expr = []
        for state_sym in state_syms:
            if state_sym in variable_state.dynamics_expressions:
                expr = variable_state.dynamics_expressions[state_sym]
                # Ensure we have a pure MX expression
                if isinstance(expr, ca.MX):
                    dynamics_expr.append(expr)
                else:
                    # Convert to MX if it's not already
                    dynamics_expr.append(ca.MX(expr))
            else:
                dynamics_expr.append(ca.MX(0))

        dynamics_vec = ca.vertcat(*dynamics_expr) if dynamics_expr else ca.MX()

        # Create CasADi function
        return ca.Function("dynamics", [states_vec, controls_vec, time, param_syms], [dynamics_vec])

    # Get cached or build function
    dynamics_func = get_global_expression_cache().get_dynamics_function(
        cache_key, build_dynamics_function
    )

    def vectorized_dynamics(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ) -> list[CasadiMX]:
        # Extract parameter values in correct order
        param_values: list[float] = []
        for name in variable_state.parameters:
            value = params.get(name, 0.0)
            try:
                param_values.append(float(value))
            except (TypeError, ValueError):
                param_values.append(0.0)

        param_vec = ca.DM(param_values) if param_values else ca.DM()
        result = dynamics_func(states_vec, controls_vec, time, param_vec)
        dynamics_output = result[0] if isinstance(result, list | tuple) else result

        if dynamics_output is None:
            raise ValueError("Dynamics function returned None")

        # Convert to list of MX elements
        num_states = int(dynamics_output.size1())
        result_list: list[CasadiMX] = []
        for i in range(num_states):
            result_list.append(cast(CasadiMX, dynamics_output[i]))

        return result_list

    return vectorized_dynamics


def get_objective_function(variable_state: VariableState) -> ObjectiveCallable:
    """Get objective function for solver with expression caching and unified storage."""
    if variable_state.objective_expression is None:
        raise ValueError("Objective expression not defined")

    # Create expression hash for caching
    obj_hash = hashlib.sha256(str(variable_state.objective_expression).encode()).hexdigest()[:16]

    cache_key = create_cache_key_from_variable_state(variable_state, "objective", obj_hash)

    def build_objective_function() -> ca.Function:
        """Build CasADi objective function - expensive operation."""
        # Use unified storage - direct O(1) access
        state_syms = variable_state.get_ordered_state_symbols()
        state_initial_syms = variable_state.get_ordered_state_initial_symbols()
        state_final_syms = variable_state.get_ordered_state_final_symbols()

        # Create inputs for unified objective function
        t0 = (
            variable_state.sym_time_initial
            if variable_state.sym_time_initial is not None
            else ca.MX.sym("t0", 1)  # type: ignore[arg-type]
        )
        tf = (
            variable_state.sym_time_final
            if variable_state.sym_time_final is not None
            else ca.MX.sym("tf", 1)  # type: ignore[arg-type]
        )
        x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]
        xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]
        q = (
            ca.vertcat(*variable_state.integral_symbols)
            if variable_state.integral_symbols
            else ca.MX.sym("q", 1)  # type: ignore[arg-type]
        )
        param_syms = (
            ca.vertcat(*variable_state.parameters.keys())
            if variable_state.parameters
            else ca.MX.sym("p", 0)  # type: ignore[arg-type]
        )

        # Substitute symbols in objective expression
        objective_expr = variable_state.objective_expression

        # Create substitution maps for all symbols
        old_syms = []
        new_syms = []

        # Map current state symbols to final state values (default behavior)
        for i, sym in enumerate(state_syms):
            old_syms.append(sym)
            if len(state_syms) == 1:
                new_syms.append(xf_vec)
            else:
                new_syms.append(xf_vec[i])

        # Map state initial symbols to initial state values
        for i, sym in enumerate(state_initial_syms):
            old_syms.append(sym)
            if len(state_syms) == 1:
                new_syms.append(x0_vec)
            else:
                new_syms.append(x0_vec[i])

        # Map state final symbols to final state values
        for i, sym in enumerate(state_final_syms):
            old_syms.append(sym)
            if len(state_syms) == 1:
                new_syms.append(xf_vec)
            else:
                new_syms.append(xf_vec[i])

        # Apply substitutions
        if old_syms:
            objective_expr = ca.substitute([objective_expr], old_syms, new_syms)[0]

        # Create unified objective function
        return ca.Function(
            "objective",
            [t0, tf, x0_vec, xf_vec, q, param_syms],
            [objective_expr],
        )

    # Get cached or build function
    obj_func = get_global_expression_cache().get_objective_function(
        cache_key, build_objective_function
    )

    def unified_objective(
        t0: CasadiMX,
        tf: CasadiMX,
        x0_vec: CasadiMX,
        xf_vec: CasadiMX,
        q: CasadiMX | None,
        params: ProblemParameters,
    ) -> CasadiMX:
        param_values = []
        for name in variable_state.parameters:
            param_val = params.get(name, 0.0)
            try:
                param_values.append(float(param_val))
            except (TypeError, ValueError):
                param_values.append(0.0)

        param_vec = ca.DM(param_values) if param_values else ca.DM()
        q_val = q if q is not None else ca.DM.zeros(len(variable_state.integral_symbols), 1)  # type: ignore[arg-type]

        result = obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)
        obj_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, obj_output)

    return unified_objective


def get_integrand_function(variable_state: VariableState) -> IntegralIntegrandCallable | None:
    """Get integrand function for solver with expression caching and unified storage."""
    if not variable_state.integral_expressions:
        return None

    # Create expression hash for caching
    integrand_exprs = [str(expr) for expr in variable_state.integral_expressions]
    expr_hash = hashlib.sha256("".join(integrand_exprs).encode()).hexdigest()[:16]

    cache_key = create_cache_key_from_variable_state(variable_state, "integrand", expr_hash)

    def build_integrand_functions() -> list[ca.Function]:
        """Build CasADi integrand functions - expensive operation."""
        # Use unified storage - direct O(1) access
        state_syms = variable_state.get_ordered_state_symbols()
        control_syms = variable_state.get_ordered_control_symbols()

        # Create combined vectors
        states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
        controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
        time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
        param_syms = (
            ca.vertcat(*variable_state.parameters.keys())
            if variable_state.parameters
            else ca.MX.sym("p", 0)  # type: ignore[arg-type]
        )

        # Create separate functions for each integrand
        integrand_funcs = []
        for expr in variable_state.integral_expressions:
            integrand_funcs.append(
                ca.Function("integrand", [states_vec, controls_vec, time, param_syms], [expr])
            )

        return integrand_funcs

    # Get cached or build functions
    integrand_funcs = get_global_expression_cache().get_integrand_functions(
        cache_key, build_integrand_functions
    )

    def vectorized_integrand(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        integral_idx: int,
        params: ProblemParameters,
    ) -> CasadiMX:
        if integral_idx >= len(integrand_funcs):
            return ca.MX(0.0)

        param_values = []
        for name in variable_state.parameters:
            param_values.append(params.get(name, 0.0))

        param_vec = ca.DM(param_values) if param_values else ca.DM()
        result = integrand_funcs[integral_idx](states_vec, controls_vec, time, param_vec)
        integrand_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, integrand_output)

    return vectorized_integrand


def get_path_constraints_function_for_problem(
    constraint_state: ConstraintState, variable_state: VariableState
) -> PathConstraintsCallable | None:
    """Get path constraints function for solver using unified storage."""
    return get_path_constraints_function(constraint_state, variable_state)


def get_event_constraints_function_for_problem(
    constraint_state: ConstraintState, variable_state: VariableState
) -> EventConstraintsCallable | None:
    """Get event constraints function for solver using unified storage."""
    return get_event_constraints_function(constraint_state, variable_state)
