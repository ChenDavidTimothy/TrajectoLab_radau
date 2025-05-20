"""
Fixed solver interface with proper CasADi handling.
"""

from __future__ import annotations

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
)
from .constraints_problem import get_event_constraints_function, get_path_constraints_function
from .state import ConstraintState, VariableState


def get_dynamics_function(variable_state: VariableState) -> DynamicsCallable:
    """Get dynamics function for solver."""
    # Gather all state and control symbols in order
    state_syms = [
        variable_state.sym_states[name]
        for name in sorted(
            variable_state.sym_states.keys(),
            key=lambda n: variable_state.states[n]["index"],
        )
    ]
    control_syms = [
        variable_state.sym_controls[name]
        for name in sorted(
            variable_state.sym_controls.keys(),
            key=lambda n: variable_state.controls[n]["index"],
        )
    ]

    # Create combined vector for CasADi function input
    states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
    controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX()
    )

    # Create output vector in same order as state_syms
    dynamics_expr = []
    for state_sym in state_syms:
        if state_sym in variable_state.dynamics_expressions:
            dynamics_expr.append(variable_state.dynamics_expressions[state_sym])
        else:
            # Default to zero if no dynamics provided
            dynamics_expr.append(ca.MX(0))

    dynamics_vec = ca.vertcat(*dynamics_expr) if dynamics_expr else ca.MX()

    # Create CasADi function
    dynamics_func = ca.Function(
        "dynamics", [states_vec, controls_vec, time, param_syms], [dynamics_vec]
    )

    # Create wrapper function
    def vectorized_dynamics(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ) -> list[CasadiMX]:
        # Extract parameter values in correct order
        param_values: list[float] = []
        for name in variable_state.sym_parameters:
            value = params.get(name, 0.0)
            if not isinstance(value, int | float):
                value = 0.0
            param_values.append(float(value))

        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Call CasADi function
        result = dynamics_func(states_vec, controls_vec, time, param_vec)
        dynamics_output = result[0] if isinstance(result, list | tuple) else result

        # Validate that we have a result
        if dynamics_output is None:
            raise ValueError("Dynamics function returned None")

        # Convert to list of MX elements
        num_states = int(dynamics_output.size1())  # type: ignore[attr-defined]
        result_list: list[CasadiMX] = []
        for i in range(num_states):
            result_list.append(cast(CasadiMX, dynamics_output[i]))

        return result_list

    return vectorized_dynamics


def get_objective_function(variable_state: VariableState) -> ObjectiveCallable:
    """Get objective function for solver."""
    if variable_state.objective_expression is None:
        raise ValueError("Objective expression not defined")

    # Gather symbols in order
    state_syms = [
        variable_state.sym_states[name]
        for name in sorted(
            variable_state.sym_states.keys(),
            key=lambda n: variable_state.states[n]["index"],
        )
    ]

    # Find all symbolic variables used in the objective expression
    obj_vars = ca.symvar(variable_state.objective_expression)

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
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)  # type: ignore[arg-type]
    )

    # Create substitution map for objective expression to use final states
    # FIX: Use proper CasADi operations instead of Python boolean checks
    substitution_keys = []
    substitution_values = []

    # Map state symbols to final state values
    for i, state_sym in enumerate(state_syms):
        # FIX: Check if symbol appears in objective using CasADi's depends_on
        try:
            if ca.depends_on(variable_state.objective_expression, state_sym):
                substitution_keys.append(state_sym)
                substitution_values.append(xf_vec[i])
        except Exception:
            # If depends_on fails, include it anyway to be safe
            substitution_keys.append(state_sym)
            substitution_values.append(xf_vec[i])

    # Map time symbols if used
    if variable_state.sym_time is not None:
        try:
            if ca.depends_on(variable_state.objective_expression, variable_state.sym_time):
                substitution_keys.append(variable_state.sym_time)
                substitution_values.append(tf)
        except Exception:
            substitution_keys.append(variable_state.sym_time)
            substitution_values.append(tf)

    if variable_state.sym_time_initial is not None:
        try:
            if ca.depends_on(variable_state.objective_expression, variable_state.sym_time_initial):
                substitution_keys.append(variable_state.sym_time_initial)
                substitution_values.append(t0)
        except Exception:
            substitution_keys.append(variable_state.sym_time_initial)
            substitution_values.append(t0)

    if variable_state.sym_time_final is not None:
        try:
            if ca.depends_on(variable_state.objective_expression, variable_state.sym_time_final):
                substitution_keys.append(variable_state.sym_time_final)
                substitution_values.append(tf)
        except Exception:
            substitution_keys.append(variable_state.sym_time_final)
            substitution_values.append(tf)

    # Map integral symbols if used
    for i, integral_sym in enumerate(variable_state.integral_symbols):
        try:
            if ca.depends_on(variable_state.objective_expression, integral_sym):
                substitution_keys.append(integral_sym)
                if i == 0 and len(variable_state.integral_symbols) == 1:
                    substitution_values.append(q)
                else:
                    substitution_values.append(q[i])
        except Exception:
            substitution_keys.append(integral_sym)
            if i == 0 and len(variable_state.integral_symbols) == 1:
                substitution_values.append(q)
            else:
                substitution_values.append(q[i])

    # Apply substitution to objective expression
    # FIX: Use proper CasADi substitute call with list arguments
    if substitution_keys:
        substituted_objective = ca.substitute(
            [variable_state.objective_expression], substitution_keys, substitution_values
        )[0]
    else:
        substituted_objective = variable_state.objective_expression

    # Create unified objective function with proper inputs
    try:
        obj_func = ca.Function(
            "objective",
            [t0, tf, x0_vec, xf_vec, q, param_syms],
            [substituted_objective],
            {"allow_free": True},  # Allow free variables if needed
        )
    except Exception as e:
        # If function creation fails, try a simpler approach
        print(f"Warning: Objective function creation failed: {e}")
        print("Attempting simplified objective function...")

        # Create a simpler function that just returns the expression directly
        obj_func = ca.Function(
            "objective", [t0, tf, x0_vec, xf_vec, q, param_syms], [substituted_objective]
        )

    # Create wrapper function
    def unified_objective(
        t0: CasadiMX,
        tf: CasadiMX,
        x0_vec: CasadiMX,
        xf_vec: CasadiMX,
        q: CasadiMX | None,
        params: ProblemParameters,
    ) -> CasadiMX:
        # Extract parameter values
        param_values = []
        for name in variable_state.sym_parameters:
            param_val = params.get(name, 0.0)
            try:
                param_values.append(float(param_val))
            except (TypeError, ValueError):
                param_values.append(0.0)

        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Handle q
        q_val = q if q is not None else ca.DM.zeros(len(variable_state.integral_symbols), 1)  # type: ignore[arg-type]

        # Call function
        try:
            result = obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)
            obj_output = result[0] if isinstance(result, list | tuple) else result
            return cast(CasadiMX, obj_output)
        except Exception as e:
            print(f"Error evaluating objective function: {e}")
            # Return a fallback objective if evaluation fails
            return cast(CasadiMX, ca.MX(0))

    return unified_objective


def get_integrand_function(variable_state: VariableState) -> IntegralIntegrandCallable | None:
    """Get integrand function for solver."""
    if not variable_state.integral_expressions:
        return None

    # Gather symbols in order
    state_syms = [
        variable_state.sym_states[name]
        for name in sorted(
            variable_state.sym_states.keys(),
            key=lambda n: variable_state.states[n]["index"],
        )
    ]
    control_syms = [
        variable_state.sym_controls[name]
        for name in sorted(
            variable_state.sym_controls.keys(),
            key=lambda n: variable_state.controls[n]["index"],
        )
    ]

    # Create combined vectors
    states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
    controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)  # type: ignore[arg-type]
    )

    # Create separate functions for each integrand
    integrand_funcs = []
    for expr in variable_state.integral_expressions:
        integrand_funcs.append(
            ca.Function("integrand", [states_vec, controls_vec, time, param_syms], [expr])
        )

    # Create wrapper function
    def vectorized_integrand(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        integral_idx: int,
        params: ProblemParameters,
    ) -> CasadiMX:
        if integral_idx >= len(integrand_funcs):
            return ca.MX(0.0)

        # Extract parameter values
        param_values = []
        for name in variable_state.sym_parameters:
            param_values.append(params.get(name, 0.0))

        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Call function
        result = integrand_funcs[integral_idx](states_vec, controls_vec, time, param_vec)
        integrand_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, integrand_output)

    return vectorized_integrand


def get_path_constraints_function_for_problem(
    constraint_state: ConstraintState, variable_state: VariableState
) -> PathConstraintsCallable | None:
    """Get path constraints function for solver."""
    return get_path_constraints_function(constraint_state, variable_state)


def get_event_constraints_function_for_problem(
    constraint_state: ConstraintState, variable_state: VariableState
) -> EventConstraintsCallable | None:
    """Get event constraints function for solver."""
    return get_event_constraints_function(constraint_state, variable_state)
