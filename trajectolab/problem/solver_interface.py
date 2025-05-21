"""
Solver interface conversion functions for optimal control problems.
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
    """Get dynamics function for solver with automatic scaling handling."""
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

    # State and control names in order (for scaling reference)
    sorted(variable_state.sym_states.keys(), key=lambda n: variable_state.states[n]["index"])
    sorted(variable_state.sym_controls.keys(), key=lambda n: variable_state.controls[n]["index"])

    # Create combined vector for CasADi function input
    states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
    controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)  # type: ignore[arg-type]
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
            try:
                param_values.append(float(value))
            except (TypeError, ValueError):
                param_values.append(0.0)

        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Standard call
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
    """Get objective function for solver with automatic state substitution."""
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

    # Create inputs for unified objective function
    t0 = (
        variable_state.sym_time_initial
        if variable_state.sym_time_initial is not None
        else ca.MX.sym("t0", 1)
    )
    tf = (
        variable_state.sym_time_final
        if variable_state.sym_time_final is not None
        else ca.MX.sym("tf", 1)
    )
    x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}", 1) for i in range(len(state_syms))])
    xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}", 1) for i in range(len(state_syms))])
    q = (
        ca.vertcat(*variable_state.integral_symbols)
        if variable_state.integral_symbols
        else ca.MX.sym("q", 1)
    )
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)
    )

    # Get the objective expression
    objective_expr = variable_state.objective_expression

    # Automatically substitute state variables with final state values
    if state_syms:
        # Create substitution mapping
        old_syms = []
        new_syms = []

        for i, sym in enumerate(state_syms):
            old_syms.append(sym)
            # Handle dimension mismatch for single state
            if len(state_syms) == 1:
                new_syms.append(xf_vec)
            else:
                new_syms.append(xf_vec[i])

        # Apply substitution
        objective_expr = ca.substitute([objective_expr], old_syms, new_syms)[0]

    # Create unified objective function
    obj_func = ca.Function(
        "objective",
        [t0, tf, x0_vec, xf_vec, q, param_syms],
        [objective_expr],
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
        result = obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)
        obj_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, obj_output)

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
