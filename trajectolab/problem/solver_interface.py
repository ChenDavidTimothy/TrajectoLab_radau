"""
Redesigned solver interface implementing proper optimal control scaling.

Key principles:
1. Use scaled variables for NLP optimization
2. Transform to physical variables during expression evaluation
3. Apply proper scaling factors (W_f, W_g, w_0) per scale.txt rules
4. Preserve original expression structure (especially for integrals)
"""

from __future__ import annotations

from typing import cast

import casadi as ca

from ..scaling.core_scale import AutoScalingManager
from ..tl_types import (
    CasadiMX,
    DynamicsCallable,
    EventConstraintsCallable,
    IntegralIntegrandCallable,
    ObjectiveCallable,
    PathConstraintsCallable,
    ProblemParameters,
)
from .constraints_problem import get_event_constraints_function as _get_event_constraints_original
from .constraints_problem import get_path_constraints_function as _get_path_constraints_original
from .state import ConstraintState, VariableState


def get_dynamics_function(
    variable_state: VariableState,
    scaling_manager: AutoScalingManager | None = None,
    physical_state_names: list[str] | None = None,
    physical_control_names: list[str] | None = None,
) -> DynamicsCallable:
    """
    Get dynamics function with proper scaling following Rules 2 & 3.

    Implementation:
    1. NLP uses scaled variables (x_scaled, u_scaled)
    2. Transform to physical variables during evaluation
    3. Apply ODE defect scaling W_f = V_y (Rule 3)
    """
    if scaling_manager is None:
        # No scaling - use original implementation
        return _get_dynamics_function_original(variable_state)

    print("🔧 Creating dynamics function with proper scaling")

    # Get physical names and scaling info
    state_names = physical_state_names or []
    control_names = physical_control_names or []

    # Build scaled variable vectors for NLP (input to dynamics function)
    scaled_state_syms = []
    scaled_control_syms = []

    # Get scaled symbols from variable state
    for name in state_names:
        scaled_name = f"{name}_scaled"
        if scaled_name in variable_state.sym_states:
            scaled_state_syms.append(variable_state.sym_states[scaled_name])

    for name in control_names:
        scaled_name = f"{name}_scaled"
        if scaled_name in variable_state.sym_controls:
            scaled_control_syms.append(variable_state.sym_controls[scaled_name])

    # Create NLP input vectors (scaled variables)
    states_vec = ca.vertcat(*scaled_state_syms) if scaled_state_syms else ca.MX()
    controls_vec = ca.vertcat(*scaled_control_syms) if scaled_control_syms else ca.MX()
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)
    )

    # Create substitution map: original physical symbols -> scaled variable expressions
    substitution_map = scaling_manager.create_physical_to_scaled_substitution_map(
        states_vec, controls_vec, state_names, control_names
    )

    # Transform dynamics expressions and apply ODE defect scaling
    scaled_dynamics_expressions = []
    ode_defect_scaling = scaling_manager.ode_defect_scaling

    for i, physical_name in enumerate(state_names):
        if physical_name in scaling_manager.original_physical_symbols:
            original_symbol = scaling_manager.original_physical_symbols[physical_name]

            if original_symbol in variable_state.dynamics_expressions:
                # Get original dynamics expression
                original_expr = variable_state.dynamics_expressions[original_symbol]

                # Transform expression: substitute original symbols with scaled expressions
                transformed_expr = ca.substitute(
                    [original_expr], list(substitution_map.keys()), list(substitution_map.values())
                )[0]

                # Apply ODE defect scaling W_f = V_y (Rule 3)
                w_f = ode_defect_scaling.get(physical_name, 1.0)
                scaled_expr = w_f * transformed_expr
                scaled_dynamics_expressions.append(scaled_expr)

                print(f"  📐 State '{physical_name}': W_f = {w_f:.3e}")
            else:
                scaled_dynamics_expressions.append(ca.MX(0))  # Default
        else:
            scaled_dynamics_expressions.append(ca.MX(0))

    dynamics_vec = (
        ca.vertcat(*scaled_dynamics_expressions) if scaled_dynamics_expressions else ca.MX()
    )

    # Create CasADi function
    dynamics_func = ca.Function(
        "dynamics_proper_scaling", [states_vec, controls_vec, time, param_syms], [dynamics_vec]
    )

    def vectorized_dynamics(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ) -> list[CasadiMX]:
        # Extract parameter values
        param_values = []
        for name in variable_state.sym_parameters:
            param_values.append(params.get(name, 0.0))
        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Evaluate dynamics
        result = dynamics_func(states_vec, controls_vec, time, param_vec)
        dynamics_output = result[0] if isinstance(result, list | tuple) else result

        # Convert to list format
        num_states = int(dynamics_output.size1())
        result_list = []
        for i in range(num_states):
            result_list.append(cast(CasadiMX, dynamics_output[i]))

        return result_list

    print("  ✅ Dynamics function created with proper scaling")
    return vectorized_dynamics


def get_objective_function(
    variable_state: VariableState,
    scaling_manager: AutoScalingManager | None = None,
    physical_state_names: list[str] | None = None,
) -> ObjectiveCallable:
    """
    Get objective function with proper scaling following Rule 5.

    Implementation:
    1. Use original objective expression (no substitution corruption)
    2. Apply multiplicative scaling w_0 * J
    3. Transform variables only during evaluation
    """
    if scaling_manager is None:
        # No scaling - use original implementation
        return _get_objective_function_original(variable_state)

    if variable_state.objective_expression is None:
        raise ValueError("Objective expression not defined")

    print("🔧 Creating objective function with proper scaling (Rule 5)")

    # Get scaling factor
    w_0 = scaling_manager.objective_scaling_factor
    print(f"  📐 Objective scaling factor: w_0 = {w_0:.3e}")

    # Create function inputs (scaled variables for NLP)
    t0 = variable_state.sym_time_initial if variable_state.sym_time_initial else ca.MX.sym("t0", 1)
    tf = variable_state.sym_time_final if variable_state.sym_time_final else ca.MX.sym("tf", 1)

    # Build scaled state vectors
    state_names = physical_state_names or []
    num_states = len(state_names)

    x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}", 1) for i in range(num_states)])
    xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}", 1) for i in range(num_states)])

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

    # Create substitution map for final state values
    substitution_map = {}
    for i, physical_name in enumerate(state_names):
        if physical_name in scaling_manager.original_physical_symbols:
            original_symbol = scaling_manager.original_physical_symbols[physical_name]

            if physical_name in scaling_manager.variable_scaling_factors:
                factors = scaling_manager.variable_scaling_factors[physical_name]
                # Physical = (scaled - r) / v
                physical_expr = (xf_vec[i] - factors.r) / factors.v
                substitution_map[original_symbol] = physical_expr
            else:
                substitution_map[original_symbol] = xf_vec[i]  # No scaling

    # Transform objective expression
    objective_expr = variable_state.objective_expression
    if substitution_map:
        objective_expr = ca.substitute(
            [objective_expr], list(substitution_map.keys()), list(substitution_map.values())
        )[0]

    # Apply multiplicative objective scaling (Rule 5: w_0 * J)
    scaled_objective_expr = w_0 * objective_expr

    # Create objective function
    obj_func = ca.Function(
        "objective_proper_scaling",
        [t0, tf, x0_vec, xf_vec, q, param_syms],
        [scaled_objective_expr],
    )

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
            param_values.append(params.get(name, 0.0))
        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Handle integrals
        q_val = q if q is not None else ca.DM.zeros(len(variable_state.integral_symbols), 1)

        # Evaluate objective
        result = obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)
        obj_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, obj_output)

    print("  ✅ Objective function created with multiplicative scaling")
    return unified_objective


def get_integrand_function(
    variable_state: VariableState,
    scaling_manager: AutoScalingManager | None = None,
    physical_state_names: list[str] | None = None,
    physical_control_names: list[str] | None = None,
) -> IntegralIntegrandCallable | None:
    """
    Get integrand function with proper scaling - CRITICAL for integral objectives.

    Implementation:
    1. Use original integrand expressions (NO CORRUPTION!)
    2. Transform variables during evaluation only
    3. Preserve integrand structure: ∫ f(x, u) dt
    """
    if not variable_state.integral_expressions:
        return None

    if scaling_manager is None:
        # No scaling - use original implementation
        return _get_integrand_function_original(variable_state)

    print("🔧 Creating integrand function with proper scaling")
    print("  ✅ Preserving original integrand structure")

    # Get physical names
    state_names = physical_state_names or []
    control_names = physical_control_names or []

    # Build scaled variable vectors for NLP
    scaled_state_syms = []
    scaled_control_syms = []

    for name in state_names:
        scaled_name = f"{name}_scaled"
        if scaled_name in variable_state.sym_states:
            scaled_state_syms.append(variable_state.sym_states[scaled_name])

    for name in control_names:
        scaled_name = f"{name}_scaled"
        if scaled_name in variable_state.sym_controls:
            scaled_control_syms.append(variable_state.sym_controls[scaled_name])

    states_vec = ca.vertcat(*scaled_state_syms) if scaled_state_syms else ca.MX()
    controls_vec = ca.vertcat(*scaled_control_syms) if scaled_control_syms else ca.MX()
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)
    )

    # Create substitution map
    substitution_map = scaling_manager.create_physical_to_scaled_substitution_map(
        states_vec, controls_vec, state_names, control_names
    )

    # Create functions for each integrand (preserve original structure)
    integrand_funcs = []
    for i, original_expr in enumerate(variable_state.integral_expressions):
        print(f"  📊 Processing integrand {i}: preserving f(x,u) structure")

        # Transform expression: substitute original symbols with scaled expressions
        transformed_expr = ca.substitute(
            [original_expr], list(substitution_map.keys()), list(substitution_map.values())
        )[0]

        # Create function - NOTE: No additional scaling applied to integrand
        integrand_funcs.append(
            ca.Function(
                f"integrand_{i}_proper",
                [states_vec, controls_vec, time, param_syms],
                [transformed_expr],
            )
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

        # Extract parameter values
        param_values = []
        for name in variable_state.sym_parameters:
            param_values.append(params.get(name, 0.0))
        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Evaluate integrand
        result = integrand_funcs[integral_idx](states_vec, controls_vec, time, param_vec)
        integrand_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, integrand_output)

    print("  ✅ Integrand functions created with preserved structure")
    return vectorized_integrand


def get_path_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
    scaling_manager: AutoScalingManager | None = None,
    physical_state_names: list[str] | None = None,
    physical_control_names: list[str] | None = None,
) -> PathConstraintsCallable | None:
    """Get path constraints function with proper scaling (Rule 4: W_g)."""
    if scaling_manager is None:
        # No scaling - use original implementation
        return _get_path_constraints_original(constraint_state, variable_state)

    print("🔧 Creating path constraints function with proper scaling (Rule 4)")

    # For now, use original implementation with transformation
    # TODO: Implement proper W_g scaling
    original_func = _get_path_constraints_original(constraint_state, variable_state)

    if original_func is None:
        return None

    # Create wrapper that applies constraint scaling
    def scaled_path_constraints(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ):
        # Get original constraints
        constraints = original_func(states_vec, controls_vec, time, params)

        # TODO: Apply W_g scaling to constraints
        # For now, return as-is
        return constraints

    print("  ⚠️  Using original constraints (W_g scaling TODO)")
    return scaled_path_constraints


def get_event_constraints_function(
    constraint_state: ConstraintState,
    variable_state: VariableState,
    scaling_manager: AutoScalingManager | None = None,
    physical_state_names: list[str] | None = None,
) -> EventConstraintsCallable | None:
    """Get event constraints function with proper scaling."""
    if scaling_manager is None:
        # No scaling - use original implementation
        return _get_event_constraints_original(constraint_state, variable_state)

    print("🔧 Creating event constraints function with proper scaling")

    # For now, use original implementation
    # TODO: Implement proper constraint scaling for event constraints
    original_func = _get_event_constraints_original(constraint_state, variable_state)

    print("  ⚠️  Using original event constraints (scaling TODO)")
    return original_func


# Original implementations for fallback
def _get_dynamics_function_original(variable_state: VariableState) -> DynamicsCallable:
    """Original dynamics function implementation (no scaling)."""
    # Get all state and control symbols in order
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
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)
    )

    # Create output vector in same order as state_syms
    dynamics_expr = []
    for state_sym in state_syms:
        if state_sym in variable_state.dynamics_expressions:
            dynamics_expr.append(variable_state.dynamics_expressions[state_sym])
        else:
            dynamics_expr.append(ca.MX(0))

    dynamics_vec = ca.vertcat(*dynamics_expr) if dynamics_expr else ca.MX()

    # Create CasADi function
    dynamics_func = ca.Function(
        "dynamics_original", [states_vec, controls_vec, time, param_syms], [dynamics_vec]
    )

    def vectorized_dynamics(
        states_vec: CasadiMX,
        controls_vec: CasadiMX,
        time: CasadiMX,
        params: ProblemParameters,
    ) -> list[CasadiMX]:
        # Extract parameter values
        param_values = []
        for name in variable_state.sym_parameters:
            value = params.get(name, 0.0)
            try:
                param_values.append(float(value))
            except (TypeError, ValueError):
                param_values.append(0.0)

        param_vec = ca.DM(param_values) if param_values else ca.DM()

        # Call function
        result = dynamics_func(states_vec, controls_vec, time, param_vec)
        dynamics_output = result[0] if isinstance(result, list | tuple) else result

        if dynamics_output is None:
            raise ValueError("Dynamics function returned None")

        # Convert to list
        num_states = int(dynamics_output.size1())
        result_list = []
        for i in range(num_states):
            result_list.append(cast(CasadiMX, dynamics_output[i]))

        return result_list

    return vectorized_dynamics


def _get_objective_function_original(variable_state: VariableState) -> ObjectiveCallable:
    """Original objective function implementation (no scaling)."""
    if variable_state.objective_expression is None:
        raise ValueError("Objective expression not defined")

    # Get state symbols in order
    state_syms = [
        variable_state.sym_states[name]
        for name in sorted(
            variable_state.sym_states.keys(),
            key=lambda n: variable_state.states[n]["index"],
        )
    ]

    # Create function inputs
    t0 = variable_state.sym_time_initial if variable_state.sym_time_initial else ca.MX.sym("t0", 1)
    tf = variable_state.sym_time_final if variable_state.sym_time_final else ca.MX.sym("tf", 1)
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

    # Get objective expression
    objective_expr = variable_state.objective_expression

    # Substitute state variables with final state values
    if state_syms:
        old_syms = []
        new_syms = []
        for i, sym in enumerate(state_syms):
            old_syms.append(sym)
            if len(state_syms) == 1:
                new_syms.append(xf_vec)
            else:
                new_syms.append(xf_vec[i])
        objective_expr = ca.substitute([objective_expr], old_syms, new_syms)[0]

    # Create objective function
    obj_func = ca.Function(
        "objective_original", [t0, tf, x0_vec, xf_vec, q, param_syms], [objective_expr]
    )

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
        q_val = q if q is not None else ca.DM.zeros(len(variable_state.integral_symbols), 1)

        # Call function
        result = obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)
        obj_output = result[0] if isinstance(result, list | tuple) else result
        return cast(CasadiMX, obj_output)

    return unified_objective


def _get_integrand_function_original(
    variable_state: VariableState,
) -> IntegralIntegrandCallable | None:
    """Original integrand function implementation (no scaling)."""
    if not variable_state.integral_expressions:
        return None

    # Get symbols in order
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
    time = variable_state.sym_time if variable_state.sym_time is not None else ca.MX.sym("t", 1)
    param_syms = (
        ca.vertcat(*variable_state.sym_parameters.values())
        if variable_state.sym_parameters
        else ca.MX.sym("p", 0)
    )

    # Create functions for each integrand
    integrand_funcs = []
    for expr in variable_state.integral_expressions:
        integrand_funcs.append(
            ca.Function("integrand_original", [states_vec, controls_vec, time, param_syms], [expr])
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
