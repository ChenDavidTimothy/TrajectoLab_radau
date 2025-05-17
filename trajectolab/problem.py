from __future__ import annotations

import inspect
from typing import Any, Callable  # Removed 'cast' as it's not used in the final version

import casadi as ca
import numpy as np

from .tl_types import (
    EventConstraint,
    PathConstraint,
    _CasadiMX,
    _ConstraintFuncType,
    _ConstraintValue,
    _DynamicsCallable,
    _DynamicsFuncType,
    _EventConstraintFuncType,
    _EventConstraintsCallable,
    _FloatArray,
    _IntegralIntegrandCallable,
    _IntegrandFuncType,
    _ObjectiveCallable,
    _ObjectiveFuncType,
    _PathConstraintsCallable,
    _ProblemParameters,
)

# if not using explicit casts.


def adapt_function_call(func: Callable, **kwargs: Any) -> Any:
    sig = inspect.signature(func)
    filtered_kwargs = {name: value for name, value in kwargs.items() if name in sig.parameters}
    return func(**filtered_kwargs)


class Constraint:
    def __init__(
        self,
        val: _ConstraintValue | None = None,
        lower: float | None = None,
        upper: float | None = None,
        equals: float | None = None,
    ) -> None:
        self.val = val
        self.lower = lower
        self.upper = upper
        self.equals = equals

        if equals is not None:
            self.lower = equals
            self.upper = equals


class Problem:
    def __init__(self, name: str = "Unnamed Problem") -> None:
        self.name = name
        self._states: dict[str, dict[str, Any]] = {}
        self._controls: dict[str, dict[str, Any]] = {}
        self._parameters: _ProblemParameters = {}
        self._t0_bounds: tuple[float, float] = (0.0, 0.0)
        self._tf_bounds: tuple[float, float] = (1.0, 1.0)
        self._dynamics_func: _DynamicsFuncType | None = None
        self._objective_type: str | None = None
        self._objective_func: _ObjectiveFuncType | None = None
        self._path_constraints: list[_ConstraintFuncType] = []
        self._event_constraints: list[_EventConstraintFuncType] = []
        self._num_integrals: int = 0
        self._integral_functions: list[_IntegrandFuncType] = []
        self.collocation_points_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: _FloatArray | None = None
        self.initial_guess: Any = None
        self.default_initial_guess_values: Any = None
        self.solver_options: dict[str, Any] = {}

    def set_time_bounds(
        self,
        t0: float = 0.0,
        tf: float = 1.0,
        t0_bounds: tuple[float, float] | None = None,
        tf_bounds: tuple[float, float] | None = None,
    ) -> Problem:
        if t0_bounds is None:
            t0_bounds = (t0, t0)
        if tf_bounds is None:
            tf_bounds = (tf, tf)
        self._t0_bounds = t0_bounds
        self._tf_bounds = tf_bounds
        return self

    def add_state(
        self,
        name: str,
        initial_constraint: Constraint | None = None,
        final_constraint: Constraint | None = None,
        bounds: tuple[float, float] | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> Problem:
        if bounds is not None:
            lower, upper = bounds
        self._states[name] = {
            "index": len(self._states),
            "initial_constraint": initial_constraint,
            "final_constraint": final_constraint,
            "lower": lower,
            "upper": upper,
        }
        return self

    def add_control(
        self,
        name: str,
        bounds: tuple[float, float] | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> Problem:
        if bounds is not None:
            lower, upper = bounds
        self._controls[name] = {"index": len(self._controls), "lower": lower, "upper": upper}
        return self

    def add_parameter(self, name: str, value: Any) -> Problem:
        self._parameters[name] = value
        return self

    def set_dynamics(self, dynamics_func: _DynamicsFuncType) -> Problem:
        self._dynamics_func = dynamics_func
        return self

    def set_objective(self, objective_type: str, objective_func: _ObjectiveFuncType) -> Problem:
        if objective_type == "mayer" and self._num_integrals > 0:
            objective_type = "integral"
        self._objective_type = objective_type
        self._objective_func = objective_func
        return self

    def add_integral(self, integral_func: _IntegrandFuncType) -> Problem:
        self._num_integrals += 1
        self._integral_functions.append(integral_func)
        return self

    def add_path_constraint(self, constraint_func: _ConstraintFuncType) -> Problem:
        self._path_constraints.append(constraint_func)
        return self

    def add_event_constraint(self, constraint_func: _EventConstraintFuncType) -> Problem:
        self._event_constraints.append(constraint_func)
        return self

    def get_dynamics_function(self) -> _DynamicsCallable:
        if self._dynamics_func is None:
            raise ValueError("Dynamics function not set")
        dynamics_func = self._dynamics_func
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_dynamics(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            params: _ProblemParameters,
        ) -> list[_CasadiMX]:
            params = params or {}
            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}
            result_dict = adapt_function_call(
                dynamics_func, states=states_dict, controls=controls_dict, time=time, params=params
            )
            result_vec = [result_dict[name] for name in state_names]
            return result_vec

        return vectorized_dynamics

    def get_objective_function(self) -> _ObjectiveCallable:
        if self._objective_func is None:
            raise ValueError("Objective function not set")
        objective_func = self._objective_func
        objective_type = self._objective_type
        state_names = list(self._states.keys())

        def vectorized_objective(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0_vec: _CasadiMX,  # Changed from x0 to x0_vec for clarity
            xf_vec: _CasadiMX,  # Changed from xf to xf_vec for clarity
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> _CasadiMX:
            params = params or {}
            if objective_type == "mayer":
                initial_states = {name: x0_vec[i] for i, name in enumerate(state_names)}
                final_states = {name: xf_vec[i] for i, name in enumerate(state_names)}
                result = adapt_function_call(
                    objective_func,
                    initial_time=t0,
                    final_time=tf,
                    initial_states=initial_states,
                    final_states=final_states,
                    integrals=q,
                    params=params,
                )
                return result
            else:
                if q is None:
                    raise ValueError("Integral value q is None for integral objective")
                if isinstance(q, (list, np.ndarray)) and len(q) > 0 and self._num_integrals == 1:
                    return q[0]  # For single integral, return the element
                return q  # For multiple integrals or if q is already a scalar MX

        return vectorized_objective

    def get_integrand_function(self) -> _IntegralIntegrandCallable | None:
        if not self._integral_functions:
            return None
        integral_functions = self._integral_functions
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_integrand(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            integral_idx: int,
            params: _ProblemParameters,
        ) -> _CasadiMX:
            params = params or {}
            if integral_idx >= len(integral_functions):
                return ca.MX(0.0)
            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}
            result = adapt_function_call(
                integral_functions[integral_idx],
                states=states_dict,
                controls=controls_dict,
                time=time,
                params=params,
            )
            return result

        return vectorized_integrand

    def get_path_constraints_function(self) -> _PathConstraintsCallable | None:
        has_explicit_path_constraints = bool(self._path_constraints)
        # Add more conditions if states/controls can define path constraints directly
        # For example:
        # has_state_path_constraints = any(s.get("path_constraints") for s in self._states.values())
        # if not has_explicit_path_constraints and not has_state_path_constraints:
        if not has_explicit_path_constraints:
            return None

        def vectorized_path_constraints(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            params: _ProblemParameters,
        ) -> list[PathConstraint]:
            params = params or {}
            state_names = list(self._states.keys())
            control_names = list(self._controls.keys())
            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            result: list[PathConstraint] = []
            for constraint_func in self._path_constraints:
                constraint_result: Any = adapt_function_call(
                    constraint_func,
                    states=states_dict,
                    controls=controls_dict,
                    time=time,
                    params=params,
                )
                constraints_to_process: list[Constraint] = []
                if isinstance(constraint_result, Constraint):
                    constraints_to_process.append(constraint_result)
                elif isinstance(constraint_result, list):
                    constraints_to_process.extend(
                        c for c in constraint_result if isinstance(c, Constraint)
                    )

                for c_item in constraints_to_process:
                    if c_item.val is not None:
                        val_for_solver: _CasadiMX | float
                        if isinstance(c_item.val, np.ndarray):
                            val_for_solver = ca.MX(c_item.val)
                        elif isinstance(c_item.val, (ca.MX, float)):
                            val_for_solver = c_item.val
                        else:
                            raise TypeError(
                                f"Unexpected type for path constraint value: {type(c_item.val)}"
                            )
                        result.append(
                            PathConstraint(
                                val=val_for_solver,
                                min_val=c_item.lower,
                                max_val=c_item.upper,
                                equals=c_item.equals,
                            )
                        )
            return result

        return vectorized_path_constraints

    def get_event_constraints_function(self) -> _EventConstraintsCallable | None:
        has_explicit_event_constraints = bool(self._event_constraints)
        has_state_boundary_constraints = any(
            s.get("initial_constraint") or s.get("final_constraint") for s in self._states.values()
        )
        if not has_explicit_event_constraints and not has_state_boundary_constraints:
            return None

        def auto_event_constraints(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0_vec: _CasadiMX,
            xf_vec: _CasadiMX,
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> list[EventConstraint]:
            params = params or {}
            result: list[EventConstraint] = []
            state_names = list(self._states.keys())
            initial_states_dict = {name: x0_vec[i] for i, name in enumerate(state_names)}
            final_states_dict = {name: xf_vec[i] for i, name in enumerate(state_names)}

            for _name_idx, state_name in enumerate(state_names):
                state_def = self._states[state_name]
                initial_constraint = state_def.get("initial_constraint")
                if isinstance(initial_constraint, Constraint):
                    val_to_constrain = initial_states_dict[state_name]
                    if initial_constraint.val is not None:
                        if isinstance(initial_constraint.val, np.ndarray):
                            val_to_constrain = ca.MX(initial_constraint.val)
                        elif isinstance(initial_constraint.val, (ca.MX, float)):
                            val_to_constrain = initial_constraint.val
                        else:
                            raise TypeError(
                                f"Unexpected type for initial_constraint.val: {type(initial_constraint.val)}"
                            )

                    if initial_constraint.equals is not None:
                        result.append(
                            EventConstraint(val=val_to_constrain, equals=initial_constraint.equals)
                        )
                    else:
                        if initial_constraint.lower is not None:
                            result.append(
                                EventConstraint(
                                    val=val_to_constrain, min_val=initial_constraint.lower
                                )
                            )
                        if initial_constraint.upper is not None:
                            result.append(
                                EventConstraint(
                                    val=val_to_constrain, max_val=initial_constraint.upper
                                )
                            )

                final_constraint = state_def.get("final_constraint")
                if isinstance(final_constraint, Constraint):
                    val_to_constrain = final_states_dict[state_name]
                    if final_constraint.val is not None:
                        if isinstance(final_constraint.val, np.ndarray):
                            val_to_constrain = ca.MX(final_constraint.val)
                        elif isinstance(final_constraint.val, (ca.MX, float)):
                            val_to_constrain = final_constraint.val
                        else:
                            raise TypeError(
                                f"Unexpected type for final_constraint.val: {type(final_constraint.val)}"
                            )

                    if final_constraint.equals is not None:
                        result.append(
                            EventConstraint(val=val_to_constrain, equals=final_constraint.equals)
                        )
                    else:
                        if final_constraint.lower is not None:
                            result.append(
                                EventConstraint(
                                    val=val_to_constrain, min_val=final_constraint.lower
                                )
                            )
                        if final_constraint.upper is not None:
                            result.append(
                                EventConstraint(
                                    val=val_to_constrain, max_val=final_constraint.upper
                                )
                            )

            for constraint_func in self._event_constraints:
                constraint_result: Any = adapt_function_call(
                    constraint_func,
                    initial_time=t0,
                    final_time=tf,
                    initial_states=initial_states_dict,
                    final_states=final_states_dict,
                    integrals=q,
                    params=params,
                )
                constraints_to_process: list[Constraint] = []
                if isinstance(constraint_result, Constraint):
                    constraints_to_process.append(constraint_result)
                elif isinstance(constraint_result, list):
                    constraints_to_process.extend(
                        c for c in constraint_result if isinstance(c, Constraint)
                    )
                for c_item in constraints_to_process:
                    if c_item.val is not None:
                        val_for_solver: _CasadiMX | float
                        if isinstance(c_item.val, np.ndarray):
                            val_for_solver = ca.MX(c_item.val)
                        elif isinstance(c_item.val, (ca.MX, float)):
                            val_for_solver = c_item.val
                        else:
                            raise TypeError(
                                f"Unexpected type for event constraint value: {type(c_item.val)}"
                            )
                        result.append(
                            EventConstraint(
                                val=val_for_solver,
                                min_val=c_item.lower,
                                max_val=c_item.upper,
                                equals=c_item.equals,
                            )
                        )
            return result

        return auto_event_constraints
