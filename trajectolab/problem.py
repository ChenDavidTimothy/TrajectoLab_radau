from __future__ import annotations

from typing import Any, cast

import numpy as np

# Import types from tl_types
from .tl_types import (  # Import new types we added
    EventConstraint,
    PathConstraint,
    _CasadiMX,
    _ConstraintFuncType,
    _DynamicsCallable,
    _DynamicsFuncType,
    _EventConstraintFuncType,
    _IntegralIntegrandCallable,
    _IntegrandFuncType,
    _ObjectiveCallable,
    _ObjectiveFuncType,
    _PathConstraintsCallable,
    _ProblemParameters,
)


class Constraint:
    val: Any  # Use Any since it can be None or CasADI values
    lower: float | None
    upper: float | None
    equals: float | None

    def __init__(
        self,
        val: Any = None,
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
    name: str
    _states: dict[str, dict[str, Any]]
    _controls: dict[str, dict[str, Any]]
    _parameters: _ProblemParameters
    _t0_bounds: tuple[float, float]
    _tf_bounds: tuple[float, float]
    _dynamics_func: _DynamicsFuncType | None
    _objective_type: str | None
    _objective_func: _ObjectiveFuncType | None
    _path_constraints: list[_ConstraintFuncType]
    _event_constraints: list[_EventConstraintFuncType]
    _num_integrals: int
    _integral_functions: list[_IntegrandFuncType]

    def __init__(self, name: str = "Unnamed Problem") -> None:
        self.name = name
        self._states = {}
        self._controls = {}
        self._parameters = {}
        self._t0_bounds = (0.0, 0.0)  # Use tuple instead of list
        self._tf_bounds = (1.0, 1.0)  # Use tuple instead of list
        self._dynamics_func = None
        self._objective_type = None
        self._objective_func = None
        self._path_constraints = []
        self._event_constraints = []
        self._num_integrals = 0
        self._integral_functions = []

    def set_time_bounds(
        self,
        t0: float = 0.0,
        tf: float = 1.0,
        t0_bounds: tuple[float, float] | None = None,
        tf_bounds: tuple[float, float] | None = None,
    ) -> Problem:
        if t0_bounds is None:
            t0_bounds = (t0, t0)  # Use tuple instead of list
        if tf_bounds is None:
            tf_bounds = (tf, tf)  # Use tuple instead of list

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
        # Auto-correct the objective type if it's referencing integrals
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

    def _convert_to_legacy_problem(
        self,
    ) -> Any:  # Returns OptimalControlProblem, using Any to avoid circular import
        from trajectolab.direct_solver import EventConstraint, OptimalControlProblem

        # Create adapter functions
        vectorized_dynamics = self._create_vectorized_dynamics()
        vectorized_objective = self._create_vectorized_objective()
        vectorized_integrand = self._create_vectorized_integrand()
        vectorized_path_constraints = self._create_vectorized_path_constraints()

        # Create function that adds initial and final state constraints
        def auto_event_constraints(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0: _CasadiMX,
            xf: _CasadiMX,
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> list[EventConstraint]:
            # params is guaranteed to be non-None in this context
            params = params or {}  # Default to empty dict if None
            result: list[EventConstraint] = []

            # Add initial state constraints
            for i, name in enumerate(self._states.keys()):
                state_def = self._states[name]
                if state_def.get("initial_constraint"):
                    constraint = state_def["initial_constraint"]
                    if constraint is not None:  # Type narrowing
                        # Make sure we have a valid value for the constraint
                        if constraint.val is None:
                            # Use x0[i] as the value if constraint.val is None
                            val = x0[i]
                        else:
                            val = constraint.val

                        result.append(
                            EventConstraint(
                                val=val,  # Use the valid value
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

                # Add final state constraints
                if state_def.get("final_constraint"):
                    constraint = state_def["final_constraint"]
                    if constraint is not None:  # Type narrowing
                        # Make sure we have a valid value for the constraint
                        if constraint.val is None:
                            # Use xf[i] as the value if constraint.val is None
                            val = xf[i]
                        else:
                            val = constraint.val

                        result.append(
                            EventConstraint(
                                val=val,  # Use the valid value
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

            # Add custom event constraints
            for constraint_func in self._event_constraints:
                # Use any to avoid type conflicts
                constraint_result: Any = constraint_func(
                    t0,
                    tf,
                    {name: x0[i] for i, name in enumerate(self._states.keys())},
                    {name: xf[i] for i, name in enumerate(self._states.keys())},
                    q,
                    params,
                )

                # Handle different return types explicitly
                if isinstance(constraint_result, Constraint):
                    # Handle single constraint
                    if constraint_result.val is not None:
                        result.append(
                            EventConstraint(
                                val=constraint_result.val,
                                min_val=constraint_result.lower,
                                max_val=constraint_result.upper,
                                equals=constraint_result.equals,
                            )
                        )
                else:
                    # Handle list of constraints
                    constraint_list = constraint_result  # This is now known to be a list
                    for c in constraint_list:
                        if c.val is not None:  # Skip constraints with None values
                            result.append(
                                EventConstraint(
                                    val=c.val, min_val=c.lower, max_val=c.upper, equals=c.equals
                                )
                            )

            return result

        # Create legacy problem with the auto event constraints
        return OptimalControlProblem(
            num_states=len(self._states),
            num_controls=len(self._controls),
            dynamics_function=vectorized_dynamics,
            objective_function=vectorized_objective,
            t0_bounds=self._t0_bounds,  # Removed redundant cast
            tf_bounds=self._tf_bounds,
            num_integrals=self._num_integrals,
            integral_integrand_function=vectorized_integrand if self._num_integrals > 0 else None,
            path_constraints_function=(
                vectorized_path_constraints if self._path_constraints else None
            ),
            event_constraints_function=auto_event_constraints,  # Always include auto constraints
            problem_parameters=self._parameters,
        )

    def _create_vectorized_dynamics(self) -> _DynamicsCallable:
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
            # params is guaranteed to be non-None in this context
            params = params or {}  # Default to empty dict if None

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            result_dict = dynamics_func(states_dict, controls_dict, time, params)
            result_vec = [cast(_CasadiMX, result_dict[name]) for name in state_names]
            return result_vec

        return vectorized_dynamics

    def _create_vectorized_objective(self) -> _ObjectiveCallable:
        if self._objective_func is None:
            raise ValueError("Objective function not set")

        objective_func = self._objective_func
        objective_type = self._objective_type
        state_names = list(self._states.keys())

        def vectorized_objective(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0: _CasadiMX,
            xf: _CasadiMX,
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> _CasadiMX:
            # params is guaranteed to be non-None in this context
            params = params or {}  # Default to empty dict if None

            if objective_type == "mayer":
                initial_states = {name: x0[i] for i, name in enumerate(state_names)}
                final_states = {name: xf[i] for i, name in enumerate(state_names)}
                # Cast to _CasadiMX to ensure correct return type
                return cast(
                    _CasadiMX, objective_func(t0, tf, initial_states, final_states, q, params)
                )
            else:
                # For integral objectives, just return the integral value
                if q is None:
                    raise ValueError("Integral value q is None for integral objective")
                # Cast to _CasadiMX to ensure correct return type
                return cast(
                    _CasadiMX, q[0] if isinstance(q, (list, np.ndarray)) and len(q) > 0 else q
                )

        return vectorized_objective

    def _create_vectorized_integrand(self) -> _IntegralIntegrandCallable:
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
            # params is guaranteed to be non-None in this context
            params = params or {}  # Default to empty dict if None

            if integral_idx >= len(integral_functions):
                return cast(
                    _CasadiMX, 0.0
                )  # Return a numeric value when no function exists, cast to _CasadiMX

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            # Just return the result directly - DO NOT try to convert to float!
            # Cast to _CasadiMX to ensure correct return type
            return cast(
                _CasadiMX,
                integral_functions[integral_idx](states_dict, controls_dict, time, params),
            )

        return vectorized_integrand

    def _create_vectorized_path_constraints(self) -> _PathConstraintsCallable | None:
        path_constraints = self._path_constraints
        if not path_constraints:
            return None

        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_path_constraints(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            params: _ProblemParameters,
        ) -> list[PathConstraint]:
            # params is guaranteed to be non-None in this context
            params = params or {}  # Default to empty dict if None

            from trajectolab.direct_solver import PathConstraint

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            result: list[PathConstraint] = []
            for constraint_func in path_constraints:
                # Use Any to avoid type conflicts
                constraint_result: Any = constraint_func(states_dict, controls_dict, time, params)

                # Handle different return types explicitly
                if isinstance(constraint_result, Constraint):
                    # Skip constraints with None values
                    if constraint_result.val is not None:
                        result.append(
                            PathConstraint(
                                val=constraint_result.val,
                                min_val=constraint_result.lower,
                                max_val=constraint_result.upper,
                                equals=constraint_result.equals,
                            )
                        )
                else:
                    # Handle list of constraints
                    constraint_list = constraint_result  # This is now known to be a list
                    for c in constraint_list:
                        if c.val is not None:  # Skip constraints with None values
                            result.append(
                                PathConstraint(
                                    val=c.val,
                                    min_val=c.lower,
                                    max_val=c.upper,
                                    equals=c.equals,
                                )
                            )

            return result

        return vectorized_path_constraints
