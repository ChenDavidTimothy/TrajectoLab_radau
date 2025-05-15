from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

# Use simpler type aliases
_Vector: TypeAlias = np.ndarray  # For 1D arrays
_StateDict: TypeAlias = dict[str, Any]  # Using Any for flexibility
_ControlDict: TypeAlias = dict[str, Any]
_ParamDict: TypeAlias = dict[str, Any]

# Define function type protocols without circular references
_DynamicsFunc: TypeAlias = Callable[[_StateDict, _ControlDict, float, _ParamDict], dict[str, float]]
_ObjectiveFunc: TypeAlias = Callable[[float, float, _StateDict, _StateDict, Any, _ParamDict], float]
_IntegrandFunc: TypeAlias = Callable[[_StateDict, _ControlDict, float, _ParamDict], float]
_ConstraintFunc: TypeAlias = Callable[[_StateDict, _ControlDict, float, _ParamDict], Any]
_EventConstraintFunc: TypeAlias = Callable[
    [float, float, _StateDict, _StateDict, Any, _ParamDict], Any
]


class Constraint:
    def __init__(
        self,
        val: float | None = None,
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
        self._parameters: dict[str, Any] = {}
        self._t0_bounds: list[float] = [0.0, 0.0]
        self._tf_bounds: list[float] = [1.0, 1.0]
        self._dynamics_func: _DynamicsFunc | None = None
        self._objective_type: str | None = None
        self._objective_func: _ObjectiveFunc | None = None
        self._path_constraints: list[_ConstraintFunc] = []
        self._event_constraints: list[_EventConstraintFunc] = []
        self._num_integrals: int = 0
        self._integral_functions: list[_IntegrandFunc] = []

    def set_time_bounds(
        self,
        t0: float = 0.0,
        tf: float = 1.0,
        t0_bounds: list[float] | None = None,
        tf_bounds: list[float] | None = None,
    ) -> "Problem":
        if t0_bounds is None:
            t0_bounds = [t0, t0]
        if tf_bounds is None:
            tf_bounds = [tf, tf]

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
    ) -> "Problem":
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
    ) -> "Problem":
        if bounds is not None:
            lower, upper = bounds

        self._controls[name] = {
            "index": len(self._controls),
            "lower": lower,
            "upper": upper,
        }
        return self

    def add_parameter(self, name: str, value: Any) -> "Problem":
        self._parameters[name] = value
        return self

    def set_dynamics(self, dynamics_func: _DynamicsFunc) -> "Problem":
        self._dynamics_func = dynamics_func
        return self

    def set_objective(self, objective_type: str, objective_func: _ObjectiveFunc) -> "Problem":
        # Auto-correct the objective type if it's referencing integrals
        if objective_type == "mayer" and self._num_integrals > 0:
            objective_type = "integral"

        self._objective_type = objective_type
        self._objective_func = objective_func
        return self

    def add_integral(self, integral_func: _IntegrandFunc) -> "Problem":
        self._num_integrals += 1
        self._integral_functions.append(integral_func)
        return self

    def add_path_constraint(self, constraint_func: _ConstraintFunc) -> "Problem":
        self._path_constraints.append(constraint_func)
        return self

    def _convert_to_legacy_problem(self) -> Any:
        from trajectolab.direct_solver import EventConstraint, OptimalControlProblem

        # Create adapter functions
        vectorized_dynamics = self._create_vectorized_dynamics()
        vectorized_objective = self._create_vectorized_objective()
        vectorized_integrand = self._create_vectorized_integrand()
        vectorized_path_constraints = self._create_vectorized_path_constraints()

        # NEW: Add auto-generated event constraints from state definitions
        def auto_event_constraints(
            t0: float, tf: float, x0: np.ndarray, xf: np.ndarray, q: Any, params: dict[str, Any]
        ) -> list:
            result = []

            # Add initial state constraints
            for i, name in enumerate(self._states.keys()):
                state_def = self._states[name]
                initial_constraint = state_def.get("initial_constraint")
                if initial_constraint is not None and isinstance(initial_constraint, Constraint):
                    result.append(
                        EventConstraint(
                            val=x0[i],
                            min_val=initial_constraint.lower,
                            max_val=initial_constraint.upper,
                            equals=initial_constraint.equals,
                        )
                    )

                # Add final state constraints
                final_constraint = state_def.get("final_constraint")
                if final_constraint is not None and isinstance(final_constraint, Constraint):
                    result.append(
                        EventConstraint(
                            val=xf[i],
                            min_val=final_constraint.lower,
                            max_val=final_constraint.upper,
                            equals=final_constraint.equals,
                        )
                    )

            # Add custom event constraints
            for constraint_func in self._event_constraints:
                constraint = constraint_func(
                    t0,
                    tf,
                    {name: x0[i] for i, name in enumerate(self._states.keys())},
                    {name: xf[i] for i, name in enumerate(self._states.keys())},
                    q,
                    params,
                )
                if isinstance(constraint, Constraint):
                    result.append(
                        EventConstraint(
                            val=constraint.val,
                            min_val=constraint.lower,
                            max_val=constraint.upper,
                            equals=constraint.equals,
                        )
                    )
                elif isinstance(constraint, list):
                    for c in constraint:
                        if isinstance(c, Constraint):
                            result.append(
                                EventConstraint(
                                    val=c.val,
                                    min_val=c.lower,
                                    max_val=c.upper,
                                    equals=c.equals,
                                )
                            )

            return result

        # Create legacy problem with the auto event constraints
        return OptimalControlProblem(
            num_states=len(self._states),
            num_controls=len(self._controls),
            dynamics_function=vectorized_dynamics,
            objective_function=vectorized_objective,
            t0_bounds=self._t0_bounds,
            tf_bounds=self._tf_bounds,
            num_integrals=self._num_integrals,
            integral_integrand_function=(vectorized_integrand if self._num_integrals > 0 else None),
            path_constraints_function=(
                vectorized_path_constraints if self._path_constraints else None
            ),
            event_constraints_function=auto_event_constraints,
            problem_parameters=self._parameters,
        )

    def _create_vectorized_dynamics(self) -> Callable:
        dynamics_func = self._dynamics_func
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_dynamics(
            states_vec: np.ndarray, controls_vec: np.ndarray, time: float, params: dict[str, Any]
        ) -> list[float]:
            states_dict = {name: float(states_vec[i]) for i, name in enumerate(state_names)}
            controls_dict = {name: float(controls_vec[i]) for i, name in enumerate(control_names)}

            if dynamics_func is None:
                raise ValueError("Dynamics function not defined")

            result_dict = dynamics_func(states_dict, controls_dict, time, params)
            result_vec = [result_dict[name] for name in state_names]

            return result_vec

        return vectorized_dynamics

    def _create_vectorized_objective(self) -> Callable:
        objective_func = self._objective_func
        objective_type = self._objective_type
        state_names = list(self._states.keys())

        def vectorized_objective(
            t0: float, tf: float, x0: np.ndarray, xf: np.ndarray, q: Any, params: dict[str, Any]
        ) -> float:
            if objective_func is None:
                raise ValueError("Objective function not defined")

            if objective_type == "mayer":
                initial_states = {name: float(x0[i]) for i, name in enumerate(state_names)}
                final_states = {name: float(xf[i]) for i, name in enumerate(state_names)}
                return objective_func(t0, tf, initial_states, final_states, q, params)
            else:
                # For integral objectives, just return the integral value
                if isinstance(q, (list, np.ndarray)) and len(q) > 0:
                    # Handle first element specifically
                    if isinstance(q[0], (float, int, np.number)):
                        return float(q[0])
                    return 0.0  # Fallback
                if isinstance(q, (float, int, np.number)):
                    return float(q)
                return 0.0  # Fallback

        return vectorized_objective

    def _create_vectorized_integrand(self) -> Callable:
        integral_functions = self._integral_functions
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_integrand(
            states_vec: np.ndarray,
            controls_vec: np.ndarray,
            time: float,
            integral_idx: int,
            params: dict[str, Any],
        ) -> float:
            if integral_idx >= len(integral_functions):
                return 0.0  # Return a numeric value when no function exists

            states_dict = {name: float(states_vec[i]) for i, name in enumerate(state_names)}
            controls_dict = {name: float(controls_vec[i]) for i, name in enumerate(control_names)}

            # Get the result
            result = integral_functions[integral_idx](states_dict, controls_dict, time, params)
            return result  # Should be a float based on _IntegrandFunc typing

        return vectorized_integrand

    def _create_vectorized_path_constraints(self) -> Callable:
        from trajectolab.direct_solver import PathConstraint

        path_constraints = self._path_constraints
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_path_constraints(
            states_vec: np.ndarray, controls_vec: np.ndarray, time: float, params: dict[str, Any]
        ) -> list:
            states_dict = {name: float(states_vec[i]) for i, name in enumerate(state_names)}
            controls_dict = {name: float(controls_vec[i]) for i, name in enumerate(control_names)}

            result = []
            for constraint_func in path_constraints:
                constraint = constraint_func(states_dict, controls_dict, time, params)
                if isinstance(constraint, Constraint):
                    result.append(
                        PathConstraint(
                            val=constraint.val,
                            min_val=constraint.lower,
                            max_val=constraint.upper,
                            equals=constraint.equals,
                        )
                    )
                elif isinstance(constraint, list):
                    for c in constraint:
                        if isinstance(c, Constraint):
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
