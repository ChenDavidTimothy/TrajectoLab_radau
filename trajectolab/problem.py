from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# Define proper type variables and protocols
T = TypeVar("T")
TState = TypeVar("TState", bound=Dict[str, float])
TControl = TypeVar("TControl", bound=Dict[str, float])
TParams = TypeVar("TParams", bound=Dict[str, Any])


# Protocol classes for callback functions
class DynamicsFunction(Protocol):
    def __call__(
        self,
        states: Dict[str, float],
        controls: Dict[str, float],
        time: float,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        ...


class ObjectiveFunction(Protocol):
    def __call__(
        self,
        t0: float,
        tf: float,
        initial_states: Dict[str, float],
        final_states: Dict[str, float],
        q: Union[float, List[float], NDArray[np.float64]],
        params: Dict[str, Any],
    ) -> float:
        ...


class IntegralFunction(Protocol):
    def __call__(
        self,
        states: Dict[str, float],
        controls: Dict[str, float],
        time: float,
        params: Dict[str, Any],
    ) -> float:
        ...


class ConstraintFunction(Protocol):
    def __call__(
        self,
        states: Dict[str, float],
        controls: Dict[str, float],
        time: float,
        params: Dict[str, Any],
    ) -> Union["Constraint", List["Constraint"]]:
        ...


class EventConstraintFunction(Protocol):
    def __call__(
        self,
        t0: float,
        tf: float,
        initial_states: Dict[str, float],
        final_states: Dict[str, float],
        q: Union[float, List[float], NDArray[np.float64]],
        params: Dict[str, Any],
    ) -> Union["Constraint", List["Constraint"]]:
        ...


class Constraint:
    def __init__(
        self,
        val: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        equals: Optional[float] = None,
    ):
        self.val: Optional[float] = val
        self.lower: Optional[float] = lower
        self.upper: Optional[float] = upper
        self.equals: Optional[float] = equals

        if equals is not None:
            self.lower = equals
            self.upper = equals


class Problem:
    def __init__(self, name: str = "Unnamed Problem"):
        self.name: str = name
        self._states: Dict[str, Dict[str, Any]] = {}
        self._controls: Dict[str, Dict[str, Any]] = {}
        self._parameters: Dict[str, Any] = {}
        self._t0_bounds: List[float] = [0.0, 0.0]
        self._tf_bounds: List[float] = [1.0, 1.0]
        self._dynamics_func: Optional[DynamicsFunction] = None
        self._objective_type: Optional[str] = None
        self._objective_func: Optional[ObjectiveFunction] = None
        self._path_constraints: List[ConstraintFunction] = []
        self._event_constraints: List[EventConstraintFunction] = []
        self._num_integrals: int = 0
        self._integral_functions: List[IntegralFunction] = []

    def set_time_bounds(
        self,
        t0: float = 0.0,
        tf: float = 1.0,
        t0_bounds: Optional[List[float]] = None,
        tf_bounds: Optional[List[float]] = None,
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
        initial_constraint: Optional[Constraint] = None,
        final_constraint: Optional[Constraint] = None,
        bounds: Optional[Tuple[float, float]] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
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
        bounds: Optional[Tuple[float, float]] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> "Problem":
        if bounds is not None:
            lower, upper = bounds

        self._controls[name] = {"index": len(self._controls), "lower": lower, "upper": upper}
        return self

    def add_parameter(self, name: str, value: Any) -> "Problem":
        self._parameters[name] = value
        return self

    def set_dynamics(
        self,
        dynamics_func: DynamicsFunction,
    ) -> "Problem":
        self._dynamics_func = dynamics_func
        return self

    def set_objective(
        self,
        objective_type: str,
        objective_func: ObjectiveFunction,
    ) -> "Problem":
        # Auto-correct the objective type if it's referencing integrals
        if objective_type == "mayer" and self._num_integrals > 0:
            objective_type = "integral"

        self._objective_type = objective_type
        self._objective_func = objective_func
        return self

    def add_integral(
        self,
        integral_func: IntegralFunction,
    ) -> "Problem":
        self._num_integrals += 1
        self._integral_functions.append(integral_func)
        return self

    def add_path_constraint(
        self,
        constraint_func: ConstraintFunction,
    ) -> "Problem":
        self._path_constraints.append(constraint_func)
        return self

    def _convert_to_legacy_problem(self):
        from trajectolab.direct_solver import OptimalControlProblem

        # Create adapter functions
        vectorized_dynamics = self._create_vectorized_dynamics()
        vectorized_objective = self._create_vectorized_objective()
        vectorized_integrand = self._create_vectorized_integrand()
        vectorized_path_constraints = self._create_vectorized_path_constraints()

        # This creates a function that adds all the initial and final state constraints
        def auto_event_constraints(
            t0: float,
            tf: float,
            x0: List[float],
            xf: List[float],
            q: Union[float, List[float], NDArray[np.float64]],
            params: Dict[str, Any],
        ) -> List[Any]:
            from trajectolab.direct_solver import EventConstraint

            result: List[Any] = []

            # Add initial state constraints
            for i, name in enumerate(self._states.keys()):
                state_def = self._states[name]
                if state_def.get("initial_constraint"):
                    constraint = state_def["initial_constraint"]
                    if constraint is not None:  # Type check
                        result.append(
                            EventConstraint(
                                val=x0[i],
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

                # Add final state constraints
                if state_def.get("final_constraint"):
                    constraint = state_def["final_constraint"]
                    if constraint is not None:  # Type check
                        result.append(
                            EventConstraint(
                                val=xf[i],
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

            # Add custom event constraints
            for constraint_func in self._event_constraints:
                constraint_result = constraint_func(
                    t0,
                    tf,
                    {name: x0[i] for i, name in enumerate(self._states.keys())},
                    {name: xf[i] for i, name in enumerate(self._states.keys())},
                    q,
                    params,
                )
                if isinstance(constraint_result, Constraint):
                    result.append(
                        EventConstraint(
                            val=constraint_result.val,
                            min_val=constraint_result.lower,
                            max_val=constraint_result.upper,
                            equals=constraint_result.equals,
                        )
                    )
                else:
                    result.extend(
                        [
                            EventConstraint(
                                val=c.val,
                                min_val=c.lower,
                                max_val=c.upper,
                                equals=c.equals,
                            )
                            for c in constraint_result
                        ]
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
            integral_integrand_function=vectorized_integrand if self._num_integrals > 0 else None,
            path_constraints_function=(
                vectorized_path_constraints if self._path_constraints else None
            ),
            event_constraints_function=auto_event_constraints,  # Always include auto constraints
            problem_parameters=self._parameters,
        )

    def _create_vectorized_dynamics(
        self,
    ) -> Callable[[List[float], List[float], float, Dict[str, Any]], List[float]]:
        dynamics_func = self._dynamics_func
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_dynamics(
            states_vec: List[float], controls_vec: List[float], time: float, params: Dict[str, Any]
        ) -> List[float]:
            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            if dynamics_func is None:
                raise ValueError("Dynamics function not set")

            result_dict = dynamics_func(states_dict, controls_dict, time, params)
            result_vec = [result_dict[name] for name in state_names]

            return result_vec

        return vectorized_dynamics

    def _create_vectorized_objective(
        self,
    ) -> Callable[
        [
            float,
            float,
            List[float],
            List[float],
            Union[float, List[float], NDArray[np.float64]],
            Dict[str, Any],
        ],
        float,
    ]:
        objective_func = self._objective_func
        objective_type = self._objective_type
        state_names = list(self._states.keys())

        def vectorized_objective(
            t0: float,
            tf: float,
            x0: List[float],
            xf: List[float],
            q: Union[float, List[float], NDArray[np.float64]],
            params: Dict[str, Any],
        ) -> float:
            if objective_type == "mayer":
                initial_states = {name: x0[i] for i, name in enumerate(state_names)}
                final_states = {name: xf[i] for i, name in enumerate(state_names)}

                if objective_func is None:
                    raise ValueError("Objective function not set")

                return objective_func(t0, tf, initial_states, final_states, q, params)
            else:
                # For integral objectives, just return the integral value
                if isinstance(q, (list, np.ndarray)):
                    if len(q) > 0:
                        # Handle lists and arrays properly
                        if isinstance(q, np.ndarray):
                            return float(q.item(0) if q.size > 0 else 0.0)
                        else:  # It's a list
                            return float(q[0])
                    return 0.0
                return float(q)

        return vectorized_objective

    def _create_vectorized_integrand(
        self,
    ) -> Callable[[List[float], List[float], float, int, Dict[str, Any]], float]:
        integral_functions = self._integral_functions
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_integrand(
            states_vec: List[float],
            controls_vec: List[float],
            time: float,
            integral_idx: int,
            params: Dict[str, Any],
        ) -> float:
            if integral_idx >= len(integral_functions):
                return 0.0  # Return a numeric value when no function exists

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            # Just return the result directly
            result = integral_functions[integral_idx](states_dict, controls_dict, time, params)
            return float(result)

        return vectorized_integrand

    def _create_vectorized_path_constraints(
        self,
    ) -> Callable[[List[float], List[float], float, Dict[str, Any]], List[Any]]:
        path_constraints = self._path_constraints
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_path_constraints(
            states_vec: List[float], controls_vec: List[float], time: float, params: Dict[str, Any]
        ) -> List[Any]:
            from trajectolab.direct_solver import PathConstraint

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            result: List[Any] = []
            for constraint_func in path_constraints:
                constraint_result = constraint_func(states_dict, controls_dict, time, params)
                if isinstance(constraint_result, Constraint):
                    result.append(
                        PathConstraint(
                            val=constraint_result.val,
                            min_val=constraint_result.lower,
                            max_val=constraint_result.upper,
                            equals=constraint_result.equals,
                        )
                    )
                else:
                    result.extend(
                        [
                            PathConstraint(
                                val=c.val,
                                min_val=c.lower,
                                max_val=c.upper,
                                equals=c.equals,
                            )
                            for c in constraint_result
                        ]
                    )

            return result

        return vectorized_path_constraints
