from __future__ import annotations

from typing import Any, cast

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


class Constraint:
    """
    Represents a constraint on a state, control, or parameter.

    Attributes:
        val: The value being constrained
        lower: Lower bound for the constraint
        upper: Upper bound for the constraint
        equals: Equality constraint value (sets both lower and upper to the same value)
    """

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
    """
    Defines an optimal control problem with states, controls, parameters,
    dynamics, objectives, and constraints.
    """

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

        # Solver configuration
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
        """
        Sets the bounds for the initial and final times of the optimal control problem.

        Args:
            t0: Initial time value
            tf: Final time value
            t0_bounds: Tuple (lower_bound, upper_bound) for initial time
            tf_bounds: Tuple (lower_bound, upper_bound) for final time

        Returns:
            Self for method chaining
        """
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
        """
        Adds a state variable to the optimal control problem.

        Args:
            name: Name of the state
            initial_constraint: Constraint on the initial state
            final_constraint: Constraint on the final state
            bounds: Tuple (lower_bound, upper_bound) for the state
            lower: Lower bound for the state
            upper: Upper bound for the state

        Returns:
            Self for method chaining
        """
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
        """
        Adds a control variable to the optimal control problem.

        Args:
            name: Name of the control
            bounds: Tuple (lower_bound, upper_bound) for the control
            lower: Lower bound for the control
            upper: Upper bound for the control

        Returns:
            Self for method chaining
        """
        if bounds is not None:
            lower, upper = bounds

        self._controls[name] = {"index": len(self._controls), "lower": lower, "upper": upper}
        return self

    def add_parameter(self, name: str, value: Any) -> Problem:
        """
        Adds a parameter to the optimal control problem.

        Args:
            name: Name of the parameter
            value: Value of the parameter

        Returns:
            Self for method chaining
        """
        self._parameters[name] = value
        return self

    def set_dynamics(self, dynamics_func: _DynamicsFuncType) -> Problem:
        """
        Sets the dynamics function for the optimal control problem.

        Args:
            dynamics_func: Function defining the system dynamics
                function(states, controls, time, params) -> dict of state derivatives

        Returns:
            Self for method chaining
        """
        self._dynamics_func = dynamics_func
        return self

    def set_objective(self, objective_type: str, objective_func: _ObjectiveFuncType) -> Problem:
        """
        Sets the objective function for the optimal control problem.

        Args:
            objective_type: Type of objective ('mayer' or 'integral')
            objective_func: Function defining the objective

        Returns:
            Self for method chaining
        """
        # Auto-correct the objective type if it's referencing integrals
        if objective_type == "mayer" and self._num_integrals > 0:
            objective_type = "integral"

        self._objective_type = objective_type
        self._objective_func = objective_func
        return self

    def add_integral(self, integral_func: _IntegrandFuncType) -> Problem:
        """
        Adds an integral term to the optimal control problem.

        Args:
            integral_func: Function defining the integrand

        Returns:
            Self for method chaining
        """
        self._num_integrals += 1
        self._integral_functions.append(integral_func)
        return self

    def add_path_constraint(self, constraint_func: _ConstraintFuncType) -> Problem:
        """
        Adds a path constraint to the optimal control problem.

        Args:
            constraint_func: Function defining the path constraint

        Returns:
            Self for method chaining
        """
        self._path_constraints.append(constraint_func)
        return self

    def add_event_constraint(self, constraint_func: _EventConstraintFuncType) -> Problem:
        """
        Adds an event constraint to the optimal control problem.

        Args:
            constraint_func: Function defining the event constraint

        Returns:
            Self for method chaining
        """
        self._event_constraints.append(constraint_func)
        return self

    def get_dynamics_function(self) -> _DynamicsCallable:
        """
        Gets the vectorized dynamics function for the solver.

        Returns:
            Vectorized dynamics function for use with CasADi solver

        Raises:
            ValueError: If dynamics function has not been set
        """
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
            params = params or {}  # Default to empty dict if None

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            result_dict = dynamics_func(states_dict, controls_dict, time, params)
            result_vec = [cast(_CasadiMX, result_dict[name]) for name in state_names]
            return result_vec

        return vectorized_dynamics

    def get_objective_function(self) -> _ObjectiveCallable:
        """
        Gets the vectorized objective function for the solver.

        Returns:
            Vectorized objective function for use with CasADi solver

        Raises:
            ValueError: If objective function has not been set
        """
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
            params = params or {}  # Default to empty dict if None

            if objective_type == "mayer":
                initial_states = {name: x0[i] for i, name in enumerate(state_names)}
                final_states = {name: xf[i] for i, name in enumerate(state_names)}
                return cast(
                    _CasadiMX, objective_func(t0, tf, initial_states, final_states, q, params)
                )
            else:
                # For integral objectives, just return the integral value
                if q is None:
                    raise ValueError("Integral value q is None for integral objective")
                return cast(
                    _CasadiMX, q[0] if isinstance(q, (list, np.ndarray)) and len(q) > 0 else q
                )

        return vectorized_objective

    def get_integrand_function(self) -> _IntegralIntegrandCallable | None:
        """
        Gets the vectorized integrand function for the solver.

        Returns:
            Vectorized integrand function for use with CasADi solver, or None if no integrals
        """
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
            params = params or {}  # Default to empty dict if None

            if integral_idx >= len(integral_functions):
                return cast(_CasadiMX, 0.0)

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            return cast(
                _CasadiMX,
                integral_functions[integral_idx](states_dict, controls_dict, time, params),
            )

        return vectorized_integrand

    def get_path_constraints_function(self) -> _PathConstraintsCallable | None:
        """
        Gets the vectorized path constraints function for the solver.

        Returns:
            Vectorized path constraints function for use with CasADi solver, or None if no path constraints
        """
        if not self._path_constraints:
            return None

        path_constraints = self._path_constraints
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_path_constraints(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            params: _ProblemParameters,
        ) -> list[PathConstraint]:
            params = params or {}  # Default to empty dict if None

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            result: list[PathConstraint] = []
            for constraint_func in path_constraints:
                constraint_result: Any = constraint_func(states_dict, controls_dict, time, params)

                if isinstance(constraint_result, Constraint):
                    if constraint_result.val is not None:
                        result.append(
                            PathConstraint(
                                val=cast(_CasadiMX, constraint_result.val),  # Add cast here
                                min_val=constraint_result.lower,
                                max_val=constraint_result.upper,
                                equals=constraint_result.equals,
                            )
                        )
                elif isinstance(constraint_result, list):
                    constraint_list = constraint_result
                    for c in constraint_list:
                        if c.val is not None:
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

    def get_event_constraints_function(self) -> _EventConstraintsCallable:
        """
        Gets the vectorized event constraints function for the solver.

        Returns:
            Vectorized event constraints function for use with CasADi solver
        """
        state_names = list(self._states.keys())
        event_constraints = self._event_constraints

        def auto_event_constraints(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0: _CasadiMX,
            xf: _CasadiMX,
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> list[EventConstraint]:
            params = params or {}
            result: list[EventConstraint] = []

            # Add initial state constraints
            for i, name in enumerate(self._states.keys()):
                state_def = self._states[name]
                if state_def.get("initial_constraint"):
                    constraint = state_def["initial_constraint"]
                    if constraint is not None:
                        if constraint.val is None:
                            val = x0[i]
                        else:
                            val = constraint.val

                        result.append(
                            EventConstraint(
                                val=val,
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

                # Add final state constraints
                if state_def.get("final_constraint"):
                    constraint = state_def["final_constraint"]
                    if constraint is not None:
                        if constraint.val is None:
                            val = xf[i]
                        else:
                            val = constraint.val

                        result.append(
                            EventConstraint(
                                val=val,
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

            # Add custom event constraints
            for constraint_func in event_constraints:
                constraint_result: Any = constraint_func(
                    t0,
                    tf,
                    {name: x0[i] for i, name in enumerate(state_names)},
                    {name: xf[i] for i, name in enumerate(state_names)},
                    q,
                    params,
                )

                if isinstance(constraint_result, Constraint):
                    if constraint_result.val is not None:
                        result.append(
                            EventConstraint(
                                val=cast(_CasadiMX, constraint_result.val),  # Add cast here
                                min_val=constraint_result.lower,
                                max_val=constraint_result.upper,
                                equals=constraint_result.equals,
                            )
                        )
                elif isinstance(constraint_result, list):
                    constraint_list = constraint_result
                    for c in constraint_list:
                        if c.val is not None:
                            result.append(
                                EventConstraint(
                                    val=c.val, min_val=c.lower, max_val=c.upper, equals=c.equals
                                )
                            )

            return result

        return auto_event_constraints
