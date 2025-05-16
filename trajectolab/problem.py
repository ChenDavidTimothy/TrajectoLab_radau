"""
Problem definition for trajectory optimization.
"""

from collections.abc import Callable, Sequence
from typing import Any, Dict

import numpy as np

from trajectolab.trajectolab_types import (
    Constraint,
    DynamicsFunction,
    IntegrandFunction,
    ObjectiveFunction,
    _ConstraintFunc,
    _ControlDict,
    _EventConstraintFunc,
    _ParamDict,
    _StateDict,
    _TimePoint,
)


class Problem:
    """Defines an optimal control problem using a builder pattern."""

    def __init__(self, name: str = "Unnamed Problem"):
        self.name = name
        self._states: Dict[str, Dict[str, Any]] = {}
        self._controls: Dict[str, Dict[str, Any]] = {}
        self._parameters: _ParamDict = {}
        self._t0_bounds: list[float] = [0.0, 0.0]
        self._tf_bounds: list[float] = [1.0, 1.0]
        self._dynamics_func: DynamicsFunction | None = None
        self._objective_type: str | None = None
        self._objective_func: ObjectiveFunction | None = None
        self._path_constraints: list[_ConstraintFunc] = []
        self._event_constraints: list[_EventConstraintFunc] = []
        self._num_integrals: int = 0
        self._integral_functions: list[IntegrandFunction] = []

    def set_time_bounds(
        self,
        t0: _TimePoint = 0.0,
        tf: _TimePoint = 1.0,
        t0_bounds: list[float] | None = None,
        tf_bounds: list[float] | None = None,
    ) -> "Problem":
        """Set the time bounds for the problem."""
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
        bounds: tuple[float | None, float | None] | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> "Problem":
        """Add a state variable to the problem."""
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
        bounds: tuple[float | None, float | None] | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> "Problem":
        """Add a control variable to the problem."""
        if bounds is not None:
            lower, upper = bounds

        self._controls[name] = {
            "index": len(self._controls),
            "lower": lower,
            "upper": upper,
        }
        return self

    def add_parameter(self, name: str, value: object) -> "Problem":
        """Add a parameter to the problem."""
        self._parameters[name] = value
        return self

    def set_dynamics(self, dynamics_func: DynamicsFunction) -> "Problem":
        """Set the dynamics function for the problem."""
        self._dynamics_func = dynamics_func
        return self

    def set_objective(self, objective_type: str, objective_func: ObjectiveFunction) -> "Problem":
        """Set the objective function and type."""
        if objective_type == "mayer" and self._num_integrals > 0:
            objective_type = "integral"

        self._objective_type = objective_type
        self._objective_func = objective_func
        return self

    def add_integral(self, integral_func: IntegrandFunction) -> "Problem":
        """Add an integral term to the objective."""
        self._num_integrals += 1
        self._integral_functions.append(integral_func)
        return self

    def add_path_constraint(self, constraint_func: _ConstraintFunc) -> "Problem":
        """Add a path constraint to the problem."""
        self._path_constraints.append(constraint_func)
        return self

    def add_event_constraint(self, constraint_func: _EventConstraintFunc) -> "Problem":
        """Add an event constraint to the problem."""
        self._event_constraints.append(constraint_func)
        return self

    def _convert_to_legacy_problem(self):
        """Convert to the legacy problem format for the solver."""
        from trajectolab.direct_solver import EventConstraint, OptimalControlProblem

        vectorized_dynamics = self._create_vectorized_dynamics()
        vectorized_objective = self._create_vectorized_objective()
        vectorized_integrand = (
            self._create_vectorized_integrand() if self._num_integrals > 0 else None
        )
        vectorized_path_constraints = (
            self._create_vectorized_path_constraints() if self._path_constraints else None
        )

        # MODIFIED: Signature of auto_event_constraints to align with solver expectations
        def auto_event_constraints(
            t0: float,
            tf: float,
            x0: Any,  # Was _FloatArray, solver provides CasADi-like vector
            xf: Any,  # Was _FloatArray, solver provides CasADi-like vector
            q: Any,  # Was Sequence[float] | float | None, solver may provide CasADi-like scalar
            params: _ParamDict,
        ):
            result = []
            # Initial state constraints
            for i, name in enumerate(self._states.keys()):
                state_def = self._states[name]
                if state_def.get("initial_constraint"):
                    constraint = state_def["initial_constraint"]  # This is problem_types.Constraint
                    if constraint is not None:
                        # x0[i] is now an element from the solver's vector (e.g., CasADi scalar)
                        # constraint.lower/upper/equals are float | None
                        # Assuming direct_solver.EventConstraint can handle float bounds as constants
                        result.append(
                            EventConstraint(  # This is direct_solver.EventConstraint
                                val=x0[
                                    i
                                ],  # Should be fine if x0[i] is CasADi-like and EventConstraint.val expects it
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

                # Add final state constraints
                if state_def.get("final_constraint"):
                    constraint = state_def["final_constraint"]
                    if constraint is not None:
                        result.append(
                            EventConstraint(
                                val=xf[i],  # xf[i] is CasADi-like
                                min_val=constraint.lower,
                                max_val=constraint.upper,
                                equals=constraint.equals,
                            )
                        )

            # Add custom event constraints
            for constraint_func in self._event_constraints:
                # Constructing StateDicts for user's function.
                # If x0, xf are CasADi vectors, x0[i] etc. are CasADi scalars.
                # _StateDict now allows 'Any' for values, so this is compatible.
                initial_states: _StateDict = {
                    name: x0[i]
                    for i, name in enumerate(self._states.keys())  # x0[i] could be symbolic
                }
                final_states: _StateDict = {
                    name: xf[i]
                    for i, name in enumerate(self._states.keys())  # xf[i] could be symbolic
                }

                constraint_obj = constraint_func(t0, tf, initial_states, final_states, q, params)

                processed_constraints = []
                if isinstance(constraint_obj, Constraint):
                    processed_constraints.append(constraint_obj)
                else:  # Assuming it's a list of Constraints
                    processed_constraints.extend(constraint_obj)

                for c in processed_constraints:  # c is problem_types.Constraint
                    # c.val can now be symbolic (due to trajectolab_types.Constraint change)
                    # Assuming EventConstraint from direct_solver expects symbolic val
                    result.append(
                        EventConstraint(
                            val=c.val,
                            min_val=c.lower,
                            max_val=c.upper,
                            equals=c.equals,
                        )
                    )
            return result

        return OptimalControlProblem(
            num_states=len(self._states),
            num_controls=len(self._controls),
            dynamics_function=vectorized_dynamics,
            objective_function=vectorized_objective,
            t0_bounds=self._t0_bounds,
            tf_bounds=self._tf_bounds,
            num_integrals=self._num_integrals,
            integral_integrand_function=vectorized_integrand,
            path_constraints_function=vectorized_path_constraints,
            event_constraints_function=auto_event_constraints,  # Error 3 should be fixed by signature change
            problem_parameters=self._parameters,
        )

    def _create_vectorized_dynamics(self) -> Callable:
        if self._dynamics_func is None:
            raise ValueError("Dynamics function not set")

        dynamics_func = self._dynamics_func
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        # MODIFIED: states_vec, controls_vec are from solver, potentially CasADi-like
        def vectorized_dynamics(
            states_vec: Any, controls_vec: Any, time: _TimePoint, params: _ParamDict
        ) -> list[Any]:  # Return list of CasADi-like scalars
            # _StateDict and _ControlDict now allow Any (CasADi scalars)
            states_dict: _StateDict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict: _ControlDict = {
                name: controls_vec[i] for i, name in enumerate(control_names)
            }
            # User's dynamics_func receives dicts with CasADi scalars, returns dict with CasADi scalars
            result_dict = dynamics_func(states_dict, controls_dict, time, params)
            # Ensure we return a list of values in correct order for the solver
            result_vec = [result_dict[name] for name in state_names]
            return result_vec

        return vectorized_dynamics

    def _create_vectorized_objective(self) -> Callable:
        if self._objective_func is None:
            raise ValueError("Objective function not set")

        objective_func = self._objective_func
        objective_type = self._objective_type
        state_names = list(self._states.keys())

        # MODIFIED: x0, xf, q are from solver, potentially CasADi-like
        def vectorized_objective(
            t0: _TimePoint,
            tf: _TimePoint,
            x0: Any,  # Was _FloatArray
            xf: Any,  # Was _FloatArray
            q: Any,  # Was Sequence[float] | float | None
            params: _ParamDict,
        ) -> Any:  # Objective value could be CasADi scalar
            if objective_type == "mayer":
                initial_states: _StateDict = {  # Values can be CasADi symbolic
                    name: x0[i] for i, name in enumerate(state_names)
                }
                final_states: _StateDict = {  # Values can be CasADi symbolic
                    name: xf[i] for i, name in enumerate(state_names)
                }
                return objective_func(t0, tf, initial_states, final_states, q, params)
            else:  # Integral objective
                if q is None:
                    return 0.0  # Or CasADi DM(0.0) if strict
                # If q is a CasADi scalar (common for single integral objective) or sequence
                if isinstance(
                    q, (list, np.ndarray, Sequence)
                ):  # Check if Sequence is appropriate for CasADi
                    if len(q) > 0:
                        return q[0]  # q[0] could be CasADi scalar
                    return 0.0  # Or CasADi DM(0.0)
                return q  # q is already a scalar (potentially CasADi)

        return vectorized_objective

    def _create_vectorized_integrand(self) -> Callable:
        integral_functions = self._integral_functions
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        # MODIFIED: states_vec, controls_vec are from solver
        def vectorized_integrand(
            states_vec: Any,  # Was _FloatArray
            controls_vec: Any,  # Was _FloatArray
            time: _TimePoint,
            integral_idx: int,
            params: _ParamDict,
        ) -> Any:  # Integrand value could be CasADi scalar
            if integral_idx >= len(integral_functions):
                return 0.0  # Or CasADi DM(0.0)

            states_dict: _StateDict = {  # Values can be CasADi symbolic
                name: states_vec[i] for i, name in enumerate(state_names)
            }
            controls_dict: _ControlDict = {  # Values can be CasADi symbolic
                name: controls_vec[i] for i, name in enumerate(control_names)
            }
            # User's integral_func receives dicts with CasADi scalars, returns CasADi scalar
            result = integral_functions[integral_idx](states_dict, controls_dict, time, params)
            return result

        return vectorized_integrand

    def _create_vectorized_path_constraints(self) -> Callable:
        """Create vectorized path constraints for the solver."""
        path_constraints = self._path_constraints
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        # MODIFIED: states_vec, controls_vec are from solver
        def vectorized_path_constraints(
            states_vec: Any, controls_vec: Any, time: _TimePoint, params: _ParamDict
        ):
            from trajectolab.direct_solver import PathConstraint  # direct_solver.PathConstraint

            states_dict: _StateDict = {  # Values can be CasADi symbolic
                name: states_vec[i] for i, name in enumerate(state_names)
            }
            controls_dict: _ControlDict = {  # Values can be CasADi symbolic
                name: controls_vec[i] for i, name in enumerate(control_names)
            }

            result = []
            for constraint_func in path_constraints:
                # constraint_func is user-defined, gets symbolic dicts, returns problem_types.Constraint
                # where .val can now be symbolic due to trajectolab_types.Constraint change
                constraint_obj = constraint_func(states_dict, controls_dict, time, params)

                processed_constraints = []
                if isinstance(constraint_obj, Constraint):  # problem_types.Constraint
                    processed_constraints.append(constraint_obj)
                else:  # Assuming list of problem_types.Constraint
                    processed_constraints.extend(constraint_obj)

                for c in processed_constraints:  # c is problem_types.Constraint
                    # c.val can be symbolic. direct_solver.PathConstraint.val expects symbolic.
                    # c.lower/upper/equals are float | None. Assuming PathConstraint handles float bounds.
                    result.append(
                        PathConstraint(
                            val=c.val,  # Error 4/5 should be fixed
                            min_val=c.lower,
                            max_val=c.upper,
                            equals=c.equals,
                        )
                    )
            return result

        return vectorized_path_constraints
