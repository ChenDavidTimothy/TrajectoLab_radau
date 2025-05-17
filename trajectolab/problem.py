from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import casadi as ca
import numpy as np

from .tl_types import (
    EventConstraint,
    PathConstraint,
    _CasadiMX,
    _FloatArray,
    _ProblemParameters,
    _SymExpr,
    _SymType,
)


class TimeVariableImpl:
    """
    Symbolic time variable with initial and final time properties.
    """

    def __init__(self, sym_var: _SymType, sym_initial: _SymType, sym_final: _SymType):
        self._sym_var = sym_var
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    def __call__(self, other=None):
        """Return the time symbol when called with no arguments."""
        if other is None:
            return self._sym_var
        raise NotImplementedError("Time indexing not yet implemented")

    @property
    def initial(self) -> _SymType:
        """Get initial time symbol."""
        return self._sym_initial

    @property
    def final(self) -> _SymType:
        """Get final time symbol."""
        return self._sym_final

    # Allow direct use in expressions
    def __add__(self, other):
        return self._sym_var + other

    def __radd__(self, other):
        return other + self._sym_var

    def __sub__(self, other):
        return self._sym_var - other

    def __rsub__(self, other):
        return other - self._sym_var

    def __mul__(self, other):
        return self._sym_var * other

    def __rmul__(self, other):
        return other * self._sym_var

    def __truediv__(self, other):
        return self._sym_var / other

    def __rtruediv__(self, other):
        return other / self._sym_var

    def __pow__(self, other):
        return self._sym_var**other

    def __neg__(self):
        return -self._sym_var

    # Comparison operators
    def __lt__(self, other):
        return self._sym_var < other

    def __le__(self, other):
        return self._sym_var <= other

    def __gt__(self, other):
        return self._sym_var > other

    def __ge__(self, other):
        return self._sym_var >= other

    def __eq__(self, other):
        return self._sym_var == other

    def __ne__(self, other):
        return self._sym_var != other


class Constraint:
    """
    Represents a constraint in an optimal control problem.
    Can be used for both path and event constraints.
    """

    def __init__(
        self,
        val: _SymExpr | None = None,
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
    Defines an optimal control problem using symbolic expressions.
    """

    def __init__(self, name: str = "Unnamed Problem") -> None:
        self.name = name

        # Symbolic variables
        self._sym_states: Dict[str, _SymType] = {}
        self._sym_controls: Dict[str, _SymType] = {}
        self._sym_parameters: Dict[str, _SymType] = {}
        self._sym_time: Optional[_SymType] = None
        self._sym_time_initial: Optional[_SymType] = None
        self._sym_time_final: Optional[_SymType] = None

        # Store state and control metadata
        self._states: Dict[str, Dict[str, Any]] = {}
        self._controls: Dict[str, Dict[str, Any]] = {}
        self._parameters: _ProblemParameters = {}

        # Expressions for dynamics, objectives, and constraints
        self._dynamics_expressions: Dict[_SymType, _SymExpr] = {}
        self._objective_expression: Optional[_SymExpr] = None
        self._objective_type: Optional[str] = None
        self._constraints: List[_SymExpr] = []

        # Integral expressions and symbols
        self._integral_expressions: List[_SymExpr] = []
        self._integral_symbols: List[_SymType] = []
        self._num_integrals: int = 0

        # Time bounds
        self._t0_bounds: tuple[float, float] = (0.0, 0.0)
        self._tf_bounds: tuple[float, float] = (1.0, 1.0)

        # Mesh configuration
        self.collocation_points_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: Optional[_FloatArray] = None

        # Initial guess and solver options
        self.initial_guess: Any = None
        self.default_initial_guess_values: Any = None
        self.solver_options: Dict[str, Any] = {}

    def time(
        self, initial: float = 0.0, final: float | None = None, free_final: bool = False
    ) -> TimeVariableImpl:
        """
        Create symbolic time variable with initial/final values.

        Args:
            initial: Initial time value
            final: Final time value (None means free)
            free_final: Whether final time is free or fixed

        Returns:
            TimeVariable with initial and final time properties
        """
        # Create symbolic variables for time, initial time, and final time
        sym_time = ca.MX.sym("t", 1)
        sym_t0 = ca.MX.sym("t0", 1)
        sym_tf = ca.MX.sym("tf", 1)

        # Store the time bounds
        t0_bounds = (initial, initial)  # Fixed initial time

        if free_final:
            # Free final time
            if final is not None:
                tf_bounds = (0.0, final)  # Upper bound provided
            else:
                tf_bounds = (0.0, 1e6)  # Default large upper bound
        else:
            # Fixed final time (if not provided, same as initial)
            tf_val = final if final is not None else initial
            tf_bounds = (tf_val, tf_val)

        self._t0_bounds = t0_bounds
        self._tf_bounds = tf_bounds

        # Store symbolic variables
        self._sym_time = sym_time
        self._sym_time_initial = sym_t0
        self._sym_time_final = sym_tf

        return TimeVariableImpl(sym_time, sym_t0, sym_tf)

    def state(
        self,
        name: str,
        initial: float | None = None,
        final: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> _SymType:
        """
        Create a symbolic state variable with boundary conditions.

        Args:
            name: State variable name
            initial: Initial value constraint
            final: Final value constraint
            lower: Lower bound
            upper: Upper bound

        Returns:
            Symbolic state variable
        """
        sym_var = ca.MX.sym(name, 1)

        # Store metadata
        self._states[name] = {
            "index": len(self._states),
            "initial_constraint": None if initial is None else Constraint(equals=initial),
            "final_constraint": None if final is None else Constraint(equals=final),
            "lower": lower,
            "upper": upper,
        }

        # Store symbolic variable
        self._sym_states[name] = sym_var

        return sym_var

    def control(
        self, name: str, lower: float | None = None, upper: float | None = None
    ) -> _SymType:
        """
        Create a symbolic control variable with bounds.

        Args:
            name: Control variable name
            lower: Lower bound
            upper: Upper bound

        Returns:
            Symbolic control variable
        """
        sym_var = ca.MX.sym(name, 1)

        # Store metadata
        self._controls[name] = {"index": len(self._controls), "lower": lower, "upper": upper}

        # Store symbolic variable
        self._sym_controls[name] = sym_var

        return sym_var

    def parameter(self, name: str, value: Any) -> _SymType:
        """
        Create a symbolic parameter with a value.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Symbolic parameter variable
        """
        sym_var = ca.MX.sym(name, 1)

        # Store parameter value
        self._parameters[name] = value

        # Store symbolic variable
        self._sym_parameters[name] = sym_var

        return sym_var

    def dynamics(self, dynamics_dict: Dict[_SymType, _SymExpr]) -> None:
        """
        Set system dynamics using symbolic expressions.

        Args:
            dynamics_dict: Dictionary mapping state variables to their derivatives
        """
        self._dynamics_expressions = dynamics_dict

        # Verify all keys correspond to defined state variables
        for state_sym in dynamics_dict.keys():
            # Find which state this corresponds to
            found = False
            for _name, sym in self._sym_states.items():
                if state_sym is sym:  # Use 'is' to check object identity
                    found = True
                    break

            if not found:
                raise ValueError("Dynamics provided for undefined state variable")

    def add_integral(self, integrand_expr: _SymExpr) -> _SymType:
        """
        Add an integral cost term to the problem.

        Args:
            integrand_expr: Expression to be integrated

        Returns:
            Symbolic variable representing the integral value
        """
        # Create symbolic variable for the integral
        integral_name = f"integral_{len(self._integral_expressions)}"
        integral_sym = ca.MX.sym(integral_name, 1)

        # Store the integrand expression
        self._integral_expressions.append(integrand_expr)
        self._integral_symbols.append(integral_sym)

        # Increment internal counter
        self._num_integrals = len(self._integral_expressions)

        return integral_sym

    def minimize(self, objective_expr: _SymExpr) -> None:
        """
        Set objective function to minimize.

        Args:
            objective_expr: Expression to minimize
        """
        self._objective_expression = objective_expr
        self._objective_type = self._determine_objective_type(objective_expr)

    def _determine_objective_type(self, expr: _SymExpr) -> str:
        """
        Determine objective type based on expression structure.

        Args:
            expr: Objective expression

        Returns:
            'integral' or 'mayer' based on expression content
        """
        # Check if expression contains any integral symbols
        for integral_sym in self._integral_symbols:
            if self._expression_contains_symbol(expr, integral_sym):
                return "integral"

        # Default to Mayer
        return "mayer"

    def _expression_contains_symbol(self, expr: _SymExpr, symbol: _SymType) -> bool:
        """
        Check if an expression contains a specific symbol.

        Args:
            expr: Expression to check
            symbol: Symbol to look for

        Returns:
            True if expression contains the symbol
        """
        if isinstance(expr, (int, float)):
            return False

        # Use CasADi's depends_on to check dependency
        return bool(ca.depends_on(expr, symbol))

    def subject_to(self, constraint_expr: _SymExpr) -> None:
        """
        Add a constraint to the problem.

        Args:
            constraint_expr: Constraint expression (e.g., x >= 0)
        """
        self._constraints.append(constraint_expr)

    def set_initial_guess(self, var: _SymType, value: Union[float, np.ndarray]) -> None:
        """
        Set initial guess for a symbolic variable.

        Args:
            var: Symbolic variable
            value: Initial guess value (scalar or array)
        """
        from .direct_solver import InitialGuess

        # Initialize initial_guess if needed
        if self.initial_guess is None:
            self.initial_guess = InitialGuess()

        # Find which category this variable belongs to
        for name, sym in self._sym_states.items():
            if var is sym:
                if isinstance(value, (int, float)):
                    # Create array matching expected format
                    if (
                        not hasattr(self.initial_guess, "states")
                        or self.initial_guess.states is None
                    ):
                        self.initial_guess.states = []

                    # Create default guesses for all intervals
                    if not self.initial_guess.states:
                        num_intervals = len(self.collocation_points_per_interval)
                        if num_intervals == 0:
                            # Default to 1 interval with 5 nodes if not configured
                            num_intervals = 1
                            n_nodes = 5
                            self.initial_guess.states = [
                                np.zeros((len(self._states), n_nodes), dtype=np.float64)
                            ]
                        else:
                            self.initial_guess.states = [
                                np.zeros((len(self._states), n + 1), dtype=np.float64)
                                for n in self.collocation_points_per_interval
                            ]

                    # Set the value for this state
                    idx = self._states[name]["index"]
                    for interval_guess in self.initial_guess.states:
                        interval_guess[idx, :] = value

                    return
                elif isinstance(value, np.ndarray):
                    # TODO: Handle array value case
                    pass

        # Check controls
        for name, sym in self._sym_controls.items():
            if var is sym:
                if isinstance(value, (int, float)):
                    # Create array matching expected format
                    if (
                        not hasattr(self.initial_guess, "controls")
                        or self.initial_guess.controls is None
                    ):
                        self.initial_guess.controls = []

                    # Create default guesses for all intervals
                    if not self.initial_guess.controls:
                        num_intervals = len(self.collocation_points_per_interval)
                        if num_intervals == 0:
                            # Default to 1 interval with 4 control nodes if not configured
                            num_intervals = 1
                            n_nodes = 4
                            self.initial_guess.controls = [
                                np.zeros((len(self._controls), n_nodes), dtype=np.float64)
                            ]
                        else:
                            self.initial_guess.controls = [
                                np.zeros((len(self._controls), n), dtype=np.float64)
                                for n in self.collocation_points_per_interval
                            ]

                    # Set the value for this control
                    idx = self._controls[name]["index"]
                    for interval_guess in self.initial_guess.controls:
                        interval_guess[idx, :] = value

                    return
                elif isinstance(value, np.ndarray):
                    # TODO: Handle array value case
                    pass

    # Conversion methods for solver compatibility

    def get_dynamics_function(self) -> Callable:
        """
        Convert symbolic dynamics to a function compatible with solver.

        Returns:
            Function that maps (state_vector, control_vector, time, params)
            to state derivative vector
        """
        # Gather all state and control symbols in order
        state_syms = [
            self._sym_states[name]
            for name in sorted(self._sym_states.keys(), key=lambda n: self._states[n]["index"])
        ]
        control_syms = [
            self._sym_controls[name]
            for name in sorted(self._sym_controls.keys(), key=lambda n: self._controls[n]["index"])
        ]

        # Create combined vector for CasADi function input
        states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
        controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
        time = self._sym_time if self._sym_time is not None else ca.MX.sym("t", 1)
        param_syms = ca.vertcat(*self._sym_parameters.values()) if self._sym_parameters else ca.MX()

        # Create output vector in same order as state_syms
        dynamics_expr = []
        for state_sym in state_syms:
            if state_sym in self._dynamics_expressions:
                dynamics_expr.append(self._dynamics_expressions[state_sym])
            else:
                # Default to zero if no dynamics provided
                dynamics_expr.append(ca.MX(0))

        dynamics_vec = ca.vertcat(*dynamics_expr) if dynamics_expr else ca.MX()

        # Create CasADi function
        dynamics_func = ca.Function(
            "dynamics", [states_vec, controls_vec, time, param_syms], [dynamics_vec]
        )

        # Create wrapper function that matches existing API
        def vectorized_dynamics(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            params: _ProblemParameters,
        ) -> List[_CasadiMX]:
            # Extract parameter values in correct order
            param_values = []
            for name in self._sym_parameters:
                param_values.append(params.get(name, 0.0))

            param_vec = ca.DM(param_values) if param_values else ca.DM()

            # Call CasADi function and convert to list
            result = dynamics_func(states_vec, controls_vec, time, param_vec)

            # Convert to list of MX elements
            num_states = result.size1()
            result_list = []
            for i in range(num_states):
                result_list.append(result[i])

            return result_list

        return vectorized_dynamics

    def get_objective_function(self) -> Callable:
        """
        Convert symbolic objective to a function compatible with solver.

        Returns:
            Function that maps (t0, tf, x0_vec, xf_vec, q, params)
            to objective value
        """
        if self._objective_expression is None:
            raise ValueError("Objective expression not defined")

        # Gather symbols in order
        state_syms = [
            self._sym_states[name]
            for name in sorted(self._sym_states.keys(), key=lambda n: self._states[n]["index"])
        ]

        # Create inputs for the function
        t0 = self._sym_time_initial if self._sym_time_initial is not None else ca.MX.sym("t0", 1)
        tf = self._sym_time_final if self._sym_time_final is not None else ca.MX.sym("tf", 1)
        x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}", 1) for i in range(len(state_syms))])
        xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}", 1) for i in range(len(state_syms))])
        q = ca.vertcat(*self._integral_symbols) if self._integral_symbols else ca.MX.sym("q", 1)
        param_syms = (
            ca.vertcat(*self._sym_parameters.values())
            if self._sym_parameters
            else ca.MX.sym("p", 0)
        )

        # Create a substitution map for the objective expression
        if self._objective_type == "integral":
            # Use q directly if it's an integral objective
            obj_func = ca.Function(
                "objective", [t0, tf, x0_vec, xf_vec, q, param_syms], [self._objective_expression]
            )
        else:
            # For Mayer objectives, we need to substitute state symbols
            # with their final values (xf) for endpoint objectives
            subs_map = {}

            # Map each state symbol to its final value from xf_vec
            for i, state_sym in enumerate(state_syms):
                subs_map[state_sym] = xf_vec[i]

            # Also substitute initial/final time symbols
            if self._sym_time_initial is not None:
                subs_map[self._sym_time_initial] = t0
            if self._sym_time_final is not None:
                subs_map[self._sym_time_final] = tf

            # Substitute these values in the objective expression
            substituted_obj = ca.substitute(
                [self._objective_expression], list(subs_map.keys()), list(subs_map.values())
            )[0]

            obj_func = ca.Function(
                "objective", [t0, tf, x0_vec, xf_vec, q, param_syms], [substituted_obj]
            )

        # Create wrapper function that matches existing API
        def vectorized_objective(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0_vec: _CasadiMX,
            xf_vec: _CasadiMX,
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> _CasadiMX:
            # Extract parameter values in correct order
            param_values = []
            for name in self._sym_parameters:
                param_values.append(params.get(name, 0.0))

            param_vec = ca.DM(param_values) if param_values else ca.DM()

            # If q is None but we need it, create zero vector
            q_val = q if q is not None else ca.DM.zeros(len(self._integral_symbols), 1)

            # Call CasADi function
            return obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)

        return vectorized_objective

    def get_integrand_function(self) -> Callable | None:
        """
        Convert symbolic integrand expressions to a function compatible with solver.

        Returns:
            Function that maps (state_vector, control_vector, time, integral_idx, params)
            to integrand value or None if no integrals are defined
        """
        if not self._integral_expressions:
            return None

        # Gather symbols in order
        state_syms = [
            self._sym_states[name]
            for name in sorted(self._sym_states.keys(), key=lambda n: self._states[n]["index"])
        ]
        control_syms = [
            self._sym_controls[name]
            for name in sorted(self._sym_controls.keys(), key=lambda n: self._controls[n]["index"])
        ]

        # Create combined vector for CasADi function input
        states_vec = ca.vertcat(*state_syms) if state_syms else ca.MX()
        controls_vec = ca.vertcat(*control_syms) if control_syms else ca.MX()
        time = self._sym_time if self._sym_time is not None else ca.MX.sym("t", 1)
        param_syms = (
            ca.vertcat(*self._sym_parameters.values())
            if self._sym_parameters
            else ca.MX.sym("p", 0)
        )

        # Create separate CasADi functions for each integrand
        integrand_funcs = []
        for expr in self._integral_expressions:
            integrand_funcs.append(
                ca.Function("integrand", [states_vec, controls_vec, time, param_syms], [expr])
            )

        # Create wrapper function that matches existing API
        def vectorized_integrand(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            integral_idx: int,
            params: _ProblemParameters,
        ) -> _CasadiMX:
            if integral_idx >= len(integrand_funcs):
                return ca.MX(0.0)

            # Extract parameter values in correct order
            param_values = []
            for name in self._sym_parameters:
                param_values.append(params.get(name, 0.0))

            param_vec = ca.DM(param_values) if param_values else ca.DM()

            # Call appropriate CasADi function
            return integrand_funcs[integral_idx](states_vec, controls_vec, time, param_vec)

        return vectorized_integrand

    def _is_path_constraint(self, expr: _SymExpr) -> bool:
        """
        Determine if constraint is a path constraint.

        Args:
            expr: Constraint expression

        Returns:
            True if constraint is a path constraint
        """
        # Path constraints only depend on states, controls and time (t)
        # Not on initial/final specific values (t0/tf)
        depends_on_t0_tf = (
            self._sym_time_initial is not None and ca.depends_on(expr, self._sym_time_initial)
        ) or (self._sym_time_final is not None and ca.depends_on(expr, self._sym_time_final))

        return not depends_on_t0_tf

    def _symbolic_constraint_to_path_constraint(self, expr: _SymExpr) -> PathConstraint:
        """
        Convert symbolic constraint to PathConstraint object.

        Args:
            expr: Constraint expression

        Returns:
            PathConstraint object
        """
        # Handle equality constraints: expr == value
        if isinstance(expr, ca.MX) and expr.is_op(ca.OP_EQ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs - rhs, equals=0)

        # Handle inequality constraints: expr <= value or expr >= value
        elif isinstance(expr, ca.MX) and expr.is_op(ca.OP_LE):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs, max_val=float(rhs))

        elif isinstance(expr, ca.MX) and expr.is_op(ca.OP_GE):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs, min_val=float(rhs))

        # Default case
        return PathConstraint(val=expr, equals=0)

    def _symbolic_constraint_to_event_constraint(self, expr: _SymExpr) -> EventConstraint:
        """
        Convert symbolic constraint to EventConstraint object.

        Args:
            expr: Constraint expression

        Returns:
            EventConstraint object
        """
        # Handle equality constraints: expr == value
        if isinstance(expr, ca.MX) and expr.is_op(ca.OP_EQ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs - rhs, equals=0)

        # Handle inequality constraints: expr <= value or expr >= value
        elif isinstance(expr, ca.MX) and expr.is_op(ca.OP_LE):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs, max_val=float(rhs))

        elif isinstance(expr, ca.MX) and expr.is_op(ca.OP_GE):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs, min_val=float(rhs))

        # Default case
        return EventConstraint(val=expr, equals=0)

    def get_path_constraints_function(self) -> Callable | None:
        """
        Convert symbolic path constraints to a function compatible with solver.

        Returns:
            Function that maps (state_vector, control_vector, time, params)
            to list of path constraints
        """
        # Filter constraints
        path_constraints = [expr for expr in self._constraints if self._is_path_constraint(expr)]

        if not path_constraints:
            # Additionally check for state/control bounds
            has_bounds = any(
                self._states[s].get("lower") is not None or self._states[s].get("upper") is not None
                for s in self._states
            ) or any(
                self._controls[c].get("lower") is not None
                or self._controls[c].get("upper") is not None
                for c in self._controls
            )

            if not has_bounds:
                return None

        # Gather symbols in order
        state_syms = [
            self._sym_states[name]
            for name in sorted(self._sym_states.keys(), key=lambda n: self._states[n]["index"])
        ]
        control_syms = [
            self._sym_controls[name]
            for name in sorted(self._sym_controls.keys(), key=lambda n: self._controls[n]["index"])
        ]

        def vectorized_path_constraints(
            states_vec: _CasadiMX,
            controls_vec: _CasadiMX,
            time: _CasadiMX,
            params: _ProblemParameters,
        ) -> list[PathConstraint]:
            result: list[PathConstraint] = []

            # Create substitution map
            subs_map = {}

            # Map each state symbol to its value from states_vec
            for i, state_sym in enumerate(state_syms):
                subs_map[state_sym] = states_vec[i]

            # Map each control symbol to its value from controls_vec
            for i, control_sym in enumerate(control_syms):
                subs_map[control_sym] = controls_vec[i]

            # Map time symbol
            if self._sym_time is not None:
                subs_map[self._sym_time] = time

            # Process each path constraint
            for expr in path_constraints:
                # Substitute values
                substituted_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]

                # Convert to PathConstraint
                result.append(self._symbolic_constraint_to_path_constraint(substituted_expr))

            # Add state bounds as constraints
            for i, name in enumerate(
                sorted(self._states.keys(), key=lambda n: self._states[n]["index"])
            ):
                state_def = self._states[name]
                if state_def.get("lower") is not None:
                    result.append(PathConstraint(val=states_vec[i], min_val=state_def["lower"]))
                if state_def.get("upper") is not None:
                    result.append(PathConstraint(val=states_vec[i], max_val=state_def["upper"]))

            # Add control bounds as constraints
            for i, name in enumerate(
                sorted(self._controls.keys(), key=lambda n: self._controls[n]["index"])
            ):
                control_def = self._controls[name]
                if control_def.get("lower") is not None:
                    result.append(PathConstraint(val=controls_vec[i], min_val=control_def["lower"]))
                if control_def.get("upper") is not None:
                    result.append(PathConstraint(val=controls_vec[i], max_val=control_def["upper"]))

            return result

        return vectorized_path_constraints

    def _has_initial_or_final_state_constraints(self) -> bool:
        """
        Check if problem has initial or final state constraints.

        Returns:
            True if any states have initial or final constraints
        """
        return any(
            s.get("initial_constraint") is not None or s.get("final_constraint") is not None
            for s in self._states.values()
        )

    def get_event_constraints_function(self) -> Callable | None:
        """
        Convert symbolic event constraints to a function compatible with solver.

        Returns:
            Function that maps (t0, tf, x0_vec, xf_vec, q, params)
            to list of event constraints
        """
        # Filter constraints
        event_constraints = [
            expr for expr in self._constraints if not self._is_path_constraint(expr)
        ]

        if not event_constraints and not self._has_initial_or_final_state_constraints():
            return None

        # Gather symbols in order
        state_syms = [
            self._sym_states[name]
            for name in sorted(self._sym_states.keys(), key=lambda n: self._states[n]["index"])
        ]

        def vectorized_event_constraints(
            t0: _CasadiMX,
            tf: _CasadiMX,
            x0_vec: _CasadiMX,
            xf_vec: _CasadiMX,
            q: _CasadiMX | None,
            params: _ProblemParameters,
        ) -> list[EventConstraint]:
            result: list[EventConstraint] = []

            # Create substitution map
            subs_map = {}

            # Map each state symbol to its initial/final value
            for i, state_sym in enumerate(state_syms):
                # We could use a special symbol for initial/final states later
                # For now, just use the state symbol directly
                subs_map[state_sym] = xf_vec[i]  # Default to final for regular constraints

            # Map time symbols
            if self._sym_time_initial is not None:
                subs_map[self._sym_time_initial] = t0
            if self._sym_time_final is not None:
                subs_map[self._sym_time_final] = tf
            if self._sym_time is not None:
                subs_map[self._sym_time] = tf  # Default to final time

            # Map integral symbols
            if q is not None and len(self._integral_symbols) > 0:
                for i, integral_sym in enumerate(self._integral_symbols):
                    if i < q.shape[0]:
                        subs_map[integral_sym] = q[i]

            # Process each event constraint
            for expr in event_constraints:
                # Substitute values
                substituted_expr = ca.substitute(
                    [expr], list(subs_map.keys()), list(subs_map.values())
                )[0]

                # Convert to EventConstraint
                result.append(self._symbolic_constraint_to_event_constraint(substituted_expr))

            # Add initial state constraints
            for i, name in enumerate(
                sorted(self._states.keys(), key=lambda n: self._states[n]["index"])
            ):
                state_def = self._states[name]
                initial_constraint = state_def.get("initial_constraint")
                if initial_constraint is not None:
                    if initial_constraint.equals is not None:
                        result.append(
                            EventConstraint(val=x0_vec[i], equals=initial_constraint.equals)
                        )
                    else:
                        if initial_constraint.lower is not None:
                            result.append(
                                EventConstraint(val=x0_vec[i], min_val=initial_constraint.lower)
                            )
                        if initial_constraint.upper is not None:
                            result.append(
                                EventConstraint(val=x0_vec[i], max_val=initial_constraint.upper)
                            )

            # Add final state constraints
            for i, name in enumerate(
                sorted(self._states.keys(), key=lambda n: self._states[n]["index"])
            ):
                state_def = self._states[name]
                final_constraint = state_def.get("final_constraint")
                if final_constraint is not None:
                    if final_constraint.equals is not None:
                        result.append(
                            EventConstraint(val=xf_vec[i], equals=final_constraint.equals)
                        )
                    else:
                        if final_constraint.lower is not None:
                            result.append(
                                EventConstraint(val=xf_vec[i], min_val=final_constraint.lower)
                            )
                        if final_constraint.upper is not None:
                            result.append(
                                EventConstraint(val=xf_vec[i], max_val=final_constraint.upper)
                            )

            return result

        return vectorized_event_constraints
