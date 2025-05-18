from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

import casadi as ca
import numpy as np

from .tl_types import (
    CasadiMX,
    EventConstraint,
    FloatArray,
    FloatMatrix,
    PathConstraint,
    ProblemParameters,
    SymExpr,
    SymType,
)


# Local type aliases for improved readability
_SymStateDict: TypeAlias = dict[str, SymType]
_SymControlDict: TypeAlias = dict[str, SymType]
_SymParameterDict: TypeAlias = dict[str, SymType]
_StateMetadata: TypeAlias = dict[str, Any]
_ControlMetadata: TypeAlias = dict[str, Any]
_ConstraintList: TypeAlias = list[SymExpr]
_IntegralExprList: TypeAlias = list[SymExpr]
_IntegralSymList: TypeAlias = list[SymType]
_DynamicsExprs: TypeAlias = dict[SymType, SymExpr]

# Callback function type aliases
_DynamicsCallback: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters], list[CasadiMX] | CasadiMX
]
_ObjectiveCallback: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters], CasadiMX
]
_IntegrandCallback: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, int, ProblemParameters], CasadiMX
]
_PathConstraintsCallback: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, ProblemParameters], list[PathConstraint]
]
_EventConstraintsCallback: TypeAlias = Callable[
    [CasadiMX, CasadiMX, CasadiMX, CasadiMX, CasadiMX | None, ProblemParameters],
    list[EventConstraint],
]


class TimeVariableImpl:
    def __init__(self, sym_var: SymType, sym_initial: SymType, sym_final: SymType) -> None:
        self._sym_var = sym_var
        self._sym_initial = sym_initial
        self._sym_final = sym_final

    def __call__(self, other: Any = None) -> SymType:
        if other is None:
            return self._sym_var
        raise NotImplementedError("Time indexing not yet implemented")

    @property
    def initial(self) -> SymType:
        return self._sym_initial

    @property
    def final(self) -> SymType:
        return self._sym_final

    # Arithmetic operators
    def __add__(self, other: Any) -> SymType:
        return self._sym_var + other

    def __radd__(self, other: Any) -> SymType:
        return other + self._sym_var

    def __sub__(self, other: Any) -> SymType:
        return self._sym_var - other

    def __rsub__(self, other: Any) -> SymType:
        return other - self._sym_var

    def __mul__(self, other: Any) -> SymType:
        return self._sym_var * other

    def __rmul__(self, other: Any) -> SymType:
        return other * self._sym_var

    def __truediv__(self, other: Any) -> SymType:
        return self._sym_var / other

    def __rtruediv__(self, other: Any) -> SymType:
        return other / self._sym_var

    def __pow__(self, other: Any) -> SymType:
        return self._sym_var**other

    def __neg__(self) -> SymType:
        return cast(SymType, -self._sym_var)

    # Comparison operators
    def __lt__(self, other: Any) -> SymType:
        return self._sym_var < other

    def __le__(self, other: Any) -> SymType:
        return self._sym_var <= other

    def __gt__(self, other: Any) -> SymType:
        return self._sym_var > other

    def __ge__(self, other: Any) -> SymType:
        return self._sym_var >= other

    def __eq__(self, other: Any) -> SymType:
        return self._sym_var == other

    def __ne__(self, other: Any) -> SymType:
        return self._sym_var != other


class Constraint:
    def __init__(
        self,
        val: SymExpr | None = None,
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


class InitialGuessRequirements:
    def __init__(
        self,
        states_shapes: list[tuple[int, int]],
        controls_shapes: list[tuple[int, int]],
        needs_initial_time: bool,
        needs_terminal_time: bool,
        integrals_length: int,
    ) -> None:
        self.states_shapes = states_shapes
        self.controls_shapes = controls_shapes
        self.needs_initial_time = needs_initial_time
        self.needs_terminal_time = needs_terminal_time
        self.integrals_length = integrals_length

    def __str__(self) -> str:
        req = ["Initial Guess Requirements:"]
        req.append(f"  States: {len(self.states_shapes)} intervals")
        for k, shape in enumerate(self.states_shapes):
            req.append(f"    Interval {k}: array of shape {shape}")
        req.append(f"  Controls: {len(self.controls_shapes)} intervals")
        for k, shape in enumerate(self.controls_shapes):
            req.append(f"    Interval {k}: array of shape {shape}")
        if self.needs_initial_time:
            req.append("  Initial time: float")
        if self.needs_terminal_time:
            req.append("  Terminal time: float")
        if self.integrals_length > 0:
            req.append(f"  Integrals: array of length {self.integrals_length}")
        return "\n".join(req)


class SolverInputSummary:
    def __init__(
        self,
        mesh_intervals: int,
        polynomial_degrees: list[int],
        mesh_points: FloatArray,
        initial_guess_source: str,
        states_guess_shapes: list[tuple[int, int]] | None,
        controls_guess_shapes: list[tuple[int, int]] | None,
        initial_time_guess: float | None,
        terminal_time_guess: float | None,
        integrals_guess_length: int | None,
    ) -> None:
        self.mesh_intervals = mesh_intervals
        self.polynomial_degrees = polynomial_degrees
        self.mesh_points = mesh_points
        self.initial_guess_source = initial_guess_source
        self.states_guess_shapes = states_guess_shapes
        self.controls_guess_shapes = controls_guess_shapes
        self.initial_time_guess = initial_time_guess
        self.terminal_time_guess = terminal_time_guess
        self.integrals_guess_length = integrals_guess_length

    def __str__(self) -> str:
        summary = ["Solver Input Summary:"]
        summary.append(f"  Mesh intervals: {self.mesh_intervals}")
        summary.append(f"  Polynomial degrees: {self.polynomial_degrees}")
        summary.append(f"  Mesh points: {np.array2string(self.mesh_points, precision=4)}")
        summary.append(f"  Initial guess source: {self.initial_guess_source}")
        if self.states_guess_shapes:
            summary.append(f"  States guess shapes: {self.states_guess_shapes}")
        if self.controls_guess_shapes:
            summary.append(f"  Controls guess shapes: {self.controls_guess_shapes}")
        if self.initial_time_guess is not None:
            summary.append(f"  Initial time guess: {self.initial_time_guess}")
        if self.terminal_time_guess is not None:
            summary.append(f"  Terminal time guess: {self.terminal_time_guess}")
        if self.integrals_guess_length is not None:
            summary.append(f"  Integrals guess length: {self.integrals_guess_length}")
        return "\n".join(summary)


class Problem:
    def __init__(self, name: str = "Unnamed Problem") -> None:
        self.name = name

        # Symbolic variables
        self._sym_states: _SymStateDict = {}
        self._sym_controls: _SymControlDict = {}
        self._sym_parameters: _SymParameterDict = {}
        self._sym_time: SymType | None = None
        self._sym_time_initial: SymType | None = None
        self._sym_time_final: SymType | None = None

        # Store state and control metadata
        self._states: dict[str, _StateMetadata] = {}
        self._controls: dict[str, _ControlMetadata] = {}
        self._parameters: ProblemParameters = {}

        # Expressions for dynamics, objectives, and constraints
        self._dynamics_expressions: _DynamicsExprs = {}
        self._objective_expression: SymExpr | None = None
        self._objective_type: str | None = None
        self._constraints: _ConstraintList = []

        # Integral expressions and symbols
        self._integral_expressions: _IntegralExprList = []
        self._integral_symbols: _IntegralSymList = []
        self._num_integrals: int = 0

        # Time bounds
        self._t0_bounds: tuple[float, float] = (0.0, 0.0)
        self._tf_bounds: tuple[float, float] = (1.0, 1.0)

        # Mesh configuration - MUST be explicitly set
        self.collocation_points_per_interval: list[int] = []
        self.global_normalized_mesh_nodes: FloatArray | None = None
        self._mesh_configured: bool = False

        # Initial guess - MUST be explicitly provided
        from .direct_solver import InitialGuess

        self.initial_guess: InitialGuess | None = None
        self.solver_options: dict[str, Any] = {}

    def time(
        self, initial: float = 0.0, final: float | None = None, free_final: bool = False
    ) -> TimeVariableImpl:
        # Create symbolic variables for time, initial time, and final time
        sym_time = ca.MX.sym("t", 1)  # type: ignore[arg-type]
        sym_t0 = ca.MX.sym("t0", 1)  # type: ignore[arg-type]
        sym_tf = ca.MX.sym("tf", 1)  # type: ignore[arg-type]

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
    ) -> SymType:
        sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

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

    def control(self, name: str, lower: float | None = None, upper: float | None = None) -> SymType:
        sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

        # Store metadata
        self._controls[name] = {"index": len(self._controls), "lower": lower, "upper": upper}

        # Store symbolic variable
        self._sym_controls[name] = sym_var

        return sym_var

    def parameter(self, name: str, value: Any) -> SymType:
        sym_var = ca.MX.sym(name, 1)  # type: ignore[arg-type]

        # Store parameter value
        self._parameters[name] = value

        # Store symbolic variable
        self._sym_parameters[name] = sym_var

        return sym_var

    def dynamics(self, dynamics_dict: dict[SymType, SymExpr]) -> None:
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

    def add_integral(self, integrand_expr: SymExpr) -> SymType:
        # Create symbolic variable for the integral
        integral_name = f"integral_{len(self._integral_expressions)}"
        integral_sym = ca.MX.sym(integral_name, 1)  # type: ignore[arg-type]

        # Store the integrand expression
        self._integral_expressions.append(integrand_expr)
        self._integral_symbols.append(integral_sym)

        # Increment internal counter
        self._num_integrals = len(self._integral_expressions)

        return integral_sym

    def minimize(self, objective_expr: SymExpr) -> None:
        self._objective_expression = objective_expr
        self._objective_type = self._determine_objective_type(objective_expr)

    def _determine_objective_type(self, expr: SymExpr) -> str:
        # Check if expression contains any integral symbols
        for integral_sym in self._integral_symbols:
            if self._expression_contains_symbol(expr, integral_sym):
                return "integral"

        # Default to Mayer
        return "mayer"

    def _expression_contains_symbol(self, expr: SymExpr, symbol: SymType) -> bool:
        if isinstance(expr, int | float):
            return False

        # Use CasADi's depends_on to check dependency
        return bool(ca.depends_on(expr, symbol))  # type: ignore[attr-defined]

    def subject_to(self, constraint_expr: SymExpr) -> None:
        self._constraints.append(constraint_expr)

    def set_mesh(
        self, polynomial_degrees: list[int], mesh_points: FloatArray | list[float]
    ) -> None:
        """
        Explicitly configure the mesh structure.
        This clears any existing initial guess since mesh configuration has changed.

        Args:
            polynomial_degrees: Number of collocation nodes per interval
            mesh_points: Normalized mesh points in [-1, 1] - accepts any array-like

        Raises:
            ValueError: If mesh specification is invalid
        """
        # Convert to numpy array if needed, ensuring float64 for consistency
        if isinstance(mesh_points, list):
            mesh_array = np.array(mesh_points, dtype=np.float64)
        else:
            # Handle any numpy array dtype by converting to float64
            mesh_array = np.asarray(mesh_points, dtype=np.float64)

        # Validate mesh structure
        if len(polynomial_degrees) != len(mesh_array) - 1:
            raise ValueError(
                f"Number of polynomial degrees ({len(polynomial_degrees)}) must be exactly "
                f"one less than number of mesh points ({len(mesh_array)})"
            )

        # Validate polynomial degrees
        for k, degree in enumerate(polynomial_degrees):
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError(
                    f"Polynomial degree for interval {k} must be positive integer, got {degree}"
                )

        # Validate mesh points
        if not np.isclose(mesh_array[0], -1.0):
            raise ValueError(f"First mesh point must be -1.0, got {mesh_array[0]}")

        if not np.isclose(mesh_array[-1], 1.0):
            raise ValueError(f"Last mesh point must be 1.0, got {mesh_array[-1]}")

        if not np.all(np.diff(mesh_array) > 1e-9):
            raise ValueError("Mesh points must be strictly increasing with minimum spacing of 1e-9")

        # Set mesh configuration
        self.collocation_points_per_interval = polynomial_degrees
        self.global_normalized_mesh_nodes = mesh_array
        self._mesh_configured = True

        # Always clear existing initial guess when mesh changes
        # User must call set_initial_guess() again if they want one for the new mesh
        self.initial_guess = None

    def set_initial_guess(
        self,
        states: Sequence[FloatMatrix] | None = None,
        controls: Sequence[FloatMatrix] | None = None,
        initial_time: float | None = None,
        terminal_time: float | None = None,
        integrals: float | FloatArray | None = None,
    ) -> None:
        """
        Set explicit initial guess for variables.
        All parameters are optional - if not provided, CasADi will use defaults (typically zeros).

        Args:
            states: Optional sequence of state arrays, one per interval.
                   Each array shape: (num_states, Nk+1)
            controls: Optional sequence of control arrays, one per interval.
                     Each array shape: (num_controls, Nk)
            initial_time: Optional initial time guess
            terminal_time: Optional terminal time guess
            integrals: Optional integral values guess

        Raises:
            ValueError: If mesh not configured or dimensions don't match
        """
        if not self._mesh_configured:
            raise ValueError(
                "Mesh must be configured before setting initial guess. "
                "Call set_mesh(polynomial_degrees, mesh_points) first."
            )

        # Validate number of intervals if states/controls provided
        num_intervals = len(self.collocation_points_per_interval)
        num_states = len(self._states)
        num_controls = len(self._controls)

        # Validate states if provided
        states_list: list[FloatMatrix] | None = None
        if states is not None:
            if len(states) != num_intervals:
                raise ValueError(
                    f"If providing states, must provide exactly {num_intervals} arrays, got {len(states)}"
                )

            # Validate state dimensions
            for k, state_array in enumerate(states):
                expected_shape = (num_states, self.collocation_points_per_interval[k] + 1)
                if state_array.shape != expected_shape:
                    raise ValueError(
                        f"State array for interval {k} has shape {state_array.shape}, "
                        f"expected {expected_shape} (num_states={num_states}, "
                        f"num_nodes={self.collocation_points_per_interval[k] + 1})"
                    )

            states_list = [np.array(s, dtype=np.float64) for s in states]

        # Validate controls if provided
        controls_list: list[FloatMatrix] | None = None
        if controls is not None:
            if len(controls) != num_intervals:
                raise ValueError(
                    f"If providing controls, must provide exactly {num_intervals} arrays, got {len(controls)}"
                )

            # Validate control dimensions
            for k, control_array in enumerate(controls):
                expected_shape = (num_controls, self.collocation_points_per_interval[k])
                if control_array.shape != expected_shape:
                    raise ValueError(
                        f"Control array for interval {k} has shape {control_array.shape}, "
                        f"expected {expected_shape} (num_controls={num_controls}, "
                        f"num_nodes={self.collocation_points_per_interval[k]})"
                    )

            controls_list = [np.array(c, dtype=np.float64) for c in controls]

        # Validate integrals if provided
        validated_integrals: float | FloatArray | None = None
        if integrals is not None:
            if self._num_integrals == 0:
                raise ValueError("Problem has no integrals, but integral guess was provided")

            if self._num_integrals == 1:
                if not isinstance(integrals, int | float):
                    raise ValueError(f"For single integral, provide scalar, got {type(integrals)}")
                validated_integrals = integrals
            else:
                integrals_array = np.array(integrals)
                if integrals_array.size != self._num_integrals:
                    raise ValueError(
                        f"Integral guess must have {self._num_integrals} elements, "
                        f"got {integrals_array.size}"
                    )
                validated_integrals = integrals_array

        # Store initial guess (even if partially specified or None)
        from .direct_solver import InitialGuess

        self.initial_guess = InitialGuess(
            initial_time_variable=initial_time,
            terminal_time_variable=terminal_time,
            states=states_list,
            controls=controls_list,
            integrals=validated_integrals,
        )

    def get_initial_guess_requirements(self) -> InitialGuessRequirements:
        """
        Get detailed requirements for initial guess specification.

        Returns:
            InitialGuessRequirements object with exact specifications

        Raises:
            ValueError: If mesh not configured
        """
        if not self._mesh_configured:
            raise ValueError(
                "Mesh must be configured before getting initial guess requirements. "
                "Call set_mesh(polynomial_degrees, mesh_points) first."
            )

        num_states = len(self._states)
        num_controls = len(self._controls)

        states_shapes = [(num_states, N + 1) for N in self.collocation_points_per_interval]
        controls_shapes = [(num_controls, N) for N in self.collocation_points_per_interval]

        needs_initial_time = self._t0_bounds[0] != self._t0_bounds[1]
        needs_terminal_time = self._tf_bounds[0] != self._tf_bounds[1]

        return InitialGuessRequirements(
            states_shapes=states_shapes,
            controls_shapes=controls_shapes,
            needs_initial_time=needs_initial_time,
            needs_terminal_time=needs_terminal_time,
            integrals_length=self._num_integrals,
        )

    def validate_initial_guess(self) -> None:
        """
        Validate that the initial guess (if provided) is complete and correct.
        Initial guess is optional - this only validates if it exists.

        Raises:
            ValueError: If provided initial guess is invalid
        """
        if not self._mesh_configured:
            raise ValueError("Mesh must be configured before validating initial guess")

        # If no initial guess provided, validation passes (CasADi will handle it)
        if self.initial_guess is None:
            return

        ig = self.initial_guess

        # Validate states if provided
        if ig.states is not None:
            requirements = self.get_initial_guess_requirements()

            if len(ig.states) != len(requirements.states_shapes):
                raise ValueError(
                    f"States guess has {len(ig.states)} arrays, "
                    f"but problem needs {len(requirements.states_shapes)}"
                )

            for k, (state_array, expected_shape) in enumerate(
                zip(ig.states, requirements.states_shapes, strict=False)
            ):
                if state_array.shape != expected_shape:
                    raise ValueError(
                        f"State array {k} has shape {state_array.shape}, expected {expected_shape}"
                    )

        # Validate controls if provided
        if ig.controls is not None:
            requirements = self.get_initial_guess_requirements()

            if len(ig.controls) != len(requirements.controls_shapes):
                raise ValueError(
                    f"Controls guess has {len(ig.controls)} arrays, "
                    f"but problem needs {len(requirements.controls_shapes)}"
                )

            for k, (control_array, expected_shape) in enumerate(
                zip(ig.controls, requirements.controls_shapes, strict=False)
            ):
                if control_array.shape != expected_shape:
                    raise ValueError(
                        f"Control array {k} has shape {control_array.shape}, expected {expected_shape}"
                    )

        # Validate integrals if provided
        if ig.integrals is not None:
            if self._num_integrals == 0:
                raise ValueError("Integral guess provided but problem has no integrals")

            if self._num_integrals == 1:
                if not isinstance(ig.integrals, int | float):
                    raise ValueError("Single integral guess must be scalar")
            else:
                integrals_array = np.array(ig.integrals)
                if integrals_array.size != self._num_integrals:
                    raise ValueError(
                        f"Integral guess must have {self._num_integrals} elements, "
                        f"got {integrals_array.size}"
                    )

    def get_solver_input_summary(self) -> SolverInputSummary:
        """
        Get summary of exactly what will be passed to the solver.

        Returns:
            SolverInputSummary object with complete input specification
        """
        if not self._mesh_configured:
            raise ValueError("Mesh must be configured to get solver input summary")

        states_shapes = None
        controls_shapes = None
        initial_time_guess = None
        terminal_time_guess = None
        integrals_length = None
        source = "no_initial_guess"

        if self.initial_guess is not None:
            source = "partial_user_provided"
            states_shapes = (
                [cast(tuple[int, int], s.shape) for s in self.initial_guess.states]
                if self.initial_guess.states
                else None
            )
            controls_shapes = (
                [cast(tuple[int, int], c.shape) for c in self.initial_guess.controls]
                if self.initial_guess.controls
                else None
            )
            initial_time_guess = self.initial_guess.initial_time_variable
            terminal_time_guess = self.initial_guess.terminal_time_variable

            if self.initial_guess.integrals is not None:
                if isinstance(self.initial_guess.integrals, int | float):
                    integrals_length = 1
                else:
                    integrals_length = len(np.array(self.initial_guess.integrals))

            # Check if guess is complete
            is_complete = True
            if self.initial_guess.states is None or len(self.initial_guess.states) != len(
                self.collocation_points_per_interval
            ):
                is_complete = False
            if self.initial_guess.controls is None or len(self.initial_guess.controls) != len(
                self.collocation_points_per_interval
            ):
                is_complete = False
            if self._num_integrals > 0 and self.initial_guess.integrals is None:
                is_complete = False

            if is_complete:
                source = "complete_user_provided"

        return SolverInputSummary(
            mesh_intervals=len(self.collocation_points_per_interval),
            polynomial_degrees=self.collocation_points_per_interval.copy(),
            mesh_points=self.global_normalized_mesh_nodes.copy()
            if self.global_normalized_mesh_nodes is not None
            else np.array([]),
            initial_guess_source=source,
            states_guess_shapes=states_shapes,
            controls_guess_shapes=controls_shapes,
            initial_time_guess=initial_time_guess,
            terminal_time_guess=terminal_time_guess,
            integrals_guess_length=integrals_length,
        )

    # Conversion methods for solver compatibility

    def get_dynamics_function(self) -> _DynamicsCallback:
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
        time = self._sym_time if self._sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
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
            states_vec: CasadiMX,
            controls_vec: CasadiMX,
            time: CasadiMX,
            params: ProblemParameters,
        ) -> list[CasadiMX]:
            # Extract parameter values in correct order
            param_values: list[float] = []
            for name in self._sym_parameters:
                value = params.get(name, 0.0)
                if not isinstance(value, int | float):
                    value = 0.0
                param_values.append(float(value))

            param_vec = ca.DM(param_values) if param_values else ca.DM()

            # Call CasADi function and convert to list
            result = dynamics_func(states_vec, controls_vec, time, param_vec)
            if result is None:
                return []

            # Handle CasADi function result properly
            # CasADi functions return a list/tuple, we want the first element which is the output vector
            dynamics_output = result[0] if isinstance(result, list | tuple) else result

            # Convert to list of MX elements
            num_states = int(dynamics_output.size1())  # type: ignore[attr-defined]
            result_list: list[CasadiMX] = []
            for i in range(num_states):
                result_list.append(cast(CasadiMX, dynamics_output[i]))

            return result_list

        return vectorized_dynamics

    def get_objective_function(self) -> _ObjectiveCallback:
        if self._objective_expression is None:
            raise ValueError("Objective expression not defined")

        # Gather symbols in order
        state_syms = [
            self._sym_states[name]
            for name in sorted(self._sym_states.keys(), key=lambda n: self._states[n]["index"])
        ]

        # Create inputs for the function
        t0 = self._sym_time_initial if self._sym_time_initial is not None else ca.MX.sym("t0", 1)  # type: ignore[arg-type]
        tf = self._sym_time_final if self._sym_time_final is not None else ca.MX.sym("tf", 1)  # type: ignore[arg-type]
        x0_vec = ca.vertcat(*[ca.MX.sym(f"x0_{i}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]
        xf_vec = ca.vertcat(*[ca.MX.sym(f"xf_{i}", 1) for i in range(len(state_syms))])  # type: ignore[arg-type]
        q = ca.vertcat(*self._integral_symbols) if self._integral_symbols else ca.MX.sym("q", 1)  # type: ignore[arg-type]
        param_syms = (
            ca.vertcat(*self._sym_parameters.values())
            if self._sym_parameters
            else ca.MX.sym("p", 0)  # type: ignore[arg-type]
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
            t0: CasadiMX,
            tf: CasadiMX,
            x0_vec: CasadiMX,
            xf_vec: CasadiMX,
            q: CasadiMX | None,
            params: ProblemParameters,
        ) -> CasadiMX:
            # Extract parameter values in correct order
            param_values = []
            for name in self._sym_parameters:
                param_val = params.get(name, 0.0)
                try:
                    param_values.append(float(param_val))
                except (TypeError, ValueError):
                    param_values.append(0.0)

            param_vec = ca.DM(param_values) if param_values else ca.DM()

            # If q is None but we need it, create zero vector
            q_val = q if q is not None else ca.DM.zeros(len(self._integral_symbols), 1)  # type: ignore[arg-type]

            # Call CasADi function
            result = obj_func(t0, tf, x0_vec, xf_vec, q_val, param_vec)
            # Handle CasADi function result properly
            obj_output = result[0] if isinstance(result, list | tuple) else result
            return cast(CasadiMX, obj_output)

        return vectorized_objective

    def get_integrand_function(self) -> _IntegrandCallback | None:
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
        time = self._sym_time if self._sym_time is not None else ca.MX.sym("t", 1)  # type: ignore[arg-type]
        param_syms = (
            ca.vertcat(*self._sym_parameters.values())
            if self._sym_parameters
            else ca.MX.sym("p", 0)  # type: ignore[arg-type]
        )

        # Create separate CasADi functions for each integrand
        integrand_funcs = []
        for expr in self._integral_expressions:
            integrand_funcs.append(
                ca.Function("integrand", [states_vec, controls_vec, time, param_syms], [expr])
            )

        # Create wrapper function that matches existing API
        def vectorized_integrand(
            states_vec: CasadiMX,
            controls_vec: CasadiMX,
            time: CasadiMX,
            integral_idx: int,
            params: ProblemParameters,
        ) -> CasadiMX:
            if integral_idx >= len(integrand_funcs):
                return ca.MX(0.0)

            # Extract parameter values in correct order
            param_values = []
            for name in self._sym_parameters:
                param_values.append(params.get(name, 0.0))

            param_vec = ca.DM(param_values) if param_values else ca.DM()

            # Call appropriate CasADi function
            result = integrand_funcs[integral_idx](states_vec, controls_vec, time, param_vec)
            # Handle CasADi function result properly
            integrand_output = result[0] if isinstance(result, list | tuple) else result
            return cast(CasadiMX, integrand_output)

        return vectorized_integrand

    def _is_path_constraint(self, expr: SymExpr) -> bool:
        # Path constraints only depend on states, controls and time (t)
        # Not on initial/final specific values (t0/tf)
        depends_on_t0_tf = (
            self._sym_time_initial is not None and ca.depends_on(expr, self._sym_time_initial)  # type: ignore[attr-defined]
        ) or (self._sym_time_final is not None and ca.depends_on(expr, self._sym_time_final))  # type: ignore[attr-defined]

        return not depends_on_t0_tf

    def _symbolic_constraint_to_path_constraint(self, expr: SymExpr) -> PathConstraint:
        # Handle equality constraints: expr == value
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_EQ", "eq"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return PathConstraint(val=lhs - rhs, equals=0.0)

        # Handle inequality constraints: expr <= value or expr >= value
        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_LE", "le"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            # Reformulate as lhs - rhs <= 0 to handle symbolic rhs
            return PathConstraint(val=lhs - rhs, max_val=0.0)

        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_GE", "ge"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            # Reformulate as lhs - rhs >= 0 to handle symbolic rhs
            return PathConstraint(val=lhs - rhs, min_val=0.0)

        # Default case
        return PathConstraint(val=expr, equals=0.0)

    def _symbolic_constraint_to_event_constraint(self, expr: SymExpr) -> EventConstraint:
        # Handle equality constraints: expr == value
        if (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_EQ", "eq"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            return EventConstraint(val=lhs - rhs, equals=0.0)

        # Handle inequality constraints: expr <= value or expr >= value
        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_LE", "le"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            # Reformulate as lhs - rhs <= 0 to handle symbolic rhs
            return EventConstraint(val=lhs - rhs, max_val=0.0)

        elif (
            isinstance(expr, ca.MX)
            and hasattr(expr, "is_op")
            and expr.is_op(getattr(ca, "OP_GE", "ge"))
        ):
            lhs = expr.dep(0)
            rhs = expr.dep(1)
            # Reformulate as lhs - rhs >= 0 to handle symbolic rhs
            return EventConstraint(val=lhs - rhs, min_val=0.0)

        # Default case
        return EventConstraint(val=expr, equals=0.0)

    def get_path_constraints_function(self) -> _PathConstraintsCallback | None:
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
            states_vec: CasadiMX,
            controls_vec: CasadiMX,
            time: CasadiMX,
            params: ProblemParameters,
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
        return any(
            s.get("initial_constraint") is not None or s.get("final_constraint") is not None
            for s in self._states.values()
        )

    def get_event_constraints_function(self) -> _EventConstraintsCallback | None:
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
            t0: CasadiMX,
            tf: CasadiMX,
            x0_vec: CasadiMX,
            xf_vec: CasadiMX,
            q: CasadiMX | None,
            params: ProblemParameters,
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
