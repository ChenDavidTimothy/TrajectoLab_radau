import re

import sympy as sm
import sympy.physics.mechanics as me


def _sympy_to_casadi_string(expressions):
    # Handle single expression case
    if not isinstance(expressions, (list, tuple)):
        expressions = [expressions]

    # Function mappings
    functions = {
        "atan2": "ca.atan2",
        "sqrt": "ca.sqrt",
        "sin": "ca.sin",
        "cos": "ca.cos",
        "tan": "ca.tan",
        "exp": "ca.exp",
        "log": "ca.log",
        "Abs": "ca.fabs",
        "asin": "ca.asin",
        "acos": "ca.acos",
        "atan": "ca.atan",
        "sinh": "ca.sinh",
        "cosh": "ca.cosh",
        "tanh": "ca.tanh",
    }

    # Derivative patterns - order matters!
    patterns = [
        # Handle second derivatives first
        (re.compile(r"Derivative\(([^,\(\)]+)\(t\),\s*\(t,\s*2\)\)"), r"\1_ddot"),
        # Handle first derivatives
        (re.compile(r"Derivative\(([^,\(\)]+)\(t\),\s*t\)"), r"\1_dot"),
        # Handle alternative derivative notations
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)ddot\b"), r"\1_ddot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)dot\b"), r"\1_dot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)dd\b"), r"\1_ddot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)d\b"), r"\1_dot"),
        # Remove (t) from base variables - MUST BE LAST
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)\(t\)"), r"\1"),
    ]

    func_pattern = re.compile(r"\b(" + "|".join(re.escape(f) for f in functions.keys()) + r")\b")

    def _convert_single(expr):
        expr_str = str(expr)

        # Convert derivatives
        for pattern, replacement in patterns:
            expr_str = pattern.sub(replacement, expr_str)

        # Convert functions
        expr_str = func_pattern.sub(lambda m: functions[m.group(1)], expr_str)

        return expr_str

    # Convert all expressions
    converted_expressions = []
    for expr in expressions:
        converted = _convert_single(expr)

        # Post-process Matrix format if present
        if converted.startswith("Matrix([") and converted.endswith("])"):
            # Extract equations from Matrix([[eq1], [eq2]]) format
            inner = converted[9:-3]  # Remove 'Matrix([[' and ']])'
            equations = inner.split("], [")
            converted_expressions.extend([eq.strip() for eq in equations])
        else:
            converted_expressions.append(converted)

    return converted_expressions


def lagrangian_to_maptor_dynamics(lagranges_method, coordinates):
    """
    Convert SymPy LagrangesMethod to MAPTOR dynamics format.

    Args:
        lagranges_method: SymPy LagrangesMethod object
        coordinates: List of generalized coordinates (e.g., [q1, q2])

    Returns:
        tuple: (casadi_equations, state_names) ready for MAPTOR dynamics
    """
    # Get implicit equations of motion
    eom_implicit = lagranges_method.form_lagranges_equations()

    # Solve for accelerations explicitly
    second_derivatives = [coord.diff(me.dynamicsymbols._t, 2) for coord in coordinates]
    accelerations = sm.solve(eom_implicit, second_derivatives)

    # Extract and simplify explicit accelerations
    explicit_accelerations = []
    for sd in second_derivatives:
        explicit_accelerations.append(sm.simplify(accelerations[sd]))

    # Create first-order system: [q1_dot, q2_dot, ..., q1_ddot, q2_ddot, ...]
    first_derivatives = [coord.diff(me.dynamicsymbols._t) for coord in coordinates]
    first_order_system = first_derivatives + explicit_accelerations

    # Convert to CasADi syntax
    casadi_equations = _sympy_to_casadi_string(first_order_system)

    # Generate state names: [q1, q2, ..., q1_dot, q2_dot, ...]
    coordinate_names = [str(coord).replace("(t)", "") for coord in coordinates]
    state_names = coordinate_names + [name + "_dot" for name in coordinate_names]

    return casadi_equations, state_names


def print_maptor_dynamics(casadi_equations, state_names):
    """Print MAPTOR dynamics in copy-paste ready format."""
    print("CasADi MAPTOR Dynamics:")
    print("=" * 60)

    print("# State variables:")
    for name in state_names:
        print(f"# {name} = phase.state('{name}')")

    print("\n# MAPTOR dynamics dictionary:")
    print("phase.dynamics({")
    for name, eq_str in zip(state_names, casadi_equations, strict=False):
        print(f"    {name}: {eq_str},")
    print("})")
