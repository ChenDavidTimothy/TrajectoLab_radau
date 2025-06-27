import re


def sympy_to_casadi_string(eom):
    """
    Convert SymPy equations to CasADi syntax strings.
    NO CASADI IMPORT REQUIRED - pure string conversion.
    """

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

    # Derivative patterns
    patterns = [
        (re.compile(r"Derivative\(([^,\(\)]+)\(t\),\s*\(t,\s*2\)\)"), r"\1_ddot"),
        (re.compile(r"Derivative\(([^,\(\)]+)\(t\),\s*t\)"), r"\1_dot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)ddot\b"), r"\1_ddot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)dot\b"), r"\1_dot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)dd\b"), r"\1_ddot"),
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)d\b"), r"\1_dot"),
        # Remove (t) from base variables - MUST BE LAST
        (re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)\(t\)"), r"\1"),
    ]

    func_pattern = re.compile(r"\b(" + "|".join(re.escape(f) for f in functions.keys()) + r")\b")

    def convert_single(expr):
        expr_str = str(expr)
        # Convert derivatives
        for pattern, replacement in patterns:
            expr_str = pattern.sub(replacement, expr_str)
        # Convert functions
        expr_str = func_pattern.sub(lambda m: functions[m.group(1)], expr_str)
        return expr_str

    # Convert the expression
    converted = convert_single(eom)

    # Post-process Matrix format if present
    if converted.startswith("Matrix([") and converted.endswith("])"):
        # Extract equations from Matrix([[eq1], [eq2]]) format
        # Remove Matrix([[ and ]])
        inner = converted[9:-3]  # Remove 'Matrix([[' and ']])'

        # Split by '], [' to get individual equations
        equations = inner.split("], [")
        return [eq.strip() for eq in equations]
    else:
        return [converted]
