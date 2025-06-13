import sympy as sp


# Set up pretty printing for the console output
sp.init_printing(use_unicode=True)

# --- 1. Define all symbols from the text ---

# State variables
v, beta, gamma, psi, p = sp.symbols("v beta gamma psi p")

# State derivatives (the variables we want to solve for)
v_dot, beta_dot, gamma_dot, p_dot = sp.symbols("v_dot beta_dot gamma_dot p_dot")

# Control variables (only delta_f appears in the dynamics equations)
delta_f = sp.symbols("delta_f")

# Vehicle parameters
m, m_b, h_b, l_f, l_r = sp.symbols("m m_b h_b l_f l_r")
I_z, I_x, I_xz = sp.symbols("I_z I_x I_xz")
K_phi, C_phi = sp.symbols("K_phi C_phi")
g = sp.symbols("g")

# External forces (treated as known inputs for this algebraic manipulation)
F_xf, F_yf, F_xr, F_yr, F_d = sp.symbols("F_xf F_yf F_xr F_yr F_d")


# --- 2. Transcribe the original coupled equations (1-4) ---
# The equations are rearranged into the form: expression = 0

# Equation (1): Longitudinal dynamics
eq1 = sp.Eq(
    m * v_dot * sp.cos(beta) - m * v * sp.sin(beta) * (gamma + beta_dot) + m_b * h_b * p * gamma,
    F_xf * sp.cos(delta_f) - F_yf * sp.sin(delta_f) + F_xr - F_d,
)

# Equation (2): Lateral dynamics
eq2 = sp.Eq(
    m * v_dot * sp.sin(beta) + m * v * sp.cos(beta) * (gamma + beta_dot) - m_b * h_b * p_dot,
    F_xf * sp.sin(delta_f) + F_yf * sp.cos(delta_f) + F_yr,
)

# Equation (3): Yaw dynamics
eq3 = sp.Eq(
    I_z * gamma_dot - I_xz * p_dot,
    l_f * (F_xf * sp.sin(delta_f) + F_yf * sp.cos(delta_f)) - l_r * F_yr,
)

# Equation (4): Roll dynamics
eq4 = sp.Eq(
    I_x * p_dot
    - I_xz * gamma_dot
    - m_b * h_b * (v_dot * sp.sin(beta) + v * sp.cos(beta) * (gamma + beta_dot)),
    m_b * g * h_b * sp.sin(psi) - K_phi * psi - C_phi * p,
)

# --- 3. Solve the system of equations for the state derivatives ---

system_of_equations = [eq1, eq2, eq3, eq4]
derivatives_to_solve = [v_dot, beta_dot, gamma_dot, p_dot]

solution = sp.solve(system_of_equations, derivatives_to_solve, dict=True)
solution_dict = solution[0]

# --- NEW: 4. Simplify the results ---
# This is the crucial new step. We iterate through the solution dictionary
# and apply the simplify() function to each expression.

print("\n--- Simplifying the results... ---\n")
simplified_solution_dict = {}
for key, value in solution_dict.items():
    simplified_solution_dict[key] = sp.simplify(value)


# --- 5. Display the SIMPLIFIED decoupled state-space equations ---

print("\n--- SIMPLIFIED Decoupled State-Space Equations ---")
print("These are the final, human-readable, and computationally efficient")
print("expressions for the state derivatives.\n")

# Display simplified solution for v_dot
print("1. Solution for v_dot (dv/dt):")
sp.pprint(simplified_solution_dict[v_dot])
print("\nLaTeX format for v_dot:")
print(f"$$ \\dot{{v}} = {sp.latex(simplified_solution_dict[v_dot])} $$")

# Display simplified solution for beta_dot
print("\n2. Solution for beta_dot (d(beta)/dt):")
sp.pprint(simplified_solution_dict[beta_dot])
print("\nLaTeX format for beta_dot:")
print(f"$$ \\dot{{\\beta}} = {sp.latex(simplified_solution_dict[beta_dot])} $$")

# Display simplified solution for gamma_dot
print("\n3. Solution for gamma_dot (d(gamma)/dt):")
sp.pprint(simplified_solution_dict[gamma_dot])
print("\nLaTeX format for gamma_dot:")
print(f"$$ \\dot{{\\gamma}} = {sp.latex(simplified_solution_dict[gamma_dot])} $$")

# The equation for psi_dot is definitional
print("\n4. Solution for psi_dot (d(psi)/dt):")
print("By definition of the state p:")
sp.pprint(p)
print("\nLaTeX format for psi_dot:")
print("$$ \\dot{\\psi} = p $$")

# Display simplified solution for p_dot
print("\n5. Solution for p_dot (dp/dt = d^2(psi)/dt^2):")
sp.pprint(simplified_solution_dict[p_dot])
print("\nLaTeX format for p_dot:")
print(f"$$ \\dot{{p}} = {sp.latex(simplified_solution_dict[p_dot])} $$")
