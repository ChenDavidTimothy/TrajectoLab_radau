import sympy as sm
import sympy.physics.mechanics as me

from maptor.mechanics.symbolic import _sympy_to_casadi_string


# === Define Symbols ===
m1, m2 = sm.symbols("m1 m2")
l1, l2 = sm.symbols("l1 l2")
lc1, lc2 = sm.symbols("lc1 lc2")
I1, I2 = sm.symbols("I1 I2")
g = sm.symbols("g")
tau1, tau2 = sm.symbols("tau1 tau2")

q1, q2 = me.dynamicsymbols("q1 q2")
q1d, q2d = me.dynamicsymbols("q1 q2", 1)


# === Reference Frames ===
N = me.ReferenceFrame("N")
A = N.orientnew("A", "Axis", (q1, N.z))
B = A.orientnew("B", "Axis", (q2, A.z))


# === Points and Velocities ===
O = me.Point("O")
O.set_vel(N, 0)

P1 = O.locatenew("P1", l1 * A.x)
P1.v2pt_theory(O, N, A)

G1 = O.locatenew("G1", lc1 * A.x)
G1.v2pt_theory(O, N, A)

G2 = P1.locatenew("G2", lc2 * B.x)
G2.v2pt_theory(P1, N, B)


# === Rigid Bodies ===
I1_dyadic = I1 * me.inertia(A, 0, 0, 1)
link1_body = me.RigidBody("link1", G1, A, m1, (I1_dyadic, G1))

I2_dyadic = I2 * me.inertia(B, 0, 0, 1)
link2_body = me.RigidBody("link2", G2, B, m2, (I2_dyadic, G2))


# === Lagrangian Mechanics ===
loads = [(G1, -m1 * g * N.y), (G2, -m2 * g * N.y)]

L = me.Lagrangian(N, link1_body, link2_body)
LM = me.LagrangesMethod(L, [q1, q2], forcelist=loads, frame=N)
LM.form_lagranges_equations()

# Extract dynamics components
mass_matrix_inv = LM.mass_matrix.inv()
passive_dynamics = mass_matrix_inv * LM.forcing
control_coupling = mass_matrix_inv * sm.Matrix([tau1, tau2])
total_dynamics = passive_dynamics + control_coupling

# Convert to CasADi format
coordinates = [q1, q2]
velocities = [q1d, q2d]
first_order_system = velocities + [total_dynamics[0], total_dynamics[1]]

casadi_equations = _sympy_to_casadi_string(first_order_system)
coordinate_names = _sympy_to_casadi_string(coordinates)
state_names = coordinate_names + [name + "_dot" for name in coordinate_names]

print("MAPTOR Dynamics (General 2DOF Manipulator):")
print("=" * 50)
print("# State variables:")
for name in state_names:
    print(f"# {name} = phase.state('{name}')")
print("\n# Control variables:")
print("# tau1 = phase.control('tau1')")
print("# tau2 = phase.control('tau2')")
print("\n# Dynamics:")
print("phase.dynamics({")
for name, eq_str in zip(state_names, casadi_equations, strict=False):
    print(f"    {name}: {eq_str},")
print("})")

# Output:
# State variables:
# q1 = phase.state('q1')
# q2 = phase.state('q2')
# q1_dot = phase.state('q1_dot')
# q2_dot = phase.state('q2_dot')

# Control variables:
# tau1 = phase.control('tau1')
# tau2 = phase.control('tau2')

# Dynamics:
# phase.dynamics({
#    q1: q1_dot,
#    q2: q2_dot,
#    q1_dot: tau1*(-I2 - lc2**2*m2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2) + tau2*(I2 + l1*lc2*m2*ca.cos(q2) + lc2**2*m2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2) + (-I2 - lc2**2*m2)*(-g*l1*m2*ca.cos(q1) - g*lc1*m1*ca.cos(q1) - g*lc2*m2*(-ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) - m2*(-2*l1*lc2*(q1_dot + q2_dot)*ca.sin(q2)*q2_dot - 2*l1*lc2*ca.sin(q2)*q1_dot*q2_dot)/2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2) + (I2 + l1*lc2*m2*ca.cos(q2) + lc2**2*m2)*(-g*lc2*m2*(-ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) - l1*lc2*m2*(q1_dot + q2_dot)*ca.sin(q2)*q1_dot + l1*lc2*m2*ca.sin(q2)*q1_dot*q2_dot)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2),
#    q2_dot: tau1*(I2 + l1*lc2*m2*ca.cos(q2) + lc2**2*m2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2) + tau2*(-I1 - I2 - l1**2*m2 - 2*l1*lc2*m2*ca.cos(q2) - lc1**2*m1 - lc2**2*m2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2) + (I2 + l1*lc2*m2*ca.cos(q2) + lc2**2*m2)*(-g*l1*m2*ca.cos(q1) - g*lc1*m1*ca.cos(q1) - g*lc2*m2*(-ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) - m2*(-2*l1*lc2*(q1_dot + q2_dot)*ca.sin(q2)*q2_dot - 2*l1*lc2*ca.sin(q2)*q1_dot*q2_dot)/2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2) + (-g*lc2*m2*(-ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) - l1*lc2*m2*(q1_dot + q2_dot)*ca.sin(q2)*q1_dot + l1*lc2*m2*ca.sin(q2)*q1_dot*q2_dot)*(-I1 - I2 - l1**2*m2 - 2*l1*lc2*m2*ca.cos(q2) - lc1**2*m1 - lc2**2*m2)/(-I1*I2 - I1*lc2**2*m2 - I2*l1**2*m2 - I2*lc1**2*m1 + l1**2*lc2**2*m2**2*ca.cos(q2)**2 - l1**2*lc2**2*m2**2 - lc1**2*lc2**2*m1*m2),
# })
