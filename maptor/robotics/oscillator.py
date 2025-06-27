import sympy as sm
import sympy.physics.mechanics as me
from symbolic import sympy_to_casadi_string


# Initialize pretty printing
me.init_vprinting()

# === Define Symbols ===
# System parameters
M, m, l, k1, k2, c1, g, h, w, d, r = sm.symbols("M m l k1 k2 c1 g h w d r")

# Generalized coordinates and their derivatives
q1, q2 = me.dynamicsymbols("q1 q2")  # q1: block position, q2: pendulum angle
q1d, q2d = me.dynamicsymbols("q1 q2", 1)  # first derivatives

# === Reference Frames ===
N = me.ReferenceFrame("N")  # Inertial frame
B = N.orientnew("B", "Axis", (q2, N.z))  # Pendulum frame rotated by q2

# === Points and Velocities ===
O = me.Point("O")  # Fixed origin
O.set_vel(N, 0)

block_point = O.locatenew("block", q1 * N.y)
block_point.set_vel(N, q1d * N.y)

pendulum_point = block_point.locatenew("pendulum", l * B.y)
pendulum_point.v2pt_theory(block_point, N, B)

# === Inertia and Rigid Bodies ===
# Inertia tensor for the block (approximate as a cuboid)
I_block = (M / 12) * me.inertia(N, h**2 + d**2, w**2 + d**2, w**2 + h**2)
block_body = me.RigidBody("block", block_point, N, M, (I_block, block_point))

# Inertia tensor for the pendulum (approximate as a sphere)
I_pendulum = (2 * m * r**2 / 5) * me.inertia(B, 1, 0, 1)
pendulum_body = me.RigidBody("pendulum", pendulum_point, B, m, (I_pendulum, pendulum_point))

# === Forces: Duffing Spring and Damper ===
path = me.LinearPathway(O, block_point)
spring = me.DuffingSpring(k1, k2, path, 0)
damper = me.LinearDamper(c1, path)
loads = spring.to_loads() + damper.to_loads()

# Add gravitational forces
bodies = [block_body, pendulum_body]
for body in bodies:
    loads.append((body.masscenter, body.mass * g * N.y))

# === Lagrangian Mechanics ===
L = me.Lagrangian(N, block_body, pendulum_body)

# Apply Lagrange's method
LM = me.LagrangesMethod(L, [q1, q2], forcelist=loads, frame=N)
eom = sm.simplify(LM.form_lagranges_equations())

# === Display Result ===
print("Equations of Motion (Simplified):")
sm.pprint(eom, use_unicode=True)


if __name__ == "__main__":
    # After your mechanics code, add these 3 lines:

    casadi_strings = sympy_to_casadi_string(eom)

    print("\nCasADi Equation Strings:")
    print("=" * 50)
    for i, eq_str in enumerate(casadi_strings):
        print(f"f{i + 1} = {eq_str}")
