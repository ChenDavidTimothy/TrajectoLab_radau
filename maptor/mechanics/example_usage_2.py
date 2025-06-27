import sympy as sm
import sympy.physics.mechanics as me
from symbolic import lagrangian_to_maptor_dynamics


# Initialize pretty printing
me.init_vprinting()

# === Define Symbols ===
M, m, l, k1, k2, c1, g, h, w, d, r = sm.symbols("M m l k1 k2 c1 g h w d r")
q1, q2 = me.dynamicsymbols("q1 q2")
q1d, q2d = me.dynamicsymbols("q1 q2", 1)

# === Reference Frames ===
N = me.ReferenceFrame("N")
B = N.orientnew("B", "Axis", (q2, N.z))

# === Points and Velocities ===
O = me.Point("O")
O.set_vel(N, 0)

block_point = O.locatenew("block", q1 * N.y)
block_point.set_vel(N, q1d * N.y)

pendulum_point = block_point.locatenew("pendulum", l * B.y)
pendulum_point.v2pt_theory(block_point, N, B)

# === Inertia and Rigid Bodies ===
I_block = (M / 12) * me.inertia(N, h**2 + d**2, w**2 + d**2, w**2 + h**2)
block_body = me.RigidBody("block", block_point, N, M, (I_block, block_point))

I_pendulum = (2 * m * r**2 / 5) * me.inertia(B, 1, 0, 1)
pendulum_body = me.RigidBody("pendulum", pendulum_point, B, m, (I_pendulum, pendulum_point))

# === Forces ===
path = me.LinearPathway(O, block_point)
spring = me.DuffingSpring(k1, k2, path, 0)
damper = me.LinearDamper(c1, path)
loads = spring.to_loads() + damper.to_loads()

bodies = [block_body, pendulum_body]
for body in bodies:
    loads.append((body.masscenter, body.mass * g * N.y))

# === Lagrangian Mechanics ===
L = me.Lagrangian(N, block_body, pendulum_body)
LM = me.LagrangesMethod(L, [q1, q2], forcelist=loads, frame=N)

# === Convert to MAPTOR Format ===
lagrangian_to_maptor_dynamics(LM, [q1, q2])
