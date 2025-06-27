import sympy as sm
import sympy.physics.mechanics as me

from maptor.mechanics import lagrangian_to_maptor_dynamics


# Initialize pretty printing
me.init_vprinting()

# === Define Symbols ===
M, m, l, g = sm.symbols("M m l g")  # Cart mass, pole mass, pole length, gravity
F = sm.symbols("F")  # Applied force on cart
x, theta = me.dynamicsymbols("x theta")  # Cart position, pole angle from vertical
xd, thetad = me.dynamicsymbols("x theta", 1)  # First derivatives

# === Reference Frames ===
N = me.ReferenceFrame("N")  # Inertial frame
B = N.orientnew("B", "Axis", (theta, N.z))  # Pole body frame

# === Points and Velocities ===
O = me.Point("O")  # Fixed origin
O.set_vel(N, 0)

# Cart center of mass
cart_point = O.locatenew("cart", x * N.x)
cart_point.set_vel(N, xd * N.x)

# Pole center of mass (l/2 from cart along pole)
pole_point = cart_point.locatenew("pole", (l / 2) * B.y)
pole_point.v2pt_theory(cart_point, N, B)

# === Inertia and Rigid Bodies ===
# Cart: point mass (no rotational inertia about center)
I_cart = me.inertia(N, 0, 0, 0)
cart_body = me.RigidBody("cart", cart_point, N, M, (I_cart, cart_point))

# Pole: uniform rod inertia about center of mass
I_pole_cm = (m * l**2 / 12) * me.inertia(B, 0, 0, 1)
pole_body = me.RigidBody("pole", pole_point, B, m, (I_pole_cm, pole_point))

# === Forces ===
loads = [
    (cart_point, F * N.x),  # Applied force on cart (horizontal)
    (cart_point, -M * g * N.y),  # Gravity on cart
    (pole_point, -m * g * N.y),  # Gravity on pole
]

# === Lagrangian Mechanics ===
L = me.Lagrangian(N, cart_body, pole_body)
LM = me.LagrangesMethod(L, [x, theta], forcelist=loads, frame=N)

# === Convert to MAPTOR Format ===
lagrangian_to_maptor_dynamics(LM, [x, theta])
