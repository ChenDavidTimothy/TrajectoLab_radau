import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, "legendre"))

# Coefficients of the collocation equation
C = np.zeros((d + 1, d + 1))

# Coefficients of the continuity equation
D = np.zeros(d + 1)

# Coefficients of the quadrature function
B = np.zeros(d + 1)

# Construct polynomial basis
for j in range(d + 1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d + 1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d + 1):
        C[j, r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)

# Time horizon
T = 1.0

# Declare model variables
y = ca.SX.sym("y", 10)  # 10 states # type: ignore[arg-type]
u = ca.SX.sym("u", 4)  # 4 controls # type: ignore[arg-type]

# Parameters
cx = 0.5
rx = 0.1
ux = 2.0
cz = 0.1
uz = 0.1

# Helper functions
E = ca.exp(-(((y[0] - cx) / rx) ** 2))
Rx = -ux * E * (y[0] - cx) * ((y[2] - cz) / cz) ** 2
Rz = -uz * E * ((y[2] - cz) / cz) ** 2

# Model equations (underwater vehicle dynamics)
ydot = ca.vertcat(
    y[6] * ca.cos(y[5]) * ca.cos(y[4]) + Rx,  # y1dot
    y[6] * ca.sin(y[5]) * ca.cos(y[4]),  # y2dot
    -y[6] * ca.sin(y[4]) + Rz,  # y3dot
    y[7] + y[8] * ca.sin(y[3]) * ca.tan(y[4]) + y[9] * ca.cos(y[3]) * ca.tan(y[4]),  # y4dot
    y[8] * ca.cos(y[3]) - y[9] * ca.sin(y[3]),  # y5dot
    (y[8] * ca.sin(y[3]) + y[9] * ca.cos(y[3])) / ca.cos(y[4]),  # y6dot
    u[0],  # y7dot
    u[1],  # y8dot
    u[2],  # y9dot
    u[3],  # y10dot
)

# Objective term (minimize control energy)
L = u[0] ** 2 + u[1] ** 2 + u[2] ** 2 + u[3] ** 2

# Continuous time dynamics
f = ca.Function("f", [y, u], [ydot, L], ["y", "u"], ["ydot", "L"])

# Control discretization
N = 150  # number of control intervals
h = T / N

# Create an optimization instance
opti = ca.Opti()

# Create variables for states at all N+1 time steps
Y = opti.variable(10, N + 1)
U = opti.variable(4, N)  # Control inputs at each interval

# Set initial conditions
opti.subject_to(Y[:, 0] == [0, 0, 0.2, np.pi / 2, 0.1, -np.pi / 4, 1, 0, 0.5, 0.1])

# Set final conditions
opti.subject_to(Y[:, N] == [1, 0.5, 0, np.pi / 2, 0, 0, 0, 0, 0, 0])

# Control bounds
for k in range(N):
    for i in range(4):
        opti.subject_to(U[i, k] >= -15)
        opti.subject_to(U[i, k] <= 15)

# Path constraints (y4 bounded)
for k in range(N + 1):
    opti.subject_to(Y[3, k] >= np.pi / 2 - 0.02)  # y4 >= π/2 - 0.02
    opti.subject_to(Y[3, k] <= np.pi / 2 + 0.02)  # y4 <= π/2 + 0.02

# Objective function
J = 0

# Loop over control intervals
for k in range(N):
    # Create variables for collocation states
    Yc = []
    for j in range(d):
        Ykj = opti.variable(10)
        Yc.append(Ykj)
        # Apply path constraints to collocation points
        opti.subject_to(Ykj[3] >= np.pi / 2 - 0.02)
        opti.subject_to(Ykj[3] <= np.pi / 2 + 0.02)

    # Collocation constraints and objective contribution
    Yk_end = D[0] * Y[:, k]
    for j in range(1, d + 1):
        # State derivative at the collocation point
        yp = C[0, j] * Y[:, k]
        for r in range(d):
            yp = yp + C[r + 1, j] * Yc[r]

        # Collocation equations
        fj, qj = f(Yc[j - 1], U[:, k])
        opti.subject_to(h * fj == yp)

        # Contribution to end state
        Yk_end = Yk_end + D[j] * Yc[j - 1]

        # Contribution to quadrature function
        J = J + B[j] * qj * h

    # Continuity constraint
    opti.subject_to(Y[:, k + 1] == Yk_end)

# Set objective
opti.minimize(J)

# Set options and solve
opti.solver("ipopt")
sol = opti.solve()

# Extract solution
y_opt = sol.value(Y)
u_opt = sol.value(U)

# Literature comparison
J_literature = 236.527851
J_computed = sol.value(J)
print(f"Literature solution: J* = {J_literature}")
print(f"Computed solution:   J* = {J_computed:.6f}")
print(f"Relative error: {abs(J_computed - J_literature) / J_literature * 100:.4f}%")

# Plot the results
tgrid = np.linspace(0, T, N + 1)
tgrid_u = np.linspace(0, T, N)

plt.figure(figsize=(15, 10))

# States plot
plt.subplot(3, 2, 1)
for i in range(6):
    plt.plot(tgrid, y_opt[i, :], label=f"y{i + 1}")
plt.xlabel("Time")
plt.ylabel("States y1-y6")
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
for i in range(6, 10):
    plt.plot(tgrid, y_opt[i, :], label=f"y{i + 1}")
plt.xlabel("Time")
plt.ylabel("States y7-y10")
plt.legend()
plt.grid()

# Controls plot
plt.subplot(3, 2, 3)
for i in range(4):
    plt.step(tgrid_u, u_opt[i, :], where="post", label=f"u{i + 1}")
plt.xlabel("Time")
plt.ylabel("Controls")
plt.legend()
plt.grid()

# Position trajectory (y1, y2, y3)
plt.subplot(3, 2, 4)
plt.plot(y_opt[0, :], y_opt[1, :], "b-", linewidth=2)
plt.plot(y_opt[0, 0], y_opt[1, 0], "go", markersize=8, label="Start")
plt.plot(y_opt[0, -1], y_opt[1, -1], "ro", markersize=8, label="End")
plt.xlabel("y1 (x-position)")
plt.ylabel("y2 (y-position)")
plt.title("Trajectory (x-y plane)")
plt.legend()
plt.grid()

# Depth profile
plt.subplot(3, 2, 5)
plt.plot(tgrid, y_opt[2, :], "b-", linewidth=2)
plt.xlabel("Time")
plt.ylabel("y3 (depth)")
plt.title("Depth Profile")
plt.grid()

# Attitude angles
plt.subplot(3, 2, 6)
plt.plot(tgrid, y_opt[3, :], label="y4 (pitch)")
plt.plot(tgrid, y_opt[4, :], label="y5 (roll)")
plt.plot(tgrid, y_opt[5, :], label="y6 (yaw)")
plt.axhline(y=np.pi / 2, color="r", linestyle="--", alpha=0.5, label="π/2")
plt.xlabel("Time")
plt.ylabel("Attitude angles")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("Underwater vehicle optimal control problem solved successfully!")
