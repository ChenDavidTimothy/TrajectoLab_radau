import csv

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

# Declare model variables (symbolic)
x1_sym = ca.SX.sym("x1_sym")  # pos
x2_sym = ca.SX.sym("x2_sym")  # speed
x_sym = ca.vertcat(x1_sym, x2_sym)
u_sym = ca.SX.sym("u_sym")  # throttle

# Model equations
# dx1/dt = x2 (pos_dot = speed)
# dx2/dt = u - x2 (speed_dot = throttle - speed)
xdot_sym = ca.vertcat(x2_sym, u_sym - x2_sym)

# Objective term (Lagrangian) L. For minimum time, L=1
L_sym = 1.0

# Continuous time dynamics function
f = ca.Function("f", [x_sym, u_sym], [xdot_sym, L_sym], ["x", "u"], ["xdot", "L"])

# Control discretization
N = 100  # number of control intervals

# Create an optimization instance
opti = ca.Opti()

# --- Decision Variables ---
# States at control interval boundaries
X = opti.variable(2, N + 1)  # Two states: pos, speed
pos = X[0, :]
speed = X[1, :]

# Control input at each interval (piecewise constant)
U = opti.variable(1, N)  # One control: throttle

# Final time T (make it a variable)
T_var = opti.variable()
opti.subject_to(T_var >= 0.01)  # Time must be positive (small lower bound for stability)
h = T_var / N  # Length of a control interval (now depends on T_var)


# --- Constraints ---
# Initial conditions
opti.subject_to(pos[0] == 0)  # Start at position 0
opti.subject_to(speed[0] == 0)  # ... from stand-still

# Terminal condition
opti.subject_to(pos[N] == 1)  # Finish line at position 1 (Note: X index goes up to N)

# Control bounds
for k in range(N):
    opti.subject_to(opti.bounded(0, U[0, k], 1))  # Control is limited [0,1]


# Path constraints (speed limit)
# Applied at all mesh points X[:,k] for k=0...N
# And at all collocation points Xkj
def limit_func(p):
    return 1 - ca.sin(2 * ca.pi * p) / 2


for k in range(N + 1):  # For X0, X1, ..., XN
    opti.subject_to(speed[k] <= limit_func(pos[k]))  # Track speed limit

# Objective function (Quadrature sum)
J = 0

# Loop over control intervals to set up collocation constraints
for k in range(N):
    # Create variables for states at collocation points within the interval
    Xc_k = []  # List to hold collocation states for interval k
    for j_col in range(d):  # d collocation points per interval (Xc1, Xc2, ..., Xcd)
        Xkj = opti.variable(2)  # State (pos, speed) at collocation point j_col in interval k
        Xc_k.append(Xkj)

        # Apply path constraints (speed limit) at collocation points
        opti.subject_to(Xkj[1] <= limit_func(Xkj[0]))  # Xkj[1] is speed, Xkj[0] is pos

    # Collocation constraints and objective contribution
    Xk_end_poly = D[0] * X[:, k]  # Contribution from X_k (start of interval)

    for j_col in range(1, d + 1):  # Loop over d+1 basis polynomials (index j in B,C,D)
        # Relates to d collocation states Xc_k[0]...Xc_k[d-1]
        # State derivative approximation at collocation point tau_root[j_col]
        xp_approx = C[0, j_col] * X[:, k]  # Term from X_k
        for r_col in range(d):  # Sum over d collocation states Xc_k
            xp_approx = xp_approx + C[r_col + 1, j_col] * Xc_k[r_col]

        # Evaluate dynamics f(x,u) and L(x,u) at Xc_k[j_col-1]
        # Xc_k has d elements (0 to d-1). Collocation points tau_root[1] to tau_root[d]
        # So Xc_k[j_col-1] corresponds to the state at tau_root[j_col]
        fj, qj = f(Xc_k[j_col - 1], U[0, k])  # U[0,k] because U is (1,N)

        # Collocation equations: h * f(Xc, U) = polynomial_derivative_at_Xc
        opti.subject_to(h * fj == xp_approx)

        # Contribution to state at the end of the interval using polynomial
        Xk_end_poly = Xk_end_poly + D[j_col] * Xc_k[j_col - 1]

        # Contribution to quadrature (objective)
        J = J + B[j_col] * qj * h

    # Continuity constraint: state at end of interval k via polynomial must equal X[:, k+1]
    opti.subject_to(X[:, k + 1] == Xk_end_poly)

# Set objective to minimize
opti.minimize(J)  # J represents T_final because L=1

# --- Initial values for solver (optional, but can help) ---
opti.set_initial(T_var, 1.5)  # Initial guess for final time (original was 1)
# Initial guess for speed: a simple ramp then constant (can be refined)
initial_speed_profile = np.linspace(0, 1, N + 1)  # Simple linear increase
opti.set_initial(speed, initial_speed_profile)
opti.set_initial(pos, np.linspace(0, 1, N + 1))  # Linear guess for position
opti.set_initial(U, 0.5)  # Initial guess for control

# Set options and solve
opti.solver("ipopt")
try:
    sol = opti.solve()
except RuntimeError as e:
    print(f"Solver failed: {e}")
    print("Try adjusting initial guesses or problem parameters.")
    # You could inspect opti.debug.value(variable) here for failed step
    exit()


# --- Post-processing and Plotting ---
T_opt = sol.value(T_var)
x_opt = sol.value(X)
u_opt = sol.value(U)  # u_opt will likely be 1D array of shape (N,)

print(f"Optimal Time T: {T_opt}")

# Time grid for plotting
tgrid = np.linspace(0, T_opt, N + 1)
# tgrid_u = np.linspace(0, T_opt, N) # Not strictly needed if using step plot correctly

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(tgrid, x_opt[0, :], "-", label="Position (pos)")
plt.plot(tgrid, x_opt[1, :], "--", label="Speed")
# Plot speed limit based on solved positions
# Evaluate limit_func with CasADi symbolic variables before solving, or use numerical values
# For plotting, it's often easier to re-evaluate with numpy if limit_func is simple
# Or, if limit_func was defined with CasADi symbols, sol.value(limit_func(X[0,:])) is fine.
pos_values_for_limit = x_opt[0, :]  # Use the solved numerical position values
speed_limit_values = 1 - np.sin(2 * np.pi * pos_values_for_limit) / 2
plt.plot(tgrid, speed_limit_values, "r:", label="Speed Limit (evaluated)")
plt.xlabel("Time (s)")
plt.ylabel("States")
plt.legend()
plt.grid(True)
plt.title(f"Optimal Trajectory (Direct Collocation) - Minimized Time: {T_opt:.3f}s")

plt.subplot(2, 1, 2)
# Use step plot for piecewise constant control
# To align, u_opt has N values. tgrid has N+1 values.
# np.append u_opt with its last value to make it N+1 for step 'post'
u_plot = np.append(u_opt, u_opt[-1])  # CORRECTED: u_opt is 1D
plt.step(tgrid, u_plot, "-.", where="post", label="Throttle (U)")
plt.xlabel("Time (s)")
plt.ylabel("Control")
plt.ylim([-0.1, 1.1])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Prepare data for CSV ---
csv_data = []
csv_data.append(["Time", "Position (x1)", "Speed (x2)", "Throttle (u)"])  # Header

# Add data rows
for i in range(N):
    csv_data.append([tgrid[i], x_opt[0, i], x_opt[1, i], u_opt[i]])  # CORRECTED: u_opt[i]
# Add final state (control at N-1 is the last active control for interval N-1 to N)
csv_data.append(
    [tgrid[N], x_opt[0, N], x_opt[1, N], u_opt[N - 1] if N > 0 else None]
)  # Show last applied control

# Write to CSV file
csv_filename = "car_race_collocation_results.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"Results exported to '{csv_filename}'")
