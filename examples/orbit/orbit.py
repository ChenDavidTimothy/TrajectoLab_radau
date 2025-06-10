import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import maptor as mtor


# Constants from C++ PSOPT code (unchanged)
ISP = 450.0  # [sec]
MU = 1.407645794e16  # [ft^3/sec^2]
G0 = 32.174  # [ft/sec^2]
T_THRUST = 4.446618e-3  # [lb]
RE = 20925662.73  # [ft]
J2 = 1082.639e-6
J3 = -2.565e-6
J4 = -1.608e-6

# Physical initial/final conditions (unchanged)
PTI = 21837080.052835
FTI = 0.0
GTI = 0.0
HTI = -0.25396764647494
KTI = 0.0
LTI = np.pi
WTI = 1.0

PTF = 40007346.015232
EVENT_FINAL_9 = 0.73550320568829
EVENT_FINAL_10 = 0.61761258786099
EVENT_FINAL_11 = 0.0
EVENT_FINAL_12_LOWER = -10.0
EVENT_FINAL_12_UPPER = 0.0

EQ_TOL = 0.001

# Scaling factors following shuttle_reentry.py pattern
P_SCALE = 1e7  # Position parameter scaling [ft]
L_SCALE = np.pi  # Longitude scaling [rad]
T_SCALE = 1e4  # Time scaling [sec]


def generate_advanced_initial_guess():
    """
    Generate sophisticated initial guess using variable step ODE solver
    following the methodology from Section 5.3 of the reference paper.

    Returns:
        tuple: (states_guess, controls_guess, final_time_guess)
    """

    def legendre_polynomial(x, n):
        """Legendre polynomials P_n(x) for n=2,3,4"""
        if n == 2:
            return 0.5 * (3.0 * x**2 - 1.0)
        elif n == 3:
            return 0.5 * (5.0 * x**3 - 3.0 * x)
        elif n == 4:
            return (1.0 / 8.0) * (35.0 * x**4 - 30.0 * x**2 + 3.0)
        else:
            raise ValueError(f"Legendre polynomial not implemented for n={n}")

    def legendre_polynomial_derivative(x, n):
        """Derivatives of Legendre polynomials P'_n(x) for n=2,3,4"""
        if n == 2:
            return 0.5 * (2.0 * 3.0 * x)
        elif n == 3:
            return 0.5 * (3.0 * 5.0 * x**2 - 3.0)
        elif n == 4:
            return (1.0 / 8.0) * (4.0 * 35.0 * x**3 - 2.0 * 30.0 * x)
        else:
            raise ValueError(f"Legendre polynomial derivative not implemented for n={n}")

    def cross_product(a, b):
        """Cross product of two 3D vectors"""
        return np.array(
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        )

    def dot_product(a, b):
        """Dot product of two 3D vectors"""
        return np.sum(a * b)

    def orbital_dynamics(t, state):
        """
        Modified equinoctial orbital dynamics with J2, J3, J4 perturbations
        and control law u = Q^T * V/||V|| (inertial velocity direction)

        Args:
            t: Time [sec]
            state: [p, f, g, h, k, L, w] in physical units

        Returns:
            State derivatives [pdot, fdot, gdot, hdot, kdot, Ldot, wdot]
        """
        p, f, g, h, k, L, w = state

        # Integration parameters
        tau = -25.0  # Throttle parameter

        # Numerical safeguards
        eps = 1e-12
        p = max(p, eps)
        w = max(w, eps)

        # Dependent variables
        q = 1.0 + f * np.cos(L) + g * np.sin(L)
        q = max(abs(q), eps) * np.sign(q) if q != 0 else eps
        r = p / q
        alpha2 = h * h - k * k
        X = np.sqrt(h * h + k * k)
        s2 = 1 + X * X

        # Position vector
        r1 = r / s2 * (np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L))
        r2 = r / s2 * (np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L))
        r3 = 2 * r / s2 * (h * np.sin(L) - k * np.cos(L))
        rvec = np.array([r1, r2, r3])

        # Velocity vector
        sqrt_mu_p = np.sqrt(MU / max(p, eps))
        v1 = (
            -(1.0 / s2)
            * sqrt_mu_p
            * (
                np.sin(L)
                + alpha2 * np.sin(L)
                - 2 * h * k * np.cos(L)
                + g
                - 2 * f * h * k
                + alpha2 * g
            )
        )
        v2 = (
            -(1.0 / s2)
            * sqrt_mu_p
            * (
                -np.cos(L)
                + alpha2 * np.cos(L)
                + 2 * h * k * np.sin(L)
                - f
                + 2 * g * h * k
                + alpha2 * f
            )
        )
        v3 = (2.0 / s2) * sqrt_mu_p * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
        vvec = np.array([v1, v2, v3])

        # Reference frame construction
        rv = cross_product(rvec, vvec)
        rvr = cross_product(rv, rvec)
        norm_r = np.sqrt(max(dot_product(rvec, rvec), eps))
        norm_rv = np.sqrt(max(dot_product(rv, rv), eps))

        ir = rvec / norm_r
        ith = rvr / (norm_rv * norm_r)
        ih = rv / norm_rv

        # Compute in vector
        en = np.array([0.0, 0.0, 1.0])
        enir = dot_product(en, ir)
        in_vec = en - enir * ir
        norm_in = np.sqrt(max(dot_product(in_vec, in_vec), eps))
        in_normalized = in_vec / norm_in

        # Gravitational perturbations
        r_safe = max(r, RE / 100.0)
        sin_phi = rvec[2] / norm_r
        sin_phi = max(min(sin_phi, 1.0 - eps), -1.0 + eps)
        cos_phi = np.sqrt(1.0 - sin_phi**2)

        deltagn = 0.0
        deltagr = 0.0
        for j in [2, 3, 4]:
            J_coeff = [0, 0, J2, J3, J4][j]
            P_j = legendre_polynomial(sin_phi, j)
            Pdash_j = legendre_polynomial_derivative(sin_phi, j)
            deltagn += -MU * cos_phi / (r_safe * r_safe) * (RE / r_safe) ** j * Pdash_j * J_coeff
            deltagr += -MU / (r_safe * r_safe) * (j + 1) * (RE / r_safe) ** j * P_j * J_coeff

        # Gravitational perturbation vector
        delta_g = deltagn * in_normalized - deltagr * ir

        # Transform to orbital frame
        DELTA_g1 = dot_product(ir, delta_g)
        DELTA_g2 = dot_product(ith, delta_g)
        DELTA_g3 = dot_product(ih, delta_g)

        # Control law: u = Q^T * V/||V|| (inertial velocity direction)
        v_norm = np.sqrt(max(dot_product(vvec, vvec), eps))
        v_unit = vvec / v_norm
        u1 = dot_product(ir, v_unit)
        u2 = dot_product(ith, v_unit)
        u3 = dot_product(ih, v_unit)

        # Thrust acceleration
        DELTA_T1 = G0 * T_THRUST * (1.0 + 0.01 * tau) * u1 / w
        DELTA_T2 = G0 * T_THRUST * (1.0 + 0.01 * tau) * u2 / w
        DELTA_T3 = G0 * T_THRUST * (1.0 + 0.01 * tau) * u3 / w

        # Total acceleration
        delta1 = DELTA_g1 + DELTA_T1
        delta2 = DELTA_g2 + DELTA_T2
        delta3 = DELTA_g3 + DELTA_T3

        # State derivatives (identical to C++ reference)
        sqrt_p_mu = np.sqrt(max(p, eps) / MU)

        pdot = 2 * p / q * sqrt_p_mu * delta2
        fdot = (
            sqrt_p_mu * np.sin(L) * delta1
            + sqrt_p_mu * (1.0 / q) * ((q + 1.0) * np.cos(L) + f) * delta2
            - sqrt_p_mu * (g / q) * (h * np.sin(L) - k * np.cos(L)) * delta3
        )
        gdot = (
            -sqrt_p_mu * np.cos(L) * delta1
            + sqrt_p_mu * (1.0 / q) * ((q + 1.0) * np.sin(L) + g) * delta2
            + sqrt_p_mu * (f / q) * (h * np.sin(L) - k * np.cos(L)) * delta3
        )
        hdot = sqrt_p_mu * s2 * np.cos(L) / (2.0 * q) * delta3
        kdot = sqrt_p_mu * s2 * np.sin(L) / (2.0 * q) * delta3
        Ldot = (
            sqrt_p_mu * (1.0 / q) * (h * np.sin(L) - k * np.cos(L)) * delta3
            + np.sqrt(MU * max(p, eps)) * (q / max(p, eps)) ** 2
        )
        wdot = -T_THRUST * (1.0 + 0.01 * tau) / ISP

        return np.array([pdot, fdot, gdot, hdot, kdot, Ldot, wdot])

    # Initial state vector
    state0 = np.array([PTI, FTI, GTI, HTI, KTI, LTI, WTI])
    t_final = 9e4  # Integration final time [sec]

    print("Integrating orbital dynamics with advanced control law...")

    # Integrate using variable step ODE solver
    sol = solve_ivp(
        orbital_dynamics,
        [0, t_final],
        state0,
        method="DOP853",  # High-order adaptive Runge-Kutta
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # Extract solution
    t_integration = sol.t
    states_integration = sol.y

    print(f"Integration completed: {len(t_integration)} time points")
    print(f"Final mass: {states_integration[6, -1]:.6f}")
    print(f"Final longitude: {states_integration[5, -1] / np.pi:.2f} π")

    # Generate guess for mesh structure matching orbit.py
    mesh_intervals = 8  # Fixed to match orbit.py mesh
    polynomial_degrees = [8] * mesh_intervals

    # Generate guess for each mesh interval
    states_guess = []
    controls_guess = []

    for interval_idx in range(mesh_intervals):
        N = polynomial_degrees[interval_idx]

        # Time points for this interval (normalized to [-1, 1])
        tau_interval = np.linspace(-1, 1, N + 1)
        t_interval_start = (interval_idx / mesh_intervals) * t_final
        t_interval_end = ((interval_idx + 1) / mesh_intervals) * t_final
        t_interval_physical = (tau_interval + 1) / 2 * (
            t_interval_end - t_interval_start
        ) + t_interval_start

        # Interpolate states at interval points
        states_interpolated = np.zeros((7, N + 1))
        for state_idx in range(7):
            interp_func = interp1d(
                t_integration,
                states_integration[state_idx, :],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            states_interpolated[state_idx, :] = interp_func(t_interval_physical)

        # Convert to scaled units
        p_s_vals = states_interpolated[0, :] / P_SCALE
        f_vals = states_interpolated[1, :]
        g_vals = states_interpolated[2, :]
        h_vals = states_interpolated[3, :]
        k_vals = states_interpolated[4, :]
        L_s_vals = states_interpolated[5, :] / L_SCALE
        w_vals = states_interpolated[6, :]

        states_guess.append(np.vstack([p_s_vals, f_vals, g_vals, h_vals, k_vals, L_s_vals, w_vals]))

        # Generate control points for this interval
        t_control_physical = (
            np.linspace(0, 1, N) * (t_interval_end - t_interval_start) + t_interval_start
        )

        # Interpolate controls using the same method as in integration
        controls_interpolated = np.zeros((3, N))
        for control_idx in range(N):
            t_ctrl = t_control_physical[control_idx]

            # Evaluate state at control time
            state_at_t = np.zeros(7)
            for state_idx in range(7):
                interp_func = interp1d(
                    t_integration,
                    states_integration[state_idx, :],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                state_at_t[state_idx] = interp_func(t_ctrl)

            # Compute control using the same law as in dynamics
            p, f, g, h, k, L, w = state_at_t
            eps = 1e-12
            q = 1.0 + f * np.cos(L) + g * np.sin(L)
            r = p / max(abs(q), eps)
            alpha2 = h * h - k * k
            X = np.sqrt(h * h + k * k)
            s2 = 1 + X * X

            # Velocity vector
            sqrt_mu_p = np.sqrt(MU / max(p, eps))
            v1 = (
                -(1.0 / s2)
                * sqrt_mu_p
                * (
                    np.sin(L)
                    + alpha2 * np.sin(L)
                    - 2 * h * k * np.cos(L)
                    + g
                    - 2 * f * h * k
                    + alpha2 * g
                )
            )
            v2 = (
                -(1.0 / s2)
                * sqrt_mu_p
                * (
                    -np.cos(L)
                    + alpha2 * np.cos(L)
                    + 2 * h * k * np.sin(L)
                    - f
                    + 2 * g * h * k
                    + alpha2 * f
                )
            )
            v3 = (2.0 / s2) * sqrt_mu_p * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
            vvec = np.array([v1, v2, v3])

            # Position vector
            r1 = r / s2 * (np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L))
            r2 = r / s2 * (np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L))
            r3 = 2 * r / s2 * (h * np.sin(L) - k * np.cos(L))
            rvec = np.array([r1, r2, r3])

            # Reference frame
            rv = cross_product(rvec, vvec)
            rvr = cross_product(rv, rvec)
            norm_r = np.sqrt(max(dot_product(rvec, rvec), eps))
            norm_rv = np.sqrt(max(dot_product(rv, rv), eps))

            ir = rvec / norm_r
            ith = rvr / (norm_rv * norm_r)
            ih = rv / norm_rv

            # Control law: u = Q^T * V/||V||
            v_norm = np.sqrt(max(dot_product(vvec, vvec), eps))
            v_unit = vvec / v_norm
            u1 = dot_product(ir, v_unit)
            u2 = dot_product(ith, v_unit)
            u3 = dot_product(ih, v_unit)

            controls_interpolated[:, control_idx] = [u1, u2, u3]

        controls_guess.append(controls_interpolated)

    final_time_guess = t_final / T_SCALE

    print("Generated initial guess:")
    print(f"  Final time (scaled): {final_time_guess:.1f}")
    print(f"  Final mass: {states_guess[-1][6, -1]:.6f}")
    print(f"  Final longitude (scaled): {states_guess[-1][5, -1]:.2f}")

    return states_guess, controls_guess, final_time_guess


# Problem setup
problem = mtor.Problem("Low thrust transfer problem")
phase = problem.set_phase(1)

# Scaled variables with O(1) bounds
t_s = phase.time(initial=0.0, final=(50000.0 / T_SCALE, 100000.0 / T_SCALE))
p_s = phase.state(
    "p_scaled",
    initial=PTI / P_SCALE,
    boundary=(10.0e6 / P_SCALE, 60.0e6 / P_SCALE),
)
f = phase.state("f", initial=FTI, boundary=(-0.20, 0.20))  # Already O(1)
g = phase.state("g", initial=GTI, boundary=(-0.10, 1.0))  # Already O(1)
h = phase.state("h", initial=HTI, boundary=(-1.0, 1.0))  # Already O(1)
k = phase.state("k", initial=KTI, boundary=(-0.20, 0.20))  # Already O(1)
L_s = phase.state(
    "L_scaled",
    initial=LTI / L_SCALE,
    boundary=(np.pi / L_SCALE, 20 * np.pi / L_SCALE),
)
w = phase.state("w", initial=WTI, boundary=(0.0, 2.0))  # Already O(1)

u1 = phase.control("u1", boundary=(-1.0, 1.0))
u2 = phase.control("u2", boundary=(-1.0, 1.0))
u3 = phase.control("u3", boundary=(-1.0, 1.0))

tau = problem.parameter("tau", boundary=(-50.0, 0.0))

# Convert scaled variables to physical variables for all physics calculations
t = t_s * T_SCALE
p = p_s * P_SCALE
L = L_s * L_SCALE

# Numerical safeguards
eps = 1e-12

# Physics calculations using physical variables (identical to C++ code)
q = 1.0 + f * ca.cos(L) + g * ca.sin(L)
r = p / ca.fmax(q, eps)
alpha2 = h * h - k * k
X = ca.sqrt(h * h + k * k)
s2 = 1 + X * X

# Position vector components
r1 = r / s2 * (ca.cos(L) + alpha2 * ca.cos(L) + 2 * h * k * ca.sin(L))
r2 = r / s2 * (ca.sin(L) - alpha2 * ca.sin(L) + 2 * h * k * ca.cos(L))
r3 = 2 * r / s2 * (h * ca.sin(L) - k * ca.cos(L))
rvec = ca.vertcat(r1, r2, r3)

# Velocity vector components
sqrt_mu_p = ca.sqrt(MU / ca.fmax(p, eps))
v1 = (
    -(1.0 / s2)
    * sqrt_mu_p
    * (ca.sin(L) + alpha2 * ca.sin(L) - 2 * h * k * ca.cos(L) + g - 2 * f * h * k + alpha2 * g)
)
v2 = (
    -(1.0 / s2)
    * sqrt_mu_p
    * (-ca.cos(L) + alpha2 * ca.cos(L) + 2 * h * k * ca.sin(L) - f + 2 * g * h * k + alpha2 * f)
)
v3 = (2.0 / s2) * sqrt_mu_p * (h * ca.cos(L) + k * ca.sin(L) + f * h + g * k)
vvec = ca.vertcat(v1, v2, v3)


# Cross and dot products (unchanged)
def cross_product(a, b):
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# Reference frames (unchanged)
rv = cross_product(rvec, vvec)
rvr = cross_product(rv, rvec)
norm_r = ca.sqrt(ca.fmax(dot_product(rvec, rvec), eps))
norm_rv = ca.sqrt(ca.fmax(dot_product(rv, rv), eps))

ir = rvec / norm_r
ith = rvr / (norm_rv * norm_r)
ih = rv / norm_rv

# Compute in vector
en = ca.vertcat(0.0, 0.0, 1.0)
enir = dot_product(en, ir)
in_vec = en - enir * ir
norm_in = ca.sqrt(ca.fmax(dot_product(in_vec, in_vec), eps))
in_normalized = in_vec / norm_in

# Gravitational perturbations (unchanged physics)
sin_phi = rvec[2] / norm_r
sin_phi_clamped = ca.fmax(ca.fmin(sin_phi, 1.0 - eps), -1.0 + eps)
cos_phi = ca.sqrt(1.0 - sin_phi_clamped**2)

# J2, J3, J4 terms (identical to original)
P2 = 0.5 * (3.0 * sin_phi_clamped**2 - 1.0)
Pdash2 = 3.0 * sin_phi_clamped
r_safe = ca.fmax(r, RE / 100.0)
deltagn_j2 = -MU * cos_phi / (r_safe * r_safe) * (RE / r_safe) ** 2 * Pdash2 * J2
deltagr_j2 = -MU / (r_safe * r_safe) * 3.0 * (RE / r_safe) ** 2 * P2 * J2

P3 = 0.5 * (5.0 * sin_phi_clamped**3 - 3.0 * sin_phi_clamped)
Pdash3 = 0.5 * (15.0 * sin_phi_clamped**2 - 3.0)
deltagn_j3 = -MU * cos_phi / (r_safe * r_safe) * (RE / r_safe) ** 3 * Pdash3 * J3
deltagr_j3 = -MU / (r_safe * r_safe) * 4.0 * (RE / r_safe) ** 3 * P3 * J3

P4 = (1.0 / 8.0) * (35.0 * sin_phi_clamped**4 - 30.0 * sin_phi_clamped**2 + 3.0)
Pdash4 = (1.0 / 8.0) * (140.0 * sin_phi_clamped**3 - 60.0 * sin_phi_clamped)
deltagn_j4 = -MU * cos_phi / (r_safe * r_safe) * (RE / r_safe) ** 4 * Pdash4 * J4
deltagr_j4 = -MU / (r_safe * r_safe) * 5.0 * (RE / r_safe) ** 4 * P4 * J4

deltagn = deltagn_j2 + deltagn_j3 + deltagn_j4
deltagr = deltagr_j2 + deltagr_j3 + deltagr_j4

# Gravitational perturbation vector
delta_g = deltagn * in_normalized - deltagr * ir

# Transform to orbital frame
DELTA_g1 = dot_product(ir, delta_g)
DELTA_g2 = dot_product(ith, delta_g)
DELTA_g3 = dot_product(ih, delta_g)

# Thrust acceleration components
w_safe = ca.fmax(w, eps)
DELTA_T1 = G0 * T_THRUST * (1.0 + 0.01 * tau) * u1 / w_safe
DELTA_T2 = G0 * T_THRUST * (1.0 + 0.01 * tau) * u2 / w_safe
DELTA_T3 = G0 * T_THRUST * (1.0 + 0.01 * tau) * u3 / w_safe

# Total acceleration
delta1 = DELTA_g1 + DELTA_T1
delta2 = DELTA_g2 + DELTA_T2
delta3 = DELTA_g3 + DELTA_T3

# Physical derivatives (identical to C++ code)
sqrt_p_mu = ca.sqrt(ca.fmax(p, eps) / MU)
q_safe = ca.fmax(ca.fabs(q), eps)

pdot = 2 * p / q_safe * sqrt_p_mu * delta2
fdot = (
    sqrt_p_mu * ca.sin(L) * delta1
    + sqrt_p_mu * (1.0 / q_safe) * ((q + 1.0) * ca.cos(L) + f) * delta2
    - sqrt_p_mu * (g / q_safe) * (h * ca.sin(L) - k * ca.cos(L)) * delta3
)
gdot = (
    -sqrt_p_mu * ca.cos(L) * delta1
    + sqrt_p_mu * (1.0 / q_safe) * ((q + 1.0) * ca.sin(L) + g) * delta2
    + sqrt_p_mu * (f / q_safe) * (h * ca.sin(L) - k * ca.cos(L)) * delta3
)
hdot = sqrt_p_mu * s2 * ca.cos(L) / (2.0 * q_safe) * delta3
kdot = sqrt_p_mu * s2 * ca.sin(L) / (2.0 * q_safe) * delta3
Ldot = (
    sqrt_p_mu * (1.0 / q_safe) * (h * ca.sin(L) - k * ca.cos(L)) * delta3
    + ca.sqrt(MU * ca.fmax(p, eps)) * (q / ca.fmax(p, eps)) ** 2
)
wdot = -T_THRUST * (1.0 + 0.01 * tau) / ISP

# Scaled dynamics: Apply the chain rule for time scaling
phase.dynamics(
    {
        p_s: (pdot / P_SCALE) * T_SCALE,
        f: fdot * T_SCALE,
        g: gdot * T_SCALE,
        h: hdot * T_SCALE,
        k: kdot * T_SCALE,
        L_s: (Ldot / L_SCALE) * T_SCALE,
        w: wdot * T_SCALE,
    }
)

# Path constraint using physical variables
thrust_magnitude_squared = u1**2 + u2**2 + u3**2
phase.path_constraints(
    thrust_magnitude_squared >= 1.0 - EQ_TOL,
    thrust_magnitude_squared <= 1.0 + EQ_TOL,
)

# Event constraints using scaled state variables with scaled constraint values
phase.event_constraints(
    p_s.final == PTF / P_SCALE,  # Convert constraint value to scaled units
    ca.sqrt(f.final**2 + g.final**2) == EVENT_FINAL_9,  # f,g already O(1)
    ca.sqrt(h.final**2 + k.final**2) == EVENT_FINAL_10,  # h,k already O(1)
    f.final * h.final + g.final * k.final == EVENT_FINAL_11,  # Already O(1)
)

gtf_htf_minus_ktf_ftf = g.final * h.final - k.final * f.final
phase.event_constraints(
    gtf_htf_minus_ktf_ftf >= EVENT_FINAL_12_LOWER,
    gtf_htf_minus_ktf_ftf <= EVENT_FINAL_12_UPPER,
)

# Objective using physical variable
problem.minimize(-w.final)

# Mesh configuration
phase.mesh([8, 8, 8, 8, 8, 8, 8, 8], [-1.0, -6 / 7, -4 / 7, -2 / 7, 0, 2 / 7, 4 / 7, 6 / 7, 1.0])

# Generate advanced initial guess
states_guess, controls_guess, final_time_guess = generate_advanced_initial_guess()

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: final_time_guess},
    static_parameters=np.array([-25.0]),
)

# Solve
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-4,
    max_iterations=20,
    min_polynomial_degree=3,
    max_polynomial_degree=8,
    nlp_options={
        "ipopt.print_level": 5,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-4,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
    },
)

# Results (convert back to physical units)
if solution.status["success"]:
    final_mass = solution[(1, "w")][-1]
    final_time_scaled = solution.phases[1]["times"]["final"]
    final_time = final_time_scaled * T_SCALE

    print(f"Final mass: {final_mass:.6f}")
    print(f"Final time: {final_time:.1f} seconds ({final_time / 3600:.2f} hours)")

    # Show scaling effects
    p_final_scaled = solution[(1, "p_scaled")][-1]
    p_final_physical = p_final_scaled * P_SCALE
    print(f"Final p (scaled): {p_final_scaled:.6f}")
    print(f"Final p (physical): {p_final_physical:.1f} ft")

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
