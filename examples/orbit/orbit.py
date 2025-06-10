import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import maptor as mtor


# =============================================================================
# 1. Physical Constants and Scaling
# =============================================================================
T_phys = 4.446618e-3
Isp_phys = 450
mu_phys = 1.407645794e16
gs_phys = 32.174
Re_phys = 20925662.73
J2 = 1082.639e-6
J3 = -2.565e-6
J4 = -1.608e-6
p0_phys = 21837080.052835
w0_phys = 1.0
LU = p0_phys
MU = w0_phys
TU = np.sqrt(LU**3 / mu_phys)
mu_s = 1.0
Re_s = Re_phys / LU
T_s = T_phys * TU**2 / (MU * LU)
gs_s = gs_phys * TU**2 / LU
Isp_s = Isp_phys / TU

# =============================================================================
# 2. Scaled Problem Definition
# =============================================================================
p0_s = p0_phys / LU
f0 = 0
g0 = 0
h0 = -0.25396764647494
k0 = 0
L0 = np.pi
w0_s = w0_phys / MU
pf_phys = 40007346.015232
pf_s = pf_phys / LU
pmin_s, pmax_s = 20000000 / LU, 60000000 / LU
fmin, fmax = -1, 1
gmin, gmax = -1, 1
hmin, hmax = -1, 1
kmin, kmax = -1, 1
Lmin, Lmax = L0, 9 * 2 * np.pi
wmin_s, wmax_s = 0.1 / MU, w0_phys / MU
urmin, urmax = -1, 1
utmin, utmax = -1, 1
uhmin, uhmax = -1, 1
taumin, taumax = -50, 0
t0_s = 0
tfmin_s, tfmax_s = 50000 / TU, 100000 / TU
ff_target_sq = 0.73550320568829**2
hf_target_sq = 0.61761258786099**2

problem = mtor.Problem("Low-Thrust Orbit Transfer (Scaled)")
phase = problem.set_phase(1)

t = phase.time(initial=t0_s, final=(tfmin_s, tfmax_s))
p = phase.state("p_scaled", initial=p0_s, final=pf_s, boundary=(pmin_s, pmax_s))
f = phase.state("f", initial=f0, final=(fmin, fmax), boundary=(fmin, fmax))
g = phase.state("g", initial=g0, final=(gmin, gmax), boundary=(gmin, gmax))
h = phase.state("h", initial=h0, final=(hmin, hmax), boundary=(hmin, hmax))
k = phase.state("k", initial=k0, final=(kmin, kmax), boundary=(kmin, kmax))
L = phase.state("L", initial=L0, final=(Lmin, Lmax), boundary=(Lmin, Lmax))
w = phase.state("w_scaled", initial=w0_s, final=(wmin_s, wmax_s), boundary=(wmin_s, wmax_s))
ur = phase.control("ur", boundary=(urmin, urmax))
ut = phase.control("ut", boundary=(utmin, utmax))
uh = phase.control("uh", boundary=(uhmin, uhmax))
tau = problem.parameter("tau", boundary=(taumin, taumax))

# =============================================================================
# 3. Scaled Dynamics
# =============================================================================
q = 1 + f * ca.cos(L) + g * ca.sin(L)
r_s = p / q
alpha2 = h * h - k * k
chi = ca.sqrt(h * h + k * k)
s2 = 1 + chi * chi
rX_s = (r_s / s2) * (ca.cos(L) + alpha2 * ca.cos(L) + 2 * h * k * ca.sin(L))
rY_s = (r_s / s2) * (ca.sin(L) - alpha2 * ca.sin(L) + 2 * h * k * ca.cos(L))
rZ_s = (2 * r_s / s2) * (h * ca.sin(L) - k * ca.cos(L))
rMag_s = ca.sqrt(ca.fmax(1e-9, rX_s**2 + rY_s**2 + rZ_s**2))
rXZMag_s = ca.sqrt(ca.fmax(1e-9, rX_s**2 + rZ_s**2))
v_factor = ca.sqrt(mu_s / p)
vX_s = (
    -(1 / s2)
    * v_factor
    * (ca.sin(L) + alpha2 * ca.sin(L) - 2 * h * k * ca.cos(L) + g - 2 * f * h * k + alpha2 * g)
)
vY_s = (
    -(1 / s2)
    * v_factor
    * (-ca.cos(L) + alpha2 * ca.cos(L) + 2 * h * k * ca.sin(L) - f + 2 * g * h * k + alpha2 * f)
)
vZ_s = (2 / s2) * v_factor * (h * ca.cos(L) + k * ca.sin(L) + f * h + g * k)
ir1, ir2, ir3 = rX_s / rMag_s, rY_s / rMag_s, rZ_s / rMag_s
rCrossv_x = rY_s * vZ_s - rZ_s * vY_s
rCrossv_y = rZ_s * vX_s - rX_s * vZ_s
rCrossv_z = rX_s * vY_s - rY_s * vX_s
rCrossvMag = ca.sqrt(ca.fmax(1e-9, rCrossv_x**2 + rCrossv_y**2 + rCrossv_z**2))
rCrossvCrossr_x = rCrossv_y * rZ_s - rCrossv_z * rY_s
rCrossvCrossr_y = rCrossv_z * rX_s - rCrossv_x * rZ_s
rCrossvCrossr_z = rCrossv_x * rY_s - rCrossv_y * rX_s
it1 = rCrossvCrossr_x / (rCrossvMag * rMag_s)
it2 = rCrossvCrossr_y / (rCrossvMag * rMag_s)
it3 = rCrossvCrossr_z / (rCrossvMag * rMag_s)
ih1, ih2, ih3 = (
    rCrossv_x / rCrossvMag,
    rCrossv_y / rCrossvMag,
    rCrossv_z / rCrossvMag,
)
enir = ir3
enirir1, enirir2, enirir3 = enir * ir1, enir * ir2, enir * ir3
enenirir1, enenirir2, enenirir3 = -enirir1, -enirir2, 1 - enirir3
enenirirMag = np.sqrt(np.fmax(1e-9, enenirir1**2 + enenirir2**2 + enenirir3**2))
in1, in2, in3 = (
    enenirir1 / enenirirMag,
    enenirir2 / enenirirMag,
    enenirir3 / enenirirMag,
)
sinphi = rZ_s / rXZMag_s
cosphi = ca.sqrt(ca.fmax(1e-9, 1 - sinphi**2))
P2 = (3 * sinphi**2 - 2) / 2
P3 = (5 * sinphi**3 - 3 * sinphi) / 2
P4 = (35 * sinphi**4 - 30 * sinphi**2 + 3) / 8
dP2 = 3 * sinphi
dP3 = (15 * sinphi**2 - 3) / 2
dP4 = (140 * sinphi**3 - 60 * sinphi) / 8
sumn = (Re_s / r_s) ** 2 * dP2 * J2 + (Re_s / r_s) ** 3 * dP3 * J3 + (Re_s / r_s) ** 4 * dP4 * J4
sumr = (
    3 * (Re_s / r_s) ** 2 * P2 * J2
    + 4 * (Re_s / r_s) ** 3 * P3 * J3
    + 5 * (Re_s / r_s) ** 4 * P4 * J4
)
delta_gn_s = -(mu_s * cosphi / (r_s**2)) * sumn
delta_gr_s = -(mu_s / (r_s**2)) * sumr
delta_g1 = delta_gn_s * in1 - delta_gr_s * ir1
delta_g2 = delta_gn_s * in2 - delta_gr_s * ir2
delta_g3 = delta_gn_s * in3 - delta_gr_s * ir3
Deltag1 = ir1 * delta_g1 + ir2 * delta_g2 + ir3 * delta_g3
Deltag2 = it1 * delta_g1 + it2 * delta_g2 + it3 * delta_g3
Deltag3 = ih1 * delta_g1 + ih2 * delta_g2 + ih3 * delta_g3
thrust_factor = (gs_s * T_s * (1 + 0.01 * tau)) / w
DeltaT1 = thrust_factor * ur
DeltaT2 = thrust_factor * ut
DeltaT3 = thrust_factor * uh
Delta1 = Deltag1 + DeltaT1
Delta2 = Deltag2 + DeltaT2
Delta3 = Deltag3 + DeltaT3
dp = (2 * p / q) * ca.sqrt(p / mu_s) * Delta2
df = (
    ca.sqrt(p / mu_s) * ca.sin(L) * Delta1
    + ca.sqrt(p / mu_s) * (1 / q) * ((q + 1) * ca.cos(L) + f) * Delta2
    - ca.sqrt(p / mu_s) * (g / q) * (h * ca.sin(L) - k * ca.cos(L)) * Delta3
)
dg = (
    -ca.sqrt(p / mu_s) * ca.cos(L) * Delta1
    + ca.sqrt(p / mu_s) * (1 / q) * ((q + 1) * ca.sin(L) + g) * Delta2
    + ca.sqrt(p / mu_s) * (f / q) * (h * ca.sin(L) - k * ca.cos(L)) * Delta3
)
dh = ca.sqrt(p / mu_s) * (s2 * ca.cos(L) / (2 * q)) * Delta3
dk = ca.sqrt(p / mu_s) * (s2 * ca.sin(L) / (2 * q)) * Delta3
dL = ca.sqrt(p / mu_s) * (1 / q) * (h * ca.sin(L) - k * ca.cos(L)) * Delta3 + ca.sqrt(mu_s * p) * (
    (q / p) ** 2
)
dw = -(T_s * (1 + 0.01 * tau)) / (gs_s * Isp_s)
phase.dynamics({p: dp, f: df, g: dg, h: dh, k: dk, L: dL, w: dw})

# =============================================================================
# 4. High-Fidelity Initial Guess Generation
# =============================================================================


def get_dynamics_and_controls_for_guess(y):
    p, f, g, h, k, L, w = y
    q = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / q
    alpha2 = h * h - k * k
    chi = np.sqrt(h * h + k * k)
    s2 = 1 + chi * chi
    rX = (r / s2) * (np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L))
    rY = (r / s2) * (np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L))
    rZ = (2 * r / s2) * (h * np.sin(L) - k * np.cos(L))
    rMag = np.sqrt(np.fmax(1e-9, rX**2 + rY**2 + rZ**2))
    rXZMag = np.sqrt(np.fmax(1e-9, rX**2 + rZ**2))
    v_factor = np.sqrt(mu_phys / p)
    vX = (
        -(1 / s2)
        * v_factor
        * (np.sin(L) + alpha2 * np.sin(L) - 2 * h * k * np.cos(L) + g - 2 * f * h * k + alpha2 * g)
    )
    vY = (
        -(1 / s2)
        * v_factor
        * (-np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L) - f + 2 * g * h * k + alpha2 * f)
    )
    vZ = (2 / s2) * v_factor * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
    v_vec = np.array([vX, vY, vZ])
    v_mag = np.linalg.norm(v_vec)
    v_hat = v_vec / v_mag
    ir_vec = np.array([rX, rY, rZ]) / rMag
    h_vec = np.cross(ir_vec * rMag, v_vec)
    ih_vec = h_vec / np.linalg.norm(h_vec)
    it_vec = np.cross(ih_vec, ir_vec)
    ur = np.dot(v_hat, ir_vec)
    ut = np.dot(v_hat, it_vec)
    uh = np.dot(v_hat, ih_vec)
    ir1, ir2, ir3 = ir_vec
    it1, it2, it3 = it_vec
    ih1, ih2, ih3 = ih_vec
    enir = ir3
    enirir1, enirir2, enirir3 = enir * ir1, enir * ir2, enir * ir3
    enenirir1, enenirir2, enenirir3 = -enirir1, -enirir2, 1 - enirir3
    enenirirMag = np.sqrt(np.fmax(1e-9, enenirir1**2 + enenirir2**2 + enenirir3**2))
    in1, in2, in3 = (
        enenirir1 / enenirirMag,
        enenirir2 / enenirirMag,
        enenirir3 / enenirirMag,
    )
    sinphi = rZ / rXZMag
    cosphi = np.sqrt(np.fmax(1e-9, 1 - sinphi**2))
    P2 = (3 * sinphi**2 - 2) / 2
    P3 = (5 * sinphi**3 - 3 * sinphi) / 2
    P4 = (35 * sinphi**4 - 30 * sinphi**2 + 3) / 8
    dP2 = 3 * sinphi
    dP3 = (15 * sinphi**2 - 3) / 2
    dP4 = (140 * sinphi**3 - 60 * sinphi) / 8
    sumn = (
        (Re_phys / r) ** 2 * dP2 * J2
        + (Re_phys / r) ** 3 * dP3 * J3
        + (Re_phys / r) ** 4 * dP4 * J4
    )
    sumr = (
        3 * (Re_phys / r) ** 2 * P2 * J2
        + 4 * (Re_phys / r) ** 3 * P3 * J3
        + 5 * (Re_phys / r) ** 4 * P4 * J4
    )
    delta_gn = -(mu_phys * cosphi / (r**2)) * sumn
    delta_gr = -(mu_phys / (r**2)) * sumr
    delta_g1 = delta_gn * in1 - delta_gr * ir1
    delta_g2 = delta_gn * in2 - delta_gr * ir2
    delta_g3 = delta_gn * in3 - delta_gr * ir3
    Deltag1 = ir1 * delta_g1 + ir2 * delta_g2 + ir3 * delta_g3
    Deltag2 = it1 * delta_g1 + it2 * delta_g2 + it3 * delta_g3
    Deltag3 = ih1 * delta_g1 + ih2 * delta_g2 + ih3 * delta_g3
    tau_guess = -25.0
    thrust_factor = (T_phys * (1 + 0.01 * tau_guess)) / w
    DeltaT1 = thrust_factor * ur
    DeltaT2 = thrust_factor * ut
    DeltaT3 = thrust_factor * uh
    Delta1 = Deltag1 + DeltaT1
    Delta2 = Deltag2 + DeltaT2
    Delta3 = Deltag3 + DeltaT3
    dp = (2 * p / q) * np.sqrt(p / mu_phys) * Delta2
    df = (
        np.sqrt(p / mu_phys) * np.sin(L) * Delta1
        + np.sqrt(p / mu_phys) * (1 / q) * ((q + 1) * np.cos(L) + f) * Delta2
        - np.sqrt(p / mu_phys) * (g / q) * (h * np.sin(L) - k * np.cos(L)) * Delta3
    )
    dg = (
        -np.sqrt(p / mu_phys) * np.cos(L) * Delta1
        + np.sqrt(p / mu_phys) * (1 / q) * ((q + 1) * np.sin(L) + g) * Delta2
        + np.sqrt(p / mu_phys) * (f / q) * (h * np.sin(L) - k * np.cos(L)) * Delta3
    )
    dh = np.sqrt(p / mu_phys) * (s2 * np.cos(L) / (2 * q)) * Delta3
    dk = np.sqrt(p / mu_phys) * (s2 * np.sin(L) / (2 * q)) * Delta3
    dL = np.sqrt(p / mu_phys) * (1 / q) * (h * np.sin(L) - k * np.cos(L)) * Delta3 + np.sqrt(
        mu_phys * p
    ) * ((q / p) ** 2)
    dw = -(T_phys * (1 + 0.01 * tau_guess)) / (gs_phys * Isp_phys)
    return np.array([dp, df, dg, dh, dk, dL, dw]), np.array([ur, ut, uh])


def dynamics_for_ode(t, y):
    dydt, _ = get_dynamics_and_controls_for_guess(y)
    return dydt


def generate_physics_based_guess(poly_degrees):
    print("Generating high-fidelity physics-based initial guess...")
    y0_phys = np.array([p0_phys, f0, g0, h0, k0, L0, w0_phys])
    t_final_guess_phys = 90000.0
    t_span = [0, t_final_guess_phys]
    sol = solve_ivp(
        dynamics_for_ode,
        t_span,
        y0_phys,
        method="RK45",
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )
    num_intervals = len(poly_degrees)
    L_start = sol.y[5, 0]
    L_end = sol.y[5, -1]
    L_grid = np.linspace(L_start, L_end, num_intervals + 1)
    time_interpolator = interp1d(sol.y[5], sol.t, kind="linear", fill_value="extrapolate")
    t_grid = time_interpolator(L_grid)
    states_guess = []
    controls_guess = []
    for i in range(num_intervals):
        num_state_points = poly_degrees[i] + 1
        num_control_points = poly_degrees[i]
        t_interval_states = np.linspace(t_grid[i], t_grid[i + 1], num_state_points)
        y_interval_states = sol.sol(t_interval_states)
        t_interval_controls = t_interval_states[:-1]
        y_interval_for_controls = sol.sol(t_interval_controls)
        u_interval = np.array(
            [
                get_dynamics_and_controls_for_guess(y_interval_for_controls[:, j])[1]
                for j in range(num_control_points)
            ]
        ).T
        y_interval_scaled = y_interval_states / np.array([LU, 1, 1, 1, 1, 1, MU]).reshape(-1, 1)
        states_guess.append(y_interval_scaled)
        controls_guess.append(u_interval)
    tf_guess_scaled = t_final_guess_phys / TU
    print("Initial guess generation complete.")
    return states_guess, controls_guess, tf_guess_scaled


# =============================================================================
# 5. Constraints, Objective, and Solver Setup
# =============================================================================
phase.path_constraints(ur**2 + ut**2 + uh**2 == 1)
phase.event_constraints(
    f.final**2 + g.final**2 == ff_target_sq,
    h.final**2 + k.final**2 == hf_target_sq,
    f.final * h.final + g.final * k.final == 0,
    g.final * h.final - k.final * f.final >= -3,
    g.final * h.final - k.final * f.final <= 0,
)
problem.minimize(-w.final)
poly_degrees = [8] * 8
phase.mesh(
    poly_degrees,
    [-1.0, -6 / 7, -4 / 7, -2 / 7, 0, 2 / 7, 4 / 7, 6 / 7, 1.0],
)
states_guess, controls_guess, tf_guess = generate_physics_based_guess(poly_degrees)
problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: tf_guess},
    static_parameters=np.array([-25]),
)
solution = mtor.solve_adaptive(
    problem,
    nlp_options={
        "ipopt.print_level": 5,
        "ipopt.max_iter": 5000,
        "ipopt.tol": 1e-7,
        "ipopt.constr_viol_tol": 1e-7,
        "ipopt.linear_solver": "mumps",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.warm_start_init_point": "yes",
    },
)

# =============================================================================
# 6. Results
# =============================================================================
if solution.status["success"]:
    final_mass_phys = solution[(1, "w_scaled")][-1] * MU
    final_time_phys = solution.phases[1]["times"]["final"] * TU
    print("\n" + "=" * 50)
    print("Optimization Successful!")
    print("=" * 50)
    print(f"Final mass: {final_mass_phys:.6f} [kg or slug]")
    print(f"Transfer time: {final_time_phys:.1f} s ({final_time_phys / 3600:.2f} hr)")
    print(f"Objective (negative scaled mass): {solution.status['objective']:.6f}")
    pf_actual_phys = solution[(1, "p_scaled")][-1] * LU
    ff_actual = solution[(1, "f")][-1]
    gf_actual = solution[(1, "g")][-1]
    hf_actual = solution[(1, "h")][-1]
    kf_actual = solution[(1, "k")][-1]
    print("\nFinal orbital elements (physical units):")
    print(f"p: {pf_actual_phys:.1f} ft (target: {pf_phys:.1f} ft)")
    print(f"f: {ff_actual:.6f}")
    print(f"g: {gf_actual:.6f}")
    print(f"h: {hf_actual:.6f}")
    print(f"k: {kf_actual:.6f}")
    eccentricity_sq = ff_actual**2 + gf_actual**2
    inclination_sq = hf_actual**2 + kf_actual**2
    print("\nConstraint verification:")
    print(f"f² + g²: {eccentricity_sq:.6f} (target: {ff_target_sq:.6f})")
    print(f"h² + k²: {inclination_sq:.6f} (target: {hf_target_sq:.6f})")
    solution.plot()
else:
    print(f"\nOptimization failed: {solution.status['message']}")
