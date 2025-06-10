import casadi as ca
import numpy as np

import maptor as mtor


# Physical constants (exact values from MATLAB code)
T = 4.446618e-3  # [lb] - Maximum thrust
Isp = 450  # [s] - Specific impulse
mu = 1.407645794e16  # [ft^3/s^2] - Earth gravitational parameter
gs = 32.174  # [ft/s^2] - Standard gravity
Re = 20925662.73  # [ft] - Earth radius
J2 = 1082.639e-6  # J2 harmonic coefficient
J3 = -2.565e-6  # J3 harmonic coefficient
J4 = -1.608e-6  # J4 harmonic coefficient

# Initial conditions (exact values from MATLAB)
p0 = 21837080.052835  # [ft] - Initial semi-latus rectum
f0 = 0  # Initial f eccentricity component
g0 = 0  # Initial g eccentricity component
h0 = -0.25396764647494  # Initial h inclination component
k0 = 0  # Initial k inclination component
L0 = np.pi  # [rad] - Initial true longitude
w0 = 1  # Initial normalized mass

# Final conditions
pf = 40007346.015232  # [ft] - Final semi-latus rectum

# State bounds (from MATLAB bounds structure)
pmin, pmax = 20000000, 60000000  # [ft]
fmin, fmax = -1, 1
gmin, gmax = -1, 1
hmin, hmax = -1, 1
kmin, kmax = -1, 1
Lmin, Lmax = L0, 9 * 2 * np.pi
wmin, wmax = 0.1, w0

# Control bounds
urmin, urmax = -1, 1
utmin, utmax = -1, 1
uhmin, uhmax = -1, 1

# Parameter bounds
taumin, taumax = -50, 0

# Time bounds
t0 = 0
tfmin, tfmax = 50000, 100000

# Event constraint bounds (from MATLAB bounds.eventgroup)
ff_target_sq = 0.73550320568829**2
hf_target_sq = 0.61761258786099**2

# Problem setup
problem = mtor.Problem("Low-Thrust Orbit Transfer")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=t0, final=(tfmin, tfmax))

# States with exact bounds from MATLAB
p = phase.state("semi_latus_rectum", initial=p0, final=pf, boundary=(pmin, pmax))
f = phase.state("f_eccentricity", initial=f0, final=(fmin, fmax), boundary=(fmin, fmax))
g = phase.state("g_eccentricity", initial=g0, final=(gmin, gmax), boundary=(gmin, gmax))
h = phase.state("h_inclination", initial=h0, final=(hmin, hmax), boundary=(hmin, hmax))
k = phase.state("k_inclination", initial=k0, final=(kmin, kmax), boundary=(kmin, kmax))
L = phase.state("true_longitude", initial=L0, final=(Lmin, Lmax), boundary=(Lmin, Lmax))
w = phase.state("mass", initial=w0, final=(wmin, wmax), boundary=(wmin, wmax))

# Controls
ur = phase.control("radial_thrust", boundary=(urmin, urmax))
ut = phase.control("tangential_thrust", boundary=(utmin, utmax))
uh = phase.control("normal_thrust", boundary=(uhmin, uhmax))

# Static parameter
tau = problem.parameter("thrust_efficiency", boundary=(taumin, taumax))

# Orbital mechanics calculations (exact translation from MATLAB)
q = 1 + f * ca.cos(L) + g * ca.sin(L)
r = p / q
alpha2 = h * h - k * k
chi = ca.sqrt(h * h + k * k)
s2 = 1 + chi * chi

# Position vector components
rX = (r / s2) * (ca.cos(L) + alpha2 * ca.cos(L) + 2 * h * k * ca.sin(L))
rY = (r / s2) * (ca.sin(L) - alpha2 * ca.sin(L) + 2 * h * k * ca.cos(L))
rZ = (2 * r / s2) * (h * ca.sin(L) - k * ca.cos(L))

rMag = ca.sqrt(rX**2 + rY**2 + rZ**2)
rXZMag = ca.sqrt(rX**2 + rZ**2)

# Velocity vector components
vX = (
    -(1 / s2)
    * ca.sqrt(mu / p)
    * (ca.sin(L) + alpha2 * ca.sin(L) - 2 * h * k * ca.cos(L) + g - 2 * f * h * k + alpha2 * g)
)
vY = (
    -(1 / s2)
    * ca.sqrt(mu / p)
    * (-ca.cos(L) + alpha2 * ca.cos(L) + 2 * h * k * ca.sin(L) - f + 2 * g * h * k + alpha2 * f)
)
vZ = (2 / s2) * ca.sqrt(mu / p) * (h * ca.cos(L) + k * ca.sin(L) + f * h + g * k)

# Cross products for reference frame
rCrossv_x = rY * vZ - rZ * vY
rCrossv_y = rZ * vX - rX * vZ
rCrossv_z = rX * vY - rY * vX
rCrossvMag = ca.sqrt(rCrossv_x**2 + rCrossv_y**2 + rCrossv_z**2)

# Unit vectors
ir1, ir2, ir3 = rX / rMag, rY / rMag, rZ / rMag

# Transverse unit vector (r × v × r normalized)
rCrossvCrossr_x = rCrossv_y * rZ - rCrossv_z * rY
rCrossvCrossr_y = rCrossv_z * rX - rCrossv_x * rZ
rCrossvCrossr_z = rCrossv_x * rY - rCrossv_y * rX

it1 = rCrossvCrossr_x / (rCrossvMag * rMag)
it2 = rCrossvCrossr_y / (rCrossvMag * rMag)
it3 = rCrossvCrossr_z / (rCrossvMag * rMag)

# Normal unit vector
ih1, ih2, ih3 = rCrossv_x / rCrossvMag, rCrossv_y / rCrossvMag, rCrossv_z / rCrossvMag

# North direction calculation for perturbations
enir = ir3
enirir1, enirir2, enirir3 = enir * ir1, enir * ir2, enir * ir3
enenirir1, enenirir2, enenirir3 = -enirir1, -enirir2, 1 - enirir3
enenirirMag = ca.sqrt(enenirir1**2 + enenirir2**2 + enenirir3**2)
in1, in2, in3 = enenirir1 / enenirirMag, enenirir2 / enenirirMag, enenirir3 / enenirirMag

# Geocentric latitude
sinphi = rZ / rXZMag
cosphi = ca.sqrt(1 - sinphi**2)

# Legendre polynomials
P2 = (3 * sinphi**2 - 2) / 2
P3 = (5 * sinphi**3 - 3 * sinphi) / 2
P4 = (35 * sinphi**4 - 30 * sinphi**2 + 3) / 8
dP2 = 3 * sinphi
dP3 = (15 * sinphi**2 - 3) / 2
dP4 = (140 * sinphi**3 - 60 * sinphi) / 8

# Oblate earth perturbations
sumn = (Re / r) ** 2 * dP2 * J2 + (Re / r) ** 3 * dP3 * J3 + (Re / r) ** 4 * dP4 * J4
sumr = 3 * (Re / r) ** 2 * P2 * J2 + 4 * (Re / r) ** 3 * P3 * J3 + 5 * (Re / r) ** 4 * P4 * J4

delta_gn = -(mu * cosphi / (r**2)) * sumn
delta_gr = -(mu / (r**2)) * sumr

# Gravitational perturbation components
delta_g1 = delta_gn * in1 - delta_gr * ir1
delta_g2 = delta_gn * in2 - delta_gr * ir2
delta_g3 = delta_gn * in3 - delta_gr * ir3

# Project to orbital frame
Deltag1 = ir1 * delta_g1 + ir2 * delta_g2 + ir3 * delta_g3
Deltag2 = it1 * delta_g1 + it2 * delta_g2 + it3 * delta_g3
Deltag3 = ih1 * delta_g1 + ih2 * delta_g2 + ih3 * delta_g3

# Thrust acceleration components (with parameter tau)
thrust_factor = (gs * T * (1 + 0.01 * tau)) / w
DeltaT1 = thrust_factor * ur
DeltaT2 = thrust_factor * ut
DeltaT3 = thrust_factor * uh

# Total acceleration
Delta1 = Deltag1 + DeltaT1
Delta2 = Deltag2 + DeltaT2
Delta3 = Deltag3 + DeltaT3

# Dynamics (exact translation from MATLAB lowThrustContinuous)
dp = (2 * p / q) * ca.sqrt(p / mu) * Delta2

df = (
    ca.sqrt(p / mu) * ca.sin(L) * Delta1
    + ca.sqrt(p / mu) * (1 / q) * ((q + 1) * ca.cos(L) + f) * Delta2
    - ca.sqrt(p / mu) * (g / q) * (h * ca.sin(L) - k * ca.cos(L)) * Delta3
)

dg = (
    -ca.sqrt(p / mu) * ca.cos(L) * Delta1
    + ca.sqrt(p / mu) * (1 / q) * ((q + 1) * ca.sin(L) + g) * Delta2
    + ca.sqrt(p / mu) * (f / q) * (h * ca.sin(L) - k * ca.cos(L)) * Delta3
)

dh = ca.sqrt(p / mu) * (s2 * ca.cos(L) / (2 * q)) * Delta3

dk = ca.sqrt(p / mu) * (s2 * ca.sin(L) / (2 * q)) * Delta3

dL = ca.sqrt(p / mu) * (1 / q) * (h * ca.sin(L) - k * ca.cos(L)) * Delta3 + ca.sqrt(mu * p) * (
    (q / p) ** 2
)

dw = -(T * (1 + 0.01 * tau) / Isp)

phase.dynamics({p: dp, f: df, g: dg, h: dh, k: dk, L: dL, w: dw})

# Path constraint: thrust magnitude limit (exact from MATLAB)
phase.path_constraints(ur**2 + ut**2 + uh**2 <= 1)

# Event constraints: final orbital characteristics (exact from MATLAB)
phase.event_constraints(
    f.final**2 + g.final**2 == ff_target_sq,
    h.final**2 + k.final**2 == hf_target_sq,
    f.final * h.final + g.final * k.final == 0,
    g.final * h.final - k.final * f.final >= -3,
    g.final * h.final - k.final * f.final <= 0,
)

# Objective: maximize final mass (exact from MATLAB)
problem.minimize(-w.final)

# High-density mesh configuration (following GPOPS hp-adaptive approach)
phase.mesh([8, 8, 8, 8, 8, 8, 8, 8], [-1.0, -6 / 7, -4 / 7, -2 / 7, 0, 2 / 7, 4 / 7, 6 / 7, 1.0])


# Physics-based initial guess following GPOPS approach
def generate_physics_based_guess():
    """Generate initial guess by propagating modified equinoctial dynamics"""

    # Time span for guess propagation
    t_guess = 75000  # seconds
    n_points_total = sum([8, 8, 8, 8, 8, 8, 8, 8]) + 1  # Total collocation points

    # Propagate simplified dynamics for guess
    time_span = np.linspace(0, t_guess, n_points_total)

    # Initialize state arrays
    p_traj = np.zeros(n_points_total)
    f_traj = np.zeros(n_points_total)
    g_traj = np.zeros(n_points_total)
    h_traj = np.zeros(n_points_total)
    k_traj = np.zeros(n_points_total)
    L_traj = np.zeros(n_points_total)
    w_traj = np.zeros(n_points_total)

    # Control guess: thrust along velocity direction (GPOPS approach)
    ur_traj = np.zeros(n_points_total - 1)
    ut_traj = np.zeros(n_points_total - 1)
    uh_traj = np.zeros(n_points_total - 1)

    # Initial conditions
    p_traj[0] = p0
    f_traj[0] = f0
    g_traj[0] = g0
    h_traj[0] = h0
    k_traj[0] = k0
    L_traj[0] = L0
    w_traj[0] = w0

    # Simple forward propagation with physics-based control
    dt = time_span[1] - time_span[0]
    tau_guess = -25  # From GPOPS paper

    for i in range(n_points_total - 1):
        # Current state
        p_curr = p_traj[i]
        f_curr = f_traj[i]
        g_curr = g_traj[i]
        h_curr = h_traj[i]
        k_curr = k_traj[i]
        L_curr = L_traj[i]
        w_curr = w_traj[i]

        # Basic orbital mechanics for control direction
        q_curr = 1 + f_curr * np.cos(L_curr) + g_curr * np.sin(L_curr)

        # Thrust along velocity direction (simplified)
        thrust_magnitude = 0.3  # Moderate thrust
        ur_traj[i] = 0.1 * thrust_magnitude  # Small radial
        ut_traj[i] = thrust_magnitude  # Main tangential
        uh_traj[i] = 0.05 * thrust_magnitude  # Small normal

        # Simplified dynamics integration (Euler step)
        sqrt_p_mu = np.sqrt(p_curr / mu)
        sqrt_mu_p = np.sqrt(mu / p_curr)

        # Simplified derivatives (without full perturbations for guess)
        dp_dt = (
            (2 * p_curr / q_curr)
            * sqrt_p_mu
            * (gs * T * (1 + 0.01 * tau_guess) / w_curr)
            * ut_traj[i]
        )

        dL_dt = sqrt_mu_p * (q_curr / p_curr) ** 2

        dw_dt = -(T * (1 + 0.01 * tau_guess) / Isp)

        # Simple integration
        p_traj[i + 1] = p_curr + dp_dt * dt
        f_traj[i + 1] = f_curr  # Keep approximately constant for guess
        g_traj[i + 1] = g_curr  # Keep approximately constant for guess
        h_traj[i + 1] = h_curr  # Keep approximately constant for guess
        k_traj[i + 1] = k_curr  # Keep approximately constant for guess
        L_traj[i + 1] = L_curr + dL_dt * dt
        w_traj[i + 1] = w_curr + dw_dt * dt

        # Enforce bounds
        p_traj[i + 1] = np.clip(p_traj[i + 1], pmin, pmax)
        w_traj[i + 1] = np.clip(w_traj[i + 1], wmin, wmax)

    return (
        p_traj,
        f_traj,
        g_traj,
        h_traj,
        k_traj,
        L_traj,
        w_traj,
        ur_traj,
        ut_traj,
        uh_traj,
        t_guess,
    )


# Generate physics-based guess
(
    p_guess,
    f_guess,
    g_guess,
    h_guess,
    k_guess,
    L_guess,
    w_guess,
    ur_guess,
    ut_guess,
    uh_guess,
    tf_guess,
) = generate_physics_based_guess()

# Distribute guess across mesh intervals
states_guess = []
controls_guess = []
poly_degrees = [8, 8, 8, 8, 8, 8, 8, 8]
start_idx = 0

for N in poly_degrees:
    end_idx = start_idx + N + 1

    # Extract state points for this interval
    p_interval = p_guess[start_idx:end_idx]
    f_interval = f_guess[start_idx:end_idx]
    g_interval = g_guess[start_idx:end_idx]
    h_interval = h_guess[start_idx:end_idx]
    k_interval = k_guess[start_idx:end_idx]
    L_interval = L_guess[start_idx:end_idx]
    w_interval = w_guess[start_idx:end_idx]

    states_guess.append(
        np.vstack(
            [p_interval, f_interval, g_interval, h_interval, k_interval, L_interval, w_interval]
        )
    )

    # Extract control points for this interval (N points, not N+1)
    ur_interval = ur_guess[start_idx : start_idx + N]
    ut_interval = ut_guess[start_idx : start_idx + N]
    uh_interval = uh_guess[start_idx : start_idx + N]

    controls_guess.append(np.vstack([ur_interval, ut_interval, uh_interval]))

    start_idx = end_idx - 1  # Overlap by 1 point

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_terminal_times={1: tf_guess},
    static_parameters=np.array([-25]),
)

# Solve with robust solver configuration
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-4,  # Relaxed tolerance initially
    max_iterations=15,
    min_polynomial_degree=4,
    max_polynomial_degree=12,
    nlp_options={
        "ipopt.print_level": 5,  # Reduced output for clarity
        "ipopt.max_iter": 5000,
        "ipopt.tol": 1e-6,  # Relaxed tolerance
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.linear_solver": "mumps",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.obj_scaling_factor": 1e3,  # Scale objective for better conditioning
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-9,
        "ipopt.warm_start_mult_bound_push": 1e-9,
        "ipopt.mu_init": 1e-1,  # Larger initial barrier parameter
        "ipopt.adaptive_mu_globalization": "kkt-error",
        "ipopt.acceptable_tol": 1e-4,  # Accept looser solution if needed
        "ipopt.acceptable_iter": 15,
    },
)

# Results
if solution.status["success"]:
    final_mass = solution[(1, "mass")][-1]
    final_time = solution.phases[1]["times"]["final"]

    print(f"Final mass: {final_mass:.6f}")
    print(f"Transfer time: {final_time:.1f} seconds ({final_time / 3600:.2f} hours)")
    print(f"Objective (negative final mass): {solution.status['objective']:.6f}")

    # Final orbital elements
    pf_actual = solution[(1, "semi_latus_rectum")][-1]
    ff_actual = solution[(1, "f_eccentricity")][-1]
    gf_actual = solution[(1, "g_eccentricity")][-1]
    hf_actual = solution[(1, "h_inclination")][-1]
    kf_actual = solution[(1, "k_inclination")][-1]

    print("\nFinal orbital elements:")
    print(f"p: {pf_actual:.1f} ft (target: {pf:.1f} ft)")
    print(f"f: {ff_actual:.6f}")
    print(f"g: {gf_actual:.6f}")
    print(f"h: {hf_actual:.6f}")
    print(f"k: {kf_actual:.6f}")

    # Verify constraints
    eccentricity_sq = ff_actual**2 + gf_actual**2
    inclination_sq = hf_actual**2 + kf_actual**2
    print("\nConstraint verification:")
    print(f"f² + g²: {eccentricity_sq:.6f} (target: {ff_target_sq:.6f})")
    print(f"h² + k²: {inclination_sq:.6f} (target: {hf_target_sq:.6f})")

    solution.plot()
else:
    print(f"Optimization failed: {solution.status['message']}")
