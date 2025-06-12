import casadi as ca
import numpy as np

import maptor as mtor


def _oe2rv(oe, mu):
    """Convert orbital elements to position and velocity vectors (PSOPT exact)."""
    a, e, i, Om, om, nu = oe
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(nu))

    # Position in perifocal frame
    rv_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])

    # Velocity in perifocal frame
    vv_pf = np.array([-np.sin(nu), e + np.cos(nu), 0.0]) * np.sqrt(mu / p)

    # Rotation matrix from perifocal to inertial frame
    cO, sO = np.cos(Om), np.sin(Om)
    co, so = np.cos(om), np.sin(om)
    ci, si = np.cos(i), np.sin(i)

    R = np.array(
        [
            [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si, co * si, ci],
        ]
    )

    ri = R @ rv_pf
    vi = R @ vv_pf

    return ri, vi


def _cross_product(a, b):
    """Cross product of two 3D vectors."""
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )


def _dot_product(a, b):
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _smooth_heaviside(x, a_eps=0.1):
    """Smooth Heaviside function using tanh (PSOPT exact)."""
    return 0.5 * (1 + ca.tanh(x / a_eps))


def _rv2oe(rv, vv, mu):
    """Convert position and velocity to orbital elements (PSOPT exact with smooth Heaviside)."""
    eps = 1e-12

    K = ca.vertcat(0.0, 0.0, 1.0)
    hv = _cross_product(rv, vv)
    nv = _cross_product(K, hv)

    n = ca.sqrt(ca.fmax(_dot_product(nv, nv), eps))
    h2 = ca.fmax(_dot_product(hv, hv), eps)
    v2 = ca.fmax(_dot_product(vv, vv), eps)
    r = ca.sqrt(ca.fmax(_dot_product(rv, rv), eps))

    rv_dot_vv = _dot_product(rv, vv)

    ev = ca.vertcat(
        (1 / mu) * ((v2 - mu / r) * rv[0] - rv_dot_vv * vv[0]),
        (1 / mu) * ((v2 - mu / r) * rv[1] - rv_dot_vv * vv[1]),
        (1 / mu) * ((v2 - mu / r) * rv[2] - rv_dot_vv * vv[2]),
    )

    p = h2 / mu
    e = ca.sqrt(ca.fmax(_dot_product(ev, ev), eps))
    a = p / (1 - e * e)
    i = ca.acos(ca.fmax(ca.fmin(hv[2] / ca.sqrt(h2), 1.0 - eps), -1.0 + eps))

    # PSOPT smooth Heaviside implementation for RAAN
    a_eps = 0.1
    nv_dot_ev = _dot_product(nv, ev)

    Om = _smooth_heaviside(nv[1] + eps, a_eps) * ca.acos(
        ca.fmax(ca.fmin(nv[0] / n, 1.0 - eps), -1.0 + eps)
    ) + _smooth_heaviside(-(nv[1] + eps), a_eps) * (
        2 * np.pi - ca.acos(ca.fmax(ca.fmin(nv[0] / n, 1.0 - eps), -1.0 + eps))
    )

    # PSOPT smooth Heaviside implementation for argument of periapsis
    om = _smooth_heaviside(ev[2], a_eps) * ca.acos(
        ca.fmax(ca.fmin(nv_dot_ev / (n * e), 1.0 - eps), -1.0 + eps)
    ) + _smooth_heaviside(-ev[2], a_eps) * (
        2 * np.pi - ca.acos(ca.fmax(ca.fmin(nv_dot_ev / (n * e), 1.0 - eps), -1.0 + eps))
    )

    # PSOPT smooth Heaviside implementation for true anomaly
    ev_dot_rv = _dot_product(ev, rv)
    nu = _smooth_heaviside(rv_dot_vv, a_eps) * ca.acos(
        ca.fmax(ca.fmin(ev_dot_rv / (e * r), 1.0 - eps), -1.0 + eps)
    ) + _smooth_heaviside(-rv_dot_vv, a_eps) * (
        2 * np.pi - ca.acos(ca.fmax(ca.fmin(ev_dot_rv / (e * r), 1.0 - eps), -1.0 + eps))
    )

    return ca.vertcat(a, e, i, Om, om, nu)


# PSOPT exact constants
omega = 7.29211585e-5  # Earth rotation rate (rad/s)
mu = 3.986012e14  # Gravitational parameter (m^3/s^2)
cd = 0.5  # Drag coefficient
sa = 4 * np.pi  # Surface area (m^2)
rho0 = 1.225  # Sea level density (kg/m^3)
H = 7200.0  # Density scale height (m)
Re = 6378145.0  # Radius of earth (m)
g0 = 9.80665  # Sea level gravity (m/s^2)

# Scaling factors for numerical conditioning (NO TIME SCALING)
R_SCALE = 1e6  # Position scaling (m)
V_SCALE = 1e3  # Velocity scaling (m/s)
M_SCALE = 1e4  # Mass scaling (kg)

# Earth rotation matrix
OMEGA_MATRIX = ca.MX(3, 3)
OMEGA_MATRIX[0, 0] = 0.0
OMEGA_MATRIX[0, 1] = -omega
OMEGA_MATRIX[0, 2] = 0.0
OMEGA_MATRIX[1, 0] = omega
OMEGA_MATRIX[1, 1] = 0.0
OMEGA_MATRIX[1, 2] = 0.0
OMEGA_MATRIX[2, 0] = 0.0
OMEGA_MATRIX[2, 1] = 0.0
OMEGA_MATRIX[2, 2] = 0.0

# Initial conditions (Cape Canaveral) - PSOPT exact
lat0 = 28.5 * np.pi / 180.0
x0 = Re * np.cos(lat0)
z0 = Re * np.sin(lat0)
y0 = 0.0
r0_vals = [x0 / R_SCALE, y0 / R_SCALE, z0 / R_SCALE]  # Scaled

# Initial velocity due to Earth rotation
omega_matrix_np = np.array([[0.0, -omega, 0.0], [omega, 0.0, 0.0], [0.0, 0.0, 0.0]])
v0_phys = omega_matrix_np @ np.array([x0, y0, z0])
v0_vals = [v0_phys[0] / V_SCALE, v0_phys[1] / V_SCALE, v0_phys[2] / V_SCALE]  # Scaled

# PSOPT exact timing (NO SCALING)
t0, t1, t2, t3, t4 = 0.0, 75.2, 150.4, 261.0, 961.0

# PSOPT exact mass parameters
m_tot_srb = 19290.0
m_prop_srb = 17010.0
m_dry_srb = m_tot_srb - m_prop_srb
m_tot_first = 104380.0
m_prop_first = 95550.0
m_dry_first = m_tot_first - m_prop_first
m_tot_second = 19300.0
m_prop_second = 16820.0
m_dry_second = m_tot_second - m_prop_second
m_payload = 4164.0

# PSOPT exact thrust parameters
thrust_srb = 628500.0
thrust_first = 1083100.0
thrust_second = 110094.0
mdot_srb = m_prop_srb / t1
mdot_first = m_prop_first / (t3 - t0)
mdot_second = m_prop_second / 700.0
ISP_srb = thrust_srb / (g0 * mdot_srb)
ISP_first = thrust_first / (g0 * mdot_first)
ISP_second = thrust_second / (g0 * mdot_second)

# PSOPT exact target orbital elements
af = 24361140.0
ef = 0.7308
incf = 28.5 * np.pi / 180.0
Omf = 269.8 * np.pi / 180.0
omf = 130.5 * np.pi / 180.0
nuguess = 0.0

# Convert target orbital elements to position/velocity and scale
target_oe = [af, ef, incf, Omf, omf, nuguess]
rout_phys, vout_phys = _oe2rv(target_oe, mu)
rout = rout_phys / R_SCALE  # Scaled
vout = vout_phys / V_SCALE  # Scaled

# PSOPT exact initial masses (scaled)
m10 = (m_payload + m_tot_second + m_tot_first + 9 * m_tot_srb) / M_SCALE
m1f = (m10 * M_SCALE - (6 * mdot_srb + mdot_first) * t1) / M_SCALE
m20 = (m1f * M_SCALE - 6 * m_dry_srb) / M_SCALE
m2f = (m20 * M_SCALE - (3 * mdot_srb + mdot_first) * (t2 - t1)) / M_SCALE
m30 = (m2f * M_SCALE - 3 * m_dry_srb) / M_SCALE
m3f = (m30 * M_SCALE - mdot_first * (t3 - t2)) / M_SCALE
m40 = (m3f * M_SCALE - m_dry_first) / M_SCALE
m4f = m_payload / M_SCALE

# PSOPT exact bounds (scaled)
rmin = -2 * Re / R_SCALE
rmax = 2 * Re / R_SCALE
vmin = -10000.0 / V_SCALE
vmax = 10000.0 / V_SCALE

# Problem setup
problem = mtor.Problem("Multiphase Vehicle Launch - PSOPT Exact Scaled")

# Phase 1: 6 SRBs + First Stage (0-75.2s)
phase1 = problem.set_phase(1)
t_1 = phase1.time(initial=t0, final=t1)  # Time not scaled
r1_s = [phase1.state(f"r{i}_scaled", initial=r0_vals[i], boundary=(rmin, rmax)) for i in range(3)]
v1_s = [phase1.state(f"v{i}_scaled", initial=v0_vals[i], boundary=(vmin, vmax)) for i in range(3)]
m1_s = phase1.state("mass_scaled", initial=m10, boundary=(m1f, m10))
u1 = [phase1.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]

# Phase 2: 3 SRBs + First Stage (75.2-150.4s)
phase2 = problem.set_phase(2)
t_2 = phase2.time(initial=t1, final=t2)  # Time not scaled
r2_s = [
    phase2.state(f"r{i}_scaled", initial=r1_s[i].final, boundary=(rmin, rmax)) for i in range(3)
]
v2_s = [
    phase2.state(f"v{i}_scaled", initial=v1_s[i].final, boundary=(vmin, vmax)) for i in range(3)
]
m2_s = phase2.state(
    "mass_scaled", initial=m1_s.final - 6 * m_dry_srb / M_SCALE, boundary=(m2f, m20)
)
u2 = [phase2.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]

# Phase 3: First Stage Only (150.4-261.0s)
phase3 = problem.set_phase(3)
t_3 = phase3.time(initial=t2, final=t3)  # Time not scaled
r3_s = [
    phase3.state(f"r{i}_scaled", initial=r2_s[i].final, boundary=(rmin, rmax)) for i in range(3)
]
v3_s = [
    phase3.state(f"v{i}_scaled", initial=v2_s[i].final, boundary=(vmin, vmax)) for i in range(3)
]
m3_s = phase3.state(
    "mass_scaled", initial=m2_s.final - 3 * m_dry_srb / M_SCALE, boundary=(m3f, m30)
)
u3 = [phase3.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]

# Phase 4: Second Stage Only (261.0-961.0s)
phase4 = problem.set_phase(4)
t_4 = phase4.time(initial=t3, final=(t3, t4))  # Time not scaled
r4_s = [
    phase4.state(f"r{i}_scaled", initial=r3_s[i].final, boundary=(rmin, rmax)) for i in range(3)
]
v4_s = [
    phase4.state(f"v{i}_scaled", initial=v3_s[i].final, boundary=(vmin, vmax)) for i in range(3)
]
m4_s = phase4.state("mass_scaled", initial=m3_s.final - m_dry_first / M_SCALE, boundary=(m4f, m40))
u4 = [phase4.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]


def _define_phase_dynamics(phase, r_vars_s, v_vars_s, m_var_s, u_vars, phase_num):
    """Define dynamics with manual scaling (NO TIME SCALING)."""
    # Convert scaled variables to physical units for dynamics calculations
    r_vec_s = ca.vertcat(*r_vars_s)
    v_vec_s = ca.vertcat(*v_vars_s)
    u_vec = ca.vertcat(*u_vars)

    r_vec = r_vec_s * R_SCALE  # Physical position
    v_vec = v_vec_s * V_SCALE  # Physical velocity
    m_phys = m_var_s * M_SCALE  # Physical mass

    rad = ca.sqrt(ca.fmax(_dot_product(r_vec, r_vec), 1e-12))

    # Relative velocity accounting for Earth rotation
    vrel = v_vec - OMEGA_MATRIX @ r_vec
    speedrel = ca.sqrt(ca.fmax(_dot_product(vrel, vrel), 1e-12))

    # Atmospheric effects (physical units)
    altitude = rad - Re
    rho = rho0 * ca.exp(-altitude / H)
    bc = (rho / (2 * m_phys)) * sa * cd
    bcspeed = bc * speedrel
    drag = -vrel * bcspeed

    # Gravitational force (physical units)
    muoverradcubed = mu / (rad**3)
    grav = -muoverradcubed * r_vec

    # Thrust and mass flow (PSOPT exact logic)
    if phase_num == 1:
        T_tot = 6 * thrust_srb + thrust_first
        mdot = -(6 * thrust_srb / (g0 * ISP_srb) + thrust_first / (g0 * ISP_first))
    elif phase_num == 2:
        T_tot = 3 * thrust_srb + thrust_first
        mdot = -(3 * thrust_srb / (g0 * ISP_srb) + thrust_first / (g0 * ISP_first))
    elif phase_num == 3:
        T_tot = thrust_first
        mdot = -thrust_first / (g0 * ISP_first)
    elif phase_num == 4:
        T_tot = thrust_second
        mdot = -thrust_second / (g0 * ISP_second)

    # Thrust vector (physical units)
    Toverm = T_tot / m_phys
    thrust = Toverm * u_vec

    # Total acceleration (physical units)
    acceleration = thrust + drag + grav

    # Manual scaling of derivatives (NO TIME SCALING)
    # dr_s/dt = (dr/dt) / R_SCALE
    r_dot_scaled = v_vec / R_SCALE

    # dv_s/dt = (dv/dt) / V_SCALE
    v_dot_scaled = acceleration / V_SCALE

    # dm_s/dt = (dm/dt) / M_SCALE
    m_dot_scaled = mdot / M_SCALE

    # Set dynamics
    dynamics_dict = {}
    for i in range(3):
        dynamics_dict[r_vars_s[i]] = r_dot_scaled[i]
        dynamics_dict[v_vars_s[i]] = v_dot_scaled[i]
    dynamics_dict[m_var_s] = m_dot_scaled

    phase.dynamics(dynamics_dict)

    # PSOPT exact path constraint: unit thrust vector
    thrust_magnitude_squared = _dot_product(u_vec, u_vec)
    phase.path_constraints(thrust_magnitude_squared == 1.0)


# Define dynamics for all phases
_define_phase_dynamics(phase1, r1_s, v1_s, m1_s, u1, 1)
_define_phase_dynamics(phase2, r2_s, v2_s, m2_s, u2, 2)
_define_phase_dynamics(phase3, r3_s, v3_s, m3_s, u3, 3)
_define_phase_dynamics(phase4, r4_s, v4_s, m4_s, u4, 4)

# PSOPT exact constraint: 5 orbital elements (NOT true anomaly)
# Convert scaled final state back to physical units for orbital elements
r_final_s = ca.vertcat(*[r4_s[i].final for i in range(3)])
v_final_s = ca.vertcat(*[v4_s[i].final for i in range(3)])
r_final = r_final_s * R_SCALE  # Physical position
v_final = v_final_s * V_SCALE  # Physical velocity

oe_final = _rv2oe(r_final, v_final, mu)

phase4.event_constraints(
    oe_final[0] == af,  # semi-major axis
    oe_final[1] == ef,  # eccentricity
    oe_final[2] == incf,  # inclination
    oe_final[3] == Omf,  # RAAN
    oe_final[4] == omf,  # argument of periapsis
    # Note: true anomaly (oe_final[5]) is NOT constrained in PSOPT
)

# PSOPT exact objective: maximize final mass
problem.minimize(-m4_s.final)

# PSOPT mesh configuration
phase1.mesh([4, 4], [-1.0, 0.0, 1.0])
phase2.mesh([4, 4], [-1.0, 0.0, 1.0])
phase3.mesh([4, 4], [-1.0, 0.0, 1.0])
phase4.mesh([4, 4], [-1.0, 0.0, 1.0])


# PSOPT exact initial guess (scaled)
def _generate_psopt_guess(N_list, r_init_s, v_init_s, m_init_s, m_final_s):
    """Generate PSOPT exact initial guess format (scaled)."""
    states_guess = []
    controls_guess = []

    for N in N_list:
        # State guess (7 states × N+1 points)
        states = np.zeros((7, N + 1))
        for i in range(3):
            states[i, :] = r_init_s[i]  # Constant scaled position
            states[i + 3, :] = v_init_s[i]  # Constant scaled velocity
        states[6, :] = np.linspace(m_init_s, m_final_s, N + 1)  # Linear scaled mass variation
        states_guess.append(states)

        # Control guess (3 controls × N points)
        controls = np.zeros((3, N))
        controls[0, :] = 1.0  # [1, 0, 0] unit vector
        controls_guess.append(controls)

    return states_guess, controls_guess


# Generate PSOPT exact guesses (scaled)
states_p1, controls_p1 = _generate_psopt_guess([4, 4], r0_vals, v0_vals, m10, m1f)
states_p2, controls_p2 = _generate_psopt_guess([4, 4], r0_vals, v0_vals, m20, m2f)
states_p3, controls_p3 = _generate_psopt_guess([4, 4], r0_vals, v0_vals, m30, m3f)
states_p4, controls_p4 = _generate_psopt_guess([4, 4], rout, vout, m40, m4f)

problem.guess(
    phase_states={1: states_p1, 2: states_p2, 3: states_p3, 4: states_p4},
    phase_controls={1: controls_p1, 2: controls_p2, 3: controls_p3, 4: controls_p4},
)

# Solve with PSOPT-equivalent settings
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-6,
    max_iterations=20,
    min_polynomial_degree=3,
    max_polynomial_degree=8,
    nlp_options={
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 5,
    },
)

# Results (convert back to physical units)
if solution.status["success"]:
    final_mass_scaled = -solution.status["objective"]  # Convert back from minimization
    final_mass = final_mass_scaled * M_SCALE  # Physical mass
    mission_time = solution.status["total_mission_time"]  # Time not scaled

    print("PSOPT-Exact MAPTOR Solution (with Manual Scaling):")
    print(f"Final mass: {final_mass:.1f} kg")
    print(f"Mission time: {mission_time:.1f} seconds")
    print(f"Payload fraction: {final_mass / (m10 * M_SCALE) * 100:.2f}%")

    # Verify final orbital elements
    r_final_scaled = (
        solution[(4, "r0_scaled")][-1],
        solution[(4, "r1_scaled")][-1],
        solution[(4, "r2_scaled")][-1],
    )
    v_final_scaled = (
        solution[(4, "v0_scaled")][-1],
        solution[(4, "v1_scaled")][-1],
        solution[(4, "v2_scaled")][-1],
    )

    print("\nFinal state verification:")
    print(
        f"Final position (scaled): [{r_final_scaled[0]:.6f}, {r_final_scaled[1]:.6f}, {r_final_scaled[2]:.6f}]"
    )
    print(
        f"Final velocity (scaled): [{v_final_scaled[0]:.6f}, {v_final_scaled[1]:.6f}, {v_final_scaled[2]:.6f}]"
    )

    solution.plot()
else:
    print(f"Failed: {solution.status['message']}")
