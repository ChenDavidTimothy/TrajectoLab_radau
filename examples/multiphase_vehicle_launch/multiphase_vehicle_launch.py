import casadi as ca
import numpy as np

import maptor as mtor


def _oe2rv(orbital_elements, mu):
    """Convert orbital elements to position and velocity vectors (physical units)."""
    a, e, i, Om, om, nu = orbital_elements

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
    """Compute cross product of two 3D vectors."""
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot_product(a, b):
    """Compute dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _smooth_heaviside(x, steepness=10.0):
    """Smooth approximation of Heaviside step function using hyperbolic tangent."""
    return 0.5 * (1 + ca.tanh(steepness * x))


def _rv_to_orbital_elements(rv, vv, mu):
    """Convert position and velocity to orbital elements using numerically safe operations."""
    K = ca.vertcat(0.0, 0.0, 1.0)
    hv = _cross_product(rv, vv)
    nv = _cross_product(K, hv)

    n = ca.sqrt(ca.fmax(_dot_product(nv, nv), EPS))
    h2 = ca.fmax(_dot_product(hv, hv), EPS)
    v2 = ca.fmax(_dot_product(vv, vv), EPS)
    r = ca.sqrt(ca.fmax(_dot_product(rv, rv), EPS))

    rv_dot_vv = _dot_product(rv, vv)
    mu_safe = ca.fmax(mu, EPS)
    r_safe = ca.fmax(r, EPS)

    ev = ca.vertcat(
        (1 / mu_safe) * ((v2 - mu_safe / r_safe) * rv[0] - rv_dot_vv * vv[0]),
        (1 / mu_safe) * ((v2 - mu_safe / r_safe) * rv[1] - rv_dot_vv * vv[1]),
        (1 / mu_safe) * ((v2 - mu_safe / r_safe) * rv[2] - rv_dot_vv * vv[2]),
    )

    p = h2 / mu_safe
    e = ca.sqrt(ca.fmax(_dot_product(ev, ev), EPS))
    e_safe = ca.fmax(e, EPS)
    e_clamped = ca.fmin(e_safe, 0.999)
    a = p / (1 - e_clamped * e_clamped)

    h_mag = ca.sqrt(h2)
    h_mag_safe = ca.fmax(h_mag, EPS)
    cos_i = ca.fmax(ca.fmin(hv[2] / h_mag_safe, 1.0 - EPS), -1.0 + EPS)
    i = ca.acos(cos_i)

    steepness = 10.0

    n_safe = ca.fmax(n, EPS)
    cos_Om = ca.fmax(ca.fmin(nv[0] / n_safe, 1.0 - EPS), -1.0 + EPS)
    Om_base = ca.acos(cos_Om)
    Om_alternative = 2 * np.pi - Om_base
    H_Om = _smooth_heaviside(nv[1], steepness)
    Om = H_Om * Om_base + (1 - H_Om) * Om_alternative

    nv_dot_ev = _dot_product(nv, ev)
    cos_om = ca.fmax(ca.fmin(nv_dot_ev / (n_safe * e_safe), 1.0 - EPS), -1.0 + EPS)
    om_base = ca.acos(cos_om)
    om_alternative = 2 * np.pi - om_base
    H_om = _smooth_heaviside(ev[2], steepness)
    om = H_om * om_base + (1 - H_om) * om_alternative

    ev_dot_rv = _dot_product(ev, rv)
    cos_nu = ca.fmax(ca.fmin(ev_dot_rv / (e_safe * r_safe), 1.0 - EPS), -1.0 + EPS)
    nu_base = ca.acos(cos_nu)
    nu_alternative = 2 * np.pi - nu_base
    H_nu = _smooth_heaviside(rv_dot_vv, steepness)
    nu = H_nu * nu_base + (1 - H_nu) * nu_alternative

    return ca.vertcat(a, e, i, Om, om, nu)


# Physical constants - EXACT PSOPT VALUES
MU = 3.986012e14
RE = 6378145.0
OMEGA = 7.29211585e-5
RHO0 = 1.225
H = 7200.0
CD = 0.5
SA = 4 * np.pi
G0 = 9.80665

# Scaling factors (NO TIME SCALING - like C++ PSOPT)
R_SCALE = 1e6
V_SCALE = 1e3
M_SCALE = 1e4

# Numerical safety epsilon
EPS = 1e-12

# Earth rotation matrix
OMEGA_MATRIX = ca.MX(3, 3)
OMEGA_MATRIX[0, 0] = 0.0
OMEGA_MATRIX[0, 1] = -OMEGA
OMEGA_MATRIX[0, 2] = 0.0
OMEGA_MATRIX[1, 0] = OMEGA
OMEGA_MATRIX[1, 1] = 0.0
OMEGA_MATRIX[1, 2] = 0.0
OMEGA_MATRIX[2, 0] = 0.0
OMEGA_MATRIX[2, 1] = 0.0
OMEGA_MATRIX[2, 2] = 0.0

# Propulsion parameters - EXACT PSOPT VALUES
THRUST_SRB = 628500.0
THRUST_FIRST = 1083100.0
THRUST_SECOND = 110094.0

# Stage characteristics - EXACT PSOPT VALUES
BT_SRB = 75.2
BT_FIRST = 261.0
BT_SECOND = 700.0

# Mass parameters (kg) - EXACT PSOPT VALUES
M_TOT_SRB = 19290.0
M_PROP_SRB = 17010.0
M_DRY_SRB = M_TOT_SRB - M_PROP_SRB

M_TOT_FIRST = 104380.0
M_PROP_FIRST = 95550.0
M_DRY_FIRST = M_TOT_FIRST - M_PROP_FIRST

M_TOT_SECOND = 19300.0
M_PROP_SECOND = 16820.0
M_DRY_SECOND = M_TOT_SECOND - M_PROP_SECOND

M_PAYLOAD = 4164.0

# Specific impulse calculations - EXACT PSOPT LOGIC
MDOT_SRB = M_PROP_SRB / BT_SRB
ISP_SRB = THRUST_SRB / (G0 * MDOT_SRB)

MDOT_FIRST = M_PROP_FIRST / BT_FIRST
ISP_FIRST = THRUST_FIRST / (G0 * MDOT_FIRST)

MDOT_SECOND = M_PROP_SECOND / BT_SECOND
ISP_SECOND = THRUST_SECOND / (G0 * MDOT_SECOND)

# Initial conditions (Cape Canaveral) - EXACT PSOPT VALUES
LAT0 = 28.5 * np.pi / 180.0
X0 = RE * np.cos(LAT0)
Y0 = 0.0
Z0 = RE * np.sin(LAT0)

# Convert to scaled values
R0_values = [X0 / R_SCALE, Y0 / R_SCALE, Z0 / R_SCALE]

# Calculate initial velocity due to Earth rotation (scaled)
OMEGA_MATRIX_np = np.array([[0.0, -OMEGA, 0.0], [OMEGA, 0.0, 0.0], [0.0, 0.0, 0.0]])
V0_phys = OMEGA_MATRIX_np @ np.array([X0, Y0, Z0])
V0_values = [V0_phys[0] / V_SCALE, V0_phys[1] / V_SCALE, V0_phys[2] / V_SCALE]

# Target orbital elements - EXACT PSOPT VALUES
AF = 24361140.0
EF = 0.7308
INCF = 28.5 * np.pi / 180.0
OMF = 269.8 * np.pi / 180.0
OMEGAF = 130.5 * np.pi / 180.0
NU_GUESS = 0.0  # From C++ PSOPT

# Convert target orbital elements to position/velocity
target_oe = [AF, EF, INCF, OMF, OMEGAF, NU_GUESS]
R_TARGET_phys, V_TARGET_phys = _oe2rv(target_oe, MU)

# Scale target coordinates
R_TARGET_values = [
    R_TARGET_phys[0] / R_SCALE,
    R_TARGET_phys[1] / R_SCALE,
    R_TARGET_phys[2] / R_SCALE,
]
V_TARGET_values = [
    V_TARGET_phys[0] / V_SCALE,
    V_TARGET_phys[1] / V_SCALE,
    V_TARGET_phys[2] / V_SCALE,
]

# Phase timing - EXACT PSOPT VALUES (NO SCALING)
T0 = 0.0
T1 = 75.2
T2 = 150.4
T3 = 261.0
T4 = 961.0

# Initial masses for each phase (scaled) - EXACT PSOPT CALCULATIONS
M10 = (M_PAYLOAD + M_TOT_SECOND + M_TOT_FIRST + 9 * M_TOT_SRB) / M_SCALE
M1F = (M10 * M_SCALE - (6 * MDOT_SRB + MDOT_FIRST) * T1) / M_SCALE
M20 = (M1F * M_SCALE - 6 * M_DRY_SRB) / M_SCALE
M2F = (M20 * M_SCALE - (3 * MDOT_SRB + MDOT_FIRST) * (T2 - T1)) / M_SCALE
M30 = (M2F * M_SCALE - 3 * M_DRY_SRB) / M_SCALE
M3F = (M30 * M_SCALE - MDOT_FIRST * (T3 - T2)) / M_SCALE
M40 = (M3F * M_SCALE - M_DRY_FIRST) / M_SCALE
M4F = M_PAYLOAD / M_SCALE

# Bounds (scaled) - EXACT PSOPT VALUES
RMIN = -2 * RE / R_SCALE
RMAX = 2 * RE / R_SCALE
VMIN = -10000.0 / V_SCALE
VMAX = 10000.0 / V_SCALE

# Problem setup
problem = mtor.Problem("Multiphase Vehicle Launch")

# Phase 1: 6 SRBs + First Stage (0 - 75.2s) - TIMES IN PHYSICAL UNITS
phase1 = problem.set_phase(1)
t1 = phase1.time(initial=T0, final=T1)
r1 = [phase1.state(f"r{i}", initial=R0_values[i], boundary=(RMIN, RMAX)) for i in range(3)]
v1 = [phase1.state(f"v{i}", initial=V0_values[i], boundary=(VMIN, VMAX)) for i in range(3)]
m1 = phase1.state("mass", initial=M10, boundary=(M1F, M10))
u1 = [phase1.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]

# Phase 2: 3 SRBs + First Stage (75.2 - 150.4s)
phase2 = problem.set_phase(2)
t2 = phase2.time(initial=T1, final=T2)
r2 = [phase2.state(f"r{i}", initial=r1[i].final, boundary=(RMIN, RMAX)) for i in range(3)]
v2 = [phase2.state(f"v{i}", initial=v1[i].final, boundary=(VMIN, VMAX)) for i in range(3)]
m2 = phase2.state("mass", initial=m1.final - 6 * M_DRY_SRB / M_SCALE, boundary=(M2F, M20))
u2 = [phase2.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]

# Phase 3: First Stage Only (150.4 - 261.0s)
phase3 = problem.set_phase(3)
t3 = phase3.time(initial=T2, final=T3)
r3 = [phase3.state(f"r{i}", initial=r2[i].final, boundary=(RMIN, RMAX)) for i in range(3)]
v3 = [phase3.state(f"v{i}", initial=v2[i].final, boundary=(VMIN, VMAX)) for i in range(3)]
m3 = phase3.state("mass", initial=m2.final - 3 * M_DRY_SRB / M_SCALE, boundary=(M3F, M30))
u3 = [phase3.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]

# Phase 4: Second Stage Only (261.0 - 961.0s)
phase4 = problem.set_phase(4)
t4 = phase4.time(initial=T3, final=(T3, T4))
r4 = [phase4.state(f"r{i}", initial=r3[i].final, boundary=(RMIN, RMAX)) for i in range(3)]
v4 = [phase4.state(f"v{i}", initial=v3[i].final, boundary=(VMIN, VMAX)) for i in range(3)]
m4 = phase4.state("mass", initial=m3.final - M_DRY_FIRST / M_SCALE, boundary=(M4F, M40))
u4 = [phase4.control(f"u{i}", boundary=(-1.0, 1.0)) for i in range(3)]


def _define_phase_dynamics(phase, r_vars, v_vars, m_var, u_vars, phase_num):
    """Define dynamics for a given phase with CORRECT manual scaling (NO TIME SCALING)."""
    r_vec_scaled = ca.vertcat(*r_vars)
    v_vec_scaled = ca.vertcat(*v_vars)
    m_scaled = m_var
    u_vec = ca.vertcat(*u_vars)

    # Convert to physical units for dynamics calculations
    r_vec = r_vec_scaled * R_SCALE
    v_vec = v_vec_scaled * V_SCALE
    m_phys = m_scaled * M_SCALE

    m_safe = ca.fmax(m_phys, EPS)

    # Relative velocity (accounting for Earth rotation)
    v_rel = v_vec - OMEGA_MATRIX @ r_vec
    speed_rel_squared = ca.fmax(_dot_product(v_rel, v_rel), EPS)
    speed_rel = ca.sqrt(speed_rel_squared)

    # Safe radius and altitude calculations
    rad_squared = ca.fmax(_dot_product(r_vec, r_vec), EPS)
    rad = ca.sqrt(rad_squared)
    rad_safe = ca.fmax(rad, RE + 1000.0)
    altitude = rad_safe - RE
    altitude_safe = ca.fmax(altitude, 0.0)

    # Atmospheric density
    rho = RHO0 * ca.exp(-ca.fmin(altitude_safe / H, 50.0))

    # Drag force (physical units)
    bc = (rho / (2 * m_safe)) * SA * CD
    bc_speed = bc * speed_rel
    drag = -v_rel * bc_speed

    # Gravitational force (physical units)
    mu_over_rad_cubed = MU / (rad_safe**3)
    grav = -mu_over_rad_cubed * r_vec

    # Thrust and mass flow (physical units) - EXACT PSOPT LOGIC
    if phase_num == 1:
        T_total = 6 * THRUST_SRB + THRUST_FIRST
        mdot = -(6 * THRUST_SRB / (G0 * ISP_SRB) + THRUST_FIRST / (G0 * ISP_FIRST))
    elif phase_num == 2:
        T_total = 3 * THRUST_SRB + THRUST_FIRST
        mdot = -(3 * THRUST_SRB / (G0 * ISP_SRB) + THRUST_FIRST / (G0 * ISP_FIRST))
    elif phase_num == 3:
        T_total = THRUST_FIRST
        mdot = -THRUST_FIRST / (G0 * ISP_FIRST)
    elif phase_num == 4:
        T_total = THRUST_SECOND
        mdot = -THRUST_SECOND / (G0 * ISP_SECOND)

    # Thrust vector (physical units)
    T_over_m = T_total / m_safe
    thrust = T_over_m * u_vec

    # Total acceleration (physical units)
    acceleration = thrust + drag + grav

    # CORRECT manual scaling (NO TIME SCALING - like C++ PSOPT)
    # dr_scaled/dt = dr_physical/dt * (1/R_SCALE) = v_physical/R_SCALE
    r_dot_scaled = v_vec / R_SCALE

    # dv_scaled/dt = dv_physical/dt * (1/V_SCALE) = acceleration/V_SCALE
    v_dot_scaled = acceleration / V_SCALE

    # dm_scaled/dt = dm_physical/dt * (1/M_SCALE) = mdot/M_SCALE
    m_dot_scaled = mdot / M_SCALE

    # Set dynamics
    dynamics_dict = {}
    for i in range(3):
        dynamics_dict[r_vars[i]] = r_dot_scaled[i]
        dynamics_dict[v_vars[i]] = v_dot_scaled[i]
    dynamics_dict[m_var] = m_dot_scaled

    phase.dynamics(dynamics_dict)

    # Path constraint: unit thrust vector
    thrust_magnitude_squared = ca.fmax(_dot_product(u_vec, u_vec), EPS)
    phase.path_constraints(thrust_magnitude_squared == 1.0)


# Define dynamics for all phases
_define_phase_dynamics(phase1, r1, v1, m1, u1, 1)
_define_phase_dynamics(phase2, r2, v2, m2, u2, 2)
_define_phase_dynamics(phase3, r3, v3, m3, u3, 3)
_define_phase_dynamics(phase4, r4, v4, m4, u4, 4)

# Final orbital element constraints (Phase 4)
r_final_scaled = ca.vertcat(*[r4[i].final for i in range(3)])
v_final_scaled = ca.vertcat(*[v4[i].final for i in range(3)])

# Convert back to physical units for orbital elements calculation
r_final = r_final_scaled * R_SCALE
v_final = v_final_scaled * V_SCALE

oe_final = _rv_to_orbital_elements(r_final, v_final, MU)

# Constrain the 5 target orbital elements (PSOPT does not constrain true anomaly)
phase4.event_constraints(
    oe_final[0] == AF,
    oe_final[1] == EF,
    oe_final[2] == INCF,
    oe_final[3] == OMF,
    oe_final[4] == OMEGAF,
)

# Objective: Maximize final mass
problem.minimize(-m4.final)

# Mesh configuration - Coarser mesh for realistic convergence
phase1.mesh([4, 5], [-1.0, 0.0, 1.0])
phase2.mesh([4, 5], [-1.0, 0.0, 1.0])
phase3.mesh([4, 5, 4], [-1.0, -0.3, 0.3, 1.0])
phase4.mesh([5, 6, 7], [-1.0, -0.2, 0.4, 1.0])


def _generate_phase_guess(polynomial_degrees, r_init, v_init, m_init, m_final):
    """Generate initial guess for a phase matching C++ PSOPT format."""
    states_guess = []
    controls_guess = []
    for N in polynomial_degrees:
        N_state_points = N + 1
        states = np.zeros((7, N_state_points))
        # Constant position and velocity (matching C++ linspace with same start/end)
        for i in range(3):
            states[i, :] = r_init[i]  # Constant position
            states[i + 3, :] = v_init[i]  # Constant velocity
        # Linear mass variation
        states[6, :] = np.linspace(m_init, m_final, N_state_points)
        states_guess.append(states)

        N_control_points = N
        controls = np.zeros((3, N_control_points))
        controls[0, :] = 1.0  # [1, 0, 0] control guess
        # controls[1,:] and controls[2,:] remain 0.0
        controls_guess.append(controls)
    return states_guess, controls_guess


# CORRECTED initial guesses matching new coarser mesh
states_p1, controls_p1 = _generate_phase_guess([4, 5], R0_values, V0_values, M10, M1F)
states_p2, controls_p2 = _generate_phase_guess([4, 5], R0_values, V0_values, M20, M2F)
states_p3, controls_p3 = _generate_phase_guess([4, 5, 4], R0_values, V0_values, M30, M3F)

# Phase 4 uses TARGET orbital coordinates (CRITICAL FIX)
states_p4, controls_p4 = _generate_phase_guess(
    [5, 6, 7], R_TARGET_values, V_TARGET_values, M40, M4F
)

problem.guess(
    phase_states={1: states_p1, 2: states_p2, 3: states_p3, 4: states_p4},
    phase_controls={1: controls_p1, 2: controls_p2, 3: controls_p3, 4: controls_p4},
)

# Solve with settings appropriate for this complex problem
solution = mtor.solve_adaptive(
    problem,
    error_tolerance=1e-5,
    max_iterations=20,
    min_polynomial_degree=4,
    max_polynomial_degree=10,
    nlp_options={
        "ipopt.print_level": 5,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
    },
)

# Results
if solution.status["success"]:
    final_mass_scaled = solution[(4, "mass")][-1]
    final_mass = final_mass_scaled * M_SCALE
    mission_time = solution.status["total_mission_time"]

    print("\n--- OPTIMIZATION SUCCESSFUL ---")
    print(f"Final mass: {final_mass:.1f} kg")
    print(f"Mission time: {mission_time:.1f} seconds")
    print(f"Payload fraction: {final_mass / (M10 * M_SCALE) * 100:.2f}%")

    solution.plot()
else:
    print("\n--- OPTIMIZATION FAILED ---")
    print(f"Failed: {solution.status['message']}")
