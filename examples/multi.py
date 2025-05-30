# examples/tumor_antiangiogenesis_multiphase.py
"""
TrajectoLab Example: Tumor Antiangiogenesis Two-Phase Optimal Control Problem
Based on Example 10.142 from optimal control literature.

Phase 1: Treatment phase with antiangiogenic therapy
Phase 2: No treatment phase (drug washout/recovery)
Objective: Minimize final tumor endothelial cell population
"""

import casadi as ca
import numpy as np

import trajectolab as tl


# Tumor model parameters from Table 10.33
xi = 0.084  # ξ - endothelial cell death rate
b = 5.85  # b - endothelial cell birth rate
d = 0.00873  # d - tumor growth parameter
G = 0.15  # G - treatment efficacy parameter
mu = 0.02  # μ - natural death rate
a = 75  # a - maximum treatment rate
A = 15  # A - maximum cumulative treatment

# Derived parameters
p_bar = ((b - mu) / d) ** (3 / 2)  # p̄ = q̄ = [(b-μ)/d]^(3/2)
q_bar = p_bar
p_0 = p_bar / 2  # p₀ = p̄/2
q_0 = q_bar / 4  # q₀ = q̄/4

print("Tumor Model Parameters:")
print(f"  ξ = {xi}, b = {b}, d = {d}")
print(f"  G = {G}, μ = {mu}, a_max = {a}, A = {A}")
print(f"  Computed: p̄ = q̄ = {p_bar:.2f}")
print(f"  Initial: p₀ = {p_0:.2f}, q₀ = {q_0:.2f}")

# Create multiphase tumor treatment problem
problem = tl.Problem("Tumor Antiangiogenesis Two-Phase Treatment")

# Phase 1: Treatment Phase (0 to t_F^(1))
with problem.phase(1) as treatment:
    print("\nDefining Phase 1: Treatment Phase")

    # Time for treatment phase (free final time with minimum bound)
    t1 = treatment.time(initial=0.0, final=(0.01, None))

    # States: endothelial cells (p), tumor cells (q), cumulative treatment (y)
    p1 = treatment.state(
        "p", initial=p_0, final=(0.01, p_bar), boundary=(0.01, p_bar)
    )  # Endothelial cells
    q1 = treatment.state(
        "q", initial=q_0, final=(0.01, q_bar), boundary=(0.01, q_bar)
    )  # Tumor cells
    y1 = treatment.state(
        "y", initial=0.0, final=(0.0, A), boundary=(0, None)
    )  # Cumulative treatment

    # Dynamics for treatment phase
    # ṗ = -ξp ln(p/q)
    # q̇ = q[b - (μ + dp^(2/3) + Ga)]
    # ẏ = a
    treatment.dynamics(
        {p1: -xi * p1 * ca.log(p1 / q1), q1: q1 * (b - (mu + d * p1 ** (2 / 3) + G * a)), y1: a}
    )

    # Mesh for treatment phase - refined for complex dynamics
    treatment.set_mesh([10, 10, 10], [-1.0, -1 / 3, 1 / 3, 1.0])

# Phase 2: No Treatment Phase (t_F^(1) to t_F^(2))
with problem.phase(2) as recovery:
    print("Defining Phase 2: Recovery Phase (No Treatment)")

    # Time for recovery phase (continues from treatment phase)
    t2 = recovery.time(initial=t1.final)

    # States: continue from treatment phase with state continuity
    p2 = recovery.state("p", initial=p1.final, final=(0.01, p_bar), boundary=(0.01, p_bar))
    q2 = recovery.state("q", initial=q1.final, final=(0.01, q_bar), boundary=(0.01, q_bar))
    y2 = recovery.state("y", initial=y1.final, final=(0.0, A), boundary=(0, None))

    # Dynamics for recovery phase (no treatment term)
    # ṗ = -ξp ln(p/q)
    # q̇ = q[b - (μ + dp^(2/3))]  [Note: no Ga term]
    # ẏ = 0  [no treatment]
    recovery.dynamics(
        {
            p2: -xi * p2 * ca.log(p2 / q2),
            q2: q2 * (b - (mu + d * p2 ** (2 / 3))),  # No treatment term
            y2: 0,  # No change in cumulative treatment
        }
    )

    # Mesh for recovery phase
    recovery.set_mesh([8, 8], [-1.0, 0.0, 1.0])

# Cross-phase constraints (state continuity - automatically handled by initial conditions)
print("Cross-phase constraints: State continuity enforced through initial conditions")

# Objective: Minimize final endothelial cell population p(t_F^(2))
problem.minimize(p2.final)

print("Objective: Minimize final endothelial cell population p(t_F^(2))")


# Solve the two-phase tumor treatment problem
print("\nSolving Two-Phase Tumor Treatment Problem...")
print("=" * 60)

solution = tl.solve_fixed_mesh(
    problem,
    nlp_options={
        "ipopt.print_level": 3,
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1e-8,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
    },
)

# Results and Analysis
if solution.success:
    print("\n" + "=" * 60)
    print("TUMOR TREATMENT OPTIMIZATION SUCCESS!")
    print("=" * 60)

    # Treatment summary
    print(f"Objective Value (Final p): {solution.objective:.6f}")
    print("Expected Optimal Value: ~7571.67 (from reference)")
    print(f"Total Treatment Duration: {solution.get_total_mission_time():.4f} time units")

    # Phase 1 results (Treatment)
    t1_duration = solution.get_phase_duration(1)
    print(f"\nPhase 1 (Treatment): {t1_duration:.4f} time units")
    print(f"  Initial endothelial cells (p): {solution[(1, 'p')][0]:.2f}")
    print(f"  Final endothelial cells (p): {solution[(1, 'p')][-1]:.2f}")
    print(f"  Initial tumor cells (q): {solution[(1, 'q')][0]:.2f}")
    print(f"  Final tumor cells (q): {solution[(1, 'q')][-1]:.2f}")
    print(f"  Total treatment given (y): {solution[(1, 'y')][-1]:.4f}")
    print(f"  Average treatment rate: {np.mean(solution[(1, 'a')]):.4f}")
    print(f"  Max treatment rate used: {np.max(solution[(1, 'a')]):.4f}")

    # Phase 2 results (Recovery)
    t2_duration = solution.get_phase_duration(2)
    print(f"\nPhase 2 (Recovery): {t2_duration:.4f} time units")
    print(f"  Initial endothelial cells (p): {solution[(2, 'p')][0]:.2f}")
    print(f"  Final endothelial cells (p): {solution[(2, 'p')][-1]:.2f}")
    print(f"  Initial tumor cells (q): {solution[(2, 'q')][0]:.2f}")
    print(f"  Final tumor cells (q): {solution[(2, 'q')][-1]:.2f}")
    print(f"  Cumulative treatment (constant): {solution[(2, 'y')][-1]:.4f}")

    # Treatment strategy analysis
    total_treatment = solution[(1, "y")][-1]
    treatment_efficiency = total_treatment / A * 100
    reduction_p = (solution[(1, "p")][0] - solution[(2, "p")][-1]) / solution[(1, "p")][0] * 100

    print("\nTreatment Strategy Analysis:")
    print(f"  Treatment utilization: {treatment_efficiency:.1f}% of maximum allowed")
    print(f"  Endothelial cell reduction: {reduction_p:.1f}%")
    print(f"  Final p/initial p ratio: {solution[(2, 'p')][-1] / solution[(1, 'p')][0]:.4f}")

    # Verify state continuity at phase boundary
    p1_final = solution[(1, "p")][-1]
    p2_initial = solution[(2, "p")][0]
    q1_final = solution[(1, "q")][-1]
    q2_initial = solution[(2, "q")][0]
    y1_final = solution[(1, "y")][-1]
    y2_initial = solution[(2, "y")][0]

    print("\nState Continuity Verification:")
    print(f"  p: {p1_final:.6f} → {p2_initial:.6f} (diff: {abs(p1_final - p2_initial):.2e})")
    print(f"  q: {q1_final:.6f} → {q2_initial:.6f} (diff: {abs(q1_final - q2_initial):.2e})")
    print(f"  y: {y1_final:.6f} → {y2_initial:.6f} (diff: {abs(y1_final - y2_initial):.2e})")

    # Plot multiphase solution
    print("\nPlotting tumor treatment trajectories...")
    solution.plot(show_phase_boundaries=True)

    # Plot individual phases for detailed analysis
    solution.plot(phase_id=1, figsize=(12, 10))  # Treatment phase
    solution.plot(phase_id=2, figsize=(12, 10))  # Recovery phase

    print("\n" + "=" * 70)
    print("- Comparison with reference optimal value J* = 7571.67158")
    print("=" * 70)

else:
    # Print problem structure for debugging
    print("\nProblem Structure:")
    print(f"  Phases: {solution.get_phase_ids()}")
    for phase_id in problem.get_phase_ids():
        num_states, num_controls = problem.get_phase_variable_counts(phase_id)
        print(f"  Phase {phase_id}: {num_states} states, {num_controls} controls")

    print("  Parameter values used:")
    print(f"    p̄ = q̄ = {p_bar:.2f}")
    print(f"    p₀ = {p_0:.2f}, q₀ = {q_0:.2f}")
