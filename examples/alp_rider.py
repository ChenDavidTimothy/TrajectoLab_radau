"""
TrajectoLab Example: Alp Rider Problem
"""

# ===============================================================================
# DEBUG CODE - Add this at the very top, before any trajectolab imports
# ===============================================================================
import logging


# Set up detailed logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")


# Create a debug wrapper for the ODE solver to trace when it's called
class DebugODESolverWrapper:
    def __init__(self, actual_solver, name="ODE Solver"):
        self.actual_solver = actual_solver
        self.name = name
        self.call_count = 0

    def __call__(self, fun, t_span, y0, t_eval=None, **kwargs):
        self.call_count += 1
        print(f"\n{'=' * 80}")
        print(f"🚨 {self.name} CALLED (Call #{self.call_count})")
        print(f"{'=' * 80}")
        print(f"t_span: {t_span}")
        print(f"y0: {y0}")
        print(f"t_eval: {t_eval}")
        print(f"kwargs: {kwargs}")

        # Check for our invalid settings
        method = kwargs.get("method", "DEFAULT")
        rtol = kwargs.get("rtol", "DEFAULT")
        atol = kwargs.get("atol", "DEFAULT")
        max_step = kwargs.get("max_step", "DEFAULT")

        print("\n🔍 PARAMETER ANALYSIS:")
        print(f"   method: {method}")
        print(f"   rtol: {rtol}")
        print(f"   atol: {atol}")
        print(f"   max_step: {max_step}")

        # Check if our invalid values made it through
        if method == "this is gibberish":
            print("✅ INVALID METHOD DETECTED! Should fail now...")
        if isinstance(rtol, (int, float)) and rtol < 0:
            print("✅ NEGATIVE RTOL DETECTED! Should fail now...")
        if isinstance(atol, (int, float)) and atol < 0:
            print("✅ NEGATIVE ATOL DETECTED! Should fail now...")
        if isinstance(max_step, (int, float)) and max_step < 0:
            print("✅ NEGATIVE MAX_STEP DETECTED! Should fail now...")

        print("\n🏃 Calling actual scipy.integrate.solve_ivp...")
        print(f"{'=' * 80}")

        # Call the actual solver - any configuration errors should happen here
        try:
            result = self.actual_solver(fun, t_span, y0, t_eval=t_eval, **kwargs)
            print("✅ ODE solve completed successfully")
            return result
        except Exception as e:
            print(f"💥 ODE solve FAILED with: {type(e).__name__}: {e}")
            raise  # Re-raise to let the error bubble up


print("🔍 Debug instrumentation ready!")

# ===============================================================================
# END DEBUG CODE - Now import trajectolab
# ===============================================================================

import casadi as ca
import numpy as np

import trajectolab as tl

# NOW monkey patch after trajectolab is imported
from trajectolab.adaptive.phs.data_structures import AdaptiveParameters


# Monkey patch the AdaptiveParameters.get_ode_solver method
original_get_ode_solver = AdaptiveParameters.get_ode_solver


def debug_get_ode_solver(self):
    print("\n🔧 AdaptiveParameters.get_ode_solver() called")
    print(f"   ode_method: {self.ode_method}")
    print(f"   ode_solver_tolerance: {self.ode_solver_tolerance}")
    print(f"   ode_atol_factor: {self.ode_atol_factor}")
    print(f"   ode_max_step: {self.ode_max_step}")

    # Call original method to get the configured solver
    configured_solver = original_get_ode_solver(self)

    # Wrap it with our debug wrapper
    return DebugODESolverWrapper(configured_solver, "Configured ODE Solver")


# Apply the monkey patch
AdaptiveParameters.get_ode_solver = debug_get_ode_solver

# Also add a way to check if error estimation is being called at all
import trajectolab.adaptive.phs.error_estimation as error_est


original_simulate = error_est.simulate_dynamics_for_phase_interval_error_estimation


def debug_simulate_dynamics(*args, **kwargs):
    print("\n📊 ERROR ESTIMATION CALLED!")
    print("   Function: simulate_dynamics_for_phase_interval_error_estimation")
    print(f"   Args: {len(args)} arguments")
    return original_simulate(*args, **kwargs)


error_est.simulate_dynamics_for_phase_interval_error_estimation = debug_simulate_dynamics

print("🔍 Debug instrumentation installed!")

# ===============================================================================
# ORIGINAL CODE CONTINUES HERE
# ===============================================================================

# Problem setup
problem = tl.Problem("Alp Rider")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0, final=20.0)
y1 = phase.state("y1", initial=2.0, final=2.0)
y2 = phase.state("y2", initial=1.0, final=3.0)
y3 = phase.state("y3", initial=2.0, final=1.0)
y4 = phase.state("y4", initial=1.0, final=-2.0)
u1 = phase.control("u1")
u2 = phase.control("u2")

# Dynamics
phase.dynamics(
    {
        y1: -10 * y1 + u1 + u2,
        y2: -2 * y2 + u1 + 2 * u2,
        y3: -3 * y3 + 5 * y4 + u1 - u2,
        y4: 5 * y3 - 3 * y4 + u1 + 3 * u2,
    }
)


# Path constraint: y1² + y2² + y3² + y4² ≥ terrain following function
def p_function(t_val, a, b):
    return ca.exp(-b * (t_val - a) ** 2)


terrain_function = (
    3 * p_function(t, 3, 12)
    + 3 * p_function(t, 6, 10)
    + 3 * p_function(t, 10, 6)
    + 8 * p_function(t, 15, 4)
    + 0.01
)

state_norm_squared = y1**2 + y2**2 + y3**2 + y4**2
phase.subject_to(state_norm_squared >= terrain_function)

# Objective
integrand = 100 * (y1**2 + y2**2 + y3**2 + y4**2) + 0.01 * (u1**2 + u2**2)
integral_var = phase.add_integral(integrand)
problem.minimize(integral_var)

# Mesh and guess
phase.mesh([12, 12, 12], [-1.0, -1 / 3, 1 / 3, 1.0])

states_guess = []
controls_guess = []
for N in [12, 12, 12]:
    tau = np.linspace(-1, 1, N + 1)
    t_norm = (tau + 1) / 2

    # Linear interpolation between initial and final conditions
    y1_vals = 2.0 + (2.0 - 2.0) * t_norm  # 2 to 2
    y2_vals = 1.0 + (3.0 - 1.0) * t_norm  # 1 to 3
    y3_vals = 2.0 + (1.0 - 2.0) * t_norm  # 2 to 1
    y4_vals = 1.0 + (-2.0 - 1.0) * t_norm  # 1 to -2

    states_guess.append(np.vstack([y1_vals, y2_vals, y3_vals, y4_vals]))
    controls_guess.append(np.vstack([np.zeros(N), np.zeros(N)]))

problem.guess(
    phase_states={1: states_guess},
    phase_controls={1: controls_guess},
    phase_integrals={1: 2000.0},
)

# Solve
solution = tl.solve_adaptive(
    problem,
    error_tolerance=1e-4,
    max_iterations=20,
    min_polynomial_degree=4,
    max_polynomial_degree=15,
    ode_method="this is gibberish",
    ode_atol_factor=-1468454846548,
    ode_max_step=-25158454865153,
    ode_solver_tolerance=-14864564,
    nlp_options={
        "ipopt.max_iter": 3000,
        "ipopt.mumps_pivtol": 5e-7,
        "ipopt.mumps_mem_percent": 50000,
        "ipopt.linear_solver": "mumps",
        "ipopt.constr_viol_tol": 1e-7,
        "ipopt.print_level": 0,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.check_derivatives_for_naninf": "yes",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-8,
    },
)

# Results
if solution.success:
    print(f"Objective: {solution.objective:.5f}")
    print(
        f"Reference: 2030.85609 (Error: {abs(solution.objective - 2030.85609) / 2030.85609 * 100:.3f}%)"
    )

    # Final state values
    y1_final = solution[(1, "y1")][-1]
    y2_final = solution[(1, "y2")][-1]
    y3_final = solution[(1, "y3")][-1]
    y4_final = solution[(1, "y4")][-1]
    print(
        f"Final states: y1={y1_final:.6f}, y2={y2_final:.6f}, y3={y3_final:.6f}, y4={y4_final:.6f}"
    )

    solution.plot()
else:
    print(f"Failed: {solution.message}")
