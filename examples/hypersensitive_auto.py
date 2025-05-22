import matplotlib.pyplot as plt
import numpy as np

import trajectolab as tl


# --- Copied from shuttle_auto.py: get_scaling_params_generalized ---
def get_scaling_params_generalized(
    var_name,
    explicit_lower_bound,
    explicit_upper_bound,
    initial_guess_min,
    initial_guess_max,
    output_scaling_factors_dict,
):
    vk = 1.0
    rk = 0.0
    rule_applied = "2.4 (Default)"
    if (
        explicit_lower_bound is not None
        and explicit_upper_bound is not None
        and not np.isclose(explicit_upper_bound, explicit_lower_bound)
    ):
        vk = 1.0 / (explicit_upper_bound - explicit_lower_bound)
        rk = 0.5 - explicit_upper_bound / (explicit_upper_bound - explicit_lower_bound)
        rule_applied = "2.1.a (Explicit Bounds)"
    elif (
        initial_guess_min is not None
        and initial_guess_max is not None
        and not np.isclose(initial_guess_max, initial_guess_min)
    ):
        vk = 1.0 / (initial_guess_max - initial_guess_min)
        rk = 0.5 - initial_guess_max / (initial_guess_max - initial_guess_min)
        rule_applied = "2.1.b (Initial Guess Range)"
    output_scaling_factors_dict[var_name] = {"v": vk, "r": rk, "rule": rule_applied}
    return vk, rk


# --- Copied and adapted from shuttle_auto.py: AutoscaledProblem class ---
class AutoscaledProblem:
    def __init__(self, name, initial_guess_ranges_physical=None):
        self.internal_problem = tl.Problem(name, auto_scaling=False)
        self.initial_guess_ranges_physical = (
            initial_guess_ranges_physical if initial_guess_ranges_physical else {}
        )
        self.scaling_factors = {}
        self._physical_sx = {}
        self._tilde_sx = {}
        self._state_names_ordered = []
        self._control_names_ordered = []
        self.time_symbol = None
        self.objective_scale_weight_ = 1.0

        self._raw_states_physical_guess_flat = None
        self._raw_controls_physical_guess_flat = None
        self._raw_initial_time_guess = None
        self._raw_terminal_time_guess = None
        self._raw_integrals_physical_guess = None

    def time(self, initial, final=None, free_final=None):
        if final is not None and free_final is not None:
            raise ValueError("Cannot specify both 'final' (fixed) and 'free_final'.")
        if final is not None:
            self.time_symbol = self.internal_problem.time(initial=initial, final=final)
        elif free_final is not None:
            self.time_symbol = self.internal_problem.time(initial=initial, free_final=free_final)
        else:
            raise ValueError("Must specify either 'final' (fixed time) or 'free_final'.")
        return self.time_symbol

    def _determine_and_store_scaling(self, var_name, scale_guide_L, scale_guide_U, op_L, op_U):
        guess_min = self.initial_guess_ranges_physical.get(var_name, {}).get("min")
        guess_max = self.initial_guess_ranges_physical.get(var_name, {}).get("max")
        vk, rk = get_scaling_params_generalized(
            var_name, scale_guide_L, scale_guide_U, guess_min, guess_max, self.scaling_factors
        )
        return vk, rk

    def _get_scaled_props(self, var_name, vk, rk, initial_P, final_P, lower_P, upper_P):
        scaled_def = {}
        if initial_P is not None:
            scaled_def["initial"] = vk * initial_P + rk
        if final_P is not None:
            scaled_def["final"] = vk * final_P + rk

        if self.scaling_factors[var_name]["rule"] != "2.4 (Default)":
            scaled_def["lower"] = -0.5
            scaled_def["upper"] = 0.5
        else:
            scaled_def["lower"] = (vk * lower_P + rk) if lower_P is not None else None
            scaled_def["upper"] = (vk * upper_P + rk) if upper_P is not None else None
        return scaled_def

    def state(
        self,
        name_physical,
        initial=None,
        final=None,
        lower=None,
        upper=None,
        scale_guide_lower=None,
        scale_guide_upper=None,
    ):
        vk, rk = self._determine_and_store_scaling(
            name_physical, scale_guide_lower, scale_guide_upper, lower, upper
        )
        scaled_props = self._get_scaled_props(name_physical, vk, rk, initial, final, lower, upper)

        tilde_name = f"{name_physical}_tilde"
        tilde_symbol = self.internal_problem.state(tilde_name, **scaled_props)
        self._tilde_sx[name_physical] = tilde_symbol
        if np.isclose(vk, 0):
            raise ValueError(f"Scaling factor 'v' for {name_physical} is zero.")
        self._physical_sx[name_physical] = (tilde_symbol - rk) / vk
        if name_physical not in self._state_names_ordered:
            self._state_names_ordered.append(name_physical)
        return self._physical_sx[name_physical]

    def control(
        self, name_physical, lower=None, upper=None, scale_guide_lower=None, scale_guide_upper=None
    ):
        vk, rk = self._determine_and_store_scaling(
            name_physical, scale_guide_lower, scale_guide_upper, lower, upper
        )
        scaled_props = self._get_scaled_props(name_physical, vk, rk, None, None, lower, upper)

        tilde_name = f"{name_physical}_tilde"
        tilde_symbol = self.internal_problem.control(tilde_name, **scaled_props)
        self._tilde_sx[name_physical] = tilde_symbol
        if np.isclose(vk, 0):
            raise ValueError(f"Scaling factor 'v' for {name_physical} is zero.")
        self._physical_sx[name_physical] = (tilde_symbol - rk) / vk
        if name_physical not in self._control_names_ordered:
            self._control_names_ordered.append(name_physical)
        return self._physical_sx[name_physical]

    def symbol(self, name_physical):
        if name_physical in self._physical_sx:
            return self._physical_sx[name_physical]
        raise KeyError(
            f"Physical symbol for '{name_physical}' not defined. Define it as state or control first."
        )

    def dynamics(self, dynamics_dict_physical_str_keys):
        scaled_dynamics_dict = {}
        for name_physical_state_str, physical_rhs_expr in dynamics_dict_physical_str_keys.items():
            if (
                name_physical_state_str not in self.scaling_factors
                or name_physical_state_str not in self._tilde_sx
            ):
                raise KeyError(
                    f"Scaling factors or tilde symbol for state '{name_physical_state_str}' not found."
                )
            vk = self.scaling_factors[name_physical_state_str]["v"]
            tilde_symbol_for_state = self._tilde_sx[name_physical_state_str]

            scaled_dynamics_dict[tilde_symbol_for_state] = vk * physical_rhs_expr
        self.internal_problem.dynamics(scaled_dynamics_dict)

    def add_integral(self, integrand_expr_physical):
        integral_var = self.internal_problem.add_integral(integrand_expr_physical)
        return integral_var

    def minimize(self, objective_expr_physical, objective_scale_weight=1.0):
        self.objective_scale_weight_ = objective_scale_weight
        self.internal_problem.minimize(objective_scale_weight * objective_expr_physical)

    def subject_to(self, constraint_expr_physical, constraint_scale_weight=1.0):
        self.internal_problem.subject_to(constraint_scale_weight * constraint_expr_physical)

    def set_mesh(self, degrees, points):
        self._current_mesh_degrees = degrees
        self.internal_problem.set_mesh(degrees, points)

    def set_initial_guess(
        self,
        states_physical_flat_dict=None,
        controls_physical_flat_dict=None,
        initial_time=None,
        terminal_time=None,
        integrals_physical_guess=None,
    ):
        self._raw_states_physical_guess_flat = states_physical_flat_dict
        self._raw_controls_physical_guess_flat = controls_physical_flat_dict
        self._raw_initial_time_guess = initial_time
        self._raw_terminal_time_guess = terminal_time
        self._raw_integrals_physical_guess = integrals_physical_guess

        self.internal_problem.set_initial_guess(
            states=self._raw_states_physical_guess_flat,
            controls=self._raw_controls_physical_guess_flat,
            initial_time=self._raw_initial_time_guess,
            terminal_time=self._raw_terminal_time_guess,
            integrals=self._raw_integrals_physical_guess,
        )

    def get_symbolic_vars_for_postprocessing(self):
        output = {
            "scaling_factors": self.scaling_factors,
            "objective_scale_weight": getattr(self, "objective_scale_weight_", 1.0),
        }
        if self.time_symbol is not None:
            output["t"] = self.time_symbol
        output.update(self._physical_sx)
        for name, sym in self._tilde_sx.items():  # Ensure tilde symbols are added correctly
            output[f"{name}_tilde"] = sym
        return output


# --- Main script for Hypersensitive Problem ---
def main_hypersensitive():
    initial_guess_ranges_physical = {"x": {"min": 1.0, "max": 1.5}, "u": {"min": 0.0, "max": 0.0}}

    ap = AutoscaledProblem("Hypersensitive Autoscaled", initial_guess_ranges_physical)

    t_ap = ap.time(initial=0.0, final=40.0)

    x_ap = ap.state("x", initial=1.5, final=1.0, scale_guide_lower=1.0, scale_guide_upper=1.5)
    u_ap = ap.control("u")

    ap.dynamics({"x": -(x_ap**3) + u_ap})

    integrand_expr_physical = 0.5 * (x_ap**2 + u_ap**2)
    integral_var_physical = ap.add_integral(integrand_expr_physical)

    objective_w0 = 1.0
    print(f"Using objective_scale_weight (w0): {objective_w0}")
    ap.minimize(integral_var_physical, objective_scale_weight=objective_w0)

    print("\n--- Scaling Information (from AutoscaledProblem) ---")
    for var_name, sf_info in sorted(ap.scaling_factors.items()):
        print(
            f"Var: {var_name:<3s} | Rule: {sf_info['rule']:<28s} | v: {sf_info['v']:.3e} | r: {sf_info['r']:.3f}"
        )
    print("--------------------------------------------------\n")

    fixed_polynomial_degrees = [20, 12, 20]
    fixed_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]
    ap.set_mesh(fixed_polynomial_degrees, fixed_mesh_points)

    states_physical_guess_intervals = []
    controls_physical_guess_intervals = []

    for N_poly_degree in fixed_polynomial_degrees:
        tau_points = np.linspace(-1, 1, N_poly_degree + 1)
        x_phys_vals_interval = 1.5 + (1.0 - 1.5) * (tau_points + 1) / 2
        states_physical_guess_intervals.append(x_phys_vals_interval.reshape(1, -1))

        controls_physical_guess_intervals.append(np.zeros((1, N_poly_degree)))

    scaled_states_guess_for_internal_problem = []
    if "x" in ap.scaling_factors:
        vk_x, rk_x = ap.scaling_factors["x"]["v"], ap.scaling_factors["x"]["r"]
        for phys_interval_guess in states_physical_guess_intervals:
            scaled_states_guess_for_internal_problem.append(vk_x * phys_interval_guess + rk_x)
    else:
        print("Warning: Scaling factors for 'x' not found. Using physical guess for states.")
        scaled_states_guess_for_internal_problem = states_physical_guess_intervals

    scaled_controls_guess_for_internal_problem = []
    if "u" in ap.scaling_factors:
        vk_u, rk_u = ap.scaling_factors["u"]["v"], ap.scaling_factors["u"]["r"]
        for phys_interval_guess in controls_physical_guess_intervals:
            scaled_controls_guess_for_internal_problem.append(vk_u * phys_interval_guess + rk_u)
    else:
        print("Warning: Scaling factors for 'u' not found. Using physical guess for controls.")
        scaled_controls_guess_for_internal_problem = controls_physical_guess_intervals

    ap.set_initial_guess(
        states_physical_flat_dict=scaled_states_guess_for_internal_problem,
        controls_physical_flat_dict=scaled_controls_guess_for_internal_problem,
        initial_time=0.0,
        terminal_time=40.0,
        integrals_physical_guess=0.1,
    )

    print("Solving with fixed mesh...")
    fixed_solution = tl.solve_fixed_mesh(
        ap.internal_problem,
        nlp_options={
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 2000,
        },
    )

    if fixed_solution.success:
        physical_objective = fixed_solution.objective / objective_w0
        print("Fixed mesh solution successful!")
        print(f"  Solver's Objective (w0 * J_phys): {fixed_solution.objective:.6f}")
        print(f"  Physical Objective (J_phys):      {physical_objective:.6f}")

        symbolic_vars = ap.get_symbolic_vars_for_postprocessing()

        def unscale_var(var_tilde_vals, var_name, scaling_factors_dict):
            # Ensure var_name is a string and exists in scaling_factors_dict
            if not isinstance(var_name, str) or var_name not in scaling_factors_dict:
                raise KeyError(
                    f"Scaling factors for variable '{var_name}' not found or var_name is not a string."
                )
            sf = scaling_factors_dict[var_name]
            if np.isclose(sf["v"], 0):
                return var_tilde_vals
            return (var_tilde_vals - sf["r"]) / sf["v"]

        # Use string names for tilde symbols from symbolic_vars
        time_x_plot, x_tilde_vals = fixed_solution.get_trajectory(symbolic_vars["x_tilde"])
        x_phys_vals_plot = unscale_var(x_tilde_vals, "x", symbolic_vars["scaling_factors"])

        time_u_plot, u_tilde_vals = fixed_solution.get_trajectory(
            symbolic_vars["u_tilde"]
        )  # CORRECTED: Get time for u
        u_phys_vals_plot = unscale_var(u_tilde_vals, "u", symbolic_vars["scaling_factors"])

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_x_plot, x_phys_vals_plot, label="x (physical)")  # CORRECTED: Use time_x_plot
        plt.title("State x (physical)")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(
            time_u_plot, u_phys_vals_plot, label="u (physical)", linestyle="--"
        )  # CORRECTED: Use time_u_plot
        plt.title("Control u (physical)")
        plt.grid(True)
        plt.legend()

        plt.suptitle("Hypersensitive Problem Solution (Autoscaled Wrapper)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print("\nTrajectoLab's default plot (scaled variables of internal problem):")
        fixed_solution.plot()

    else:
        print(f"Fixed mesh solution failed: {fixed_solution.message}")


# [Existing code for get_scaling_params_generalized and AutoscaledProblem class remains here]
# ...


# --- Main script for CST2 Reactor Problem ---
def main_cst2_reactor():
    print("\n--- Running CST2 Reactor Problem ---")
    # Constants from the problem description [cite: 1]
    p_rho = 0.01
    c1 = 2.83374
    c2 = -0.80865
    c3 = 0.71265
    c4 = 17.2656
    c5 = 27.0756

    # For states x3 and x6, which have initial=0, final=0,
    # we provide a broader scale_guide range as they might vary in between.
    # For states/controls with explicit bounds, those bounds are used as scale_guides.
    # No initial_guess_ranges_physical needed for constructor if all vars get scale_guides
    ap = AutoscaledProblem("CST2_Reactor_Autoscaled")

    # Time
    t_final_cst2 = 9.0
    ap.time(initial=0.0, final=t_final_cst2)

    # States [cite: 1]
    # x1: initial=0, final=10
    x1_ap = ap.state("x1", initial=0.0, final=10.0, scale_guide_lower=0.0, scale_guide_upper=10.0)
    # x2: initial=22, final=14
    x2_ap = ap.state("x2", initial=22.0, final=14.0, scale_guide_lower=14.0, scale_guide_upper=22.0)
    # x3: initial=0, final=0. Assume it varies, objective has x3^2
    x3_ap = ap.state("x3", initial=0.0, final=0.0, scale_guide_lower=-0.5, scale_guide_upper=0.5)
    # x4: initial=-1, final=2.3. Path constraints: -2.5 <= x4 <= 2.5
    x4_ap = ap.state(
        "x4",
        initial=0.0,
        final=2.5,
        lower=-2.5,
        upper=2.5,
        scale_guide_lower=-2.5,
        scale_guide_upper=2.5,
    )
    # x5: initial=0, final=0. Path constraints: -1 <= x5 <= 1
    x5_ap = ap.state(
        "x5",
        initial=-1.0,
        final=0.0,
        lower=-1.0,
        upper=1.0,
        scale_guide_lower=-1.0,
        scale_guide_upper=1.0,
    )
    # x6: initial=0, final=0. Assume it varies, objective has x6^2
    x6_ap = ap.state("x6", initial=0.0, final=0.0, scale_guide_lower=-0.5, scale_guide_upper=0.5)

    # Controls [cite: 1]
    # u1: -c1 <= u1 <= c1
    u1_ap = ap.control("u1", lower=-c1, upper=c1, scale_guide_lower=-c1, scale_guide_upper=c1)
    # u2: c2 <= u2 <= c3
    u2_ap = ap.control("u2", lower=c2, upper=c3, scale_guide_lower=c2, scale_guide_upper=c3)

    # Dynamics [cite: 1]
    # dx1/dt = x4
    # dx2/dt = x5
    # dx3/dt = x6
    # dx4/dt = u1 + c4*x3
    # dx5/dt = u2
    # dx6/dt = -(u1 + c5*x3 + 2*x5*x6) / x2
    ap.dynamics(
        {
            "x1": x4_ap,
            "x2": x5_ap,
            "x3": x6_ap,
            "x4": u1_ap + c4 * x3_ap,
            "x5": u2_ap,
            "x6": -(u1_ap + c5 * x3_ap + 2.0 * x5_ap * x6_ap) / x2_ap,
        }
    )

    # Objective function [cite: 1]
    # J = 0.5 * integral(x3^2 + x6^2 + p*(u1^2 + u2^2)) dt
    integrand_cst2 = 0.5 * (x3_ap**2 + x6_ap**2 + p_rho * (u1_ap**2 + u2_ap**2))
    integral_var_cst2 = ap.add_integral(integrand_cst2)

    objective_w0_cst2 = 1.0  # Keep weight as 1 for direct comparison
    ap.minimize(integral_var_cst2, objective_scale_weight=objective_w0_cst2)

    print("\n--- Scaling Information (CST2 Problem) ---")
    for var_name, sf_info in sorted(ap.scaling_factors.items()):
        print(
            f"Var: {var_name:<3s} | Rule: {sf_info['rule']:<28s} | v: {sf_info['v']:.3e} | r: {sf_info['r']:.3f}"
        )
    print("----------------------------------------\n")

    # Mesh configuration
    # Using a single segment of high degree, or multiple segments
    # Let's try 3 segments of degree 20 for a start (total 60th degree polynomial approx over interval)
    # The time horizon is 9, not 40 like hypersensitive.
    fixed_polynomial_degrees = [25, 25, 25]  # Adjusted degrees
    fixed_mesh_points = [-1.0, -1 / 3, 1 / 3, 1.0]  # Standard 3-segment mesh
    ap.set_mesh(fixed_polynomial_degrees, fixed_mesh_points)

    # Initial Guess Generation
    num_states = 6
    num_controls = 2

    initial_states_physical = {
        "x1": (0.0, 10.0),
        "x2": (22.0, 14.0),
        "x3": (0.0, 0.0),
        "x4": (-1.0, 2.3),
        "x5": (0.0, 0.0),
        "x6": (0.0, 0.0),
    }
    state_names_ordered = ["x1", "x2", "x3", "x4", "x5", "x6"]  # Order for guess array

    initial_controls_physical = {"u1": (-c1, c1), "u2": (c2, c3)}
    control_names_ordered = ["u1", "u2"]

    states_physical_guess_intervals = []
    controls_physical_guess_intervals = []

    for N_poly_degree in fixed_polynomial_degrees:
        # States: N_poly_degree + 1 points per interval
        tau_points_states = np.linspace(-1, 1, N_poly_degree + 1)
        current_interval_states_guess = np.zeros((num_states, N_poly_degree + 1))
        for i, s_name in enumerate(state_names_ordered):
            s_init, s_final = initial_states_physical[s_name]
            # Linear interpolation for the whole trajectory (0 to t_final_cst2)
            # We need to map tau_points (-1 to 1 for interval) to global time fraction later if needed,
            # but for simple linear guess between overall problem initial/final, this is simpler.
            # For now, just simple linear interpolation across each state's own initial/final.
            current_interval_states_guess[i, :] = (
                s_init + (s_final - s_init) * (tau_points_states + 1) / 2
            )
        states_physical_guess_intervals.append(current_interval_states_guess)

        # Controls: N_poly_degree points per interval
        current_interval_controls_guess = np.zeros((num_controls, N_poly_degree))
        for i, c_name in enumerate(control_names_ordered):
            c_min, c_max = initial_controls_physical[c_name]
            current_interval_controls_guess[i, :] = (c_min + c_max) / 2  # Midpoint
        controls_physical_guess_intervals.append(current_interval_controls_guess)

    # Scale the initial guesses for the internal problem
    scaled_states_guess_list = []
    for (
        phys_interval_guess_states
    ) in states_physical_guess_intervals:  # list of (num_states, N+1) arrays
        scaled_interval_states = np.zeros_like(phys_interval_guess_states)
        for i, s_name in enumerate(state_names_ordered):
            if s_name in ap.scaling_factors:
                vk, rk = ap.scaling_factors[s_name]["v"], ap.scaling_factors[s_name]["r"]
                scaled_interval_states[i, :] = vk * phys_interval_guess_states[i, :] + rk
            else:  # Should not happen if all states defined
                scaled_interval_states[i, :] = phys_interval_guess_states[i, :]
        scaled_states_guess_list.append(scaled_interval_states)

    scaled_controls_guess_list = []
    for (
        phys_interval_guess_controls
    ) in controls_physical_guess_intervals:  # list of (num_controls, N) arrays
        scaled_interval_controls = np.zeros_like(phys_interval_guess_controls)
        for i, c_name in enumerate(control_names_ordered):
            if c_name in ap.scaling_factors:
                vk, rk = ap.scaling_factors[c_name]["v"], ap.scaling_factors[c_name]["r"]
                scaled_interval_controls[i, :] = vk * phys_interval_guess_controls[i, :] + rk
            else:  # Should not happen
                scaled_interval_controls[i, :] = phys_interval_guess_controls[i, :]
        scaled_controls_guess_list.append(scaled_interval_controls)

    ap.set_initial_guess(
        states_physical_flat_dict=scaled_states_guess_list,  # Already scaled
        controls_physical_flat_dict=scaled_controls_guess_list,  # Already scaled
        initial_time=0.0,
        terminal_time=t_final_cst2,
        integrals_physical_guess=0.1,  # Guess for the integral value
    )

    print("Solving CST2 Reactor with fixed mesh...")
    # May need more iterations for this problem
    nlp_max_iter = 2000
    print(f"NLP max iterations: {nlp_max_iter}")
    fixed_solution = tl.solve_fixed_mesh(
        ap.internal_problem,
        nlp_options={
            "ipopt.print_level": 5,  # Increased print level for diagnostics
            "ipopt.sb": "yes",
            "print_time": 1,
            "ipopt.max_iter": nlp_max_iter,
            "ipopt.tol": 1e-4,  # Default is 1e-8, can try tighter if needed
            "ipopt.constr_viol_tol": 1e-4,  # Default is 1e-4
        },
    )

    if fixed_solution.success:
        physical_objective_cst2 = fixed_solution.objective / objective_w0_cst2
        print("CST2 Fixed mesh solution successful!")
        print(f"  Solver's Objective (w0 * J_phys): {fixed_solution.objective:.8f}")
        print(f"  Physical Objective (J_phys):      {physical_objective_cst2:.8f}")

        reference_objective_cst2 = 0.0375194596  # [cite: 1]
        print(f"  Reference Objective (J*):         {reference_objective_cst2:.8f}")
        error_percentage = (
            abs(physical_objective_cst2 - reference_objective_cst2) / reference_objective_cst2 * 100
        )
        print(f"  Error from reference:             {error_percentage:.4f}%")

        symbolic_vars = ap.get_symbolic_vars_for_postprocessing()

        def unscale_var(var_tilde_vals, var_name, scaling_factors_dict):
            if not isinstance(var_name, str) or var_name not in scaling_factors_dict:
                raise KeyError(
                    f"Scaling factors for variable '{var_name}' not found or var_name is not a string."
                )
            sf = scaling_factors_dict[var_name]
            if np.isclose(sf["v"], 0):
                return var_tilde_vals
            return (var_tilde_vals - sf["r"]) / sf["v"]

        time_plot, x1_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["x1_tilde"])
        x1_phys_sol = unscale_var(x1_tilde_sol, "x1", symbolic_vars["scaling_factors"])

        _, x2_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["x2_tilde"])
        x2_phys_sol = unscale_var(x2_tilde_sol, "x2", symbolic_vars["scaling_factors"])

        _, x3_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["x3_tilde"])
        x3_phys_sol = unscale_var(x3_tilde_sol, "x3", symbolic_vars["scaling_factors"])

        _, x4_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["x4_tilde"])
        x4_phys_sol = unscale_var(x4_tilde_sol, "x4", symbolic_vars["scaling_factors"])

        _, x5_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["x5_tilde"])
        x5_phys_sol = unscale_var(x5_tilde_sol, "x5", symbolic_vars["scaling_factors"])

        _, x6_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["x6_tilde"])
        x6_phys_sol = unscale_var(x6_tilde_sol, "x6", symbolic_vars["scaling_factors"])

        time_plot_u, u1_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["u1_tilde"])
        u1_phys_sol = unscale_var(u1_tilde_sol, "u1", symbolic_vars["scaling_factors"])

        _, u2_tilde_sol = fixed_solution.get_trajectory(symbolic_vars["u2_tilde"])
        u2_phys_sol = unscale_var(u2_tilde_sol, "u2", symbolic_vars["scaling_factors"])

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 2, 1)
        plt.plot(time_plot, x1_phys_sol, label="x1 (physical)")
        plt.title("x1")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(time_plot, x2_phys_sol, label="x2 (physical)")
        plt.title("x2")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(time_plot, x3_phys_sol, label="x3 (physical)")
        plt.title("x3")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(time_plot, x4_phys_sol, label="x4 (physical)")
        plt.hlines(
            [-2.5, 2.5],
            xmin=time_plot[0],
            xmax=time_plot[-1],
            colors="r",
            linestyles="--",
            label="bounds",
        )
        plt.title("x4")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(time_plot, x5_phys_sol, label="x5 (physical)")
        plt.hlines(
            [-1.0, 1.0],
            xmin=time_plot[0],
            xmax=time_plot[-1],
            colors="r",
            linestyles="--",
            label="bounds",
        )
        plt.title("x5")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(time_plot, x6_phys_sol, label="x6 (physical)")
        plt.title("x6")
        plt.grid(True)
        plt.legend()

        plt.suptitle("CST2 Reactor States (Autoscaled)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time_plot_u, u1_phys_sol, label="u1 (physical)")
        plt.hlines(
            [-c1, c1],
            xmin=time_plot_u[0],
            xmax=time_plot_u[-1],
            colors="r",
            linestyles="--",
            label="bounds",
        )
        plt.title("u1")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(time_plot_u, u2_phys_sol, label="u2 (physical)")
        plt.hlines(
            [c2, c3],
            xmin=time_plot_u[0],
            xmax=time_plot_u[-1],
            colors="r",
            linestyles="--",
            label="bounds",
        )
        plt.title("u2")
        plt.grid(True)
        plt.legend()

        plt.suptitle("CST2 Reactor Controls (Autoscaled)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    else:
        print(f"CST2 Fixed mesh solution failed: {fixed_solution.message}")


# To run this new problem, modify the main execution block:
if __name__ == "__main__":
    # main_hypersensitive() # Comment out or remove previous problem run
    main_cst2_reactor()
