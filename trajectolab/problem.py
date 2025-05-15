import numpy as np
import casadi as ca
from typing import Dict, List, Callable, Union, Any, Optional, Tuple

class Constraint:
    def __init__(self, val=None, lower=None, upper=None, equals=None):
        self.val = val
        self.lower = lower
        self.upper = upper
        self.equals = equals
        
        if equals is not None:
            self.lower = equals
            self.upper = equals

class Problem:
    def __init__(self, name="Unnamed Problem"):
        self.name = name
        self._states = {}
        self._controls = {}
        self._parameters = {}
        self._t0_bounds = [0.0, 0.0]
        self._tf_bounds = [1.0, 1.0]
        self._dynamics_func = None
        self._objective_type = None
        self._objective_func = None
        self._path_constraints = []
        self._event_constraints = []
        self._num_integrals = 0
        self._integral_functions = []
    
    def set_time_bounds(self, t0=0.0, tf=1.0, t0_bounds=None, tf_bounds=None):
        if t0_bounds is None:
            t0_bounds = [t0, t0]
        if tf_bounds is None:
            tf_bounds = [tf, tf]
            
        self._t0_bounds = t0_bounds
        self._tf_bounds = tf_bounds
        return self
    
    def add_state(self, name, initial_constraint=None, final_constraint=None, 
                  bounds=None, lower=None, upper=None):
        if bounds is not None:
            lower, upper = bounds
            
        self._states[name] = {
            'index': len(self._states),
            'initial_constraint': initial_constraint,
            'final_constraint': final_constraint,
            'lower': lower,
            'upper': upper
        }
        return self
    
    def add_control(self, name, bounds=None, lower=None, upper=None):
        if bounds is not None:
            lower, upper = bounds
            
        self._controls[name] = {
            'index': len(self._controls),
            'lower': lower,
            'upper': upper
        }
        return self
    
    def add_parameter(self, name, value):
        self._parameters[name] = value
        return self
    
    def set_dynamics(self, dynamics_func):
        self._dynamics_func = dynamics_func
        return self
    
    def set_objective(self, objective_type, objective_func):
        """Set the objective function for the problem."""
        # Auto-correct the objective type if it's referencing integrals
        if objective_type == "mayer" and self._num_integrals > 0:
            objective_type = "integral"

        self._objective_type = objective_type
        self._objective_func = objective_func
        return self
    
    def add_integral(self, integral_func):
        self._num_integrals += 1
        self._integral_functions.append(integral_func)
        return self
    
    def add_path_constraint(self, constraint_func):
        self._path_constraints.append(constraint_func)
        return self
        
    def _convert_to_legacy_problem(self):
        from trajectolab.direct_solver import OptimalControlProblem, PathConstraint, EventConstraint

        # Create adapter functions
        vectorized_dynamics = self._create_vectorized_dynamics()
        vectorized_objective = self._create_vectorized_objective()
        vectorized_integrand = self._create_vectorized_integrand()
        vectorized_path_constraints = self._create_vectorized_path_constraints()

        # NEW: Add auto-generated event constraints from state definitions
        # This creates a function that adds all the initial and final state constraints
        def auto_event_constraints(t0, tf, x0, xf, q, params):
            from trajectolab.direct_solver import EventConstraint
            result = []

            # Add initial state constraints
            for i, name in enumerate(self._states.keys()):
                state_def = self._states[name]
                if state_def.get('initial_constraint'):
                    constraint = state_def['initial_constraint']
                    result.append(EventConstraint(
                        val=x0[i],
                        min_val=constraint.lower,
                        max_val=constraint.upper,
                        equals=constraint.equals
                    ))

                # Add final state constraints
                if state_def.get('final_constraint'):
                    constraint = state_def['final_constraint']
                    result.append(EventConstraint(
                        val=xf[i],
                        min_val=constraint.lower,
                        max_val=constraint.upper,
                        equals=constraint.equals
                    ))

            # Add custom event constraints
            for constraint_func in self._event_constraints:
                constraint = constraint_func(t0, tf, {name: x0[i] for i, name in enumerate(self._states.keys())}, 
                                            {name: xf[i] for i, name in enumerate(self._states.keys())}, q, params)
                if isinstance(constraint, Constraint):
                    result.append(EventConstraint(
                        val=constraint.val,
                        min_val=constraint.lower,
                        max_val=constraint.upper,
                        equals=constraint.equals
                    ))
                else:
                    result.extend([
                        EventConstraint(
                            val=c.val,
                            min_val=c.lower,
                            max_val=c.upper,
                            equals=c.equals
                        ) for c in constraint
                    ])

            return result
    
        # Create legacy problem with the auto event constraints
        return OptimalControlProblem(
            num_states=len(self._states),
            num_controls=len(self._controls),
            dynamics_function=vectorized_dynamics,
            objective_function=vectorized_objective,
            t0_bounds=self._t0_bounds,
            tf_bounds=self._tf_bounds,
            num_integrals=self._num_integrals,
            integral_integrand_function=vectorized_integrand if self._num_integrals > 0 else None,
            path_constraints_function=vectorized_path_constraints if self._path_constraints else None,
            event_constraints_function=auto_event_constraints,  # Always include auto constraints
            problem_parameters=self._parameters
        )
    
    def _create_vectorized_dynamics(self):
        dynamics_func = self._dynamics_func
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())
        
        def vectorized_dynamics(states_vec, controls_vec, time, params):
            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}
            
            result_dict = dynamics_func(states_dict, controls_dict, time, params)
            result_vec = [result_dict[name] for name in state_names]
            
            return result_vec
            
        return vectorized_dynamics
    
    def _create_vectorized_objective(self):
        objective_func = self._objective_func
        objective_type = self._objective_type
        state_names = list(self._states.keys())
        
        def vectorized_objective(t0, tf, x0, xf, q, params):
            if objective_type == "mayer":
                initial_states = {name: x0[i] for i, name in enumerate(state_names)}
                final_states = {name: xf[i] for i, name in enumerate(state_names)}
                return objective_func(t0, tf, initial_states, final_states, q, params)
            else:
                # For integral objectives, just return the integral value
                return q[0] if isinstance(q, (list, np.ndarray)) and len(q) > 0 else q
                
        return vectorized_objective
    
    def _create_vectorized_integrand(self):
        integral_functions = self._integral_functions
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())

        def vectorized_integrand(states_vec, controls_vec, time, integral_idx, params):
            if integral_idx >= len(integral_functions):
                return 0.0  # Return a numeric value when no function exists

            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}

            # Just return the result directly - DO NOT try to convert to float!
            return integral_functions[integral_idx](states_dict, controls_dict, time, params)

        return vectorized_integrand
    
    def _create_vectorized_path_constraints(self):
        path_constraints = self._path_constraints
        state_names = list(self._states.keys())
        control_names = list(self._controls.keys())
        
        def vectorized_path_constraints(states_vec, controls_vec, time, params):
            from trajectolab.direct_solver import PathConstraint
            
            states_dict = {name: states_vec[i] for i, name in enumerate(state_names)}
            controls_dict = {name: controls_vec[i] for i, name in enumerate(control_names)}
            
            result = []
            for constraint_func in path_constraints:
                constraint = constraint_func(states_dict, controls_dict, time, params)
                if isinstance(constraint, Constraint):
                    result.append(PathConstraint(
                        val=constraint.val,
                        min_val=constraint.lower,
                        max_val=constraint.upper,
                        equals=constraint.equals
                    ))
                else:
                    result.extend([
                        PathConstraint(
                            val=c.val,
                            min_val=c.lower,
                            max_val=c.upper,
                            equals=c.equals
                        ) for c in constraint
                    ])
            
            return result
            
        return vectorized_path_constraints
    
def _create_vectorized_event_constraints(self):
    """Create a function that handles all event constraints including boundary conditions."""
    event_constraints = self._event_constraints
    state_names = list(self._states.keys())
    
    def vectorized_event_constraints(t0, tf, x0, xf, q, params):
        from trajectolab.direct_solver import EventConstraint
        
        result = []
        
        # Add initial and final state constraints
        for i, name in enumerate(state_names):
            state_info = self._states[name]
            
            # Handle initial constraints
            if state_info.get('initial_constraint'):
                constraint = state_info['initial_constraint']
                if constraint.equals is not None:
                    result.append(EventConstraint(
                        val=x0[i],
                        equals=constraint.equals
                    ))
                elif constraint.lower is not None or constraint.upper is not None:
                    result.append(EventConstraint(
                        val=x0[i],
                        min_val=constraint.lower,
                        max_val=constraint.upper
                    ))
            
            # Handle final constraints
            if state_info.get('final_constraint'):
                constraint = state_info['final_constraint']
                if constraint.equals is not None:
                    result.append(EventConstraint(
                        val=xf[i],
                        equals=constraint.equals
                    ))
                elif constraint.lower is not None or constraint.upper is not None:
                    result.append(EventConstraint(
                        val=xf[i],
                        min_val=constraint.lower,
                        max_val=constraint.upper
                    ))
        
        # Add user-defined event constraints
        for constraint_func in event_constraints:
            constraint = constraint_func(t0, tf, initial_states, final_states, q, params)
            # (rest of processing for user constraints...)
        
        return result
    
    return vectorized_event_constraints