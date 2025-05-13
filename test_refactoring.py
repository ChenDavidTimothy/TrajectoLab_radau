"""
Test script to verify that the refactored code produces identical results to the original.
"""

import os
import sys
import numpy as np
import traceback
import importlib.util
import pickle

def import_module_from_path(module_name, file_path):
    """Dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_original_code():
    """Run the original code and return the results."""
    print("Running original code...")
    
    # Add the original directory to path so we can import the original modules
    original_dir = '.'  # Adjust as needed - points to where the original files are
    sys.path.insert(0, original_dir)
    
    # Import the original modules directly from their files
    rpm_solver = import_module_from_path('rpm_solver', os.path.join(original_dir, 'rpm_solver.py'))
    phs_adaptive = import_module_from_path('phs_adaptive', os.path.join(original_dir, 'phs_adaptive.py'))
    hypersensitive_direct = import_module_from_path('hypersensitive_direct', os.path.join(original_dir, 'hypersensitive_direct.py'))
    hypersensitive = import_module_from_path('hypersensitive', os.path.join(original_dir, 'hypersensitive.py'))
    
    # Run the examples
    # 1. Direct solve
    direct_solution = rpm_solver.solve_single_phase_radau_collocation(
        hypersensitive_direct.fixed_mesh_problem_def_hypersensitive
    )
    
    # 2. Adaptive solve
    adaptive_solution = phs_adaptive.run_phs_adaptive_mesh_refinement(
        hypersensitive.initial_problem_def_hypersensitive,
        hypersensitive.adaptive_params_hypersensitive
    )
    
    # Remove references to CasADi objects that can't be easily compared
    for soln in [direct_solution, adaptive_solution]:
        if soln:
            soln.pop('raw_solution', None)
            soln.pop('opti_object', None)
    
    # Reset path
    sys.path.pop(0)
    
    return direct_solution, adaptive_solution

def run_refactored_code():
    """Run the refactored code and return the results."""
    print("Running refactored code...")
    
    # Import from the new package structure
    import trajectolab as tl
    from trajectolab.examples.hypersensitive.direct import fixed_mesh_problem_def_hypersensitive
    from trajectolab.examples.hypersensitive.adaptive import (
        initial_problem_def_hypersensitive, 
        adaptive_params_hypersensitive
    )
    
    # Run the examples
    # 1. Direct solve
    direct_solution = tl.solve_single_phase_radau_collocation(
        fixed_mesh_problem_def_hypersensitive
    )
    
    # 2. Adaptive solve
    adaptive_solution = tl.run_phs_adaptive_mesh_refinement(
        initial_problem_def_hypersensitive,
        adaptive_params_hypersensitive
    )
    
    # Remove references to CasADi objects that can't be easily compared
    for soln in [direct_solution, adaptive_solution]:
        if soln:
            soln.pop('raw_solution', None)
            soln.pop('opti_object', None)
    
    return direct_solution, adaptive_solution

def compare_results(original_results, refactored_results, tolerance=1e-10):
    """Compare the results and print differences."""
    direct_original, adaptive_original = original_results
    direct_refactored, adaptive_refactored = refactored_results
    
    # Compare direct solutions
    print("\nComparing direct solutions:")
    compare_dictionaries(direct_original, direct_refactored, "direct", tolerance)
    
    # Compare adaptive solutions
    print("\nComparing adaptive solutions:")
    compare_dictionaries(adaptive_original, adaptive_refactored, "adaptive", tolerance)

def compare_dictionaries(dict1, dict2, name, tolerance=1e-10):
    """Compare two dictionaries recursively."""
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in sorted(all_keys):
        if key not in dict1:
            print(f"  {name}: Key '{key}' missing in original")
            continue
        if key not in dict2:
            print(f"  {name}: Key '{key}' missing in refactored")
            continue
        
        val1 = dict1[key]
        val2 = dict2[key]
        
        if isinstance(val1, dict) and isinstance(val2, dict):
            print(f"  Comparing {name}.{key} dictionaries:")
            compare_dictionaries(val1, val2, f"{name}.{key}", tolerance)
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                print(f"  {name}.{key}: Shape mismatch: {val1.shape} vs {val2.shape}")
            elif not np.allclose(val1, val2, rtol=tolerance, atol=tolerance):
                diff = np.abs(val1 - val2)
                max_diff = np.max(diff)
                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  {name}.{key}: Arrays not equal, max diff: {max_diff} at {max_idx}")
                print(f"    Original: {val1[max_idx]}, Refactored: {val2[max_idx]}")
            else:
                print(f"  {name}.{key}: Arrays match")
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                print(f"  {name}.{key}: List length mismatch: {len(val1)} vs {len(val2)}")
            else:
                print(f"  {name}.{key}: Comparing lists of length {len(val1)}")
                for i, (item1, item2) in enumerate(zip(val1, val2)):
                    if isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
                        if not np.allclose(item1, item2, rtol=tolerance, atol=tolerance):
                            diff = np.abs(item1 - item2)
                            max_diff = np.max(diff) if diff.size > 0 else 0
                            max_idx = np.unravel_index(np.argmax(diff), diff.shape) if diff.size > 0 else (0,0)
                            print(f"  {name}.{key}[{i}]: Arrays not equal, max diff: {max_diff} at {max_idx}")
                        else:
                            print(f"  {name}.{key}[{i}]: Arrays match")
                    else:
                        if item1 != item2:
                            print(f"  {name}.{key}[{i}]: Values differ: {item1} vs {item2}")
                        else:
                            print(f"  {name}.{key}[{i}]: Values match")
        else:
            if val1 != val2:
                print(f"  {name}.{key}: Values differ: {val1} vs {val2}")
            else:
                print(f"  {name}.{key}: Values match")

def main():
    """Main test function."""
    try:
        # Install the package in development mode first
        print("Installing trajectolab package in development mode...")
        os.system("pip install -e .")
        
        # Run and save original code results
        print("\nRunning original code...")
        original_results = run_original_code()
        
        # Run and save refactored code results
        print("\nRunning refactored code...")
        refactored_results = run_refactored_code()
        
        # Compare results
        print("\nComparing results...")
        compare_results(original_results, refactored_results)
        
        print("\nVerification complete!")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()