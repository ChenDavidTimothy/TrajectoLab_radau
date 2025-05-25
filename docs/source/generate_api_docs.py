#!/usr/bin/env python3
"""Auto-generate API documentation for TrajectoLab - CORRECTED VERSION."""

import pkgutil
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_api_docs():
    """Generate API documentation automatically."""

    print("🔍 Discovering TrajectoLab modules...")

    try:
        import trajectolab
    except ImportError:
        print("⚠ TrajectoLab not installed - install with 'pip install -e .'")
        return

    # Create API directory
    api_dir = Path(__file__).parent / "api"
    api_dir.mkdir(exist_ok=True)

    # Discover all modules
    modules = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        trajectolab.__path__, trajectolab.__name__ + "."
    ):
        # FIXED: Better filtering of test/internal modules
        if not any(exclude in modname for exclude in ["test", "__pycache__", ".pytest"]):
            modules.append(modname)

    print(f"✓ Found {len(modules)} modules")

    # Generate RST file for each module
    for module in modules:
        safe_name = module.replace(".", "_")
        rst_file = api_dir / f"{safe_name}.rst"

        # FIXED: Better module documentation format
        content = f"""
{module}
{"=" * len(module)}

.. automodule:: {module}
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: __dict__,__weakref__
"""
        rst_file.write_text(content.strip())

    # Generate API index with better organization
    api_index = api_dir / "index.rst"
    content = """
API Reference
=============

Core Modules
------------

.. toctree::
   :maxdepth: 1

   trajectolab

High-Level Interface
--------------------

.. toctree::
   :maxdepth: 1

"""

    # Organize modules by category
    core_modules = []
    solver_modules = []
    problem_modules = []
    other_modules = []

    for module in sorted(modules):
        if module == "trajectolab":
            continue  # Already listed above
        elif "solver" in module:
            solver_modules.append(module)
        elif "problem" in module:
            problem_modules.append(module)
        else:
            other_modules.append(module)

    # Add solver modules
    if solver_modules:
        content += "\nSolver Modules\n--------------\n\n.. toctree::\n   :maxdepth: 1\n\n"
        for module in solver_modules:
            safe_name = module.replace(".", "_")
            content += f"   {safe_name}\n"

    # Add problem modules
    if problem_modules:
        content += "\nProblem Definition\n------------------\n\n.. toctree::\n   :maxdepth: 1\n\n"
        for module in problem_modules:
            safe_name = module.replace(".", "_")
            content += f"   {safe_name}\n"

    # Add other modules
    if other_modules:
        content += "\nOther Modules\n-------------\n\n.. toctree::\n   :maxdepth: 1\n\n"
        for module in other_modules:
            safe_name = module.replace(".", "_")
            content += f"   {safe_name}\n"

    api_index.write_text(content)
    print(f"✓ Generated organized API documentation in {api_dir}")


if __name__ == "__main__":
    generate_api_docs()
