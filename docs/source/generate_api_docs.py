#!/usr/bin/env python3
"""
Auto-generate API documentation for TrajectoLab.

This script discovers the main TrajectoLab package and its submodules,
then generates reStructuredText (.rst) files for each.
It also creates an index.rst file that organizes these modules
into categories for Sphinx documentation.
"""

import pkgutil
import sys
from pathlib import Path


def generate_api_docs():
    """
    Generates API documentation .rst files for Sphinx.
    """
    # Determine the project root. This assumes the script is located
    # in a subdirectory of the project root, e.g., 'docs/source/'.
    # Adjust if your script is located elsewhere (e.g., project_root/docs).
    # If generate_api_docs.py is in 'docs/source/', then project_root is two levels up.
    # If it's in 'docs/', then project_root is one level up.
    # The original script had .parent.parent.parent which implies docs/source/something/
    # Let's assume it's in docs/source/
    try:
        # Assuming this script is in 'docs/source/' relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., in some interactive environments)
        project_root = Path(".").resolve().parent.parent  # Adjust as needed

    print(f"ℹ️  Project root identified as: {project_root}")
    sys.path.insert(0, str(project_root))

    print("🔍 Discovering TrajectoLab modules...")

    try:
        import trajectolab

        print(f"✓ Successfully imported TrajectoLab from: {trajectolab.__file__}")
    except ImportError as e:
        print("❌ Error: TrajectoLab not found or could not be imported.")
        print(f"   Details: {e}")
        print(
            "   Ensure TrajectoLab is installed (e.g., 'pip install -e .') and sys.path is correct:"
        )
        print(f"   Current sys.path includes: {project_root}")
        return

    # Define the output directory for the generated .rst files
    # This should be relative to this script's location.
    # If this script is docs/source/generate_api_docs.py, then api_dir is docs/source/api/
    api_dir = Path(__file__).parent / "api"
    api_dir.mkdir(exist_ok=True)
    print(f"ℹ️  API documentation output directory: {api_dir.resolve()}")

    # --- 1. Generate RST for the main trajectolab package ---
    main_package_name = trajectolab.__name__  # Should be "trajectolab"
    # The .rst filename for the main package (e.g., "trajectolab.rst")
    # This must match the entry in the toctree in index.rst
    main_package_rst_filename = main_package_name
    main_rst_file_path = api_dir / f"{main_package_rst_filename}.rst"

    main_package_rst_content = f"""
{main_package_name}
{"=" * len(main_package_name)}

.. automodule:: {main_package_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __dict__,__weakref__
"""
    main_rst_file_path.write_text(main_package_rst_content.strip() + "\n")
    print(f"✓ Generated RST for main package: {main_rst_file_path.name}")

    # --- 2. Discover and generate RST for submodules ---
    submodules_found = []
    # trajectolab.__path__ gives the directory (or directories) of the package
    # trajectolab.__name__ + "." ensures we get fully qualified submodule names
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=trajectolab.__path__, prefix=trajectolab.__name__ + "."
    ):
        # Filter out test modules or other private/internal modules
        if not any(
            exclude_pattern in modname for exclude_pattern in ["test", "__pycache__", ".pytest"]
        ):
            submodules_found.append(modname)

    print(f"✓ Found {len(submodules_found)} submodules to document.")

    # Generate .rst file for each discovered submodule
    for submodule_fullname in submodules_found:  # e.g., "trajectolab.solver"
        # Create a "safe" version of the module name for the .rst filename
        # e.g., "trajectolab.solver" becomes "trajectolab_solver.rst"
        submodule_filename_safe = submodule_fullname.replace(".", "_")
        submodule_rst_file_path = api_dir / f"{submodule_filename_safe}.rst"

        submodule_rst_content = f"""
{submodule_fullname}
{"=" * len(submodule_fullname)}

.. automodule:: {submodule_fullname}
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __dict__,__weakref__
"""
        submodule_rst_file_path.write_text(submodule_rst_content.strip() + "\n")
        # print(f"  ✓ Generated RST for submodule: {submodule_rst_file_path.name}")

    # --- 3. Generate the API index.rst file ---
    api_index_rst_path = api_dir / "index.rst"

    # Start with the header and the main package toctree entry
    index_content = f"""
API Reference
=============

This section provides an overview of all modules within the TrajectoLab library.

Core Package
------------

The main ``{main_package_name}`` package.

.. toctree::
   :maxdepth: 1

   {main_package_rst_filename}

"""

    # Categorize submodules for better organization in the index
    # The names used here (e.g., "trajectolab_solver") must match the .rst filenames
    categorized_submodules = {
        "Solver Modules": [],
        "Problem Definition Modules": [],
        "Adaptive Mesh Refinement Modules": [],
        "Utility Modules": [],
        "Other Modules": [],  # Fallback category
    }

    for submodule_fullname in sorted(submodules_found):
        submodule_filename_safe = submodule_fullname.replace(".", "_")
        if "solver" in submodule_fullname:
            categorized_submodules["Solver Modules"].append(submodule_filename_safe)
        elif "problem" in submodule_fullname:
            categorized_submodules["Problem Definition Modules"].append(submodule_filename_safe)
        elif "adaptive" in submodule_fullname:  # Example category
            categorized_submodules["Adaptive Mesh Refinement Modules"].append(
                submodule_filename_safe
            )
        elif "util" in submodule_fullname:  # Example category
            categorized_submodules["Utility Modules"].append(submodule_filename_safe)
        else:
            categorized_submodules["Other Modules"].append(submodule_filename_safe)

    # Append categorized submodules to the index content
    for category_title, module_filenames in categorized_submodules.items():
        if module_filenames:  # Only add category if it has modules
            index_content += f"\n{category_title}\n{'-' * len(category_title)}\n\n"
            index_content += ".. toctree::\n   :maxdepth: 1\n\n"
            for filename in sorted(module_filenames):
                index_content += f"   {filename}\n"

    api_index_rst_path.write_text(index_content)
    print(f"✓ Generated API index: {api_index_rst_path.name}")
    print(f"🎉 API documentation generation complete in {api_dir.resolve()}")


if __name__ == "__main__":
    generate_api_docs()
