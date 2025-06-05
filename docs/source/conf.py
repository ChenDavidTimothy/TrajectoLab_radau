# TrajectoLab Sphinx Configuration - CORRECTED VERSION
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "TrajectoLab"
copyright = "2024, TrajectoLab Authors"
author = "TrajectoLab Authors"

# Get version
try:
    import trajectolab

    version = trajectolab.__version__
    release = version
except ImportError:
    version = "0.2.1"
    release = version

# Extensions - Removed problematic sphinx_gallery
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Auto-generate everything
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# Autodoc configuration
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# HTML theme configuration
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # Load custom CSS

# Intersphinx mapping - Added proper mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Warning handling
suppress_warnings = ["autodoc.import_object"]

# Napoleon settings for better docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False


# Auto-run API and examples generation
def setup(app):
    """Auto-generate API and examples docs on build."""
    import subprocess

    # Generate API documentation
    api_script_path = Path(__file__).parent / "generate_api_docs.py"
    if api_script_path.exists():
        try:
            subprocess.run([sys.executable, str(api_script_path)], check=True)
            print("✓ API documentation generated")
        except subprocess.CalledProcessError:
            print("⚠ API generation had issues")

    # Generate examples documentation
    examples_script_path = Path(__file__).parent / "generate_examples_docs.py"
    if examples_script_path.exists():
        try:
            subprocess.run([sys.executable, str(examples_script_path)], check=True)
            print("✓ Examples documentation generated")
        except subprocess.CalledProcessError:
            print("⚠ Examples generation had issues")
