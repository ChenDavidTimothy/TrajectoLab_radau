import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "MAPTOR"
copyright = "2025, MAPTOR"
author = "David Timothy"

# Get version
try:
    import maptor

    version = maptor.__version__
    release = version
except ImportError:
    version = "0.1.0"
    release = version

# Extensions - Added MyST parser for markdown support
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",  # Added for markdown support
]

# MyST parser configuration for markdown and LaTeX support
myst_enable_extensions = [
    "dollarmath",  # For $...$ and $$...$$ LaTeX math
    "amsmath",  # For advanced math environments
    "deflist",  # For definition lists
    "fieldlist",  # For field lists
    "html_admonition",  # For admonitions
    "html_image",  # For HTML image handling
    "colon_fence",  # For ::: fenced directives
    "smartquotes",  # For smart quotes
    "strikethrough",  # For ~~strikethrough~~
    "substitution",  # For variable substitution
    "tasklist",  # For - [ ] task lists
]

# Configure math rendering
myst_dmath_double_inline = True  # Allows $$ for display math
myst_dmath_allow_labels = True  # Allows equation labels

# Set supported file extensions - use list format
source_suffix = [".rst", ".md"]

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

# Brand integration - Professional logo placement
html_logo = "_static/MAPTOR_banner.svg"
html_title = "MAPTOR Documentation"

# Theme options for better branding
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2d2d2d",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

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
            print("API documentation generated")
        except subprocess.CalledProcessError:
            print("API generation had issues")

    # Generate examples documentation
    examples_script_path = Path(__file__).parent / "generate_examples_docs.py"
    if examples_script_path.exists():
        try:
            subprocess.run([sys.executable, str(examples_script_path)], check=True)
            print("Examples documentation generated")
        except subprocess.CalledProcessError:
            print("Examples generation had issues")
