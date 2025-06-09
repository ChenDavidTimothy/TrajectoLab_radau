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

# Extensions - Professional scientific Python setup
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST parser configuration for markdown and LaTeX support
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Configure math rendering
myst_dmath_double_inline = True
myst_dmath_allow_labels = True

# Set supported file extensions
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

# HTML theme configuration - Professional PyData theme like SciPy
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["maptor_brand.css"]

# MAPTOR branding
html_logo = "_static/MAPTOR_banner.svg"
html_title = "MAPTOR Documentation"
html_favicon = "_static/favicon.ico"  # Add this file to _static/

# Modern sidebar configuration
html_sidebars = {"index": ["search-button-field"], "**": ["search-button-field", "sidebar-nav-bs"]}

# Professional theme options - adapted from SciPy
html_theme_options = {
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ChenDavidTimothy/maptor",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/maptor/",
            "icon": "fa-solid fa-box",
        },
    ],
    "logo": {
        "text": "MAPTOR",
        "image_light": "_static/MAPTOR_banner.svg",
        "image_dark": "_static/MAPTOR_banner.svg",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    # MAPTOR brand colors
    "navbar_style": "dark",
    "primary_sidebar_end": [],
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "casadi": ("https://web.casadi.org/", None),
}

# Warning handling
suppress_warnings = ["autodoc.import_object"]

# Napoleon settings for better docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.{3,}: | {5,8}: "
copybutton_prompt_is_regexp = True


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
