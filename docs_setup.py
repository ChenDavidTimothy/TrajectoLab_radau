#!/usr/bin/env python3
"""
CORRECTED working documentation setup script for TrajectoLab.
Fixes all compatibility issues with the actual codebase.
"""

from pathlib import Path


def create_docs_structure():
    """Create the complete documentation system - FIXED VERSION."""

    print("🚀 Setting up TrajectoLab Documentation (CORRECTED)...")

    # Get repository root
    repo_root = Path.cwd()
    print(f"📁 Working in: {repo_root}")

    # Create directory structure
    dirs_to_create = [
        "docs",
        "docs/source",
        "docs/source/_static",
        "docs/source/_templates",
        "docs/source/examples",
        ".github",
        ".github/workflows",
    ]

    for dir_path in dirs_to_create:
        full_path = repo_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {dir_path}/")

    # File contents dictionary
    files = {
        # FIXED Sphinx Configuration
        "docs/source/conf.py": '''# TrajectoLab Sphinx Configuration - CORRECTED VERSION
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = 'TrajectoLab'
copyright = '2024, TrajectoLab Authors'
author = 'TrajectoLab Authors'

# Get version
try:
    import trajectolab
    version = trajectolab.__version__
    release = version
except ImportError:
    version = '0.2.1'
    release = version

# Extensions - Removed problematic sphinx_gallery
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

# Auto-generate everything
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Autodoc configuration
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

# HTML theme configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']  # Load custom CSS

# Intersphinx mapping - Added proper mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Warning handling
suppress_warnings = ['autodoc.import_object']

# Napoleon settings for better docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Auto-run API generation
def setup(app):
    """Auto-generate API docs on build."""
    import subprocess
    script_path = Path(__file__).parent / 'generate_api_docs.py'
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
            print("✓ API documentation generated")
        except subprocess.CalledProcessError:
            print("⚠ API generation had issues")
''',
        # FIXED API Auto-Discovery Script
        "docs/source/generate_api_docs.py": '''#!/usr/bin/env python3
"""Auto-generate API documentation for TrajectoLab - CORRECTED VERSION."""

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
        # Better filtering of test/internal modules
        if not any(exclude in modname for exclude in ["test", "__pycache__", ".pytest"]):
            modules.append(modname)

    print(f"✓ Found {len(modules)} modules")

    # Generate RST file for each module
    for module in modules:
        safe_name = module.replace(".", "_")
        rst_file = api_dir / f"{safe_name}.rst"

        # Better module documentation format
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

''',
        # FIXED Main Documentation Pages
        "docs/source/index.rst": """
TrajectoLab: Optimal Control Made Simple
========================================

TrajectoLab is a Python framework for solving optimal control problems using the Radau Pseudospectral Method.

🚀 Quick Example
----------------

.. code-block:: python

    import trajectolab as tl
    import numpy as np

    # Create problem
    problem = tl.Problem("Car Race")

    # Define variables
    t = problem.time(initial=0.0)
    pos = problem.state("position", initial=0.0, final=1.0)
    speed = problem.state("speed", initial=0.0)
    throttle = problem.control("throttle", boundary=(0.0, 1.0))

    # Dynamics and objective
    problem.dynamics({pos: speed, speed: throttle - speed})
    problem.minimize(t.final)

    # Solve
    problem.mesh([8, 8], np.linspace(-1, 1, 3))
    solution = tl.solve_adaptive(problem)

    if solution.status["success"]:
        print(f"Optimal time: {solution.final_time:.3f}")
        solution.plot()

📚 Documentation
----------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   examples/index
   api/index

🔧 Installation
---------------

.. code-block:: bash

    pip install trajectolab

✨ Features
-----------

* Simple problem definition API
* Adaptive mesh refinement
* High-precision solutions
* Built-in plotting and analysis
* Comprehensive type hints

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""",
        # FIXED Examples page (manual, not sphinx-gallery)
        "docs/source/examples/index.rst": """
Examples Gallery
================

TrajectoLab comes with several example problems demonstrating different optimal control scenarios.

Basic Examples
--------------

Car Race Problem
~~~~~~~~~~~~~~~~

Minimum time problem with speed limits.

.. literalinclude:: ../../../examples/car.py
   :language: python
   :caption: examples/car.py

Hypersensitive Problem
~~~~~~~~~~~~~~~~~~~~~~

Stiff optimal control problem with boundary conditions.

.. literalinclude:: ../../../examples/hypersensitive.py
   :language: python
   :caption: examples/hypersensitive.py

Advanced Examples
-----------------

HIV Immunology Model
~~~~~~~~~~~~~~~~~~~~

Multi-state biomedical control problem.

.. literalinclude:: ../../../examples/hiv.py
   :language: python
   :caption: examples/hiv.py

Space Shuttle Reentry
~~~~~~~~~~~~~~~~~~~~~

High-fidelity aerospace trajectory optimization.

.. literalinclude:: ../../../examples/shuttle.py
   :language: python
   :caption: examples/shuttle.py

Chemical Reactor
~~~~~~~~~~~~~~~~

Industrial process optimization.

.. literalinclude:: ../../../examples/crane.py
   :language: python
   :caption: examples/crane.py

Running Examples
----------------

All examples can be run directly:

.. code-block:: bash

    cd examples
    python car.py
    python hiv.py
    # etc.

Each example demonstrates different features:

* **car.py**: Basic problem setup, adaptive solving
* **hiv.py**: Multi-state dynamics, control bounds
* **hypersensitive.py**: Stiff systems, mesh refinement
* **shuttle.py**: Complex aerospace dynamics
* **crane.py**: Industrial optimization
""",
        "docs/source/installation.rst": """
Installation
============

Quick Installation
------------------

.. code-block:: bash

    pip install trajectolab

Development Installation
------------------------

.. code-block:: bash

    git clone https://github.com/trajectolab/trajectolab.git
    cd trajectolab
    pip install -e .

Requirements
------------

* Python 3.10+
* NumPy ≥ 1.18.0
* SciPy ≥ 1.4.0
* CasADi ≥ 3.5.0
* Matplotlib ≥ 3.1.0
* Pandas ≥ 1.0.0

Optional Dependencies
---------------------

For development:

.. code-block:: bash

    pip install -e ".[dev]"

This installs additional tools:

* Ruff (linting and formatting)
* MyPy (type checking)
* Pytest (testing)

Verification
------------

.. code-block:: python

    import trajectolab as tl
    print(f"TrajectoLab {tl.__version__} installed successfully!")

    # Run a quick test
    problem = tl.Problem("Test")
    print("✓ TrajectoLab working correctly!")
""",
        "docs/source/quickstart.rst": """
5-Minute Quickstart
===================

This guide gets you solving optimal control problems in 5 minutes.

Basic Problem Structure
-----------------------

Every TrajectoLab problem follows this pattern:

1. **Create Problem**
2. **Define Variables** (states, controls, time)
3. **Set Dynamics**
4. **Define Objective**
5. **Configure Mesh**
6. **Solve**

Example: Minimum Time Problem
-----------------------------

.. code-block:: python

    import trajectolab as tl
    import numpy as np

    # 1. Create problem
    problem = tl.Problem("Minimum Time")

    # 2. Define variables
    t = problem.time(initial=0.0)                          # Free final time
    x = problem.state("position", initial=0.0, final=1.0)  # Position: 0 → 1
    v = problem.state("velocity", initial=0.0)             # Velocity: start at rest
    u = problem.control("force", boundary=(-2.0, 2.0))     # Bounded control

    # 3. Set dynamics
    problem.dynamics({
        x: v,        # dx/dt = v
        v: u         # dv/dt = u
    })

    # 4. Define objective
    problem.minimize(t.final)  # Minimize final time

    # 5. Configure mesh and solve
    problem.mesh([10], np.array([-1.0, 1.0]))
    solution = tl.solve_fixed_mesh(problem)

    # 6. Results
    if solution.status["success"]:
        print(f"Minimum time: {solution.final_time:.3f} seconds")
        solution.plot()

Key Patterns
------------

**Constraint Specification:**

.. code-block:: python

    # Equality constraints
    x = problem.state("x", initial=5.0)           # x(0) = 5
    x = problem.state("x", final=10.0)            # x(tf) = 10

    # Inequality constraints
    x = problem.state("x", boundary=(-1.0, 1.0))  # -1 ≤ x(t) ≤ 1
    u = problem.control("u", boundary=(0.0, None)) # u ≥ 0

**Solver Selection:**

.. code-block:: python

    # Fixed mesh - fast
    solution = tl.solve_fixed_mesh(problem)

    # Adaptive mesh - high accuracy
    solution = tl.solve_adaptive(problem, error_tolerance=1e-8)

**Working with Solutions:**

.. code-block:: python

    if solution.status["success"]:
        # Get trajectory data
        time, position = solution.get_trajectory("position")
        time, velocity = solution.get_trajectory("velocity")

        # Plot results
        solution.plot()

        # Access final values
        print(f"Final time: {solution.final_time}")
        print(f"Objective: {solution.status['objective']}")

Next Steps
----------

* Explore the examples gallery
* Check the API reference
* Try adaptive mesh refinement for high-accuracy solutions
""",
        # FIXED Makefile
        "docs/Makefile": """# Makefile for Sphinx documentation

SPHINXOPTS    ?= -W --keep-going -j auto
SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean-api

# Generate API docs before building
api:
	@echo "🔄 Generating API documentation..."
	cd $(SOURCEDIR) && python generate_api_docs.py

# Build HTML with API generation
html: api
	@echo "🔨 Building HTML documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "✓ Documentation built in _build/html/"

# Clean everything including API docs
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf $(SOURCEDIR)/api/
	@echo "✓ Cleaned all build files"

# Quick build without API regeneration
html-fast:
	@echo "🔨 Building HTML documentation (fast mode)..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
""",
        # FIXED Requirements
        "docs/requirements.txt": """# TrajectoLab Documentation Requirements - FIXED VERSIONS
# Match versions with main package to avoid conflicts

# Core Sphinx
sphinx>=6.0.0,<8.0.0
sphinx-rtd-theme>=1.3.0
sphinx-autodoc-typehints>=1.20.0

# Main package dependencies (matching pyproject.toml)
numpy>=1.18.0
matplotlib>=3.1.0
scipy>=1.4.0
casadi>=3.5.0
pandas>=1.0.0

# Additional doc tools
myst-parser>=0.18.0
""",
        # FIXED GitHub Actions
        ".github/workflows/docs.yml": """name: Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'trajectolab/**'
      - 'examples/**'
      - 'docs/**'
      - 'pyproject.toml'
  pull_request:
    branches: [ main ]
    paths:
      - 'trajectolab/**'
      - 'examples/**'
      - 'docs/**'

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y build-essential

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install -r docs/requirements.txt

    - name: Build documentation
      run: |
        cd docs
        make html

    - name: Check for build warnings
      run: |
        if [ -f docs/_build/html/.buildinfo ]; then
          echo "✓ Documentation built successfully"
        else
          echo "❌ Documentation build failed"
          exit 1
        fi

    - name: Upload pages artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: docs/_build/html

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    runs-on: ubuntu-latest

    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v3
""",
        # FIXED Custom CSS
        "docs/source/_static/custom.css": """/* TrajectoLab Documentation Styling - CORRECTED */

/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Modern typography */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    font-size: 16px;
}

/* Code styling */
code, .highlight {
    font-family: 'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', monospace;
    font-size: 14px;
}

code {
    background-color: #f8f9fa;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    border: 1px solid #e9ecef;
}

/* Headers */
h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.5rem;
    font-weight: 600;
}

h2 {
    color: #34495e;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 0.3rem;
    font-weight: 500;
}

h3 {
    color: #2c3e50;
    font-weight: 500;
}

/* Admonitions */
.admonition {
    border-radius: 6px;
    border-left: 4px solid #3498db;
    background-color: #ebf3fd;
    padding: 1rem;
    margin: 1.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.admonition.note {
    border-left-color: #3498db;
    background-color: #ebf3fd;
}

.admonition.warning {
    border-left-color: #f39c12;
    background-color: #fef9e7;
}

.admonition.danger {
    border-left-color: #e74c3c;
    background-color: #fdedec;
}

/* Code blocks */
div.highlight {
    border-radius: 6px;
    border: 1px solid #e1e8ed;
    overflow: hidden;
    margin: 1rem 0;
}

/* Tables */
table.docutils {
    border-collapse: collapse;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-radius: 6px;
    overflow: hidden;
    width: 100%;
}

table.docutils th {
    background-color: #3498db;
    color: white;
    font-weight: 600;
    padding: 12px;
    text-align: left;
}

table.docutils td {
    padding: 12px;
    border-bottom: 1px solid #ecf0f1;
}

table.docutils tr:hover {
    background-color: #f8f9fa;
}

/* Links */
a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    color: #2980b9;
    text-decoration: underline;
}

/* Sidebar customization */
.wy-side-nav-search {
    background-color: #2c3e50;
}

.wy-side-nav-search > a {
    color: white;
    font-weight: 600;
}

/* Navigation */
.wy-menu-vertical a {
    color: #2c3e50;
}

.wy-menu-vertical a:hover {
    background-color: #ecf0f1;
    color: #2c3e50;
}

/* API documentation styling */
dl.class, dl.function, dl.method {
    border: 1px solid #e1e8ed;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
    background-color: #fafbfc;
}

dt.sig {
    background-color: transparent;
    border: none;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 500;
}

/* Responsive improvements */
@media (max-width: 768px) {
    body {
        font-size: 14px;
    }

    h1 {
        font-size: 1.8em;
    }

    h2 {
        font-size: 1.4em;
    }
}
""",
        # Windows batch file for convenience
        "docs/make.bat": """@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "api" goto api

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
echo Cleaning documentation...
%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
if exist %SOURCEDIR%\\api rmdir /s /q %SOURCEDIR%\\api
echo ✓ Cleaned all build files
goto end

:api
echo 🔄 Generating API documentation...
cd %SOURCEDIR% && python generate_api_docs.py
goto end

:html
echo 🔄 Generating API documentation...
cd %SOURCEDIR% && python generate_api_docs.py
cd ..
echo 🔨 Building HTML documentation...
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo ✓ Documentation built in _build/html/
goto end

:end
popd
""",
    }

    # Write all files
    print("📝 Creating files...")
    for file_path, content in files.items():
        full_path = repo_root / file_path
        full_path.write_text(content, encoding="utf-8")
        print(f"  ✓ {file_path}")

    print("\n🎉 CORRECTED Documentation system created!")
    print("\n📋 Next steps:")
    print("1. Install documentation dependencies:")
    print("   cd docs && pip install -r requirements.txt")
    print("\n2. Test the build:")
    print("   Windows: make.bat html")
    print("   Unix:    make html")
    print("\n3. Push to GitHub and enable Pages in repository settings")
    print("\n4. Your docs will be at: https://[username].github.io/[repo]/")
    print("\n✨ Key fixes applied:")
    print("  - Removed problematic sphinx-gallery configuration")
    print("  - Fixed dependency versions to match main package")
    print("  - Added missing pandas dependency")
    print("  - Created manual examples gallery using literalinclude")
    print("  - Fixed CSS loading and GitHub Actions versions")
    print("  - Added Windows batch support")
    print("  - Improved API documentation organization")


if __name__ == "__main__":
    create_docs_structure()
