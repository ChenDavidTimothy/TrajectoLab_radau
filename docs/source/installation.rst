
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
