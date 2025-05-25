
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

.. literalinclude:: ../../../examples/shuttle_simple.py
   :language: python
   :caption: examples/shuttle_simple.py

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
* **shuttle_simple.py**: Complex aerospace dynamics
* **crane.py**: Industrial optimization
