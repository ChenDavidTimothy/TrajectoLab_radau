
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
    problem.set_mesh([8, 8], np.linspace(-1, 1, 3))
    solution = tl.solve_adaptive(problem)

    if solution.success:
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
