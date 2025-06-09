MAPTOR: Multiphase Adaptive Trajectory Optimizer
================================================

**Date**: |today| **Version**: |version|

**Useful links**:
`Install <https://pypi.org/project/maptor/>`__ |
`Source Repository <https://github.com/maptor/maptor>`__ |
`Issues & Ideas <https://github.com/maptor/maptor/issues>`__ |

**MAPTOR** is a Python framework for solving optimal control problems using the Radau Pseudospectral Method.

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :text-align: center

        **Quick Start**
        ^^^

        Get up and running with MAPTOR in 5 minutes.
        Learn the basic problem definition pattern and solve your first optimal control problem.

        +++

        .. button-ref:: quickstart
            :color: primary
            :click-parent:

            Get Started

    .. grid-item-card::
        :text-align: center

        **Tutorials**
        ^^^

        In-depth guides covering solution access, advanced features,
        and best practices for multiphase optimal control problems.

        +++

        .. button-ref:: tutorials/index
            :color: primary
            :click-parent:

            Learn More

    .. grid-item-card::
        :text-align: center

        **Examples Gallery**
        ^^^

        Complete, runnable examples from aerospace to robotics.
        Each example includes mathematical formulation and detailed implementation.

        +++

        .. button-ref:: examples/index
            :color: primary
            :click-parent:

            Browse Examples

    .. grid-item-card::
        :text-align: center

        **API Reference**
        ^^^

        Comprehensive reference documentation for all public classes,
        functions, and methods in the MAPTOR framework.

        +++

        .. button-ref:: api/index
            :color: primary
            :click-parent:

            API Docs

Quick Example
-------------

.. code-block:: python

    import maptor as mtor
    import numpy as np

    # Create problem
    problem = mtor.Problem("Car Race")

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
    solution = mtor.solve_adaptive(problem)

    if solution.status["success"]:
        print(f"Optimal time: {solution.final_time:.3f}")
        solution.plot()

Key Features
------------

**Simple API**: Intuitive problem definition with automatic mesh generation

**Adaptive Refinement**: High-precision solutions with automatic error control

**Multiphase Support**: Complex mission profiles with phase transitions

**Built-in Analysis**: Comprehensive plotting and solution diagnostics

**Type Safety**: Full type hints for robust development

Installation
------------

.. code-block:: bash

    pip install maptor

Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tutorials/index
   examples/index
   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
