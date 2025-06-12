# MAPTOR: Multiphase Adaptive Trajectory Optimizer

[![PyPI version](https://badge.fury.io/py/maptor.svg)](https://badge.fury.io/py/maptor)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/maptor/badge/?version=latest)](https://maptor.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python framework for **trajectory optimization** using optimal control theory. MAPTOR solves problems involving the motion of vehicles, robots, spacecraft, and other dynamic systems through space and time using the Legendre-Gauss-Radau pseudospectral method.

## What is Trajectory Optimization?

Trajectory optimization finds the best path for a dynamic system to follow, considering:
- **Physics**: System dynamics and constraints
- **Objectives**: Minimize time, fuel, energy, or tracking error
- **Constraints**: Safety limits, obstacle avoidance, boundary conditions

**Mathematical Foundation**: MAPTOR applies optimal control theory to transform continuous trajectory optimization problems into solvable nonlinear programming problems through spectral collocation methods.

## Core Methodology

MAPTOR implements the **Legendre-Gauss-Radau pseudospectral method** with:

- **Spectral accuracy**: Exponential convergence for smooth solutions
- **Adaptive mesh refinement**: Automatic error control through phs-Adaptive refinement method
- **Multiphase capability**: Complex missions with automatic phase linking
- **Symbolic computation**: Built on CasADi for exact differentiation and optimization

## Quick Start

```python
import maptor as mtor
import numpy as np

# Minimum-time trajectory: reach target with bounded control
problem = mtor.Problem("Minimum Time to Target")
phase = problem.set_phase(1)

# Variables
t = phase.time(initial=0.0)  # Free final time
position = phase.state("position", initial=0.0, final=1.0)
velocity = phase.state("velocity", initial=0.0, final=0.0)
force = phase.control("force", boundary=(-2.0, 2.0))

# Dynamics and objective
phase.dynamics({position: velocity, velocity: force})
problem.minimize(t.final)

# Solve
phase.mesh([8], [-1.0, 1.0])
solution = mtor.solve_adaptive(problem)

if solution.status["success"]:
    print(f"Optimal time: {solution.status['objective']:.3f}")
    solution.plot()
```

## Problem Classes

**Beyond Spatial Trajectories**: MAPTOR also handles abstract optimal control problems where "trajectory" refers to the evolution of any system state over time (chemical processes, financial optimization, resource allocation).

## Installation

```bash
pip install maptor
```

**Requirements**: Python 3.10+, NumPy, SciPy, CasADi, Matplotlib

**Development Installation**:
```bash
git clone https://github.com/maptor/maptor.git
cd maptor
pip install -e .
```

## Documentation

| Resource | Description |
|----------|-------------|
| **[Installation Guide](https://maptor.readthedocs.io/en/latest/installation.html)** | Setup and dependencies |
| **[Quick Start](https://maptor.readthedocs.io/en/latest/quickstart.html)** | Basic workflow and first example |
| **[Problem Definition Tutorial](https://maptor.readthedocs.io/en/latest/tutorials/problem_definition.html)** | Comprehensive problem construction guide |
| **[Solution Analysis Tutorial](https://maptor.readthedocs.io/en/latest/tutorials/solution_access.html)** | Working with optimization results |
| **[Examples Gallery](https://maptor.readthedocs.io/en/latest/examples/index.html)** | Complete problems with mathematical formulations |
| **[API Reference](https://maptor.readthedocs.io/en/latest/api/index.html)** | Detailed function documentation |

## Example Trajectories

The examples gallery demonstrates trajectory optimization across multiple domains:

### Aerospace Applications
- **[Spacecraft Orbit Transfer](https://maptor.readthedocs.io/en/latest/examples/low_thrust_orbit_transfer.html)**: Low-thrust trajectory optimization with orbital mechanics
- **[Multiphase Launch Vehicle](https://maptor.readthedocs.io/en/latest/examples/multiphase_vehicle_launch.html)**: Complex mission with stage separations
- **[Hang Glider Flight](https://maptor.readthedocs.io/en/latest/examples/hang_glider.html)**: Atmospheric flight with thermal updrafts
- **[Shuttle Reentry](https://maptor.readthedocs.io/en/latest/examples/shuttle_reentry.html)**: Entry trajectory optimization

### Robotics and Control
- **[Robot Arm Motion](https://maptor.readthedocs.io/en/latest/examples/robot_arm.html)**: Minimum-time manipulator trajectories
- **[Cart-Pole Swing-Up](https://maptor.readthedocs.io/en/latest/examples/cartpole.html)**: Underactuated system control
- **[Dynamic Obstacle Avoidance](https://maptor.readthedocs.io/en/latest/examples/dynamic_obstacle_avoidance.html)**: Real-time path planning
- **[Two-Phase Robot Tracking](https://maptor.readthedocs.io/en/latest/examples/two_phase_robot.html)**: Multiphase trajectory following

### Vehicle and Racing
- **[Race Car Optimization](https://maptor.readthedocs.io/en/latest/examples/race_car.html)**: Minimum-time racing line with speed constraints

### Classical Problems
- **[Brachistochrone](https://maptor.readthedocs.io/en/latest/examples/brachistochrone.html)**: Fastest descent under gravity
- **[LQR Control](https://maptor.readthedocs.io/en/latest/examples/lqr.html)**: Linear-quadratic regulator
- **[Hypersensitive Problem](https://maptor.readthedocs.io/en/latest/examples/hypersensitive.html)**: Challenging optimal control benchmark

## Architecture

MAPTOR provides a layered architecture separating trajectory design from numerical implementation:

```
User API (Problem, solve_adaptive, solve_fixed_mesh)
         ↓
Trajectory Definition (States, controls, dynamics, constraints)
         ↓
Mathematical Framework (Radau pseudospectral method)
         ↓
Symbolic Computation (CasADi expressions and differentiation)
         ↓
Optimization (IPOPT nonlinear programming solver)
```

**Key Design Principles**:
- **Intuitive API**: Define trajectories naturally without numerical details
- **Automatic differentiation**: CasADi handles complex derivative computations
- **Adaptive precision**: Mesh refinement ensures solution accuracy
- **Multiphase support**: Complex missions with automatic phase transitions



## Contributing

We welcome contributions from the trajectory optimization and optimal control community.

## License

MAPTOR is licensed under the [GNU Lesser General Public License v3.0](LICENSE). This allows use in both open source and proprietary applications while ensuring improvements to the core library remain open.

## Citation

If you use MAPTOR in academic research, please cite:

```bibtex
@software{maptor2025,
  title={MAPTOR: Multiphase Adaptive Trajectory Optimizer},
  author={Timothy, David},
  year={2025},
  url={https://github.com/maptor/maptor},
  version={0.1.0}
}
```

## References

MAPTOR builds upon established methods in computational optimal control:

**Pseudospectral Methods**:
- Agamawi, Y. M., & Rao, A. V. (2020). CGPOPS: A C++ Software for Solving Multiple-Phase Optimal Control Problems Using Adaptive Gaussian Quadrature Collocation and Sparse Nonlinear Programming. *ACM Transactions on Mathematical Software*, 46(3), Article 25. https://doi.org/10.1145/3390463

**Adaptive Mesh Refinement**:
- Haman III, G. V., & Rao, A. V. (2024). Adaptive Mesh Refinement and Error Estimation Method for Optimal Control Using Direct Collocation. *arXiv preprint arXiv:2410.07488*. https://arxiv.org/abs/2410.07488

**Symbolic Computation Framework**:
- Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., & Diehl, M. (2019). CasADi -- A software framework for nonlinear optimization and optimal control. *Mathematical Programming Computation*, 11(1), 1-36. https://doi.org/10.1007/s12532-018-0139-4

## Support

- **Documentation**: [https://maptor.readthedocs.io](https://maptor.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/maptor/maptor/issues)

## Acknowledgments

MAPTOR implements methods from the computational optimal control literature, particularly pseudospectral collocation techniques and adaptive mesh refinement strategies. The framework leverages CasADi for symbolic computation and automatic differentiation.

---

**Next Steps**: Begin with the [Quick Start Guide](https://maptor.readthedocs.io/en/latest/quickstart.html) or explore the [Examples Gallery](https://maptor.readthedocs.io/en/latest/examples/index.html) to see MAPTOR applied to trajectory optimization problems in your domain.
