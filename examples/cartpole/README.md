# Cart-Pole Swing-Up

## Mathematical Formulation

The cart-pole swing-up problem demonstrates optimal control of an underactuated nonlinear system.

### Problem Statement

Find the optimal control force $f_x(t)$ that minimizes:

$J = t_f + 0.01 \int_0^{t_f} f_x^2 \, dt$

Subject to the dynamic constraints:

$2\ddot{x} + \ddot{\theta} \cos \theta - \dot{\theta}^2 \sin \theta = f_x$

$\ddot{x} \cos \theta + \ddot{\theta} + \sin \theta = 0$

With boundary conditions:
- Initial condition: $[x, \theta, \dot{x}, \dot{\theta}](0) = [0, 0, 0, 0]$
- Final condition: $[x, \theta, \dot{x}, \dot{\theta}](t_f) = [0, \pi, 0, 0]$
- Control bounds: $|f_x| \leq 20$ N

Where $\theta = 0$ is pendulum hanging down and $\theta = \pi$ is upright.

## Running This Example

```bash
cd examples/cart_pole_swing_up
python cart_pole_swing_up.py
```
