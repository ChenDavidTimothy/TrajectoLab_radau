# Cart-Pole Swing-Up

## Mathematical Formulation

### Problem Statement

Find the optimal force $f_x(t)$ that minimizes the combined time and control effort:

$$J = t_f + \int_0^{t_f} 0.01 f_x^2 dt$$

Subject to the cart-pole dynamic constraints:

$$\frac{dx}{dt} = \dot{x}$$

$$\frac{d\theta}{dt} = \dot{\theta}$$

$$\frac{d\dot{x}}{dt} = \frac{f_x + \sin\theta \cos\theta + \dot{\theta}^2 \sin\theta}{2 - \cos^2\theta}$$

$$\frac{d\dot{\theta}}{dt} = \frac{-\cos\theta \cdot f_x - \dot{\theta}^2 \sin\theta \cos\theta - 2\sin\theta}{2 - \cos^2\theta}$$

### Boundary Conditions

- **Initial conditions**: $x(0) = 0$, $\theta(0) = 0$, $\dot{x}(0) = 0$, $\dot{\theta}(0) = 0$
- **Final conditions**: $x(t_f) = 0$, $\theta(t_f) = \pi$, $\dot{x}(t_f) = 0$, $\dot{\theta}(t_f) = 0$
- **Control bounds**: $-20 \leq f_x \leq 20$ N

### Physical Parameters

- Cart mass: $m_c = 1$ kg (normalized)
- Pendulum mass: $m_p = 1$ kg (normalized)
- Pendulum length: $l = 1$ m (normalized)
- Gravitational acceleration: $g = 1$ m/s² (normalized)

### State Variables

- $x(t)$: Cart position (m)
- $\theta(t)$: Pendulum angle (rad, 0 = down, $\pi$ = up)
- $\dot{x}(t)$: Cart velocity (m/s)
- $\dot{\theta}(t)$: Pendulum angular velocity (rad/s)

### Control Variable

- $f_x(t)$: Horizontal force applied to cart (N)

### Notes

This is the classic cart-pole swing-up problem where the goal is to swing the pendulum from the stable downward position to the unstable upward position while returning the cart to its original location.

## Running This Example

```bash
cd examples/cartpole
python cartpole.py
```
