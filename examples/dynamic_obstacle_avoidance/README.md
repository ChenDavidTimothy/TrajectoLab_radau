# Dynamic Obstacle Avoidance

## Mathematical Formulation

### Problem Statement

Find the optimal control inputs $\delta(t)$ and $a(t)$ that minimize the transit time:

$$J = t_f$$

Subject to the bicycle model dynamics:

$$\frac{dx}{dt} = v \cos\theta$$

$$\frac{dy}{dt} = v \sin\theta$$

$$\frac{d\theta}{dt} = \frac{v \tan\delta}{L}$$

$$\frac{dv}{dt} = a$$

And the collision avoidance constraint:

$$(x - x_{\text{obs}}(t))^2 + (y - y_{\text{obs}}(t))^2 \geq d_{\min}^2$$

### Boundary Conditions

- **Initial conditions**: $x(0) = 0$, $y(0) = 0$, $\theta(0) = \frac{\pi}{4}$, $v(0) = 1.0$
- **Final conditions**: $x(t_f) = 20$, $y(t_f) = 20$, $\theta(t_f) = \text{free}$, $v(t_f) = \text{free}$
- **Control bounds**: $-0.5 \leq \delta \leq 0.5$ rad, $-3.0 \leq a \leq 3.0$ m/s²
- **State bounds**: $0.5 \leq v \leq 20$ m/s, $-5 \leq x,y \leq 25$ m

### Physical Parameters

- Vehicle wheelbase: $L = 2.5$ m
- Vehicle radius: $r_v = 1.5$ m
- Obstacle radius: $r_o = 2.5$ m
- Minimum separation: $d_{\min} = r_v + r_o = 4.0$ m

### Obstacle Trajectory

The obstacle follows a predetermined path with waypoints:
- $(5.0, 5.0)$ at $t = 0$ s
- $(12.0, 12.0)$ at $t = 3$ s
- $(15.0, 15.0)$ at $t = 6$ s
- $(20.0, 20.0)$ at $t = 12$ s

### State Variables

- $x(t)$: Vehicle x-position (m)
- $y(t)$: Vehicle y-position (m)
- $\theta(t)$: Vehicle heading angle (rad)
- $v(t)$: Vehicle speed (m/s)

### Control Variables

- $\delta(t)$: Steering angle (rad)
- $a(t)$: Acceleration (m/s²)

### Notes

This problem demonstrates real-time collision avoidance with a moving obstacle using a kinematic bicycle model for the vehicle dynamics.

## Running This Example

```bash
cd examples/dynamic_obstacle_avoidance
python dynamic_obstacle_avoidance.py
```
