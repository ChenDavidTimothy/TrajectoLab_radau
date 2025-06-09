# Dynamic Obstacle Avoidance

## Mathematical Formulation

The dynamic obstacle avoidance problem demonstrates real-time trajectory optimization for a vehicle navigating around a moving obstacle.

### Problem Statement

Find the optimal steering angle $\delta(t)$ and acceleration $a(t)$ that minimize the mission time:

$$J = t_f$$

Subject to the bicycle model dynamics:

$$\frac{dx}{dt} = v \cos \theta$$
$$\frac{dy}{dt} = v \sin \theta$$
$$\frac{d\theta}{dt} = \frac{v \tan \delta}{L}$$
$$\frac{dv}{dt} = a$$

With boundary conditions:
- Initial conditions: $x(0) = 0$, $y(0) = 0$, $\theta(0) = \pi/4$, $v(0) = 1$
- Final conditions: $x(t_f) = 20$, $y(t_f) = 20$
- Control bounds: $|\delta| \leq 0.5$ rad, $|a| \leq 3.0$ m/s²
- Velocity bounds: $0.5 \leq v \leq 20$ m/s

### Collision Avoidance Constraint

The vehicle must maintain a safe distance from the moving obstacle:

$$\sqrt{(x - x_{obs}(t))^2 + (y - y_{obs}(t))^2} \geq r_{vehicle} + r_{obstacle}$$

Where:
- $r_{vehicle} = 1.5$ m (vehicle safety radius)
- $r_{obstacle} = 2.5$ m (obstacle radius)
- The obstacle follows a predefined trajectory with waypoints

## Expected Results

The solution demonstrates real-time collision avoidance with smooth trajectory generation and minimal mission time while respecting all safety constraints.

## Running This Example

```bash
cd examples/dynamic_obstacle_avoidance
python dynamic_obstacle_avoidance.py
```

### Animation

The example includes an animation script to visualize the vehicle's path and obstacle motion:

```bash
python dynamic_obstacle_avoidance_animate.py
```

This creates an MP4 animation showing the vehicle trajectory and moving obstacle, demonstrating the effectiveness of the collision avoidance strategy.
