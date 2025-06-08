# Brachistochrone

## Mathematical Formulation

The brachistochrone problem is a classic optimal control problem that finds the curve of fastest descent between two points under the influence of gravity.

### Problem Statement

Find the optimal control $u(t)$ that minimizes the time:

$$J = t_f$$

Subject to the dynamic constraints:

$$\frac{dx}{dt} = v \cos u$$
$$\frac{dy}{dt} = v \sin u$$
$$\frac{dv}{dt} = g_0 \sin u$$

With boundary conditions:
- Initial conditions: $x(0) = 0$, $y(0) = 0$, $v(0) = 0$
- Final condition: $x(t_f) = 1$
- Path constraints: $0 \leq x \leq 10$, $0 \leq y \leq 10$, $0 \leq v \leq 10$
- Control bounds: $0 \leq u \leq \pi/2$

Where $g_0 = 32.174 \text{ ft/sec}^2$  is the gravitational acceleration.

## Expected Results

Reference objective value: $J^* = 0.312480130$ (from literature [29, Ex. 4.10]; [66, pp. 81, 119])

## Running This Example

```bash
cd examples/brachistochrone
python brachistochrone.py
```
