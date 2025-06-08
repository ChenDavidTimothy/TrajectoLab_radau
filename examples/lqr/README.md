# LQR Problem

## Mathematical Formulation

The LQR (Linear Quadratic Regulator) problem is a classic optimal control benchmark featuring linear dynamics and quadratic cost structure.

### Problem Statement

Find the optimal control $u(t)$ that minimizes the cost functional:

$$J = \int_0^{1} (0.625x^2 + 0.5xu + 0.5u^2) \, dt$$

Subject to the dynamic constraint:

$$\frac{dx}{dt} = 0.5x + u$$

With boundary conditions:
- Initial condition: $x(0) = 1$
- Final condition: Free (unconstrained)

## Expected Results

Reference objective value: $J^* \approx 0.380797077977481140$ (TOMLAB reference)

The reference objective is compared with TOMLAB Optimization Inc. solution.

## Running This Example

```bash
cd examples/lqr
python lqr.py
```
