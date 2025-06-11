# Alp Rider Problem

## Mathematical Formulation

The Alp rider problem is a classic optimal control benchmark that demonstrates terrain-following trajectory optimization with state constraints.

### Problem Statement

Find the optimal controls $u_1(t)$ and $u_2(t)$ that minimize the cost functional:

$$J = \int_0^{20} \left[100(x_1^2 + x_2^2 + x_3^2 + x_4^2) + 0.01(u_1^2 + u_2^2)\right] dt$$

Subject to the linear dynamic system:

$$\frac{dx_1}{dt} = -10x_1 + u_1 + u_2$$
$$\frac{dx_2}{dt} = -2x_2 + u_1 + 2u_2$$
$$\frac{dx_3}{dt} = -3x_3 + 5x_4 + u_1 - u_2$$
$$\frac{dx_4}{dt} = 5x_3 - 3x_4 + u_1 + 3u_2$$

With boundary conditions:
- Initial conditions: $[x_1, x_2, x_3, x_4](0) = [2, 1, 2, 1]$
- Final conditions: $[x_1, x_2, x_3, x_4](20) = [2, 3, 1, -2]$

Control bounds:
- $-500 \leq u_1, u_2 \leq 500$

State bounds:
- $-4 \leq x_i \leq 4$ for all states

### Terrain-Following Constraint

The path constraint enforces terrain following:

$$x_1^2 + x_2^2 + x_3^2 + x_4^2 \geq p(t)$$

Where the terrain function is:

$$p(t) = 3e^{-12(t-3)^2} + 3e^{-10(t-6)^2} + 3e^{-6(t-10)^2} + 8e^{-4(t-15)^2} + 0.01$$

This creates terrain peaks at different time points that the vehicle must navigate around.

## Expected Results

Reference objective value: $J^* \approx 2030.85609$ (PSOPT reference)

The reference objective is compared with PSOPT optimization software.

## Running This Example

```bash
cd examples/alp_rider_psopt_exact
python alp_rider_psopt_exact.py
```
