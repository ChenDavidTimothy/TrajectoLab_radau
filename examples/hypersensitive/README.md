# Hypersensitive

## Mathematical Formulation

The hypersensitive problem is a classic optimal control benchmark that tests the numerical robustness of pseudospectral methods due to its rapid state variations and boundary layer behavior.

### Problem Statement

Find the optimal control $u(t)$ that minimizes the cost functional:

$$J = \int_0^{40} \frac{1}{2}(x^2 + u^2) \ dt$$

Subject to the dynamic constraint:

$$\frac{dx}{dt} = -x^3 + u(t)$$

With boundary conditions:
- Initial condition: $x(0) = 1.5$
- Final condition: $x(40) = 1.0$


## Expected Results

Reference objective value: $J^* \approx 0.035$ (problem-dependent based on discretization)

The reference objective is compared with J.T.Betts Practical Optimal control


## Running This Example

```bash
cd examples/hypersensitive
python hypersensitive.py
```
