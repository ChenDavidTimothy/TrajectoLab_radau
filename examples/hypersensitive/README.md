### Problem Statement

Find the optimal control $u(t)$ that minimizes the cost functional:

$$J = \int_0^{40} \frac{1}{2}(x^2 + u^2) \, dt$$

Subject to the dynamic constraint:

$$\frac{dx}{dt} = -x^3 + u(t)$$

With boundary conditions:
- Initial condition: $x(0) = 1.5$
- Final condition: $x(40) = 1.0$

### Problem Characteristics

This problem exhibits **hypersensitive** behavior due to:

1. **Rapid State Transitions**: The cubic nonlinearity $-x^3$ creates steep gradients
2. **Boundary Layer Structure**: Sharp transitions occur near the boundaries
3. **Stiff Dynamics**: The system requires fine temporal resolution to capture solution features accurately

### Solution Features

The optimal solution typically exhibits:
- **Initial Boundary Layer**: Rapid decrease from $x(0) = 1.5$
- **Interior Region**: Gradual evolution with balanced state-control trade-off
- **Terminal Boundary Layer**: Sharp adjustment to reach $x(40) = 1.0$

### Numerical Implementation

This example demonstrates:
- **Multi-interval mesh**: Uses variable mesh density `[8, 8, 8]` with nodes $[-1.0, -1/3, 1/3, 1.0]$
- **Adaptive refinement**: High-degree polynomials (5-15) with tight tolerance ($10^{-3}$)
- **Comparison approach**: Validates adaptive vs fixed mesh solutions

### Expected Results

Reference objective value: $J^* \approx 0.035$ (problem-dependent based on discretization)

The adaptive algorithm should converge within 30 iterations, automatically refining the mesh in regions with steep gradients while maintaining efficiency in smooth regions.
