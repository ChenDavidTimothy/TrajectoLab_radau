# Two Phase Path Tracking Robot

## Mathematical Formulation

The two-phase path tracking robot problem demonstrates multiphase trajectory optimization with time-varying reference tracking objectives.

### Problem Statement

Find optimal control inputs $u_1(t)$ and $u_2(t)$ that minimize the total tracking cost:

$$J = \int_0^1 \left[ w_1(x_1 - x_{1ref})^2 + w_2(x_2 - x_{2ref})^2 + w_3(x_3 - x_{3ref})^2 + w_4(x_4 - x_{4ref})^2 \right] dt$$
$$+ \int_1^2 \left[ w_1(x_1 - x_{1ref})^2 + w_2(x_2 - x_{2ref})^2 + w_3(x_3 - x_{3ref})^2 + w_4(x_4 - x_{4ref})^2 \right] dt$$

Subject to the integrator dynamics:

$$\frac{dx_1}{dt} = x_3$$
$$\frac{dx_2}{dt} = x_4$$
$$\frac{dx_3}{dt} = u_1$$
$$\frac{dx_4}{dt} = u_2$$

### Phase Structure

**Phase 1 (0 ≤ t ≤ 1):**
- Initial conditions: $x_1(0) = 0$, $x_2(0) = 0$, $x_3(0) = 0.5$, $x_4(0) = 0$
- Reference trajectory: $x_{1ref} = t/2$, $x_{2ref} = 0$, $x_{3ref} = 0.5$, $x_{4ref} = 0$

**Phase 2 (1 ≤ t ≤ 2):**
- Automatic continuity from Phase 1
- Final conditions: $x_1(2) = 0.5$, $x_2(2) = 0.5$, $x_3(2) = 0$, $x_4(2) = 0.5$
- Reference trajectory: $x_{1ref} = 0.5$, $x_{2ref} = (t-1)/2$, $x_{3ref} = 0$, $x_{4ref} = 0.5$

### Constraints

- State bounds: $-10 \leq x_i \leq 10$ for all states
- Control bounds: $-10 \leq u_j \leq 10$ for all controls
- Automatic phase continuity: $x_i(1^-) = x_i(1^+)$ for all states

### Weighting Parameters

- Position tracking weights: $w_1 = w_2 = 100$
- Velocity tracking weights: $w_3 = w_4 = 500$

## Expected Results

The solution demonstrates smooth reference tracking with automatic phase transitions. The robot follows the time-varying reference in Phase 1, then transitions smoothly to track the Phase 2 reference trajectory.

## Running This Example

```bash
cd examples/two_phase_robot
python two_phase_robot.py
```
