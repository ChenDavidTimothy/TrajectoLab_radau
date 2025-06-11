# Hang Glider Problem

## Mathematical Formulation

The hang glider problem demonstrates optimal trajectory planning for a glider in the presence of updraft currents, maximizing horizontal distance traveled.

### Problem Statement

Find the optimal lift coefficient $C_L(t)$ that maximizes the horizontal distance:

$$J = x(t_f)$$

Subject to the aerodynamic dynamics:

$$\frac{dx}{dt} = v_x$$
$$\frac{dy}{dt} = v_y$$
$$\frac{dv_x}{dt} = \frac{1}{m}(-L \sin \eta - D \cos \eta)$$
$$\frac{dv_y}{dt} = \frac{1}{m}(L \cos \eta - D \sin \eta - mg)$$

### Aerodynamic Model

The aerodynamic forces are computed as:

- **Drag coefficient**: $C_D = C_0 + k C_L^2$
- **Relative velocity**: $v_r = \sqrt{v_x^2 + v_y^2}$
- **Drag force**: $D = \frac{1}{2} C_D \rho S v_r^2$
- **Lift force**: $L = \frac{1}{2} C_L \rho S v_r^2$

### Updraft Model

The vertical updraft velocity varies spatially:

$$u_a = u_M (1 - X) e^{-X}$$

where $X = \left(\frac{x}{R} - 2.5\right)^2$

The relative vertical velocity is: $V_y = v_y - u_a$

Flight path angle components:
- $\sin \eta = \frac{V_y}{v_r}$
- $\cos \eta = \frac{v_x}{v_r}$

### Physical Parameters

- Mass: $m = 100$ kg
- Wing area: $S = 14$ m²
- Air density: $\rho = 1.13$ kg/m³
- Base drag coefficient: $C_0 = 0.034$
- Induced drag factor: $k = 0.069662$
- Maximum updraft: $u_M = 2.5$ m/s
- Updraft length scale: $R = 100$ m

### Boundary Conditions

- **Initial conditions**: $x(0) = 0$ m, $y(0) = 1000$ m, $v_x(0) = 13.2275675$ m/s, $v_y(0) = -1.28750052$ m/s
- **Final conditions**: $y(t_f) = 900$ m, $v_x(t_f) = 13.2275675$ m/s, $v_y(t_f) = -1.28750052$ m/s
- **Control bounds**: $0 \leq C_L \leq 1.4$
- **State bounds**: $0 \leq x \leq 1500$ m, $0 \leq y \leq 1100$ m, $0 \leq v_x \leq 15$ m/s, $-4 \leq v_y \leq 4$ m/s

## Expected Results

The optimal solution demonstrates how the glider exploits updraft currents to maximize horizontal distance while satisfying altitude and velocity constraints at the final time.

## Running This Example

```bash
cd examples/hang_glider
python hang_glider.py
```
