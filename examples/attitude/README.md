# Space Station Attitude Control

## Mathematical Formulation

The space station attitude control problem demonstrates optimal control of spacecraft orientation using Euler-Rodrigues parameters and angular momentum management.

### Problem Statement

Find the optimal control torque $\mathbf{u}(t)$ that minimizes the cost functional:

$$J = \frac{1}{2} \int_{0}^{1800} \mathbf{u}^T \mathbf{u} \, dt$$

Subject to the dynamic constraints:

$$\dot{\boldsymbol{\omega}} = \mathbf{J}^{-1}\left\{\boldsymbol{\tau}_{gg}(\mathbf{r}) - \boldsymbol{\omega}^{\otimes}[\mathbf{J}\boldsymbol{\omega} + \mathbf{h}] - \mathbf{u}\right\}$$

$$\dot{\mathbf{r}} = \frac{1}{2}\left[\mathbf{r}\mathbf{r}^T + \mathbf{I} + \mathbf{r}^{\otimes}\right][\boldsymbol{\omega} - \boldsymbol{\omega}_0(\mathbf{r})]$$

$$\dot{\mathbf{h}} = \mathbf{u}$$

With boundary conditions:
- Initial conditions: $\boldsymbol{\omega}(0) = \bar{\boldsymbol{\omega}}_0$, $\mathbf{r}(0) = \bar{\mathbf{r}}_0$, $\mathbf{h}(0) = \bar{\mathbf{h}}_0$
- Final equilibrium: $\dot{\boldsymbol{\omega}}(t_f) = 0$, $\dot{\mathbf{r}}(t_f) = 0$
- Mission time: $t_f = 1800$ seconds

### Path Constraints

Angular momentum magnitude constraint:
$$\|\mathbf{h}\| \leq h_{max} = 10000$$

### System Parameters

Where $(\boldsymbol{\omega}, \mathbf{r}, \mathbf{h})$ is the state and $\mathbf{u}$ is the control:
- $\boldsymbol{\omega}$: angular velocity vector
- $\mathbf{r}$: Euler-Rodrigues parameter vector
- $\mathbf{h}$: angular momentum vector
- $\mathbf{u}$: input moment vector (control)

Auxiliary relations:
- $\boldsymbol{\omega}_0(\mathbf{r}) = -\omega_{orb} \mathbf{C}_2$
- $\boldsymbol{\tau}_{gg} = 3\omega_{orb}^2 \mathbf{C}_3^{\otimes} \mathbf{J} \mathbf{C}_3$
- $\omega_{orb} = 0.6511 \frac{\pi}{180}$ rad/s
- $\mathbf{C} = \mathbf{I} + \frac{2}{1+\mathbf{r}^T\mathbf{r}}(\mathbf{r}^{\otimes}\mathbf{r}^{\otimes} - \mathbf{r}^{\otimes})$

## Expected Results

Reference objective value: $J^* = 3.586751 \times 10^{-6}$ (CGPOPS solution)

The reference objective is compared with CGPOPS and GPOPS-II solutions from literature using identical initial guess conditions (constant state values equal to initial conditions, zero control inputs).

## Running This Example

```bash
cd examples/space_station_attitude_control
python space_station_attitude_control.py
```
