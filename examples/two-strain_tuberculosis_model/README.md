# Two-Strain Tuberculosis Model

## Mathematical Formulation

### Problem Statement

Find the optimal treatment controls $u_1(t)$ and $u_2(t)$ that minimize the total infected population and control costs:

$$J = \int_0^{t_f} \left[ L_2 + I_2 + \frac{1}{2}B_1 u_1^2 + \frac{1}{2}B_2 u_2^2 \right] dt$$

Subject to the epidemiological dynamics:

$$\frac{dS}{dt} = \Lambda - \beta_1 S \frac{I_1}{N} - \beta^* S \frac{I_2}{N} - \mu S$$

$$\frac{dT}{dt} = u_1 r_1 L_1 - \mu T + (1-(1-u_2)(p+q)) r_2 I_1 - \beta_2 T \frac{I_1}{N} - \beta^* T \frac{I_2}{N}$$

$$\frac{dL_1}{dt} = \beta_1 S \frac{I_1}{N} - (\mu + k_1) L_1 - u_1 r_1 L_1 + (1-u_2) p r_2 I_1 + \beta_2 T \frac{I_1}{N} - \beta^* L_1 \frac{I_2}{N}$$

$$\frac{dL_2}{dt} = (1-u_2) q r_2 I_1 - (\mu + k_2) L_2 + \beta^* (S + L_1 + T) \frac{I_2}{N}$$

$$\frac{dI_1}{dt} = k_1 L_1 - (\mu + d_1) I_1 - r_2 I_1$$

$$\frac{dI_2}{dt} = k_2 L_2 - (\mu + d_2) I_2$$

### Boundary Conditions

- **Initial conditions**: $S(0) = 19000$, $T(0) = 250$, $L_1(0) = 9000$, $I_1(0) = 1000$, $L_2(0) = 500$, $I_2(0) = 250$
- **Final conditions**: All states free
- **Control bounds**: $0.05 \leq u_1, u_2 \leq 0.95$

### Physical Parameters

- Population recruitment rate: $\Lambda = 429$ per year
- Transmission rates: $\beta_1 = 13$, $\beta_2 = 13$, $\beta^* = 0.029$ per year
- Natural death rate: $\mu = 0.0143$ per year
- Disease-induced death rates: $d_1 = 0$, $d_2 = 0$ per year
- Progression rates: $k_1 = 0.5$, $k_2 = 1$ per year
- Recovery rates: $r_1 = 2$, $r_2 = 1$ per year
- Treatment parameters: $p = 0.4$, $q = 0.1$
- Total population: $N = 30000$
- Control costs: $B_1 = 50$, $B_2 = 500$

### State Variables

- $S(t)$: Susceptible population
- $T(t)$: Population under treatment
- $L_1(t)$: Latent population with drug-sensitive TB
- $I_1(t)$: Infectious population with drug-sensitive TB
- $L_2(t)$: Latent population with drug-resistant TB
- $I_2(t)$: Infectious population with drug-resistant TB

### Control Variables

- $u_1(t)$: Treatment rate for latent drug-sensitive TB (fraction)
- $u_2(t)$: Treatment rate for infectious drug-sensitive TB (fraction)

### Notes

This model represents the optimal control of a two-strain tuberculosis epidemic with drug-sensitive and drug-resistant strains, where treatment decisions must balance infection reduction with control costs.

## Running This Example

```bash
cd examples/two-strain_tuberculosis_model
python two-strain_tuberculosis_model.py
```
