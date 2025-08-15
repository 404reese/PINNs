# Physics-Informed Neural Networks (PINNs) for Projectile Motion with Drag

This repository implements Physics-Informed Neural Networks to solve projectile motion with quadratic air resistance and learn the drag coefficient from noisy observational data.

## Problem Overview

We model a projectile moving under the influence of gravity and quadratic air resistance. The neural network learns both the trajectory and the unknown drag coefficient simultaneously by enforcing physical laws as constraints.

## Mathematical Formulation

### 1. Equations of Motion

The projectile motion with quadratic drag is governed by:

```
m * d²x/dt² = -k * (dx/dt) * √((dx/dt)² + (dy/dt)²)
m * d²y/dt² = -mg - k * (dy/dt) * √((dx/dt)² + (dy/dt)²)
```

Where:
- `m` = mass of the projectile (kg)
- `g` = gravitational acceleration (9.81 m/s²)
- `k` = drag coefficient (to be learned)
- `x(t), y(t)` = position coordinates
- `dx/dt, dy/dt` = velocity components
- `d²x/dt², d²y/dt²` = acceleration components

### 2. Drag Force Components

The drag force acts opposite to velocity direction with magnitude proportional to speed squared:

**Horizontal drag force:**
```
F_drag_x = -k * vx * |v|
```

**Vertical drag force:**
```
F_drag_y = -k * vy * |v|
```

**Velocity magnitude:**
```
|v| = √(vx² + vy²)
```

### 3. Initial Conditions

**Position at t=0:**
```
x(0) = x₀
y(0) = y₀
```

**Velocity at t=0:**
```
vx(0) = v₀ * cos(θ)
vy(0) = v₀ * sin(θ)
```

Where:
- `v₀` = initial speed
- `θ` = launch angle

## PINN Architecture

### Neural Network Structure
```
Input: t (time)
Hidden layers: 3 layers × 50 neurons each with tanh activation
Output: [x(t), y(t)] (position coordinates)
Learnable parameter: k (drag coefficient)
```

### 4. Physics Residuals

The PINN enforces the differential equations by minimizing residuals:

**Horizontal physics residual:**
```
R_x = m * d²x/dt² + k * (dx/dt) * √((dx/dt)² + (dy/dt)²)
```

**Vertical physics residual:**
```
R_y = m * d²y/dt² + mg + k * (dy/dt) * √((dx/dt)² + (dy/dt)²)
```

### 5. Loss Function Components

#### Basic Loss Function
```
L_total = L_physics + L_ic + L_data
```

#### Weighted Loss Function
```
L_total = λ_physics * L_physics + λ_ic * L_ic + λ_data * L_data + λ_reg * L_reg
```

**Physics Loss:**
```
L_physics = (1/N) * Σ(R_x² + R_y²)
```

**Initial Condition Loss:**
```
L_ic = (x(0) - x₀)² + (y(0) - y₀)² + (vx(0) - vx₀)² + (vy(0) - vy₀)²
```

**Data Loss:**
```
L_data = (1/M) * Σ[(x_pred - x_obs)² + (y_pred - y_obs)²]
```

**Regularization Loss:**
```
L_reg = (k - k_expected)²
```

Where:
- `N` = number of collocation points
- `M` = number of data points
- `λ` = weighting coefficients

### 6. Automatic Differentiation

PINN uses automatic differentiation to compute derivatives:

**First derivatives (velocity):**
```
vx = ∂x/∂t
vy = ∂y/∂t
```

**Second derivatives (acceleration):**
```
ax = ∂²x/∂t² = ∂vx/∂t
ay = ∂²y/∂t² = ∂vy/∂t
```

## Training Process

### 7. Optimization

**Adam optimizer with learning rate scheduling:**
```
lr(step) = lr_initial * decay_rate^(step/decay_steps)
```

**Gradient descent update:**
```
θ_{t+1} = θ_t - α * ∇L(θ_t)
```

Where `θ` includes both neural network weights and the drag coefficient `k`.

### 8. Collocation Points

Random sampling in time domain:
```
t_collocation ~ Uniform(0, t_max)
```

## Performance Metrics

### 9. Error Measures

**Root Mean Square Error (RMSE):**
```
RMSE = √[(1/N) * Σ(||pred_i - true_i||²)]
```

**Parameter Estimation Error:**
```
k_error = |k_learned - k_true|
```

**Position Error:**
```
position_error = √[(x_pred - x_true)² + (y_pred - y_true)²]
```

## Key Features

- **Parameter Discovery**: Learns unknown drag coefficient from data
- **Physics Constraints**: Enforces differential equations during training
- **Noise Robustness**: Handles noisy observational data
- **Gradient-based**: Uses automatic differentiation for exact derivatives
- **Multi-objective**: Balances physics compliance and data fitting

## Implementation Variants

1. **Fixed Drag Coefficient**: k is predetermined
2. **Learnable Drag Coefficient**: k is a trainable parameter
3. **Weighted Loss**: Different loss components with adaptive weights
4. **Enhanced Training**: Learning rate scheduling and regularization

## Results Summary

The implementation demonstrates:
- Accurate trajectory prediction from noisy data
- Successful parameter estimation (drag coefficient)
- Physics-compliant solutions
- Robustness to measurement noise

<!-- Insert trajectory comparison image here -->

<!-- Insert loss evolution image here -->

<!-- Insert parameter learning curve here -->

<!-- Insert error analysis plots here -->

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
- Classical mechanics: Projectile motion with air resistance
- Automatic differentiation in TensorFlow

## Usage

Run the Jupyter notebook `PINNS1.ipynb` to:
1. Generate synthetic noisy data
2. Train PINN variants
3. Compare results with analytical solutions
4. Visualize trajectories and learning progress

## Dependencies

```
tensorflow
numpy
matplotlib
scipy
```
