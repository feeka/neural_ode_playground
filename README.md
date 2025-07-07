# Neural ODE: Quadratic Dynamics Learning

A PyTorch implementation demonstrating how Neural Ordinary Differential Equations (Neural ODEs) can learn complex nonlinear dynamics. This project specifically focuses on learning quadratic state-dependent dynamics of the form `dx/dt = ax² + bx + c`.

## Overview

Neural ODEs combine neural networks with differential equation solvers to model continuous-time dynamics. Unlike traditional discrete neural networks, Neural ODEs can model smooth, continuous trajectories and are particularly useful for:

- Time series with irregular sampling
- Physics-informed modeling
- Continuous normalizing flows
- Systems where the rate of change depends on the current state

## Mathematical Foundation

This implementation learns a 3D dynamical system where each component follows:

```
dx₁/dt = 0.1x₁² - 0.5x₁ + 0.2
dx₂/dt = -0.2x₂² + 0.3x₂ - 0.1  
dx₃/dt = 0.05x₃² + 0.1x₃ + 0.05
```

The neural network learns to approximate the function `f(y) = dy/dt` given only trajectory data.

## Features

- **State-dependent quadratic dynamics**: Learn complex nonlinear relationships
- **Extrapolation testing**: Evaluate model performance beyond training time horizon
- **Derivative accuracy validation**: Direct comparison of learned vs. true dynamics
- **Comprehensive visualization**: Training loss and trajectory plots
- **Reproducible results**: Fixed random seeds for consistent outputs

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Additional dependencies listed below

### Install Dependencies

```bash
pip install torch torchdiffeq matplotlib numpy
```

### Alternative: Using requirements.txt

Install using:

```bash
pip3 install -r requirements.txt
```
or if it does not work
```
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete Neural ODE training and testing pipeline:

```bash
python3 neural_ode.py
```

### What the Script Does

1. **Data Generation**: Creates synthetic trajectory data by solving the true quadratic dynamics
2. **Training**: Trains a neural network to learn the dynamics function
3. **Testing**: Evaluates the model on:
   - Training data (interpolation)
   - Extended time horizon (extrapolation)
   - Direct derivative accuracy at specific points
4. **Visualization**: Generates plots showing training progress and trajectory comparisons

### Expected Output

```
Neural ODE Quadratic Function Test
==================================================
Learning dynamics: dx/dt = ax² + bx + c

Training Neural ODE to learn quadratic dynamics...
True system: dx/dt = ax^2 + bx + c for each component
Epoch 0, Loss: 0.123456
Epoch 100, Loss: 0.001234
Epoch 200, Loss: 0.000123

==================================================
TESTING NEURAL ODE
==================================================
Training MSE: 0.000045
Test MSE (same time range): 0.000045
Extrapolation MSE (t > 2): 0.001234

Derivative Accuracy Test:
Point                True dx/dt              Predicted dx/dt         Error     
--------------------------------------------------------------------------------
Point 1:           [ 1.   0.5 -0.5]      [ 0.25  0.08 -0.1 ]      [ 0.251  0.079 -0.098]      0.0234
...

Test completed! Results saved to 'neural_ode_quadratic_test.png'
```

## Code Structure

### Core Components

- **`ODEFunc`**: Neural network class defining the dynamics function
- **`generate_quadratic_data()`**: Creates synthetic training data
- **`train_neural_ode()`**: Training loop with Adam optimizer
- **`test_neural_ode()`**: Comprehensive testing and validation
- **`plot_results()`**: Visualization of results

### Key Parameters

- **Network architecture**: 3 → 50 → 3 with ReLU activation
- **Training epochs**: 250
- **Learning rate**: 0.01
- **Time horizon**: Training on [0, 2], testing on [0, 3]
- **Initial conditions**: `[1.0, 0.5, -0.5]`

## Customization

### Modify the Dynamics

Change the quadratic coefficients in `generate_quadratic_data()`:

```python
def true_dynamics(t, y):
    x1, x2, x3 = y[0], y[1], y[2]
    # Modify these coefficients
    dx1_dt = 0.2 * x1**2 - 0.3 * x1 + 0.1  # New coefficients
    dx2_dt = -0.1 * x2**2 + 0.4 * x2 - 0.2
    dx3_dt = 0.03 * x3**2 + 0.2 * x3 + 0.03
    return torch.stack([dx1_dt, dx2_dt, dx3_dt])
```

### Adjust Network Architecture

Modify the `ODEFunc` class:

```python
self.net = nn.Sequential(
    nn.Linear(3, 100),    # Increase hidden size
    nn.ReLU(),
    nn.Linear(100, 50),   # Add another layer
    nn.ReLU(),
    nn.Linear(50, 3)
)
```

### Change Training Parameters

In `train_neural_ode()`:

```python
optimizer = torch.optim.Adam(func.parameters(), lr=0.001)  # Lower learning rate
for epoch in range(500):  # More epochs
```

## Results Interpretation

### Training Loss Plot
Shows convergence of the neural network during training. Should decrease steadily and plateau.

### Trajectory Plots
- **Circles**: Original training data points
- **Dashed lines**: True trajectories (ground truth)
- **Solid lines**: Neural ODE predictions
- **Vertical line**: Boundary between training and extrapolation regions

### Performance Metrics
- **Training MSE**: Error on training data
- **Test MSE**: Error on same time range as training
- **Extrapolation MSE**: Error when predicting beyond training time
- **Derivative Error**: Direct comparison of learned dynamics function

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `torchdiffeq` is installed
   ```bash
   pip install torchdiffeq
   ```

2. **Poor Convergence**: Try:
   - Lower learning rate
   - More training epochs
   - Different network architecture
   - Different initial conditions

3. **Numerical Instability**: The quadratic dynamics can lead to exponential growth. Consider:
   - Shorter time horizons
   - Different coefficient values
   - Gradient clipping

### Performance Tips

- Use GPU if available by adding `.cuda()` to tensors
- Experiment with different ODE solvers in `torchdiffeq`
- Try different activation functions (Tanh, Swish, etc.)

## Extensions

### Possible Improvements

1. **Multiple Initial Conditions**: Train on trajectories from different starting points
2. **Parameter Variation**: Test generalization across different quadratic coefficients
3. **Regularization**: Add L2 regularization or dropout
4. **Adaptive Time Steps**: Use adaptive ODE solvers
5. **Uncertainty Quantification**: Add Bayesian layers for uncertainty estimation

### Research Directions

- Compare with other continuous-time models
- Investigate different functional forms (cubic, exponential, etc.)
- Apply to real-world time series data
- Explore physics-informed loss functions

## References

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. NeurIPS.
- [torchdiffeq Documentation](https://github.com/rtqichen/torchdiffeq)
- [Neural ODE Tutorial](https://github.com/msurtsukov/neural-ode-tutorial)

## License

MIT License - feel free to use and modify for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
