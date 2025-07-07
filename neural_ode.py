import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )

    def forward(self, t, y):
        return self.net(y)

# Proof of concept: Learn dynamics of a quadratic system
# We'll model a 3D system where each component follows: dx/dt = ax^2 + bx + c

def generate_quadratic_data():
    """
    Generate synthetic data for a 3D quadratic system:
    dx1/dt = 0.1*x1^2 - 0.5*x1 + 0.2
    dx2/dt = -0.2*x2^2 + 0.3*x2 - 0.1  
    dx3/dt = 0.05*x3^2 + 0.1*x3 + 0.05
    """
    def true_dynamics(t, y):
        x1, x2, x3 = y[0], y[1], y[2]
        dx1_dt = 0.1 * x1**2 - 0.5 * x1 + 0.2
        dx2_dt = -0.2 * x2**2 + 0.3 * x2 - 0.1
        dx3_dt = 0.05 * x3**2 + 0.1 * x3 + 0.05
        return torch.stack([dx1_dt, dx2_dt, dx3_dt])
    
    # Time points
    t = torch.linspace(0, 2, 50)
    
    # Initial conditions
    y0 = torch.tensor([1.0, 0.5, -0.5])
    
    # Generate true trajectory
    true_y = odeint(true_dynamics, y0, t)
    
    return t, true_y, true_dynamics

def train_neural_ode():
    """Train the Neural ODE to learn the quadratic dynamics"""
    
    # Generate training data
    t_train, y_train, true_dynamics = generate_quadratic_data()
    
    # Initialize Neural ODE
    func = ODEFunc()
    optimizer = torch.optim.Adam(func.parameters(), lr=0.01)
    
    print("Training Neural ODE to learn quadratic dynamics...")
    print("True system: dx/dt = ax^2 + bx + c for each component")
    
    losses = []
    for epoch in range(250):
        optimizer.zero_grad()
        
        # Predict trajectory using Neural ODE
        y_pred = odeint(func, y_train[0], t_train)
        
        # Compute loss
        loss = torch.mean((y_pred - y_train)**2)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return func, t_train, y_train, true_dynamics, losses

def test_neural_ode(func, t_train, y_train, true_dynamics):
    """Test the trained Neural ODE"""
    
    print("\n" + "="*50)
    print("TESTING NEURAL ODE")
    print("="*50)
    
    # Test on training data
    with torch.no_grad():
        y_pred = odeint(func, y_train[0], t_train)
        train_error = torch.mean((y_pred - y_train)**2).item()
        print(f"Training MSE: {train_error:.6f}")
    
    # Test on longer time horizon
    t_test = torch.linspace(0, 3, 75)  # Longer than training
    
    with torch.no_grad():
        # Neural ODE prediction
        y_pred_long = odeint(func, y_train[0], t_test)
        
        # True dynamics for comparison
        y_true_long = odeint(true_dynamics, y_train[0], t_test)
        
        test_error = torch.mean((y_pred_long[:len(t_train)] - y_train)**2).item()
        extrapolation_error = torch.mean((y_pred_long[len(t_train):] - y_true_long[len(t_train):])**2).item()
        
        print(f"Test MSE (same time range): {test_error:.6f}")
        print(f"Extrapolation MSE (t > 2): {extrapolation_error:.6f}")
    
    # Test derivative accuracy at specific points
    test_points = torch.tensor([[1.0, 0.5, -0.5], [0.0, 1.0, 0.0], [-1.0, -0.5, 1.0]])
    
    print(f"\nDerivative Accuracy Test:")
    print(f"{'Point':<20} {'True dx/dt':<25} {'Predicted dx/dt':<25} {'Error':<10}")
    print("-" * 80)
    
    with torch.no_grad():
        for i, point in enumerate(test_points):
            true_deriv = true_dynamics(0, point)
            pred_deriv = func(0, point)
            error = torch.norm(true_deriv - pred_deriv).item()
            
            print(f"Point {i+1}:           {point.numpy()}      {true_deriv.numpy()}      {pred_deriv.numpy()}      {error:.4f}")
    
    return y_pred_long, y_true_long, t_test

def plot_results(t_train, y_train, t_test, y_pred, y_true, losses):
    """Plot training results and predictions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    # Plot trajectories for each component
    components = ['x₁', 'x₂', 'x₃']
    colors = ['red', 'blue', 'green']
    
    for i in range(3):
        row = (i + 1) // 2
        col = (i + 1) % 2
        
        axes[row, col].plot(t_train.numpy(), y_train[:, i].numpy(), 'o', 
                           color=colors[i], label=f'Training data ({components[i]})', markersize=4)
        axes[row, col].plot(t_test.numpy(), y_true[:, i].numpy(), '--', 
                           color=colors[i], label=f'True trajectory ({components[i]})', linewidth=2)
        axes[row, col].plot(t_test.numpy(), y_pred[:, i].numpy(), '-', 
                           color=colors[i], label=f'Neural ODE ({components[i]})', linewidth=2, alpha=0.8)
        
        axes[row, col].axvline(x=2, color='black', linestyle=':', alpha=0.5, label='Training boundary')
        axes[row, col].set_title(f'Component {components[i]} (Quadratic Dynamics)')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel(f'{components[i]}(t)')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_ode_quadratic_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete test"""
    print("Neural ODE Quadratic Function Test")
    print("="*50)
    print("Learning dynamics: dx/dt = ax² + bx + c")
    print()
    
    # Train the model
    func, t_train, y_train, true_dynamics, losses = train_neural_ode()
    
    # Test the model
    y_pred, y_true, t_test = test_neural_ode(func, t_train, y_train, true_dynamics)
    
    # Plot results
    plot_results(t_train, y_train, t_test, y_pred, y_true, losses)
    
    print(f"\nTest completed! Results saved to 'neural_ode_quadratic_test.png'")
    
    return func, t_train, y_train, y_pred, y_true

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the test
    main()