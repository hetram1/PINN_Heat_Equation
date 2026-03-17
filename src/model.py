"""
model.py
Defines the Physics-Informed Neural Network (PINN) architecture.
"""

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        # Input layer: 2 inputs (x and t) -> 32 hidden neurons
        self.hidden1 = nn.Linear(2, 32)
        
        # Hidden layers: 32 -> 32
        self.hidden2 = nn.Linear(32, 32)
        self.hidden3 = nn.Linear(32, 32)
        self.hidden4 = nn.Linear(32, 32)
        
        # Output layer: 32 -> 1 output (Temperature 'u')
        self.output = nn.Linear(32, 1)
        
        # Activation function: Tanh (smooth, continuously differentiable)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.activation(self.hidden4(x))
        u = self.output(x)
        return u

# --- Quick Test Block ---
# This code only runs if you execute this file directly. 
# It won't run when we import this model into our main file later.
if __name__ == "__main__":
    print("Testing the PINN architecture...")
    model = PINN()
    
    # Create a dummy input (e.g., 5 random points with x and t)
    dummy_input = torch.rand((5, 2)) 
    dummy_output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print("Model module is ready to go!")