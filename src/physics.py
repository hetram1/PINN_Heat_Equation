"""
physics.py
Handles data generation (domain, boundaries, initial conditions)
and the physical loss functions (Heat Equation PDE).
"""

import torch
import numpy as np

# --- Physical Parameters ---
alpha = 0.01
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

def generate_training_data(n_physics=10000, n_ic=500, n_bc=500):
    """
    Generates PyTorch tensors for the physics collocation points and BC/IC points.
    """
    # 1. Physics Collocation Points (Inside the rod)
    x_physics = np.random.uniform(x_min, x_max, (n_physics, 1))
    t_physics = np.random.uniform(t_min, t_max, (n_physics, 1))
    X_physics = np.hstack((x_physics, t_physics))
    
    # 2. Initial Condition Points (t = 0)
    x_ic = np.random.uniform(x_min, x_max, (n_ic, 1))
    t_ic = np.zeros((n_ic, 1))
    u_ic = -np.sin(np.pi * x_ic)
    
    # 3. Boundary Condition Points (x = -1 and x = 1)
    x_bc_left = np.full((n_bc, 1), x_min)
    t_bc_left = np.random.uniform(t_min, t_max, (n_bc, 1))
    u_bc_left = np.zeros((n_bc, 1))
    
    x_bc_right = np.full((n_bc, 1), x_max)
    t_bc_right = np.random.uniform(t_min, t_max, (n_bc, 1))
    u_bc_right = np.zeros((n_bc, 1))
    
    # Combine BC and IC data
    X_bc_ic = np.vstack([np.hstack([x_ic, t_ic]), 
                         np.hstack([x_bc_left, t_bc_left]), 
                         np.hstack([x_bc_right, t_bc_right])])
    U_bc_ic = np.vstack([u_ic, u_bc_left, u_bc_right])
    
    # Convert to PyTorch Tensors
    # Note: 'requires_grad=True' is only needed for the physics points
    X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32, requires_grad=True)
    X_bc_ic_tensor = torch.tensor(X_bc_ic, dtype=torch.float32)
    U_bc_ic_tensor = torch.tensor(U_bc_ic, dtype=torch.float32)
    
    return X_physics_tensor, X_bc_ic_tensor, U_bc_ic_tensor

def physics_loss(model, x_t_points):
    """
    Calculates the mean squared error of the Heat Equation residual.
    """
    x = x_t_points[:, 0:1]
    t = x_t_points[:, 1:2]
    
    x_t_combined = torch.cat([x, t], dim=1)
    u = model(x_t_combined)
    
    # Calculate derivatives using Autograd
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # PDE Residual
    f = u_t - alpha * u_xx
    
    return torch.mean(f ** 2)

def data_loss(model, X_train, U_train):
    """
    Calculates the MSE between the model predictions and the true BC/IC values.
    """
    predictions = model(X_train)
    return torch.mean((predictions - U_train) ** 2)