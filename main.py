"""
main.py
The control center for the Physics-Informed Neural Network.
Now includes FDM benchmarking and advanced visualization.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

from src.model import PINN
from src.physics import generate_training_data, physics_loss, data_loss, alpha, x_min, x_max, t_max
from src.solver import solve_fdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN().to(device)
    
    X_physics, X_bc_ic, U_bc_ic = generate_training_data()
    X_physics, X_bc_ic, U_bc_ic = X_physics.to(device), X_bc_ic.to(device), U_bc_ic.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 3000
    
    # Upgrade: Track loss for plotting
    loss_history = []
    
    print("Starting PINN training loop...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_d = data_loss(model, X_bc_ic, U_bc_ic)
        loss_p = physics_loss(model, X_physics)
        loss_total = loss_d + loss_p
        
        loss_total.backward()
        optimizer.step()
        
        loss_history.append(loss_total.item())
        
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Total Loss: {loss_total.item():.6f}")

    print(f"PINN Training finished in {time.time() - start_time:.2f} seconds!\n")
    return model, device, loss_history

def evaluate_and_plot(model, device, loss_history):
    print("Running numerical FDM solver for comparison...")
    # Run FDM
    nx, nt = 100, 2500
    x_fdm, t_fdm, u_fdm = solve_fdm(alpha, x_min, x_max, t_max, nx, nt)
    
    print("Generating PINN predictions on the exact same FDM grid...")
    # Create the exact same grid for the PINN
    X_plot, T_plot = np.meshgrid(x_fdm, t_fdm)
    x_flat = X_plot.flatten()[:, None]
    t_flat = T_plot.flatten()[:, None]
    X_pred = np.hstack((x_flat, t_flat))
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pinn_flat = model(X_pred_tensor).cpu().numpy()
        
    u_pinn = u_pinn_flat.reshape(X_plot.shape)
    
    # Calculate Absolute Error
    error = np.abs(u_fdm - u_pinn)
    
    print("Generating professional visualization suite...")
    # Plot 1: Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.yscale('log')
    plt.title('PINN Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: The 3-Panel Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # FDM Plot
    c1 = axes[0].contourf(T_plot, X_plot, u_fdm, levels=100, cmap='coolwarm')
    axes[0].set_title('Numerical Baseline (FDM)')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Position (x)')
    fig.colorbar(c1, ax=axes[0])
    
    # PINN Plot
    c2 = axes[1].contourf(T_plot, X_plot, u_pinn, levels=100, cmap='coolwarm')
    axes[1].set_title('Neural Network Prediction (PINN)')
    axes[1].set_xlabel('Time (t)')
    fig.colorbar(c2, ax=axes[1])
    
    # Error Plot
    c3 = axes[2].contourf(T_plot, X_plot, error, levels=100, cmap='inferno')
    axes[2].set_title('Absolute Error |FDM - PINN|')
    axes[2].set_xlabel('Time (t)')
    fig.colorbar(c3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, compute_device, history = train()
    evaluate_and_plot(trained_model, compute_device, history)