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
from src.solver import solve_fdm, solve_analytical

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
    from src.solver import solve_fdm, solve_analytical
    print("Running numerical FDM solver for comparison...")
    nx, nt = 100, 2500
    x_fdm, t_fdm, u_fdm = solve_fdm(alpha, x_min, x_max, t_max, nx, nt)
    
    print("Calculating Exact Analytical Solution...")
    _, _, u_exact = solve_analytical(alpha, x_min, x_max, t_max, nx, nt)
    
    print("Generating PINN predictions and calculating PDE Residuals...")
    X_plot, T_plot = np.meshgrid(x_fdm, t_fdm)
    x_flat = X_plot.flatten()[:, None]
    t_flat = T_plot.flatten()[:, None]
    
    # --- THE FIX IS HERE ---
    # We must define x and t as separate tensors BEFORE passing them to the model
    x_t = torch.tensor(x_flat, dtype=torch.float32, requires_grad=True).to(device)
    t_t = torch.tensor(t_flat, dtype=torch.float32, requires_grad=True).to(device)
    
    # Combine them for the model input
    X_pred_tensor = torch.cat([x_t, t_t], dim=1)
    
    # 1. Get PINN Predictions
    u_pinn_tensor = model(X_pred_tensor)
    
    # 2. Calculate the PDE Residual across the entire grid
    u_t = torch.autograd.grad(u_pinn_tensor, t_t, grad_outputs=torch.ones_like(u_pinn_tensor), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pinn_tensor, x_t, grad_outputs=torch.ones_like(u_pinn_tensor), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Calculate residual and convert everything back to numpy for plotting
    residual = (u_t - alpha * u_xx).detach().cpu().numpy().reshape(X_plot.shape)
    u_pinn = u_pinn_tensor.detach().cpu().numpy().reshape(X_plot.shape)
    # -----------------------
    
    l2_error = np.linalg.norm(u_exact - u_pinn) / np.linalg.norm(u_exact)
    print(f"\n---> L2 Relative Error (PINN vs Exact): {l2_error:.6f} <---")
    
    print("Generating Visualizations...")
    
    # --- FIGURE 1: The 4-Panel Comparison ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    
    c1 = axes1[0, 0].contourf(T_plot, X_plot, u_fdm, levels=100, cmap='coolwarm')
    axes1[0, 0].set_title('Numerical Baseline (FDM)')
    axes1[0, 0].set_ylabel('Position (x)')
    fig1.colorbar(c1, ax=axes1[0, 0])
    
    c2 = axes1[0, 1].contourf(T_plot, X_plot, u_exact, levels=100, cmap='coolwarm')
    axes1[0, 1].set_title('Exact Analytical Solution')
    fig1.colorbar(c2, ax=axes1[0, 1])
    
    c3 = axes1[1, 0].contourf(T_plot, X_plot, u_pinn, levels=100, cmap='coolwarm')
    axes1[1, 0].set_title('Neural Network (PINN)')
    axes1[1, 0].set_xlabel('Time (t)')
    axes1[1, 0].set_ylabel('Position (x)')
    fig1.colorbar(c3, ax=axes1[1, 0])
    
    error = np.abs(u_exact - u_pinn)
    c4 = axes1[1, 1].contourf(T_plot, X_plot, error, levels=100, cmap='inferno')
    axes1[1, 1].set_title(f'Absolute Error |Exact - PINN|\nL2 Error: {l2_error:.5f}')
    axes1[1, 1].set_xlabel('Time (t)')
    fig1.colorbar(c4, ax=axes1[1, 1])
    
    plt.tight_layout()
    
    # --- FIGURE 2: Engineering Diagnostics ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    axes2[0].plot(loss_history, color='blue', linewidth=2)
    axes2[0].set_yscale('log')
    axes2[0].set_title('Training Convergence (Total Loss)')
    axes2[0].set_xlabel('Epochs')
    axes2[0].set_ylabel('Loss (Log Scale)')
    axes2[0].grid(True, alpha=0.3)
    
    c_res = axes2[1].contourf(T_plot, X_plot, np.abs(residual), levels=100, cmap='viridis')
    axes2[1].set_title('PDE Residual |f(x,t)| (Closer to 0 is better)')
    axes2[1].set_xlabel('Time (t)')
    axes2[1].set_ylabel('Position (x)')
    fig2.colorbar(c_res, ax=axes2[1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, compute_device, history = train()
    evaluate_and_plot(trained_model, compute_device, history)