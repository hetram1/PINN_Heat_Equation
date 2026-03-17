"""
main.py
The control center for the Physics-Informed Neural Network.
Now includes 3D Visualization and PINN Loss Component Breakdown.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.animation as animation

from src.model import PINN
from src.physics import generate_training_data, physics_loss, data_loss, alpha, x_min, x_max, t_min, t_max

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN().to(device)
    
    X_physics, X_bc_ic, U_bc_ic = generate_training_data()
    X_physics, X_bc_ic, U_bc_ic = X_physics.to(device), X_bc_ic.to(device), U_bc_ic.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 3000
    
    # --- UPGRADE: Track all loss components separately ---
    loss_history_total = []
    loss_history_p = []
    loss_history_d = []
    
    print("Starting PINN training loop...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_d = data_loss(model, X_bc_ic, U_bc_ic)
        loss_p = physics_loss(model, X_physics)
        loss_total = loss_d + loss_p
        
        loss_total.backward()
        optimizer.step()
        
        # Save components
        loss_history_total.append(loss_total.item())
        loss_history_p.append(loss_p.item())
        loss_history_d.append(loss_d.item())
        
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Total: {loss_total.item():.6f} | Physics: {loss_p.item():.6f} | Data: {loss_d.item():.6f}")

    print(f"PINN Training finished in {time.time() - start_time:.2f} seconds!\n")
    return model, device, loss_history_total, loss_history_p, loss_history_d

def evaluate_and_plot(model, device, loss_total, loss_p, loss_d):
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
    
    x_t = torch.tensor(x_flat, dtype=torch.float32, requires_grad=True).to(device)
    t_t = torch.tensor(t_flat, dtype=torch.float32, requires_grad=True).to(device)
    X_pred_tensor = torch.cat([x_t, t_t], dim=1)
    
    u_pinn_tensor = model(X_pred_tensor)
    
    u_t = torch.autograd.grad(u_pinn_tensor, t_t, grad_outputs=torch.ones_like(u_pinn_tensor), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pinn_tensor, x_t, grad_outputs=torch.ones_like(u_pinn_tensor), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = (u_t - alpha * u_xx).detach().cpu().numpy().reshape(X_plot.shape)
    u_pinn = u_pinn_tensor.detach().cpu().numpy().reshape(X_plot.shape)
    
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
    
    # --- FIGURE 2: Engineering Diagnostics (UPGRADED) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plotting all three loss components
    axes2[0].plot(loss_total, color='black', linewidth=2, label='Total Loss')
    axes2[0].plot(loss_p, color='red', linestyle='--', alpha=0.8, label='Physics Loss (PDE)')
    axes2[0].plot(loss_d, color='blue', linestyle='-.', alpha=0.8, label='Data Loss (BC/IC)')
    
    axes2[0].set_yscale('log')
    axes2[0].set_title('Training Convergence Component Breakdown')
    axes2[0].set_xlabel('Epochs')
    axes2[0].set_ylabel('Loss (Log Scale)')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    c_res = axes2[1].contourf(T_plot, X_plot, np.abs(residual), levels=100, cmap='viridis')
    axes2[1].set_title('PDE Residual |f(x,t)| (Closer to 0 is better)')
    axes2[1].set_xlabel('Time (t)')
    axes2[1].set_ylabel('Position (x)')
    fig2.colorbar(c_res, ax=axes2[1])
    plt.tight_layout()

    # --- FIGURE 3: 3D Surface Plot ---
    fig3 = plt.figure(figsize=(10, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(T_plot, X_plot, u_pinn, cmap='coolwarm', linewidth=0, antialiased=True)
    ax3.set_title('3D Surface: PINN Temperature Evolution')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Position (x)')
    ax3.set_zlabel('Temperature (u)')
    fig3.colorbar(surf, shrink=0.5, aspect=10, label='Temperature')
    plt.tight_layout()

    plt.show()


def generate_gif(model, device):
    print("Generating Heat Diffusion GIF Animation...")
    
    # Create a grid for the animation (200 spatial points, 60 time frames)
    x_plot = np.linspace(x_min, x_max, 200)
    t_frames = np.linspace(t_min, t_max, 60)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], color='red', lw=2.5, label='PINN Prediction')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Temperature (u)')
    ax.set_title('PINN: 1D Heat Diffusion Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        t_val = t_frames[i]
        
        # Create input tensor for this specific frame's time step
        t_tensor = np.full((len(x_plot), 1), t_val)
        x_tensor = x_plot.reshape(-1, 1)
        X_pred = np.hstack((x_tensor, t_tensor))
        X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(device)

        with torch.no_grad():
            u_pred = model(X_pred_tensor).cpu().numpy()

        line.set_data(x_plot, u_pred)
        time_text.set_text(f'Time: {t_val:.3f} s')
        return line, time_text

    # Generate the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(t_frames), init_func=init, blit=True)

    # Save it as a GIF using Pillow (built into matplotlib)
    gif_path = 'plots/heat_diffusion.gif'
    ani.save(gif_path, writer='pillow', fps=15)
    print(f"Animation successfully saved to {gif_path}!")
    plt.close()

if __name__ == "__main__":
    trained_model, compute_device, l_tot, l_p, l_d = train()
    evaluate_and_plot(trained_model, compute_device, l_tot, l_p, l_d)
    
    # Add this final line!
    generate_gif(trained_model, compute_device)