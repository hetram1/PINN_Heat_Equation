"""
solver.py
Implements the traditional Finite Difference Method (FDM) 
to establish a numerical baseline for the Heat Equation.
"""

import numpy as np

def solve_fdm(alpha, x_min, x_max, t_max, nx=100, nt=2500):
    """
    Solves the 1D Heat Equation using an Explicit Finite Difference scheme.
    """
    dx = (x_max - x_min) / (nx - 1)
    dt = t_max / (nt - 1)
    
    # Stability condition for explicit FDM (Courant-Friedrichs-Lewy condition)
    r = alpha * dt / (dx**2)
    if r > 0.5:
        print(f"Warning: FDM scheme might be unstable. r = {r:.3f} > 0.5")
        
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0, t_max, nt)
    
    # Initialize the grid: rows are time steps, columns are spatial points
    u = np.zeros((nt, nx))
    
    # Apply Initial Condition: u(x, 0) = -sin(pi * x)
    u[0, :] = -np.sin(np.pi * x)
    
    # Apply Boundary Conditions: u(-1, t) = 0 and u(1, t) = 0
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Time-stepping loop
    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
    return x, t, u


def solve_analytical(alpha, x_min, x_max, t_max, nx=100, nt=2500):
    """
    Computes the exact analytical solution for the 1D Heat Equation 
    given the specific IC: u(x,0) = -sin(pi*x) and BCs: u(-1,t)=u(1,t)=0.
    """
    import numpy as np
    
    # Create the exact same grid we use for FDM
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0, t_max, nt)
    X, T = np.meshgrid(x, t)
    
    # The exact mathematical solution
    u_exact = -np.exp(-alpha * (np.pi**2) * T) * np.sin(np.pi * X)
    
    return x, t, u_exact

