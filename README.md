# Physics-Informed Neural Network (PINN) vs. Finite Difference Method

## 📌 Problem Statement

Simulating transport phenomena and continuous physical systems traditionally requires discrete numerical grids. This project solves the 1D Heat Diffusion Equation by implementing a Physics-Informed Neural Network (PINN) in PyTorch, replacing the discrete mesh with a continuous, differentiable neural network approximation.

To validate the accuracy of the machine learning model, the PINN's predictions are directly benchmarked against a traditional Explicit Finite Difference Method (FDM) solver.

## 🧮 The Governing Equation

The transport of heat is governed by the parabolic Partial Differential Equation (PDE):
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

**Physical Constraints:**

- **Spatial Domain:** $x \in [-1, 1]$
- **Temporal Domain:** $t \in [0, 1]$
- **Thermal Diffusivity:** $\alpha = 0.01$
- **Initial Condition (IC):** $u(x, 0) = -\sin(\pi x)$
- **Boundary Conditions (BC):** $u(-1, t) = 0$ and $u(1, t) = 0$

## ⚙️ Approach & Architecture

1. **Neural Network Approximation:** A 4-layer Multi-Layer Perceptron (MLP) with 32 neurons per hidden layer maps coordinates $(x, t)$ to temperature $u$. A $\tanh$ activation function is used to ensure the network is twice-differentiable.
2. **Physics Loss (Autograd):** Instead of calculating loss against a labeled dataset, PyTorch's Automatic Differentiation is used to compute exact spatial $\left(\frac{\partial^2 u}{\partial x^2}\right)$ and temporal $\left(\frac{\partial u}{\partial t}\right)$ derivatives at 10,000 randomly sampled collocation points.
3. **Numerical Baseline:** An explicit FDM solver is implemented from scratch to provide a ground-truth temperature surface for error analysis.

## 📊 Results & Error Analysis

After training for 3,000 epochs using the Adam optimizer, the PINN successfully converged, driving the PDE residual near zero.

- **Convergence:** The network achieved a stable, logarithmic decay in total loss.
- **Accuracy:** The absolute spatiotemporal error between the FDM numerical baseline and the continuous PINN prediction is $< 0.01$ across the entire domain.

_(Note: Add your 3-panel plot and your loss curve image here!)_
`![FDM vs PINN Comparison](plots/comparison.png)`
`![Training Loss](plots/loss.png)`

## 📁 Repository Structure

Designed for modularity and high-performance scientific computing:

```text
pinn_heat_equation/
│
├── main.py                 # Execution script, FDM comparison, and visualization
├── README.md               # Project documentation
│
└── src/
    ├── __init__.py
    ├── model.py            # PyTorch PINN architecture
    ├── physics.py          # Autograd PDE residuals and BC/IC generation
    └── solver.py           # Traditional FDM numerical solver baseline

## 🚀 Tech Stack

- **Language:** `Python`
- **Deep Learning:** `PyTorch` (Autograd, Optimizers)
- **Scientific Computing:** `NumPy`, `Matplotlib`
- **Core Concepts:** Partial Differential Equations, Transport Phenomena, Scientific Machine Learning (SciML)

## Author

- **Name:** Het Ram
```
