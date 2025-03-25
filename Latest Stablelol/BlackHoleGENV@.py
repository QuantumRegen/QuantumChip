import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set up the grid and simulation parameters
N = 500  # Grid size (500x500)
L = 5.0  # Physical size of the grid
dx = L / (N - 1)  # Spatial step
dt = 0.05  # Time step
steps = 4000  # Total steps to run

# Initialize the wavefunction psi (complex field for superfluid)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)  # Radial distance from center

psi = np.ones((N, N), dtype=complex) * 0.1  # Initial uniform wavefunction
density = np.abs(psi)**2  # Initial density (|psi|^2)

# Vortex parameters
vortex_speed = 1.5  # Reduced from 1.8 to encourage inward flow
vortex_flow = np.tanh(R / 0.1)  # Vortex flow with tanh profile
horizon = 0.1  # Horizon radius for vortex sink
sink = -0.05 * (R < horizon)  # Sink term to draw density inward

# Gravity well (centered at x=250, y=250)
center_x, center_y = N//2, N//2  # Center of the grid (250, 250)
X_shifted = X + L/2 - x[center_x]  # Shift X to center at 250
Y_shifted = Y + L/2 - y[center_y]  # Shift Y to center at 250
R_shifted = np.sqrt(X_shifted**2 + Y_shifted**2)  # Radial distance from center

# Output directory
output_dir = "/home/wayne/Desktop/THEORY/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Time evolution loop
for t in range(steps):
    # Update gravity well with tighter radius and stronger pull
    t_scaled = t / 250
    gravity_radius = (0.0002 + t_scaled)**2  # Tightened radius
    V_gravity = -8.5e7 * np.exp(-R_shifted**2 / gravity_radius)  # Centered at (250, 250)

    # Nonlinear term
    nonlinear = 80000 * density  # Nonlinear interaction term

    # Damping term (adjusted threshold)
    damping = -0.05 * density * (density > 0.1)  # Lowered threshold from 0.15 to 0.1

    # Vortex dynamics
    vortex_term = vortex_speed * vortex_flow * (psi / (np.abs(psi) + 1e-10)) + sink * psi

    # Hawking noise (random fluctuations)
    hawking_noise = 0.01 * (np.random.randn(N, N) + 1j * np.random.randn(N, N))

    # Total potential and evolution
    V_total = V_gravity + nonlinear + damping + vortex_term + hawking_noise

    # Simplified time evolution (using a basic split-step method)
    # Kinetic term (Laplacian in Fourier space)
    kx = 2 * np.pi * np.fft.fftfreq(N, dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, dx)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-1j * dt * K2 / 2)  # Kinetic evolution
    psi = np.fft.ifft2(psi_k)

    # Potential term
    psi *= np.exp(-1j * dt * V_total)

    # Update density
    density = np.abs(psi)**2

    # Absorbing boundary conditions (set density to 0 at edges)
    density[0:10, :] = 0
    density[-10:, :] = 0
    density[:, 0:10] = 0
    density[:, -10:] = 0
    psi[density == 0] = 0  # Reset psi where density is zero

    # Compute phase coherence
    phase = np.angle(psi)
    coherence = np.abs(np.mean(np.exp(1j * phase)))

    # Plot every 50 steps
    if t % 50 == 0:
        # Density plot
        plt.figure(figsize=(8, 6))
        plt.imshow(density, cmap="viridis", vmin=0, vmax=0.5)
        plt.colorbar(label="Density")
        plt.title(f"Step {t} - Density")
        plt.savefig(os.path.join(output_dir, f"density_step_{t}.png"))
        plt.close()

        # Phase plot (3D)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, phase, cmap='viridis')
        ax.set_zlabel("Coherence")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlim(-0.03, 0.03)
        plt.title(f"Step {t} - Vortex Sim Phase Shift")
        fig.colorbar(surf, label="Coherence")
        plt.savefig(os.path.join(output_dir, f"phase_step_{t}.png"))
        plt.close()

        # Track central density
        central_density = density[center_x, center_y]
        with open(os.path.join(output_dir, "central_density.txt"), "a") as f:
            f.write(f"{t} {central_density}\n")

# Final output
print(f"Simulation complete. Check outputs in {output_dir}")

# Plot central density over time
central_data = np.loadtxt(os.path.join(output_dir, "central_density.txt"))
plt.figure(figsize=(8, 6))
plt.plot(central_data[:, 0], central_data[:, 1])
plt.xlabel("Step")
plt.ylabel("Central Density")
plt.title("Central Density Over Time")
plt.savefig(os.path.join(output_dir, "central_density.png"))
plt.close()