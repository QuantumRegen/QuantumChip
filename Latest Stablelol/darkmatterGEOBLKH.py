import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
N = 500  # Grid size (500x500)
L = 5.0  # Physical size
dx = L / (N - 1)  # Spatial step
dt = 0.00005  # Time step
total_steps = 4000
output_dir = "/home/wayne/Desktop/THEORY/"
os.makedirs(output_dir, exist_ok=True)

# Grid setup
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
X_c, Y_c = N // 2, N // 2  # Center at (250, 250)
R = np.sqrt((X - X[X_c, Y_c])**2 + (Y - Y[Y_c, X_c])**2)

# Initial wavefunction with reduced bump
psi = np.ones((N, N), dtype=complex) * np.sqrt(0.1)  # Uniform density = 0.1
gaussian_bump = 0.05 * np.exp(-((X - X[X_c, Y_c])**2 + (Y - Y[Y_c, X_c])**2) / 0.05**2)
psi += gaussian_bump
density = np.abs(psi)**2

# Vortex and collapse tracking
vortex_speed = 0.0  # Starts at 0, ramps up
vortex_active_steps = 0
dark_matter_factor = 1.0  # Gradual transition to dark matter geometry

# Time evolution
for step in range(total_steps):
    t = step * dt

    # Central density
    central_density = density[X_c, Y_c]

    # Check for overflow and halt if necessary
    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)) or central_density > 1e12:
        print(f"Simulation halted at step {step} due to overflow or extreme values.")
        break

    # Normalize psi if magnitude becomes too large (lower threshold)
    psi_magnitude = np.max(np.abs(psi))
    if psi_magnitude > 100:  # Reduced from 1e3
        psi /= psi_magnitude
        density = np.abs(psi)**2
        print(f"Normalized psi at step {step}, max magnitude = {psi_magnitude}")

    # Vortex dynamics (delayed activation)
    if central_density > 0.2 and vortex_speed < 1.5:
        vortex_active_steps += 1
        vortex_speed = min(1.5, vortex_active_steps * 0.00375)  # Slower ramp over ~400 steps

    # Gravity potential with gradual dark matter geometry
    R_shifted = R
    if central_density > 0.05:
        dark_matter_factor = min(1.2, dark_matter_factor + 0.00025)
        R_eff = np.sqrt((X - X[X_c, Y_c])**2 + dark_matter_factor * (Y - Y[Y_c, X_c])**2)
        R_shifted = R_eff
    gravity_radius = 0.0001 + t / 250
    V_gravity = -1e5 * np.exp(-R_shifted**2 / gravity_radius**2)  # Reduced from -5e5

    # Nonlinear term (further reduced)
    nonlinear = 100 * density  # Reduced from 500

    # Damping (increased)
    initial_damping = -0.5 * psi  # Increased from -0.1
    damping = np.where(density > 0.1, -1.0 * density, 0.0)  # Increased from -0.5

    # Sink term (increased)
    sink = -0.1 * (R < 0.1)  # Increased from -0.05

    # Hawking noise (reduced)
    noise = 0.005 * (np.random.randn(N, N) + 1j * np.random.randn(N, N))  # Reduced from 0.01

    # Schrödinger-like evolution (simplified)
    laplacian = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
                 np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi) / dx**2
    dpsi_dt = -1j * (0.5 * laplacian + V_gravity * psi + nonlinear * psi +
                     damping * psi + sink * psi + noise + initial_damping)
    psi += dpsi_dt * dt

    # Clip psi to prevent overflow
    psi = np.clip(psi, -1e3, 1e3)

    # Update density
    density = np.abs(psi)**2

    # Absorbing boundaries
    psi[:10, :] *= 0.5
    psi[-10:, :] *= 0.5
    psi[:, :10] *= 0.5
    psi[:, -10:] *= 0.5

    # Output every 25 steps
    if step % 25 == 0:
        phase = np.angle(psi)
        max_density = np.max(density)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(density, cmap='hot', vmin=0, vmax=max_density if max_density > 0 else 0.5)
        plt.title(f"Density, Step {step}")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.title(f"Phase, Step {step}")
        plt.colorbar()
        plt.savefig(f"{output_dir}/step_{step:04d}.png")
        plt.close()

        print(f"Step {step}: Central Density = {central_density:.4f}, "
              f"Phase Range = {np.min(phase):.2f} to {np.max(phase):.2f}")

# Final output
np.save(f"{output_dir}/final_density.npy", density)
np.save(f"{output_dir}/final_phase.npy", np.angle(psi))
print("Simulation complete!")