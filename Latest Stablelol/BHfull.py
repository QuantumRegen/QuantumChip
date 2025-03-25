import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 250
L = 5.0
dx = L / (N - 1)
dt = 0.00005
steps = 500
m = 1.0

# Grid
X, Y = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))
R = np.sqrt((X - L / 2) ** 2 + (Y - L / 2) ** 2)

# Initial Condition
psi = np.ones((N, N), dtype=np.complex128) * np.sqrt(0.1)
psi += 0.05 * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / 0.01)
density = np.abs(psi) ** 2
phase = np.angle(psi)

# Qiskit Data (Placeholder—Insert Your Results)
qiskit_density_factor = 1.0  # From Nv-Si regen (e.g., fidelity scaling)
qiskit_phase_std = 0.03      # From Qiskit runs today

# Storage
steps_to_save = [0, 52, 200, 500]
density_history = [density.copy()]
phase_history = [phase.copy()]

# Time Evolution
for step in range(1, steps + 1):
    t = step * dt

    # Gravitational Potential
    V_gravity = -2e4 * np.exp(-R ** 2 / (0.0001 + t / 250) ** 2)

    # Pulse (Quantum Trigger)
    V_laser = np.zeros((N, N), dtype=np.complex128)
    if 50 <= step <= 52:
        V_laser = -2500 * np.exp(-R ** 2 / 0.01) * np.cos(100 * t)

    # EM Hint (New—Simplified Charge Effect)
    V_em = 0.01 * np.sin(50 * t) * np.exp(-R ** 2 / 0.1)  # Weak EM oscillation

    # Total Potential
    V = V_gravity + V_laser + V_em

    # Nonlinear and Damping
    nonlinear = 25 * density * qiskit_density_factor  # Qiskit tweak
    damping = (-0.5 * psi - 2.0 * (density > 0.1) * density * psi - 0.1 * R / L * psi).astype(np.complex128)

    # Quantum Foam Resistance
    F_foam = np.zeros((N, N), dtype=np.complex128)
    if 50 <= step <= 52:
        grad_density_x = np.gradient(density, axis=0) / dx
        grad_density_y = np.gradient(density, axis=1) / dx
        F_foam = (-0.01 * (grad_density_x ** 2 + grad_density_y ** 2) * psi).astype(np.complex128)

    # Dark Matter Geometry
    rho_max = np.max(density)
    R_eff = np.sqrt(X ** 2 + (1.0 + 0.2 * density / rho_max) * Y ** 2) + 0.1 * np.abs(np.cos(100 * t))
    if np.max(density) > 0.05:
        V += -0.05 * R_eff  # Geometric feedback

    # Vortex (Wayne Pulse Refinement)
    central_density = density[N // 2, N // 2]
    if central_density > 1.0:
        phase_std = np.std(phase)
        v_speed = 1.5 * np.exp(-(phase_std / qiskit_phase_std) ** 2 / 0.1)  # Qiskit-tuned
        psi *= np.exp(1j * v_speed * dt * (X - L / 2) / R)  # Spin phase shift

    # Phase Damping (Post-Pulse)
    if step > 52:
        phase_mean = np.mean(phase)
        damping += (-0.05 * (phase - phase_mean) * psi).astype(np.complex128)

    # Laplacian
    laplacian = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
                 np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi) / dx ** 2

    # Time Step
    dpsi_dt = ((-1j / (2 * m)) * laplacian + (-1j * V * psi) + (-1j * nonlinear * psi) +
               damping + F_foam + 0.0005 * np.random.normal(size=(N, N)).astype(np.complex128))
    psi = (psi + dt * dpsi_dt).astype(np.complex128)

    # Update
    density = np.abs(psi) ** 2
    phase = np.angle(psi)

    # Save
    if step in steps_to_save:
        density_history.append(density.copy())
        phase_history.append(phase.copy())

    # Normalize
    if step % 10 == 0:
        psi /= np.sqrt(np.mean(np.abs(psi) ** 2))

# Save Data
np.save('/home/wayne/Desktop/THEORY/sim_data_uft.npy',
        {'density': density_history, 'phase': phase_history, 'steps': steps_to_save})

print(f"UFT Sim complete. Data saved. Final density: {density[N // 2, N // 2]:.2f}, Phase std dev: {np.std(phase):.2f}")

# Plots
plt.figure(figsize=(12, 10))
for i, step in enumerate(steps_to_save):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.log10(density_history[i] + 1e-10), extent=[0, 5, 0, 5], cmap='viridis')
    plt.colorbar(label='Log10 Density')
    plt.title(f'Density at Step {step}')
plt.tight_layout()
plt.savefig('/home/wayne/Desktop/THEORY/density_plots_uft.png')
plt.close()

plt.figure(figsize=(12, 10))
for i, step in enumerate(steps_to_save):
    plt.subplot(2, 2, i + 1)
    plt.imshow(phase_history[i], extent=[0, 5, 0, 5], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Phase (radians)')
    plt.title(f'Phase at Step {step}')
plt.tight_layout()
plt.savefig('/home/wayne/Desktop/THEORY/phase_plots_uft.png')
plt.close()