import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

# Start timer
start_time = time.time()
print("Starting 2D time-evolution sim on T630 with CuPy and CUDA 12.4...")

# Parameters
L = 1000
N = 500  # 500x500 grid
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = 1e-3  # Time step
time_steps = 10000  # 1 µs to 10 s (scaled)
save_interval = 100  # 100 PNGs
c_s = 1.0
v_max = 1.5
horizon_pos = 0
m = 1.0
hbar = 1.0

# Initial wavefunction (vacuum fluctuation)
np.random.seed(42)
psi = 0.1 * np.random.normal(size=(N, N)) + 0j
psi = psi / np.linalg.norm(psi)
psi_gpu = cp.array(psi)

# Flow velocity (2D, x-dependent)
def flow_velocity(x):
    return v_max * np.tanh(x / 100)

V = cp.array(flow_velocity(X))
horizon = cp.where(cp.abs(V - c_s) < 0.1, 1, 0)

# Time-dependent potential (gravity well)
def potential(X, Y, t):
    strength = 10 * (1 - cp.exp(-t / 2000))  # Grows over time
    return -strength * cp.exp(-cp.sqrt(X**2 + Y**2) / 50)

# Phonon pairs (Hawking radiation proxy)
def generate_phonons(psi, t):
    noise_amplitude = 0.05 * (1 + t / 10000)
    phonon_inside = noise_amplitude * cp.random.normal(size=(N, N))
    phonon_outside = cp.copy(phonon_inside)
    phonon_inside = cp.where(X < horizon_pos, -phonon_inside, phonon_inside)
    decay_scale = 50
    decay = cp.exp(-cp.sqrt(X**2 + Y**2) / decay_scale)
    return phonon_inside * decay, phonon_outside * decay

# Split-step Fourier for 2D Gross-Pitaevskii
kx = 2 * np.pi * np.fft.fftfreq(N, dx)
ky = 2 * np.pi * np.fft.fftfreq(N, dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

# Parallel evolution function with sliced K2
def evolve_chunk(args):
    chunk, t, k2_slice = args
    chunk_gpu = cp.array(chunk)
    chunk_k = cp.fft.fft2(chunk_gpu)
    chunk_k *= cp.exp(-1j * hbar * cp.array(k2_slice) * dt / (2 * m))
    chunk = cp.asnumpy(cp.fft.ifft2(chunk_k))
    V_chunk = cp.asnumpy(potential(X, Y, t))
    chunk *= np.exp(-1j * V_chunk * dt / hbar)
    chunk *= np.exp(-1j * 0.1 * np.abs(chunk)**2 * dt / hbar)
    return chunk

# Main loop
pool = Pool(36)  # Use all 36 cores
psi_chunks = np.array_split(cp.asnumpy(psi_gpu), 36, axis=0)
k2_chunks = np.array_split(K2, 36, axis=0)

for step in range(time_steps + 1):
    t = step * dt
    psi_chunks = pool.map(evolve_chunk, [(chunk, t, k2_chunk) for chunk, k2_chunk in zip(psi_chunks, k2_chunks)])
    psi = np.concatenate(psi_chunks, axis=0)
    psi_gpu = cp.array(psi)
    psi_gpu /= cp.linalg.norm(psi_gpu)

    # Phonons
    phonon_inside, phonon_outside = generate_phonons(psi_gpu, t)

    # Save visuals every save_interval
    if step % save_interval == 0:
        density = cp.asnumpy(cp.abs(psi_gpu)**2)
        phonon_inside_sum = cp.asnumpy(phonon_inside).sum(axis=0)
        phonon_outside_sum = cp.asnumpy(phonon_outside).sum(axis=0)

        plt.figure(figsize=(15, 5))
        # Density
        plt.subplot(1, 3, 1)
        plt.imshow(density, extent=[-L/2, L/2, -L/2, L/2], cmap='viridis')
        plt.colorbar(label='Density')
        plt.title(f'Density - t={t:.3f}')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Phonons
        plt.subplot(1, 3, 2)
        plt.plot(x, phonon_inside_sum, label='Inside', color='green')
        plt.plot(x, phonon_outside_sum, label='Outside', color='orange')
        plt.axvline(x[horizon_idx], color='black', linestyle=':')
        plt.title('Phonon Pairs')
        plt.xlabel('X')
        plt.ylabel('Amplitude')
        plt.legend()

        # Correlation
        correlation = np.correlate(phonon_inside_sum, phonon_outside_sum, mode='same') / N
        plt.subplot(1, 3, 3)
        plt.plot(x, correlation, label='Correlation', color='purple')
        plt.axvline(x[horizon_idx], color='black', linestyle=':')
        plt.title('Correlation')
        plt.xlabel('X')
        plt.ylabel('Strength')
        plt.legend()

        plt.tight_layout()
        filename = f'evolution_2d_step_{step}.png'
        plt.savefig(filename, dpi=100)
        plt.close()
        print(f"Saved {filename}")

pool.close()
print(f"Simulation completed in {(time.time() - start_time) / 60:.1f} minutes!")