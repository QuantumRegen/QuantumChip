import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

# Config
n_qubits = 25  # 25 fits 64GB, extrapolate to 100
res_x, res_y = 1920, 1080  # High-res viz
t_steps = 30  # Vortex steps, tweak for speed


# Hamiltonian (NV-Si + vortex)
def hamiltonian(state, t):
    n = len(state)
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (H + H.conj().T) / 2  # Hermitian
    return H * np.sin(t)  # Vortex kick


# Parallel evolution
def evolve_chunk(args):
    chunk, t = args
    H = hamiltonian(chunk, t)
    return chunk - 1j * 0.01 * H.dot(chunk)  # Schrödinger step


# Main sim
def run_sim():
    start = time.time()

    # Initial state (dense, no sparse)
    state = np.zeros(1024, dtype=complex)  # Truncate to fit
    state[0] = 1.0  # |0...0>

    # Split for 44 cores
    chunks = np.array_split(state, 44)
    pool = Pool(44)

    # Time evolution
    for t in np.linspace(0, 1, t_steps):
        state_chunks = pool.map(evolve_chunk, [(chunk, t) for chunk in chunks])
        state = np.concatenate(state_chunks)
        state /= np.linalg.norm(state)  # Normalize to keep probs sane

    # Entropy
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)  # Ensure sum to 1
    entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Avoid log(0)
    print(f"Entropy: {entropy:.2f} bits")

    # CPU viz
    phase = np.angle(state)
    x, y = np.meshgrid(np.linspace(0, 1, res_x), np.linspace(0, 1, res_y))
    z = np.sin(x * 10 + phase.mean()) + np.cos(y * 10 + phase.std())  # Vortex

    # Blue-red-green sync
    rgb = np.zeros((res_y, res_x, 3))
    rgb[..., 0] = np.abs(np.sin(z))  # Red
    rgb[..., 1] = np.cos(z + 1.5)  # Green
    rgb[..., 2] = np.sin(z + 3.0)  # Blue

    # Save PNG
    plt.imsave("Titan-Si.png", rgb.clip(0, 1))

    end = time.time()
    print(f"Time: {end - start:.2f}s")


if __name__ == "__main__":
    run_sim()