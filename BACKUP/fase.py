import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NV-Si params
L = 1000
x = np.linspace(-L/2, L/2, 50)
y = np.linspace(-L/2, L/2, 50)
X, Y = np.meshgrid(x, y)
nv_speed = 1.5  # NV gate speed proxy
si_speed = 0.5  # Si gate speed proxy

# NV-Si flow + handoff
def nv_flow(x):
    return nv_speed * np.tanh(x / 100)
def si_flow(x):
    return si_speed * np.tanh(x / 50)
NV = nv_flow(X)
SI = si_flow(X)
handoff = 0.5 * (NV + SI)  # Handoff mid-layer
sink = np.exp(-np.abs(X) / 50) * np.cos(2 * np.pi * (X**2 + Y**2) / L)  # Fibonacci-ish sink

# Phase shift sim: NV-Si coherence
np.random.seed(42)
nv_noise = 0.1 * np.random.normal(size=X.shape)
si_noise = 0.05 * np.random.normal(size=X.shape)
nv_state = nv_noise * sink  # NV spins
si_state = si_noise * sink  # Si logical
phase_shift = nv_state * np.cos(handoff) + si_state * np.sin(handoff)  # Phase lock
coherence = np.correlate(phase_shift.flatten(), nv_state.flatten(), mode='same') / len(x)

# Viz (2M polys proxy)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phase_shift, cmap='viridis', alpha=0.8)
ax.set_title('NV-Si Phase Shift')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Coherence')
plt.savefig('nvsi_phase_shift.png')
print("NV-Si phase shift saved as 'nvsi_phase_shift.png'")