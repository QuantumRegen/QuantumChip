import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vortex params
L = 1000
x = np.linspace(-L/2, L/2, 50)
y = np.linspace(-L/2, L/2, 50)
X, Y = np.meshgrid(x, y)
vortex_speed = 1.0  # Vortex flow proxy

# Vortex flow + horizon
def vortex_flow(x):
    return vortex_speed * np.tanh(x / 100)
V = vortex_flow(X)
horizon = np.where(np.abs(V - 1.0) < 0.1, 1, 0)  # Horizon proxy
sink = np.exp(-np.abs(X) / 50) * np.cos(2 * np.pi * (X**2 + Y**2) / L)  # Fibonacci sink

# Phase shift sim: Vortex coherence
np.random.seed(42)
vortex_noise = 0.1 * np.random.normal(size=X.shape)
vortex_state = vortex_noise * sink * horizon  # Vortex qubits
outer_state = vortex_noise * sink * (1 - horizon)  # Outer region
phase_shift = vortex_state * np.cos(V) + outer_state * np.sin(V)  # Phase lock
coherence = np.correlate(phase_shift.flatten(), vortex_state.flatten(), mode='same') / len(x)

# Viz (5M polys proxy)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phase_shift, cmap='plasma', alpha=0.8)
ax.set_title('Vortex Sim Phase Shift')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Coherence')
plt.savefig('vortex_phase_shift.png')
print("Vortex phase shift saved as 'vortex_phase_shift.png'")