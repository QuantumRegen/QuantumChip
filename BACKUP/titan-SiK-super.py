import numpy as np
import tensornetwork as tn
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import datetime

# Simulation parameters
dt = 0.01  # Time step
steps = 200  # Depth (extended)
qubits = 200  # Scaled from 25 to 200
bond_dim = 2
b_field_strength = 0.3  # Adjusted B-field
noise_scale = 0.05  # Adjusted noise
damping_factor = 0.01  # Damping to reduce fluctuations

# Pauli matrices
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

# Noise generation function
def add_noise(H_local, scale):
    return H_local + np.random.normal(0, scale, H_local.shape)

# Hamiltonian: Local terms with photon coupling and Kagome structure
def hamiltonian(t):
    H = []
    for i in range(qubits):
        if i == 0:  # Photon/vortex seed with adjustable B-field
            H_local = Z + b_field_strength * X - damping_factor * I
            H.append(add_noise(H_local, noise_scale))
        elif i % 3 == 0:  # Kagome nodes with triangular coupling
            coupling = 0.05 * (X + np.roll(X, 1) + np.roll(X, 2))
            H_local = Z + coupling - damping_factor * I
            H.append(add_noise(H_local, noise_scale))
        else:
            H_local = I - damping_factor * I
            H.append(add_noise(H_local, noise_scale))
    return H

# Evolve MPS by applying local unitaries to each qubit
def evolve_chunk(nodes, t):
    H = hamiltonian(t)
    new_nodes = []
    for i, node in enumerate(nodes):
        tensor = node.tensor
        U = expm(-1j * dt * H[i])
        if i == 0:
            evolved = np.dot(U, tensor)
        elif i == qubits - 1:
            evolved = np.dot(tensor.T, U.T).T
        else:
            evolved = np.tensordot(U, tensor, axes=([1], [1])).transpose(1, 0, 2)
        new_nodes.append(tn.Node(evolved))
    return new_nodes

# Compute reduced density matrix for the first qubit
def get_rho_first_qubit(nodes):
    conj_nodes = [tn.Node(np.conj(node.tensor)) for node in nodes]
    for i in range(1, qubits):
        nodes[i][1] ^ conj_nodes[i][1]  # Physical indices for tracing out
    for i in range(qubits - 1):
        nodes[i][1 if i == 0 else 2] ^ nodes[i + 1][0]
    for i in range(qubits - 1):
        conj_nodes[i][1 if i == 0 else 2] ^ conj_nodes[i + 1][0]
    network = tn.replicate_nodes(nodes + conj_nodes)
    rho = tn.contractors.auto(network, output_edge_order=[network[0][0], network[qubits][0]]).tensor
    trace_rho = np.trace(rho)
    if trace_rho != 0:
        rho /= trace_rho
    return rho

# Initialize MPS
nodes = []
for i in range(qubits):
    if i == 0:
        tensor = np.zeros((2, bond_dim), dtype=complex)
        tensor[0, 0] = 1.0  # |0>
    elif i == qubits - 1:
        tensor = np.zeros((bond_dim, 2), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1] = 1.0 / np.sqrt(2)  # |+>
    else:
        tensor = np.zeros((bond_dim, 2, bond_dim), dtype=complex)
        tensor[0, 0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1, 0] = 1.0 / np.sqrt(2)  # |+>
    nodes.append(tn.Node(tensor))

# Run simulation
entropies = []
coherences = []
uptimes = []
for t in range(steps):
    nodes = evolve_chunk(nodes, t * dt)
    rho = get_rho_first_qubit(nodes)
    eigvals = np.linalg.eigvalsh(rho)
    entropy = -np.sum([x * np.log2(x + 1e-10) for x in eigvals if x > 0])
    coherence = np.real(np.trace(rho @ X))
    uptime_flag = 1 if abs(coherence) > 0.2 else 0  # Adjusted threshold
    entropies.append(entropy)
    coherences.append(coherence)
    uptimes.append(uptime_flag)
    if t % 10 == 0:
        print(f"Step {t}, Entropy: {entropy:.4f}, Coherence: {coherence:.4f}, Uptime: {uptime_flag}")

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))
time = np.arange(steps) * dt
ax1.plot(time, entropies, 'b-', label='Entropy')
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(time, coherences, 'r-', label='Coherence <σ_x>')
ax2.set_ylabel('Coherence', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(time, uptimes, 'g--', label='Uptime')
ax3.set_ylabel('Uptime', color='g')
ax3.tick_params(axis='y', labelcolor='g')
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
title = f"Titan-SiK Simulation\nWayne Spratley\n{current_time}\nQubits: {qubits}, Depth: {steps}\nB-field: {b_field_strength}, Noise: {noise_scale}"
plt.title(title)
lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
plt.savefig('titan_sik_simulation_optimized.png')
plt.show()
print(f"Simulation complete. Scaled to {qubits} qubits successfully.")