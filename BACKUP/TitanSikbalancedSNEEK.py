import numpy as np
import tensornetwork as tn
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import datetime

# Simulation parameters
dt = 0.01
steps = 400
qubits = 200
bond_dim = 2
b_field_strength = 0.2
noise_scale = 0.05
damping_factor = 0.01

# Pauli matrices
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)


# Noise generation function
def add_noise(H_local, scale):
    return H_local + np.random.normal(0, scale, H_local.shape)


# Hamiltonian with adjusted ZZ coupling
def hamiltonian(t):
    H = []
    for i in range(qubits):
        if i == 0:
            H_local = Z + b_field_strength * X - damping_factor * I + 0.3 * Z
            H.append(add_noise(H_local, noise_scale))
        elif i == 1:
            H_local = Z + 0.3 * Z - damping_factor * I
            H.append(add_noise(H_local, noise_scale))
        elif i % 3 == 0:
            coupling = 0.05 * (X + np.roll(X, 1) + np.roll(X, 2))
            H_local = Z + coupling - damping_factor * I
            H.append(add_noise(H_local, noise_scale))
        else:
            H_local = I - damping_factor * I
            H.append(add_noise(H_local, noise_scale))
    return H


# Evolve MPS
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
    for i in range(len(new_nodes) - 1):
        new_nodes[i][1 if i == 0 else 2] ^ new_nodes[i + 1][0]
    return new_nodes


# Reduced density matrix for qubits 0 and 1
def get_rho_two_qubits(nodes):
    nodes_copy = [tn.Node(node.tensor.copy()) for node in nodes]
    conj_nodes = [tn.Node(np.conj(node.tensor)) for node in nodes_copy]

    for i in range(len(nodes_copy) - 1):
        nodes_copy[i][1 if i == 0 else 2] ^ nodes_copy[i + 1][0]
    for i in range(len(conj_nodes) - 1):
        conj_nodes[i][1 if i == 0 else 2] ^ conj_nodes[i + 1][0]

    for i in range(2, qubits):
        nodes_copy[i][1] ^ conj_nodes[i][1]

    network = nodes_copy + conj_nodes
    rho = tn.contractors.auto(network, output_edge_order=[nodes_copy[0][0], nodes_copy[1][1], conj_nodes[0][0],
                                                          conj_nodes[1][1]]).tensor
    rho = rho.reshape(4, 4)
    trace_rho = np.trace(rho)
    if abs(trace_rho) > 1e-10:
        rho /= trace_rho
    return rho.reshape(2, 2, 2, 2)


# Reduced density matrix for qubit 0
def get_rho_first_qubit(nodes):
    nodes_copy = [tn.Node(node.tensor.copy()) for node in nodes]
    conj_nodes = [tn.Node(np.conj(node.tensor)) for node in nodes_copy]

    for i in range(len(nodes_copy) - 1):
        nodes_copy[i][1 if i == 0 else 2] ^ nodes_copy[i + 1][0]
    for i in range(len(conj_nodes) - 1):
        conj_nodes[i][1 if i == 0 else 2] ^ conj_nodes[i + 1][0]

    for i in range(1, qubits):
        nodes_copy[i][1] ^ conj_nodes[i][1]

    network = nodes_copy + conj_nodes
    rho = tn.contractors.auto(network, output_edge_order=[nodes_copy[0][0], conj_nodes[0][0]]).tensor
    trace_rho = np.trace(rho)
    if abs(trace_rho) > 1e-10:
        rho /= trace_rho
    return rho


# Initialize MPS with Bell state
nodes = []
for i in range(qubits):
    if i == 0:
        tensor = np.zeros((2, bond_dim), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)  # |00> + |11>
        tensor[1, 1] = 1.0 / np.sqrt(2)
    elif i == 1:
        tensor = np.zeros((bond_dim, 2, bond_dim), dtype=complex)
        tensor[0, 0, 0] = 1.0 / np.sqrt(2)  # Matching |00> + |11>
        tensor[1, 1, 0] = 1.0 / np.sqrt(2)
    elif i == qubits - 1:
        tensor = np.zeros((bond_dim, 2), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1] = 1.0 / np.sqrt(2)
    else:
        tensor = np.zeros((bond_dim, 2, bond_dim), dtype=complex)
        tensor[0, 0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1, 0] = 1.0 / np.sqrt(2)
    nodes.append(tn.Node(tensor))

for i in range(len(nodes) - 1):
    nodes[i][1 if i == 0 else 2] ^ nodes[i + 1][0]

# Run simulation
entropies = []
coherences = []
uptimes = []
entanglements = []
for t in range(steps):
    nodes = evolve_chunk(nodes, t * dt)
    rho = get_rho_first_qubit(nodes)
    rho_two = get_rho_two_qubits(nodes)
    eigvals = np.linalg.eigvalsh(rho)
    entropy = -np.sum([x * np.log2(x + 1e-10) for x in eigvals if x > 0])
    sigma_x_x = np.kron(X, X)
    coherence = np.real(np.trace(rho_two.reshape(4, 4) @ sigma_x_x))
    rho_two_reduced = np.trace(rho_two.reshape(2, 2, 2, 2), axis1=2, axis2=3)
    eigvals_two = np.linalg.eigvalsh(rho_two_reduced)
    entanglement = -np.sum([x * np.log2(x + 1e-10) for x in eigvals_two if x > 0])
    uptime_flag = 1 if entanglement > 0.5 else 0
    entropies.append(entropy)
    coherences.append(coherence)
    uptimes.append(uptime_flag)
    entanglements.append(entanglement)
    if t % 10 == 0:
        print(
            f"Step {t}, Entropy: {entropy:.4f}, Coherence: {coherence:.4f}, Uptime: {uptime_flag}, Entanglement: {entanglement:.4f}")

# Plot results with entanglement included
fig, ax1 = plt.subplots(figsize=(10, 6))
time = np.arange(steps) * dt
ax1.plot(time, entropies, 'b-', label='Entropy')
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(time, coherences, 'r-', label='Coherence <σ_x ⊗ σ_x>')
ax2.plot(time, entanglements, 'm-', label='Entanglement', alpha=0.5)  # Add entanglement
ax2.set_ylabel('Coherence / Entanglement', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(time, uptimes, 'g--', label='Uptime')
ax3.set_ylabel('Uptime', color='g')
ax3.tick_params(axis='y', labelcolor='g')

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
title = f"Titan-SiK Simulation\nWayne Spratley\n{current_time}\nQubits: {qubits}, Depth: {steps}\nB-field: {b_field_strength}, Noise: {noise_scale}"
plt.title(title)
lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax2.get_lines()[1], ax3.get_lines()[0]]
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
plt.savefig('titan_sik_simulation_with_entanglement_plot.png')
plt.show()
print(f"Simulation complete. Scaled to {qubits} qubits successfully.")