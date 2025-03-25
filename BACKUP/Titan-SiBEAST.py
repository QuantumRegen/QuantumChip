import numpy as np
import tensornetwork as tn
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import datetime

# Simulation parameters
dt = 0.01
steps = 500
qubits = 25
initial_bond_dim = 8
max_bond_dim = 16

# Pauli matrices
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

# Hamiltonian
def hamiltonian(t):
    H = []
    for i in range(qubits):
        if i == 0:
            H.append(Z * (1 + 0.5 * np.sin(t)) + 0.5 * X)
        elif i % 2 == 0:
            H.append(Z)
        else:
            H.append(I)
    return H

# Two-qubit XX Hamiltonian
def two_qubit_hamiltonian():
    H_xx = 0.1 * np.kron(X, X)
    return expm(-1j * dt * H_xx)

# Evolve MPS with single-qubit and two-qubit gates
def evolve_chunk(nodes, bond_dims, t):
    # Single-qubit gates
    H = hamiltonian(t)
    new_nodes = []
    for i, node in enumerate(nodes):
        tensor = node.tensor
        U = expm(-1j * dt * H[i])
        if i == 0:
            evolved = np.dot(U, tensor)
        elif i == qubits - 1:
            evolved = np.dot(tensor, U)
        else:
            evolved = np.tensordot(U, tensor, axes=([1], [1])).transpose(1, 0, 2)
        new_nodes.append(tn.Node(evolved))

    # Two-qubit gates
    nodes = new_nodes
    U_two = two_qubit_hamiltonian()
    for i in range(0, qubits - 1, 2):  # Odd pairs
        node1, node2 = nodes[i], nodes[i + 1]
        tensor1, tensor2 = node1.tensor, node2.tensor
        if i == 0:
            contracted = np.tensordot(tensor1, tensor2, axes=([1], [0])).reshape(4, -1)
            evolved = np.dot(U_two, contracted)
            U, S, Vh = np.linalg.svd(evolved, full_matrices=False)
            dim = min(len(S), max_bond_dim)
            U = U[:, :dim]
            S = S[:dim]
            Vh = Vh[:dim, :]
            nodes[i] = tn.Node(U.reshape(2, 2, dim)[:, 0, :])  # (2, dim)
            nodes[i + 1] = tn.Node((np.diag(S) @ Vh).reshape(dim, 2, -1))
            bond_dims[i] = dim  # Update bond dimension
        elif i == qubits - 2:
            contracted = np.tensordot(tensor1, tensor2, axes=([2], [0])).reshape(bond_dims[i - 1], 4)
            evolved = np.dot(contracted, U_two)
            U, S, Vh = np.linalg.svd(evolved, full_matrices=False)
            dim = min(len(S), max_bond_dim)
            U = U[:, :dim]
            S = S[:dim]
            Vh = Vh[:dim, :]
            nodes[i] = tn.Node(U.reshape(bond_dims[i - 1], 2, dim))
            nodes[i + 1] = tn.Node((np.diag(S) @ Vh))
            bond_dims[i] = dim
        else:
            contracted = np.tensordot(tensor1, tensor2, axes=([2], [0])).reshape(bond_dims[i - 1], 4, bond_dims[i])
            evolved = np.tensordot(U_two, contracted, axes=([1], [1])).transpose(0, 2, 1)
            matrix = evolved.reshape(bond_dims[i - 1] * 4, bond_dims[i])
            U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
            dim = min(len(S), max_bond_dim)
            U = U[:, :dim]
            S = S[:dim]
            Vh = Vh[:dim, :]
            nodes[i] = tn.Node(U.reshape(bond_dims[i - 1], 4, dim)[:, :2, :])
            nodes[i + 1] = tn.Node((np.diag(S) @ Vh).reshape(dim, 2, -1))
            bond_dims[i] = dim
    return nodes, bond_dims

# Compute reduced density matrix for a given qubit
def get_rho_qubit(nodes, qubit_idx, bond_dims):
    conj_nodes = [tn.Node(np.conj(node.tensor)) for node in nodes]
    # Trace out all qubits except the target
    for i in range(qubits):
        if i != qubit_idx:
            nodes[i][1] ^ conj_nodes[i][1]
    # Connect bond indices with dynamic bond dimensions
    for i in range(qubits - 1):
        # Get the actual bond dimension from the tensor shapes
        current_bond_dim = nodes[i].tensor.shape[-1] if i < qubits - 1 else nodes[i].tensor.shape[-1]
        next_bond_dim = nodes[i + 1].tensor.shape[0] if i < qubits - 2 else nodes[i + 1].tensor.shape[0]
        if current_bond_dim != next_bond_dim and i < qubits - 2:
            raise ValueError(f"Bond dimension mismatch at i={i}: {current_bond_dim} vs {next_bond_dim}")
        nodes[i][1 if i == 0 else 2] ^ nodes[i + 1][0]
        conj_nodes[i][1 if i == 0 else 2] ^ conj_nodes[i + 1][0]
    network = nodes + conj_nodes
    if qubit_idx == 0:
        rho = tn.contractors.auto(network, output_edge_order=[network[0][0], network[qubits][0]]).tensor
    elif qubit_idx == qubits - 1:
        rho = tn.contractors.auto(network, output_edge_order=[network[qubits - 1][1], network[2 * qubits - 1][1]]).tensor
    else:
        rho = tn.contractors.auto(network, output_edge_order=[network[qubit_idx][1], network[qubits + qubit_idx][1]]).tensor
    trace_rho = np.trace(rho)
    if trace_rho != 0:
        rho /= trace_rho
    return rho

# Initialize MPS
nodes = []
bond_dims = [initial_bond_dim] * (qubits - 1)
for i in range(qubits):
    if i == 0:
        tensor = np.zeros((2, initial_bond_dim), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)
        tensor[1, 0] = 1.0 / np.sqrt(2)
    elif i == qubits - 1:
        tensor = np.zeros((initial_bond_dim, 2), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1] = 1.0 / np.sqrt(2)
    else:
        tensor = np.zeros((initial_bond_dim, 2, initial_bond_dim), dtype=complex)
        tensor[0, 0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1, 0] = 1.0 / np.sqrt(2)
    nodes.append(tn.Node(tensor))

# Run simulation and collect metrics for all qubits
metrics = {q: {'entropy': [], 'coherence': [], 'uptime': []} for q in range(qubits)}
for t in range(steps):
    nodes, bond_dims = evolve_chunk(nodes, bond_dims, t * dt)
    for qubit_idx in range(qubits):
        rho = get_rho_qubit(nodes, qubit_idx, bond_dims)
        # Entropy
        eigvals = np.linalg.eigvalsh(rho)
        entropy = -np.sum([x * np.log2(x) if x > 0 else 0 for x in eigvals])
        # Coherence
        coherence = np.real(np.trace(rho @ X))
        # Uptime
        uptime_flag = 1 if abs(coherence) > 0.3 else 0
        metrics[qubit_idx]['entropy'].append(entropy if not np.isnan(entropy) else 0)
        metrics[qubit_idx]['coherence'].append(coherence)
        metrics[qubit_idx]['uptime'].append(uptime_flag)
    if t % 50 == 0:
        print(f"Step {t}, Qubit 0 - Entropy: {metrics[0]['entropy'][-1]:.4f}, "
              f"Coherence: {metrics[0]['coherence'][-1]:.4f}, Uptime: {metrics[0]['uptime'][-1]}")

# Plotting
fig = plt.figure(figsize=(14, 12), dpi=300)

# Heatmap for Coherence
plt.subplot(2, 1, 1)
coherence_data = np.array([metrics[q]['coherence'] for q in range(qubits)])
plt.imshow(coherence_data, aspect='auto', cmap='viridis', extent=[0, steps * dt, 0, qubits - 1])
plt.colorbar(label='Coherence <σ_x>')
plt.xticks(np.arange(0, steps * dt + dt, 1.0))
plt.yticks(range(qubits), [f'Qubit {q}' for q in range(qubits)])
plt.xlabel('Time')
plt.ylabel('Qubit Index')
plt.title('Heatmap of Coherence Across All Qubits')

# Line plot for Qubit 0 metrics
plt.subplot(2, 1, 2)
time = np.arange(steps) * dt
ax1 = plt.gca()
ax1.plot(time, metrics[0]['entropy'], 'b-', label='Entropy')
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(time, metrics[0]['coherence'], 'r-', label='Coherence <σ_x>')
ax2.set_ylabel('Coherence', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(time, metrics[0]['uptime'], 'g--', label='Uptime')
ax3.set_ylabel('Uptime', color='g')
ax3.tick_params(axis='y', labelcolor='g')

# Title with metadata
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
title = f"Quantum Chip Simulation\nWayne Spratley\n{current_time}\nQubits: {qubits}, Steps: {steps}"
fig.suptitle(title)

# Legend
lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('quantum_chip_metrics.png', dpi=300)
plt.show()

print("Simulation complete. Scaled to 25 qubits with full metrics visualized.")