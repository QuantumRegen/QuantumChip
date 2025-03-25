import numpy as np
import tensornetwork as tn
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import datetime

# Simulation parameters (user-adjustable)
dt = 0.01  # Time step
steps = 500  # Depth
qubits = 200  # Number of qubits
bond_dim = 4  # Bond dimension for MPS

# Adjustable parameters
b_field_spin = 7  # Spin direction of B-field (0 = +Z, 1 = -Z, 2 = +X, 3 = -X)
b_field_strength = 5  # Strength of the B-field
vortex_spin = -5  # Spin of the vortex (0 = no spin, 1 = clockwise, -1 = counterclockwise)
laser_decoupling = True  # Enable/disable laser decoupling
laser_strength = 1  # Strength of laser decoupling field

# Pauli matrices
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

# Hamiltonian with adjustable B-field and vortex spin
def hamiltonian(t):
    H = []
    for i in range(qubits):
        if i == 0:
            # Photon coupling with B-field and vortex effect
            b_term = Z if b_field_spin in [0, 1] else X
            if b_field_spin in [1, 3]:
                b_term = -b_term  # Flip direction for -Z or -X
            vortex_phase = np.exp(1j * vortex_spin * t * 0.1) if vortex_spin != 0 else 1.0
            H.append(b_field_strength * b_term * vortex_phase + 0.05 * X)  # NV-Si photon teaser
        elif i % 2 == 0:
            # Even indices with B-field
            b_term = Z if b_field_spin in [0, 1] else X
            if b_field_spin in [1, 3]:
                b_term = -b_term
            H.append(b_field_strength * b_term)
        else:
            H.append(I)  # Odd indices (can be modified for full interaction)
    return H

# Laser decoupling term (simple time-dependent control)
def laser_decoupling_term(t):
    if laser_decoupling:
        return laser_strength * np.sin(t) * X  # Sinusoidal control field
    return 0 * X

# Evolve MPS by applying local unitaries to each qubit
def evolve_chunk(nodes, t):
    H = hamiltonian(t)
    new_nodes = []
    for i, node in enumerate(nodes):
        tensor = node.tensor
        U_ham = expm(-1j * dt * H[i])  # Hamiltonian evolution
        if i == 0:
            decoupling = laser_decoupling_term(t)
            U_dec = expm(-1j * dt * decoupling)  # Decoupling evolution
        else:
            U_dec = np.eye(2, dtype=complex)  # Identity for no decoupling
        U = U_ham @ U_dec  # Combined unitary
        if i == 0:
            # Shape (2, bond_dim)
            evolved = np.dot(U, tensor)  # Apply U to physical index
        elif i == qubits - 1:
            # Shape (bond_dim, 2)
            evolved = np.dot(tensor, U)  # Apply U to physical index
        else:
            # Shape (bond_dim, 2, bond_dim)
            evolved = np.tensordot(U, tensor, axes=([1], [1])).transpose(1, 0, 2)
        new_nodes.append(tn.Node(evolved))
    return new_nodes

# Compute reduced density matrix for the first qubit
def get_rho_first_qubit(nodes):
    conj_nodes = [tn.Node(np.conj(node.tensor)) for node in nodes]
    # Connect physical indices for qubits 1 to N-1
    for i in range(1, qubits):
        nodes[i][1] ^ conj_nodes[i][1]  # Physical indices for tracing out
    # Connect bond indices for MPS
    for i in range(qubits - 1):
        nodes[i][1 if i == 0 else 2] ^ nodes[i + 1][0]
    # Connect bond indices for conjugate MPS
    for i in range(qubits - 1):
        conj_nodes[i][1 if i == 0 else 2] ^ conj_nodes[i + 1][0]
    # Contract network
    network = tn.replicate_nodes(nodes + conj_nodes)
    rho = tn.contractors.auto(network, output_edge_order=[network[0][0], network[qubits][0]]).tensor
    # Normalize
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

# Run simulation and collect data
entropies = []
coherences = []
uptimes = []
for t in range(steps):
    nodes = evolve_chunk(nodes, t * dt)
    rho = get_rho_first_qubit(nodes)
    # Entropy: -Tr(rho log2 rho)
    eigvals = np.linalg.eigvalsh(rho)
    entropy = -np.sum([x * np.log2(x + 1e-10) for x in eigvals if x > 0])
    # Coherence: <σ_x>
    coherence = np.real(np.trace(rho @ X))
    # Uptime: 1 if |<σ_x>| > 0.5, else 0
    uptime_flag = 1 if abs(coherence) > 0.5 else 0
    entropies.append(entropy)
    coherences.append(coherence)
    uptimes.append(uptime_flag)
    if t % 10 == 0:
        print(f"Step {t}, Entropy: {entropy:.4f}, Coherence: {coherence:.4f}")

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))
time = np.arange(steps) * dt

# Entropy (left y-axis)
ax1.plot(time, entropies, 'b-', label='Entropy')
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Coherence (right y-axis)
ax2 = ax1.twinx()
ax2.plot(time, coherences, 'r-', label='Coherence <σ_x>')
ax2.set_ylabel('Coherence', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Uptime (second right y-axis)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(time, uptimes, 'g--', label='Uptime')
ax3.set_ylabel('Uptime', color='g')
ax3.tick_params(axis='y', labelcolor='g')

# Header with metadata
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
title = f"Titan-Si Simulation\nWayne Spratley\n{current_time}\nQubits: {qubits}, Depth: {steps}"
plt.title(title)

# Legend
lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')

# Save and display
plt.savefig('titan_si_simulation.png')
plt.show()
print("Simulation complete. Scaled to 200 qubits successfully.")