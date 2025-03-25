import numpy as np
import tensornetwork as tn
from scipy.linalg import expm

dt = 0.01  # Time step
steps = 100
qubits = 50  # Scaling up
bond_dim = 4  # More room for entanglement

# Pauli matrices
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

def hamiltonian(nodes, t):
    H = [Z if i % 2 == 0 else I for i in range(len(nodes))]
    # Photon coupling: X-X between neighbors
    for i in range(len(nodes) - 1):
        H[i] += 0.1 * X  # NV-Si sync boost
        H[i + 1] += 0.1 * X
    return H

def evolve_chunk(args):
    nodes, t = args
    H = hamiltonian(nodes, t)
    new_nodes = []
    for i, node in enumerate(nodes):
        tensor = node.tensor.reshape(2, -1)
        U = expm(-1j * dt * H[i])
        evolved = np.dot(U, tensor)
        new_nodes.append(tn.Node(evolved.reshape(2, *node.tensor.shape[1:])))
    return new_nodes

# Init MPS
nodes = []
for i in range(qubits):
    if i == 0:
        tensor = np.zeros((2, bond_dim), dtype=complex)
        tensor[0, 0] = 1.0
    elif i == qubits - 1:
        tensor = np.zeros((bond_dim, 2), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1] = 1.0 / np.sqrt(2)
    else:
        tensor = np.zeros((bond_dim, 2, bond_dim), dtype=complex)
        tensor[0, 0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1, 0] = 1.0 / np.sqrt(2)
    nodes.append(tn.Node(tensor))

# Link MPS chain
mps_nodes = nodes.copy()
for i in range(qubits - 1):
    edge = mps_nodes[i][-1] ^ mps_nodes[i + 1][0]
    contracted = tn.contract(edge)
    mps_nodes[i + 1] = contracted
    if i == 0:
        mps_nodes[i] = contracted
    else:
        mps_nodes[i] = mps_nodes[i - 1]

# Run sim
for t in range(steps):
    mps_nodes = evolve_chunk((mps_nodes, t * dt))
    norm = np.linalg.norm(mps_nodes[0].tensor.flatten())
    print(f"Step {t}, norm: {norm:.4f}")

# Entropy check
entropy = tn.von_neumann_entropy(mps_nodes[0])
print(f"Entropy: {entropy:.4f} bits")
print("50-qubit test done. Next stop: 500?")