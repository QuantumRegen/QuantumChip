import numpy as np
import tensornetwork as tn
from scipy.linalg import expm
import psutil
import gc
from multiprocessing import Pool
import sys

dt = 0.01
steps = 30
qubits = 25  # Start at 25, scale to 500 later
bond_dim = 4

# Pauli matrices—yelling complex
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

def hamiltonian(nodes, t):
    print(f"Hamiltonian time {t:.3f}, nodes: {len(nodes)}")
    H = [Z if i % 2 == 0 else I for i in range(len(nodes))]
    for i in range(len(nodes)):
        H[i] += 0.2 * X  # Photon sync—loud boost
    H_pairs = [1.0 * np.kron(X, X) for _ in range(len(nodes) - 1)]  # X-X roar
    return H, H_pairs

def evolve_node(args):
    i, node, H, H_pairs, dt = args
    shape = node.tensor.shape
    print(f"Node {i} shape in: {shape}")
    tensor = node.tensor.reshape(2, -1)
    U = expm(-1j * dt * H[i])
    evolved = np.dot(U, tensor)
    if i == 0:
        out_shape = (2, bond_dim)
    elif i == qubits - 1:
        out_shape = (bond_dim, 2)
    else:
        out_shape = (bond_dim, 2, bond_dim)
    evolved = evolved.reshape(out_shape)
    print(f"Node {i} shape out: {out_shape}")
    return tn.Node(evolved)

def evolve_chunk(args):
    print("Entering evolve_chunk...")
    nodes, t = args
    H, H_pairs = hamiltonian(nodes, t)
    print(f"Starting chunk, t={t:.3f}, nodes={len(nodes)}")
    try:
        with Pool(processes=72) as pool:  # 72 threads—CPU feast
            print("Pool started...")
            new_nodes = pool.map(evolve_node, [(i, node, H, H_pairs, dt) for i, node in enumerate(nodes)])
            print("Pool map done!")
    except Exception as pool_err:
        print(f"Pool crashed: {pool_err}")
        raise
    # X-X sweep—screaming dims, rebuild legit
    for i in range(len(nodes) - 1):
        edge1, edge2 = new_nodes[i][-1], new_nodes[i + 1][0]
        print(f"X-X at {i}: edge1={edge1.dimension}, edge2={edge2.dimension}")
        if edge1.dimension != bond_dim or edge2.dimension != bond_dim:
            raise ValueError(f"Edge dims off at {i}: {edge1.dimension} vs {edge2.dimension}")
        theta = tn.contract(edge1 ^ edge2).tensor  # (2, 2)
        print(f"Theta shape pre-reshape: {theta.shape}")
        theta = theta.reshape(4, 4)  # 2 qubits, 4 states
        print(f"Theta shape post-reshape: {theta.shape}")
        U = expm(-1j * dt * H_pairs[i])
        print(f"U shape: {U.shape}")
        evolved = np.dot(U, theta)  # (4, 4)
        print(f"Evolved shape: {evolved.shape}")
        u, s, vh = np.linalg.svd(evolved, full_matrices=False)
        s = s[:bond_dim]
        u = u[:, :bond_dim]  # (4, 4)
        vh = vh[:bond_dim, :]  # (4, 4)
        print(f"SVD: u={u.shape}, s={s.shape}, vh={vh.shape}")
        left_shape = (2, bond_dim) if i == 0 else (bond_dim, 2, bond_dim)
        right_shape = (bond_dim, 2) if i == len(nodes) - 2 else (bond_dim, 2, bond_dim)
        # Rebuild tensors legit—no reshape hacks
        if i == 0:
            left_tensor = u[:2, :]  # (2, 4)
            right_tensor = np.dot(np.diag(s), vh)  # (4, 4) -> reshape to (4, 2, 4)
        elif i == len(nodes) - 2:
            left_tensor = u  # (4, 4)
            right_tensor = vh[-2:, :]  # (2, 4)
        else:
            left_tensor = u  # (4, 4)
            right_tensor = np.dot(np.diag(s), vh)  # (4, 4)
        new_nodes[i] = tn.Node(left_tensor.reshape(left_shape))
        if i == len(nodes) - 2:
            new_nodes[i + 1] = tn.Node(right_tensor.reshape(right_shape))  # (4, 2)
        else:
            # Rebuild right_tensor to match (4, 2, 4)
            right_tensor_expanded = np.tensordot(np.diag(s), vh, axes=0)  # (4, 4, 4)
            new_nodes[i + 1] = tn.Node(right_tensor_expanded.reshape(right_shape))  # (4, 2, 4)
        print(f"Node {i} new shape: {new_nodes[i].tensor.shape}")
        print(f"Node {i+1} new shape: {new_nodes[i+1].tensor.shape}")
    print("Chunk done!")
    return new_nodes

# Init MPS—yell it out
nodes = []
for i in range(qubits):
    if i == 0:
        tensor = np.zeros((2, bond_dim), dtype=complex)
        tensor[0, 0] = 1.0
        print(f"Node {i}: (2, {bond_dim})")
    elif i == qubits - 1:
        tensor = np.zeros((bond_dim, 2), dtype=complex)
        tensor[0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1] = 1.0 / np.sqrt(2)
        print(f"Node {i}: ({bond_dim}, 2)")
    else:
        tensor = np.zeros((bond_dim, 2, bond_dim), dtype=complex)
        tensor[0, 0, 0] = 1.0 / np.sqrt(2)
        tensor[0, 1, 0] = 1.0 / np.sqrt(2)
        print(f"Node {i}: ({bond_dim}, 2, {bond_dim})")
    nodes.append(tn.Node(tensor))

# Link MPS chain—fix over-contraction
mps_nodes = nodes.copy()
print("Linking MPS chain...")
for i in range(qubits - 1):
    print(f"Contracting edge {i} to {i+1}")
    edge = mps_nodes[i][-1] ^ mps_nodes[i + 1][0]
    tn.contract(edge)  # Connect, don’t collapse
print("Chain linked—keeping nodes intact.")

# Entropy calc—loud result
def compute_entropy(nodes):
    mid = len(nodes) // 2
    print(f"Entropy calc—contracting {mid} nodes...")
    rho = tn.contract_between(nodes[:mid], nodes[:mid]).tensor
    _, s, _ = np.linalg.svd(rho.reshape(2**mid, -1), full_matrices=False)
    s = s[s > 1e-10]
    ent = -np.sum(s**2 * np.log2(s**2)) if s.size > 0 else 0.0
    print(f"Entropy computed: {ent:.4f} bits")
    return ent

# Run with stats—verbose as hell
print("Starting main loop...")
try:
    for t in range(steps):
        print(f"\nStep {t} kicking off...")
        mps_nodes = evolve_chunk((mps_nodes, t * dt))
        norm = np.linalg.norm(mps_nodes[0].tensor.flatten())
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"Step {t}, norm: {norm:.4f}, RAM: {mem.percent}%, Swap: {swap.percent}%, CPU: {cpu}%")
        gc.collect()
    entropy = compute_entropy(mps_nodes)
    print(f"Entropy: {entropy:.4f} bits")
    print("25-qubit test done. Scaling next.")
except Exception as e:
    print(f"Crash caught in main loop: {e}")
    sys.exit(1)
print("Main loop finished!")