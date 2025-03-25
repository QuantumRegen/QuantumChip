import numpy as np
import tensornetwork as tn

def apply_two_qubit_gate(nodes, i, U_two, max_bond_dim=16):
    bond_dim = nodes[i].tensor.shape[0] if i > 0 else nodes[i + 1].tensor.shape[1]
    tensor1, tensor2 = nodes[i].tensor, nodes[i + 1].tensor

    # Step 1: Contract nodes i and i+1
    if i == 0:
        contracted = np.tensordot(tensor1, tensor2, axes=([1], [0]))  # (2, bond_dim) @ (bond_dim, 2, bond_dim)
        contracted = contracted.reshape(4, -1)  # (4, bond_dim)
    elif i == qubits - 2:
        contracted = np.tensordot(tensor1, tensor2, axes=([2], [0]))  # (bond_dim, 2, bond_dim) @ (bond_dim, 2)
        contracted = contracted.reshape(bond_dim, 4)  # (bond_dim, 4)
    else:
        contracted = np.tensordot(tensor1, tensor2, axes=([2], [0]))  # (bond_dim, 2, bond_dim) @ (bond_dim, 2, bond_dim)
        contracted = contracted.reshape(bond_dim, 4, -1)  # (bond_dim, 4, bond_dim)

    # Step 2: Apply the two-qubit gate
    if i == 0:
        evolved = np.dot(U_two, contracted)  # (4, 4) @ (4, bond_dim) -> (4, bond_dim)
    elif i == qubits - 2:
        evolved = np.dot(contracted, U_two)  # (bond_dim, 4) @ (4, 4) -> (bond_dim, 4)
    else:
        evolved = np.tensordot(U_two, contracted, axes=([1], [1])).transpose(0, 2, 1)  # (bond_dim, 4, bond_dim)

    # Step 3: Reshape and perform SVD
    if i == 0:
        U, S, Vh = np.linalg.svd(evolved, full_matrices=False)
        dim = min(len(S), max_bond_dim)
        U = U[:, :dim]
        S = S[:dim]
        Vh = Vh[:dim, :]
        new_tensor1 = U.reshape(2, 2, dim)[:, 0, :]  # (2, dim)
        new_tensor2 = (np.diag(S) @ Vh).reshape(dim, 2, -1)
    elif i == qubits - 2:
        # Reshape for last pair
        evolved_reshaped = evolved.reshape(bond_dim, 2, 2)  # (bond_dim, 2, 2)
        matrix = evolved_reshaped.reshape(bond_dim * 2, 2)  # (bond_dim * 2, 2)
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        dim = min(len(S), max_bond_dim)
        U = U[:, :dim]
        S = S[:dim]
        Vh = Vh[:dim, :]
        new_tensor1 = U.reshape(bond_dim, 2, dim)  # (bond_dim, 2, dim)
        new_tensor2 = (np.diag(S) @ Vh)  # (dim, 2)
    else:
        U, S, Vh = np.linalg.svd(evolved.reshape(bond_dim * 4, -1), full_matrices=False)
        dim = min(len(S), max_bond_dim)
        U = U[:, :dim]
        S = S[:dim]
        Vh = Vh[:dim, :]
        new_tensor1 = U.reshape(bond_dim, 4, dim)[:, :2, :]  # (bond_dim, 2, dim)
        new_tensor2 = (np.diag(S) @ Vh).reshape(dim, 2, -1)

    # Step 4: Update nodes
    nodes[i] = tn.Node(new_tensor1)
    nodes[i + 1] = tn.Node(new_tensor2)