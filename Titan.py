import numpy as np
import tensornetwork as tn
from scipy.linalg import expm
from multiprocessing import Pool
import os


class QuantumSystem:
    def __init__(self, qubits):
        self.qubits = qubits
        self.dt = 0.01
        # Initialize all nodes with consistent shape (2, 2, 2)
        self.nodes = [tn.Node(np.random.random((2, 2, 2)) / np.sqrt(2)) for _ in range(qubits)]
        self.noise_scale = 0.0
        self.damping = 0.0
        self.kagome_coupling = 0

    def set_hamiltonian(self, Z_terms, X_terms, B_field, ZZ_coupling, XX_coupling, noise_scale, damping,
                        kagome_coupling):
        self.Z_terms = Z_terms
        self.X_terms = X_terms
        self.B_field = B_field
        self.ZZ_coupling = ZZ_coupling
        self.XX_coupling = XX_coupling
        self.noise_scale = noise_scale
        self.damping = damping
        self.kagome_coupling = kagome_coupling

    def initialize_bell_state(self, qubit_pair):
        q0, q1 = qubit_pair
        if q1 != q0 + 1:
            raise ValueError("Bell state qubits must be consecutive (e.g., [0, 1])")

        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bell_tensor = bell_state.reshape(2, 2)

        # Initialize q0 with (2, 2, 2) tensor
        q0_tensor = np.zeros((2, 2, 2))
        q0_tensor[:, 0, 0] = bell_tensor[:, 0]
        q0_tensor[:, 1, 1] = bell_tensor[:, 1]
        self.nodes[q0] = tn.Node(q0_tensor)

        # Initialize q1 with (2, 2, 2) tensor
        q1_tensor = np.zeros((2, 2, 2))
        q1_tensor[0, :, 0] = bell_tensor[0, :]
        q1_tensor[1, :, 1] = bell_tensor[1, :]
        self.nodes[q1] = tn.Node(q1_tensor)

        # Connect q0 and q1
        self.nodes[q0][1] ^ self.nodes[q1][0]

        # Connect remaining nodes
        for i in range(self.qubits - 1):
            if i < q0 or i >= q1:
                self.nodes[i][2] ^ self.nodes[i + 1][0]

    def hamiltonian(self, dt):
        H = []
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        for i in range(self.qubits):
            H_i = self.B_field * self.X_terms[i] * sigma_x + self.Z_terms[i] * sigma_z
            H_i += self.noise_scale * np.random.randn(2, 2) * dt + self.damping * np.eye(2)
            H.append(H_i)
        return H

    def compute_U(self, i, H):
        U = expm(-1j * self.dt * H[i])
        tensor = self.nodes[i].tensor
        if i == 0:
            evolved = np.tensordot(U, tensor, axes=([1], [0])).transpose(1, 0, 2)
        elif i == self.qubits - 1:
            evolved = np.tensordot(tensor, U, axes=([1], [0])).transpose(0, 2, 1)
        else:
            evolved = np.tensordot(U, tensor, axes=([1], [1])).transpose(1, 0, 2)
        return i, evolved

    def apply_two_qubit_interaction(self, q0, q1):
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        zz_term = self.ZZ_coupling * np.kron(sigma_z, sigma_z)
        xx_term = self.XX_coupling * np.kron(sigma_x, sigma_x)
        H_pair = zz_term + xx_term
        U = expm(-1j * self.dt * H_pair)

        tensor0 = self.nodes[q0].tensor
        tensor1 = self.nodes[q1].tensor

        # Safely disconnect the bond between q0 and q1
        edge_q0 = self.nodes[q0][1]
        edge_q1 = self.nodes[q1][0]
        if not edge_q0.is_dangling():
            try:
                tn.disconnect(edge_q0)
            except ValueError:
                pass

        # Contract over the bond
        combined = np.tensordot(tensor0, tensor1, axes=([1], [0]))  # Shape: (2, 2, 2, 2)
        combined_reshaped = combined.transpose(0, 2, 1, 3).reshape(4, 2, 2)
        evolved = np.tensordot(U, combined_reshaped, axes=([1], [0]))  # Shape: (4, 2, 2)
        evolved_reshaped = evolved.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)

        # Split using SVD
        matrix = evolved_reshaped.reshape(4, 4)  # (phys0 × left0, phys1 × right1)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        bond_dim = min(2, len(s))  # Ensure bond dimension ≤ 2
        u = u[:, :bond_dim]
        s = s[:bond_dim]
        vh = vh[:bond_dim, :]

        # Reconstruct tensors with bond dimension
        sqrt_s = np.sqrt(s)
        u_tensor = u.reshape(2, 2, bond_dim)
        vh_tensor = vh.reshape(bond_dim, 2, 2)
        evolved0 = np.zeros((2, 2, 2), dtype=complex)
        evolved1 = np.zeros((2, 2, 2), dtype=complex)
        for k in range(bond_dim):
            evolved0 += sqrt_s[k] * u_tensor[:, :, k].reshape(2, 2, 1) * np.ones((1, 1, 2))
            evolved1 += sqrt_s[k] * np.ones((2, 1, 1)) * vh_tensor[k, :, :].reshape(1, 2, 2)

        return evolved0, evolved1

    def evolve_one_step(self):
        H = self.hamiltonian(self.dt)

        # Step 1: Apply single-qubit evolution in parallel
        with Pool(processes=min(44, os.cpu_count())) as pool:
            results = pool.starmap(self.compute_U, [(i, H) for i in range(self.qubits)],
                                   chunksize=self.qubits // 44 or 1)

        new_nodes = [None] * self.qubits
        for i, evolved in sorted(results):
            new_nodes[i] = tn.Node(evolved)

        self.nodes = new_nodes

        # Step 2: Apply two-qubit interactions sequentially
        for i in range(self.qubits - 1):
            evolved0, evolved1 = self.apply_two_qubit_interaction(i, i + 1)
            self.nodes[i] = tn.Node(evolved0)
            self.nodes[i + 1] = tn.Node(evolved1)
            # Reconnect the nodes
            self.nodes[i][1] ^ self.nodes[i + 1][0]

    def apply_gate(self, gate_type, qubit):
        if gate_type == "X":
            gate = np.array([[0, 1], [1, 0]])
        elif gate_type == "Z":
            gate = np.array([[1, 0], [0, -1]])
        elif gate_type == "iY":
            gate = np.array([[0, -1j], [1j, 0]])
        else:
            raise ValueError("Unsupported gate type")

        U = gate
        tensor = self.nodes[qubit].tensor
        if qubit == 0:
            evolved = np.tensordot(U, tensor, axes=([1], [0])).transpose(1, 0, 2)
        elif qubit == self.qubits - 1:
            evolved = np.tensordot(tensor, U, axes=([1], [0])).transpose(0, 2, 1)
        else:
            evolved = np.tensordot(U, tensor, axes=([1], [1])).transpose(1, 0, 2)
        self.nodes[qubit].tensor = evolved

    def get_rho_two_qubits(self):
        q0, q1 = 0, 1

        # Extract tensors for q0 and q1
        tensor0 = self.nodes[q0].tensor  # Shape: (2, 2, 2)
        tensor1 = self.nodes[q1].tensor  # Shape: (2, 2, 2)

        # Contract q0 and q1: (phys0, left0, right0) with (left1, phys1, right1)
        combined = np.tensordot(tensor0, tensor1, axes=([1], [0]))  # Shape: (2, 2, 2, 2)
        state = combined.transpose(0, 2, 1, 3).reshape(4, 2, 2)  # Shape: (4, 2, 2)

        # Contract with neighbors
        if self.qubits > 2:
            # Left of q0 (q0 = 0, so skip)
            if q0 > 0:
                left_tensor = self.nodes[q0 - 1].tensor  # Shape: (2, 2, 2)
                state = np.tensordot(left_tensor, state, axes=([2], [1]))  # Shape: (2, 2, 4, 2)
                state = np.sum(state, axis=(0, 1))  # Shape: (4, 2)
            else:
                # Since q0 is 0, left bond is dummy (size 2), sum over it
                state = np.sum(state, axis=1)  # Shape: (4, 2)

            # Right of q1
            if q1 < self.qubits - 1:
                right_tensor = self.nodes[q1 + 1].tensor  # Shape: (2, 2, 2)
                state = np.tensordot(state, right_tensor, axes=([1], [0]))  # Shape: (4, 2, 2)
                state = np.sum(state, axis=(1, 2))  # Shape: (4,)
            else:
                # Since q1 is not the last qubit, but let's sum the right bond
                state = np.sum(state, axis=1)  # Shape: (4,)

        # Ensure state is (4,)
        state = state.flatten()  # Shape: (4,)

        # Compute density matrix ρ = |ψ⟩⟨ψ|
        rho = np.outer(state, np.conj(state))  # Shape: (4, 4)
        return rho

    def measure_two_qubit_coherence(self, q0, q1, operator):
        sigma_x = np.array([[0, 1], [1, 0]])
        if operator == "X":
            op = np.kron(sigma_x, sigma_x)
        rho = self.get_rho_two_qubits()
        return np.trace(np.dot(op, rho)).real

    def measure_entanglement(self, q0, q1):
        rho = self.get_rho_two_qubits()
        eigvals = np.linalg.eigvals(rho)
        eigvals = np.array([x for x in eigvals if x > 1e-10])  # Convert to NumPy array
        ent = -np.sum(eigvals * np.log2(eigvals + 1e-10))
        return ent if ent >= 0 else 0.0

    def measure_bell_basis(self, qubit_pair):
        q0, q1 = qubit_pair
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        hadamard_gate = np.kron(hadamard, np.eye(2))

        rho_two = self.get_rho_two_qubits()
        print(f"Density matrix for qubits {q0},{q1}: {rho_two.flatten()}")

        rho_after_cnot = np.dot(cnot, np.dot(rho_two, cnot.T.conj()))
        rho_after_hadamard = np.dot(hadamard_gate, np.dot(rho_after_cnot, hadamard_gate.T.conj()))

        probs = np.abs(np.diag(rho_after_hadamard))
        print(f"Probabilities: {probs}")

        max_idx = np.argmax(probs)
        if max_idx == 0:
            print("Detected |00> state, decoding as '11'")
            return "11"
        elif max_idx == 1:
            print("Detected |01> state, decoding as '10'")
            return "10"
        elif max_idx == 2:
            print("Detected |10> state, decoding as '01'")
            return "01"
        elif max_idx == 3:
            print("Detected |11> state, decoding as '00'")
            return "00"
        print(f"Unexpected dominant state (index {max_idx}), falling back to '00'")
        return "00"