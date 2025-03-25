import cirq
import qsimcirq
import numpy as np
import matplotlib.pyplot as plt
import time

print(f"Cirq Version: {cirq.__version__}")
start_time = time.time()
print("Kicking off—Grover's Search (Simplified) with Cirq qsim...")

# Parameters
n_qubits = 30  # Reduced to 30 qubits to test
shots = 64

# Qubits
qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
ancilla = [cirq.LineQubit(i) for i in range(n_qubits, 2 * n_qubits - 2)]


# Custom multi-controlled Toffoli decomposition
def multi_controlled_toffoli(controls, target, ancilla):
    n_controls = len(controls)
    if n_controls <= 2:
        return [cirq.TOFFOLI(*controls, target)]

    # Use ancilla to build a tree of AND operations
    operations = []
    current_ancilla_idx = 0

    # Layer 1: Pair controls into groups
    layer1_results = []
    for i in range(0, n_controls - 1, 2):
        if i + 1 < n_controls:
            operations.append(cirq.TOFFOLI(controls[i], controls[i + 1], ancilla[current_ancilla_idx]))
            layer1_results.append(ancilla[current_ancilla_idx])
            current_ancilla_idx += 1
        else:
            layer1_results.append(controls[i])  # Odd control passes through

    # Subsequent layers: Pair results until we reach the target
    while len(layer1_results) > 1:
        next_layer = []
        for i in range(0, len(layer1_results) - 1, 2):
            if i + 1 < len(layer1_results):
                operations.append(cirq.TOFFOLI(layer1_results[i], layer1_results[i + 1], ancilla[current_ancilla_idx]))
                next_layer.append(ancilla[current_ancilla_idx])
                current_ancilla_idx += 1
            else:
                next_layer.append(layer1_results[i])
        layer1_results = next_layer

    # Final gate to target
    if len(layer1_results) == 1:
        operations.append(cirq.CNOT(layer1_results[0], target))
    else:
        operations.append(cirq.TOFFOLI(layer1_results[0], layer1_results[1], target))

    # Uncomputation: Reverse the operations
    operations.extend(reversed(operations[:-1]))

    return operations


# Circuit
circuit = cirq.Circuit()

# Initial state: Hadamard on all qubits
circuit.append(cirq.H(q) for q in qubits)

# Grover iterations (capped at 10 to reduce depth)
optimal_iterations = int(np.pi / 4 * np.sqrt(float(2 ** n_qubits)) + 0.5)
iterations = min(10, optimal_iterations)
print(f"Total iterations (capped): {iterations}")

for i in range(iterations):
    # Oracle (mark |11...1>)
    circuit.append(cirq.H(qubits[-1]))
    circuit.append(cirq.X(qubits[-1]))
    circuit.append(multi_controlled_toffoli(qubits[:-1], qubits[-1], ancilla))
    circuit.append(cirq.X(qubits[-1]))
    circuit.append(cirq.H(qubits[-1]))

    # Diffusion
    circuit.append(cirq.H(q) for q in qubits)
    circuit.append(cirq.X(q) for q in qubits)
    circuit.append(cirq.H(qubits[-1]))
    circuit.append(multi_controlled_toffoli(qubits[:-1], qubits[-1], ancilla))
    circuit.append(cirq.H(qubits[-1]))
    circuit.append(cirq.X(q) for q in qubits)
    circuit.append(cirq.H(q) for q in qubits)

    if i % max(1, iterations // 10) == 0:
        print(f"Iteration {i + 1}/{iterations} ({(i + 1) / iterations * 100:.1f}%)")

# Measurement
circuit.append(cirq.measure(*qubits, key='result'))

# Simulator (qsim)
simulator = qsimcirq.QSimSimulator()
print("Running with qsim...")

# Run
result = simulator.run(circuit, repetitions=shots)
counts = result.histogram(key='result', fold_func=lambda bits: ''.join(map(str, bits)))

# Results
max_count = max(counts.values())
density_proxy = max_count / shots
state_spread = len(counts) / 2 ** n_qubits
print(f"Grover's Search Results: Density: {density_proxy:.4f}, Spread: {state_spread:.4f}")

plt.figure(figsize=(8, 3))
dominant_state = max(counts, key=counts.get)
plt.barh(f'Dominant State: {dominant_state}', counts[dominant_state], color='blue')
plt.xlabel('Count')
plt.title(f'{n_qubits}-Qubit Grover\'s Search (Density {density_proxy:.3f})')
plt.xlim(0, shots * 1.1)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'grover_search_{n_qubits}q_mps.png')
plt.close()

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")