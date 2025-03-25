from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = "/home/wayne/PycharmProjects/Theory/.venv/share/"
os.makedirs(output_dir, exist_ok=True)

# Parameters
n_qubits = 20
steps = 1000
dt = 0.01
shots = 1024

# Simulator
simulator = AerSimulator()


# Evolution function
def evolve_circuit(qc, t):
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(np.random.uniform(0, 2 * np.pi), q)
    for q in range(n_qubits - 1):
        qc.rzz(dt * t, q, q + 1)
    return qc


# Main sim
print("Starting Qiskit sim on T630...")
counts_history = []
for t in range(0, steps, 10):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc = evolve_circuit(qc, t * dt)
    qc.measure(range(n_qubits), range(n_qubits))

    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    counts_history.append(counts)

    if t % 100 == 0:
        probs = np.zeros(2 ** n_qubits)
        for state, count in counts.items():
            idx = int(state, 2)
            probs[idx] = count / shots
        plt.plot(probs, 'o-', label=f"Step {t}")
        plt.title(f"Qubit State Probabilities, Step {t}")
        plt.xlabel("State Index")
        plt.ylabel("Probability")
        plt.savefig(os.path.join(output_dir, f"qiskit_evolution_step_{t}.png"))
        plt.close()
        print(f"Saved {os.path.join(output_dir, f'qiskit_evolution_step_{t}.png')}")

print("Qiskit simulation complete!")