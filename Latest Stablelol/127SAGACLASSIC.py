from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Kicking off—127-Qubit Classical Sim for UFT (857 Hz Vortex Boost)...")

n_qubits = 127
shots = 512
time_steps = np.logspace(-6, -3, 11)  # 1 µs to 1 ms
T2_avg = 100e-6  # 100 µs
fidelity_init = 0.99
regen_time_qubit = 1.166e-3  # 1 / 857 Hz ≈ 1.166 ms
regen_recovery_qubit = 0.999999
radiation_noise = 0.0005

qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(n_qubits // 5, 'c')
qc = QuantumCircuit(qreg, creg)

for q in range(n_qubits):
    qc.h(q)

for _ in range(5):
    qc.barrier()
    qc.x(range(n_qubits))
    qc.barrier()
    qc.x(range(n_qubits))

# Sim backend—MPS for memory
backend = AerSimulator(method='matrix_product_state')  # Less memory, still juicy
print("Running on: AerSimulator (matrix_product_state)")

# Fake coupling map
coupling_map = [[i, i+1] for i in range(n_qubits-1)]
print("Coupling Map (sim):", coupling_map[:10], "...")

for i in range(0, n_qubits, 5):
    if i + 4 < n_qubits:
        if i+1 < n_qubits:
            qc.cx(qreg[i], qreg[i+1])
        if i+2 < n_qubits:
            qc.cx(qreg[i], qreg[i+2])
        qc.measure(qreg[i+1], creg[i // 5])

for t in time_steps:
    try:
        qc_temp = qc.copy()
        cycles = np.floor(t / regen_time_qubit)  # 857 Hz resets
        for _ in range(int(cycles)):
            for q in range(n_qubits):
                qc_temp.reset(q)
                if np.random.random() > regen_recovery_qubit:
                    qc_temp.x(q)

        qc_temp = transpile(qc_temp, backend=backend, optimization_level=3, coupling_map=coupling_map)

        result = backend.run(qc_temp, shots=shots).result()
        counts = result.get_counts(qc_temp)
        max_count = max(counts.values())
        density_proxy = max_count / shots
        state_spread = len(counts) / 2**(n_qubits // 5)
        t_eff = t % regen_time_qubit
        gate_error = 0.003
        noise_penalty = (1 - gate_error) ** (n_qubits * cycles * 2)
        fidelity_proxy = (fidelity_init * np.exp(-t_eff / T2_avg) *
                         (1 - radiation_noise) * (regen_recovery_qubit ** cycles) * noise_penalty)
        print(f"t={t:.1e} s, Density: {density_proxy:.4f}, Spread: {state_spread:.4f}, Fidelity: {fidelity_proxy:.4f}")

        plt.figure(figsize=(8, 3))
        dominant_state = max(counts, key=counts.get)
        plt.barh(f'Dominant State: {dominant_state}', counts[dominant_state], color='blue')
        plt.xlabel('Count')
        plt.title(f'127-Qubit Sim Run at t={t:.1e} s (Fidelity {fidelity_proxy:.3f})')
        plt.xlim(0, shots * 1.1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/home/wayne/Desktop/ibm_127qubit_857hz_sim_t{t:.1e}.png')
        plt.close()
    except Exception as e:
        print(f"Error at t={t:.1e} s: {str(e)}")
        continue

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")