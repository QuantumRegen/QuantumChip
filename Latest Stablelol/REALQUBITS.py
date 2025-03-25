from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Kicking off—5-Qubit IBM Quantum Test for UFT...")

n_qubits = 5
shots = 32
time_steps = np.logspace(-6, -3, 5)
T2_avg = 100e-6
fidelity_init = 0.99
regen_time_qubit = 5e-5
regen_recovery_qubit = 0.9999
radiation_noise = 0.001

qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(n_qubits, 'c')
qc = QuantumCircuit(qreg, creg)

for q in range(n_qubits):
    qc.h(q)

for _ in range(10):
    qc.barrier()
    qc.x(range(n_qubits))
    qc.barrier()
    qc.x(range(n_qubits))

qc.cx(0, 1)
qc.cx(0, 2)
qc.measure([1, 2], [0, 1])

service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
print(f"Running on: {backend.name}")

for t in time_steps:
    qc_temp = qc.copy()
    cycles = np.floor(t / regen_time_qubit)
    for _ in range(int(cycles)):
        for q in range(n_qubits):
            qc_temp.reset(q)
            if np.random.random() > regen_recovery_qubit:
                qc_temp.x(q)

    # Transpile for the backend
    qc_temp = transpile(qc_temp, backend=backend)

    sampler = Sampler(mode=backend)
    job = sampler.run([qc_temp], shots=shots)
    print(f"Job submitted for t={t:.1e} s, Job ID: {job.job_id()}")

    result = job.result()
    counts = result[0].data.c.get_counts()
    max_count = max(counts.values())
    density_proxy = max_count / shots
    state_spread = len(counts) / 2**n_qubits
    t_eff = t % regen_time_qubit
    fidelity_proxy = (fidelity_init * np.exp(-t_eff / T2_avg) *
                     (1 - radiation_noise) * (regen_recovery_qubit ** cycles))
    print(f"t={t:.1e} s, Density: {density_proxy:.4f}, Spread: {state_spread:.4f}, Fidelity: {fidelity_proxy:.4f}")

    plt.figure(figsize=(8, 3))
    dominant_state = max(counts, key=counts.get)
    plt.barh('Dominant State', counts[dominant_state], color='blue')
    plt.xlabel('Count')
    plt.title(f'5-Qubit IBM Run at t={t:.1e} s (Fidelity {fidelity_proxy:.3f})')
    plt.xlim(0, shots * 1.1)
    plt.tight_layout()
    plt.savefig(f'ibm_5qubit_t{t:.1e}.png')
    plt.close()

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")