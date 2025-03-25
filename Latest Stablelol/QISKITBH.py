import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram, plot_state_city
import matplotlib.pyplot as plt

# Parameters
n_qubits = 20
shots = 1024
steps_to_save = [0, 50, 200, 500]

# Quantum Registers
qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(n_qubits, 'c')
qc = QuantumCircuit(qreg, creg)

# Initial State (Superfluid-like)
for i in range(n_qubits):
    qc.h(i)
    qc.rz(0.05, i)

# Time Evolution Stages
for step in range(1, max(steps_to_save) + 1):
    t = step * 0.00005

    # Gravitational Collapse (Dial back for balance)
    for i in range(n_qubits):
        qc.rz(-0.8 * np.exp(-((i - n_qubits / 2) / 4) ** 2), i)  # Softer focus

    # Pulse (Steps 50-52)
    if 50 <= step <= 52:
        qc.rx(-1.5 * np.cos(100 * t), range(n_qubits))  # Softer pulse

    # Quantum Foam (Reduce entanglement layers)
    if 50 <= step <= 52:
        for i in range(0, n_qubits - 1):
            qc.cz(i, i + 1)
        for i in range(0, n_qubits - 2, 2):
            qc.cz(i, i + 2)

    # Dark Matter Geometry
    rho_sim = step / 500
    for i in range(n_qubits):
        qc.rz(1.0 * rho_sim * (i / n_qubits), i)

    # Vortex (Wayne Pulse with balanced feedback)
    if step > 200:
        qc.rx(1.0, range(n_qubits))
        if step == 201:
            qc.reset(range(n_qubits // 2, n_qubits))
            qc.h(range(n_qubits // 2, n_qubits))
        for i in range(n_qubits):
            qc.rz(0.5 * (i - n_qubits / 2) / n_qubits, i)

    # Measure at key steps
    if step in steps_to_save:
        qc.measure(qreg, creg)
        break

# Local Sim (TITAN)
sim = AerSimulator(method='matrix_product_state', max_parallel_threads=72)
job = sim.run(qc, shots=shots)
counts = job.result().get_counts()
print(f"Raw counts: {counts}")
plot_histogram(counts)
plt.savefig('/home/wayne/Desktop/THEORY/qiskit_density_20.png')
plt.close()

# Cloud Run (IBM with Transpilation)
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
qc_transpiled = pm.run(qc)
sampler = Sampler(mode=backend)
job_cloud = sampler.run([qc_transpiled], shots=shots)
print(f"Job ID: {job_cloud.job_id()}")

# Metrics
density_proxy = max(counts.values()) / shots
state_spread = len(counts) / 2**n_qubits
print(f"Local density proxy: {density_proxy:.4f}, State spread: {state_spread:.4f}")