from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Kicking off—127-Qubit IBM Quantum Test for UFT (857 Hz Vortex Boost)...")

# Parameters
n_qubits = 127
shots = 256
time_steps = np.logspace(-6, -3, 11)  # 1 µs to 1 ms
T2_avg = 260e-6  # Updated to your achieved T2 with DD
fidelity_init = 0.99
regen_time_qubit = 1.166e-3  # 1 / 857 Hz ≈ 1.166 ms
regen_recovery_qubit = 0.999999
radiation_noise = 0.0005
gate_error = 0.003

# Backend setup
service = QiskitRuntimeService(channel="ibm_quantum")
backends = service.backends(min_num_qubits=127, operational=True, simulator=False)
if not backends:
    raise Exception("No 127-qubit backends available!")
backend = backends[0]
print(f"Running on: {backend.name}")

coupling_map = backend.configuration().coupling_map
print("Coupling Map:", coupling_map)

# Circuit setup
qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(2, 'c')  # Reduced to 2 classical bits
qc = QuantumCircuit(qreg, creg)

# Initialize
for q in range(n_qubits):
    qc.h(q)

# Reduced X-gate cycles (3 instead of 5)
for _ in range(3):
    qc.barrier()
    qc.x(range(n_qubits))
    qc.barrier()
    qc.x(range(n_qubits))

# Create a 2D cluster state across the 10 measured qubits
for i in range(0, 50, 5):  # 5 groups of 10 qubits
    for j in range(i, i+5):  # First row: 0→1→2→3→4
        if j+1 < n_qubits and (j, j+1) in coupling_map:
            qc.cx(qreg[j], qreg[j+1])
    for j in range(i+5, i+10):  # Second row: 5→6→7→8→9
        if j+1 < n_qubits and (j, j+1) in coupling_map:
            qc.cx(qreg[j], qreg[j+1])
    # Connect rows: 0→5, 1→6, etc.
    for j in range(5):
        if (i+j, i+j+5) in coupling_map:
            qc.cx(qreg[i+j], qreg[i+j+5])

# Measure only 2 qubits
for i in range(0, 10, 5):
    qc.measure(qreg[i], creg[i // 5])

# Dynamical Decoupling
def add_dd_pulses(circuit, qubit):
    circuit.rx(np.pi / 2, qubit)
    circuit.rx(np.pi, qubit)
    circuit.rx(np.pi / 2, qubit)

# Log Job IDs
with open('/home/wayne/Desktop/ibm_127qubit_job_ids.txt', 'a') as f:
    f.write(f"Run started at {time.ctime()}\n")

# Run in batches
successful_jobs = 0
total_jobs = len(time_steps)
for t_batch in [time_steps[i:i+5] for i in range(0, len(time_steps), 5)]:
    for t in t_batch:
        for attempt in range(3):
            try:
                qc_temp = qc.copy()
                cycles = np.floor(t / regen_time_qubit)
                for cycle in range(int(cycles)):
                    for batch in range(0, n_qubits, 32):
                        for q in range(batch, min(batch + 32, n_qubits)):
                            add_dd_pulses(qc_temp, qreg[q])
                            qc_temp.reset(q)
                            if np.random.random() > regen_recovery_qubit:
                                qc_temp.x(q)

                initial_layout = list(range(n_qubits))
                qc_temp = transpile(qc_temp, backend=backend, optimization_level=2,
                                    basis_gates=['x', 'sx', 'rz', 'cx'], initial_layout=initial_layout)

                sampler = Sampler(mode=backend)
                job = sampler.run([qc_temp], shots=shots)
                job_id = job.job_id()
                print(f"Job submitted for t={t:.1e} s, Job ID: {job_id}")
                with open('/home/wayne/Desktop/ibm_127qubit_job_ids.txt', 'a') as f:
                    f.write(f"t={t:.1e} s, Job ID: {job_id}\n")

                result = job.result(timeout=1800)  # 30-minute timeout
                counts = result[0].data.c.get_counts()
                max_count = max(counts.values())
                density_proxy = max_count / shots
                state_spread = len(counts) / 2**2  # Updated for 2 classical bits
                t_eff = t % regen_time_qubit
                noise_penalty = (1 - gate_error) ** (n_qubits * cycles * 2)
                # Add gate error penalty for H, X, CX gates
                h_gates = n_qubits
                x_gates = n_qubits * 3 * 2
                cx_gates = 25
                total_gates = h_gates + x_gates + cx_gates
                gate_error_per_gate = 0.003
                gate_noise_penalty = (1 - gate_error_per_gate) ** total_gates
                fidelity_proxy = (fidelity_init * np.exp(-t_eff / T2_avg) *
                                 (1 - radiation_noise) * (regen_recovery_qubit ** cycles) *
                                 noise_penalty * gate_noise_penalty)
                print(f"t={t:.1e} s, Density: {density_proxy:.4f}, Spread: {state_spread:.4f}, Fidelity: {fidelity_proxy:.4f}")

                plt.figure(figsize=(8, 3))
                dominant_state = max(counts, key=counts.get)
                plt.barh(f'Dominant State: {dominant_state}', counts[dominant_state], color='blue')
                plt.xlabel('Count')
                plt.title(f'127-Qubit IBM Run at t={t:.1e} s (Fidelity {fidelity_proxy:.3f})')
                plt.xlim(0, shots * 1.1)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'/home/wayne/Desktop/ibm_127qubit_857hz_t{t:.1e}.png')
                plt.close()
                successful_jobs += 1
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed for t={t:.1e} s: {str(e)}")
                if attempt == 2:
                    print(f"Skipping t={t:.1e} s after 3 failures")
                    continue

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")
print(f"Completed {successful_jobs}/{total_jobs} jobs successfully")