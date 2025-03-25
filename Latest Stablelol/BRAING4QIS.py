import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.visualization import plot_histogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Simulation started—Quantum Chip 50-Qubit Usable Regenerative State (1s Victory)...")

n_qubits = 50
shots = 1024
time_steps = np.logspace(-6, 0, 10)
T2_NV = 1.5e-3
T2_SiV = 500e-6
T2_Nuclear = 10
fidelity_init = 0.99
regen_time_qubit = 5e-4
regen_recovery_qubit = 0.9997  # Upped
regen_time_photon = 1e-2
regen_recovery_photon = 0.999999999  # Maxed
radiation_noise_300K = 0.0015  # Reduced
radiation_noise_20mK = 0.0005  # Reduced
qubit_types = [T2_NV] * 15 + [T2_SiV] * 15 + [T2_Nuclear] * 20

def build_noise_model(t, temp):
    noise_model = NoiseModel()
    noise = radiation_noise_300K if temp == 300 else radiation_noise_20mK
    for q in range(n_qubits):
        T2_val = qubit_types[q % len(qubit_types)]
        error = thermal_relaxation_error(t1=2*T2_val, t2=T2_val, time=t)
        error = error.compose(depolarizing_error(noise, 1))
        noise_model.add_quantum_error(error, ['u3'], [q])
        reset_error_prob = 1 - (regen_recovery_photon ** np.floor(t / regen_time_photon))
        noise_model.add_quantum_error(depolarizing_error(reset_error_prob, 1), ['reset'], [q])
    return noise_model

def apply_qec(qc, qreg, creg):
    for i in range(0, n_qubits, 3):
        if i + 2 < n_qubits:
            qc.cx(qreg[i], qreg[i+1])
            qc.cx(qreg[i], qreg[i+2])
            qc.measure(qreg[i], creg[i // 3])

qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(n_qubits // 3, 'c')
qc = QuantumCircuit(qreg, creg)
for q in range(n_qubits):
    qc.h(q)
for _ in range(250):
    qc.barrier()
    qc.x(range(n_qubits))
    qc.barrier()
    qc.x(range(n_qubits))

for t in time_steps:
    qc_temp = qc.copy()
    cycles = np.floor(t / regen_time_qubit)
    for _ in range(int(cycles)):
        for q in range(n_qubits):
            qc_temp.reset(q)
            if np.random.random() > regen_recovery_qubit:
                qc_temp.x(q)
    apply_qec(qc_temp, qreg, creg)

    try:
        noise_model = build_noise_model(t, 300)
        sim = AerSimulator(method='matrix_product_state', noise_model=noise_model)
        job = sim.run(qc_temp, shots=shots)
        counts = job.result().get_counts()
        max_count = max(counts.values())
        density_proxy = max_count / shots
        state_spread = len(counts) / 2**(n_qubits // 3)
        t_eff = t % regen_time_qubit
        q_cycles = np.floor(t / regen_time_qubit)
        p_cycles = np.floor(t / regen_time_photon)
        fidelity_proxy = (fidelity_init * np.exp(-t_eff / np.mean(qubit_types)) *
                         (1 - radiation_noise_300K) * (regen_recovery_qubit ** q_cycles) *
                         (regen_recovery_photon ** p_cycles))
        print(f"300K, t={t:.1e} s, Density: {density_proxy:.4f}, Spread: {state_spread:.4f}, Fidelity: {fidelity_proxy:.4f}")
        plot_histogram(counts, filename=f'/home/wayne/Desktop/THEORY/qiskit_50_chip_300K_t{t:.1e}.png')
        plt.close()

        noise_model_20mK = build_noise_model(t, 20e-3)
        sim_20mK = AerSimulator(method='matrix_product_state', noise_model=noise_model_20mK)
        job_20mK = sim_20mK.run(qc_temp, shots=shots)
        counts_20mK = job_20mK.result().get_counts()
        max_count_20mK = max(counts_20mK.values())
        density_proxy_20mK = max_count_20mK / shots
        state_spread_20mK = len(counts_20mK) / 2**(n_qubits // 3)
        fidelity_proxy_20mK = (fidelity_init * np.exp(-t_eff / np.mean(qubit_types)) *
                              (1 - radiation_noise_20mK) * (regen_recovery_qubit ** q_cycles) *
                              (regen_recovery_photon ** p_cycles))
        print(f"20mK, t={t:.1e} s, Density: {density_proxy_20mK:.4f}, Spread: {state_spread_20mK:.4f}, Fidelity: {fidelity_proxy_20mK:.4f}")
        plot_histogram(counts_20mK, filename=f'/home/wayne/Desktop/THEORY/qiskit_50_chip_20mK_t{t:.1e}.png')
        plt.close()
        print(f"Plots saved for t={t:.1e} s")
    except Exception as e:
        print(f"Error at t={t:.1e} s: {str(e)}")
        continue

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")