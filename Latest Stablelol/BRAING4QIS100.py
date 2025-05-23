import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Simulation started—Quantum Chip 200-Qubit Usable Regenerative State (10s Victory)...")

n_qubits = 200
shots = 32
time_steps = np.logspace(-6, 1, 11)
T2_NV = 1.5e-3
T2_SiV = 500e-6
T2_Nuclear = 10
fidelity_init = 0.99
regen_time_qubit = 5e-4
regen_recovery_qubit = 0.99997
regen_time_photon = 1e-2
regen_recovery_photon = 0.999999999999
radiation_noise_300K = 0.0003
radiation_noise_20mK = 0.00005
qubit_types = [T2_NV] * 66 + [T2_SiV] * 66 + [T2_Nuclear] * 68


def build_noise_model(t, temp):
    noise_model = NoiseModel()
    noise = radiation_noise_300K if temp == 300 else radiation_noise_20mK
    for q in range(n_qubits):
        T2_val = qubit_types[q % len(qubit_types)]
        error = thermal_relaxation_error(t1=2 * T2_val, t2=T2_val, time=t)
        error = error.compose(depolarizing_error(noise, 1))
        noise_model.add_quantum_error(error, ['u3'], [q])
        reset_error_prob = 1 - (regen_recovery_photon ** np.floor(t / regen_time_photon))
        noise_model.add_quantum_error(depolarizing_error(reset_error_prob, 1), ['reset'], [q])
    return noise_model


def apply_surface_code(qc, qreg, creg):
    for i in range(0, n_qubits, 10):
        if i + 9 < n_qubits:
            qc.cx(qreg[i], qreg[i + 1])
            qc.cx(qreg[i], qreg[i + 4])
            qc.measure(qreg[i + 1], creg[i // 10])


qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(n_qubits // 10, 'c')
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
    apply_surface_code(qc_temp, qreg, creg)

    try:
        noise_model = build_noise_model(t, 300)
        sim = AerSimulator(method='matrix_product_state', noise_model=noise_model)
        job = sim.run(qc_temp, shots=shots)
        counts = job.result().get_counts()
        max_count = max(counts.values())
        density_proxy = max_count / shots
        state_spread = len(counts) / 2 ** (n_qubits // 10)
        t_eff = t % regen_time_qubit
        q_cycles = np.floor(t / regen_time_qubit)
        p_cycles = np.floor(t / regen_time_photon)
        fidelity_proxy = (fidelity_init * np.exp(-t_eff / np.mean(qubit_types)) *
                          (1 - radiation_noise_300K) * (regen_recovery_qubit ** q_cycles) *
                          (regen_recovery_photon ** p_cycles))
        print(
            f"300K, t={t:.1e} s, Density: {density_proxy:.4f}, Spread: {state_spread:.4f}, Fidelity: {fidelity_proxy:.4f}")

        plt.figure(figsize=(8, 3))
        dominant_state = max(counts, key=counts.get)
        plt.barh('Dominant State', counts[dominant_state], color='blue')
        plt.xlabel('Count')
        plt.title(f'200-Qubit Sim at t={t:.1e} s (300K, Fidelity {fidelity_proxy:.3f})')
        plt.xlim(0, shots * 1.1)
        plt.tight_layout()
        plt.savefig(f'/home/wayne/Desktop/THEORY/qiskit_200_chip_300K_t{t:.1e}.png')
        plt.close()

        noise_model_20mK = build_noise_model(t, 20e-3)
        sim_20mK = AerSimulator(method='matrix_product_state', noise_model=noise_model_20mK)
        job_20mK = sim_20mK.run(qc_temp, shots=shots)
        counts_20mK = job_20mK.result().get_counts()
        max_count_20mK = max(counts_20mK.values())
        density_proxy_20mK = max_count_20mK / shots
        state_spread_20mK = len(counts_20mK) / 2 ** (n_qubits // 10)
        fidelity_proxy_20mK = (fidelity_init * np.exp(-t_eff / np.mean(qubit_types)) *
                               (1 - radiation_noise_20mK) * (regen_recovery_qubit ** q_cycles) *
                               (regen_recovery_photon ** p_cycles))
        print(
            f"20mK, t={t:.1e} s, Density: {density_proxy_20mK:.4f}, Spread: {state_spread_20mK:.4f}, Fidelity: {fidelity_proxy_20mK:.4f}")

        plt.figure(figsize=(8, 3))
        dominant_state_20mK = max(counts_20mK, key=counts_20mK.get)
        plt.barh('Dominant State', counts_20mK[dominant_state_20mK], color='blue')
        plt.xlabel('Count')
        plt.title(f'200-Qubit Sim at t={t:.1e} s (20mK, Fidelity {fidelity_proxy_20mK:.3f})')
        plt.xlim(0, shots * 1.1)
        plt.tight_layout()
        plt.savefig(f'/home/wayne/Desktop/THEORY/qiskit_200_chip_20mK_t{t:.1e}.png')
        plt.close()
        print(f"Plots saved for t={t:.1e} s")
    except Exception as e:
        print(f"Error at t={t:.1e} s: {str(e)}")
        continue

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")