import qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import numpy as np
import matplotlib.pyplot as plt
import time

# Version check for Qiskit Aer
print(f"Qiskit Aer Version: {qiskit_aer.__version__}")

start_time = time.time()
print("Kicking off—20-Qubit Test for Enhanced Quantum System with Kagome Lattice and Fibonacci Spiral...")

# Parameters
n_qubits = 20  # Matching your Mini-BEC
shots = 256
time_steps = np.logspace(-6, -3, 11)  # 1 µs to 1 ms
T2_avg = 260e-6  # Your achieved T2 with DD (260 µs)
fidelity_init = 0.99
regen_time_qubit = 1.25e-6  # 800 MHz optics (1 pulse every 1.25 µs)
regen_recovery_qubit = 0.999999
radiation_noise = 0.0005
gate_error = 0.003

# Fibonacci Spiral for qubit positioning (simplified as a 2D layout)
def fibonacci_spiral(n):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    theta = np.arange(n) * 2 * np.pi * phi  # Golden angle ~137.5° in radians
    r = np.sqrt(np.arange(n))  # Radius scales as sqrt(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# Kagome lattice coupling (simplified as a 2D triangular lattice with 3-point basis)
def kagome_coupling(qc, qreg, n_qubits):
    for i in range(0, n_qubits-2, 3):  # Every 3rd qubit for Kagome-like coupling
        if i+1 < n_qubits:
            qc.cx(qreg[i], qreg[i+1])
        if i+2 < n_qubits:
            qc.cx(qreg[i], qreg[i+2])

# Simulate gradient B-field (0.5 T at center, decreasing radially)
def b_field_gradient(n_qubits, r_max=1.0):
    B_0 = 0.5  # 0.5 T at center
    B_inner = 50e-6  # 50 µT for inner qubits (mu-metal shielding)
    r = np.sqrt(np.arange(n_qubits)) / np.sqrt(n_qubits - 1) * r_max  # Normalized radius
    B = np.where(r < 0.2, B_inner, B_0 * (1 - r / r_max))  # Inner qubits shielded
    return B

# Circuit setup
qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(2, 'c')  # 2 measured qubits
qc = QuantumCircuit(qreg, creg)

# Initialize with Hadamard gates
for q in range(n_qubits):
    qc.h(q)

# Apply Kagome-like coupling
kagome_coupling(qc, qreg, n_qubits)

# Dynamical Decoupling (simplified)
def add_dd_pulses(circuit, qubit):
    for _ in range(2):
        circuit.rx(np.pi / 2, qubit)
        circuit.rx(np.pi, qubit)
        circuit.rx(np.pi / 2, qubit)

# Simulate B-field and nuclear spin precession (¹⁵N at 0.344 MHz, ¹³C at 0.852 MHz)
B_field = b_field_gradient(n_qubits)
gamma_15N = 4.32e6  # Gyromagnetic ratio for ¹⁵N (Hz/T)
gamma_13C = 10.71e6  # Gyromagnetic ratio for ¹³C (Hz/T)
f_15N = gamma_15N * B_field / (2 * np.pi)  # Larmor frequencies for ¹⁵N
f_13C = gamma_13C * B_field / (2 * np.pi)  # Larmor frequencies for ¹³C
coupling_freq = (f_13C - f_15N)  # Coupling frequency for ¹³C-¹⁵N (0.508 MHz at center)

# Noise model (simplified with correction layer effect)
noise_model = NoiseModel()
print("Setting up noise model...")

# Apply thermal relaxation to 1-qubit gates
for q in range(n_qubits):
    thermal_error = thermal_relaxation_error(t1=1e-3, t2=T2_avg * 1.5, time=1e-6)  # 1.5x T2 due to correction layer
    noise_model.add_quantum_error(thermal_error, ['h', 'rx'], [q])

# Apply depolarizing error to 1-qubit gates
depolarizing_error_1q = depolarizing_error(gate_error, 1)
noise_model.add_all_qubit_quantum_error(depolarizing_error_1q, ['h', 'rx'])

# Apply depolarizing error to 2-qubit gates (cx)
depolarizing_error_2q = depolarizing_error(gate_error, 2)
noise_model.add_all_qubit_quantum_error(depolarizing_error_2q, ['cx'])

print("Noise model set up successfully.")

# Simulator setup
simulator = AerSimulator(noise_model=noise_model)

# Run simulation
for t in time_steps:
    try:
        qc_temp = qc.copy()
        cycles = np.floor(t / regen_time_qubit)  # 800 MHz optics
        for cycle in range(int(cycles)):
            for q in range(n_qubits):
                # Apply DD pulses
                add_dd_pulses(qc_temp, qreg[q])
                # Simulate nuclear spin precession (simplified as phase gates)
                phase_15N = 2 * np.pi * f_15N[q] * regen_time_qubit
                qc_temp.rz(phase_15N, qreg[q])
                # Simulate ¹³C side chain coupling (simplified as additional phase)
                phase_13C = 2 * np.pi * coupling_freq[q] * regen_time_qubit
                qc_temp.rz(phase_13C, qreg[q])
                # Reset with recovery probability
                qc_temp.reset(q)
                if np.random.random() > regen_recovery_qubit:
                    qc_temp.x(q)

        # Measure 2 qubits
        for i in range(0, 10, 5):
            qc_temp.measure(qreg[i], creg[i // 5])

        # Run simulation
        job = simulator.run(qc_temp, shots=shots)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.values())
        density_proxy = max_count / shots
        state_spread = len(counts) / 2**2  # 2 classical bits
        t_eff = t % regen_time_qubit
        noise_penalty = (1 - gate_error) ** (n_qubits * cycles * 2)
        h_gates = n_qubits
        x_gates = n_qubits * 3 * 2
        cx_gates = 25
        total_gates = h_gates + x_gates + cx_gates
        gate_noise_penalty = (1 - gate_error) ** total_gates
        fidelity_proxy = (fidelity_init * np.exp(-t_eff / T2_avg) *
                         (1 - radiation_noise) * (regen_recovery_qubit ** cycles) *
                         noise_penalty * gate_noise_penalty)
        print(f"t={t:.1e} s, Density: {density_proxy:.4f}, Spread: {state_spread:.4f}, Fidelity: {fidelity_proxy:.4f}")

        # Plot results
        plt.figure(figsize=(8, 3))
        dominant_state = max(counts, key=counts.get)
        plt.barh(f'Dominant State: {dominant_state}', counts[dominant_state], color='blue')
        plt.xlabel('Count')
        plt.title(f'20-Qubit Test at t={t:.1e} s (Fidelity {fidelity_proxy:.3f})')
        plt.xlim(0, shots * 1.1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'kagome_fibonacci_test_t{t:.1e}.png')
        plt.close()
    except Exception as e:
        print(f"Error at t={t:.1e} s: {str(e)}")
        continue

runtime = time.time() - start_time
print(f"Simulation completed in {runtime:.2f} seconds")