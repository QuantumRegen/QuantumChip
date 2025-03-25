from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import os

# Plot directory
plot_dir = "/home/wayne/PycharmProjects/Theory/plots"
os.makedirs(plot_dir, exist_ok=True)

# Simulator backend
backend = AerSimulator()
print(f"Using backend: AerSimulator")

# Parameters
n_qubits = 10
shots = 512
time_steps = 11
t2 = 130e-6
freq_laser = 872e6
dt = np.logspace(-6, -3, time_steps)

# Dummy coupling map for simulation (linear chain)
coupling_map = [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]
print(f"Coupling Map (simulated): {coupling_map[:10]}... (showing first 10 pairs)")


def add_dd_pulses(circuit, qubit):
    for _ in range(2):  # Double DD
        circuit.rx(np.pi / 2, qubit)
        circuit.rx(np.pi, qubit)
        circuit.rx(np.pi / 2, qubit)


def add_nuclear_coupling(circuit, nv_qubit, nuclear_qubit, t_step):
    if [nv_qubit, nuclear_qubit] in coupling_map or [nuclear_qubit, nv_qubit] in coupling_map:
        print(f"Adding SIL RZ on qubit {nuclear_qubit}")
        circuit.rz(np.pi / 2, nuclear_qubit)  # Fixed angle
        if [nv_qubit, nuclear_qubit] in coupling_map:
            print(f"Adding SIL CX between qubits {nv_qubit} and {nuclear_qubit}")
            circuit.barrier(nv_qubit, nuclear_qubit)
            circuit.cx(nv_qubit, nuclear_qubit)
            circuit.barrier(nv_qubit, nuclear_qubit)
        elif [nuclear_qubit, nv_qubit] in coupling_map:
            print(f"Adding SIL CX between qubits {nuclear_qubit} and {nv_qubit} (reversed)")
            circuit.barrier(nuclear_qubit, nv_qubit)
            circuit.cx(nuclear_qubit, nv_qubit)
            circuit.barrier(nuclear_qubit, nv_qubit)


# Log file for CX gates
log_file = "/home/wayne/PycharmProjects/Theory/cx_log.txt"
with open(log_file, 'w') as f:
    f.write("CX Gate Log\n")

# Run for each time step
for i, t_step in enumerate(dt):
    circ = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        circ.h(q)
        circ.rx(np.pi, q)  # Full flip to |1>

    # Custom CX pairs based on coupling map
    cx_pairs = [(1, 0), (2, 1), (3, 2), (4, 3), (4, 5), (6, 5), (6, 7), (7, 8), (8, 9)]
    for control, target in cx_pairs:
        if control < n_qubits and target < n_qubits:
            print(f"Adding CX between qubits {control} and {target}")
            circ.barrier(control, target)
            circ.cx(control, target)
            circ.barrier(control, target)

    for q in range(n_qubits // 2):
        add_dd_pulses(circ, q)
        add_nuclear_coupling(circ, q, q + 1, t_step)

    for q in range(n_qubits):
        circ.measure(q, q)

    # Debug: Print circuit stats
    print(f"Circuit stats for t={t_step:.1e} s:")
    print(f"Depth: {circ.depth()}")
    print(f"Gate counts: {circ.count_ops()}")

    # Log pre-transpile CX gates
    cx_gates = [(circ.find_bit(instr.qubits[0])[0], circ.find_bit(instr.qubits[1])[0])
                for instr in circ.data if instr.operation.name == 'cx']
    with open(log_file, 'a') as f:
        f.write(f"t={t_step:.1e} s Pre-transpile CX: {cx_gates}\n")

    # Transpile
    initial_layout = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    circ_transpiled = transpile(
        circ,
        backend=backend,
        optimization_level=1,
        initial_layout=initial_layout,
        coupling_map=coupling_map
    )

    # Log post-transpile CX gates
    cx_gates_post = [(circ_transpiled.find_bit(instr.qubits[0])[0], circ_transpiled.find_bit(instr.qubits[1])[0])
                     for instr in circ_transpiled.data if instr.operation.name == 'cx']
    with open(log_file, 'a') as f:
        f.write(f"t={t_step:.1e} s Post-transpile CX: {cx_gates_post}\n")

    # Run on simulator
    job = backend.run(circ_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Filter to top 20 states for clarity
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])

    dominant_state = max(counts, key=counts.get)
    print(f"Dominant state for t={t_step:.1e} s: {dominant_state} with {counts[dominant_state]} counts")

    # Plot histogram with spread-out settings
    plt.rcParams['figure.figsize'] = [15, 6]
    plot_path = f"{plot_dir}/counts_t_{t_step:.1e}s_sim.png"
    plot_histogram(sorted_counts, title=f"Counts for t={t_step:.1e} s")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    print(f"Histogram saved: {plot_path}")

print(f"Simulated T₂ target: {t2 * 1e6} µs")
print(f"CX gate log saved: {log_file}")