import numpy as np
import matplotlib.pyplot as plt
import time

# Simulation parameters
GRID_SIZE = 50
L = 1000  # Grid extent in arbitrary units
x = np.linspace(-L/2, L/2, GRID_SIZE)
y = np.linspace(-L/2, L/2, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# NV and Si flow functions
NV_SPEED = 1.5  # NV center influence rate
SI_SPEED = 0.5  # Silicon qubit influence rate
def nv_flow(x):
    return NV_SPEED * np.tanh(x / 100)
def si_flow(x):
    return SI_SPEED * np.tanh(x / 50)

NV = nv_flow(X)
SI = si_flow(X)
handoff = 0.5 * (NV + SI)  # Handoff interaction term
sink = np.exp(-np.abs(X) / 50) * np.cos(2 * np.pi * (X**2 + Y**2) / L)  # Sink term

# Simulation settings
TOTAL_TIME = 30 * 60  # 30 minutes in seconds
STEP_INTERVAL = 60    # 1 minute per step
NUM_STEPS = TOTAL_TIME // STEP_INTERVAL  # 30 steps
NUM_RUNS = 10  # 10 independent runs

# Arrays for storing metrics
coherence_peaks = np.zeros((NUM_RUNS, NUM_STEPS + 1))
entropies = np.zeros((NUM_RUNS, NUM_STEPS + 1))
depths = np.zeros((NUM_RUNS, NUM_STEPS + 1))
uptimes = np.zeros(NUM_RUNS)

start_time = time.time()
output_files = []

print("Starting 30-minute NV-Si grid-based simulation with 10 runs...")

for run in range(NUM_RUNS):
    np.random.seed(42 + run * 1000)  # Unique seed per run
    uptime = 0

    for step in range(NUM_STEPS + 1):
        # Introduce noise
        nv_noise = 0.1 * np.random.normal(size=(GRID_SIZE, GRID_SIZE))
        si_noise = 0.05 * np.random.normal(size=(GRID_SIZE, GRID_SIZE))
        nv_state = nv_noise * sink
        si_state = si_noise * sink
        phase_shift = nv_state * np.cos(handoff) + si_state * np.sin(handoff)

        # Coherence calculation (simplified correlation)
        coherence_raw = np.correlate(phase_shift.flatten(), nv_state.flatten(), mode='same')
        max_abs_raw = np.max(np.abs(coherence_raw))
        if max_abs_raw == 0:
            peak = 0.15
        else:
            coherence = np.abs(coherence_raw) * 0.15 / max_abs_raw
            peak = np.max(coherence)
        coherence_peaks[run, step] = peak

        # Entropy calculation
        hist, bins = np.histogram(phase_shift.flatten(), bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        entropy = -np.sum(hist * np.log2(hist)) * np.diff(bins)[0]
        entropies[run, step] = entropy

        # Depth (standard deviation)
        depth = np.std(phase_shift)
        depths[run, step] = depth

        # Uptime: time until coherence drops below threshold
        if peak >= 0.10:
            uptime = step

        # Visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, phase_shift, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_title(f'NV-Si Phase Shift - Run {run+1}, Step {step} min')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Phase Shift')
        filename = f'nvsi_grid_run{run+1}_step{step}min.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        output_files.append(filename)
        print(f"Saved {filename}")

        # Control simulation pace
        elapsed = time.time() - start_time
        if step < NUM_STEPS:
            sleep_time = (step + 1) * STEP_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    uptimes[run] = uptime

# Compute averages
avg_coherence = np.mean(coherence_peaks, axis=0)
avg_entropy = np.mean(entropies, axis=0)
avg_depth = np.mean(depths, axis=0)
avg_uptime = np.mean(uptimes)

# Summary plots
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(range(NUM_STEPS + 1), avg_coherence, 'b-')
plt.title('Average Coherence Peak Over Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Coherence Peak')

plt.subplot(3, 1, 2)
plt.plot(range(NUM_STEPS + 1), avg_entropy, 'r-')
plt.title('Average Entropy Over Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Entropy')

plt.subplot(3, 1, 3)
plt.plot(range(NUM_STEPS + 1), avg_depth, 'g-')
plt.title('Average Depth Over Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Depth (std dev)')

plt.tight_layout()
plt.savefig('nvsi_grid_summary.png', dpi=300)
plt.close()

print(f"Simulation complete! Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
print(f"Average uptime: {avg_uptime:.1f} minutes")
print("Output files:", output_files)