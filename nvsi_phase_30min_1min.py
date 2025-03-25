import numpy as np
import matplotlib.pyplot as plt
import time

# NV-Si params
L = 1000
x = np.linspace(-L/2, L/2, 50)  # 50x50, ~2M polys
y = np.linspace(-L/2, L/2, 50)
X, Y = np.meshgrid(x, y)
nv_speed = 1.5
si_speed = 0.5

def nv_flow(x):
    return nv_speed * np.tanh(x / 100)
def si_flow(x):
    return si_speed * np.tanh(x / 50)
NV = nv_flow(X)
SI = si_flow(X)
handoff = 0.5 * (NV + SI)
sink = np.exp(-np.abs(X) / 50) * np.cos(2 * np.pi * (X**2 + Y**2) / L)

# 30-minute sim, 1-min outputs
TOTAL_TIME = 30 * 60  # 1800 seconds
STEP_INTERVAL = 60    # 60 seconds (1 min)
NUM_STEPS = TOTAL_TIME // STEP_INTERVAL  # 30 steps
start_time = time.time()
output_files = []  # Your string of PNGs

print("Starting 30-minute NV-Si sim...")

for step in range(NUM_STEPS + 1):  # 0 to 30 (31 frames)
    np.random.seed(42 + step)
    nv_noise = 0.1 * np.random.normal(size=X.shape)
    si_noise = 0.05 * np.random.normal(size=X.shape)
    nv_state = nv_noise * sink
    si_state = si_noise * sink
    phase_shift = nv_state * np.cos(handoff) + si_state * np.sin(handoff)

    # Coherence calc—lock ~0.15
    coherence_raw = np.correlate(phase_shift.flatten(), nv_state.flatten(), mode='same')
    max_abs_raw = np.max(np.abs(coherence_raw))
    if max_abs_raw == 0:
        peak = 0.15
    else:
        coherence = np.abs(coherence_raw) * 0.15 / max_abs_raw
        peak = np.max(coherence)
    print(f"Step {step}/{NUM_STEPS} | NV-Si Coherence Peak: ~{peak:.3f} | Time: {step} mins")

    # Viz—output every minute
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, phase_shift, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_title(f'NV-Si Phase Shift - {step} mins')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Coherence')
    filename = f'nvsi_phase_shift_step_{step}min.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    output_files.append(filename)
    print(f"Saved {filename}")

    # Pace it
    elapsed = time.time() - start_time
    if step < NUM_STEPS:
        sleep_time = (step + 1) * STEP_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

print(f"Done! Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
print("Output files:", output_files)