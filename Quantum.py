import numpy as np
import pyvista as pv
from qutip import Qobj, sigmaz, sigmax, tensor, qeye

# Fibonacci spiral coords
def fibonacci_spiral_coords(n, offset=0, scale=0.1, shield_factor=0.5):
    phi = (1 + 5**0.5) / 2
    return [(scale * i**0.5 * np.cos(i * phi) * (shield_factor if i < n//2 else 1) + offset,
             scale * i**0.5 * np.sin(i * phi) * (shield_factor if i < n//2 else 1),
             scale * i * 0.1) for i in range(n)]

# Helper: Operator at position
def op_at(op, pos, N, coeff=1.0):
    return coeff * tensor([op if i == pos else qeye(2) for i in range(N)])

# Params
n_nv = 5    # 5 NV diamond qubits
n_si = 5    # 5 Si P:Si logical qubits
n_total = n_nv + n_si

# Hamiltonian snapshot
H_terms = []
for i in range(n_nv):
    H_terms.append(op_at(sigmaz(), i, n_total, 1.0))
for i in range(n_si):
    H_terms.append(op_at(sigmaz(), n_nv + i, n_total, 1.0))
B_field = [0.01 for _ in range(n_total)]
for i, B in enumerate(B_field):
    H_terms.append(op_at(sigmaz(), i, n_total, B))
for i in range(n_nv):
    H_terms.append(op_at(sigmax(), i, n_total, 0.1))
for i in range(n_si):
    H_terms.append(op_at(sigmax(), n_nv + i, n_total, 0.05))
for i in range(3):  # 3 handoff pairs
    nv_idx = n_nv - 3 + i
    si_idx = n_nv + i
    H_terms.append(0.067 * tensor([sigmaz() if j in [nv_idx, si_idx] else qeye(2) for j in range(n_total)]))  # 0.2 / 3
H = sum(H_terms)
# Skip dense eigs if too big, just print shape
print(f"Hamiltonian shape: {H.shape}")

# Static viz
pv.set_plot_theme("document")
plotter = pv.Plotter(window_size=[1280, 720])
nv_coords = fibonacci_spiral_coords(n_nv, scale=0.1, shield_factor=0.5)
si_coords = fibonacci_spiral_coords(n_si, offset=1, scale=0.2)
fluid_points = fibonacci_spiral_coords(n_nv//2, scale=0.05)
fluid_lines = pv.Spline(fluid_points, n_points=20)
sink_points = fibonacci_spiral_coords(5, offset=2, scale=0.3)
sink_lines = pv.Spline(sink_points, n_points=20)

for i, coord in enumerate(nv_coords):
    plotter.add_mesh(pv.Sphere(center=coord, radius=0.04), color='blue', opacity=0.8)
for i, coord in enumerate(si_coords):
    plotter.add_mesh(pv.Sphere(center=coord, radius=0.04), color='green', opacity=0.8)
for i in range(3):
    line = pv.Line(nv_coords[n_nv - 3 + i], si_coords[i])
    plotter.add_mesh(line, color='red', opacity=0.6, line_width=2)
plotter.add_mesh(fluid_lines, color='purple', line_width=3)
plotter.add_mesh(sink_lines, color='blue', line_width=3)
plotter.show()