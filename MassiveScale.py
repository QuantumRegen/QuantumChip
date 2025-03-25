import numpy as np
from qutip import Qobj, sigmaz, tensor, mesolve, basis
import pyvista as pv
from mpi4py import MPI  # For scalability

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Fibonacci spiral coords (scaled)
def fibonacci_spiral_coords(n, offset=0, scale=1):
    phi = (1 + 5**0.5) / 2
    return [(np.cos(i * phi) * (i * scale)**0.5, np.sin(i * phi) * (i * scale)**0.5, i * 0.1 + offset)
            for i in range(n)]

# NV: 100 qubits, center shielded
nv_qubits = 100
H_nv = sum([sigmaz() for _ in range(nv_qubits)])
B_field = [0.01 if i < 10 else 0.2 for i in range(nv_qubits)]  # Center shield, mid softer
H_nv += sum([B * sigmaz() for B in B_field])
noise = [0.0005 if i < 10 else 0.001 for i in range(nv_qubits)]  # T2* ~1-2ms center, ~1ms mid

# Si: 100 logical (300 physical)
si_qubits = 300
H_si = sum([sigmaz() for _ in range(si_qubits)])
handoff = tensor(sigmaz(), [qeye(2)] * (si_qubits-1)) * tensor([qeye(2)] * (nv_qubits-1), sigmaz())
H_total = H_nv + H_si + 0.2 * handoff

# Simulate: Sparse, parallel
tlist = np.linspace(0, 2e-3, 200)  # 2ms window
chunk_size = nv_qubits + si_qubits // size
local_start = rank * chunk_size
local_end = min((rank + 1) * chunk_size, nv_qubits + si_qubits)
local_state = basis(nv_qubits + si_qubits, 0)
result = mesolve(H_total, local_state, tlist, c_ops=noise, options={'method': 'sparse'})

# Live viz: Massive scale
if rank == 0:
    plotter = pv.Plotter()
    nv_coords = fibonacci_spiral_coords(nv_qubits, scale=0.1)
    si_coords = fibonacci_spiral_coords(si_qubits, offset=1, scale=0.1)
    for t, state in enumerate(result.states):
        plotter.clear()
        # NV center: Blue, shielded
        for i in range(10):  # Center cluster
            plotter.add_mesh(pv.Sphere(center=nv_coords[i], radius=0.1), color='blue', opacity=0.8 - t * 0.002)
            plotter.add_mesh(pv.Sphere(center=nv_coords[i], radius=0.15), color='grey', opacity=0.3)
        # Mid NV: Red
        for i in range(10, nv_qubits):
            plotter.add_mesh(pv.Sphere(center=nv_coords[i], radius=0.05), color='red', opacity=0.7 - t * 0.002)
        # Si outer: Green
        for i in range(si_qubits):
            plotter.add_mesh(pv.Sphere(center=si_coords[i], radius=0.05), color='green', opacity=0.8)
        # Handoff: Sampled
        for i in range(10, min(20, si_qubits)):
            line = pv.Line(nv_coords[i], si_coords[i])
            plotter.add_mesh(line, color='red', opacity=0.6 - t * 0.003, line_width=2)
            plotter.add_mesh(line, color='green', opacity=t * 0.003, line_width=2)
        # B-field: Large spiral
        b_points = fibonacci_spiral_coords(50, scale=0.2)
        b_lines = pv.Spline(b_points, n_points=100)
        plotter.add_mesh(b_lines, color='blue', line_width=3)
        plotter.show(auto_close=False, interactive_update=True)
        plotter.update()