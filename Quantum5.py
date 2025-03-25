import numpy as np
import pyvista as pv
import time


def fibonacci_spiral_coords(n, offset=0, scale=0.1, shield_factor=0.5, rotate=0, mirror=False, z_shift=0, t=0):
    phi = (1 + 5 ** 0.5) / 2
    theta = np.radians(rotate)
    coords = [(scale * i ** 0.5 * np.cos(i * phi + t * 0.03) * (shield_factor if i < n // 2 else 1),
               scale * i ** 0.5 * np.sin(i * phi + t * 0.03) * (shield_factor if i < n // 2 else 1),
               scale * i * 0.15 + z_shift) for i in range(n)]
    return [(x * np.cos(theta) - y * np.sin(theta) + offset,
             x * np.sin(theta) + y * np.cos(theta), z) if not mirror else
            (x * np.cos(theta) - (-y) * np.sin(theta) + offset,
             x * np.sin(theta) + (-y) * np.cos(theta), z) for x, y, z in coords]


pv.set_plot_theme("dark")
plotter = pv.Plotter(window_size=[2560, 1440], off_screen=False)
gap = 5.0


def update_frame(t):
    plotter.clear()
    # NV (horizon control), Vortex (horizon), Si (outer)
    nv_coords = fibonacci_spiral_coords(100, offset=0, scale=0.25, shield_factor=0.6, rotate=90, t=t)
    vortex_coords = fibonacci_spiral_coords(100, offset=0, scale=0.3, rotate=90, t=t * 0.8)
    si_coords = fibonacci_spiral_coords(100, offset=gap / 2, scale=0.4, rotate=90, t=t * 0.7)

    # NV horizon control (blue, ~50-200 ns)
    for i, coord in enumerate(nv_coords):
        pulse = 0.12 * (1 + 0.3 * np.sin(t * 10))  # ~50-200 ns
        fidelity = 0.95 + 0.03 * np.cos(t + i * 0.05)  # 95-98%
        glow = pv.Sphere(center=coord, radius=pulse)
        plotter.add_mesh(glow, color='blue', opacity=min(fidelity, 1.0), smooth_shading=True, specular=1.0)

    # Vortex horizon (blue, ~1-10 ms)
    for i, coord in enumerate(vortex_coords):
        pulse = 0.15 * (1 + 0.4 * np.sin(t * 2))  # ~1-10 ms
        fidelity = 0.92 + 0.05 * np.cos(t + i * 0.03)  # 92-97%
        glow = pv.Sphere(center=coord, radius=pulse)
        plotter.add_mesh(glow, color='blue', opacity=min(fidelity, 1.0), smooth_shading=True, specular=1.0)

    # Si outer (green, ~10-30 µs)
    for i, coord in enumerate(si_coords):
        ripple = 0.15 * (1 + 0.3 * np.sin(t * 5 + i * 0.04))  # ~10-30 µs
        opacity = 0.9 - i * 0.005
        glow = pv.Sphere(center=coord, radius=ripple)
        plotter.add_mesh(glow, color='green', opacity=min(max(opacity, 0.6), 1.0), smooth_shading=True, specular=0.8)

    # Handoffs (NV-to-vortex red, vortex-to-Si red-to-green)
    for i in range(20):
        idx = i * 5
        nv_vortex = pv.Line(nv_coords[idx], vortex_coords[idx], resolution=10)
        vortex_si = pv.Line(vortex_coords[idx], si_coords[idx], resolution=15)
        flicker_nv = 0.9 + 0.1 * np.sin(t * 100 + i)  # ~1-10 µs
        flicker_vortex = 0.85 + 0.1 * np.sin(t * 50 + i)  # ~10-100 µs
        plotter.add_mesh(nv_vortex, color='red', opacity=flicker_nv, line_width=8)
        plotter.add_mesh(vortex_si, color=[1, 0.5 - 0.5 * flicker_vortex, 0], opacity=flicker_vortex, line_width=10)

    # Hybrid sink (blue, magnetic + vortex spin)
    sink_points = fibonacci_spiral_coords(120, offset=0, scale=0.6, rotate=90 + t * 5, t=t * 0.1)
    sink_lines = pv.Spline(sink_points, n_points=800)
    noise = 0.05 * np.sin(t * 0.5)  # ~5-10% ripple
    plotter.add_mesh(sink_lines, color='blue', line_width=16 + noise * 5, opacity=0.95, specular=1.0)


# Animation loop
t = 0
while True:
    update_frame(t)
    plotter.update()
    t += 0.03
    time.sleep(0.016)  # ~60 FPS target, scales to 0.5-1 FPS on 8x A100s