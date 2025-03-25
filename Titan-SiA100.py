import numpy as np
import pyvista as pv
import time


def fibonacci_spiral_coords(n, offset=0, scale=0.1, shield_factor=0.5, rotate=0, mirror=False, z_shift=0):
    """Precompute base Fibonacci spiral coordinates (no time dependency here)."""
    phi = (1 + 5 ** 0.5) / 2
    theta = np.radians(rotate)
    coords = [(scale * i ** 0.5 * np.cos(i * phi) * (shield_factor if i < n // 2 else 1),
               scale * i ** 0.5 * np.sin(i * phi) * (shield_factor if i < n // 2 else 1),
               scale * i * 0.15 + z_shift) for i in range(n)]
    if mirror:
        return np.array([(x * np.cos(theta) - (-y) * np.sin(theta) + offset,
                          x * np.sin(theta) + (-y) * np.cos(theta), z) for x, y, z in coords])
    return np.array([(x * np.cos(theta) - y * np.sin(theta) + offset,
                      x * np.sin(theta) + y * np.cos(theta), z) for x, y, z in coords])


# Set up PyVista plotter
pv.set_plot_theme("dark")
plotter = pv.Plotter(window_size=[2560, 1440], off_screen=False)
gap = 5.0

# Precompute base coordinates
nv_base = fibonacci_spiral_coords(100, offset=0, scale=0.25, shield_factor=0.6, rotate=90)
vortex_base = fibonacci_spiral_coords(100, offset=0, scale=0.3, rotate=90)
si_base = fibonacci_spiral_coords(100, offset=gap / 2, scale=0.4, rotate=90)
sink_base = fibonacci_spiral_coords(120, offset=0, scale=0.6, rotate=90)

# Create meshes once
nv_spheres = [pv.Sphere(center=coord, radius=0.12) for coord in nv_base]
vortex_spheres = [pv.Sphere(center=coord, radius=0.15) for coord in vortex_base]
si_spheres = [pv.Sphere(center=coord, radius=0.15) for coord in si_base]
handoff_lines = [pv.Line(nv_base[i * 5], vortex_base[i * 5], resolution=10) for i in range(20)]
vortex_si_lines = [pv.Line(vortex_base[i * 5], si_base[i * 5], resolution=15) for i in range(20)]
sink_spline = pv.Spline(sink_base, n_points=800)

# Add meshes to plotter once
for sphere in nv_spheres:
    plotter.add_mesh(sphere, color='blue', opacity=0.9, smooth_shading=True, specular=1.0)
for sphere in vortex_spheres:
    plotter.add_mesh(sphere, color='blue', opacity=0.9, smooth_shading=True, specular=1.0)
for sphere in si_spheres:
    plotter.add_mesh(sphere, color='green', opacity=0.9, smooth_shading=True, specular=0.8)
for line in handoff_lines:
    plotter.add_mesh(line, color='red', opacity=0.9, line_width=8)
for line in vortex_si_lines:
    plotter.add_mesh(line, color=[1, 0.5, 0], opacity=0.85, line_width=10)
plotter.add_mesh(sink_spline, color='blue', line_width=16, opacity=0.95, specular=1.0)


def update_frame(t):
    """Update mesh properties dynamically instead of recreating them."""
    # Update NV spheres
    for i, sphere in enumerate(nv_spheres):
        pulse = 0.12 * (1 + 0.3 * np.sin(t * 10))  # ~50-200 ns
        fidelity = 0.95 + 0.03 * np.cos(t + i * 0.05)  # 95-98%
        sphere.radius = pulse
        sphere.opacity = min(fidelity, 1.0)

    # Update Vortex spheres
    for i, sphere in enumerate(vortex_spheres):
        pulse = 0.15 * (1 + 0.4 * np.sin(t * 2))  # ~1-10 ms
        fidelity = 0.92 + 0.05 * np.cos(t + i * 0.03)  # 92-97%
        sphere.radius = pulse
        sphere.opacity = min(fidelity, 1.0)

    # Update Si spheres
    for i, sphere in enumerate(si_spheres):
        ripple = 0.15 * (1 + 0.3 * np.sin(t * 5 + i * 0.04))  # ~10-30 µs
        opacity = 0.9 - i * 0.005
        sphere.radius = ripple
        sphere.opacity = min(max(opacity, 0.6), 1.0)

    # Update handoff lines
    for i, (nv_line, vs_line) in enumerate(zip(handoff_lines, vortex_si_lines)):
        flicker_nv = 0.9 + 0.1 * np.sin(t * 100 + i)  # ~1-10 µs
        flicker_vortex = 0.85 + 0.1 * np.sin(t * 50 + i)  # ~10-100 µs
        nv_line.opacity = flicker_nv
        vs_line.opacity = flicker_vortex
        vs_line.color = [1, 0.5 - 0.5 * flicker_vortex, 0]

    # Update hybrid sink
    noise = 0.05 * np.sin(t * 0.5)  # ~5-10% ripple
    sink_points = sink_base.copy()
    theta = t * 5  # Spin effect
    sink_points[:, 0] = sink_base[:, 0] * np.cos(theta) - sink_base[:, 1] * np.sin(theta) + noise * 0.1
    sink_points[:, 1] = sink_base[:, 0] * np.sin(theta) + sink_base[:, 1] * np.cos(theta)
    sink_spline.points = sink_points
    sink_spline.line_width = 16 + noise * 5


# Animation loop with adaptive timing
t = 0
target_fps = 60
frame_time = 1.0 / target_fps if not pv.OFF_SCREEN else 0  # No sleep if off-screen

while True:
    start_time = time.time()
    update_frame(t)
    plotter.update()
    t += 0.03
    elapsed = time.time() - start_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)  # Sleep only if rendering is faster than target