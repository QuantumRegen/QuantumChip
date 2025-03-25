import numpy as np
import pyvista as pv
import time


def fibonacci_spiral_coords(n, offset=0, scale=0.1, shield_factor=0.5, rotate=0, mirror=False, z_shift=0, t=0):
    phi = (1 + 5 ** 0.5) / 2
    theta = np.radians(rotate)
    coords = [(scale * i ** 0.5 * np.cos(i * phi + t * 0.1) * (shield_factor if i < n // 2 else 1),
               scale * i ** 0.5 * np.sin(i * phi + t * 0.1) * (shield_factor if i < n // 2 else 1),
               scale * i * 0.2 + z_shift) for i in range(n)]
    return [(x * np.cos(theta) - y * np.sin(theta) + offset,
             x * np.sin(theta) + y * np.cos(theta), z) if not mirror else
            (x * np.cos(theta) - (-y) * np.sin(theta) + offset,
             x * np.sin(theta) + (-y) * np.cos(theta), z) for x, y, z in coords]


pv.set_plot_theme("dark")
plotter = pv.Plotter(window_size=[2560, 1440], off_screen=False)
gap = 4.0


def update_frame(t):
    plotter.clear()
    # Spiral coords with slight time offset
    nv_coords = fibonacci_spiral_coords(15, offset=gap / 2, scale=0.35, shield_factor=0.5, rotate=90, t=t)
    nv_coords_mirror = fibonacci_spiral_coords(15, offset=-gap / 2, scale=0.35, shield_factor=0.5, rotate=-90,
                                               mirror=True, t=t)
    si_coords = fibonacci_spiral_coords(15, offset=gap / 2 + 2.5, scale=0.4, rotate=90, t=t)
    si_coords_mirror = fibonacci_spiral_coords(15, offset=-(gap / 2 + 2.5), scale=0.4, rotate=-90, mirror=True, t=t)

    # B-field glow with pulse
    for coords in [nv_coords, nv_coords_mirror]:
        for i, coord in enumerate(coords):
            b_field_strength = 0.01 * (1 + min(i, len(coords) - 1 - i) * 0.05) * (1 + 0.3 * np.sin(t))
            opacity = 0.9 - i * 0.02
            glow = pv.Sphere(center=coord, radius=0.15 + b_field_strength * 0.5)
            plotter.add_mesh(glow, color='cyan', opacity=min(max(opacity, 0.4), 1.0), smooth_shading=True, specular=1.0)
    for coords in [si_coords, si_coords_mirror]:
        for i, coord in enumerate(coords):
            b_field_strength = 0.01 * (1 + min(i, len(coords) - 1 - i) * 0.05) * (1 + 0.3 * np.sin(t))
            opacity = 0.9 - i * 0.02
            glow = pv.Sphere(center=coord, radius=0.15 + b_field_strength * 0.5)
            plotter.add_mesh(glow, color='lime', opacity=min(max(opacity, 0.4), 1.0), smooth_shading=True, specular=1.0)

    # Fluid cores with wiggle
    fluid_points = fibonacci_spiral_coords(20, offset=gap / 2, scale=0.12, rotate=90, t=t)
    fluid_lines = pv.Spline(fluid_points, n_points=300)
    fluid_points_mirror = fibonacci_spiral_coords(20, offset=-gap / 2, scale=0.12, rotate=-90, mirror=True, t=t)
    fluid_lines_mirror = pv.Spline(fluid_points_mirror, n_points=300)
    for fluid in [fluid_lines, fluid_lines_mirror]:
        plotter.add_mesh(fluid, color='purple', line_width=14, opacity=0.9, specular=0.7)

    # Sink rings with warp
    sink_points = fibonacci_spiral_coords(60, offset=gap / 2 + 3.5, scale=0.5, rotate=90, t=t * 0.5)
    sink_lines = pv.Spline(sink_points, n_points=600)
    sink_points_mirror = fibonacci_spiral_coords(60, offset=-(gap / 2 + 3.5), scale=0.5, rotate=-90, mirror=True,
                                                 t=t * 0.5)
    sink_lines_mirror = pv.Spline(sink_points_mirror, n_points=600)
    for sink in [sink_lines, sink_lines_mirror]:
        plotter.add_mesh(sink, color='blue', line_width=16, opacity=0.95, specular=1.0)

    # Handoff lines with flicker
    for i in range(7):
        line = pv.Line(nv_coords[i + 4], si_coords[i + 4], resolution=15)
        line_mirror = pv.Line(nv_coords_mirror[i + 4], si_coords_mirror[i + 4], resolution=15)
        bridge = pv.Line(nv_coords[i + 4], nv_coords_mirror[i + 4], resolution=15)
        flicker = 0.85 + 0.1 * np.sin(t * 2 + i)
        plotter.add_mesh(line, color='red', opacity=flicker, line_width=10)
        plotter.add_mesh(line_mirror, color='red', opacity=flicker, line_width=10)
        plotter.add_mesh(bridge, color='orange', opacity=flicker, line_width=9)

    # Yellow gap plane (static)
    wave_points = [(x, z, y) for x in np.linspace(-gap / 2, gap / 2, 40)
                   for y in np.linspace(-1.5, 1.5, 40) for z in [0]]
    wave_x, wave_y, wave_z = np.array(wave_points).T.reshape(40, 40, 3).transpose(1, 0, 2)
    wave = pv.StructuredGrid(wave_x, wave_y, wave_z)
    plotter.add_mesh(wave, color='yellow', opacity=0.6, specular=0.6)

    # Vortex with spin
    vortex_points = [(np.cos(i * 0.35 + t * 0.2) * gap / 4, np.sin(i * 0.35 + t * 0.2) * gap / 4, z)
                     for i in range(50) for z in np.linspace(0, 2.2, 30)]
    vortex = pv.PolyData(vortex_points)
    vortex.lines = np.hstack([[750] + list(range(750)), [750] + list(range(750, 1500))])
    plotter.add_mesh(vortex, color='white', opacity=0.5, line_width=6, specular=0.8)


# Animation loop
t = 0
while True:
    update_frame(t)
    plotter.update()
    t += 0.1
    time.sleep(0.016)  # ~60 FPS target