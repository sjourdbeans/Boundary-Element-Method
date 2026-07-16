import json
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvista import examples


# =========================
# User settings
# =========================
N = 51

data_dir = Path(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/velocity-fields/new_flag/Flowfield_new_flag_Nx-Ny-Nz={51}-{51}-{51}_vtk_export")
frames_dir = data_dir / "frames"
geom_dir = data_dir / "geometry"

body_mesh_path = geom_dir / "body_mesh.vtp"

FLAG_RADIUS = 0.05
STREAM_RADIUS = 0.07

# =========================
# Load metadata
# =========================
with open(data_dir / "metadata.json", "r") as f:
    meta = json.load(f)

n_frames = meta["n_frames"]
body_mesh = pv.read(body_mesh_path)


# =========================
# Helpers
# =========================
def frame_file(frame: int) -> Path:
    return frames_dir / f"frame_{frame:04d}.vti"


# =========================
# Compute time-averaged velocity field
# =========================
def compute_time_average() -> pv.ImageData:
    print(f"Computing time average over {n_frames} frames...")
    grid0 = pv.read(frame_file(0))
    u_sum = np.zeros_like(np.asarray(grid0["u"]))

    for frame in range(n_frames):
        print(f"  Reading frame {frame}/{n_frames - 1}", end="\r")
        grid = pv.read(frame_file(frame))
        u_sum += np.asarray(grid["u"])

    print("\nDone.")
    avg_grid = grid0.copy()
    u_avg = u_sum / n_frames
    avg_grid["u"] = u_avg
    avg_grid["umag"] = np.linalg.norm(u_avg, axis=1)
    avg_grid["vorticity"] = grid0["vorticity"]
    avg_grid["omega_z"] = avg_grid["vorticity"][:, 2]
    return avg_grid


avg_grid = compute_time_average()
max_speed = float(np.max(avg_grid["umag"]))

# Get grid bounds for slider ranges
bounds = avg_grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
x_min, x_max = bounds[0], bounds[1]
y_min, y_max = bounds[2], bounds[3]
z_min, z_max = bounds[4], bounds[5]

# =========================
# Plotter with sliders
# =========================
pl = pv.Plotter()
pl.add_mesh(body_mesh, show_edges=True, color="lightgray")
pl.show_grid()

pl.add_volume(
    avg_grid,
    scalars="umag",
    cmap="turbo",
    opacity="sigmoid",
    clim=[0, max_speed],
    scalar_bar_args={"title": "|u|"},
)

pl.add_text(
    f"Time-averaged velocity field ({n_frames} frames)",
    position="upper_left",
    font_size=12,
)

# Store slice actors so we can remove and re-add them
slice_actors = {"z": None, "y": None}


def add_glyphs(pl, slice_mesh, zero_component):
    u = np.array(slice_mesh["u"])
    u[:, zero_component] = 0.0
    slice_mesh["u_2d"] = u
    glyphs = slice_mesh.glyph(
        orient="u_2d",
        scale="umag",
        factor=0.00005,
        tolerance=0.02,
    )
    return glyphs


def update_slice_z(value):
    origin = (0, 0, value)
    if slice_actors["z"] is not None:
        pl.remove_actor(slice_actors["z"])
    new_slice = avg_grid.slice(normal="z", origin=origin)
    actor1 = pl.add_mesh(
        new_slice,
        scalars="umag",
        cmap="turbo",
        clim=[0, max_speed],
        scalar_bar_args={"title": "|u|"},
        name="slice_z",
    )
    glyphs = add_glyphs(pl, new_slice, zero_component=2)
    actor2 = pl.add_mesh(glyphs, color="white", opacity=0.8, name="glyphs_z")
    slice_actors["z"] = ["slice_z", "glyphs_z"]


def update_slice_y(value):
    origin = (0, value, 0)
    if slice_actors["y"] is not None:
        pl.remove_actor(slice_actors["y"])
    new_slice = avg_grid.slice(normal="y", origin=origin)
    actor1 = pl.add_mesh(
        new_slice,
        scalars="umag",
        cmap="turbo",
        clim=[0, max_speed],
        scalar_bar_args={"title": "|u|"},
        name="slice_y",
    )
    glyphs = add_glyphs(pl, new_slice, zero_component=1)
    actor2 = pl.add_mesh(glyphs, color="white", opacity=0.8, name="glyphs_y")
    slice_actors["y"] = ["slice_y", "glyphs_y"]


# Initialise slices
update_slice_z(0.0)
update_slice_y(10.0)

# Add sliders
pl.add_slider_widget(
    callback=update_slice_z,
    rng=[z_min, z_max],
    value=0.0,
    title="Z slice position",
    pointa=(0.1, 0.1),
    pointb=(0.4, 0.1),
    style="modern",
)

pl.add_slider_widget(
    callback=update_slice_y,
    rng=[y_min, y_max],
    value=10.0,
    title="Y slice position",
    pointa=(0.6, 0.1),
    pointb=(0.9, 0.1),
    style="modern",
)

print(f"Time-averaged speed min/max: {np.mean(avg_grid['umag']):.4f} / {max_speed:.4f}")
pl.show()