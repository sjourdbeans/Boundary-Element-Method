import json
from pathlib import Path
 
import numpy as np
import pyvista as pv
 
 
# =========================
# User settings
# =========================
data_dir = Path("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/velocity-fields/Flowfield_Nx-Ny-Nz=101-101-101_vtk_export")
frames_dir = data_dir / "frames"
geom_dir = data_dir / "geometry"
 
body_mesh_path = geom_dir / "body_mesh.vtp"
flag1_dir = geom_dir / "flagella_1"
flag2_dir = geom_dir / "flagella_2"
 
seed_plane = pv.Plane(
    center=(15.0, 0.0, 0.0),
    direction=(-1.0, 0.0, 0.0),
    i_size=40.0,
    j_size=40.0,
    i_resolution=5,
    j_resolution=5,
)
seed_plane_2 = pv.Plane(
    center=(-15.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    i_size=40.0,
    j_size=40.0,
    i_resolution=10,
    j_resolution=10,
)
seed_plane_3 = pv.Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0, 0.0, 1.0),
    i_size=40.0,
    j_size=40.0,
    i_resolution=5,
    j_resolution=5,
)



 
FLAG_RADIUS = 0.05
STREAM_RADIUS = 0.07
Q_LEVEL_FRACTION = 0.2
 
 
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
 
def flag1_file(frame: int) -> Path:
    return flag1_dir / f"frame_{frame:04d}.vtp"
 
def flag2_file(frame: int) -> Path:
    return flag2_dir / f"frame_{frame:04d}.vtp"
 
 
# =========================
# Compute time-averaged velocity field
# =========================
def compute_time_average() -> pv.ImageData:
    print(f"Computing time average over {n_frames} frames...")
 
    # Read first frame to get grid structure and initialise accumulators
    grid0 = pv.read(frame_file(0))
    u_sum = np.zeros_like(np.asarray(grid0["u"]))
 
    for frame in range(n_frames):
        print(f"  Reading frame {frame}/{n_frames - 1}", end="\r")
        grid = pv.read(frame_file(frame))
        u_sum += np.asarray(grid["u"])
 
    print("\nDone.")
 
    # Build averaged grid by copying structure from first frame
    avg_grid = grid0.copy()
    u_avg = u_sum / n_frames
    avg_grid["u"] = u_avg
    avg_grid["umag"] = np.linalg.norm(u_avg, axis=1)
    avg_grid["vorticity"] = grid0["vorticity"]  # vorticity from last frame as placeholder
    avg_grid["omega_z"] = avg_grid["vorticity"][:, 2]
 
    return avg_grid
 
 
# =========================
# Plot time-averaged field
# =========================
avg_grid = compute_time_average()
max_speed = float(np.max(avg_grid["umag"]))
 
pl = pv.Plotter()
pl.add_mesh(body_mesh, show_edges=True, color="lightgray")
pl.show_grid()
 
# Streamlines from time-averaged field
# streamlines = avg_grid.streamlines_from_source(
#     seed_plane,
#     vectors="u",
#     max_length=200.0,
#     initial_step_length=0.1,
#     terminal_speed=1e-12,
#     interpolator_type="cell",
# )
 
# if streamlines.n_cells > 0:
#     stream_tube = streamlines.tube(radius=STREAM_RADIUS)
#     if stream_tube.n_points > 0:
#         pl.add_mesh(
#             stream_tube,
#             scalars="umag",
#             cmap="turbo",
#             opacity=0.8,
#             clim=[0, max_speed],
#         )


# streamlines_2 = avg_grid.streamlines_from_source(
#     seed_plane_2,
#     vectors="u",
#     max_length=200.0,
#     initial_step_length=0.1,
#     terminal_speed=1e-12,
#     interpolator_type="cell",
# )
 
# if streamlines_2.n_cells > 0:
#     stream_tube_2 = streamlines_2.tube(radius=STREAM_RADIUS)
#     if stream_tube_2.n_points > 0:
#         pl.add_mesh(
#             stream_tube_2,
#             scalars="umag",
#             cmap="turbo",
#             opacity=0.8,
#             clim=[0, max_speed],
#         )

# streamlines_3 = avg_grid.streamlines_from_source(
#     seed_plane_3,
#     vectors="u",
#     max_length=400.0,
#     initial_step_length=0.1,
#     terminal_speed=1e-12,
#     interpolator_type="cell",
# )
 
# if streamlines_3.n_cells > 0:
#     stream_tube_3 = streamlines_3.tube(radius=STREAM_RADIUS)
#     if stream_tube_3.n_points > 0:
#         pl.add_mesh(
#             stream_tube_3,
#             scalars="umag",
#             cmap="turbo",
#             opacity=0.8,
#             clim=[0, max_speed],
#         )

slice_z = avg_grid.slice(normal="z", origin=(0, 0, 0))

pl.add_mesh(
    slice_z,
    scalars="umag",
    cmap="turbo",
    clim=[0, max_speed],
    scalar_bar_args={"title": "|u|"},
)
slice_z_2 = avg_grid.slice(normal="y", origin=(0, -10, 0))

pl.add_mesh(
    slice_z_2,
    scalars="umag",
    cmap="turbo",
    clim=[0, max_speed],
    scalar_bar_args={"title": "|u|"},
)

# Project velocity to 2D by zeroing the z-component so arrows stay in plane
u = np.array(slice_z["u"])
u[:, 2] = 0.0
slice_z["u_2d"] = u

glyphs = slice_z.glyph(
    orient="u_2d",
    scale="umag",
    factor=0.02,  # adjust arrow size
    tolerance=0.02,  # controls arrow density, higher = fewer arrows
)

pl.add_mesh(
    glyphs,
    color="white",
    opacity=0.8,
)

u2= np.array(slice_z_2["u"])
u[:, 2] = 0.0
slice_z["u_2d"] = u

glyphs = slice_z.glyph(
    orient="u_2d",
    scale="umag",
    factor=0.02,  # adjust arrow size
    tolerance=0.02,  # controls arrow density, higher = fewer arrows
)

pl.add_mesh(
    glyphs,
    color="white",
    opacity=0.8,
)


# seed_line = pv.Line(
#     pointa=(10, -20, 0),
#     pointb=(10, 20, 0),
#     resolution=50,
# )

# Slice the grid first, then compute streamlines on the slice
# slice_z = avg_grid.slice(normal="z", origin=(0, 0, 0))

# streamlines_2d = slice_z.streamlines_from_source(
#     seed_line,
#     vectors="u",
#     max_length=200.0,
#     initial_step_length=0.1,
#     terminal_speed=1e-12,
#     interpolator_type="cell",
# )

# if streamlines_2d.n_cells > 0:
#     pl.add_mesh(
#         streamlines_2d.tube(radius=STREAM_RADIUS),
#         scalars="umag",
#         # cmap="turbo",
#         clim=[0, max_speed],
#     )
# slices = avg_grid.slice_orthogonal(x=0, y=0, z=0)
# pl.add_mesh(slices, scalars="umag", cmap="turbo", clim=[0, max_speed])
 
# Seed plane outline
# pl.add_mesh(seed_plane, style="wireframe", color="black")


pl.add_volume(
    avg_grid,
    scalars="umag",
    cmap="turbo",
    opacity="sigmoid",
    clim=[0*max_speed, max_speed],
    scalar_bar_args={"title": "|u|"},
)
 
pl.add_text(
    f"Time-averaged velocity field ({n_frames} frames)",
    position="upper_left",
    font_size=12,
)
 
print(f"Time-averaged speed min/max: {np.min(avg_grid['umag']):.4f} / {max_speed:.4f}")
# print(f"Streamlines: {streamlines.n_points} points, {streamlines.n_cells} cells")
 
pl.show()
 
