import json
from pathlib import Path

import numpy as np
import pyvista as pv


N=51

# =========================
# User settings
# =========================
data_dir = Path(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/velocity-fields/new_flag/Flowfield_new_flag_Nx-Ny-Nz={N}-{N}-{N}_vtk_export")
frames_dir = data_dir / "frames"
geom_dir = data_dir / "geometry"

body_mesh_path = geom_dir / "body_mesh.vtp"
flag1_dir = geom_dir / "flagella_1"
flag2_dir = geom_dir / "flagella_2"

seed_plane = pv.Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),   # normal along z -> plane lies in xy
    i_size=40.0,
    j_size=40.0,
    i_resolution=20,
    j_resolution=20,
)

def make_cluster(center, radius=0.7, n=4):
    """Grid of seed points in a small cube around center."""
    offsets = np.linspace(-radius, radius, n)
    pts = []
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                pts.append(center + np.array([dx, dy, dz]))
    return np.array(pts)

cluster_pos_1 = make_cluster(np.array([0.0,  8.0, -1.0]))
cluster_pos_2 = make_cluster(np.array([0.0,  8.0, 1.0]))
cluster_neg_1 = make_cluster(np.array([0.0, -8.0, -1.0]))
cluster_neg_2 = make_cluster(np.array([0.0, -8.0, 1.0]))

seed_3d = pv.PolyData(np.vstack([cluster_pos_1, cluster_pos_2, cluster_neg_1, cluster_neg_2]))

GLYPH_RATE = (2, 2, 2)
FLAG_RADIUS = 0.1
STREAM_RADIUS = 0.05
ARROW_SCALE_DEFAULT = 0.4
Q_LEVEL_FRACTION = 0.2


# =========================
# Load metadata
# =========================
with open(data_dir / "metadata.json", "r") as f:
    meta = json.load(f)

n_frames = meta["n_frames"]
max_speed = 800

body_mesh = pv.read(body_mesh_path)


# =========================
# Plot state
# =========================
pl = pv.Plotter()
pl.add_mesh(body_mesh, show_edges=True, color="lightgray")
pl.show_grid()

state = {
    "frame": 0,
    "scale": ARROW_SCALE_DEFAULT,
    "body_actor": None,
    "arrow_actor": None,
    "stream_actor": None,
    "stream3d_actor": None,
    "flag1_actor": None,
    "flag2_actor": None,
    "q_actor": None,
    "seed_actor": None,
}

# state["seed_actor"] = pl.add_mesh(seed_plane, style="wireframe", color="black")


# =========================
# Helpers
# =========================
def frame_file(frame: int) -> Path:
    return frames_dir / f"frame_{frame:04d}.vti"


def flag1_file(frame: int) -> Path:
    return flag1_dir / f"frame_{frame:04d}.vtp"


def flag2_file(frame: int) -> Path:
    return flag2_dir / f"frame_{frame:04d}.vtp"


def remove_actor(name: str) -> None:
    actor = state.get(name)
    if actor is not None:
        pl.remove_actor(actor)
        state[name] = None


def redraw():
    frame = state["frame"]
    scale = state["scale"]

    grid = pv.read(frame_file(frame))
    flag1 = pv.read(flag1_file(frame))
    flag2 = pv.read(flag2_file(frame))
    grid["omega_z"] = grid["vorticity"][:, 2]

    # -------------------------
    # Streamlines (xy plane only)
    # -------------------------
    remove_actor("stream_actor")

    xy_slice = grid.slice(normal="z", origin=(0.0, 0.0, 0.0))

    u = np.array(xy_slice["u"])/1800
    u[:, 2] = 0.0
    xy_slice["u"] = u
    xy_slice["umag"] = np.linalg.norm(u, axis=1)

    # streamlines = xy_slice.streamlines_from_source(
    #     seed_plane,
    #     vectors="u",
    #     max_length=100.0,
    #     initial_step_length=0.1,
    #     terminal_speed=1e-12,
    #     interpolator_type="cell",
    # )

    # if streamlines.n_points > 0:
    #     streamlines["umag"] = np.linalg.norm(np.array(streamlines["u"]), axis=1)

    # if streamlines.n_cells > 0:
    #     stream_tube = streamlines.tube(radius=STREAM_RADIUS)
    #     if stream_tube.n_points > 0:
    #         state["stream_actor"] = pl.add_mesh(
    #             stream_tube,
    #             scalars="umag",
    #             cmap="turbo",
    #             opacity=0.8,
    #             clim=[1, 10**3],
    #             log_scale=True,
    #         )

    # -------------------------
    # 3D streamlines seeded at (0, ±9, 0)
    # -------------------------
    remove_actor("stream3d_actor")

    streamlines_3d = grid.streamlines_from_source(
        seed_3d,
        vectors="u",
        max_length=100.0,
        initial_step_length=0.1,
        terminal_speed=1e-12,
        interpolator_type="cell",
        integration_direction="both",
    )

    if streamlines_3d.n_points > 0:
        streamlines_3d["umag"] = np.linalg.norm(np.array(streamlines_3d["u"])/1800, axis=1)

    if streamlines_3d.n_cells > 0:
        stream_tube_3d = streamlines_3d.tube(radius=STREAM_RADIUS)
        if stream_tube_3d.n_points > 0:
            state["stream3d_actor"] = pl.add_mesh(
                stream_tube_3d,
                scalars="umag",
                cmap="turbo",
                opacity=0.9,
                clim=[1, 10**3],
                log_scale=True,
                show_scalar_bar=True,   # reuse the bar from the xy streamlines
            )

    # -------------------------
    # Flagella
    # -------------------------
    remove_actor("flag1_actor")
    remove_actor("flag2_actor")

    state["flag1_actor"] = pl.add_mesh(
        flag1.tube(radius=FLAG_RADIUS),
        color="black",
    )
    state["flag2_actor"] = pl.add_mesh(
        flag2.tube(radius=FLAG_RADIUS),
        color="black",
    )

    pl.add_text(
        f"Frame: {frame}",
        name="frame_label",
        position="upper_left",
        font_size=12,
    )

    print(f"Frame {frame}")
    print("speed min/max:", np.min(grid['umag']), np.max(grid['umag']))

    pl.render()


def next_frame():
    state["frame"] = (state["frame"] + 1) % n_frames
    redraw()


def prev_frame():
    state["frame"] = (state["frame"] - 1) % n_frames
    redraw()


def update_glyph_scale(value):
    state["scale"] = value
    redraw()


# =========================
# UI
# =========================
pl.add_key_event("Right", next_frame)
pl.add_key_event("Left", prev_frame)

redraw()
pl.show()