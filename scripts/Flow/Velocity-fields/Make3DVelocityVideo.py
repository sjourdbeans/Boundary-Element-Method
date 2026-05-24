import json
from pathlib import Path

import numpy as np
import pyvista as pv



pl = pv.Plotter(off_screen=True)

# =========================
# User settings
# =========================
# data_dir = Path("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/chlamy_vtk_export")  # change this
data_dir = Path("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/velocity-fields/Flowfield_Nx-Ny-Nz=51-51-51_vtk_export")
frames_dir = data_dir / "frames"
geom_dir = data_dir / "geometry"

body_mesh_path = geom_dir / "body_mesh.vtp"
flag1_dir = geom_dir / "flagella_1"
flag2_dir = geom_dir / "flagella_2"

seed_plane = pv.Plane(
    center=(20.0, 0.0, 0.0),
    direction=(-1.0, 0.0, 0.0),
    i_size=40.0,
    j_size=40.0,
    i_resolution=30,
    j_resolution=30,
)

GLYPH_RATE = (2, 2, 2)
FLAG_RADIUS = 0.05
STREAM_RADIUS = 0.03
ARROW_SCALE_DEFAULT = 0.4
Q_LEVEL_FRACTION = 0.2


# =========================
# Load metadata
# =========================
with open(data_dir / "metadata.json", "r") as f:
    meta = json.load(f)

n_frames = meta["n_frames"]
max_speed = meta["max_speed"]

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
    "flag1_actor": None,
    "flag2_actor": None,
    "q_actor": None,
    "seed_actor": None,
}

state["seed_actor"] = pl.add_mesh(seed_plane, style="wireframe", color="black")


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
    # Q-criterion isosurface
    # -------------------------
    remove_actor("q_actor")

    q = np.asarray(grid["qcriterion"])
    qmax = float(np.nanmax(q))

    # if np.isfinite(qmax) and qmax > 0:
    #     q_level = Q_LEVEL_FRACTION * qmax
    #     qsurf = grid.contour(
    #         isosurfaces=[q_level],
    #         scalars="qcriterion",
    #     )
    #     if qsurf.n_points > 0:
    #         state["q_actor"] = pl.add_mesh(
    #             qsurf,
    #             scalars="qcriterion",
    #             clim=[0.0, qmax],
    #             opacity=0.35,
    #             cmap="plasma",
    #             scalar_bar_args={"title": "Q"},
    #         )

    # -------------------------
    # Velocity glyphs
    # -------------------------
    remove_actor("arrow_actor")

    glyph_source = grid.extract_subset(
        [0, meta["Nx"] - 1, 0, meta["Ny"] - 1, 0, meta["Nz"] - 1],
        rate=GLYPH_RATE,
    )

    glyphs = glyph_source.glyph(
        orient="u",
        scale="umag",
        factor=scale,
    )

    # state["arrow_actor"] = pl.add_mesh(
    #     glyphs,
    #     scalars="speed",
    #     clim=[0.0, max_speed],
    #     cmap="turbo",
    #     opacity=0.6,
    #     scalar_bar_args={"title": "|u|"},
    # )

    # -------------------------
    # Streamlines
    # -------------------------
    remove_actor("stream_actor")

    streamlines = grid.streamlines_from_source(
        seed_plane,
        vectors="u",
        max_length=100.0,
        initial_step_length=0.1,
        terminal_speed=1e-12,
        interpolator_type="cell",
    )

    streamlines["omega_z"] = streamlines["vorticity"][:, 2]

    if streamlines.n_cells > 0:
        stream_tube = streamlines.tube(radius=STREAM_RADIUS)
        if stream_tube.n_points > 0:
            state["stream_actor"] = pl.add_mesh(
                stream_tube,
                scalars="umag",
                cmap="turbo",
                opacity=0.8,
                clim=[0,1]
            )

    # -------------------------
    # Flagella
    # -------------------------
    remove_actor("flag1_actor")
    remove_actor("flag2_actor")

    state["flag1_actor"] = pl.add_mesh(
        flag1.tube(radius=FLAG_RADIUS),
        color="red",
    )
    state["flag2_actor"] = pl.add_mesh(
        flag2.tube(radius=FLAG_RADIUS),
        color="red",
    )

    pl.add_text(
        f"Frame: {frame}",
        name="frame_label",
        position="upper_left",
        font_size=12,
    )

    print(f"Frame {frame}")
    print("speed min/max:", np.min(grid['speed']), np.max(grid['speed']))
    print("streamlines:", streamlines.n_points, streamlines.n_cells)

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
# pl.add_key_event("Right", next_frame)
# pl.add_key_event("Left", prev_frame)

# pl.add_slider_widget(
#     callback=update_glyph_scale,
#     rng=[0.1, 20.0],
#     value=ARROW_SCALE_DEFAULT,
#     title="Arrow scale",
#     pointa=(0.35, 0.1),
#     pointb=(0.64, 0.1),
# )

# redraw()


# pl.show()

pl.camera_position = [
    (-10, 60, 80),   # camera location
    (0, 0, 0),      # point it looks at
    (0, 0, 1),      # "up" direction
]
pl.open_movie("flowfield.mp4", framerate=10)

for frame in range(n_frames):
    state["frame"] = frame
    redraw()
    pl.write_frame()

pl.close()