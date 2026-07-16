import json
from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib as mpl
import os

# =========================
# Matplotlib settings
# =========================
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.latex.preamble"] += r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{upgreek}"
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["axes.labelsize"] = 15
mpl.rcParams["axes.titlesize"] = 15
mpl.rcParams["legend.fontsize"] = 13

N = 51

# =========================
# User settings
# =========================
data_dir = Path(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/velocity-fields/new_flag/Flowfield_new_flag_Nx-Ny-Nz={N}-{N}-{N}_vtk_export")
frames_dir = data_dir / "frames"
geom_dir   = data_dir / "geometry"

body_mesh_path = geom_dir / "body_mesh.vtp"
flag1_dir = geom_dir / "flagella_1"
flag2_dir = geom_dir / "flagella_2"

def make_cluster(center, radius=0.7, n=4):
    offsets = np.linspace(-radius, radius, n)
    pts = []
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                pts.append(center + np.array([dx, dy, dz]))
    return np.array(pts)

cluster_pos_1 = make_cluster(np.array([0.0,  8.0, -1.0]))
cluster_pos_2 = make_cluster(np.array([0.0,  8.0,  1.0]))
cluster_neg_1 = make_cluster(np.array([0.0, -8.0, -1.0]))
cluster_neg_2 = make_cluster(np.array([0.0, -8.0,  1.0]))

seed_3d = pv.PolyData(np.vstack([cluster_pos_1, cluster_pos_2, cluster_neg_1, cluster_neg_2]))

FLAG_RADIUS   = 0.1
STREAM_RADIUS = 0.05
FRAME = 2
CLIM  = [1, 1e3]

# =========================
# Load data (once)
# =========================
with open(data_dir / "metadata.json", "r") as f:
    meta = json.load(f)

body_mesh = pv.read(body_mesh_path)

def frame_file(frame): return frames_dir / f"frame_{frame:04d}.vti"
def flag1_file(frame): return flag1_dir  / f"frame_{frame:04d}.vtp"
def flag2_file(frame): return flag2_dir  / f"frame_{frame:04d}.vtp"

grid  = pv.read(frame_file(FRAME))
flag1 = pv.read(flag1_file(FRAME))
flag2 = pv.read(flag2_file(FRAME))
grid["omega_z"] = grid["vorticity"][:, 2]

# Compute streamlines once, reuse for both panels
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
    streamlines_3d["umag"] = np.linalg.norm(np.array(streamlines_3d["u"]) / 1800, axis=1)

stream_tube_3d = None
if streamlines_3d.n_cells > 0:
    tube = streamlines_3d.tube(radius=STREAM_RADIUS)
    if tube.n_points > 0:
        stream_tube_3d = tube

print(f"Frame {FRAME}")
print("speed min/max:", np.min(grid['umag']), np.max(grid['umag']))

# =========================
# Render helper
# =========================
CAMERA_POSITIONS = [
    (60,  10,  30),
    (-60,  -10, -30),
]
PANEL_LABELS = ["Front", "Back"]

def render_scene(camera_position):
    pl = pv.Plotter(off_screen=True, window_size=[3000, 1800])
    pl.add_mesh(body_mesh, show_edges=True, color="lightgray")

    if stream_tube_3d is not None:
        pl.add_mesh(
            stream_tube_3d,
            scalars="umag",
            cmap="turbo",
            opacity=0.9,
            clim=CLIM,
            log_scale=True,
            show_scalar_bar=False,
        )

    pl.add_mesh(flag1.tube(radius=FLAG_RADIUS), color="black")
    pl.add_mesh(flag2.tube(radius=FLAG_RADIUS), color="black")

    pl.camera.position    = camera_position
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up          = (0, 0, 1)
    pl.camera.zoom(1)
    pl.show(auto_close=False)
    img = pl.screenshot(None, return_img=True)
    pl.close()
    return img

imgs = [render_scene(pos) for pos in CAMERA_POSITIONS]

# =========================
# Composite in matplotlib
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.01)

for ax, img, label in zip(axes, imgs, PANEL_LABELS):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=16)

# Shared colorbar on the right
norm = LogNorm(vmin=CLIM[0], vmax=CLIM[1])
sm   = cm.ScalarMappable(cmap="turbo", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
cbar.set_label(r"$\|\mathbf{u}\|$ ($\upmu$m/s)")
# cbar.ax.tick_params(labelsize=10)

# fig.savefig("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Flowfield/velocity-fields/free-swimmer-3D-streamlines.png", dpi=300, bbox_inches="tight")
# fig.savefig("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Flowfield/velocity-fields/free-swimmer-3D-streamlines.pdf", dpi=600, bbox_inches="tight")
plt.show()