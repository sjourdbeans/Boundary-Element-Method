import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl
import os


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "Palatino"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"
mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

shear_rate=5

# fileswimmer = "/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_1280.pkl"
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

# file = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/shear=0.0_N=8_periods_10/rank_000.h5"
folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
outdir =  Path(folder)

manifest = np.loadtxt(folder + "/manifest.txt", dtype=str, delimiter="\t")
N_conditions = len(manifest)

frames_per_beat = swimmer_template.N_frames

periods = int(folder.split("_")[-1])

frames = frames_per_beat*periods + 1
quaternions = np.zeros((N_conditions, frames, 4), dtype=np.float32)


for rank_file in sorted(outdir.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:
        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])
            Q = grp["quaternions"][:]
            quaternions[sim_idx, :, :] = Q
            # process here

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def quat_to_director(q):
    """First column of rotation matrix = swimmer symmetry axis in lab frame."""
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])


plot_steps = 10

discard_beats = 0
new_frames = 72* frames_per_beat#int(round(0.24*frames))
strobo_idx = np.arange(discard_beats * frames_per_beat, frames, frames_per_beat)

# Shape: (n_ics, n_beats, 4)
strobo_quats    = quaternions[::plot_steps, strobo_idx, :]

# Convert to director vectors — shape: (n_ics, n_beats, 3)
directors = np.array([[quat_to_director(q) for q in ic] for ic in strobo_quats])

n_ics, n_beats, _ = directors.shape


# ── Compute spherical coords ───────────────────────────────────────────────────
theta = np.degrees(np.arctan2(directors[:, :, 1], directors[:, :, 0]))  # azimuth
phi   = np.degrees(np.arcsin(np.clip(directors[:, :, 2], -1, 1)))        # elevation

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6))

# Colour each IC distinctly
colors = plt.cm.hsv(np.linspace(0, 1, n_ics))

# ── Panel 1: Azimuth vs elevation ─────────────────────────────────────────────
# ── Panel 2: 3D sphere ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(111, projection='3d')

# Reference sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
ax2.plot_wireframe(
    np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
    color='gray', alpha=0.4, lw=0.4
)

for i, color in enumerate(colors):
    x, y, z = directors[i, :, 0], directors[i, :, 1], directors[i, :, 2]
    # ax2.plot(x[1:], y[1:], z[1:],   color=color, alpha=0.4)
    ax2.scatter(x[0],  y[0],  z[0],  s=20, color='blue', marker='.', zorder=5)

ax2.set_title('Initial Orientations (440 out of 4400)')
ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$'); ax2.set_zlabel('$z$')
ax2.view_init(elev=30, azim=30)
plt.tight_layout()

plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/Initial-orientations-unit-sphere.pdf")
plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/Initial-orientations-unit-sphere.png",dpi=600)
plt.show()