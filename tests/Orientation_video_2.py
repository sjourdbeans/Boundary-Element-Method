import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.latex.preamble"] += r"\usepackage{amsmath}"
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["axes.labelsize"] = 15
mpl.rcParams["axes.titlesize"] = 15
mpl.rcParams["legend.fontsize"] = 13

# ===================== Parameters =====================
shear_rate = 4
ratio=1.25
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/mesh=320_shear={shear_rate}_N=4500_periods_140"
# folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
# folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio={ratio}/shear={shear_rate}_N=4500_periods_140"

mode         = "sliding"
window_beats = 10
stride_beats = 1
discard_beats = 0
n_psi_bins   = 240
n_th_bins    = 120

save_video  = True
output_path = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Chlamy/Trajectories/orientation-distribution/orientation_animation_symmetric_shear={shear_rate}.mp4"
fps         = 10
dpi         = 150

# ===================== Load data =====================
with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

outdir = Path(folder)
manifest = np.loadtxt(folder + "/manifest.txt", dtype=str, delimiter="\t")
N_conditions = len(manifest)
frames_per_beat = swimmer_template.N_frames
periods = int(folder.split("_")[-1])
frames = frames_per_beat * periods + 1

initial_conditions = np.zeros((N_conditions, 4), dtype=np.float32)
quaternions = np.zeros((N_conditions, frames, 4), dtype=np.float32)

for rank_file in sorted(outdir.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:
        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])
            initial_conditions[sim_idx, :] = grp["initial_orientation"][:]
            quaternions[sim_idx, :, :] = grp["quaternions"][:]

# ===================== Deduplicate =====================
def canonicalize_quaternion_sign(Q):
    Q = Q.copy()
    for i in range(Q.shape[0]):
        q = Q[i]
        for val in q:
            if abs(val) > 1e-12:
                if val < 0:
                    Q[i] = -q
                break
    return Q

initial_q = quaternions[:, 0, :]
initial_q_canon = canonicalize_quaternion_sign(initial_q)
_, unique_idx = np.unique(np.round(initial_q_canon, 6), axis=0, return_index=True)
unique_idx = np.sort(unique_idx)
quaternions = quaternions[unique_idx, :, :]

# ===================== Director extraction =====================
def quat_to_director(q):
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])

total_beats = periods
beat_indices_all = np.arange(discard_beats, total_beats)
strobo_all = np.array([
    [quat_to_director(quaternions[i, b * frames_per_beat, :])
     for b in beat_indices_all]
    for i in range(quaternions.shape[0])
])

# ===================== Coordinate conversion =====================
def dirs_to_psi_theta(dirs):
    """
    Convert directors in lab frame (flow=x, gradient=y, vorticity=-z)
    to paper angles (Psi, Theta) matching Jing et al. Fig 5.
    
    Paper frame: flow=x, gradient=z, vorticity=y
    Mapping: ex_paper = dx, ey_paper = -dz, ez_paper = dy
    Theta = arcsin(ez_paper)  -- elevation from flow-vorticity plane
    Psi   = arctan2(ey_paper, ex_paper)  -- azimuth, Psi=pi/2 is vorticity
    """
    ex_paper =  dirs[:, 0]   # flow: same
    ey_paper = -dirs[:, 2]   # paper's vorticity = your -z
    ez_paper =  dirs[:, 1]   # paper's gradient  = your y
    Theta = np.arcsin(np.clip(-ez_paper, -1, 1))
    Psi   = np.arctan2(ey_paper, ex_paper)
    return Psi, Theta

# ===================== Precompute bin edges and area correction =====================
psi_edges = np.linspace(-np.pi, np.pi, n_psi_bins + 1)
th_edges  = np.linspace(-np.pi/2, np.pi/2, n_th_bins + 1)
psi_centers = 0.5 * (psi_edges[:-1] + psi_edges[1:])
th_centers  = 0.5 * (th_edges[:-1]  + th_edges[1:])

dPsi_val   = np.diff(psi_edges)                      # (n_psi,)
dTheta_val = np.diff(th_edges)                       # (n_th,)
cos_th     = np.cos(th_centers)                      # (n_th,)
bin_area   = np.outer(dPsi_val, dTheta_val * cos_th) # (n_psi, n_th)

# ===================== Precompute vmax from full dataset =====================
dirs_all_flat = strobo_all.reshape(-1, 3)
Psi_all, Theta_all = dirs_to_psi_theta(dirs_all_flat)
H_ref, _, _ = np.histogram2d(Psi_all, Theta_all,
                              bins=[n_psi_bins, n_th_bins],
                              range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]])
H_ref_area = H_ref / bin_area
H_ref_plot = H_ref_area / np.sum(H_ref_area * bin_area)
vmax_fixed = H_ref_plot.max()

# ===================== Build animation frames =====================
n_beats_available = len(beat_indices_all)

if mode == "snapshot":
    frame_beat_slices = [[b] for b in range(0, n_beats_available, stride_beats)]
elif mode == "cumulative":
    frame_beat_slices = [list(range(0, b + 1)) for b in range(0, n_beats_available, stride_beats)]
elif mode == "sliding":
    frame_beat_slices = [
        list(range(max(0, b - window_beats + 1), b + 1))
        for b in range(0, n_beats_available, stride_beats)
    ]

# ===================== Set up figure =====================
fig, ax = plt.subplots(figsize=(9, 6))

H0 = np.zeros((n_psi_bins, n_th_bins))
pcm = ax.pcolormesh(psi_edges, th_edges, H0.T,
                    shading="auto", cmap="turbo",
                    vmin=0, vmax=vmax_fixed)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("PDF")

ax.set_xlabel(r"$\Psi$")
ax.set_ylabel(r"$\Theta$")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_yticks([-np.pi/2, 0, np.pi/2])
ax.set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])
ax.grid(alpha=0.2)
title = ax.set_title("")

# ===================== Update function =====================
def update(frame_idx):
    beat_slice = frame_beat_slices[frame_idx]
    dirs = strobo_all[:, beat_slice, :].reshape(-1, 3)

    Psi, Theta = dirs_to_psi_theta(dirs)

    H, _, _ = np.histogram2d(Psi, Theta,
                              bins=[n_psi_bins, n_th_bins],
                              range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]])
    H_area = H / bin_area
    H_plot = H_area / np.sum(H_area * bin_area)

    pcm.set_array(H_plot.T.ravel())

    beat_start = beat_indices_all[beat_slice[0]]
    beat_end   = beat_indices_all[beat_slice[-1]]
    if mode == "snapshot":
        title.set_text(f"Orientational Distribution — beat {beat_end} ($\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$)")
    else:
        title.set_text(f"Orientational Distribution — beats {beat_start}–{beat_end} ($\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$)")

    return pcm, title

# ===================== Animate =====================
ani = animation.FuncAnimation(fig, update, frames=len(frame_beat_slices),
                               interval=1000 / fps, blit=False)
plt.tight_layout()

if save_video:
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(output_path, writer=writer, dpi=dpi)
    print(f"Saved video to {output_path}")
else:
    plt.show()