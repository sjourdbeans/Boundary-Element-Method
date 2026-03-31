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
shear_rate = 25
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
# folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/quarter_sphere/mesh=320_shear={shear_rate}_N=1100_periods_140"
folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
# --- Animation settings ---
# "sliding"     : window of `window_beats` beats slides forward each frame
# "cumulative"  : accumulate from beat 0, growing each frame
# "snapshot"    : show exactly one beat per frame
mode        = "sliding"
window_beats = 10        # only used for mode="sliding"
stride_beats = 1         # how many beats to advance per animation frame
discard_beats = 0       # transient beats to discard before animating
n_az_bins   = 240
n_z_bins    = 120

# Output
save_video  = True       # set False to show interactive window instead
output_path = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Chlamy/Trajectories/orientation-distribution/orientation_animation_shear={shear_rate}.mp4"
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
    """First column of rotation matrix = swimmer symmetry axis in lab frame."""
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])

# Precompute stroboscopic directors at every beat after discard
# Shape: (N_swimmers, N_beats_after_discard, 3)
total_beats = periods
beat_indices_all = np.arange(discard_beats, total_beats)  # beats available to animate
strobo_all = np.array([
    [quat_to_director(quaternions[i, b * frames_per_beat, :])
     for b in beat_indices_all]
    for i in range(quaternions.shape[0])
])
# strobo_all shape: (N_swimmers, N_available_beats, 3)

# ===================== Build animation frames =====================
# Each animation frame is a list of beat indices (into strobo_all axis=1)
n_beats_available = len(beat_indices_all)

if mode == "snapshot":
    frame_beat_slices = [
        [b] for b in range(0, n_beats_available, stride_beats)
    ]
elif mode == "cumulative":
    frame_beat_slices = [
        list(range(0, b + 1))
        for b in range(0, n_beats_available, stride_beats)
    ]
elif mode == "sliding":
    frame_beat_slices = [
        list(range(max(0, b - window_beats + 1), b + 1))
        for b in range(0, n_beats_available, stride_beats)
    ]

# ===================== Precompute colour scale =====================
# Use the full dataset to fix vmax so colorbar is stable
dirs_all_flat = strobo_all.reshape(-1, 3)
az_all = np.arctan2(dirs_all_flat[:, 1], dirs_all_flat[:, 0]) / np.pi
z_all  = dirs_all_flat[:, 2]
H_ref, _, _ = np.histogram2d(az_all, z_all,
                              bins=[n_az_bins, n_z_bins],
                              range=[[-1, 1], [-1, 1]])
vmax_fixed = 150#H_ref.max()

# ===================== Set up figure =====================
fig, ax = plt.subplots(figsize=(9, 6))

# Initial empty histogram
H0 = np.zeros((n_az_bins, n_z_bins))
az_edges = np.linspace(-1, 1, n_az_bins + 1)
z_edges  = np.linspace(-1, 1, n_z_bins + 1)

pcm = ax.pcolormesh(az_edges, z_edges, H0.T,
                    shading="auto", cmap="turbo",
                    vmin=0, vmax=vmax_fixed)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("Counts")

ax.set_xlabel(r"Azimuth $\phi / \pi$")
ax.set_ylabel(r"$\mathbf{p}\cdot\hat{\mathbf{z}}$")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.grid(alpha=0.2)

title = ax.set_title("")

# ===================== Update function =====================
def update(frame_idx):
    beat_slice = frame_beat_slices[frame_idx]

    # Gather directors for this frame's beats
    dirs = strobo_all[:, beat_slice, :].reshape(-1, 3)

    az = np.arctan2(dirs[:, 1], dirs[:, 0]) / np.pi
    z  = dirs[:, 2]

    H, _, _ = np.histogram2d(az, z,
                              bins=[n_az_bins, n_z_bins],
                              range=[[-1, 1], [-1, 1]])

    pcm.set_array(H.T.ravel())

    # Title: show actual beat numbers in simulation time
    beat_start = beat_indices_all[beat_slice[0]]
    beat_end   = beat_indices_all[beat_slice[-1]]
    if mode == "snapshot":
        title.set_text(
            f"Orientational Distribution — beat {beat_end} "
            f"($\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$)"
        )
    else:
        title.set_text(
            f"Orientational Distribution — beats {beat_start}–{beat_end} "
            f"($\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$)"
        )

    return pcm, title

# ===================== Animate =====================
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frame_beat_slices),
    interval=1000 / fps,
    blit=False
)

plt.tight_layout()

if save_video:
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(output_path, writer=writer, dpi=dpi)
    print(f"Saved video to {output_path}")
else:
    plt.show()
