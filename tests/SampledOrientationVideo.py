import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import matplotlib.animation as animation

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

shear_rate = 9
elements = 320
ratio = 5.8
scale_out_of_plane = 0.5
scale_amp=1.02

N_swimmers=4500
periods =140#400
h=0.6
drho=30

fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
# fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Euglena/Rossi/Free/Euglena_N=320_experimental.pkl"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/euglena"
main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/gravitaxis/mesh={elements}"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/experimental"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/scale-out-of-plane"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/scale-amplitude"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats"
# main_folder = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio=5.8/mesh=320"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/oop_gravitaxis"


# folder=f"{main_folder}/scale={scale_out_of_plane}_shear={shear_rate}_N=4500_periods_140"

output_folder =f"{main_folder}/sampled"
# output_file = f"{output_folder}/scale={scale_out_of_plane}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/amp={scale_amp}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
output_file = f"{output_folder}/mesh={elements}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/shear={shear_rate}_N={N_swimmers}_periods_{periods}_h={h}_drho={drho}.h5"
# output_file = f"{output_folder}/shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/scale={scale_out_of_plane}_shear={shear_rate}_h={h}_N={N_swimmers}_periods_{periods}.h5"

discard_beats  = 0
pdf_last_beats = 10       # how many final beats to use for the PDF
n_psi_bins     = 240
n_th_bins      = 120
n_psi_pdf      = 240
n_th_pdf       = 120
min_counts     = 5



mode         = "sliding"
window_beats = 10
stride_beats = 1

save_video  = True
output_path = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Chlamy/Trajectories/symmetric/distribution-shear={shear_rate}.mp4"
dpi         = 150
fps=10

with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

frames_per_beat = swimmer_template.N_frames


with h5py.File(output_file, "r") as f:
    beat_indices = f["beat_indices"][:]
    
    # Time-resolved stroboscopic data
    quaternions = f["stroboscopic/quaternions"][:]   # (N, N_beats, 4)
    omega       = f["stroboscopic/omega"][:]         # (N, N_beats, ...)
    X           = f["stroboscopic/X"][:]             # (N, N_beats, ...)

    # Static data
    initial_orientation = f["static/initial_orientation"][:]

# =============Plot position==================
# step =40

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111,projection='3d')

# # ax.scatter(minima[:,0],minima[:,1], minima[:,2], color=colors)
# ax.scatter(X[:,step,0],X[:,step,1],X[:,step,2])
# # ax.scatter(minima[:,0],minima[:,1], minima[:,2], color='r')
# # ax.quiver(minima[:,0],minima[:,1], minima[:,2], omega_min[:,0],omega_min[:,1],omega_min[:,2], colors=colors, norm=norm, length=0.05,arrow_length_ratio=2)
# ax.set_xlabel(r"x")
# ax.set_ylabel(r"y")
# ax.set_zlabel(r"z")
# # ax.view_init(azim=0,elev=0)

# plt.show()

N_swimmers, periods, _ = quaternions.shape  # (N, N_beats, 4)
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
beat_indices_pdf = np.arange(periods - pdf_last_beats, periods)
strobo_all = np.array([
    [quat_to_director(quaternions[i, b , :])
     for b in beat_indices_all]
    for i in range(N_swimmers)
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
    Theta = np.arcsin(np.clip( ez_paper, -1, 1))
    Psi   = np.arctan2(ey_paper, ex_paper)
    return Psi, -Theta

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
                    vmin=0,vmax=1)#, vmax=vmax_fixed)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("PDF")

ax.set_xlabel(r"$\Psi$")
ax.set_ylabel(r"$\Phi$")
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