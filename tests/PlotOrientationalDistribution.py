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

shear_rate = 10
# fileswimmer = "/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_1280.pkl"
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
# fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Euglena/Rossi/Free/Euglena_N=320_experimental.pkl"
with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

# file = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/shear=0.0_N=8_periods_10/rank_000.h5"
# folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
# folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
folder=f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/mesh=320_shear={shear_rate}_N=4500_periods_140"
# folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/euglena/mesh=320_shear={shear_rate}_N=20000_periods_32"
# folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio=5.8/mesh=320/shear={shear_rate}_N=4500_periods_140"
# folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio=1.25/shear={shear_rate}_N=10_periods_140"
# folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/quarter_sphere/mesh=320_shear={shear_rate}_N=1100_periods_140"
outdir =  Path(folder)

manifest = np.loadtxt(folder + "/manifest.txt", dtype=str, delimiter="\t")
N_conditions = len(manifest)

frames_per_beat = swimmer_template.N_frames

periods = int(folder.split("_")[-1])

frames = frames_per_beat*periods + 1
initial_conditions = np.zeros((N_conditions, 4), dtype=np.float32) 
quaternions = np.zeros((N_conditions, frames, 4), dtype=np.float32)


for rank_file in sorted(outdir.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:

        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])
            initial_conditions[sim_idx,:] = grp["initial_orientation"][:]
            Q = grp["quaternions"][:]
            quaternions[sim_idx, :, :] = Q
            # process here



# =========== Exclude duplicates ============

initial_q = quaternions[:, 0, :]

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

initial_q_canon = canonicalize_quaternion_sign(initial_q)

unique_q, unique_idx = np.unique(
    np.round(initial_q_canon, 6),
    axis=0,
    return_index=True
)

unique_idx = np.sort(unique_idx)

quaternions = quaternions[unique_idx, :, :]

# valid_idx = np.where((initial_conditions[:, 0] < 0) &   (initial_conditions[:, 1] > 0) & (initial_conditions[:, 2] > 0))[0]
# print(valid_idx)
# initial_q = quaternions[valid_idx, 0, :]
# initial_q_canon = canonicalize_quaternion_sign(initial_q)

# _, unique_subidx = np.unique(
#     np.round(initial_q_canon, 6),
#     axis=0,
#     return_index=True
# )

# unique_idx = valid_idx[np.sort(unique_subidx)]

# quaternions = quaternions[unique_idx, :, :]
# initial_conditions = initial_conditions[unique_idx]

# =================================
import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def quat_to_director(q):
    """First column of rotation matrix = swimmer symmetry axis in lab frame."""
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])
step=0

plot_steps = 1

new_frames = 20 *frames_per_beat

discard_beats = 0
strobo_idx = np.arange(discard_beats * frames_per_beat, new_frames, frames_per_beat)

# strobo_idx = np.arange(0, frames, frames_per_beat)
strobo_quats = quaternions[::plot_steps, strobo_idx, :]

# Convert quaternions to directors
directors = np.array([[quat_to_director(q) for q in ic] for ic in strobo_quats])

# Flatten all sampled orientations
dirs_flat = directors.reshape(-1, 3)

# # ======== Angle distribution ====================
#Angular coordinates
# azimuth = np.degrees(np.arctan2(dirs_flat[:, 1], dirs_flat[:, 0]))   # [-180, 180]
# elevation = np.degrees(np.arcsin(np.clip(dirs_flat[:, 2], -1, 1)))   # [-90, 90]

# # 2D histogram
# n_az = 120
# n_el = 60

# H, az_edges, el_edges = np.histogram2d(
#     azimuth,
#     elevation,
#     bins=[n_az, n_el],
#     range=[[-180, 180], [-90, 90]]
# )

# # Plot
# fig, ax = plt.subplots(figsize=(10, 5))

# pcm = ax.pcolormesh(
#     az_edges,
#     el_edges,
#     H.T,
#     shading="auto",
#     cmap="viridis"
# )

# cbar = plt.colorbar(pcm, ax=ax)
# cbar.set_label("Counts")

# ax.set_xlabel("Azimuth [deg]")
# ax.set_ylabel("Elevation [deg]")
# ax.set_title("Orientational distribution")
# ax.set_xlim(-180, 180)
# ax.set_ylim(-90, 90)
# ax.grid(alpha=0.2)

# plt.tight_layout()
# plt.show()


#=============== z component ================
# dirs_flat = directors.reshape(-1, 3)

# azimuth in units of pi: [-1, 1]
# azimuth_pi = np.arctan2(dirs_flat[:, 1], dirs_flat[:, 0]) / np.pi
# theta = np.arccos(np.clip(dirs_flat[:, 2], -1, 1))
# theta_pi = theta / np.pi

# # 2D histogram with raw counts
# H, th_edges, az_edges = np.histogram2d(
#     theta_pi,
#     azimuth_pi,
#     bins=[120, 240],
#     range=[[0, 1], [-1, 1]]
# )

# # Convert edges back to radians for area calculation
# az_edges_rad = az_edges * np.pi
# th_edges_rad = th_edges * np.pi

# # Calculate bin areas on the sphere
# dphi = np.diff(az_edges_rad)[None, :]  # (n_az, 1) delta in azimuthal angle
# dA_theta = (np.cos(th_edges_rad[:-1]) - np.cos(th_edges_rad[1:]))[:, None]  # (1, n_th) delta in polar angle
# bin_area = dphi * dA_theta  # (n_th, n_az) bin areas

# # Area-corrected density
# H_area = H / bin_area

# # Optional normalization so the sum of all densities is 1
# H_plot = H_area / np.sum(H_area * bin_area)

# # Plotting
# fig, ax = plt.subplots(figsize=(9, 6))

# # Plot using pcolormesh, shift bins to match the center of each bin
# pcm = ax.pcolormesh(
#     th_edges[:-1] + np.diff(th_edges) / 2,  # shift to bin centers
#     az_edges[:-1] + np.diff(az_edges) / 2,  # shift to bin centers
#     H_plot.T,  # Transpose because we want azimuth on x and polar on y
#     shading="auto",
#     cmap="turbo", vmax=1
# )

# # Add color bar for PDF
# cbar = plt.colorbar(pcm, ax=ax)
# cbar.set_label("PDF")

# # Label axes and set title
# ax.set_ylabel(r"In-Plane Angle $\phi / \pi$")
# ax.set_xlabel(r"Vorticity Angle $\theta / \pi$")
# ax.set_title(f"Orientational Distribution Chlamy with $\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$")
# ax.set_xlim(0, 1)
# ax.set_ylim(-1, 1)

# # Optional grid and layout adjustments
# ax.grid(alpha=0.2)
# plt.tight_layout()

# # plt.tight_layout()
# # plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/chlamy/3D/Distributions/2D/Distribution_mesh=320_shear={shear_rate}_N={N_conditions}_periods_{periods}.pdf")
# # plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/chlamy/3D/Distributions/2D/Distribution_mesh=320_shear={shear_rate}_N={N_conditions}_periods_{periods}.png",dpi=600)
# plt.show()


# ===============CLAUDE TEST=================

# Paper's convention:
# ez = sin(Theta)  ->  Theta = arcsin(ez)
# ey = cos(Theta)*sin(Psi)  ->  Psi = arctan2(ey, -ex)

ex_paper =  dirs_flat[:, 0]   # flow: same
ey_paper = -dirs_flat[:, 2]   # paper's y (vorticity) = your -z
ez_paper =  dirs_flat[:, 1]   # paper's z (gradient)  = your y

Theta = np.arcsin(np.clip(-ez_paper, -1, 1))          # elevation from x-y plane
Psi   = np.arctan2(ey_paper, ex_paper)               # azimuth in x-y plane

H, psi_edges, th_edges = np.histogram2d(
    Psi,
    Theta,
    bins=[240, 120],
    range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]]
)

# Area correction: dA = cos(Theta) * dPsi * dTheta
dPsi  = np.diff(psi_edges)[None, :]
th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
dA = np.cos(th_centers)[:, None] * np.diff(th_edges)[:, None] * dPsi[None, :]
# Actually simpler:
dPsi_val  = np.diff(psi_edges)   # shape (n_psi,)
dTheta_val = np.diff(th_edges)   # shape (n_th,)
cos_th = np.cos(0.5 * (th_edges[:-1] + th_edges[1:]))  # shape (n_th,)

bin_area = np.outer(dPsi_val, dTheta_val * cos_th)  # shape (n_psi, n_th)

H_area = H / bin_area
H_plot = H_area / np.sum(H_area * bin_area)

fig, ax = plt.subplots(figsize=(9, 6))
pcm = ax.pcolormesh(
    psi_edges[:-1] + np.diff(psi_edges) / 2,   # Psi on x-axis
    th_edges[:-1]  + np.diff(th_edges)  / 2,   # Theta on y-axis
    H_plot.T,
    shading="auto",
    cmap="turbo", vmax=1
)
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("PDF")
ax.set_xlabel(r"$\Psi$")
ax.set_ylabel(r"$\Theta$")
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_yticks([-np.pi/2, 0, np.pi/2])
ax.set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])
plt.tight_layout()
plt.show()
#=============================================================


# az_edges_rad = az_edges * np.pi
# th_edges_rad = th_edges * np.pi

# dphi   = np.diff(az_edges_rad)
# dA_th  = np.cos(th_edges_rad[:-1]) - np.cos(th_edges_rad[1:])
# bin_area = dA_th[:, None] * dphi[None, :]   # shape (n_th, n_az)

# H_area = H / bin_area
# H_plot = H_area / np.sum(H_area * bin_area)

# import plotly.graph_objects as go
# th_mid = (th_edges[:-1] + th_edges[1:]) / 2 * np.pi
# az_mid = (az_edges[:-1] + az_edges[1:]) / 2 * np.pi

# TH, AZ = np.meshgrid(th_mid, az_mid, indexing='ij')  # (n_th, n_az)

# X = np.sin(TH) * np.cos(AZ)
# Y = np.sin(TH) * np.sin(AZ)
# Z = -np.cos(TH)

# # close the azimuthal seam
# X = np.concatenate([X, X[:, :1]], axis=1)
# Y = np.concatenate([Y, Y[:, :1]], axis=1)
# Z = np.concatenate([Z, Z[:, :1]], axis=1)
# C = np.concatenate([H_plot, H_plot[:, :1]], axis=1)

# fig = go.Figure(go.Surface(
#     x=X, y=Y, z=Z,
#     surfacecolor=C,
#     colorscale='Turbo',
#     cmin=0.0,    # lower bound
#     cmax=1,    # upper bound
#     colorbar=dict(title='density', thickness=15),
#     lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1, roughness=0.8)
# ))

# fig.update_layout(
#     font=dict(family="Palatino, Palatino Linotype, Book Antiqua, serif"),
#     scene=dict(
#         xaxis_title=r'x', yaxis_title='y', zaxis_title='z',
#         aspectmode='cube',
#     ),
#     margin=dict(l=0, r=0, t=20, b=0),
# )
# fig.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize

# az_edges_rad = az_edges * np.pi
# th_edges_rad = th_edges * np.pi
# dphi  = np.diff(az_edges_rad)
# dA_th = np.cos(th_edges_rad[:-1]) - np.cos(th_edges_rad[1:])
# bin_area = dA_th[:, None] * dphi[None, :]
# H_area = H / bin_area
# H_plot = H_area / np.sum(H_area * bin_area)

# th_mid = (th_edges[:-1] + th_edges[1:]) / 2 * np.pi
# az_mid = (az_edges[:-1] + az_edges[1:]) / 2 * np.pi
# TH, AZ = np.meshgrid(th_mid, az_mid, indexing='ij')

# X = np.sin(TH) * np.cos(AZ)
# Y = np.sin(TH) * np.sin(AZ)
# Z = np.cos(TH)

# # close azimuthal seam
# X = np.concatenate([X, X[:, :1]], axis=1)
# Y = np.concatenate([Y, Y[:, :1]], axis=1)
# Z = np.concatenate([Z, Z[:, :1]], axis=1)
# C = np.concatenate([H_plot, H_plot[:, :1]], axis=1)

# norm = Normalize(vmin=0.0, vmax=1.0)
# facecolors = cm.turbo(norm(C))

# fig = plt.figure(figsize=(7, 6))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(
#     X, Y, Z,
#     facecolors=facecolors,
#     rstride=1, cstride=1,
#     antialiased=False,
#     shade=False,
# )

# mappable = cm.ScalarMappable(norm=norm, cmap='turbo')
# mappable.set_array(C)
# cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
# cbar.set_label('density')

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$z$')
# ax.set_box_aspect([1, 1, 1])
# ax.view_init(azim=45, elev=20, vertical_axis='y')
# # plt.rcParams['font.family'] = 'Palatino'  # or 'Palatino Linotype' on Windows

# # plt.tight_layout()
# plt.savefig("figure.pdf")
# plt.show()