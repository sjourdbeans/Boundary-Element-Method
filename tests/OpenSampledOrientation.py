import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

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

shear_rate =9

elements = 320  
ratio = 5.8
scale_out_of_plane = 1.2#1.2

scale_amp =1.4
h=0.6
drho=30

N_swimmers=4500#4500
periods = 400#400

# periods=400
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
# fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Euglena/Rossi/Free/Euglena_N=320_experimental.pkl"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/gravitaxis/mesh={elements}"
main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/experimental"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/scale-out-of-plane"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/scale-amplitude"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/gravitaxis"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/zero_thrust"
# main_folder = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio=5.8/mesh=320"
# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/gravitaxis"


# main_folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/euglena"


# folder=f"{main_folder}/scale={scale_out_of_plane}_shear={shear_rate}_N=4500_periods_140"

output_folder =f"{main_folder}/sampled"
# output_file = f"{output_folder}/shear={shear_rate}_N={N_swimmers}_periods_{periods}_h={h}_drho={drho}.h5"
output_file = f"{output_folder}/mesh={elements}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/scale={scale_out_of_plane}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/scale={scale_out_of_plane}_shear={shear_rate}_h={h}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/mesh={elements}_shear={shear_rate}_h={h}_N={N_swimmers}_periods_{periods}.h5"
# output_file = f"{output_folder}/amp={scale_amp}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"


discard_beats  = 0
pdf_last_beats =  74#int(periods) if periods>100 else periods # how many final beats to use for the PDF
n_psi_bins     = 40
n_th_bins      = 20
n_psi_pdf      = 240
n_th_pdf       = 120
min_counts     = 1

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

norms = np.linalg.norm(quaternions, axis=2)

print(len(np.where(np.all(norms > 0, axis=1))[0]))

N_swimmers, periods, _ = quaternions.shape  # (N, N_beats, 4)
# ===================== Director extraction =====================
def quat_to_director(q):
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])

# ===================== Coordinate conversion =====================
def dirs_to_psi_theta(dirs):
    ex_paper =  dirs[:, 0]
    ey_paper = -dirs[:, 2]
    ez_paper =  dirs[:, 1]
    Theta = np.arcsin(np.clip(ez_paper, -1, 1))
    Psi   = np.arctan2(ey_paper, ex_paper)
    return Psi, Theta

def calc_angles_old(quats):
    # Step 1: get director from quaternion
    r = R.from_quat(quats, scalar_first=True)
    
    roll, phi, psi = r.as_euler('xzy' ).T
    return psi, phi


# ===================== Extract stroboscopic directors =====================
# All beats after discard (used for streamlines)
beat_indices_all = np.arange(discard_beats, periods)
# Last N beats only (used for PDF)
beat_indices_pdf = np.arange(periods - pdf_last_beats, periods)

all_dirs = np.array([
    [quat_to_director(quaternions[i, b , :])
     for b in beat_indices_all]
    for i in range(N_swimmers)
])  # shape: (N_swimmers, N_beats_all, 3)

# PDF dirs: just slice the last pdf_last_beats from all_dirs
pdf_dirs = all_dirs[:, -pdf_last_beats:, :]  # (N_swimmers, pdf_last_beats, 3)

norms = np.linalg.norm(quaternions, axis=2)  # shape (n_traj, n_time)
nonzero_idx = np.where(np.all(norms > 0, axis=1))[0]
quats=quaternions[nonzero_idx]
pdf_quats = quats[:, -pdf_last_beats:, :]

psi_q, phi_q =calc_angles_old(pdf_quats)

# ===================== Compute PDF on fine grid =====================
psi_edges_pdf = np.linspace(-np.pi, np.pi,     n_psi_pdf + 1)
th_edges_pdf  = np.linspace(-np.pi/2, np.pi/2, n_th_pdf  + 1)

dirs_flat_pdf = pdf_dirs.reshape(-1, 3)
# Psi_pdf, Theta_pdf = dirs_to_psi_theta(dirs_flat_pdf)
Psi_pdf, Theta_pdf = psi_q.flatten(), phi_q.flatten()

H_pdf, _, _ = np.histogram2d(Psi_pdf, Theta_pdf,
                              bins=[n_psi_pdf, n_th_pdf],
                              range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]])

dPsi_pdf       = np.diff(psi_edges_pdf)
dTheta_pdf     = np.diff(th_edges_pdf)
th_centers_pdf = 0.5 * (th_edges_pdf[:-1] + th_edges_pdf[1:])
cos_th_pdf     = np.cos(th_centers_pdf)
bin_area_pdf   = np.outer(dPsi_pdf, dTheta_pdf * cos_th_pdf)

H_area_pdf = H_pdf / bin_area_pdf
PDF = H_area_pdf / np.sum(H_area_pdf * bin_area_pdf)

# Mask empty bins so they render as white
PDF_masked = np.where(H_pdf > 0, PDF, np.nan)

# # ===================== Compute flow field on coarse grid =====================
psi_edges   = np.linspace(-np.pi, np.pi,     n_psi_bins + 1)
th_edges    = np.linspace(-np.pi/2, np.pi/2, n_th_bins  + 1)
psi_centers = 0.5 * (psi_edges[:-1] + psi_edges[1:])
th_centers  = 0.5 * (th_edges[:-1]  + th_edges[1:])

sum_dPsi   = np.zeros((n_psi_bins, n_th_bins))
sum_dTheta = np.zeros((n_psi_bins, n_th_bins))
counts     = np.zeros((n_psi_bins, n_th_bins), dtype=int)

for i in range(N_swimmers):
    dirs = all_dirs[i]  # (N_beats_all, 3)

    Psi, Theta = dirs_to_psi_theta(dirs)

    dPsi   = np.diff(Psi)
    dPsi   = (dPsi + np.pi) % (2 * np.pi) - np.pi
    dTheta = np.diff(Theta)

    i_psi = np.clip(np.searchsorted(psi_edges, Psi[:-1], side='right') - 1, 0, n_psi_bins - 1)
    i_th  = np.clip(np.searchsorted(th_edges,  Theta[:-1], side='right') - 1, 0, n_th_bins  - 1)

    np.add.at(sum_dPsi,   (i_psi, i_th), dPsi)
    np.add.at(sum_dTheta, (i_psi, i_th), dTheta)
    np.add.at(counts,     (i_psi, i_th), 1)

with np.errstate(invalid='ignore'):
    avg_dPsi   = np.where(counts >= min_counts, sum_dPsi   / counts, np.nan)
    avg_dTheta = np.where(counts >= min_counts, sum_dTheta / counts, np.nan)

# ===================== Fill NaNs for streamplot =====================
PSI, TH = np.meshgrid(psi_centers, th_centers, indexing='ij')

def fill_nans(arr):
    valid = ~np.isnan(arr)
    if not valid.any():
        return arr
    pts    = np.column_stack([PSI[valid], TH[valid]])
    vals   = arr[valid]
    filled = griddata(pts, vals, (PSI, TH), method='nearest')
    out = arr.copy()
    out[~valid] = filled[~valid]
    return out

U = fill_nans(avg_dPsi)
V = fill_nans(avg_dTheta)



# ===================== Plot =====================
fig, ax = plt.subplots(figsize=(7, 4))

# White background so empty bins show as white
ax.set_facecolor('white')


from matplotlib.colors import LinearSegmentedColormap

# Take inferno from 0.1 upward, skipping the near-black start
cmap = LinearSegmentedColormap.from_list(
    'turbo_clipped',
    plt.cm.turbo(np.linspace(0, 1.0, 256))
)
# PDF as background with NaN -> white
# cmap = plt.get_cmap('turbo').copy()
cmap.set_bad(color='white')
pcm = ax.pcolormesh(psi_edges_pdf, th_edges_pdf, PDF_masked.T,
                    shading='auto', cmap=cmap,vmax=0.4)
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("PDF")

# Streamlines on top
ax.streamplot(
    psi_centers, th_centers,
    U.T, V.T,
    color='black',
    linewidth=1.0,
    density=1.5,
    arrowsize=1.2,
)

ax.set_xlabel(r"$\Psi$")
ax.set_ylabel(r"$\Phi$")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_yticks([-np.pi/2, 0, np.pi/2])
ax.set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])
ax.set_title(f"Orientational Distribution, $\\dot{{\\gamma}}={shear_rate}$ rad s$^{{-1}}$ over {pdf_last_beats} Beats")
# ax.set_title(f"Orientational Distribution Rigid Particles with $\\alpha$={ratio} and $\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$")
ax.grid(alpha=0.2)

plt.tight_layout()
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/chlamy/3D/symmetric/Distributions/Distribution_shear={shear_rate}_period={periods}_N={N_swimmers}.pdf")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/chlamy/3D/symmetric/Distributions/Distribution_shear={shear_rate}_period={periods}_N={N_swimmers}.png", dpi=600)
# plt.show()

azimuth_pi = np.arctan2(dirs_flat_pdf[:, 1], dirs_flat_pdf[:, 0]) / np.pi
theta = np.arccos(np.clip(-dirs_flat_pdf[:, 2], -1, 1))
theta_pi = theta / np.pi

# 2D histogram with raw counts
H, th_edges, az_edges = np.histogram2d(
    theta_pi,
    azimuth_pi,
    bins=[120, 240],
    range=[[0, 1], [-1, 1]]
)

# Convert edges back to radians for area calculation
az_edges_rad = az_edges * np.pi
th_edges_rad = th_edges * np.pi

# Calculate bin areas on the sphere
dphi = np.diff(az_edges_rad)[None, :]  # (n_az, 1) delta in azimuthal angle
dA_theta = (np.cos(th_edges_rad[:-1]) - np.cos(th_edges_rad[1:]))[:, None]  # (1, n_th) delta in polar angle
bin_area = dphi * dA_theta  # (n_th, n_az) bin areas

# Area-corrected density
H_area = H / bin_area

# Optional normalization so the sum of all densities is 1
H_plot = H_area / np.sum(H_area * bin_area)

PDF_masked = H_plot#np.where(H > 0, H_plot, np.nan)

# Plotting
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor('white')
# Plot using pcolormesh, shift bins to match the center of each bin
pcm = ax.pcolormesh(
    th_edges[:-1] + np.diff(th_edges) / 2,  # shift to bin centers
    az_edges[:-1] + np.diff(az_edges) / 2,  # shift to bin centers
    PDF_masked.T,  # Transpose because we want azimuth on x and polar on y
    shading="auto",
    cmap="turbo", vmax=0.4
)

# Add color bar for PDF
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("PDF")

# Label axes and set title
ax.set_ylabel(r"In-Plane Angle $\phi / \pi$")
ax.set_xlabel(r"Vorticity Angle $\theta / \pi$")
ax.set_title(f"Orientational Distribution Chlamy with $\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$")
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)

# Optional grid and layout adjustments
ax.grid(alpha=0.2)
plt.tight_layout()
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/chlamy/3D/non-symmetric/gravitaxis/Distribution_shear={shear_rate}_period={periods}_N={N_swimmers}_h={h}_sowmya_coords.pdf")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/chlamy/3D/non-symmetric/gravitaxis/Distribution_shear={shear_rate}_period={periods}_N={N_swimmers}_h={h}_sowmya_coords.png", dpi=600)
# plt.tight_layout()
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/chlamy/3D/Distributions/2D/Distribution_mesh=320_shear={shear_rate}_N={N_conditions}_periods_{periods}.pdf")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/chlamy/3D/Distributions/2D/Distribution_mesh=320_shear={shear_rate}_N={N_conditions}_periods_{periods}.png",dpi=600)
plt.show()