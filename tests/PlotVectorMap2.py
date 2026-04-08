import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata

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
shear_rate = 15
ratio = 5.8
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
# folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio={ratio}/mesh=320/shear={shear_rate}_N=4500_periods_140"

# folder=f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/mesh=320_shear={shear_rate}_N=4500_periods_140"

# folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
savefile = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/chlamy/3D/symmetric/vectorplots/shear={shear_rate}_N=4500_periods_140.png"

discard_beats  = 10
pdf_last_beats = 10   # how many final beats to use for the PDF
n_psi_bins     = 40
n_th_bins      = 20
n_psi_pdf      = 240
n_th_pdf       = 120
min_counts     = 5

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
quaternions        = np.zeros((N_conditions, frames, 4), dtype=np.float32)

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
N_swimmers = quaternions.shape[0]

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
    Theta = np.arcsin(np.clip(-ez_paper, -1, 1))
    Psi   = np.arctan2(ey_paper, ex_paper)
    return Psi, Theta

# ===================== Extract stroboscopic directors =====================
# All beats after discard (used for streamlines)
beat_indices_all = np.arange(discard_beats, periods)
# Last N beats only (used for PDF)
beat_indices_pdf = np.arange(periods - pdf_last_beats, periods)

all_dirs = np.array([
    [quat_to_director(quaternions[i, b * frames_per_beat, :])
     for b in beat_indices_all]
    for i in range(N_swimmers)
])  # shape: (N_swimmers, N_beats_all, 3)

# PDF dirs: just slice the last pdf_last_beats from all_dirs
pdf_dirs = all_dirs[:, -pdf_last_beats:, :]  # (N_swimmers, pdf_last_beats, 3)

# ===================== Compute PDF on fine grid =====================
psi_edges_pdf = np.linspace(-np.pi, np.pi,     n_psi_pdf + 1)
th_edges_pdf  = np.linspace(-np.pi/2, np.pi/2, n_th_pdf  + 1)

dirs_flat_pdf = pdf_dirs.reshape(-1, 3)
Psi_pdf, Theta_pdf = dirs_to_psi_theta(dirs_flat_pdf)

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

# ===================== Compute flow field on coarse grid =====================
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
fig, ax = plt.subplots(figsize=(9, 6))

# White background so empty bins show as white
ax.set_facecolor('white')


from matplotlib.colors import LinearSegmentedColormap

# Take inferno from 0.1 upward, skipping the near-black start
cmap = LinearSegmentedColormap.from_list(
    'turbo_clipped',
    plt.cm.turbo(np.linspace(0.2, 1.0, 256))
)
# PDF as background with NaN -> white
# cmap = plt.get_cmap('turbo').copy()
cmap.set_bad(color='white')
pcm = ax.pcolormesh(psi_edges_pdf, th_edges_pdf, PDF_masked.T,
                    shading='auto', cmap=cmap,vmax=1)
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
ax.set_ylabel(r"$\Theta$")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_yticks([-np.pi/2, 0, np.pi/2])
ax.set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])
ax.set_title(f"Orientational flow field, $\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$ (PDF: last {pdf_last_beats} beats)")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(savefile, dpi=600)
plt.show()