import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl
import os
import matplotlib.pyplot as plt


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

shear_rate = 6
scale = 1
# fileswimmer = "/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_1280.pkl"
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
# fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Euglena/Rossi/Free/Euglena_N=320_experimental.pkl"
with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

# outfile=f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/euglena/mesh=320/shear={shear_rate}_N=20000_periods_32.npz"
# outfile=f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/rigid-particles/ratio=1.25/distributions/shear={shear_rate}_N=4500_periods_140.npz"
# outfile = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-2d/non-symmetric/distributions/mesh=320_shear={shear_rate}_N=4500_periods_140.npz"
# outfile = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/non-symmetric/scale-out-of-plane/scale={scale}_shear={shear_rate}_N=4500_periods_140.npz"
# outfile = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/distributions/mesh=320_shear={shear_rate}_N=4500_periods_140.npz"
# outfile=f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/symmetric/zero_thrust/mesh=320_shear={shear_rate}_N=4500_periods_140.npz"
outfile = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/distributions/mesh=320/shear={shear_rate}_N=4500_periods_140.npz"
outdir =  Path(outfile)


frames_per_beat = swimmer_template.N_frames

periods = int(outfile.split("_")[-1].split(".")[0])


import numpy as np
import plotly.graph_objects as go

data = np.load(outdir)
H = data["H"]
th_edges = data["th_edges"]
az_edges = data["az_edges"]

az_edges_rad = az_edges * np.pi
th_edges_rad = th_edges * np.pi

dphi   = np.diff(az_edges_rad)
dA_th  = np.cos(th_edges_rad[:-1]) - np.cos(th_edges_rad[1:])
bin_area = dA_th[:, None] * dphi[None, :]   # shape (n_th, n_az)

H_area = H / bin_area
H_plot = H_area / np.sum(H_area * bin_area)



fig, ax = plt.subplots(figsize=(9, 6))

pcm = ax.pcolormesh(
    th_edges[:-1] + np.diff(th_edges) / 2,  # shift to bin centers
    az_edges[:-1] + np.diff(az_edges) / 2,  # shift to bin centers,
    H_plot.T,
    shading="auto",
    cmap="turbo",vmax=1
)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("PDF")

ax.set_ylabel(r"In-Plane Angle $\phi / \pi$")
ax.set_xlabel(r"Vorticity Angle $\theta$")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesi \pi$")
ax.set_title(
    f"Orientational Distribution Chlamy Symmetric with $\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$"
)
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.grid(alpha=0.2)

plt.tight_layout()
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/rigid-particles/ratio=1.25/Distribution_mesh=320_shear={shear_rate}_N=4500_periods_{periods}.pdf")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/rigid-particles/ratio=1.25/Distribution_mesh=320_shear={shear_rate}_N=4500_periods_{periods}.png",dpi=600)
plt.show()

#==========Plotly===========

#mid-bin angles
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
#     cmax=50,    # upper bound
#     colorbar=dict(title='density', thickness=15),
#     lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1, roughness=0.8)
# ))

# fig.update_layout(
#     scene=dict(
#         xaxis_title='x', yaxis_title='y', zaxis_title='z',
#         aspectmode='cube',
#     ),
#     margin=dict(l=0, r=0, t=20, b=0),
# )
# fig.show()