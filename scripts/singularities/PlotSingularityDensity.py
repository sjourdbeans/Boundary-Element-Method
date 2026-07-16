
import numpy as np
import bemsolver as BEM
import matplotlib as mpl

# Set tick direction globally
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

import os
os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "Palatino"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{upgreek}"
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{upgreek}'

mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=20
mpl.rcParams["axes.titlesize"]=25
mpl.rcParams["legend.fontsize"]=13



path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"

plot_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Singularity-density"
plot_image_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Singularity-density"

#======================================================================


def set_axes_equal(ax):

        xs = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        ranges = xs[:,1] - xs[:,0]
        centers = np.mean(xs, axis=1)
        radius = 0.5 * ranges.max()
        ax.set_xlim3d(centers[0]-radius, centers[0]+radius)
        ax.set_ylim3d(centers[1]-radius, centers[1]+radius)
        ax.set_zlim3d(centers[2]-radius, centers[2]+radius)
        ax.zaxis.set_rotate_label(False)

# Background flow
U = np.zeros(3)

U[0] = -1
U[1] = 0
U[2] = 0

# Background vorticity
W = np.zeros(3)

W[0] = 0
W[1] = 0
W[2] = 0

    
mesh=BEM.Mesh(path)

sys=BEM.FixedParticle(mesh)

psi, force, torque = sys.solve(U,W)



import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

figs, axes, cbars = sys.plot_singularity_density()

component = ['$x$', '$y$', '$z$']

# Create a single figure with 3 subplots
fig = plt.figure(figsize=(24, 7))

# Get global min/max across all three components for shared colorbar
psi = sys.psi.reshape((sys.mesh.elements, 3))
vmin = np.min(psi)
vmax = np.max(psi)

axes_new = []
for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    axes_new.append(ax)

    # Replot panels with shared normalization
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import cm, colors

    cmap = cm.Spectral_r
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    panels = sys.mesh.panels

    for ii in range(panels.shape[2]):
        panel = panels[1:, :, ii]
        n = int(panels[0, 0, ii])
        verts = panel[:n, :]
        color = cmap(norm(psi[ii, i]))
        poly = Poly3DCollection([verts], facecolors=color, edgecolors='k')
        ax.add_collection3d(poly)

    ax.set_title(component[i] + r" Component of $\boldsymbol{\psi}/\mu$", pad=0)
    ax.view_init(elev=20, azim=30)
    ax.set_xlabel(r'$x$ [$\upmu$m]', labelpad=15)
    ax.set_ylabel(r'$y$ [$\upmu$m]', labelpad=15)
    if i ==0:
        ax.set_zlabel(r'$z$ [$\upmu$m]', labelpad=15)
    

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.tick_params(axis='both', pad=8)
    ax.tick_params(axis='z', pad=6)
    # set_axes_equal(ax)

# Single shared colorbar on the right
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(psi)

fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"Singularity Strength [s$^{-1}$]",labelpad=5)
fig.set_size_inches(17, 7)
fig.savefig(f"{plot_path}/Singularity_density_combined.pdf")
fig.savefig(f"{plot_image_path}/Singularity_density_combined.png", dpi=600)

# Close the original separate figures
for f in figs:
    plt.close(f)
