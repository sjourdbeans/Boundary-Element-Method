
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
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"

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

sys=BEM.ResistanceProblem(mesh)

psi, force, torque = sys.solve(U,W)


figs, axes, cbars=sys.plot_singularity_density()

import matplotlib.pyplot as plt

figs, axes, cbars=sys.plot_singularity_density()

component=['$x$', '$y$', '$z$']




for i, (fig, ax, cbar) in enumerate(zip(figs,axes, cbars)):

    
    fig.set_size_inches(10,7)
    ax.set_title(component[i]+r" Component of the Singularity Density $\psi$", pad = 0)
    ax.view_init(elev=20, azim=30)
    ax.set_xlabel(r'$x$ [$\mu$m]',labelpad=15)
    ax.set_ylabel(r'$y$ [$\mu$m]',labelpad=15)
    ax.set_zlabel(r'$z$ [$\mu$m]',labelpad=20)
         

    ax.tick_params(axis='both', pad=8)
    ax.tick_params(axis='z', pad=6)

    cbar.set_label(r"Singularity Strength Coefficient", rotation=-90, labelpad=15)

    fig.subplots_adjust(left=0.0, right=0.95, top=0.95, bottom=0.0)

    set_axes_equal(ax)
    fig.savefig(f"{plot_path}/Singularity_density_{component[i][1]}.pdf")
    fig.savefig(f"{plot_image_path}/Singularity_density_{component[i][1]}.png",dpi=600)



# MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()
# print(force)

