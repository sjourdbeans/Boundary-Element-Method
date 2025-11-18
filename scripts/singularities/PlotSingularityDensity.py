
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
mpl.rcParams["font.family"]= "DejaVu Sans"
# mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"

mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13



path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"

plot_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Singularity-density"
plot_image_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Singularity-density"

#======================================================================

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
    ax.set_title(component[i]+r" Component of the Singularity Density $\psi$",fontsize=20, pad = 0)
    ax.view_init(elev=20, azim=30)
    ax.set_xlabel(r'$x$ [$\mu$m]',labelpad=15)
    ax.set_ylabel(r'$y$ [$\mu$m]',labelpad=15)
    ax.set_zlabel(r'$z$ [$\mu$m]',labelpad=15)
    ax.zaxis.set_rotate_label(False) 

    ax.tick_params(axis='both', pad=8)
    ax.tick_params(axis='z', pad=6)

    cbar.set_label(r"Singularity Strength Coefficient", rotation=-90, labelpad=15)

    fig.subplots_adjust(left=0.0, right=0.95, top=0.95, bottom=0.0)
    fig.savefig(f"{plot_path}/Singularity_density_{component[i][1]}.pdf")
    fig.savefig(f"{plot_image_path}/Singularity_density_{component[i][1]}.png",dpi=600)



# MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()
# print(force)

