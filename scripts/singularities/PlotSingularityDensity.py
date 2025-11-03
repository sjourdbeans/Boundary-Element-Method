
import numpy as np
import bemsolver as BEM
import matplotlib as mpl


# # Set tick direction globally
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['xtick.top'] = True
# mpl.rcParams['ytick.right'] = True

# mpl.rcParams['xtick.minor.visible'] = True
# mpl.rcParams['ytick.minor.visible'] = True

import os
os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=20
mpl.rcParams["legend.fontsize"]=13



path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"

#======================================================================

# Background flow
U = np.zeros(3)

U[0] = 1
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


figs,axes=sys.plot_singularity_density()

component=['$x$', '$y$', '$z$']

for i, fig in enumerate(figs):
    axes[i].set_title(component[i]+r" Component of the Singularity Density $\psi$")

import matplotlib.pyplot as plt

plt.show()



# MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()
# print(force)

