from scipy.io import loadmat
import warnings
from dataclasses import dataclass, field
import numpy as np
import bemsolver as BEM

# warnings.simplefilter("once", category=UserWarning)

path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/spheroid_mesh_b=0.55.mat"
newpath="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"

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

sys=BEM.System(mesh)

MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()


