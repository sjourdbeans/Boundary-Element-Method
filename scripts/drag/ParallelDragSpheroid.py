from scipy.io import loadmat
import warnings
from dataclasses import dataclass, field
import numpy as np
import bemsolver as BEM
import os

folder_path = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/new-variation"

entries = os.listdir(folder_path)


files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


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

force_x = np.zeros(len(files))
b_arr =np.zeros(len(files))
    
for i, file in enumerate(files):
    mesh=BEM.Mesh(file)
    b_arr[i]=mesh.b

    sys=BEM.System(mesh)

    psi, force, torque = sys.solve(U,W)

    force_x[i]=force[0]

np.savetxt("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/prolate-spheroids/force_vs_x.txt",
            np.vstack((b_arr,force_x)).T)



# MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()
# print(force)

