from scipy.io import loadmat
import warnings
from dataclasses import dataclass, field
import numpy as np
import bemsolver as BEM

# warnings.simplefilter("once", category=UserWarning)

path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/spheroid_mesh_b=0.55.mat"
newpath="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"



    

mesh=BEM.Mesh(path)


# for c in mesh.centroids[:2]:
#     m=np.array([[1,2,3],[4,5,6],[7,8,9]])
#     print(m@c.T)


# mesh.plot_mesh()

BEM.Solver(mesh).construct_mobility_matrix()

# panel=np.array([[0,0],
#                [1,0],
#                [0.5,1]])

# Xq, Yq, Wx, Wy=BEM.triquad(2,panel)
# print(Wy)

# BEM.stresslet(np.array([0,0,0]))



# panel=mesh.panels[:,:,1]

