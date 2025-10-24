from scipy.io import loadmat
import warnings
from dataclasses import dataclass, field
import numpy as np
import bemsolver as BEM

# warnings.simplefilter("once", category=UserWarning)

path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/spheroid_mesh_b=0.55.mat"
newpath="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"



    

mesh=BEM.Mesh(path)
# mesh.parameters["line_scale"]
# mesh.panels

mesh.plot_mesh()