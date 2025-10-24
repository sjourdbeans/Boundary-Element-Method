import numpy as np

from dataclasses import dataclass, field

from .mesh import Mesh
from .utils import find_panel_data

from .kernels import stresslet
from .quadrature import triquad





@dataclass
class Solver:

    mesh:Mesh

    def __post_init__(self):
        pass

    def construct_mobility_matrix(self):
        """
        Construct the mobility matrix of the geometry in the mesh interacting with itself 
        """

        N=self.mesh.elements


        # keep in mind that the amount of rows depend on collocation points
        # which in this case is the same as the elements but that is not always the case
        MATRIX = np.zeros((3*N,3*N))
        surface_matrix=np.zeros((3,3*N))
        torque_matrix=np.zeros((3,3*N))

        for i in range(N)[:2]:
            if N%10==0:
                print(f"Computing panel {i} of {N}")
            panel=self.mesh.panels[1:,:,i]

            self.calc_mobility_contribution(panel,self.mesh.centroids)

            
        

    def calc_mobility_contribution(self,
                                   panel:np.ndarray,
                                   evaluation_points:np.ndarray
                                   )->tuple[np.ndarray, np.ndarray, np.ndarray]: 

        X,Y,Z,centroid =  find_panel_data(panel)  #from utils.py

        numevals=len(evaluation_points)
        # Assemble the coordinate frame
        coord = np.vstack([X,Y,Z])

        npanel=np.zeros(np.shape(panel))

        for i,vert in enumerate(panel):
            npanel[i]=(coord @ (vert-centroid))


        #=============Compute quadrature points for the panel=============

        # Instead of using modules like quadpy, we do the quadrature ourselves because we want to avoid singularities.
        # This way the difference between the collocation point and centroid is never 0 (so we don't divide by 0)
        Xq, Yq, Wx, Wy = triquad(3, npanel[:, 0:2])
        Zq             = np.zeros(np.shape(Xq))

        #============Assemble the matrices for the stresslets and line distribution===========

        stresslet_array = np.zeros((3*numevals,3))
        stokeslet_array = np.zeros((3*numevals,3))
        rotlet_array    = np.zeros((3*numevals,3))  

        for i, eval_point in enumerate(evaluation_points):
            
            Col = coord @ eval_point    # Collocation point
            Int = coord @ centroid      # Center of current surface element for integration
            
            xx  = Col[0] - (Int[0]+Xq)
            yy  = Col[1] - (Int[1]+Yq)
            zz  = Col[2] -  Int[2]  
            
             




        



            
        print(Wx,Wy)
            # print(npanel[i])
        # print(npanel)




    