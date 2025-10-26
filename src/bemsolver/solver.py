import numpy as np

from dataclasses import dataclass

from .mesh import Mesh
from .utils import find_panel_data

from .kernels import stresslet, line_singularity,stresslet_fast, line_singularity_fast
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

        for i in range(N):
            if i % 10 == 0:
                print(f"computing panel {i} out of {N}")
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


        # Total contribution of the stresslet, stokeslet, and rotlet of the current element on each collocation point 
        singularities    = np.zeros((3*numevals,3))  

        Int = coord @ centroid      # Center of current surface element for integration

        
        for i, eval_point in enumerate(evaluation_points):
            
            Col = coord @ eval_point    # Collocation point
            
            
            T=stresslet(Col, Int, Xq, Yq, Wx, Wy)


            # Map the quadrature points of the element to the centerline of the mesh sucht that
            # the quadrature becomes a line integration instead of surface integration.

            # line_scale makes sure that the line distribution does not span the entire centerline,
            # but makes it a bit smaller. If line_scale is set to zero, the line distribution of singularities
            # collapses to a point singularity. XG is the middle of the line
            R   =   self.mesh.parameters["line_scale"] * (centroid[0] 
                                                          + Xq * coord[0,0] 
                                                          + Yq * coord[1,0] 
                                                          - self.mesh.parameters["XG"]) + self.mesh.parameters["XG"]


            S, G = line_singularity(Col, Int, coord, R, Xq, Yq, Wx, Wy)

            singularities[i:i+3] = coord.T @ ( 3/(4*np.pi) * T + 1/(8*np.pi) * (S + G) ) @ coord










    