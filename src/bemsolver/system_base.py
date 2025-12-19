import numpy as np
from dataclasses import dataclass

from .mesh import Mesh
from .utils import find_panel_data, U_colloc, skew_stack
from .kernels import stresslet_vectorized, line_singularity_vectorized
from .quadrature import triquad





@dataclass
class BaseSystem:
    """

    This class contains the core code that calculates the mobility matrix of the mesh used. 
    However, this class is not called by itself but it is instead used as a parent class.
    
    For example in stokes_problems.py there are two child classes of BaseSystem.
        - ResistanceProblem
        - MobilityProblem
    This means that when one of these classes is called it also needs the ``mesh`` argument,
    but then it also has access to all attributes and methods of BaseSystem.

    Parameters
    ----------
    mesh    : Mesh
              Mesh object which represents the cell body    

    Methods
    -------
    ``__post_init__()``
        Initializes the base system by constructing the mobility matrix and related matrices (e.g., surface,
        torque, and cross-product matrices) for the mesh.
    ``construct_mobility_matrix()``
        Builds the mobility matrix for the mesh, including double-layer and single-layer potentials.
    ``set_boundary_condition(U, W, E)``
        Sets the boundary conditions (translational velocity U, angular velocity W, strain rate E) on the mesh.
    """


    mesh:Mesh

    def __post_init__(self):
        self.AmountofPanelsBeforePrinting=10
        self.evaluation_points = self.mesh.centroids

        # If the collocation points are the same as the element centroids,
        # we need to use the integral equation of the second kind from Keaveny & Shelley
        self.UseSecondKindIntEquation=True



    def construct_mobility_matrix(self):
        """
        Construct the mobility matrix of the geometry in the mesh interacting with itself 
        """
        M = self.evaluation_points.shape[0]        # number of collocation points
        
        N = self.mesh.elements

        # keep in mind that the amount of rows depend on collocation points
        # which in this case is the same as the elements but that is not always the case
        MATRIX          = np.zeros((3*M,3*N))
        surface_matrix  = np.zeros((3,3*N))
        torque_matrix   = np.zeros((3,3*N))
        r_cross_matrix  = np.zeros((3*M,3))

        for i in range(N):
            if i % self.AmountofPanelsBeforePrinting == 0:
                print(f"computing panel {i} out of {N}")
            if self.mesh.is_mat:
                panel=self.mesh.panels[1:,:,i]
            else:
                panel=self.mesh.panels[i]

            singularity_contribution, area, torque_tensor = self.calc_mobility_contribution(panel)
            
            MATRIX[0:3*M, 3*i:3*i+3]     = singularity_contribution.reshape(3*M, 3)

            surface_matrix[:,3*i:3*i+3]  = area * np.eye(3)

            torque_matrix[:,3*i:3*i+3]   = torque_tensor
                    


        if self.UseSecondKindIntEquation:
            MATRIX= 0.5*np.eye(3*N) + MATRIX
        
        self.MATRIX          = MATRIX
        self.surface_matrix  = surface_matrix
        self.torque_matrix   = torque_matrix
        self.r_cross_matrix  = skew_stack(self.evaluation_points)    

        return MATRIX, surface_matrix, torque_matrix, r_cross_matrix
    
    
    def set_boundary_condition(self,
                               U        :np.ndarray,
                               W        :np.ndarray,
                               E        :np.ndarray=np.zeros((3,3)))->np.ndarray:
        """
        Set the boundary condition on the body coordinates.

        Parameters
        ----------
        U       : numpy array (3,)
                  Translational background flow
        W       : numpy array (3,)
                  Background vorticity vector
        E       : numpy array (3,3)
                  Background strain rate tensor
        
        Returns
        -------
        U_t+U_r+U_e : numpy array (3N,)
                      The total background flow on each element
        """
        
        
        r, c = np.shape(self.MATRIX)
        U_t, U_r, U_e =U_colloc(U,W, self.mesh.centroids,int(r/3), E)

        return U_t+U_r+U_e
        
        

            
        

    def calc_mobility_contribution(self,
                                   panel:np.ndarray)->tuple[np.ndarray, float, np.ndarray, np.ndarray]: 
        """
        Calculate the contribution to the mobility matrix of an element on all collocation points.
        By vectorising the collocation points, the singularity contributions are calculated at the same time
        and thus increasing the computation speed.

        Parameters
        -------------
        panel   : (V, 3) array with V being the amount of vertices.
                  The current element (panel) to be integrated.

        Returns
        -------
        A_global        : (M, 3, 3) array where M is the amount of collocation points.
                          The total contribution of the stresslet, stokeslet, rotlet on all collocation points.
        area            : float
                          The area of the current element calculated with quadrature.
        torque_tensor   : (3,3) array representing the torque on that element.
                          This array corresponds to the torque matrix on the current element.
                          To calculate the full torque on the mesh you multiply it with the double layer density (psi).
        r_cross         : (3,3) array representing the cross product of r with an arbitrary vector.
                          To determine the RBM, we need to calculate the cross product of r=(y-Y_c) with the double layer density.
        """

        X,Y,Z,centroid, _ =  find_panel_data(panel)  #from utils.py

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

        # In the original matlab code, there is a loop over all collocation points. If you specifically want 
        # to do it this way, it is possible by using the the function stresslet and line_singularity in kernels.py.
        # (see bottom kernels.py).
        
        # The loop has been removed by vectorising all collocation points such that the calculation is done
        # for all collocation points at the same time. This makes the computation MUCH faster, by making use
        # of numpy's C code.

        Int = coord @ centroid      # Center of current surface element for integration

        
        # All collocation points transformed to the element reference frame
        Col_all = (coord @ (self.evaluation_points.T)).T


        
        # Calculate the stresslet contribution of the element on every collocation point
        T_all=stresslet_vectorized(Col_all, Int, Xq, Yq, Wx, Wy)

        # Map the quadrature points of the element to the centerline of the mesh 

        # line_scale makes sure that the line distribution does not span the entire centerline,
        # but makes it a bit smaller. If line_scale is set to zero, the line distribution of singularities
        # collapses to a point singularity. XG is the middle of the line
        R   =   self.mesh.parameters["line_scale"] * (centroid[0] 
                                                        + Xq * coord[0,0] 
                                                        + Yq * coord[1,0] 
                                                        - self.mesh.parameters["XG"]) + self.mesh.parameters["XG"]
        

        # Calculate the Stokeslet and rotlet contribution on all collocation points 
        S_all, G_all = line_singularity_vectorized(Col_all, Int, coord, R, Xq, Yq, Wx, Wy)

        # Calculate the the total singularity contribution of an element on all collocation points 
        A_all =  (3/(4*np.pi)) * T_all +  (1/(8*np.pi)) * ( S_all + G_all ) # 

        # Transform back into the original coodinates
        A_global = np.einsum('ij,mjk,kl->mil', coord.T, A_all, coord)

        # Calculate the area of the element using the quadrature weights
        area = Wx @ np.ones(np.shape(Xq)) @ Wy

        # Convert the coordinates of the element centroid to element coordinates from the origin
        cent_pt = coord @ (centroid)   

        xx = cent_pt[0] + Xq       # shape (Q,Q)
        yy = cent_pt[1] + Yq       # shape (Q,Q)
        zz = cent_pt[2] + Zq

        # Initialise torque tensor (equivalent to r x psi = R psi,  so R = [r] x )
        torque_tensor = np.zeros((3,3))

        torque_tensor[0,1] = -Wx @ zz @ Wy
        torque_tensor[0,2] =  Wx @ yy @ Wy
        torque_tensor[1,2] = -Wx @ xx @ Wy

        # Anti-symmetric tensor
        torque_tensor[1,0] = -torque_tensor[0,1]
        torque_tensor[2,0] = -torque_tensor[0,2]
        torque_tensor[2,1] = -torque_tensor[1,2]

        torque_tensor = coord.T @ torque_tensor @ coord

        

        return A_global, area, torque_tensor
    















    