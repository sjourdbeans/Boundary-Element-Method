import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve

from .mesh import Mesh
from .utils import find_panel_data, U_colloc
from .kernels import stresslet_vectorized, line_singularity_vectorized, stresslet, line_singularity
from .quadrature import triquad





@dataclass
class System:

    mesh:Mesh
    evaluation_points: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        # If no collocation points are given, use the element centroids as collocation points
        if self.evaluation_points is None:
            self.AmountofPanelsBeforePrinting=10
            self.evaluation_points = self.mesh.centroids

            # If the collocation points are the same as the element centroids,
            # we need to use the integral equation of the second kind from Keaveny & Shelley
            self.UseSecondKindIntEquation=1
        else:
            self.AmountofPanelsBeforePrinting=100
            self.UseSecondKindIntEquation=0

        # self.mobility_matrix = np.zeros()



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
            panel=self.mesh.panels[1:,:,i]

            singularity_contribution, area, torque_tensor, r_cross = self.calc_mobility_contribution(panel)
            
            MATRIX[0:3*M, 3*i:3*i+3] = singularity_contribution.reshape(3*M, 3)

            surface_matrix[:,3*i:3*i+3]  = area * np.eye(3)

            torque_matrix[:,3*i:3*i+3]   = torque_tensor

            r_cross_matrix[3*i:3*i+3,:]  = r_cross

        MATRIX= self.UseSecondKindIntEquation*(0.5*np.eye(3*N)) + MATRIX

        
        self.MATRIX          = MATRIX
        self.surface_matrix  = surface_matrix
        self.torque_matrix   = torque_matrix
        self.r_cross         = r_cross_matrix

        return MATRIX, surface_matrix, torque_matrix, r_cross_matrix
    
        
    def solve(self,
              U:np.ndarray,
              W:np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the linear system M*psi=U, and return the double layer density,
        the total force, and the total torque on the body by the boundary condition U.
        """
        
        # self.MATRIX, self.surface_matrix, self.torque_matrix, self.r_cross=self.construct_mobility_matrix()
        
        RHS = self.set_boundary_condition(U,W)

        # lu, piv = lu_factor(self.MATRIX)
        # psi = lu_solve((lu, piv), -RHS)
        self.psi=np.linalg.solve(self.MATRIX,RHS)
        self.force   = self.surface_matrix @ self.psi
        self.torque  = self.torque_matrix @ self.psi

        return self.psi, self.force, self.torque
    
    def construct_grand_mobility_matrix(self):
        
        r, c = np.shape(self.MATRIX)        

        # Generate r/3 identity matrices stacked vertically (r,3) matrix
        V = np.tile(np.eye(3), int(r/3)).T
        A = self.r_cross

        F = self.surface_matrix
        T = self.torque_matrix

        M=np.block([
            [self.MATRIX, V, A],
            [F, np.zeros((3,6))],
            [T, np.zeros((3,6))]
        ])
        return M
    
    def set_boundary_condition(self,
                               U:np.ndarray,
                               W:np.ndarray)->np.ndarray:
        
        r, c = np.shape(self.MATRIX)
        U_t, U_r, _ =U_colloc(U,W, self.mesh.centroids,int(r/3))

        return U_t+U_r
    
    def set_shear_boundary_condition(self,
                               U        :np.ndarray,
                               W        :np.ndarray,
                               E        :np.ndarray)->np.ndarray:
        
        r, c = np.shape(self.MATRIX)
        U_t, U_r, U_e =U_colloc(U,W, self.mesh.centroids,int(r/3), E)

        return U_t+U_r+U_e

        



    
    def plot_singularity_density(self):
        """
        Plot the singularity density x,y,z components in separate plots.

        NOTE: This can only run after the simulation has been solved.

        """
        from . import plotting

        try:
            psi=self.psi.reshape((self.mesh.elements,3))
            figs=[]
            axes=[]
            for i in range(3):
                fig,ax=plotting.plot_panels_stokes(self.mesh.panels,psi[:,i])
                figs.append(fig)
                axes.append(ax)
            return figs, axes 
        except:
            raise SyntaxError("System has not been solved yet! Run System.solve(RHS) before plotting.")
        



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
            panel=self.mesh.panels[1:,:,i]

            singularity_contribution, area, torque_tensor, r_cross = self.calc_mobility_contribution(panel)
            
            MATRIX[0:3*M, 3*i:3*i+3] = singularity_contribution.reshape(3*M, 3)

            surface_matrix[:,3*i:3*i+3]  = area * np.eye(3)

            torque_matrix[:,3*i:3*i+3]   = torque_tensor

            r_cross_matrix[3*i:3*i+3,:]  = r_cross

        MATRIX= self.UseSecondKindIntEquation*(0.5*np.eye(3*N)) + MATRIX

        
        self.MATRIX          = MATRIX
        self.surface_matrix  = surface_matrix
        self.torque_matrix   = torque_matrix
        self.r_cross         = r_cross_matrix

        return MATRIX, surface_matrix, torque_matrix, r_cross_matrix
        

            
        

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

        X,Y,Z,centroid =  find_panel_data(panel)  #from utils.py

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
        Col_all = (coord @ self.evaluation_points.T).T


        
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
        A_all =  (3/(4*np.pi)) * T_all +(1/(8*np.pi)) * ( S_all + G_all ) # 

        # Transform back into the original coodinates
        A_global = np.einsum('ij,mjk,kl->mil', coord.T, A_all, coord)

        # Calculate the area of the element using the quadrature weights
        area = Wx @ np.ones(np.shape(Xq)) @ Wy

        # Convert the coordinates of the element centroid to element coordinates from the origin
        cent_pt = coord @ centroid 

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

        # Calculate [r]x which is the matrix representation of the cross product of r with an arbitrary vector.
        r_cross=np.zeros((3,3))

        r_cross[0,1] =  cent_pt[2]
        r_cross[0,2] = -cent_pt[1]
        r_cross[1,2] =  cent_pt[0]

        r_cross[1,0] = -r_cross[0,1]
        r_cross[2,0] = -r_cross[0,2]
        r_cross[2,1] = -r_cross[1,2]

        r_cross = coord.T @ r_cross @ coord

        return A_global, area, torque_tensor, r_cross
    

    

    #=========================OLD CODE===================================
    
    def calc_mobility_contribution_old(self,
                                   panel:np.ndarray)->tuple[np.ndarray, float, np.ndarray]: 
        """
        OLD VERSION
        -----------
        This function looks the most like the matlab code, and it loops over all collocation points.
        This is inefficient, which is why it has been replaced with the vectorised version.
        """

        X,Y,Z,centroid =  find_panel_data(panel)  #from utils.py

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


        Int = coord @ centroid      # Center of current surface element for integration
        numevals,_=np.shape(self.evaluation_points)
        singularities    = np.zeros((3*numevals,3)) 

        for i, eval_point in enumerate(self.evaluation_points):
            
            Col = coord @ eval_point    # Collocation point            
            
            T=stresslet(Col, Int, Xq, Yq, Wx, Wy)
            R   =   self.mesh.parameters["line_scale"] * (centroid[0] 
                                                        + Xq * coord[0,0] 
                                                        + Yq * coord[1,0] 
                                                        - self.mesh.parameters["XG"]) + self.mesh.parameters["XG"]

            S, G = line_singularity(Col, Int, coord, R, Xq, Yq, Wx, Wy)

            singularities[3*i:3*i+3] = coord.T @ (3/(4*np.pi) * T + 1/(8*np.pi) * ( S + G )) @ coord
        
       

        # Calculate the area of the element using the quadrature weights
        area = Wx @ np.ones(np.shape(Xq)) @ Wy

        # Convert the coordinates of the element centroid to element coodinates
        cent_pt = coord @ centroid 

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

        torque_tensor =coord.T @ torque_tensor @ coord

        return singularities, area, torque_tensor






        










    