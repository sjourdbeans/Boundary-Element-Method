import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve

from .mesh import Mesh
from .utils import find_panel_data, U_colloc
from .system_base import BaseSystem

@dataclass
class ResistanceProblem(BaseSystem):

    
    
    def solve(self,
              U:np.ndarray,
              W:np.ndarray,
              E:np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the linear system M*psi=U, and return the double layer density,
        the total force, and the total torque on the body by the boundary condition U.
        """
        
        self.construct_mobility_matrix()
        
        RHS = self.set_boundary_condition(U,W,E)

        lu, piv = lu_factor(self.MATRIX)
        self.psi = lu_solve((lu, piv), -RHS)
        # self.psi=np.linalg.solve(self.MATRIX,RHS)
        self.force   = self.surface_matrix @ self.psi
        self.torque  = self.torque_matrix @ self.psi

        return self.psi, self.force, self.torque
    
    
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




@dataclass
class MobilityProblem(BaseSystem):

    
    def construct_grand_mobility_matrix(self):

        self.construct_mobility_matrix()
        
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

        # self.lu, self.piv=lu_factor(M)

        return M
    

    

    # def omega_func(self,lu, piv, RHS):
    #     # Rotate lab-frame background flow into particle frame
    #     U_body = Q.T @ U
    #     W_body = Q.T @ W
    #     E_body = Q.T @ E @ Q
        

    #     # Solve BEM for current angular velocity
    #     U_rhs = self.set_boundary_condition(U_body, W_body, E_body)
    #     RHS = np.hstack((U_rhs, np.zeros(6)))
    #     phi = lu_solve((lu, piv), -RHS)

    #     psi= phi[:-6]
    #     # print(sys.surface_matrix@psi)
    #     # print(sys.torque_matrix@psi)

    #     omega = phi[-3:]
        
    #     return omega 
    
    # def plot_singularity_density(self):
    #     """
    #     Plot the singularity density x,y,z components in separate plots.

    #     NOTE: This can only run after the simulation has been solved.

    #     """
    #     from . import plotting

    #     try:
    #         psi=self.psi.reshape((self.mesh.elements,3))
    #         figs=[]
    #         axes=[]
    #         for i in range(3):
    #             fig,ax=plotting.plot_panels_stokes(self.mesh.panels,psi[:,i])
    #             figs.append(fig)
    #             axes.append(ax)
    #         return figs, axes 
    #     except:
    #         raise SyntaxError("System has not been solved yet! Run System.solve(RHS) before plotting.")






