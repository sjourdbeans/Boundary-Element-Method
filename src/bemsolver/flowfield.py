import numpy as np
from dataclasses import dataclass, field

from .system_base import BaseSystem
from .utils import U_colloc, points_in_polygon

@dataclass
class FlowStokes(BaseSystem):
    """
    This child class inherits all methods from BaseSystem to calculate the mobility matrix
    that corresponds to a collection of points not on the surface of the mesh to determine the interaction between
    the singularities of the mesh and the evaluation points. This class does not solve the matrix equation 
    M psi = U, but instead only calculates M. This matrix M can then be used to calculate the flow 
    at the evaluation points after the system has been solved with ResistanceProblem or MobilityProblem.

    NOTE: When the class is initialised, it immediately calculates the interaction matrix, which can take a while.


    Parameters
    ----------
    mesh                : Mesh instance
                          The mesh to be used is a bemsolver.Mesh python object.
    evaluation_points   : np.ndarray with shape (M, 3) where M is the amount of evaluation points.
                          The evaluation points where the singularity contribution of the mesh needs to be evaluated.

    Example
    -------

    Calculate the interaction matrix between the mesh and a collection of points.

    >>> import bemsolver as bem

    >>> mesh = bem.Mesh("/path_to_mesh/file.mat")

    >>> points = np.array([ [5,0,0],
                            [0,6,0],
                            [0,0,7]])

    >>> interaction = bem.FlowStokes(mesh,points)
    >>> M = interaction.MATRIX
    """

    evaluation_points: np.ndarray

    def __post_init__(self):
        """
        Since the the evaluation points are not on the surface of the mesh, 
        we now use the integral equation of the first kind. 

        Furthermore, we only print what panel is being evaluated every 100 panels.
        """
        self.AmountofPanelsBeforePrinting=100
        self.UseSecondKindIntEquation=False

        # When class is initialised immediately construct interaction matrix
        self.construct_mobility_matrix()


    def calc_vector_field(self, 
                          psi        : np.ndarray,
                          U          : np.ndarray,
                          W          : np.ndarray,
                          E          : np.ndarray):
        
        xg, yg, zg = self.evaluation_points.T
        Ng = np.shape(xg)[0]

        # Calculate the velocity field from the double layer density
        U_field = self.MATRIX @ psi

        # Get the surface of your mesh, r_surface is the distance from the centerline
        x_surface, r_surface = self.mesh.isosurface.T

        r = np.sqrt(yg**2 + zg**2)  # radial coordinate of each point

        

        self.inside_mask = points_in_polygon(xg, r, x_surface, r_surface)

        rows, columns = np.shape(self.MATRIX)
        U_t, U_r, U_e =U_colloc(U,W, self.evaluation_points,int(rows/3), E)
        U_boundary = U_t + U_r + U_e

        U_field =U_field + U_boundary

        U_field = U_field.reshape(Ng, 3)
        U_field[self.inside_mask,:] = 0

        return U_field
    
    def plot_vector_field(self, 
                          x                 :np.ndarray, 
                          y                 :np.ndarray, 
                          U_field           :np.ndarray,
                          vmax             :float,
                          quiver_density    :int  = 18,
                          view              :str  ='xy',
                          **kwargs):
        """

        view: not correct yet
        **kwargs:

        quiver: bool = True

        vmax_factor
    
        vector_scale: float or int The scale of the vectors in the plot
        view  : 'xy' view of the plot. Not supported yet
        """
        
        Nx = len(x)
        Ny = len(y)

        Ux = U_field[:, 0].reshape(Ny, Nx)
        Uy = U_field[:, 1].reshape(Ny, Nx)
        Uz = U_field[:, 2].reshape(Ny, Nx)

        

        inside_mask_2D = self.inside_mask.reshape(Ny, Nx)

        Ux_masked = np.copy(Ux)
        Uy_masked = np.copy(Uy)
        Uz_masked = np.copy(Uz)

        Ux_masked[inside_mask_2D] = np.nan
        Uy_masked[inside_mask_2D] = np.nan
        Uz_masked[inside_mask_2D] = np.nan

        if view=='xy':  
            U_magnitude = np.sqrt(Ux**2 + Uy**2 +Uz**2)
            Ux_quiver = Ux_masked[::quiver_density, ::quiver_density]
            Uy_quiver = Uy_masked[::quiver_density, ::quiver_density]

        elif view =="xz":
            U_magnitude = np.sqrt(Ux**2 + Uz**2 )
            Ux_quiver = Ux_masked[::quiver_density, ::quiver_density]
            Uy_quiver = Uz_masked[::quiver_density, ::quiver_density]
        elif view=="yz":
            U_magnitude = np.sqrt(0*Uy**2 + Uz**2 )
            Ux_quiver = Uy_masked[::quiver_density, ::quiver_density]
            Uy_quiver = Uz_masked[::quiver_density, ::quiver_density]
        else:
            raise ValueError(f"{view} is not a valid view, did you mean 'xy', 'xz', or 'yz'?" )
            


        x_quiver = x[::quiver_density]
        y_quiver = y[::quiver_density]

        from . import plotting 

        return plotting.plot_vector_field(x,y,U_magnitude,x_quiver,y_quiver,Ux_quiver,Uy_quiver, self.mesh.isosurface,vmax,view,**kwargs)



        

        





    def set_boundary_condition(self):
        """
        This method is only available where the boundary condition needs to be set on the surface of the mesh.
        """
        raise NotImplementedError("FlowStokes does not set boundary conditions!")

    