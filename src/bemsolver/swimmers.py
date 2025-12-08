import numpy as np
from typing import Iterable, Callable
import pickle 

from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve


from .mesh import Mesh
from .system_base  import BaseSystem
from .flagella import SlenderBody
from .flowfield import FlowStokes
from .SaveData import Solution
from .utils import points_in_polygon



@dataclass
class Swimmer(BaseSystem):

    flagellum_1         : Iterable[SlenderBody]

    flagellum_2         : Iterable[SlenderBody]     = field(default_factory=lambda: None)


    def __post_init__(self):

        N_frames = len(self.flagellum_1)
        N_h      = len(self.mesh.centroids)
        N_f1     = len(self.flagellum_1[0].r)
        N_f2     = len(self.flagellum_2[0].r) if self.flagellum_2 is not None else 0

        # Initialise matrices to store the LU decomposition matrix and pivot vector
        self.LU_matrix  = np.zeros((N_frames, 3*N_h + 3*N_f1 + 3* N_f2, 3*N_h + 3*N_f1 + 3* N_f2))
        self.piv_vector = np.zeros((N_frames, 3*N_h + 3*N_f1 + 3* N_f2))

        self.solution = Solution()

        self.solution.time = np.zeros(N_frames)
        self.solution.psi  = np.zeros((N_frames, 3*self.mesh.elements))
        self.solution.f1   = np.zeros((N_frames, 3*len(self.flagellum_1[0].r)))
        
        super().__post_init__()
        

        self.populate_mobility_matrix()

    
    def populate_mobility_matrix(self):
        """
        Load the mobility matrices of the flagella of each frame. The only thing that changes over time is
        the interaction between the cell body and the flagella.

        This function populates the LU_matrix and piv_vector attributes of the Swimmer class.

    
        """

        Mh,_,_,_  = self.construct_mobility_matrix()        


        if self.flagellum_2 is None:
            print("Populating flagellum")
            for i, frame in enumerate(self.flagellum_1):
                Mf1  = frame.construct_mobility_matrix()
                Mf1h = frame.calc_interaction(self.mesh.centroids)
                Mhf1 = FlowStokes(self.mesh, frame.r).MATRIX

                swimmer_matrix = np.block([
                    [Mh, Mf1h],
                    [Mhf1, Mf1]
                ])
                self.LU_matrix[i], self.piv_matrix[i] = lu_factor(swimmer_matrix)

        
        else:
            print("Populating both flagella")
            for i, (frame_1, frame_2) in enumerate(zip(self.flagellum_1,self.flagellum_2)):
                Mf1  = frame_1.construct_mobility_matrix()
                Mf1h = frame_1.calc_interaction(self.mesh.centroids)
                Mhf1 = FlowStokes(self.mesh, frame_1.r).MATRIX
                Mf1f2 = frame_1.calc_interaction(frame_2.r)

                Mf2  = frame_2.construct_mobility_matrix()
                Mf2h = frame_2.calc_interaction(self.mesh.centroids)
                Mhf2 = FlowStokes(self.mesh, frame_2.r).MATRIX
                Mf2f1 = frame_2.calc_interaction(frame_1.r)

                swimmer_matrix = np.block([
                    [Mh, Mf1h,  Mf2h],
                    [Mhf1, Mf1, Mf2f1],
                    [Mhf2, Mf1f2, Mf2]
                ])
                self.LU_matrix[i], self.piv_vector[i] = lu_factor(swimmer_matrix)

        print(f"Loaded {len(self.flagellum_1)} frames with {self.flagellum_1[0].Nf} elements!")

    
    

    def solve(self,find_flow:Callable, dt:float) -> Iterable[np.ndarray]:
        """
        Solve the mobility problem for all frames given a function that provides the boundary conditions.

        Parameters
        ----------
        find_flow : Callable
            A function that takes the current state and returns the boundary conditions (U, W, E), which
            has as input the current time and the position vector of the swimmer.
        dt : float
            Time step of the simulation.

        Returns
        -------
        solutions : Iterable[np.ndarray]
            An iterable of solution vectors for each frame.
        """
        self.dt = dt

        N_frames = len(self.flagellum_1)


        if self.flagellum_2 is not None:
            self.solution.f2   = np.zeros((N_frames, 3*len(self.flagellum_2[0].r)))
        

        for frame_index in range(N_frames):
            self.solution.time[frame_index] = frame_index * dt
            # pass on zeros since swimmer is positioned at origin.
            U, W, E = find_flow(frame_index*dt, np.zeros(3))

            self.solve_step(frame_index , U, W, E)
            

        return self.solution

    def solve_step(self, frame_index: int,
                    U:np.ndarray, W:np.ndarray, E:np.ndarray) -> np.ndarray:
        """
        Solve the mobility problem for a given frame index and right-hand side vector.

        Parameters
        ----------
        frame_index : int
            The index of the frame to solve for.
        U : np.ndarray
            The translational velocity boundary condition.
        W : np.ndarray
            The rotational velocity boundary condition.
        E : np.ndarray
            The strain rate tensor boundary condition.

        Returns
        -------
        solution :np.ndarray
            The solution vector which includes the double-layer density and the forces on the flagella [psi, f1, f2].
        """
        rhs_h = self.set_boundary_condition(U, W, E)

        if self.flagellum_2 is None:
            rhs_f1 = self.flagellum_1[frame_index].set_boundary_condition(U, W, E)
            rhs = np.concatenate([rhs_h, rhs_f1])

        else:
            rhs_f1 = self.flagellum_1[frame_index].set_boundary_condition(U, W, E)
            rhs_f2 = self.flagellum_2[frame_index].set_boundary_condition(U, W, E)
            rhs = np.concatenate([rhs_h, rhs_f1, rhs_f2])

        LU = self.LU_matrix[frame_index]
        piv = self.piv_vector[frame_index]

        sol = lu_solve((LU, piv), -rhs)
        psi = sol[:3 * len(self.mesh.centroids)]
        f1  = sol[3 * len(self.mesh.centroids): 3 * len(self.mesh.centroids) + 3 * len(self.flagellum_1[frame_index].r)]

        self.solution.psi[frame_index] = psi
        self.solution.f1[frame_index]  = f1

        if self.flagellum_2 is not None:
            f2  = sol[3 * len(self.mesh.centroids) + 3 * len(self.flagellum_1[frame_index].r):]
            self.solution.f2[frame_index]  = f2
            return psi, f1, f2
        else:
            return psi, f1
        
        
    def calc_vector_field(self,
                          interaction_object : FlowStokes,
                          frame_index        : int        ,
                          find_flow          : Callable[[float,np.ndarray], tuple[np.ndarray]]) -> np.ndarray:
        """
        Calculate the velocity field at the evaluation points for a given frame index.
        Parameters
        ----------
        interaction_object : FlowStokes
            The FlowStokes object that contains the evaluation points and mobility matrix.
        frame_index : int
            The index of the frame to calculate the velocity field for.
        find_flow : Callable
            A function that takes the current state and returns the boundary conditions (U, W, E), which
            has as input the current time and the position vector of the swimmer.
            
        Returns
        -------
        U_field : np.ndarray
            The velocity field at the evaluation points.
        """

        xg, yg, zg = interaction_object.evaluation_points.T
        Ng = np.shape(xg)[0]

        K1 = self.flagellum_1[frame_index].calc_interaction(interaction_object.evaluation_points)

        if self.flagellum_2 is None:
            U_field = (interaction_object.MATRIX @ self.solution.psi[frame_index] + K1 @ self.solution.f1[frame_index])           

        else:
            K2 = self.flagellum_2[frame_index].calc_interaction(interaction_object.evaluation_points)

            U_field = (interaction_object.MATRIX @ self.solution.psi[frame_index] 
                       + K1 @ self.solution.f1[frame_index] 
                       + K2 @ self.solution.f2[frame_index]) 

        # Again zeros since swimmer is at origin
        U, W, E = find_flow(frame_index*self.dt, np.zeros(3))
        


        # Get the surface of your mesh, r_surface is the distance from the centerline
        x_surface, r_surface = self.mesh.isosurface.T

        r = np.sqrt(yg**2 + zg**2)  # radial coordinate of each point

        self.inside_mask = points_in_polygon(xg, r, x_surface, r_surface)

        U_boundary = interaction_object.set_background_flow(U, W, E)

        U_field =U_field + U_boundary

        U_field = U_field.reshape(Ng, 3)
        U_field[self.inside_mask,:] = 0

        return U_field
        



        

            


            
        

        


@dataclass
class FreeSwimmer(BaseSystem):

    flagellum_1     : Iterable[SlenderBody]

    flagellum_2     : Iterable[SlenderBody]     = field(default_factory=lambda: None)


    def __post_init__(self):
        return super().__post_init__()
    


        




