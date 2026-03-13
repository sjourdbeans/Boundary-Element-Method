import numpy as np
from typing import Iterable, Callable

from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as R


from .system_base  import BaseSystem
from .flagella import SlenderBody
from .flowfield import FlowStokes
from .SaveData import Solution
from .utils import points_in_polygon
from .time_integration import rotate_BCs, forward_euler, rk2, omega_to_quat_dot, pyr_to_quat



@dataclass
class Swimmer(BaseSystem):
    """
    A class representing a swimmer with one or two flagella in a fixed position, used for solving 
    the boundary element method (BEM) for Stokes flow around the swimmer. This class computes the 
    mobility matrix for the swimmer and solves for the singularity distributions (e.g., double-layer 
    potentials on the cell body and forces on the flagella) given external flow conditions. It does 
    not perform time integration for free swimming; for that, use the `FreeSwimmer` class.

    Attributes
    ----------
    mesh : Mesh
          Mesh object which represents the cell body
    flagellum_1 : Iterable[SlenderBody]
        An iterable of `SlenderBody` objects representing the first flagellum across multiple frames.
    flagellum_2 : Iterable[SlenderBody], optional
        An iterable of `SlenderBody` objects representing the second flagellum across multiple frames. 
        Defaults to None if only one flagellum is present.

    Methods
    -------
    ``__post_init__()`` :
        Initializes the swimmer by setting up arrays for LU decomposition, solution storage, and 
        populating the mobility matrix.
    ``populate_mobility_matrix()``
        Computes and stores the LU decomposition of the mobility matrix for each frame, accounting 
        for interactions between the cell body and flagella.
    ``solve(find_flow, dt)``
        Solves the BEM for all frames given a flow function and time step, returning a `Solution` 
        object with singularity distributions and forces.
    ``solve_step(frame_index, U, W, E)``
        Solves the BEM for a single frame given boundary conditions, returning psi, f1, and optionally f2.
    ``calc_vector_field(interaction_object, frame_index, find_flow)``
        Computes the velocity field at specified evaluation points for a given frame, including 
        contributions from singularities and background flow.

    Examples
    --------
    Load a mesh, create flagella from waveform data, instantiate the swimmer, solve for flow, and 
    compute the velocity field:

    >>> import bemsolver as bem
    >>> import numpy as np
    >>> from scipy.io import loadmat
    >>> 
    >>> # Load mesh and waveform data
    >>> mesh = bem.Mesh("path/to/mesh.mat")
    >>> waveform = loadmat("path/to/waveform.mat")
    >>> 
    >>> # Create flagella (simplified example)
    >>> flagellum_1 = [bem.SlenderBody(curv, tors, ...) for curv, tors in waveform_data]
    >>> flagellum_2 = [bem.SlenderBody(curv, tors, ...) for curv, tors in waveform_data]
    >>> 
    >>> # Instantiate swimmer
    >>> swimmer = bem.Swimmer(mesh, flagellum_1=flagellum_1, flagellum_2=flagellum_2)
    >>> 
    >>> # Define flow function
    >>> def find_flow(t, x):
    ...     U = np.array([1000, 0, 0])  # Example uniform flow
    ...     W = np.zeros(3)
    ...     E = np.zeros((3, 3))
    ...     return U, W, E
    >>> 
    >>> # Solve for all frames
    >>> dt = 1 / 30  # Example time step
    >>> solution = swimmer.solve(find_flow, dt)
    >>> 
    >>> # Compute velocity field for a frame
    >>> points = np.random.rand(1000, 3)  # Example evaluation points
    >>> flow_head = bem.FlowStokes(mesh, points)
    >>> u_field = swimmer.calc_vector_field(flow_head, frame_index=0, find_flow=find_flow)
    """



    flagellum_1         : Iterable[SlenderBody]

    flagellum_2         : Iterable[SlenderBody]     = field(default_factory=lambda: None)


    def __post_init__(self):

        # Define the amount of frames and the sizes of the matrices
        N_frames = len(self.flagellum_1)
        N_h      = len(self.mesh.centroids)
        N_f1     = len(self.flagellum_1[0].r)
        N_f2     = len(self.flagellum_2[0].r) if self.flagellum_2 is not None else 0

        # Initialise matrices to store the LU decomposition matrix and pivot vector
        self.LU_matrix  = np.zeros((N_frames, 3*N_h + 3*N_f1 + 3* N_f2, 3*N_h + 3*N_f1 + 3* N_f2))
        self.piv_vector = np.zeros((N_frames, 3*N_h + 3*N_f1 + 3* N_f2))

        # from Savedata.py
        self.solution = Solution()

        # initialise the solution sizes
        self.solution.time = np.zeros(N_frames)
        self.solution.psi  = np.zeros((N_frames, 3*self.mesh.elements))
        self.solution.f1   = np.zeros((N_frames, 3*len(self.flagellum_1[0].r)))

        # Only if a second flagella is passed on, initialise it
        if self.flagellum_2 is not None:
            self.solution.f2   = np.zeros((N_frames, 3*len(self.flagellum_2[0].r)))

        #NOTE: In the future more flagella might be added, so maybe define change the code such that the amount of flagella
        #      depends on the amount of arguments.
        
        # Run the __post_init__ of BaseSystem to initialise mesh and mobility matrix
        super().__post_init__()
        
        # Populate the mobility matrices and store the LU decomposition
        self.populate_mobility_matrix()

        self.N_frames=N_frames

    
    def populate_mobility_matrix(self):
        """
        Load the mobility matrices of the flagella of each frame. The only thing that changes over time is
        the interaction between the cell body and the flagella.

        This function populates the LU_matrix and piv_vector attributes of the Swimmer class.

    
        """
        # calculate the mobility matrix of the mesh
        Mh,_,_,_  = self.construct_mobility_matrix()        

        # Two scenarios: one flagellum or two (in the future more than two)
        
        # The interaction and mobility matrices in the first scenario are also calculated in the second scenario
        # but to avoid checking whether we have a second flagella every loop we set the if statement before the loop
        if self.flagellum_2 is None:
            print("Populating flagellum")
            for i, frame in enumerate(self.flagellum_1):
                # Mobility matrix of the flagellum 1
                Mf1  = frame.construct_mobility_matrix()
                # interaction matrix of flagella acting on the head (cell body)
                Mf1h = frame.calc_interaction(self.mesh.centroids)
                # interaction matrix of the head acting on flagellum 1
                Mhf1 = FlowStokes(self.mesh, frame.r).MATRIX

                # Assemble mobility matrix of the swimmer
                swimmer_matrix = np.block([
                    [Mh, Mf1h],
                    [Mhf1, Mf1]
                ])
                # save the LU decomposition to solve the matrix equations later
                self.LU_matrix[i], self.piv_matrix[i] = lu_factor(swimmer_matrix)

        
        else:
            print("Populating both flagella")
            for i, (frame_1, frame_2) in enumerate(zip(self.flagellum_1,self.flagellum_2)):
                
                # Calculate the interaction of flagella 1 and 2 on the head (cell body)
                Mf1h = frame_1.calc_interaction(self.mesh.centroids)
                Mf2h = frame_2.calc_interaction(self.mesh.centroids)

                # Mobility matrix of flagellum 1, and the interaction of the head and flagellum 2 on flagellum 1
                Mf1  = frame_1.construct_mobility_matrix()
                Mhf1 = FlowStokes(self.mesh, frame_1.r).MATRIX
                Mf2f1 = frame_2.calc_interaction(frame_1.r)

                # Mobility matrix of flagellum 2, and the interaction of the head and flagellum 1 on flagellum 2
                Mf2  = frame_2.construct_mobility_matrix()
                Mhf2 = FlowStokes(self.mesh, frame_2.r).MATRIX
                Mf1f2 = frame_1.calc_interaction(frame_2.r)

                # Assemble mobility matrix of the swimmer
                swimmer_matrix = np.block([
                    [Mh, Mf1h,  Mf2h],
                    [Mhf1, Mf1, Mf2f1],
                    [Mhf2, Mf1f2, Mf2]
                ])
                # save the LU decomposition
                self.LU_matrix[i], self.piv_vector[i] = lu_factor(swimmer_matrix)

        print(f"Loaded {len(self.flagellum_1)} frames with {self.flagellum_1[0].Nf} elements per flagellum!")

    
    

    def solve(self,
              find_flow:Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]], 
              dt:float) -> Solution:
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
        solutions : Solution
            Solution dataclass which contains the simulation results
        """
        # make dt an attribute such that it can be used in other methods
        self.dt = dt

        # Amount of frames in the simulation
        N_frames = len(self.flagellum_1)
        
        # Loop over all frames and solve the system
        for frame_index in range(N_frames):
            self.solution.time[frame_index] = frame_index * dt
            # pass on zeros since swimmer is positioned at origin.
            U, W, E = find_flow(frame_index*dt, np.zeros(3))

            # solve the system for the current frame 
            # (this function returns psi, f1, and f2 but the function itself stores these values in Solution)
            self.solve_step(frame_index , U, W, E)
            
        return self.solution

    def solve_step(self, frame_index: int,
                    U:np.ndarray, W:np.ndarray, E:np.ndarray) -> tuple:
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
        solution :tuple
            A tuple which includes the double-layer density and the forces on the flagella [psi, f1, f2].
        """

        # Calculate the boundary condition on the cell body
        rhs_h = self.set_boundary_condition(U, W, E)

        # Calculate the boundary conditions for the flagella
        if self.flagellum_2 is None:
            rhs_f1 = self.flagellum_1[frame_index].set_boundary_condition(U, W, E)
            rhs = np.concatenate([rhs_h, rhs_f1])

        else:
            rhs_f1 = self.flagellum_1[frame_index].set_boundary_condition(U, W, E)
            rhs_f2 = self.flagellum_2[frame_index].set_boundary_condition(U, W, E)
            rhs = np.concatenate([rhs_h, rhs_f1, rhs_f2])

        # Choose the LU decomposition
        LU = self.LU_matrix[frame_index]
        piv = self.piv_vector[frame_index]
        
        # Solve the system
        sol = lu_solve((LU, piv), -rhs)
        
        # unpack solution
        psi = sol[:3 * len(self.mesh.centroids)]
        f1  = sol[3 * len(self.mesh.centroids): 3 * len(self.mesh.centroids) + 3 * len(self.flagellum_1[frame_index].r)]

        # save solution
        self.solution.psi[frame_index] = psi
        self.solution.f1[frame_index]  = f1

        # unpack solution of flagellum 2
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

        # points to evaluate the flow
        xg, yg, zg = interaction_object.evaluation_points.T
        Ng = np.shape(xg)[0]

        # interaction matrix between flagellum 1 and the evaluation points
        K1 = self.flagellum_1[frame_index].calc_interaction(interaction_object.evaluation_points)

        # Calculate the flow around the swimmer for either scenario
        if self.flagellum_2 is None:
            U_field = (interaction_object.MATRIX @ self.solution.psi[frame_index] + K1 @ self.solution.f1[frame_index])           

        else:
            # Interaction matrix of flagellum 2 and the evaluation points
            K2 = self.flagellum_2[frame_index].calc_interaction(interaction_object.evaluation_points)

            U_field = (interaction_object.MATRIX @ self.solution.psi[frame_index] 
                       + K1 @ self.solution.f1[frame_index] 
                       + K2 @ self.solution.f2[frame_index]) 

        # Again zeros since swimmer is at origin
        U, W, E = find_flow(frame_index*self.dt, np.zeros(3))
    

        # Get the surface of your mesh, r_surface is the distance from the centerline
        x_surface, r_surface = self.mesh.isosurface.T

        r = np.sqrt(yg**2 + zg**2)  # radial coordinate of each point

        # find points inside the cell body
        self.inside_mask = points_in_polygon(xg, r, x_surface, r_surface)

        # Set background flow on the evaluation points
        U_boundary = interaction_object.set_background_flow(U, W, E)

        # Find total velocity field
        U_field = U_field + U_boundary

        # Reshape to vectors
        U_field = U_field.reshape(Ng, 3)
        # set points inside cell body to zero
        U_field[self.inside_mask,:] = 0

        return U_field
        



@dataclass
class FreeSwimmer(BaseSystem):
    """
    A class representing a free-swimming organism with one or two flagella, used for solving 
    the boundary element method (BEM) for Stokes flow and integrating the swimmer's motion over time. 
    This class computes the grand mobility matrix for the swimmer and flagella, solves for singularity 
    distributions, and integrates rigid-body motion (position and orientation) using numerical methods 
    like forward Euler. It accounts for external flow conditions and flagellar waveforms to simulate 
    swimming dynamics.

    Attributes
    ----------
    mesh : Mesh
          Mesh object which represents the cell body
    flagellum_1 : Iterable[SlenderBody]
        An iterable of `SlenderBody` objects representing the first flagellum across multiple frames.
    flagellum_2 : Iterable[SlenderBody], optional
        An iterable of `SlenderBody` objects representing the second flagellum across multiple frames. 
        Defaults to None if only one flagellum is present.
    viscosity : float
        Viscosity of the surrounding fluid in Pa.s (default is water at room temperature 1e-3).

    Methods
    -------
    ``__post_init__()``
        Initializes the swimmer by setting up arrays for LU decomposition, solution storage, and 
        populating the grand mobility matrix.
    ``populate_grand_mobility_matrix()``
        Computes and stores the LU decomposition of the grand mobility matrix for each frame, 
        including force/torque balance constraints for free swimming.
    ``RBM_over_time(dt, t_end, flow_function, initial_position, initial_orientation)``
        Integrates the swimmer's motion over time using the provided flow function and initial conditions, 
        returning a `Solution` object with position, orientation, velocities, and singularity distributions.
    ``solve_RBM(x_initial, p_initial, time, dt)``
        Solves for the next state (position, orientation, etc.) using forward Euler integration.
    ``calc_Y_dot(t, Y)``
        Calculates the time derivative of the state vector (position and quaternion).
    ``calc_RBM(x, q, t)``
        Computes the rigid-body motion (velocities and singularities) for a given state.
    ``calc_vector_field(interaction_object, frame_index, find_flow, include_rbm)``
        Computes the velocity field at specified evaluation points for a given frame, optionally 
        including the swimmer's rigid-body motion.

    Examples
    --------
    Load a mesh, create flagella from waveform data, instantiate the free swimmer, and run the simulation:

    >>> import bemsolver as bem
    >>> import numpy as np
    >>> from scipy.io import loadmat
    >>> 
    >>> # Load mesh and waveform data
    >>> mesh = bem.Mesh("path/to/mesh.mat")
    >>> waveform = loadmat("path/to/waveform.mat")
    >>> 
    >>> # Create flagella (simplified example, based on waveform data)
    >>> flagellum_1 = [bem.SlenderBody(curv, tors, ...) for curv, tors in waveform_data_1]
    >>> flagellum_2 = [bem.SlenderBody(curv, tors, ...) for curv, tors in waveform_data_2]
    >>> 
    >>> # Instantiate free swimmer
    >>> swimmer = bem.FreeSwimmer(mesh, flagellum_1=flagellum_1, flagellum_2=flagellum_2)
    >>> 
    >>> # Define flow function (e.g., shear flow)
    >>> def find_flow(t, x):
    ...     gamma_dot = 0  # No shear
    ...     U = np.zeros(3)
    ...     W = np.array([0, 0, -gamma_dot/2])
    ...     E = (gamma_dot/2) * np.array([[0,1,0], [1,0,0], [0,0,0]])
    ...     return U, W, E
    >>> 
    >>> # Run simulation over time
    >>> dt = 1 / 30  # Time step
    >>> t_end = 100*dt
    >>> solution = swimmer.RBM_over_time(dt, t_end, find_flow, initial_position=np.array([0,0,0]), initial_orientation=np.array([0, 0, 0]))
    >>> 
    >>> # Access results, e.g., position and velocity
    >>> positions = solution.X[:, :3]
    >>> velocities = solution.u
    """   

    flagellum_1     : Iterable[SlenderBody]

    flagellum_2     : Iterable[SlenderBody]     = field(default_factory=lambda: None)

    viscosity       :float = field(default_factory=lambda: 1e-3)  # Pa.s (water at room temp)





    def __post_init__(self):
        self.N_frames = len(self.flagellum_1)
        self.N_h      = len(self.mesh.centroids)
        self.N_f1     = len(self.flagellum_1[0].r)
        self.N_f2     = len(self.flagellum_2[0].r) if self.flagellum_2 is not None else 0

        # Calculate the dimension of the grand mobility matrix (the extra 6 is for U and Omega)
        dimension = 3*self.N_h + 3*self.N_f1 + 3*self.N_f2 + 6 
        
        # Initialise matrices to store the LU decomposition matrix and pivot vector
        self.LU_matrix  = np.zeros((self.N_frames, dimension, dimension))
        self.piv_vector = np.zeros((self.N_frames, dimension))

        # Run the __post_init__ of BaseSystem 
        super().__post_init__()

        # Populate the grand mobility matrix of all frames
        self.populate_grand_mobility_matrix() 


    def populate_grand_mobility_matrix(self):
        """
        This method calculates the grand mobility matrix of all frames and stores these matrices as an
        LU matrix and a pivot vector such that they can be used to solve the matrix equation Ax=LUx=b 
         
        """

        # Calculate mobility matrix of the cell body
        Mh,_,_,_  =  self.construct_mobility_matrix()
        Mh = (1/self.viscosity) * Mh   
        r, c      = np.shape(Mh)     


        if self.flagellum_2 is None:
            print("Populating flagellum")
            for i, frame in enumerate(self.flagellum_1):
                
                # Interaction matrix of the flagellum acting on the head (cell body) 
                Mf1h = (1/self.viscosity) * frame.calc_interaction(self.mesh.centroids)

                # Interaction matrix of the cell body acting on the flagellum
                Mhf1 = (1/self.viscosity) * FlowStokes(self.mesh, frame.r).MATRIX
                # Mobility matrix of the flagellum
                Mf1  = (1/self.viscosity) * frame.construct_mobility_matrix()
                
                # Rigid Body Motion (RBM) terms: U_rbm = U + omega x r = V @ U + A @ omega
                # NOTE: the velocity of the fluid on the surface due to RBM is opposite in direction (v_rbm=-u_rbm)

                # Matrices relating the material points and the translational velocity (identities)
                V_h  = np.tile(np.eye(3), int(r/3)).T       
                V_f1 = np.tile(np.eye(3), int(len(Mf1)/3)).T  # matrix filled with identity blocks (3M, 3)
                
                # Matrices representing the cross product between omega and r
                A_h  = self.r_cross_matrix
                A_f1 = frame.calc_r_cross_matrix(self.mesh.center)

                # Force and torque free constraints: F_h @ psi + F_f1 @ f1 = 0
                #                                    T_h @ psi + T_f1 @ f1 = 0
                F_h  = self.surface_matrix
                F_f1 = V_f1.T               # Also a matrix filled with identity blocks but then (3, 3M)
                
                T_h  = self.torque_matrix
                T_f1 = A_f1.T   # the 3x3 matrices in A_f1 are antisymmetric so transposing is the same as multiplying by -1
        
                # Assemble the grand mobility matrix and store the LU matrix and pivot vector
                swimmer_matrix = np.block([
                    [Mh,   Mf1h,  -V_h,  -A_h    ],
                    [Mhf1, Mf1,   -V_f1, -A_f1   ],
                    [F_h,  F_f1,  np.zeros((3,6))],
                    [T_h,  T_f1,  np.zeros((3,6))]
                ])
                self.LU_matrix[i], self.piv_vector[i] = lu_factor(swimmer_matrix)

        
        else:
            print("Populating both flagella")
            for i, (frame_1, frame_2) in enumerate(zip(self.flagellum_1,self.flagellum_2)):
                

                #============Cell Body==============
                # Interaction matrices of the flagella acting on the cell body
                Mf1h = (1/self.viscosity) * frame_1.calc_interaction(self.mesh.centroids)
                Mf2h = (1/self.viscosity) * frame_2.calc_interaction(self.mesh.centroids)

                #==========Flagellum 1==============
                # Interaction matrix of the cell body acting on flagellum 1, 
                Mhf1 = (1/self.viscosity) * FlowStokes(self.mesh, frame_1.r).MATRIX
                # Mobility matrix of flagellum 1 
                Mf1  = (1/self.viscosity) * frame_1.construct_mobility_matrix()
                # Interaction matrix of flagellum 2 acting on flagellum 1
                Mf2f1 = (1/self.viscosity) * frame_2.calc_interaction(frame_1.r)

                #==========Flagellum 2==============
                # Interaction matrix of the cell body acting on flagellum 2
                Mhf2 = (1/self.viscosity) * FlowStokes(self.mesh, frame_2.r).MATRIX  
                # Interaction matrix of flagellum 1 acting on flagellum 2              
                Mf1f2 = (1/self.viscosity) * frame_1.calc_interaction(frame_2.r)
                # Mobility matrix of flagellum 2
                Mf2  = (1/self.viscosity) * frame_2.construct_mobility_matrix()
                
                # Rigid Body Motion (RBM) terms: U_rbm = U + omega x r = V @ U + A @ omega
                # NOTE: the velocity of the fluid on the surface due to RBM is opposite in direction (v_rbm=-u_rbm)

                # Matrices relating the material points and the translational velocity (identities)               

                V_h  = np.tile(np.eye(3), int(r/3)).T       
                V_f1 = np.tile(np.eye(3), int(len(Mf1)/3)).T  # matrix filled with identity blocks (3M, 3)
                V_f2 = np.tile(np.eye(3), int(len(Mf2)/3)).T  # matrix filled with identity blocks (3M, 3)

                # Matrices representing the cross product between omega and r
                A_h  = self.r_cross_matrix
                A_f1 = frame_1.calc_r_cross_matrix(self.mesh.center)
                A_f2 = frame_2.calc_r_cross_matrix(self.mesh.center)

                # Force and torque free constraints: F_h @ psi + F_f1 @ f1 + F_f2 @ f2 = 0
                #                                    T_h @ psi + T_f1 @ f1 + T_f2 @ f2 = 0

                F_h  = self.surface_matrix
                F_f1 = V_f1.T               # Also a matrix filled with identity blocks but then (3, 3M)
                F_f2 = V_f2.T

                T_h  = self.torque_matrix
                T_f1 = A_f1.T   # the 3x3 matrices in A_f1 and A_f2 are antisymmetric so transposing is the same as multiplying by -1
                T_f2 = A_f2.T   # the 3x3 matrices in A_f1 and A_f2 are antisymmetric so transposing is the same as multiplying by -1

                # Assemble the grand mobility matrix and store the LU matrix and pivot vector
                swimmer_matrix = np.block([
                    [Mh,   Mf1h,  Mf2h,    -V_h,   -A_h ],
                    [Mhf1, Mf1,   Mf2f1,   -V_f1,  -A_f1],
                    [Mhf2, Mf1f2, Mf2,     -V_f2,  -A_f2],
                    [F_h,  F_f1,  F_f2,  np.zeros((3,6))],
                    [T_h,  T_f1,  T_f2,  np.zeros((3,6))],
                ])
                self.LU_matrix[i], self.piv_vector[i] = lu_factor(swimmer_matrix)

        print(f"Loaded {len(self.flagellum_1)} frames with {self.flagellum_1[0].Nf} elements!")


    def RBM_over_time(self,  
                      dt                 :float,
                      t_end              :float|int,
                      flow_function      :Callable[[float,np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
                      initial_position   :np.ndarray = np.array([0,0,0]),
                      initial_orientation:np.ndarray = np.array([0, 0, 0]),
                      gravity_direction  :np.ndarray = np.array([0,0,-1])
                      ) -> Solution:
        """
        Solve the mobility problem over time given a function that provides the boundary conditions.
        NOTE: dt is passed along as a variable to multiple functions in this algorithm. However, it is also
        set as an attribute, so it might not be necessary to pass it along to every function. I leave it like this
        such that it is clear it depends on dt.

        Parameters
        ----------
        dt                   : float
                              Time step of the simulation in seconds.
        t_end                : float or int
                              End time of the simulation in seconds.
        flow_function        : Callable
                              A function that takes the current state and returns the boundary conditions (U, W, E), which
                              has as input the current time and the position vector of the swimmer.
        initial_position     : numpy array
                              Initial position of the swimmer in cartesian coordinates (standard set to origin)
        initial_orientation  : numpy array
                              Initial pitch, yaw, and roll of the swimmer in radians
        gravity_direction    : numpy array
                              Direction of gravity in the lab frame (standard set to -z)
        Returns
        -------
        solution          : Solution
                           Solution dataclass (SaveData.py) containing the solutions of the simulation.
        """

        pitch, yaw, roll = initial_orientation 


        r = R.from_euler('xzy', [roll, yaw, pitch])
        q = r.as_quat(scalar_first=True)
        # q = pyr_to_quat(pitch, yaw, roll)

        # Make the function that finds the flow an attribute of FreeSwimmer
        self.flow_function = flow_function

        # set gravity vector in lab frame
        self.gravity = 9.8 * gravity_direction / np.linalg.norm(gravity_direction)

        # Make dt an attribute
        self.dt = dt

        total_frames = int(round(t_end / self.dt))
        # print(t_end // self.dt) 

        # Initialise the solution dataclass
        self.solution = Solution()
        
        # Pre-allocate the arrays in Solution
        self.solution.time                  = np.zeros( total_frames )
        self.solution.psi                   = np.zeros((total_frames, 3*self.mesh.elements))
        self.solution.f1                    = np.zeros((total_frames, 3*self.N_f1))
        self.solution.u                     = np.zeros((total_frames, 3))
        self.solution.omega                 = np.zeros((total_frames, 3))
        self.solution.X                     = np.zeros((total_frames, 6))
        self.solution.rotation_matrices     = np.zeros((total_frames, 3, 3))
        self.solution.quaternions           = np.zeros((total_frames, 4))

        # Only pre-allocate f2 if it is an argument
        if self.flagellum_2 is not None:
            self.solution.f2   = np.zeros((total_frames, 3*self.N_f2))

    
        # Set initial quaternion
        self.solution.quaternions[0]  = q

        # Find initial cartesian rotation matrix from quaternion
        Q_0                                = R.from_quat(q, scalar_first=True).as_matrix()
        self.solution.rotation_matrices[0] = Q_0

        # Set initial condition vector and save it
        X_0 = np.hstack((initial_position, Q_0[:,0]))        
        self.solution.X[0] = X_0

        # Unpack initial state
        x, p = X_0[:3], X_0[3:]

        # Calculate the singularity distribution, translational-, and angular velocity for the initial state
        phi, u, omega = self.calc_RBM(x, q, self.solution.time[0])

        self.solution.psi[0]   = phi[:3*self.N_h]
        self.solution.f1[0]    = phi[3*self.N_h:3*self.N_h + 3*self.N_f1]

        if self.flagellum_2 is not None:
            self.solution.f2[0]    = phi[3*self.N_h + 3*self.N_f1:]

        # Transform (angular) velocity to lab frame and save it
        self.solution.u[0]     = Q_0 @ u
        self.solution.omega[0] = Q_0 @ omega

        # Loop over all time frames and solve for the system
        for frame_index in range(total_frames-1):
            if frame_index%100 == 0:
                print(f"Computing frame {frame_index} out of {total_frames}")

            # Update time array
            self.solution.time[frame_index + 1] = (frame_index + 1) * dt

            # Calculate the next timestep
            x, q, p, Q, phi, u, omega = self.solve_RBM(x, q, (frame_index)*dt, dt)


            # Save values to solution file
            self.solution.psi[frame_index+ 1]   = phi[:3*self.N_h]
            self.solution.f1[frame_index + 1]   = phi[3*self.N_h:3*self.N_h + 3*self.N_f1]

            if self.flagellum_2 is not None:
                self.solution.f2[frame_index + 1]    = phi[3*self.N_h + 3*self.N_f1:]
 
            # Transform (angular) velocity to lab frame and save it
            self.solution.u[frame_index+1]     = Q @ u
            self.solution.omega[frame_index+1] = Q @ omega

            self.solution.X[frame_index+1,:3]   = x
            self.solution.X[frame_index+1, 3:]  = p

            # Transform the current orientation to quaternion (not necessary for the sim but maybe for post processing)
            self.solution.quaternions[frame_index+1]        = q #vector_to_quaternion_from_x(p) 
            self.solution.rotation_matrices[frame_index+1]  = Q

        return self.solution



            


    def solve_RBM(self,
                  x_initial                :np.ndarray,
                  q_initial                :np.ndarray,
                  time                     :float|int,
                  dt                       :float)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given an initial x and p, calculate the next iteration with timestep dt using forward euler numerical integration.

        Parameters
        ----------
        x_initial   : numpy array
                      The initial position (x_i)
        q_initial   : numpy array
                      The initial orientation quaternion of the particle (q_i)
        time        : float or int
                      The current simulation time in seconds  
        dt          : float
                      timestep to use for time integration in seconds

        Returns
        -------
        x           : numpy array
                      The next position after time dt (x_{i+1})
        q           : numpy array
                      The next quaternion after time dt (q_{i+1})
        p           : numpy array
                      The next orientation after time dt (p_{i+1})
        Q           : numpy array (3,3)
                      Quaternion rotation matrix. This matrix can be used to rotate from the lab frame to the body frame.
                      Is mainly used to rotate the BCs to find the next solution, or for plotting the vector field.
        phi         : numpy array
                      The solution for the singularity density at the new timestep.
        u           : numpy array
                      The translational velocity of the particle at the new timestep.
        omega       : numpy array
                      The angular velocity of the particle at the new timestep.

        """
        
        
        # stack the initial position and quaternion vector into an array for time integration
        Y_0 = np.hstack((x_initial, q_initial))

        
        # Time integration using forward euler, self.calc_RHS returns the right hand side of the ODE
        # Y_next = forward_euler(self.calc_Y_dot, Y_0, time, dt)  
        Y_next = rk2(self.calc_Y_dot, Y_0, time, dt)       

        # Unpack new timestep
        x, q = Y_next[:3], Y_next[3:]

        # Use next time frame to solve
        phi, u, omega = self.calc_RBM(x, q, time + dt)

        # Convert quaternion vector to cartesian matrix
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        # Retrieve vector which is the swimmer frame x-direction. 
        # The sign of p depends on how your swimmer frame is defined. If the swimmer is oriented in the 
        # negative x-direction in the swimmer frame, you should multiply p with -1 in post processing, 
        # to get the actual orientation.
        # NOTE: p should not be used in the computation itself but only for post-processing as p does
        # not contain information about the orientation (pitch, roll, yaw)
        p = Q[:,0]

        return x, q, p, Q, phi, u, omega
    


    def calc_Y_dot(self,  t:float, Y:np.ndarray )->np.ndarray:
        """
        Function that returns the RHS of the ODE to be integrated over time.
        In other words, a function that calculates the time derivative of the current state of the system.

        Parameters
        ----------    
        t               : float
                          Current time  in seconds
        Y               : numpy array (1,7)
                          Array which contains the initial position and quaternion vector (orientation).
        

        Returns
        -------
        Y_dot   : numpy array (1,7)
                  The array which represents the time derivative at the current timestep (to be integrated).

        """
        # Unpack state
        x,q = Y[:3], Y[3:]

        # Normalise quaterion vector
        q /= np.linalg.norm(q)

        # compute RBM of state Y
        _, u, omega = self.calc_RBM(x ,q, t)

        # Transform velocities back to the lab frame
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        u_lab     = Q @ u
        omega_lab = Q @ omega

        # Calculate the time derivative of the quaternion vector
        q_dot = omega_to_quat_dot(q, omega_lab)

        # Make vector containing the time derivatives
        Y_dot = np.hstack((u_lab,q_dot))
        # Y_dot = np.hstack((u_lab,omega_lab))

        return Y_dot

    

        

    def calc_RBM(self, x, q, t)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the RBM of the particle at the current state with the LU decomposition
        of the grand mobility matrix.

        Parameters
        ----------
        x               : numpy array
                          Current x location of the geometrical center of the swimmer
        q               : numpy array
                          Current orientation as a quaternion vector
        t               : float
                          Current time in the simulation in seconds
        Returns
        -------
        phi     : numpy array (1, N) with N the amount of elements + flagella_1 elements + flagella_2 elements (if present)
                  The solution for the singularity density.
        u       : numpy array (1, 3)
                  Translational velocity of the particle
        omega   : numpy array (1, 3)
                : Angular velocity of the particle 
        """

        time_index = int(round(t/self.dt)) % self.N_frames


        # Find the current flowfield at x
        U, W, E = self.flow_function(t,x)  
        
        # Find cartesian rotation matrix
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        # Rotate the boundary conditions to the particle frame
        U_body, W_body, E_body = rotate_BCs(Q, U, W, E)
        

        # Construct the boundary conditions
        rhs_h = self.set_boundary_condition(U_body, W_body, E_body)
        rhs_f1 = self.flagellum_1[time_index].set_boundary_condition(U_body, W_body, E_body)

        if self.flagellum_2 is None:            
            rhs = np.concatenate([rhs_h, rhs_f1])

        else:
            rhs_f2 = self.flagellum_2[time_index].set_boundary_condition(U_body, W_body, E_body)
            rhs = np.concatenate([rhs_h, rhs_f1, rhs_f2])


        V = self.mesh.parameters["volume"] * 1e-18 # Convert to m^3

        # Calculate gravitational force in body frame
        F_gravity = V * self.mesh.parameters["Delta_rho"] * Q.T @ self.gravity  * 1e12 #Convert to pN (10^-6 term)
        # print(F_gravity)
        T_gravity = -  V *  (self.mesh.parameters["medium_rho"] 
                                      + self.mesh.parameters["Delta_rho"]) * (self.mesh.parameters["COM_offset"] * 1e-6) * Q.T @ np.cross(Q[:,0], self.gravity) *1e18 #Convert to pNum (10^-6 term)
        # T_gravity = np.cross(np.array([self.mesh.parameters["COM_offset"]*1000 , 0, 0]), F_gravity)  #Convert to pNum (10^-6 term)
        RHS = np.hstack((-rhs, F_gravity, T_gravity))

        # No slip, so solve for negative RHS
        sol = lu_solve((self.LU_matrix[time_index], self.piv_vector[time_index]), RHS)
        
        # Unpack solution
        phi   = sol[:-6]
        u     = sol[-6:-3] 
        omega = sol[-3:]


        return phi, u, omega
    
    
    def calc_vector_field(self,
                          interaction_object : FlowStokes,
                          frame_index        : int        ,
                          find_flow          : Callable[[float,np.ndarray], tuple[np.ndarray]],
                          include_rbm        : bool = True) -> np.ndarray:
        """
        Calculate the velocity field at the evaluation points for a given frame index.

        Parameters
        ----------
        interaction_object : FlowStokes
                             The FlowStokes object that contains the evaluation points and mobility matrix.
        frame_index        : int
                             The index of the frame to calculate the velocity field for.
        find_flow          : Callable
                             A function that takes the current state and returns the boundary conditions (U, W, E), which
                             has as input the current time and the position vector of the swimmer.
        include_rbm         : boolean 
                              Include the RBM into the background flow
            
        Returns
        -------
        U_field : numpy array
                  The velocity field at the evaluation points.
        """
        time_frame = int(frame_index % self.N_frames)
        # Unpack evaluation point coordinates
        xg, yg, zg = interaction_object.evaluation_points.T
        Ng = np.shape(xg)[0]

        # Interaction matrix of flagella 1 with evaluation points
        K1 = self.flagellum_1[time_frame].calc_interaction(interaction_object.evaluation_points)

        # Velocity field if only one flagellum is present
        if self.flagellum_2 is None:
            U_field = (interaction_object.MATRIX @ self.solution.psi[frame_index] + K1 @ self.solution.f1[frame_index])           

        # Velocity field if two flagella are present
        else:
            # Interaction matrix of flagellum 2 with the evaluation points
            K2 = self.flagellum_2[time_frame].calc_interaction(interaction_object.evaluation_points)

            U_field = (1/self.viscosity) * (interaction_object.MATRIX @ self.solution.psi[frame_index] 
                       + K1 @ self.solution.f1[frame_index] 
                       + K2 @ self.solution.f2[frame_index]) 

        # Find the flow at current time and location
        U, W, E = find_flow(frame_index*self.dt, self.solution.X[frame_index,:3])

        Q = self.solution.rotation_matrices[frame_index]

        # Add swimmer rigid-body translation and rotation to the background flow        
        if include_rbm:
            u_swim = self.solution.u[frame_index]
            omega_swim = self.solution.omega[frame_index]
        else:
            u_swim = np.zeros(3)
            omega_swim = np.zeros(3)
            

        U_total = U - u_swim
        W_total = W - omega_swim

        # Transform background flow to swimmer frame
        U_body = Q.T @ U_total
        W_body = Q.T @ W_total
        E_body = Q.T @ E @ Q        


        # Get the surface of your mesh, r_surface is the distance from the centerline
        x_surface, r_surface = self.mesh.isosurface.T

        r = np.sqrt(yg**2 + zg**2)  # radial coordinate of each point

        # Find points inside the cell body
        self.inside_mask = points_in_polygon(xg, r, x_surface, r_surface)       

        # Add background flow to all points
        U_background = interaction_object.set_background_flow(U_body, W_body, E_body)
        
        U_field =U_field + U_background

        # Reshape Ufield to vectors and set the velocity of points inside cell body to zero
        U_field = U_field.reshape(Ng, 3)
        U_field[self.inside_mask,:] = 0

        return U_field


    
    
    


        




