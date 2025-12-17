import numpy as np
from typing import Iterable, Callable

from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as R


from .mesh import Mesh
from .system_base  import BaseSystem
from .flagella import SlenderBody
from .flowfield import FlowStokes
from .SaveData import Solution
from .utils import points_in_polygon
from .time_integration import vector_to_quaternion_from_x, rotate_BCs, RK4, forward_euler, omega_to_quat_dot



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

        if self.flagellum_2 is not None:
            self.solution.f2   = np.zeros((N_frames, 3*len(self.flagellum_2[0].r)))
        
        # Run the __post_init__ of BaseSystem 
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

    
    

    def solve(self,
              find_flow:Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]], 
              dt:float) -> Iterable[np.ndarray]:
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

        U_field = U_field + U_boundary

        U_field = U_field.reshape(Ng, 3)
        U_field[self.inside_mask,:] = 0

        return U_field
        



@dataclass
class FreeSwimmer(BaseSystem):

    # function to find the flow field at a point x
    # flow_function   : Callable[[float,np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
    
    flagellum_1     : Iterable[SlenderBody]

    flagellum_2     : Iterable[SlenderBody]     = field(default_factory=lambda: None)




    def __post_init__(self):
        self.N_frames = len(self.flagellum_1)
        self.N_h      = len(self.mesh.centroids)
        self.N_f1     = len(self.flagellum_1[0].r)
        self.N_f2     = len(self.flagellum_2[0].r) if self.flagellum_2 is not None else 0

        # incl_f2  = int(bool(N_f2))
        dimension = 3*self.N_h + 3*self.N_f1 + 3*self.N_f2 + 6 
        
        # Initialise matrices to store the LU decomposition matrix and pivot vector
        self.LU_matrix  = np.zeros((self.N_frames, dimension, dimension))
        self.piv_vector = np.zeros((self.N_frames, dimension))

        self.solution = Solution()

        self.solution.time                  = np.zeros( self.N_frames )
        self.solution.psi                   = np.zeros((self.N_frames, 3*self.mesh.elements))
        self.solution.f1                    = np.zeros((self.N_frames, 3*self.N_f1))
        self.solution.u                     = np.zeros((self.N_frames, 3))
        self.solution.omega                 = np.zeros((self.N_frames, 3))
        self.solution.X                     = np.zeros((self.N_frames, 6))
        self.solution.rotation_matrices     = np.zeros((self.N_frames, 3, 3))
        self.solution.quaternions           = np.zeros((self.N_frames, 4))


        if self.flagellum_2 is not None:
            self.solution.f2   = np.zeros((self.N_frames, 3*self.N_f2))

        super().__post_init__()

        self.populate_grand_mobility_matrix() 


    def populate_grand_mobility_matrix(self):
        """
        Load the mobility matrices of the flagella of each frame. The only thing that changes over time is
        the interaction between the cell body and the flagella.

        This function populates the LU_matrix and piv_vector attributes of the Swimmer class.

    
        """

        Mh,_,_,_  = self.construct_mobility_matrix()   
        r, c      = np.shape(Mh)     


        if self.flagellum_2 is None:
            print("Populating flagellum")
            for i, frame in enumerate(self.flagellum_1):
                Mf1  = frame.construct_mobility_matrix()
                Mf1h = frame.calc_interaction(self.mesh.centroids)
                Mhf1 = FlowStokes(self.mesh, frame.r).MATRIX

                V_h  = np.tile(np.eye(3), int(r/3)).T       
                V_f1 = np.tile(np.eye(3), int(len(Mf1)/3)).T  # matrix filled with identity blocks (3M, 3)
                
                A_h  = self.r_cross_matrix
                A_f1 = frame.calc_r_cross_matrix(self.mesh.center)
                
                F_h  = self.surface_matrix
                F_f1 = V_f1.T               # Also a matrix filled with identity blocks but then (3, 3M)
                
                T_h  = self.torque_matrix
                T_f1 = A_f1.T
        

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
                Mf1  = frame_1.construct_mobility_matrix()
                Mf1h = frame_1.calc_interaction(self.mesh.centroids)
                Mhf1 = FlowStokes(self.mesh, frame_1.r).MATRIX
                Mf1f2 = frame_1.calc_interaction(frame_2.r)

                Mf2  = frame_2.construct_mobility_matrix()
                Mf2h = frame_2.calc_interaction(self.mesh.centroids)
                Mhf2 = FlowStokes(self.mesh, frame_2.r).MATRIX
                Mf2f1 = frame_2.calc_interaction(frame_1.r)

                V_h  = np.tile(np.eye(3), int(r/3)).T       
                V_f1 = np.tile(np.eye(3), int(len(Mf1)/3)).T  # matrix filled with identity blocks (3M, 3)
                V_f2 = np.tile(np.eye(3), int(len(Mf2)/3)).T  # matrix filled with identity blocks (3M, 3)

                A_h  = self.r_cross_matrix
                A_f1 = frame_1.calc_r_cross_matrix(self.mesh.center)
                A_f2 = frame_2.calc_r_cross_matrix(self.mesh.center)

                F_h  = self.surface_matrix
                F_f1 = V_f1.T               # Also a matrix filled with identity blocks but then (3, 3M)
                F_f2 = V_f2.T

                T_h  = self.torque_matrix
                T_f1 = A_f1.T   #removed minus sign (included in transpose)
                T_f2 = A_f2.T   #removed minus sign (included in transpose)

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
                      flow_function      :Callable[[float,np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
                      initial_position   :np.ndarray = np.array([0,0,0]),
                      initial_orientation:np.ndarray = np.array([-1, 0, 0])
                      ) -> Iterable[np.ndarray]:
        """
        Solve the mobility problem over time given a function that provides the boundary conditions.

        Parameters
        ----------
        T : float
            Total time of the simulation.
        dt : float
            Time step of the simulation.

        Returns
        -------
        
        """
        self.flow_function = flow_function

        self.dt = dt

        X_0 = np.hstack((initial_position, initial_orientation))
        
        self.solution.X[0] = X_0

        # Set initial quaternion
        q_0                           = vector_to_quaternion_from_x(initial_orientation)
        self.solution.quaternions[0]  = q_0

        # Find initial cartesian rotation matrix from quaternion
        Q_0                                = R.from_quat(q_0, scalar_first=True).as_matrix()
        self.solution.rotation_matrices[0] = Q_0


        # Unpack initial state
        x, p = X_0[:3], X_0[3:]

        time_index = 0
        # Calculate the singularity distribution, translational-, and angular velocity for the initial state
        phi, u, omega = self.calc_RBM(time_index, x, q_0, self.solution.time[0])

        self.solution.psi[0]   = phi[:3*self.N_h]
        self.solution.f1[0]    = phi[3*self.N_h:3*self.N_h + 3*self.N_f1]

        if self.flagellum_2 is not None:
            self.solution.f2[0]    = phi[3*self.N_h + 3*self.N_f1:]

        self.solution.u[0]     = Q_0 @ u
        self.solution.omega[0] = Q_0 @ omega

        for frame_index in range(self.N_frames-1):
            self.solution.time[frame_index+1] = (frame_index+1) * dt

            # Calculate the next timestep
            x, p, Q, phi, u, omega = self.solve_RBM(x, p, frame_index+1, dt)


            # Save values to solution file
            self.solution.psi[frame_index+ 1]   = phi[:3*self.N_h]
            self.solution.f1[frame_index + 1]   = phi[3*self.N_h:3*self.N_h + 3*self.N_f1]

            if self.flagellum_2 is not None:
                self.solution.f2[frame_index + 1]    = phi[3*self.N_h + 3*self.N_f1:]
 

            self.solution.u[frame_index+1]     = Q @ u
            self.solution.omega[frame_index+1] = Q @ omega

            self.solution.X[frame_index+1,:3]   = x
            self.solution.X[frame_index+1, 3:]  = p

            self.solution.quaternions[frame_index+1]        = vector_to_quaternion_from_x(p) 
            self.solution.rotation_matrices[frame_index+1]  = Q

        return self.solution



            


    def solve_RBM(self,
                  x_initial                :np.ndarray,
                  p_initial                :np.ndarray,
                  time_index               :int,
                  dt                       :float)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given an initial x and p, calculate the next iteration with timestep dt using rk4 numerical integration.

        Parameters
        ----------
        x_initial   : numpy array
                      The initial position (x_i)
        p_initial   : numpy array
                      The initial orientation of the particle (p_i)
        time_index  : int
                      The current time index 
        dt          : float
                      timestep to use for time integration

        Returns
        -------
        x           : numpy array
                      The next position after time dt (x_{i+1})
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
        
        # calculate the initial quaternion vector (size=(1,4)) from the initial orientation
        q_0 = vector_to_quaternion_from_x(p_initial)
        
        # stack the initial position and quaternion vector into an array for time integration
        Y_0 = np.hstack((x_initial, q_0))

        
        # Time integration using RK4 (forward euler), self.quaternion_RK4 is returns the right hand side of the ODE
        Y_next = forward_euler(self.quaternion_RK4, Y_0, time_index*dt, dt)       # t=0 since time dependence is not inplemented

        # Unpack new timestep
        x, q = Y_next[:3], Y_next[3:]

        phi, u, omega = self.calc_RBM(time_index, x, q, time_index*dt)

        # Convert quaternion vector to cartesian matrix
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        # Retrieve new orientation
        p = -Q[:,0]

        return x, p, Q, phi, u, omega
    


    def quaternion_RK4(self, t:float, Y :np.ndarray)->np.ndarray:
        """
        Function that returns the RHS of the ODE to be integrated over time

        Parameters
        ----------      
        t       : float
                  Current time  
        Y       : numpy array (1,7)
                  Array which contains the initial position and quaternion vector (orientation).

        Returns
        -------
        Y_dot   : numpy array (1,7)
                  The array which represents the time derivative at the current timestep (to be integrated).

        """
        index = int(t / self.dt)
        # Use the LU decomposition to solve the system
        Y_dot = self.calc_Y_dot(index, Y, t)
        
        return Y_dot
    

    def calc_Y_dot(self, time_index:int, Y:np.ndarray, t:float)->np.ndarray:
        """
        Calculates the current time derivative with the current state, and the LU decomposition
        of the grand mobility matrix.

        lu and piv are left as variables as this might change per timestep.
        """
        # Unpack state
        x,q = Y[:3], Y[3:]

        # Normalise quaterion vector
        q /= np.linalg.norm(q)

        # compute RBM of state Y
        _, u, omega = self.calc_RBM(time_index, x ,q, t)

        # Transform velocity back to the lab frame
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        u_lab     = Q @ u
        omega_lab = Q @ omega

        # Calculate the time derivative of the quaternion vector
        q_dot = omega_to_quat_dot(q, omega_lab)

        # Make vector containing the time derivatives
        Y_dot = np.hstack((u_lab,q_dot))

        return Y_dot
        

    def calc_RBM(self, time_index, x, q, t)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the RBM of the particle at the current state with the LU decomposition
        of the grand mobility matrix.

        Returns
        -------
        phi     : numpy array (1, N) with N the amount of elements + flagella_1 elements + flagella_2 elements (if present)
                  The solution for the singularity density.
        u       : numpy array (1, 3)
                  Translational velocity of the particle
        omega   : numpy array (1, 3)
                : Angular velocity of the particle 
        """

        # Find the current flowfield at x
        U, W, E = self.flow_function(t,x)  
        
        # Find cartesian rotation matrix
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        # Rotate the boundary conditions to the particle frame
        U_body, W_body, E_body = rotate_BCs(Q, U, W, E)
        
        # Impose the boundary conditionsfrom scipy.sparse.linalg import cg, gmres

        rhs_h = self.set_boundary_condition(U_body, W_body, E_body)
        rhs_f1 = self.flagellum_1[time_index].set_boundary_condition(U_body, W_body, E_body)

        if self.flagellum_2 is None:            
            rhs = np.concatenate([rhs_h, rhs_f1])

        else:
            rhs_f2 = self.flagellum_2[time_index].set_boundary_condition(U_body, W_body, E_body)
            rhs = np.concatenate([rhs_h, rhs_f1, rhs_f2])

        
        RHS = np.hstack((rhs, np.zeros(6)))

        # No slip, so solve for negative RHS
        sol = lu_solve((self.LU_matrix[time_index], self.piv_vector[time_index]), -RHS)
        
        # Unpack solution
        phi   = sol[:-6]
        u     = sol[-6:-3] 
        omega = sol[-3:]


        return phi, u, omega
    
    
    def calc_vector_field(self,
                          interaction_object : FlowStokes,
                          frame_index        : int        ,
                          find_flow          : Callable[[float,np.ndarray], tuple[np.ndarray]],
                          include_bg         : bool = True) -> np.ndarray:
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
        inc
            
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
        U, W, E = find_flow(frame_index*self.dt, self.solution.X[frame_index,:3])

        Q = self.solution.rotation_matrices[frame_index]

          # Add swimmer rigid-body translation and rotation to the background
        # flow so the vector field is returned in the lab frame. If the
        # current frame solution is not available yet, fall back to zeros.
        if include_bg:
            u_swim = self.solution.u[frame_index]
            omega_swim = self.solution.omega[frame_index]
        else:
            u_swim = np.zeros(3)
            omega_swim = np.zeros(3)
            

        U_total = U - u_swim
        W_total = W - omega_swim

        U_body = Q.T @ U_total
        W_body = Q.T @ W_total
        E_body = Q.T @ E @ Q
        


        # Get the surface of your mesh, r_surface is the distance from the centerline
        x_surface, r_surface = self.mesh.isosurface.T

        r = np.sqrt(yg**2 + zg**2)  # radial coordinate of each point

        self.inside_mask = points_in_polygon(xg, r, x_surface, r_surface)

       

        U_boundary = interaction_object.set_background_flow(U_body, W_body, E_body)
        

        U_field =U_field + U_boundary 

        U_field = U_field.reshape(Ng, 3)#- self.solution.u[frame_index]
        U_field[self.inside_mask,:] = 0

        return U_field


    
    
    


        




