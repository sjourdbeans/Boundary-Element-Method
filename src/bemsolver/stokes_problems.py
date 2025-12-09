import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as R


from .system_base import BaseSystem
from .time_integration import vector_to_quaternion_from_x, rotate_BCs, omega_to_quat_dot, RK4
from .SaveData import Solution




@dataclass
class ResistanceProblem(BaseSystem):
    """
    This child class inherits all methods from BaseSystem to calculate the mobility matrix 
    (or in this case the resistance matrix) and to set up all the necessary information to 
    solve the resistance problem of an object in a flowfield. The mobility matrix
    is constructed by finding the interaction of each element on every other
    element. On the boundary of the mesh there is a stresslet distribution, and on the centerline of the mesh there
    is a line singularity of stokeslets and rotlets.

    NOTE: This class assumes that the particle is clamped. This means that the fluid exerts a force on the particle,
    and thus the drag can be calculated and compared to theoretical/experimental values.

    Parameters
    ----------
    mesh                : Mesh instance
                          The mesh to be used is a bemsolver.Mesh python object.

    Example
    -------

    Solve the resistance problem for a given mesh for a flow in the x-direction.

    >>> import bemsolver as bem
    >>> import numpy as np

    >>> mesh = bem.Mesh("/path_to_mesh/file.mat")
    >>> sys  = bem.ResistanceProblem(mesh) 

    >>> # Background flow

    >>> U = np.zeros(3)

    >>> U[0] = 1
    >>> U[1] = 0
    >>> U[2] = 0

    >>> # No background vorticity
    >>> W = np.zeros(3)

    >>> # No rate of strain (The default is an array of zeros).
    >>> E = np.zeros((3,3))

    >>> psi, force, torque = sys.solve(U,W,E)
    >>> print(f"Total Force: {force}, Total Torque: {torque}")

    """

    
    
    def solve(self,
              U:np.ndarray,
              W:np.ndarray,
              E:np.ndarray = np.zeros((3,3)))->tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            cbars=[]
            for i in range(3):
                fig,ax,cbar=plotting.plot_panels_stokes(self.mesh.panels,psi[:,i])
                figs.append(fig)
                axes.append(ax)
                cbars.append(cbar)
            return figs, axes, cbars
        except:
            raise SyntaxError("System has not been solved yet! Run System.solve(RHS) before plotting.")




@dataclass
class MobilityProblem(BaseSystem):
    """
    This child class inherits all methods from BaseSystem to calculate the mobility matrix and other necessary
    information to solve the mobility problem of an object in a flow. The mobility matrix is constructed in the
    same way as for the ResistanceProblem by finding the interactions of each element on the other elements.
    To find the Rigid Body Motion (RBM) of the object, we must set the constraints that the particle is
    force and torque free. Furthermore, we need to include the RBM of the particle (U_t + omega x r).
    Since we have a no slip condition we get the following equations: \n

        Matrix         @ psi - U_t - omega x r = u   \n   
        surface_matrix @ psi                   = 0   \n
        torque_matrix  @ psi                   = 0   \n

    This can be written as a linear system with a grand mobility matrix \n

        M @ [psi, U_t, omega].T = [u, 0, 0].T   \n

    M includes the surface_matrix, torque_matrix, the translational velocity matrix (identities), and the 
    matrix representation of the cross product with r.

    Parameters
    ----------
    mesh                : Mesh instance
                          The mesh to be used is a bemsolver.Mesh python object.
    flow_function       : Callable representing the flowfield.
                          flow_function should be a function which returns U, W, and E for a given position x

    initial_position    : (optional) numpy array
                          Initial position of the particle [x, y, z]
    initial_orientation : (optional) numpy array default is [0, 0, 0]
                          Initial orientation unit vector of the particle [p_x, p_y, p_z] default is [1, 0, 0]
    particle_velocity   : (optional) float or integer default is 0
                          If the particle has a constant velocity in the direction of its orientation,
                          this can be set. Overall unnecessary if the particle will be self propelled.
                          

    Example
    -------

    Calculate the next orientation and position for a given mesh in a shear flow with an initial position and orientation.

    >>> import bemsolver as bem
    >>> import numpy as np

    >>> mesh = bem.Mesh("/path_to_mesh/file.mat")
    >>> gamma_dot = 1
    >>>
    >>> def find_flow(x):
    >>>        U = np.zeros(3)
    >>>
    >>>        U[0] = gamma_dot * x[1]
    >>>        U[1] = 0
    >>>        U[2] = 0
    >>>
    >>>        # Background vorticity
    >>>        W = np.zeros(3)  
    >>>
    >>>        W[0] = 0
    >>>        W[1] = 0
    >>>        W[2] = -gamma_dot/2
    >>>
    >>>        # Rate of strain tensor
    >>>        E = gamma_dot/2*np.array([[0,1,0],
    >>>                                  [1,0,0],
    >>>                                  [0,0,0]])
    >>>        return U, W, E
    >>>

    >>> sys  = bem.MobilityProblem(mesh, flow_function=find_flow) 
    >>> initial_orientation = np.array([1,0,0])
    >>> initial_position    = np.array([0,0,0])
    >>> dt=0.01
    >>>
    >>> # calculate the next position, orientation, and quaternion rotation matrix (to rotate the BCs)
    >>> x, p, Q = sys.solve_RBM(initial_position, initial_orientation, dt)


    


    """
    # function to find the flow field at a point x
    flow_function       :Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]

    initial_position    :np.ndarray = field(default_factory=lambda: np.array([0,0,0]))  
    initial_orientation :np.ndarray = field(default_factory=lambda: np.array([1,0,0]))

    particle_velocity   :float|int = field(default_factory=lambda: 0)

    

    
    def construct_grand_mobility_matrix(self):
        """
        Assemble the grand mobility matrix by first constructing the general mobility matrix, and then adding
        the force and force free constraints, in addition to the translational and rotational matrices V and A.
        """

        self.construct_mobility_matrix()
        
        r, c = np.shape(self.MATRIX)        

        # Generate r/3 identity matrices stacked vertically (r,3) 
        V = np.tile(np.eye(3), int(r/3)).T

        # cross product matrix to be multiplied with omega
        A = self.r_cross_matrix

        # force and torque free constraints
        F = self.surface_matrix
        T = self.torque_matrix

        # Grand mobility matrix
        M=np.block([
            [self.MATRIX, -V, -A],
            [F, np.zeros((3,6))],
            [T, np.zeros((3,6))]
        ])

        # LU decomposition for quick solve with changing BCs
        self.lu, self.piv=lu_factor(M)

        # Can be used outside of class if necessary
        return M
    

    def RBM_over_time(self,
                      T_max:int|float,
                      dt    :float)->Solution:
        """
        Find the Rigid Body Motion of a particle in a given time interval and timestep. The initial position and orientation are
        can be set when initialising the MobilityProblem instance. This function returns a dataclass with 
        the solutions.

        Parameters
        ----------
        T_max   : integer or float
                  Set the maximum time to be evaluated
        dt      : float
                  Timestep for the simulation

        Returns
        -------
        Solution dataclass containing

        - time                
        - X                   
        - rotation_matrices   
        - quaternions         
        - psi                 
        - u                   
        - omega    
        
        Example
        -------
        The solution for X can be accesed by running

        >>> solution = sys.RBM_over_time(T=100, dt=0.01)
        >>> X = solution.X


        """
        # Construct the grand mobility matrix
        self.construct_grand_mobility_matrix()
        
        # Initialise the solution file (dataclass)
        solution = Solution()        
        
        solution.time = np.arange(0, T_max+dt, dt)
        solution.psi  = np.zeros((len(solution.time),3*self.mesh.elements))
        solution.u    = np.zeros((len(solution.time),3))
        solution.omega= np.zeros((len(solution.time),3))

        # Set initial position and orientation vector
        X_0 = np.hstack((self.initial_position, self.initial_orientation))

        solution.X    = np.zeros((len(solution.time),6))
        solution.X[0] = X_0

        

        solution.rotation_matrices     = np.zeros((len(solution.time), 3, 3))
        solution.quaternions           = np.zeros((len(solution.time), 4))
        
        # Set initial quaternion
        q_0                           = vector_to_quaternion_from_x(self.initial_orientation)
        solution.quaternions[0]        = q_0

        # Find initial cartesian rotation matrix from quaternion
        Q_0                           = R.from_quat(q_0, scalar_first=True).as_matrix()
        solution.rotation_matrices[0] = Q_0

        # Unpack initial state
        x, p = X_0[:3], X_0[3:]

        # Calculate the singularity distribution, translational-, and angular velocity for the initial state
        self.psi, self.u, self.omega = self.calc_RBM(self.lu, self.piv, x, q_0, solution.time[0])

        # Set initial values to solution file
        solution.psi[0]   = self.psi
        solution.u[0]     = self.u
        solution.omega[0] = self.omega


        # Loop through time
        for k, t in enumerate(solution.time[:-1]):    

            # Calculate the next timestep
            x, p, Q = self.solve_RBM(x, p, t, dt)

            # Save values to solution file
            solution.psi[k+1]   = self.psi
            solution.u[k+1]     = self.u
            solution.omega[k+1] = self.omega

            solution.X[k+1,:3]   = x
            solution.X[k+1, 3:]  = p

            solution.quaternions[k+1]        = vector_to_quaternion_from_x(p) 
            solution.rotation_matrices[k+1] = Q

        return solution


    

    def solve_RBM(self,
                  x_initial                :np.ndarray,
                  p_initial                :np.ndarray,
                  t                        :float,
                  dt                       :float)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given an initial x and p, calculate the next iteration with timestep dt using rk4 numerical integration.

        Parameters
        ----------
        x_initial   : numpy array
                      The initial position (x_i)
        p_initial   : numpy array
                      The initial orientation of the particle (p_i)
        t           : float
                      current time 
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
        """
        
        # calculate the initial quaternion vector (size=(1,4)) from the initial orientation
        q_0 = vector_to_quaternion_from_x(p_initial)
        
        # stack the initial position and quaternion vector into an array for time integration
        Y_0 = np.hstack((x_initial, q_0))

        
        # Time integration using RK4, self.quaternion_RK4 is returns the right hand side of the ODE
        Y_next = RK4(self.quaternion_RK4, Y_0, t, dt)       # t=0 since time dependence is not inplemented

        # Unpack new timestep
        x, q = Y_next[:3], Y_next[3:]

        self.psi, self.u, self.omega = self.calc_RBM(self.lu, self.piv, x, q, t)

        # Convert quaternion vector to cartesian matrix
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        # Retrieve new orientation
        p = Q[:,0]

        return x, p, Q




    def quaternion_RK4(self, t:float, Y :np.ndarray)->np.ndarray:
        """
        Function that returns the RHS of the ODE to be integrated over time

        Parameters
        ----------        
        Y       : numpy array (1,7)
                  Array which contains the initial position and quaternion vector (orientation).

        Returns
        -------
        Y_dot   : numpy array (1,7)
                  The array which represents the time derivative at the current timestep (to be integrated).

        """
        # Use the LU decomposition to solve the system
        Y_dot = self.calc_Y_dot(self.lu, self.piv, Y, t)
        
        return Y_dot
    

    def calc_Y_dot(self, lu, piv, Y:np.ndarray, t:float)->np.ndarray:
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
        _, u, omega = self.calc_RBM(lu, piv, x ,q, t)

        # Transform velocity back to the lab frame
        Q = R.from_quat(q, scalar_first=True).as_matrix()

        u_lab = Q @ (u+self.particle_velocity*np.array([1, 0, 0]))

        # Calculate the time derivative of the quaternion vector
        q_dot = omega_to_quat_dot(q, omega)

        # Make vector containing the time derivatives
        Y_dot = np.hstack((u_lab,q_dot))

        return Y_dot


    def calc_RBM(self, lu, piv, x, q, t)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the RBM of the particle at the current state with the LU decomposition
        of the grand mobility matrix.

        Returns
        -------
        psi     : numpy array (1, N) with N the amount of elements
                  The solution for the singularity distribution density.
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
        
        # Impose the boundary conditions
        U_rhs = self.set_boundary_condition(U_body, W_body, E_body)
        
        RHS = np.hstack((U_rhs, np.zeros(6)))

        # No slip, so solve for negative RHS
        phi = lu_solve((lu, piv), -RHS)

        # Unpack solution
        psi   = phi[:-6]
        u     = phi[-6:-3] 
        omega = phi[-3:]


        return psi, u, omega


  

    
    



        
    




