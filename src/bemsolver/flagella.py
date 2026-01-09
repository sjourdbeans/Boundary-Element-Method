import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve

from .utils import U_colloc, skew_stack
from .kernels import stokeslet, tangential


@dataclass  
class SlenderBody:
    """
    This class calculates the cartesian coordinates of a slender body (commonly used to simulate flagella) 
    given an array of curvatures and torsions when it is called. The amount of curvatures (and torsions) you pass along
    determines the amount of nodes on the body. Therefore, if curvature has length N, then the amount of elements on the body is 
    N-1. The curve is determined using the Frenet-Serret eqs.\n

    With the calculated curve, the mobility matrix and interaction matrix can be calculated using Slender Body Theory.
    To calculate the the mobility matrix use the method SlenderBody.construct_mobility_matrix()
    To calculate the interaction matrix use the method SlenderBody.calc_interaction(points) \n

    Parameters
    ----------
    curvature       : numpy array   (N,)
                      An array containing N curvatures for N nodes. So N-1 elements.
    torsion         : numpy array   (N,)
                      An array containing N torsions for N nodes. If the body is planar use an array of zeros.
    theta_0         : float
                      Initial angle in radians between the initial tangent vector and the x-axis.
    rho_0           : float (optional) automatically set to 0
                      Initial angle in radians between the initial binormal vector and the z-axis.
    base_position   : numpy array (optional) automatically set to origin.
                      Starting location of the curve.
    flagellum_length: float or int (optional) automatically set to 10.
                      Length of the flagellum (or in general the length of the curve).
    flagellum_radius: float or int (optional) automatically set to 0.2.
                      Radius of the flagellum (or in general the radius of the curve).
    smin            : float (optional) automatically set to 0.15
                      Starting arclength of the curve in terms of its length. Autmatically set to start 
                      at 0.15L to avoid being to close to the cell body.
    velocity        : numpy array (optional) (N, 3) automatically set to zeros(N,3)
                      The velocity of the body at each element (Flagellum velocity).

    Example
    -------
    Calculate the force on a slender beam in a flow u = [1, 0, 0].

    >>> import bemsolver as bem
    >>> import numpy as np

    >>> U    = np.zeros(3)
    >>> U[0] = 1
    >>> W    = np.zeros(3)

    >>> initial_angle = np.pi/2
    >>> elements      = 20
    >>> curvature     = np.ones(elements+1)
    >>> torsion       = np.zeros_like(curvature)

    >>> beam          = bem.SlenderBody(curvature, torsion, theta_0 = initial_angle, smin =0)
    >>> M             = beam.construct_mobility_matrix()
    >>> RHS           = beam.set_boundary_condition(U, W)   # rate of strain tensor is optional

    >>> f             = np.linalg.solve(M, RHS)
    >>> # reshape f to force vectors per element
    >>> force_vectors = f.reshape(int(len(f)/3),3)

    
    """
       
 


    def construct_mobility_matrix(self)->np.ndarray:
        """
        Constructs the mobility matrix of the slender body.

        Returns
        -------
        MATRIX  : numpy array of shape (3N, 3N)
                  The mobility matrix of the slender body
        """

        self.MATRIX = self.calc_mobility()   

        return self.MATRIX


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
        U_t+U_r+U_e+U_f : numpy array (3N,)
                          The total background flow on each element
        """
        
        rows, columns = np.shape(self.MATRIX)

        U_t, U_r, U_e =U_colloc(U,W, self.r,int(rows/3), E) 

        return U_t+U_r+U_e - self.velocity.flatten()
            
            


    def calc_mobility(self):
        """
        Calculates the mobility matrix of the body using slender body theory.

        Returns
        -------
        K       : numpy array (3N, 3N)
                  Mobility matrix
        """

        # Initialise H matrix
        H = np.zeros((3*self.Nf, 3*self.Nf))

        # create a row vector full of ones
        ones_array   = np.ones((1, self.Nf))
        
        # Unpack the node coordinates
        x, y, z = self.r.T
        
        # Create matrices where each column consists of the x,y, or z locations of all elements
        Xi = np.outer(x, ones_array)
        Yi = np.outer(y, ones_array)
        Zi = np.outer(z, ones_array)

        # Create matrix which contains the length of each element on every row (columns are the same)
        Li = np.outer(self.element_lengths, ones_array)

        # Unpack tangent vector into xyz components
        t_x = self.tangents[:,0]
        t_y = self.tangents[:,1]
        t_z = self.tangents[:,2]

        # Store components in a tuple
        T= (t_x, t_y, t_z)

        # Create matrix which contains the centroid location along the arc of the curve (the same columns)
        Si = np.outer(self.flag_centroids, ones_array)

        # Compute matrix where entry ij represents ds_j / ds_i (size of elements j divided by size of element i)
        Lij = Li.T / Li

        # Compute distances between each element
        Xij = Xi - Xi.T
        Yij = Yi - Yi.T
        Zij = Zi - Zi.T

        # Difference along the arc of the curve
        Sij = np.abs(Si - Si.T) 

        # Compute stokeslet contribution
        G = stokeslet(Xij, Yij, Zij)

        # Compute the tangential contribution
        L = tangential(Lij, Sij, T)


        # Compute Lambda (Shape dependence)

        constant = np.log(self.slend_2)  
         # Loop over the elements and compute the self interacting matrices
        for i in range(self.Nf):
            ti = self.tangents[i]
            tt = np.outer(ti, ti)
            # Hi = (-constant[i] + 1)*np.eye(3) - (constant[i] + 3)*tt
            Hi = (constant[i] - 1)*np.eye(3) + (constant[i] + 3)*tt
            Hi /= self.element_lengths[i]
            H[3*i:3*i+3, 3*i:3*i+3] = Hi

        # H is a block diagonal matrix

        # Assemble the total mobility matrix
        K = 1/(8*np.pi)*(G - H - L)

        return  K
    

    def calc_interaction(self,
                         evaluation_points:np.ndarray)->np.ndarray:
        """
        Calculate the interaction matrix between the body and points that are not on the curve itself.

        Parameters
        ----------
        evaluation_points   : numpy array (M, 3)
                            : M points at which the interaction of the body is to be evaluated

        Returns
        -------
        interaction_matrix  : numpy array (3M, 3N)
                              Matrix that describes the interaction between the body and the evaluation points
        """
        

        N_p = np.shape(evaluation_points)[0]

        # Unpack the coordinates of the body and evaluation points
        xf, yf, zf = self.r.T
        xp, yp, zp =evaluation_points.T

        # Initialise interaction matrix
        M = np.zeros((3*N_p, 3*self.Nf))

        # Create column vector and row vector of ones
        ones_eval   = np.ones((N_p,1))
        ones_array  = np.ones((1, self.Nf))

        # Create matrices with xyz coordinates of the evaluation points
        Xi = np.outer(xp, ones_array)
        Yi = np.outer(yp, ones_array)
        Zi = np.outer(zp, ones_array)

        # Create matrices with xyz coordinates of the element coordinates        arc
        Xj = np.outer(ones_eval, xf)
        Yj = np.outer(ones_eval, yf)
        Zj = np.outer(ones_eval, zf)

        # Calculate difference in xyz coordinates between the elements and evaluation points
        Xij = Xi - Xj
        Yij = Yi - Yj
        Zij = Zi - Zj

        # Calculate cartesian distance      
        Rij = np.sqrt(Xij**2 + Yij**2 + Zij**2 ) + 0.5 * self.flagellum_radius

        # Create unitless xyz coordinates
        Xij = Xij / Rij
        Yij = Yij / Rij
        Zij = Zij / Rij       

        # index arrays
        idx_p = np.arange(0, 3*N_p, 3)
        idx_f = np.arange(0, 3*self.Nf, 3)

        # Populate the interaction matrix (Tornberg & Shelley, note that their prefactor is wrong)
        M[np.ix_(idx_p, idx_f)]         =  (1 + Xij * Xij) / Rij + (self.flagellum_radius**2/2) * (1 - 3 * Xij * Xij) / Rij**3
        M[np.ix_(idx_p, idx_f + 1)]     =  (    Xij * Yij) / Rij + (self.flagellum_radius**2/2) * (  - 3 * Xij * Yij) / Rij**3
        M[np.ix_(idx_p, idx_f + 2)]     =  (    Xij * Zij) / Rij + (self.flagellum_radius**2/2) * (  - 3 * Xij * Zij) / Rij**3

        M[np.ix_(idx_p + 1, idx_f)]     =   M[np.ix_(idx_p, idx_f+1)]
        M[np.ix_(idx_p + 1, idx_f + 1)] =  (1 + Yij * Yij) / Rij + (self.flagellum_radius**2/2) * (1 - 3 * Yij * Yij) / Rij**3
        M[np.ix_(idx_p + 1, idx_f + 2)] =  (    Yij * Zij) / Rij + (self.flagellum_radius**2/2) * (  - 3 * Yij * Zij) / Rij**3

        M[np.ix_(idx_p + 2, idx_f)]     =   M[np.ix_(idx_p, idx_f + 2)]
        M[np.ix_(idx_p + 2, idx_f + 1)] =   M[np.ix_(idx_p + 1, idx_f + 2)]
        M[np.ix_(idx_p + 2, idx_f + 2)] =  (1 + Zij * Zij) / Rij + (self.flagellum_radius**2/2) * (1 - 3 * Zij * Zij) / Rij**3

        interaction_matrix = 1/(8*np.pi) * M
        return interaction_matrix
    
    
    def calc_r_cross_matrix(self, X_center:np.ndarray=np.zeros(3))->np.ndarray:
        """
        Calculate the torque matrix at given center points.

        Parameters
        ----------
        X_center            : numpy array (3,)
                             Reference point for the torque, e.g. center of the cell body.
        Returns
        -------
        r_cross_matrix      : numpy array (3M, 3)
                             The matrix representing the cross product of r with an arbitrary vector.
        """

        R = self.r - X_center  # position vectors from center points to flagellum elements
        
        # Calculate r cross matrix at the center points
        r_cross_matrix = skew_stack(R)    

        return r_cross_matrix


@dataclass
class SlenderCoordinates(SlenderBody):

    points              :np.ndarray

    velocity            : np.ndarray            = field(default_factory=lambda: None)
    flagellum_radius    : int|float             = field(default_factory=lambda: 0.2)


    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.r[1:])
        
        Nf = len(self.points)
    
        self.t = self.points[1:]-self.points[:-1]       

        
        self.flagellum_length = 20#np.sum(np.linalg.norm(self.t,axis=1))

        self.ss = np.linspace(0, self.flagellum_length, Nf)
        self.Nf=len(self.ss)-1


        self.flag_centroids  = (self.ss[1:] + self.ss[:-1]) / 2
        self.element_lengths =  self.ss[1:] - self.ss[:-1]

        
        self.tangents  = self.t/np.linalg.norm(self.t, axis=1, keepdims=True)

        self.slenderness = self.flagellum_radius / self.flagellum_length

        self.slend_2 = self.slenderness**2 * np.ones_like(self.flag_centroids)

        self.r=self.points[1:]
        self.velocity=self.velocity[1:]

    



        

@dataclass
class SlenderCurvTors(SlenderBody):

    curvature           : np.ndarray[float|int]
    torsion             : np.ndarray[float|int]
    theta_0             : int|float  

    
    rho_0               : int|float             = field(default_factory=lambda: 0)
    base_position       : np.ndarray[float|int] = field(default_factory=lambda: np.array([0, 0, 0]))
    flagellum_length    : int|float             = field(default_factory=lambda: 10)
    flagellum_radius    : int|float             = field(default_factory=lambda: 0.2) 
    smin                : int|float             = field(default_factory=lambda: 0.15)
    velocity            : np.ndarray            = field(default_factory=lambda: None)


    

    def __post_init__(self):
        if len(self.curvature)!= len(self.torsion):
            raise IndexError(f"Curvatures and torsion must have the same length ({len(self.curvature)} != {len(self.torsion)})")

        if self.velocity is None:
            self.velocity = np.zeros((len(self.curvature),3))

        self.curvature       = self.curvature.copy()/self.flagellum_length

        Nf                   = len(self.curvature) - 1
        self.ssold           = np.linspace(0,self.flagellum_length, Nf+1)
        self.indstart        = np.min(np.where(self.ssold >= self.smin * self.flagellum_length))

        self.velocity        = self.velocity[self.indstart + 1 :]

        self.ds              = self.ssold[1]-self.ssold[0]

        self.ss              = self.ssold[self.indstart:]
        self.Nf              = len(self.ss) -1

        self.flag_centroids  = (self.ss[1:] + self.ss[:-1]) / 2
        self.element_lengths =  self.ss[1:] - self.ss[:-1]
        
        self.slenderness = self.flagellum_radius / self.flagellum_length

        self.r_epsilon = 2 * self.slenderness * np.sqrt(self.flag_centroids*(self.flagellum_length - self.flag_centroids))

        # slend_2 might be flaggellum specific
        # self.slend_2 = self.flagellum_radius**2/(4*(self.flagellum_length - self.flag_centroids)*self.flag_centroids)
        # self.slend_2 = self.slenderness**2/(4*(1 - self.flag_centroids/self.flagellum_length)*self.flag_centroids/self.flagellum_length)
        # self.slend_2 = (self.r_epsilon / self.flagellum_length)**2
        self.slend_2 = self.slenderness**2 * np.ones_like(self.flag_centroids)
        # Frenet-Serret setup
        self.tangents = np.zeros((len(self.ssold), 3))
        self.r        = np.zeros((len(self.ssold), 3))
        self.r[0]     = self.base_position
        
        # Initial tangent vector

        self.T_0 = np.array([np.cos(self.theta_0) * np.cos(self.rho_0), 
                             np.sin(self.theta_0) * np.cos(self.rho_0),
                             np.sin(self.rho_0)])
        
        # self.T_0 = np.array([
        #     -self.c/np.sqrt(self.R**2 + self.c**2),
        #     self.R/np.sqrt(self.R**2 + self.c**2),
        #     0.0
        # ])

        # self.N_0 = np.array([-1.0, 0.0, 0.0])
        # self.B_0 = np.array([ 0.0, 0.0, 1.0])
        
        self.tangents[0] = self.T_0
        
        # Initial normal vector
        # ref = np.array([0.0, 0.0, 1.0])        # z-axis by default

        # if np.abs(np.dot(ref, self.T_0)) > 0.9:
        #     ref = np.array([0.0, 1.0, 0.0]) 
        # self.N_0 = np.cross(self.T_0, ref)   


        self.B_0 = np.array([-np.cos(self.theta_0) * np.sin(self.rho_0), 
                             -np.sin(self.theta_0) * np.sin(self.rho_0),
                              np.cos(self.rho_0)])
        
        # self.N_0 = np.array([-np.sin(self.theta_0),np.cos(self.theta_0),0])
        self.N_0 = np.cross(self.B_0,self.T_0)
        
        # Initial binormal vector
        # self.B_0 = np.cross(self.T_0, self.N_0)

        self.calc_curve()
        
        self.r        = self.r[self.indstart+1:]
        self.tangents = self.tangents[self.indstart+1:]


    def calc_curve(self):
        """
        Calculate the curve geometry based on the curvature and torsion arrays using the Frenet-Serret equations.
        Given an initial tangent, normal, and binormal vector, integrate the Frenet-Serret equations using
        a Runge-Kutta 4 scheme.

        """
        # Define the state vector with the current position, tangent, normal, and binormal.
        Y = np.concatenate([self.base_position, self.T_0, self.N_0, self.B_0])

        # Loop over all nodes (NOT ELEMENTS) and determine the cartesian coordinates of the nodes
        for i in range(1,len(self.ssold)):

            def rhs(y,s):
                """
                Calculate the right hand side of the Frenet-Serret ODEs
                """
                r = y[0:3]
                T = y[3:6]          
                N = y[6:9]
                B = y[9:12]

                # Make a linear interpolation of the curvature at a location s between the two nodes. 
                alpha = (s-self.ssold[i-1])/(self.ssold[i]-self.ssold[i-1])
                
                kappa_i = self.curvature[i-1] + alpha * (self.curvature[i] - self.curvature[i-1]) 
                tau_i   = self.torsion[i-1] + alpha * (self.torsion[i] - self.torsion[i-1]) 

                # Frenet-Serret eqs
                dr =  T
                dT =  kappa_i*N
                dN = -kappa_i*T + tau_i*B
                dB = -tau_i*N

                return np.concatenate([dr, dT, dN, dB])
            
            # Integrate system to find the next state
            Y = _rk4_step(rhs,Y,self.ssold[i-1],self.ds)

            # Unpack state
            T_next = Y[3:6]
            N_next = Y[6:9]
            B_next = Y[9:12]

            # Make sure the Frenet basis remains orthogonal
            T_next = T_next / np.linalg.norm(T_next)
            N_next = N_next - np.dot(N_next, T_next) * T_next
            N_next = N_next / np.linalg.norm(N_next)
            B_next = np.cross(T_next, N_next)

            # Pack state
            Y[3:6]  = T_next
            Y[6:9]  = N_next
            Y[9:12] = B_next

            # Store the coordinates and tangents in attributes
            self.r[i]        = Y[:3]
            self.tangents[i] = T_next  


def _rk4_step(f: callable, 
              y:np.ndarray, 
              s:float|int, 
              ds:float|int)->np.ndarray:
    """
    Helper function to help integrate the Frenet-Serret eqs. using RK4

    Parameters
    ----------
    f   : function
          The function that returns the RHS of the Frenet-Serret eqs for a given state
    y   : numpy array
          Current state
    s   : float | int
          Current location along the curve (arclength)
    ds  : float |int
          Element size 
    """
    k1 = f(y,s)
    k2 = f(y + ds*k1/2., s + ds/2)
    k3 = f(y + ds*k2/2., s + ds/2)
    k4 = f(y + ds*k3, s + ds)
    return y + ds*(k1 + 2*k2 + 2*k3 + k4)/6.







