import numpy as np
from dataclasses import dataclass, field

from .utils import U_colloc, skew_stack
from .kernels import stokeslet, tangential


@dataclass  
class SlenderBody:
    """
    This parent class contains the code to assemble the mobility matrix, and interaction matrix of a slender body
    according to Slender Body Theory (Tornberg and Shelley) \n

    You do not directly use this class, but instead use 

    - SlenderCoordinates (if you already have the cartesian coordinates of your curve)
    - SlenderAngles      (if you have the 'in-plane' angles and 'out-of-plane' angles of the curve)
    - SlenderCurvTors    (if you have the curvatures and torsions of the curve)
    
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

        R = np.copy(self.r) - X_center  # position vectors from center points to flagellum elements
        
        # Calculate r cross matrix at the center points
        r_cross_matrix = skew_stack(R)    

        return r_cross_matrix


@dataclass
class SlenderCoordinates(SlenderBody):
    """
    This class constructs a slender body directly from a set of Cartesian points 
    describing the centerline of the body. The number of input points determines 
    the number of nodes on the body. If points has length N, then the number of 
    elements is N-1.

    The tangent vectors are computed from the spatial gradient of the coordinates.
    Using these coordinates, the mobility matrix and interaction matrix can be 
    calculated using Slender Body Theory.

    To calculate the mobility matrix use the method `SlenderCoordinates.construct_mobility_matrix()`
    To calculate the interaction matrix between the slender body and some coordinates 'coords',
    use the method `SlenderCoordinates.calc_interaction(coords)`.

    Parameters
    ----------
    points            : numpy array (N, 3)
                        Cartesian coordinates describing the centerline of the body.
                        N nodes define N-1 elements.
    velocity          : numpy array (N, 3) (optional)
                        Velocity of the body at each node. Automatically set to zero
                        if not provided.
    flagellum_radius  : float or int (optional), default 0.2
                        Radius of the slender body.
    flagellum_length  : float or int (optional), default 7
                        Total length of the body.
    smin              : float (optional), default 0
                        Fraction of the total length at which the curve starts.
                        Points before smin * flagellum_length are discarded.

    Attributes
    ----------
    r : numpy.ndarray (Nf, 3)
        Cartesian coordinates of element nodes after truncation.
    tangents : numpy.ndarray (Nf, 3)
        Unit tangent vectors along the body.
    ss : numpy.ndarray
        Arclength positions of nodes.
    flag_centroids : numpy.ndarray
        Arclength positions of element midpoints.
    element_lengths : numpy.ndarray
        Length of each element.
    Nf : int
        Number of elements.
    slenderness : float
        Radius-to-length ratio.
    slend_2 : numpy.ndarray
        Squared slenderness parameter per element.

    Example
    -------
    Construct a slender body from known coordinates.

    >>> import bemsolver as bem
    >>> import numpy as np
    >>> U = np.zeros(3)
    >>> U[0] = 1
    >>> W = np.zeros(3)
    >>>
    >>> pts = np.random.rand(21, 3)
    >>> body = bem.SlenderCoordinates(points=pts)
    >>> M = body.construct_mobility_matrix()

    >>> RHS           = body.set_boundary_condition(U, W)   # rate of strain tensor is optional

    >>> f             = np.linalg.solve(M, RHS)
    >>> # reshape f to force vectors per element
    >>> force_vectors = f.reshape(int(len(f)/3),3)
   
    """

    points              :np.ndarray

    velocity            : np.ndarray            = field(default_factory=lambda: None)
    flagellum_radius    : int|float             = field(default_factory=lambda: 0.2)
    flagellum_length    : int|float             = field(default_factory=lambda: 7)
    smin                : int|float             = field(default_factory=lambda: 0)


    def __post_init__(self):
        
        # Amount of nodes
        Nf = len(self.points)

        # Arclength of the body
        self.ssold = np.linspace(0, self.flagellum_length, Nf)

        # Find the index at which s is larger than smin*length
        self.indstart        = np.min(np.where(self.ssold >= self.smin * self.flagellum_length))
        
        # New arclength
        self.ss = self.ssold[self.indstart:]
        
        # Amount of ELEMENTS (#new nodes - 1)
        self.Nf=len(self.ss)-1

        # Calc the flagellum element centroids and element lengths
        self.flag_centroids  = (self.ss[1:] + self.ss[:-1]) / 2
        self.element_lengths =  self.ss[1:] - self.ss[:-1]

        # Slenderness of the body
        self.slenderness = self.flagellum_radius / self.flagellum_length

        #==============cylinder=================

        # self.r_epsilon = 2 * self.slenderness * np.sqrt(self.flag_centroids*(self.flagellum_length - self.flag_centroids))

        # self.slend_2 = (self.r_epsilon / self.flagellum_length)**2

        #========================================

        self.slend_2 = self.slenderness**2 * np.ones_like(self.flag_centroids)

        self.r=self.points[self.indstart+1:]

        # Calc the tangent vectors along the curve
        self.t = np.gradient(self.r, self.element_lengths[0], axis=0)
        self.tangents  = self.t/np.linalg.norm(self.t, axis=1, keepdims=True)

        # If no velocity is passed on, use an array of zeros
        if self.velocity is None:
            self.velocity = np.zeros_like(self.r[1:])
        self.velocity=self.velocity[self.indstart+1:]
    



        

@dataclass
class SlenderCurvTors(SlenderBody):
    """
    This class constructs a slender body from prescribed curvature and torsion
    distributions using the Frenet-Serret equations. The number of curvature 
    and torsion values determines the number of nodes on the body. If curvature 
    has length N, then the number of elements is N-1.

    The centerline is obtained by integrating the Frenet-Serret equations using
    a fourth-order Runge-Kutta scheme. Using the resulting curve, the mobility 
    matrix and interaction matrix can be calculated using Slender Body Theory.

    To calculate the mobility matrix use the method `SlenderCurvTors.construct_mobility_matrix()`
    To calculate the interaction matrix between the slender body and some coordinates 'coords',
    use the method `SlenderCurvTors.calc_interaction(coords)`.

    Parameters
    ----------
    curvature         : numpy array (N,)
                        Curvature values at each node.
    torsion           : numpy array (N,)
                        Torsion values at each node. Use zeros for planar curves.
    T_0               : numpy array (3,), optional
                        Initial tangent vector.
    N_0               : numpy array (3,), optional
                        Initial normal vector.
    base_position     : numpy array (3,), optional
                        Starting position of the curve.
    flagellum_length  : float or int (optional), default 10
                        Total length of the slender body.
    flagellum_radius  : float or int (optional), default 0.2
                        Radius of the slender body.
    smin              : float (optional), default 0.15
                        Fraction of total length at which the curve starts.
    velocity          : numpy array (N, 3) (optional)
                        Velocity at each node. Automatically set to zero if not provided.

    Attributes
    ----------
    r : numpy.ndarray (Nf, 3)
        Cartesian coordinates of the centerline after truncation.
    tangents : numpy.ndarray (Nf, 3)
        Unit tangent vectors obtained from Frenet-Serret integration.
    ss : numpy.ndarray
        Arclength positions of nodes.
    flag_centroids : numpy.ndarray
        Arclength positions of element midpoints.
    element_lengths : numpy.ndarray
        Length of each element.
    Nf : int
        Number of elements.
    ds : float
        Uniform arclength spacing.
    slenderness : float
        Radius-to-length ratio.
    slend_2 : numpy.ndarray
        Squared slenderness parameter per element.

    Example
    -------
    Construct a planar slender body from constant curvature.

    >>> import bemsolver as bem
    >>> import numpy as np
    >>> U = np.zeros(3)
    >>> U[0] = 1
    >>> W = np.zeros(3)
    >>>
    >>> curvature = np.ones(21)
    >>> torsion = np.zeros_like(curvature)
    >>> body = bem.SlenderCurvTors(curvature, torsion)
    >>> M = body.construct_mobility_matrix()

    >>> RHS           = body.set_boundary_condition(U, W)   # rate of strain tensor is optional

    >>> f             = np.linalg.solve(M, RHS)
    >>> # reshape f to force vectors per element
    >>> force_vectors = f.reshape(int(len(f)/3),3)
    """

    curvature           : np.ndarray[float|int]
    torsion             : np.ndarray[float|int]

    T_0                 : np.ndarray            = field(default_factory=lambda: np.array([1.0,0,0]))
    N_0                 : np.ndarray            = field(default_factory=lambda: np.array([0,1.0,0]))
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
        
        # slenderness of an ellipsoid
        self.slenderness = self.flagellum_radius / self.flagellum_length

        #==========Slenderness of a cylinder==============
        # self.r_epsilon = 2 * self.slenderness * np.sqrt(self.flag_centroids*(self.flagellum_length - self.flag_centroids))

        # self.slend_2 = (self.r_epsilon / self.flagellum_length)**2
        #=================================================

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

        # self.T_0 = np.array([np.cos(self.theta_0) * np.cos(self.rho_0), 
        #                      np.sin(self.theta_0) * np.cos(self.rho_0),
        #                      np.sin(self.rho_0)])
     
        
        self.tangents[0] = self.T_0
        
        # Initial normal vector
        # ref = np.array([0.0, 0.0, 1.0])        # z-axis by default

        # if np.abs(np.dot(ref, self.T_0)) > 0.9:
        #     ref = np.array([0.0, 1.0, 0.0]) 
        # self.N_0 = np.cross(self.T_0, ref)   


        # self.B_0 = np.array([-np.cos(self.theta_0) * np.sin(self.rho_0), 
        #                      -np.sin(self.theta_0) * np.sin(self.rho_0),
        #                       np.cos(self.rho_0)])
        # self.N_0 = _normal_from_tangent(self.T_0)
        # self.N_0 /= np.linalg.norm(self.N_0)

        # Initial binormal vector
        self.B_0 = np.cross(self.T_0, self.N_0)
        self.B_0 /= np.linalg.norm(self.B_0)
                

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




@dataclass
class SlenderAngles(SlenderBody):
    
    """
    This class constructs a slender body from prescribed angular distributions.
    The tangent vector at each node is defined by in-plane angle theta and 
    out-of-plane angle phi. The number of angle values determines the number 
    of nodes on the body. If theta has length N, then the number of elements 
    is N-1.

    The centerline coordinates are obtained by integrating the tangent vectors
    using the trapezoidal rule. Using the resulting curve, the mobility matrix 
    and interaction matrix can be calculated using Slender Body Theory.

    To calculate the mobility matrix use the method `SlenderAngles.construct_mobility_matrix()`
    To calculate the interaction matrix between the slender body and some coordinates 'coords',
    use the method `SlenderAngles.calc_interaction(coords)`.

    Parameters
    ----------
    theta             : numpy array (N,)
                        In-plane angles defining the tangent direction.
    phi               : numpy array (N,)
                        Out-of-plane angles defining the tangent direction.
    base_position     : numpy array (3,), optional
                        Starting position of the curve.
    velocity          : numpy array (N, 3) (optional)
                        Velocity at each node. Automatically set to zero if not provided.
    flagellum_radius  : float or int (optional), default 0.2
                        Radius of the slender body.
    flagellum_length  : float or int (optional), default 7
                        Total length of the slender body.
    smin              : float (optional), default 0
                        Fraction of total length at which the curve starts.

    Attributes
    ----------
    r : numpy.ndarray (Nf, 3)
        Cartesian coordinates of the centerline after truncation.
    tangents : numpy.ndarray (Nf, 3)
        Unit tangent vectors defined by theta and phi.
    ss : numpy.ndarray
        Arclength positions of nodes.
    flag_centroids : numpy.ndarray
        Arclength positions of element midpoints.
    element_lengths : numpy.ndarray
        Length of each element.
    Nf : int
        Number of elements.
    slenderness : float
        Radius-to-length ratio.
    slend_2 : numpy.ndarray
        Squared slenderness parameter per element.

    Example
    -------
    Construct a slender body from prescribed angular distributions.

    >>> import bemsolver as bem
    >>> import numpy as np
    >>> U = np.zeros(3)
    >>> U[0] = 1
    >>> W = np.zeros(3)
    >>>
    >>> theta = np.linspace(0, np.pi, 21)
    >>> phi = np.zeros_like(theta)
    >>> body = bem.SlenderAngles(theta, phi)
    >>> M = body.construct_mobility_matrix()


    >>> RHS           = body.set_boundary_condition(U, W)   # rate of strain tensor is optional

    >>> f             = np.linalg.solve(M, RHS)
    >>> # reshape f to force vectors per element
    >>> force_vectors = f.reshape(int(len(f)/3),3)
    """

    theta               : np.ndarray[float|int]
    phi                 : np.ndarray[float|int]

    base_position       : np.ndarray[float|int] = field(default_factory=lambda: np.array([0, 0, 0]))
    velocity            : np.ndarray            = field(default_factory=lambda: None)
    flagellum_radius    : int|float             = field(default_factory=lambda: 0.2)
    flagellum_length    : int|float             = field(default_factory=lambda: 7)
    smin                : int|float             = field(default_factory=lambda: 0.0)



    def __post_init__(self):
        if len(self.theta)!= len(self.phi):
            raise IndexError(f"In-plane and out-of-plane angles must have the same length ({len(self.theta)} != {len(self.phi)})")
        
        
        Nf = len(self.theta)
    
        # Arclength of flagellum
        self.ss = np.linspace(0, self.flagellum_length, Nf)

        # Find point on curve where it is larger than smin * length
        self.indstart        = np.min(np.where(self.ss >= self.smin * self.flagellum_length))
        

        

        # Calculate the center of every element of the curve and its length
        self.flag_centroids  = (self.ss[1:] + self.ss[:-1]) / 2
        self.element_lengths =  self.ss[1:] - self.ss[:-1]

        # Calculate tangent vectors from theta and phi
        tx = np.cos(self.theta) * np.cos(self.phi)
        ty = np.sin(self.theta) * np.cos(self.phi)
        tz = np.sin(self.phi)
        
        self.tangents  = np.column_stack((tx, ty, tz))

        self.r        = np.zeros((len(self.ss)-1, 3))
        self.r[0]     = self.base_position

        # print(len(self.ss))
        # Use trapezoidal rule to calculate the coordinates of the curve
        for i in range(1,len(self.ss)-1):
            self.r[i] = self.r[i-1] + 0.5 * (self.tangents[i-1] + self.tangents[i]) * self.element_lengths[i-1]

        # make copy of full curve just in case
        self.r_full = np.copy(self.r)

        # start from smin * length
        self.r               = self.r[self.indstart:]
        self.ss              = self.ss[self.indstart:]
        self.flag_centroids  = self.flag_centroids[self.indstart:]
        self.element_lengths = self.element_lengths[self.indstart:]
        self.tangents        = self.tangents[self.indstart+1:]


        # Amount of elements in curve
        self.Nf=len(self.ss)-1

        # calc slenderness of an ellipsoid for all elements (constant along the curve)
        self.slenderness = self.flagellum_radius / self.flagellum_length

        self.slend_2 = self.slenderness**2 * np.ones_like(self.flag_centroids)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.r)



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
