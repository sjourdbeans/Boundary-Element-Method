import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve

# from .mesh import Mesh
# from .utils import find_panel_data, U_colloc
from .kernels import stokeslet, tangential
# from .quadrature import triquad

@dataclass  
class SlenderBody:

    curvature           : np.ndarray
    torsion             : np.ndarray

    theta_0             : int|float  = field(default_factory=lambda: np.pi/4)
    rho_0               : int|float  = field(default_factory=lambda: 0)
    base_position       : np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    flagellum_length    : int|float  = field(default_factory=lambda: 10)
    flagellum_radius    : int|float  = field(default_factory=lambda: 0.2) 
    smin                : int|float  = field(default_factory=lambda: 0.15)


    def __post_init__(self):

        # self.evaluation_points = self.centroids

        self.curvature      /= self.flagellum_length

        Nf                   = len(self.curvature) - 1
        self.ssold           = np.linspace(0,self.flagellum_length, Nf+1)
        indstart             = np.min(np.where(self.ssold >= self.smin * self.flagellum_length))

        self.ds              = self.ssold[1]-self.ssold[0]

        self.ss              = self.ssold[indstart:]
        self.Nf              = len(self.ss) -1

        self.flag_centroids  = (self.ss[1:] + self.ss[:-1]) / 2
        self.element_lengths =  self.ss[1:] - self.ss[:-1]
        

        self.slenderness = self.flagellum_radius / self.flagellum_length

        # slend_2 might be flaggellum specific
        self.slend_2 = self.flagellum_radius**2/(4*(self.flagellum_length - self.flag_centroids)*self.flag_centroids)

        # Frenet-Serret setup
        self.tangents = np.zeros((len(self.ssold), 3))
        self.r        = np.zeros((len(self.ssold), 3))
        self.r[0]     = self.base_position
        
        # Initial tangent vector
        self.T_0 = np.array([np.cos(self.theta_0) * np.cos(self.rho_0), 
                             np.sin(self.theta_0) * np.cos(self.rho_0),
                             np.sin(self.rho_0)])
        
        self.tangents[0] = self.T_0
        
        # Initial normal vector
        self.N_0 = np.array([-np.sin(self.theta_0), 
                             np.cos(self.theta_0),
                             0])
        
        # Initial binormal vector
        self.B_0 = np.cross(self.T_0, self.N_0)

        self.calc_curve()
        
        self.r        = self.r[indstart+1:]
        self.tangents = self.tangents[indstart+1:]


        
        # Calculate cen


    

    def calc_curve(self):
        Y = np.concatenate([self.base_position, self.T_0, self.N_0, self.B_0])

        for i in range(1,len(self.ssold)):

            def rhs(y,s):
                r = y[0:3]
                T = y[3:6]          
                N = y[6:9]
                B = y[9:12]

                alpha = (s-self.ssold[i-1])/(self.ssold[i]-self.ssold[i-1])

                kappa_i = self.curvature[i] + alpha * (self.curvature[i] - self.curvature[i-1]) 
                tau_i   = self.torsion[i] + alpha * (self.torsion[i] - self.torsion[i-1]) 


                dr = T
                dT = kappa_i*N
                dN = -kappa_i*T + tau_i*B
                dB = -tau_i*N

                return np.concatenate([dr, dT, dN, dB])
            
            Y = _rk4_step(rhs,Y,self.ssold[i-1],self.ds)

            T_next = Y[3:6]
            N_next = Y[6:9]
            B_next = Y[9:12]

            T_next = T_next / np.linalg.norm(T_next)
            N_next = N_next - np.dot(N_next, T_next) * T_next
            N_next = N_next / np.linalg.norm(N_next)
            B_next = np.cross(T_next, N_next)

            Y[3:6]  = T_next
            Y[6:9]  = N_next
            Y[9:12] = B_next

            self.r[i]        = Y[:3]
            self.tangents[i] = T_next       
            
            


    def calc_mobility(self):
        K = np.zeros((3*self.Nf, 3*self.Nf))
        H            = np.zeros((3 * self.Nf, 3))
        ones_array   = np.ones((1, self.Nf))
        
        x, y, z = self.r.T
        
        Xi = np.outer(x, ones_array)
        Yi = np.outer(y, ones_array)
        Zi = np.outer(z, ones_array)

        Li = np.outer(self.element_lengths, ones_array)

        t_x = np.outer(self.tangents[:,0], ones_array)
        t_y = np.outer(self.tangents[:,1], ones_array)
        t_z = np.outer(self.tangents[:,2], ones_array)
        T= (t_x, t_y, t_z)

        Si = np.outer(self.flag_centroids, ones_array)

        Lij = Li.T / Li
        Xij = Xi - Xi.T
        Yij = Yi - Yi.T
        Zij = Zi - Zi.T

        
        Sij = np.abs(Si - Si.T) 

        G = stokeslet(Xij, Yij, Zij)

        L = tangential(Lij, Sij, T)

         

        idx = np.arange(0, 3*self.Nf, 3)

        constant = np.log(self.slend_2)  


        H[np.ix_(idx), 0]       =   (constant-1 + (constant + 3) * t_x[:,0] * t_x[:,0]) / self.element_lengths
        H[np.ix_(idx), 1]       =   (             (constant + 3) * t_x[:,0] * t_y[:,0]) / self.element_lengths
        H[np.ix_(idx), 2]       =   (             (constant + 3) * t_x[:,0] * t_z[:,0]) / self.element_lengths

        H[np.ix_(idx + 1), 0]   =   (             (constant + 3) * t_y[:,0] * t_x[:,0]) / self.element_lengths
        H[np.ix_(idx + 1), 1]   =   (constant-1 + (constant + 3) * t_y[:,0] * t_y[:,0]) / self.element_lengths
        H[np.ix_(idx + 1), 2]   =   (             (constant + 3) * t_y[:,0] * t_z[:,0]) / self.element_lengths

        H[np.ix_(idx + 2), 0]   =   (             (constant + 3) * t_z[:,0] * t_x[:,0]) / self.element_lengths
        H[np.ix_(idx + 2), 1]   =   (             (constant + 3) * t_z[:,0] * t_y[:,0]) / self.element_lengths
        H[np.ix_(idx + 2), 2]   =   (constant-1 + (constant + 3) * t_z[:,0] * t_z[:,0]) / self.element_lengths

        # Assemble mobility matrix

        K[np.ix_(idx, idx)]             = -G[np.ix_(idx,idx)]       + np.diag(H[np.ix_(idx), 0]     + L[np.ix_(idx), 0])
        K[np.ix_(idx, idx + 1)]         = -G[np.ix_(idx,idx+1)]     + np.diag(H[np.ix_(idx), 1]     + L[np.ix_(idx), 1])
        K[np.ix_(idx, idx + 2)]         = -G[np.ix_(idx,idx+2)]     + np.diag(H[np.ix_(idx), 2]     + L[np.ix_(idx), 2])


        K[np.ix_(idx + 1, idx)]         = -G[np.ix_(idx + 1,idx)]   + np.diag(H[np.ix_(idx + 1), 0] + L[np.ix_(idx + 1), 0])
        K[np.ix_(idx + 1, idx + 1)]     = -G[np.ix_(idx + 1,idx+1)] + np.diag(H[np.ix_(idx + 1), 1] + L[np.ix_(idx + 1), 1])
        K[np.ix_(idx + 1, idx + 2)]     = -G[np.ix_(idx + 1,idx+2)] + np.diag(H[np.ix_(idx + 1), 2] + L[np.ix_(idx + 1), 2])


        K[np.ix_(idx + 2, idx)]         = -G[np.ix_(idx + 2,idx)]   + np.diag(H[np.ix_(idx + 2), 0] + L[np.ix_(idx + 2), 0])
        K[np.ix_(idx + 2, idx + 1)]     = -G[np.ix_(idx + 2,idx+1)] + np.diag(H[np.ix_(idx + 2), 1] + L[np.ix_(idx + 2), 1])
        K[np.ix_(idx + 2, idx + 2)]     = -G[np.ix_(idx + 2,idx+2)] + np.diag(H[np.ix_(idx + 2), 2] + L[np.ix_(idx + 2), 2])

        return 1/(8*np.pi)* K
    

    def calc_interaction(self,
                         evaluation_points:np.ndarray):
        N_p = np.shape(evaluation_points)[0]

        xf, yf, zf = self.r.T
        xp, yp, zp =evaluation_points.T

        M = np.zeros((3*N_p, 3*self.Nf))

        ones_eval   = np.ones((N_p,1))
        ones_array  = np.ones((1, self.Nf))


        Xi = np.outer(xp, ones_array)
        Yi = np.outer(yp, ones_array)
        Zi = np.outer(zp, ones_array)
        
        Xj = np.outer(ones_eval, xf)
        Yj = np.outer(ones_eval, yf)
        Zj = np.outer(ones_eval, zf)

        Xij = Xi - Xj
        Yij = Yi - Yj
        Zij = Zi - Zj

        Rij = np.sqrt(Xij**2 + Yij**2 + Zij**2 ) + 0.5 * self.flagellum_radius

        
        Xij = Xij / Rij
        Yij = Yij / Rij
        Zij = Zij / Rij       

        idx_p = np.arange(0, 3*N_p, 3)
        idx_f = np.arange(0, 3*self.Nf, 3)


        M[np.ix_(idx_p, idx_f)]         =  (1 + Xij * Xij) / Rij + (self.flagellum_radius**2/2) * (1 - 3 * Xij * Xij) / Rij**3
        M[np.ix_(idx_p, idx_f + 1)]     =  (    Xij * Yij) / Rij + (self.flagellum_radius**2/2) * (  - 3 * Xij * Yij) / Rij**3
        M[np.ix_(idx_p, idx_f + 2)]     =  (    Xij * Zij) / Rij + (self.flagellum_radius**2/2) * (  - 3 * Xij * Zij) / Rij**3

        M[np.ix_(idx_p + 1, idx_f)]     =   M[np.ix_(idx_p, idx_f+1)]
        M[np.ix_(idx_p + 1, idx_f + 1)] =  (1 + Yij * Yij) / Rij + (self.flagellum_radius**2/2) * (1 - 3 * Yij * Yij) / Rij**3
        M[np.ix_(idx_p + 1, idx_f + 2)] =  (    Yij * Zij) / Rij + (self.flagellum_radius**2/2) * (  - 3 * Yij * Zij) / Rij**3

        M[np.ix_(idx_p + 2, idx_f)]     =   M[np.ix_(idx_p, idx_f + 2)]
        M[np.ix_(idx_p + 2, idx_f + 1)] =   M[np.ix_(idx_p + 1, idx_f + 2)]
        M[np.ix_(idx_p + 2, idx_f + 2)] =  (1 + Zij * Zij) / Rij + (self.flagellum_radius**2/2) * (1 - 3 * Zij * Zij) / Rij**3

        return 1/(8*np.pi) * M


def _rk4_step(f, y, s, ds):
    k1 = f(y,s)
    k2 = f(y + ds*k1/2., s + ds/2)
    k3 = f(y + ds*k2/2., s + ds/2)
    k4 = f(y + ds*k3, s + ds)
    return y + ds*(k1 + 2*k2 + 2*k3 + k4)/6.







