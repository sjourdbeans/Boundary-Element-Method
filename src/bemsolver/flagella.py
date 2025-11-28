import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.linalg import lu_factor, lu_solve

# from .mesh import Mesh
from .utils import U_colloc
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
        self.indstart        = np.min(np.where(self.ssold >= self.smin * self.flagellum_length))

        self.ds              = self.ssold[1]-self.ssold[0]

        self.ss              = self.ssold[self.indstart:]
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
        
        self.r        = self.r[self.indstart+1:]
        self.tangents = self.tangents[self.indstart+1:]



    

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


    def construct_mobility_matrix(self):

        self.MATRIX = self.calc_mobility()   

        return self.MATRIX


    def set_boundary_condition(self,
                               U        :np.ndarray,
                               W        :np.ndarray,
                               E        :np.ndarray=np.zeros((3,3)))->np.ndarray:
        
        
        rows, columns = np.shape(self.MATRIX)
        U_t, U_r, U_e =U_colloc(U,W, self.r,int(rows/3), E)

        return U_t+U_r+U_e 
            
            


    def calc_mobility(self):
        K = np.zeros((3*self.Nf, 3*self.Nf))

        H = np.zeros((3*self.Nf, 3*self.Nf))

        ones_array   = np.ones((1, self.Nf))
        
        x, y, z = self.r.T
        
        Xi = np.outer(x, ones_array)
        Yi = np.outer(y, ones_array)
        Zi = np.outer(z, ones_array)

        Li = np.outer(self.element_lengths, ones_array)

    
        t_x = self.tangents[:,0]
        t_y = self.tangents[:,1]
        t_z = self.tangents[:,2]
        T= (t_x, t_y, t_z)

        Si = np.outer(self.flag_centroids, ones_array)

        Lij = Li.T / Li
        Xij = Xi - Xi.T
        Yij = Yi - Yi.T
        Zij = Zi - Zi.T

        
        Sij = np.abs(Si - Si.T) 

        G = stokeslet(Xij, Yij, Zij)

        L = tangential(Lij, Sij, T)


        constant = np.log(self.slend_2)  

        for i in range(self.Nf):
            ti = self.tangents[i]
            tt = np.outer(ti, ti)
            Hi = (constant[i] - 1)*np.eye(3) + (constant[i] + 3)*tt
            Hi /= self.element_lengths[i]
            H[3*i:3*i+3, 3*i:3*i+3] = Hi


        K = -1/(8*np.pi)*(G - H - L)

        return  K
    

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







