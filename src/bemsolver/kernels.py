import numpy as np
from numba import njit


def stresslet_vectorized(collocations   :np.ndarray,
                         centroid       :np.ndarray,
                         Xq             :np.ndarray,
                         Yq             :np.ndarray,
                         Wx             :np.ndarray, 
                         Wy             :np.ndarray)->np.ndarray:
    
    """
    Calculates the stresslet contribution of an element on all collocation points and integrates them. 
    Only the normal component of the stresslet tensor is computed. This is the vectorised version.
    Non-vectorised function is simply called stresslet.


    Parameters
    ----------
    collocations : (M,3) array of evaluation points
                   The collocation points to be evaluated in the element frame
    centroid     : (3,)  element centroid
                   Center of the current element to be integrated in the element frame
    Xq           : (Q,Q) quadrature grid
                    x-coordinates of the quadrature points
    Yq           : (Q,Q) quadrature grid
                   y-coordinates of the quadrature points
    Wx           : (Q,)  quadrature weights
                   Quadrature weights of the x-axis
    Wy           : (Q,)  quadrature weights
                   Quadrature weights of the y-axis

    Returns
    -------
    T_all : (M,3,3) array of stresslet tensors
            Stresslet contribution on all collocation points.

    """


    # Compute the integration weights
    W2D = np.outer(Wx, Wy)  # shape (Q,Q)

    # --- Vectorized distance computations ---
    # Collocations shape: (M,3)
    # Quadrature grid: (Q,Q)
    # We broadcast the quadrature grid across all collocation points

    # Determine the x,y,z vectors which define the distance between col. and cent.
    # Collocations shape: (M,3)
    # Quadrature grid: (Q,Q)
    # The quadrature grid is broadcasted across all collocation points

    x = collocations[:, None, None, 0] - (centroid[0] + Xq)
    y = collocations[:, None, None, 1] - (centroid[1] + Yq)
    z = collocations[:, None, None, 2] - centroid[2]

    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r5 = r**5

    # Calculate the normal stresslet contribution T_ij3
    # The z axis is always normal to the surface

    T_11 = z * x**2     / r5
    T_12 = z * x * y    / r5
    T_13 = z * x * z    / r5

    T_22 = z * y**2     / r5
    T_23 = z * y * z    / r5

    T_33 = z * z**2     / r5

    # Gaussian quadrature
    def quad2d(A):
        # Integrate A[..., Q, Q] over the quadrature weights
        return np.sum(W2D * A, axis=(-2, -1))
    
    # Assemble stresslet tensor
    T_all = np.zeros((collocations.shape[0], 3, 3))

    T_all[:, 0, 0] = quad2d(T_11)
    T_all[:, 0, 1] = quad2d(T_12)
    T_all[:, 0, 2] = quad2d(T_13)

    T_all[:, 1, 1] = quad2d(T_22)
    T_all[:, 1, 2] = quad2d(T_23)

    T_all[:, 2, 2] = quad2d(T_33)

    # Stresslet tensor is symmetric
    T_all[:, 1, 0] = T_all[:, 0, 1]
    T_all[:, 2, 0] = T_all[:, 0, 2]
    T_all[:, 2, 1] = T_all[:, 1, 2]

    return T_all


def line_singularity_vectorized(collocations    :np.ndarray,
                                centroid        :np.ndarray, 
                                coord           :np.ndarray,
                                R               :np.ndarray, 
                                Xq              :np.ndarray, 
                                Yq              :np.ndarray,
                                Wx              :np.ndarray,
                                Wy              :np.ndarray)->tuple[np.ndarray,np.ndarray]:
    
    """
    Calculates the stokeslet and rotlet contribution of the line singularity on all collocation points and integrates them. 
    This is the vectorised version. Non-vectorised function is simply called line_singularity.


    Parameters
    ----------
    collocations : (M,3) array of evaluation points
                   The collocation points to be evaluated in the element frame
    centroid     : (3,)  element centroid
                   Center of the current element to be integrated in the element frame
    coord        : (3,3) element coordinate frame
                   Orthonormal coordinate system of the current element
    Xq           : (Q,Q) quadrature grid
                   x-coordinates of the quadrature points
    Yq           : (Q,Q) quadrature grid
                   y-coordinates of the quadrature points
    Wx           : (Q,)  quadrature weights
                   Quadrature weights of the x-axis
    Wy           : (Q,)  quadrature weights
                   Quadrature weights of the y-axis

    Returns
    -------
    S_all, G_all : (M,3,3) arrays for all collocation points
                    Stokeslet and rotlet contribution on all collocation points.
    """

    M = collocations.shape[0]
    W2D = np.outer(Wx, Wy)

    # Decompose the location R as x,y,z coordinates in the element frame
    Rx = R * coord[0, 0]
    Ry = R * coord[1, 0]
    Rz = R * coord[2, 0]

    # Define the vector P from the mapped quadrature points on the line to the collocation point
    # Broadcasting shapes: (M,1,1) - (Q,Q)
    Px = collocations[:, None, None, 0] - Rx       # Distance is in microns
    Py = collocations[:, None, None, 1] - Ry
    Pz = collocations[:, None, None, 2] - Rz

    Px2 = Px**2
    Py2 = Py**2
    Pz2 = Pz**2
    PP2 = Px2 + Py2 + Pz2
    PP = np.sqrt(PP2)

    PP3 = PP2 * PP

    # Calculate the stokeslet contribution S = I/r + rr/r^3     without prefactor 1/(8 pi mu)
    s11 = 1.0 / PP    +   Px2       / PP3
    s12 =                 Px * Py   / PP3
    s13 =                 Px * Pz   / PP3

    s21 =                 Py * Px   / PP3
    s22 = 1.0 / PP    +   Py2       / PP3
    s23 =                 Py * Pz   / PP3

    s31 =                 Pz * Px   / PP3
    s32 =                 Pz * Py   / PP3
    s33 = 1.0 / PP    +   Pz2       / PP3

    # Gaussian quadrature
    def quad2d(A):
        return np.sum(W2D * A, axis=(-2, -1))
    
    # Assemble stokeslet tensor
    S_all = np.zeros((M, 3, 3))
    S_all[:, 0, 0] = quad2d(s11)
    S_all[:, 0, 1] = quad2d(s12)
    S_all[:, 0, 2] = quad2d(s13)

    S_all[:, 1, 0] = quad2d(s21)
    S_all[:, 1, 1] = quad2d(s22)
    S_all[:, 1, 2] = quad2d(s23)

    S_all[:, 2, 0] = quad2d(s31)
    S_all[:, 2, 1] = quad2d(s32)
    S_all[:, 2, 2] = quad2d(s33)

    # Zq is always a matrix with zeros but for consistency I leave it in
    Zq = np.zeros_like(Xq)

    # Define the vector Q from the quadrature points on the centerline to the quadrature
    # points on the current element (in element frame)
    Qx = centroid[0] + Xq - Rx
    Qy = centroid[1] + Yq - Ry
    Qz = centroid[2] + Zq - Rz

    # In the singularity equations, it does not mention the trace anywhere. However, if you work out the
    # tensor calculus, it becomes convenient to write it in terms of the "trace"
    trace = Px * Qx + Py * Qy + Pz * Qz

    # Calculate the rotlet contribution R_ij= ɛ_ijk r_k / r^3
    g11  =  (trace - Qx * Px) / PP3
    g12  =  (      - Qx * Py) / PP3
    g13  =  (      - Qx * Pz) / PP3

    g21  =  (      - Qy * Px) / PP3
    g22  =  (trace - Qy * Py) / PP3
    g23  =  (      - Qy * Pz) / PP3

    g31  =  (      - Qz * Px) / PP3
    g32  =  (      - Qz * Py) / PP3
    g33  =  (trace - Qz * Pz) / PP3

    # Assemble rotlet tensor
    G_all = np.zeros((M, 3, 3))

    G_all[:, 0, 0] = quad2d(g11)
    G_all[:, 0, 1] = quad2d(g12)
    G_all[:, 0, 2] = quad2d(g13)

    G_all[:, 1, 0] = quad2d(g21)
    G_all[:, 1, 1] = quad2d(g22)
    G_all[:, 1, 2] = quad2d(g23)

    G_all[:, 2, 0] = quad2d(g31)
    G_all[:, 2, 1] = quad2d(g32)
    G_all[:, 2, 2] = quad2d(g33)

    return S_all, G_all





def stokeslet(Xij, Yij, Zij):

    N   = Xij.shape[0]

    G   = np.zeros((3 * N, 3 * N))

    Rij = np.sqrt(Xij**2 + Yij**2 + Zij**2 ) + 10**20 * np.eye(len(Xij))

    Xij = Xij / Rij
    Yij = Yij / Rij
    Zij = Zij / Rij

    # Select only the non-diagonal terms
    Mask = np.ones_like(Rij) - np.eye(len(Rij))

    # Make indices for allocation in steps of 3 (idx is for x, idx+1 is for y, and idx+2 is for z)
    idx = np.arange(0, 3*N, 3)

    # Kxx, Kxy, Kxz
    G[np.ix_(idx, idx)]         = (1 + Xij**2) / Rij * Mask
    G[np.ix_(idx, idx + 1)]     = (Xij * Yij)  / Rij * Mask
    G[np.ix_(idx, idx + 2)]     = (Xij * Zij)  / Rij * Mask


    # Kyx, Kyy, Kyz
    G[np.ix_(idx + 1, idx)]     = G[np.ix_(idx, idx + 1)]
    G[np.ix_(idx + 1, idx + 1)] = (1 + Yij**2) / Rij * Mask
    G[np.ix_(idx + 1, idx + 2)] = (Yij * Zij)  / Rij * Mask

    # Kzx, Kzy, Kzz
    G[np.ix_(idx + 2, idx)]     = (Zij * Xij)  / Rij * Mask
    G[np.ix_(idx + 2, idx + 1)] = (Zij * Yij)  / Rij * Mask
    G[np.ix_(idx + 2, idx + 2)] = (1 + Zij**2) / Rij * Mask

    return G

def tangential(Lij, Sij, T):

    N   = Lij.shape[0]

    L   = np.zeros((3 * N, 3 * N))

    t_x ,t_y, t_z = T

    t = np.column_stack((t_x, t_y, t_z))

    Mask = np.ones((N,N)) - np.eye(N)

    # Add identity matrix to avoid division by zero (The diagonal terms are set to zero afterwards with Mask)
    Sij = Sij + 10**20*np.eye(len(Sij))

    F = Lij / Sij  * Mask 

    # Assemble diagonal block matrix
    for i in range(N):
        ti = t[i]                 # shape (3,)
        tkF = t.T @ F[i]          # shape (3,), equals sum over k
        Li = np.outer(ti, tkF)    # shape (3,3)

        L[3*i:3*i+3, 3*i:3*i+3] = Li

    return L





#============================ FROM HERE ONWARDS IS UNUSED CODE===========================

@njit(fastmath=True)
def stresslet(collocation:np.ndarray,
              centroid:np.ndarray,
              Xq:np.ndarray,
              Yq:np.ndarray,
              Wx:np.ndarray,
              Wy:np.ndarray):
    """
    Calculates the stresslet contribution of an element on the collocation point and integrates it. 
    Only the normal component of the stresslet tensor is computed.

    NOTE: This function looks a lot like the original matlab code. It only calculates the contribution
    per collocation point. With the @njit header, the computation speed is comparable with matlab.
    However, using the vectorised version is much much quicker.

    Parameters
    ----------
    collocation : np.ndarray
                  The collocation point to be evaluated
    centroid    : np.ndarray
                  Center of the current element to be integrated
    Xq          : np.ndarray
                  x-coordinates of the quadrature points
    Yq          : np.ndarray
                  y-coordinates of the quadrature points
    Wx          : np.ndarray
                  Quadrature weights of the x-axis
    Wy          : np.ndarray
                  Quadrature weights of the y-axis

    """

    T=np.zeros((3,3))

    # Determine the x,y,z vectors which define the distance between col. and cent.
    # x and y are of the size (qxq) with q being the quadrature order, while z is a scalar.
    x  = collocation[0] - (centroid[0]+Xq)
    y  = collocation[1] - (centroid[1]+Yq)
    z  = collocation[2] -  centroid[2]  

    x2=x**2
    y2=y**2
    z2=z**2
    

    r2=x2 + y2 + z2
    r=np.sqrt(r2)
    r5=r**5

    # Calculate the normal stresslet contribution T_ij3
    # The z axis is always normal to the surface

    T_11 = z * x2         / r5
    T_12 = z * x   * y    / r5
    T_13 = z * x   * z    / r5

    T_22 = z * y2         / r5  
    T_23 = z * y   * z    / r5

    T_33 = z * z2         / r5

    T[0,0]=Wx @ T_11 @ Wy
    T[0,1]=Wx @ T_12 @ Wy
    T[0,2]=Wx @ T_13 @ Wy
    
    T[1,1]=Wx @ T_22 @ Wy
    T[1,2]=Wx @ T_23 @ Wy

    T[2,2]=Wx @ T_33 @ Wy

    T+= np.triu(T, 1).T

    return T

@njit(fastmath=True)
def line_singularity(collocation:np.ndarray,
                     centroid   :np.ndarray,
                     coord      :np.ndarray,
                     R          :np.ndarray,
                     Xq         :np.ndarray,
                     Yq         :np.ndarray,
                     Wx         :np.ndarray,
                     Wy         :np.ndarray):
    """
    Calculates the line singularity contribution of an element on the collocation point and integrates it. 

    NOTE: This function looks a lot like the original matlab code. It only calculates the contribution
    per collocation point. With the @njit header, the computation speed is comparable with matlab.
    However, using the vectorised version is much much quicker.

    Parameters
    ----------
    collocation : np.ndarray
                  The collocation point to be evaluated
    centroid    : np.ndarray
                  Center of the current element to be integrated
    coord       : np.ndarray
                  Orthonormal coordinate system of the current element
    R           : np.ndarray
                  x-coordinates of the quadrature points mapped to the center line (not in element coordinate system)
    Xq          : np.ndarray
                  x-coordinates of the quadrature points
    Yq          : np.ndarray
                  y-coordinates of the quadrature points
    Wx          : np.ndarray
                  Quadrature weights of the x-axis
    Wy          : np.ndarray
                  Quadrature weights of the y-axis

    """
    # Initialise Stokeslet tensor
    S = np.zeros((3,3))

    # Decompose the location R as x,y,z coordinates in the element frame
    Rx = R * coord[0,0]         # Note that Rx, Ry, and Rz are matrices because of the quadrature points
    Ry = R * coord[1,0]
    Rz = R * coord[2,0]

    # Define the vector P from the mapped quadrature points on the line to the collocation point
    Px  = collocation[0] - Rx       # Distance is given in microns
    Py  = collocation[1] - Ry
    Pz  = collocation[2] - Rz

    Px2 = Px**2
    Py2 = Py**2
    Pz2 = Pz**2

    PP  =  np.sqrt(Px2 + Py2 + Pz2)
    PP3 =  PP**3

    # Calculate the stokeslet contribution S = I/r + rr/r^3     without prefactor 1/(8 pi mu)

    s11   =  1/PP  +  Px2       /PP3        
    s12   =           Px * Py   /PP3      
    s13   =           Px * Pz   /PP3       

    s21   =           Py * Px   /PP3       
    s22   =  1/PP  +  Py2       /PP3       
    s23   =           Py * Pz   /PP3     

    s31   =           Pz * Px   /PP3    
    s32   =           Pz * Py   /PP3    
    s33   =  1./PP +  Pz2       /PP3  


    S[0,0] = Wx @ s11 @ Wy
    S[0,1] = Wx @ s12 @ Wy
    S[0,2] = Wx @ s13 @ Wy

    S[1,0] = Wx @ s21 @ Wy
    S[1,1] = Wx @ s22 @ Wy
    S[1,2] = Wx @ s23 @ Wy
    
    S[2,0] = Wx @ s31 @ Wy
    S[2,1] = Wx @ s32 @ Wy
    S[2,2] = Wx @ s33 @ Wy

    # Initialise Rotlet tensor
    G=np.zeros((3,3))

    # Zq is always a matrix with zeros but for consistency I leave it in
    Zq = np.zeros(np.shape(Xq))

    # Define the vector Q from the quadrature points on the centerline to the quadrature
    # points on the current element (in element frame)
    Qx  = centroid[0]+Xq-Rx
    Qy  = centroid[1]+Yq-Ry
    Qz  = centroid[2]+Zq-Rz


    # In the singularity equations, it does not mention the trace anywhere. However, if you work out the
    # tensor calculus, it becomes convenient to write it in terms of the "trace"
    trace = Px * Qx + Py * Qy + Pz * Qz

    # Calculate the rotlet contribution R_ij= ɛ_ijk r_k / r^3
    g11  =  (trace - Qx * Px) / PP3
    g12  =  (      - Qx * Py) / PP3
    g13  =  (      - Qx * Pz) / PP3

    g21  =  (      - Qy * Px) / PP3
    g22  =  (trace - Qy * Py) / PP3
    g23  =  (      - Qy * Pz) / PP3

    g31  =  (      - Qz * Px) / PP3
    g32  =  (      - Qz * Py) / PP3
    g33  =  (trace - Qz * Pz) / PP3


    G[0,0] = Wx @ g11 @ Wy
    G[0,1] = Wx @ g12 @ Wy
    G[0,2] = Wx @ g13 @ Wy

    G[1,0] = Wx @ g21 @ Wy
    G[1,1] = Wx @ g22 @ Wy
    G[1,2] = Wx @ g23 @ Wy
    
    G[2,0] = Wx @ g31 @ Wy
    G[2,1] = Wx @ g32 @ Wy
    G[2,2] = Wx @ g33 @ Wy  

    return S, G

# How to use stresslet and line_singularity functions (Not recommended!)

#==================ORIGINAL LOOP (NOT USED)==================
        # singularities    = np.zeros((3*numevals,3)) 
        # for i, eval_point in enumerate(evaluation_points):
            
            # Col = coord @ eval_point    # Collocation point            
            
            # T=stresslet(Col, Int, Xq, Yq, Wx, Wy)

            # S, G = line_singularity(Col, Int, coord, R, Xq, Yq, Wx, Wy)

            # singularities[i:i+3] = coord.T @ ( 3/(4*np.pi) * T + 1/(8*np.pi) * (S + G) ) @ coord

        #==============================================================


    















