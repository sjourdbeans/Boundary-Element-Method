import numpy as np
from numba import njit

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

    # Calculate the rotlet contribution R_ij= ɛ_ijk r_k / r^5
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



@njit(fastmath=True, cache=True)
def stresslet_fast(collocation, centroid, Xq, Yq, Wx, Wy):
    """
    Numba-accelerated stresslet computation.
    """
    T = np.zeros((3, 3))

    x = collocation[0] - (centroid[0] + Xq)
    y = collocation[1] - (centroid[1] + Yq)
    z = collocation[2] - centroid[2]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    r2 = x2 + y2 + z2
    r = np.sqrt(r2)
    r5 = r2 * r2 * r

    T_11 = z * x2 / r5
    T_12 = z * x * y / r5
    T_13 = z * x * z / r5
    T_22 = z * y2 / r5
    T_23 = z * y * z / r5
    T_33 = z * z2 / r5

    # Manually compute the weighted integrals (Wx and Wy are 1D)
    # Equivalent to Wx @ A @ Wy but faster for small arrays
    def quad2d(A, Wx, Wy):
        tmp = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            acc = 0.0
            for j in range(A.shape[1]):
                acc += A[i, j] * Wy[j]
            tmp[i] = acc
        res = 0.0
        for i in range(A.shape[0]):
            res += Wx[i] * tmp[i]
        return res

    T[0, 0] = quad2d(T_11, Wx, Wy)
    T[0, 1] = quad2d(T_12, Wx, Wy)
    T[0, 2] = quad2d(T_13, Wx, Wy)
    T[1, 1] = quad2d(T_22, Wx, Wy)
    T[1, 2] = quad2d(T_23, Wx, Wy)
    T[2, 2] = quad2d(T_33, Wx, Wy)

    # Fill symmetric upper triangle
    T[1, 0] = T[0, 1]
    T[2, 0] = T[0, 2]
    T[2, 1] = T[1, 2]

    return T
    

@njit(fastmath=True, cache=True)
def line_singularity_fast(collocation, centroid, coord, R, Xq, Yq, Wx, Wy):
    """
    Numba-accelerated line singularity contribution (Stokeslet + Rotlet).
    Equivalent to MATLAB-style quadrature implementation but compiled to machine code.
    """

    # --- Helper for double quadrature integration ---
    def quad2d(A, Wx, Wy):
        tmp = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            acc = 0.0
            for j in range(A.shape[1]):
                acc += A[i, j] * Wy[j]
            tmp[i] = acc
        res = 0.0
        for i in range(A.shape[0]):
            res += Wx[i] * tmp[i]
        return res

    # --- Precompute Rx, Ry, Rz in element frame ---
    Rx = R * coord[0, 0]
    Ry = R * coord[1, 0]
    Rz = R * coord[2, 0]

    # --- Vector from mapped quadrature points to collocation point ---
    Px = collocation[0] - Rx
    Py = collocation[1] - Ry
    Pz = collocation[2] - Rz

    Px2 = Px * Px
    Py2 = Py * Py
    Pz2 = Pz * Pz

    PP = np.sqrt(Px2 + Py2 + Pz2)
    PP3 = PP * PP * PP

    # --- Stokeslet tensor (S) ---
    s11 = 1.0 / PP + Px2 / PP3
    s12 = Px * Py / PP3
    s13 = Px * Pz / PP3
    s21 = Py * Px / PP3
    s22 = 1.0 / PP + Py2 / PP3
    s23 = Py * Pz / PP3
    s31 = Pz * Px / PP3
    s32 = Pz * Py / PP3
    s33 = 1.0 / PP + Pz2 / PP3

    S = np.zeros((3, 3))
    S[0, 0] = quad2d(s11, Wx, Wy)
    S[0, 1] = quad2d(s12, Wx, Wy)
    S[0, 2] = quad2d(s13, Wx, Wy)
    S[1, 0] = quad2d(s21, Wx, Wy)
    S[1, 1] = quad2d(s22, Wx, Wy)
    S[1, 2] = quad2d(s23, Wx, Wy)
    S[2, 0] = quad2d(s31, Wx, Wy)
    S[2, 1] = quad2d(s32, Wx, Wy)
    S[2, 2] = quad2d(s33, Wx, Wy)

    # --- Rotlet tensor (G) ---
    G = np.zeros((3, 3))
    Zq = np.zeros_like(Xq)

    Qx = centroid[0] + Xq - Rx
    Qy = centroid[1] + Yq - Ry
    Qz = centroid[2] + Zq - Rz

    trace = Px * Qx + Py * Qy + Pz * Qz

    g11 = (trace - Qx * Px) / PP3
    g12 = (-Qx * Py) / PP3
    g13 = (-Qx * Pz) / PP3
    g21 = (-Qy * Px) / PP3
    g22 = (trace - Qy * Py) / PP3
    g23 = (-Qy * Pz) / PP3
    g31 = (-Qz * Px) / PP3
    g32 = (-Qz * Py) / PP3
    g33 = (trace - Qz * Pz) / PP3

    G[0, 0] = quad2d(g11, Wx, Wy)
    G[0, 1] = quad2d(g12, Wx, Wy)
    G[0, 2] = quad2d(g13, Wx, Wy)
    G[1, 0] = quad2d(g21, Wx, Wy)
    G[1, 1] = quad2d(g22, Wx, Wy)
    G[1, 2] = quad2d(g23, Wx, Wy)
    G[2, 0] = quad2d(g31, Wx, Wy)
    G[2, 1] = quad2d(g32, Wx, Wy)
    G[2, 2] = quad2d(g33, Wx, Wy)

    return S, G


    















