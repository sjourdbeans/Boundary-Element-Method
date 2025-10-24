import numpy as np

def stresslet(x:np.ndarray,
              y:np.ndarray,
              z:np.ndarray):
    """
    Calculates the stresslet contribution of an element on the collocation point. 
    Only the normal component of the stresslet tensor is computed.
    """

    T=np.zeros((3,3))

    x2=x**2
    y2=y**2
    z2=z**2
    

    r2=x2 + y2 + z2
    r=np.sqrt(r2)
    r5=r**5

    # Calculate the normal stresslet contribution T_ij3
    # The z axis is always normal to the surface

    T_11 = r[2] * r2[0]         / r5
    T_22 = r[2] * r2[1]         / r5
    T_12 = r[2] * r[0]   * r[1] / r5
    T_13 = r[2] * r[0]   * r[2] / r5
    T_23 = r[2] * r[1]   * r[2] / r5
    T_33 = r[2] * r2[2]         / r5

    T[0,0]=T_11
    T[0,1]=T_12
    T[0,2]=T_13
    
    T[1,1]=T_22
    T[1,2]=T_23

    T+= np.triu(T, 1).T

    return T






