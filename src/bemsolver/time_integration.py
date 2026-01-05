import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Callable


def pyr_to_quat(pitch: float|int, 
                yaw:float|int, 
                roll:float|int):
    """
    Convert roll, pitch, yaw to a quaternion vector
    
    :param pitch: Description
    :param yaw: Description
    :param roll: Description
    """
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)

    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = cy*sp*cr + sy*cp*sr
    z = sy*cp*cr - cy*sp*sr

    return np.array([w, x, y, z])

# def quat_to_R(q):

#     w, x, y, z = q
#     return np.array([
#         [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
#         [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
#         [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
#     ])


def vector_to_quaternion_from_x(p):
    """
    Returns a quaternion that rotates the x-axis [1,0,0] into the direction of p.
    Uses SciPy's Rotation module.
    Output is in [w, x, y, z] convention.
    """
    p = np.array(p, dtype=float)    
    p /= np.linalg.norm(p)  # ensure it's a unit vector

    # Define reference vector (minus x-axis)
    ex = np.array([-1.0, 0.0, 0.0])
    rot, _ = R.align_vectors([p], [ex])

    q = rot.as_quat(scalar_first=True)

    return q

def RK4(RHS :Callable[[np.ndarray], np.ndarray],
        Y   :np.ndarray,
        t   :float,
        dt  :float)->np.ndarray:
    """
    Simple RK4 function that integrates the RHS of the ODE to the next iteration
    """
    k1 = RHS(t, Y)                      
    k2 = RHS(t + dt/2, Y + 0.5*dt*k1)
    k3 = RHS(t + dt/2, Y + 0.5*dt*k2)
    k4 = RHS(t + dt, Y + dt*k3)

    Y_next = Y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    Y_next[3:]/=np.linalg.norm(Y_next[3:])
    
    return Y_next

def forward_euler(RHS :Callable[[np.ndarray], np.ndarray],
                  Y   :np.ndarray,
                  t   :float,
                  dt  :float)->np.ndarray:
    """
    Simple Forward Euler function that integrates the RHS of the ODE to the next iteration
    """
    Y_next = Y + dt * RHS(t, Y)
    Y_next[3:]/=np.linalg.norm(Y_next[3:])
   
    return Y_next

def rotate_BCs(Q, U, W, E):
    """
    Rotate the boundary conditions to particle frame.
    """

    U_body = Q.T @ U
    W_body = Q.T @ W
    E_body = Q.T @ E @ Q

    return U_body, W_body, E_body

# def omega_to_quat_dot(q, omega):
#     """Compute quaternion derivative dq/dt = 0.5 * Ω(ω) * q for [w,x,y,z]."""
#     wx, wy, wz = omega
#     Omega = np.array([
#         [0.0, -wx, -wy, -wz],
#         [wx,  0.0,  wz, -wy],
#         [wy, -wz,  0.0,  wx],
#         [wz,  wy, -wx,  0.0]
#     ])
#     return 0.5 * Omega @ q

def omega_to_quat_dot(q, omega):
    """dq/dt = 0.5 * q ⊗ (0, omega) in the lab frame"""
    wx, wy, wz = omega
    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0, -wz,  wy],
        [wy,  wz,  0.0, -wx],
        [wz, -wy,  wx,  0.0]
    ])
    return 0.5 * Omega @ q
